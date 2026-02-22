from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch
from transformers import AutoConfig, AutoModel


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava3d-path", required=True)
    parser.add_argument("--llava3d-model-type", default="llama", choices=["llama", "mistral"])
    parser.add_argument("--siglip-path", required=True)
    parser.add_argument("--mapanything-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-token-num", type=int, default=32)
    parser.add_argument("--use-geometric-branch", action="store_true", default=True)
    parser.add_argument(
        "--disable-geometric-branch", dest="use_geometric_branch", action="store_false"
    )
    parser.add_argument("--use-spatial-token", action="store_true", default=True)
    parser.add_argument("--disable-spatial-token", dest="use_spatial_token", action="store_false")
    parser.add_argument("--safe-serialization", action="store_true")
    parser.add_argument(
        "--weight-absmax-threshold",
        type=float,
        default=1.0e4,
        help="Fail if critical tensors have absurdly large absmax values.",
    )
    parser.add_argument(
        "--post-validate-load",
        dest="post_validate_load",
        action="store_true",
        default=True,
        help="After save, reload checkpoint and validate critical parameter groups.",
    )
    parser.add_argument(
        "--no-post-validate-load",
        dest="post_validate_load",
        action="store_false",
    )
    parser.add_argument("--clean-output", dest="clean_output", action="store_true", default=True)
    parser.add_argument("--no-clean-output", dest="clean_output", action="store_false")
    return parser.parse_args()


def _has_prefix(state_dict: dict, prefix: str) -> bool:
    for k in state_dict.keys():
        if k.startswith(prefix):
            return True
    return False


def _prefix_keys(state_dict: dict, prefix: str) -> list[str]:
    return [k for k in state_dict.keys() if k.startswith(prefix)]


def _meta_names_for_prefixes(
    model: torch.nn.Module,
    prefixes: Sequence[str],
    max_items: int = 20,
) -> list[str]:
    names = []
    for name, param in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if getattr(param, "is_meta", False):
            names.append(name)
    return names[:max_items]


def _assert_prefix_tensors_finite(state_dict: dict, prefixes: list[str]) -> None:
    bad_entries = []
    for prefix in prefixes:
        keys = _prefix_keys(state_dict, prefix)
        if not keys:
            bad_entries.append((prefix, "__missing__", 0.0))
            continue
        for key in keys:
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                continue
            finite_ratio = float(torch.isfinite(value.detach().float()).float().mean().item())
            if finite_ratio < 1.0:
                bad_entries.append((prefix, key, finite_ratio))
    if bad_entries:
        head = bad_entries[:20]
        raise RuntimeError(
            "Detected missing/non-finite tensors in critical prefixes: "
            f"{head} (total={len(bad_entries)})"
        )


def _assert_prefix_absmax_reasonable(
    state_dict: dict,
    prefixes: Iterable[str],
    threshold: float,
) -> None:
    if threshold <= 0:
        return
    bad_entries = []
    for prefix in prefixes:
        keys = _prefix_keys(state_dict, str(prefix))
        for key in keys:
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                continue
            absmax = float(value.detach().float().abs().max().item())
            if absmax > threshold:
                bad_entries.append((prefix, key, absmax))
    if bad_entries:
        head = bad_entries[:20]
        raise RuntimeError(
            f"Detected abnormal absmax in critical prefixes (> {threshold}): "
            f"{head} (total={len(bad_entries)})"
        )


def _remap_layerscale_weights(state_dict: dict) -> int:
    to_add = {}
    to_del = []
    remapped = 0
    for k in list(state_dict.keys()):
        if k.endswith(".ls1.weight") or k.endswith(".ls2.weight"):
            gamma_key = k.rsplit(".", 1)[0] + ".gamma"
            if gamma_key not in state_dict:
                to_add[gamma_key] = state_dict[k]
                remapped += 1
            to_del.append(k)
    for k in to_del:
        state_dict.pop(k, None)
    state_dict.update(to_add)
    return remapped


def _prune_state_dict(state_dict: dict) -> int:
    prefixes = (
        "vision_tower.text_model.",
        "language_model.model.model.mm_projector.",
    )
    exact_keys = {"vision_tower.logit_bias", "vision_tower.logit_scale"}
    to_del = []
    for k in state_dict.keys():
        if k in exact_keys or k.startswith(prefixes):
            to_del.append(k)
    for k in to_del:
        state_dict.pop(k, None)
    return len(to_del)


def _clean_output_dir(output_dir: Path) -> int:
    patterns = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
        "model-*.safetensors",
    ]
    removed = 0
    for pat in patterns:
        for p in output_dir.glob(pat):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def _resolve_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _required_prefixes() -> list[str]:
    # Keep this in sync with modeling_mapanything_llava3d_vlm.py
    return [
        "language_model",
        "vision_tower",
        "vision_projector",
        "geometric_model",
        "geometric_projector",
        "fusion_projector",
        "semantic_anchor_attention",
        "semantic_anchor_norm",
        "vision_semantic_attention",
        "task_queries",
        "task_query_to_geo_attn",
        "task_query_to_vis_attn",
        "task_fuse_mlp",
        "task_fuse_norm",
    ]


def _active_meta_sensitive_prefixes() -> list[str]:
    # Prefixes that must never stay on meta for training to be valid.
    return [
        "geometric_model.map_anything_model.encoder",
        "geometric_model.map_anything_model.info_sharing",
        "geometric_projector",
        "fusion_projector",
        "task_queries",
        "task_query_to_geo_attn",
        "task_query_to_vis_attn",
        "task_fuse_mlp",
        "task_fuse_norm",
    ]


def _validate_model_dir(path: str, name: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    if p.is_dir():
        cfg = p / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"{name} config.json not found under: {path}")


def build_base_checkpoint(args) -> None:
    _ensure_repo_root_on_path()

    from starVLA.mapanything_llava3d.model.configuration_mapanything_llava3d import (
        MapAnythingLlava3DConfig,
    )
    from starVLA.mapanything_llava3d.model.modeling_llava3d_v2 import (
        LLaVA3DForCausalLMV2,
    )
    from starVLA.mapanything_llava3d.model.modeling_mapanything import MapAnythingWrapper
    from starVLA.mapanything_llava3d.model.modeling_mapanything_llava3d_vlm import (
        MapAnythingLlava3DForConditionalGeneration,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        removed = _clean_output_dir(output_dir)
        if removed:
            print(f"Cleaned {removed} old weight shard files in {output_dir}")

    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    llava3d_path = _resolve_path(args.llava3d_path)
    siglip_path = _resolve_path(args.siglip_path)
    mapanything_path = _resolve_path(args.mapanything_path)
    _validate_model_dir(llava3d_path, "LLaVA3D")
    _validate_model_dir(siglip_path, "SigLIP")
    _validate_model_dir(mapanything_path, "MapAnything")

    vision_tower = AutoModel.from_pretrained(
        siglip_path, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None
    )

    text_cfg = AutoConfig.from_pretrained(llava3d_path, trust_remote_code=True)
    setattr(text_cfg, "llava3d_model_type", args.llava3d_model_type)
    setattr(text_cfg, "llava3d_pretrained_path", llava3d_path)
    setattr(text_cfg, "_name_or_path", llava3d_path)
    language_model = LLaVA3DForCausalLMV2(text_cfg)

    class _Cfg:
        pass

    map_cfg = _Cfg()
    setattr(map_cfg, "mapanything_model_name_or_path", mapanything_path)
    geometric_model = MapAnythingWrapper(map_cfg)

    config = MapAnythingLlava3DConfig(
        text_config=text_cfg,
        mapanything_config={"model_name_or_path": mapanything_path},
        vision_model_name_or_path=siglip_path,
        language_model_name_or_path=llava3d_path,
        mapanything_model_name_or_path=mapanything_path,
        use_spatial_token=bool(args.use_spatial_token),
        use_geometric_branch=bool(args.use_geometric_branch),
        task_token_num=int(args.task_token_num),
        image_token_index=-200,
    )

    model = MapAnythingLlava3DForConditionalGeneration(
        config=config,
        vision_tower=vision_tower,
        language_model=language_model,
        mapanything_model=geometric_model,
    )

    active_prefixes = _active_meta_sensitive_prefixes()
    meta_names = _meta_names_for_prefixes(model, active_prefixes, max_items=50)
    if meta_names:
        raise RuntimeError(
            "Assembled model still has meta tensors in critical active groups; "
            f"meta_names_head={meta_names[:20]} total={len(meta_names)}"
        )

    state_dict = model.state_dict()
    remapped = _remap_layerscale_weights(state_dict)
    pruned = _prune_state_dict(state_dict)
    if remapped:
        print(f"Remapped {remapped} LayerScale weights (ls*.weight -> ls*.gamma).")
    if pruned:
        print(f"Pruned {pruned} unused vision/text/mm_projector weights from checkpoint.")
    required_prefixes = _required_prefixes()
    missing = [p for p in required_prefixes if not _has_prefix(state_dict, p)]
    if missing:
        raise RuntimeError(f"Missing parameter groups in assembled model: {missing}")

    critical_finite_prefixes = [
        "semantic_anchor_attention",
        "semantic_anchor_norm",
        "vision_semantic_attention",
        "task_queries",
        "task_query_to_geo_attn",
        "task_query_to_vis_attn",
        "task_fuse_mlp",
        "task_fuse_norm",
        "geometric_projector",
        "fusion_projector",
    ]
    _assert_prefix_tensors_finite(state_dict, critical_finite_prefixes)
    _assert_prefix_absmax_reasonable(
        state_dict,
        critical_finite_prefixes,
        threshold=float(args.weight_absmax_threshold),
    )
    num_new_task_tensors = sum(
        len(_prefix_keys(state_dict, p))
        for p in (
            "task_queries",
            "task_query_to_geo_attn",
            "task_query_to_vis_attn",
            "task_fuse_mlp",
            "task_fuse_norm",
        )
    )
    if num_new_task_tensors == 0:
        raise RuntimeError("No fixed-K task token parameters found in assembled state_dict.")
    print(
        "OK: critical fusion/task tensors are present+finite, "
        f"task_token_num={int(args.task_token_num)}, "
        f"num_task_related_tensors={num_new_task_tensors}."
    )
    print("OK: assembled model contains all required parameter groups.")

    requested_safe_serialization = bool(args.safe_serialization)
    try:
        model.save_pretrained(
            output_dir,
            safe_serialization=requested_safe_serialization,
            state_dict=state_dict,
        )
    except RuntimeError as e:
        # MapAnything has many intentionally shared/tied tensors.
        # Some transformers versions reject shared tensors under safetensors export.
        msg = str(e)
        if requested_safe_serialization and ("shared tensors" in msg or "tensor sharing" in msg):
            print(
                "Warning: safetensors export failed due to shared tensors. "
                "Retrying with safe_serialization=False (pytorch_model.bin)."
            )
            model.save_pretrained(
                output_dir,
                safe_serialization=False,
                state_dict=state_dict,
            )
        else:
            raise
    config.save_pretrained(output_dir)

    if bool(args.post_validate_load):
        print("Running post-save validation load...")
        reloaded = MapAnythingLlava3DForConditionalGeneration.from_pretrained(
            output_dir,
            low_cpu_mem_usage=False,
            device_map=None,
            skip_language_model_preload=True,
            skip_geometric_model_preload=True,
        )
        reloaded_state = reloaded.state_dict()
        _assert_prefix_tensors_finite(reloaded_state, critical_finite_prefixes)
        _assert_prefix_absmax_reasonable(
            reloaded_state,
            critical_finite_prefixes,
            threshold=float(args.weight_absmax_threshold),
        )
        reloaded_meta = _meta_names_for_prefixes(reloaded, active_prefixes, max_items=50)
        if reloaded_meta:
            raise RuntimeError(
                "Post-validate load still has meta tensors in active groups; "
                f"meta_names_head={reloaded_meta[:20]} total={len(reloaded_meta)}"
            )
        print("OK: post-save validation load passed.")

    print(f"Saved base VLM checkpoint to: {output_dir}")


def main():
    args = parse_args()
    build_base_checkpoint(args)


if __name__ == "__main__":
    main()
