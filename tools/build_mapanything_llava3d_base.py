from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser.add_argument("--use-geometric-branch", action="store_true", default=True)
    parser.add_argument(
        "--disable-geometric-branch", dest="use_geometric_branch", action="store_false"
    )
    parser.add_argument("--use-spatial-token", action="store_true", default=True)
    parser.add_argument("--disable-spatial-token", dest="use_spatial_token", action="store_false")
    parser.add_argument("--safe-serialization", action="store_true")
    return parser.parse_args()


def _has_prefix(state_dict: dict, prefix: str) -> bool:
    for k in state_dict.keys():
        if k.startswith(prefix):
            return True
    return False

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


def _resolve_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


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
        image_token_index=-200,
    )

    model = MapAnythingLlava3DForConditionalGeneration(
        config=config,
        vision_tower=vision_tower,
        language_model=language_model,
        mapanything_model=geometric_model,
    )

    state_dict = model.state_dict()
    remapped = _remap_layerscale_weights(state_dict)
    pruned = _prune_state_dict(state_dict)
    if remapped:
        print(f"Remapped {remapped} LayerScale weights (ls*.weight -> ls*.gamma).")
    if pruned:
        print(f"Pruned {pruned} unused vision/text/mm_projector weights from checkpoint.")
    required_prefixes = [
        "language_model",
        "vision_tower",
        "vision_projector",
        "geometric_model",
        "geometric_projector",
        "fusion_projector",
    ]
    missing = [p for p in required_prefixes if not _has_prefix(state_dict, p)]
    if missing:
        raise RuntimeError(f"Missing parameter groups in assembled model: {missing}")
    print("OK: assembled model contains all required parameter groups.")

    model.save_pretrained(
        output_dir,
        safe_serialization=bool(args.safe_serialization),
        state_dict=state_dict,
    )
    config.save_pretrained(output_dir)
    print(f"Saved base VLM checkpoint to: {output_dir}")


def main():
    args = parse_args()
    build_base_checkpoint(args)


if __name__ == "__main__":
    main()
