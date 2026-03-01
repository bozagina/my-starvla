#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

from starVLA.mapanything_llava3d.model.modeling_mapanything_llava3d_vlm import (
    MapAnythingLlava3DForConditionalGeneration,
)
from starVLA.mapanything_llava3d.model.processing_mapanything_llava3d import (
    MapAnythingLlava3DProcessor,
)
from starVLA.model.modules.vlm.MapAnythingLlava3D import _suspend_meta_device


def _to_namespace_dict(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_prompt_and_span(instruction: str, cot_prompt: Optional[str]) -> Tuple[str, Tuple[int, int]]:
    instruction = "" if instruction is None else str(instruction)
    if isinstance(cot_prompt, str) and "{instruction}" in cot_prompt:
        prompt = cot_prompt.replace("{instruction}", instruction)
        prefix = cot_prompt.split("{instruction}", 1)[0]
        return prompt, (len(prefix), len(prefix) + len(instruction))
    if isinstance(cot_prompt, str) and cot_prompt:
        prompt = cot_prompt.replace("{instruction}", instruction)
        start = prompt.find(instruction)
        if start >= 0:
            return prompt, (start, start + len(instruction))
        return prompt, (0, 0)
    return instruction, (0, len(instruction))


def _inspect_llava_vision_tower(model: MapAnythingLlava3DForConditionalGeneration) -> Dict[str, object]:
    out: Dict[str, object] = {
        "llava_vision_available_flag": bool(getattr(model, "_llava_vision_available", True)),
        "has_base_model": False,
        "has_get_vision_tower": False,
        "has_encode_images": False,
        "has_mm_projector": False,
        "vision_tower_path": None,
        "vision_tower_is_loaded": None,
        "llava3d_pretrained_path": None,
    }
    lang_model = getattr(model, "language_model", None)
    base_model = getattr(lang_model, "model", None)
    llava_cfg = getattr(lang_model, "config", None)
    if llava_cfg is not None:
        out["llava3d_pretrained_path"] = getattr(llava_cfg, "llava3d_pretrained_path", None)
    if base_model is None:
        return out
    out["has_base_model"] = True
    out["has_get_vision_tower"] = bool(hasattr(base_model, "get_vision_tower"))
    out["has_encode_images"] = bool(hasattr(base_model, "encode_images"))
    out["has_mm_projector"] = bool(hasattr(base_model, "mm_projector"))
    if hasattr(base_model, "get_vision_tower"):
        try:
            vt = base_model.get_vision_tower()
            vt_path = getattr(vt, "vision_tower", None) if vt is not None else None
            out["vision_tower_path"] = vt_path
            if vt is not None and hasattr(vt, "is_loaded"):
                out["vision_tower_is_loaded"] = bool(vt.is_loaded)
            elif vt is None:
                out["vision_tower_is_loaded"] = False
        except Exception as e:
            out["vision_tower_error"] = str(e)
    return out


def _reshape_multiview(pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError("pixel_values must be torch.Tensor")
    if pixel_values.ndim == 5:
        b, v = pixel_values.shape[:2]
        return pixel_values.reshape(b * v, *pixel_values.shape[2:]), (b, v)
    return pixel_values, None


def _encode_llava_vision_tokens(
    model: MapAnythingLlava3DForConditionalGeneration,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    base_model = getattr(getattr(model, "language_model", None), "model", None)
    if base_model is None or not hasattr(base_model, "encode_images"):
        raise RuntimeError("LLaVA base model or encode_images is unavailable.")

    pv, mv_shape = _reshape_multiview(pixel_values)
    with torch.no_grad():
        vision_feats = base_model.encode_images(pv)
    if not isinstance(vision_feats, torch.Tensor):
        raise RuntimeError("encode_images did not return Tensor.")

    if vision_feats.shape[-1] != int(model.hidden_size):
        vision_feats = model.vision_projector(vision_feats)

    if mv_shape is not None:
        b, v = mv_shape
        vision_feats = vision_feats.view(b, v * vision_feats.shape[1], vision_feats.shape[2])
    return vision_feats


def _load_clip_vision_tower_class(llava_repo: Path):
    clip_file = llava_repo / "llava" / "model" / "multimodal_encoder" / "clip_encoder.py"
    if not clip_file.exists():
        raise FileNotFoundError(f"Cannot find clip_encoder.py under repo: {clip_file}")
    spec = importlib.util.spec_from_file_location("llava3d_clip_encoder_local", str(clip_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {clip_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "CLIPVisionTower", None)
    if cls is None:
        raise RuntimeError("CLIPVisionTower class not found in local LLaVA-3D clip_encoder.py")
    return cls


def _resolve_llava_pretrained_path(model: MapAnythingLlava3DForConditionalGeneration) -> Optional[str]:
    candidates = []
    lang_model = getattr(model, "language_model", None)
    if lang_model is not None:
        cfg = getattr(lang_model, "config", None)
        if cfg is not None:
            candidates.append(getattr(cfg, "llava3d_pretrained_path", None))
        base_model = getattr(lang_model, "model", None)
        if base_model is not None:
            bcfg = getattr(base_model, "config", None)
            if bcfg is not None:
                candidates.append(getattr(bcfg, "llava3d_pretrained_path", None))
    candidates.append(getattr(model.config, "language_model_name_or_path", None))
    for c in candidates:
        if isinstance(c, str) and c:
            return c
    return None


def _encode_llava_vision_tokens_with_local_repo(
    model: MapAnythingLlava3DForConditionalGeneration,
    image: Image.Image,
    llava_repo: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {"mode": "local_repo_rebuild"}
    llava_pretrained_path = _resolve_llava_pretrained_path(model)
    info["llava_pretrained_path"] = llava_pretrained_path
    if llava_pretrained_path is None:
        raise RuntimeError("Cannot resolve llava3d_pretrained_path from current model config.")

    llava_cfg = AutoConfig.from_pretrained(llava_pretrained_path, trust_remote_code=True)
    vision_tower_name = getattr(llava_cfg, "mm_vision_tower", getattr(llava_cfg, "vision_tower", None))
    info["resolved_vision_tower_name"] = vision_tower_name
    if not isinstance(vision_tower_name, str) or not vision_tower_name:
        raise RuntimeError(f"Invalid mm_vision_tower in {llava_pretrained_path}: {vision_tower_name}")

    CLIPVisionTower = _load_clip_vision_tower_class(llava_repo)
    tower_args = SimpleNamespace(
        mm_vision_select_layer=int(getattr(llava_cfg, "mm_vision_select_layer", -2)),
        mm_vision_select_feature=str(getattr(llava_cfg, "mm_vision_select_feature", "patch")),
        unfreeze_mm_vision_tower=False,
    )
    tower = CLIPVisionTower(vision_tower_name, args=tower_args, delay_load=False)
    tower = tower.to(device=device)
    tower.eval()

    pixel_values = tower.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"].to(device=device)
    with torch.no_grad():
        feats = tower(pixel_values)
    if not isinstance(feats, torch.Tensor):
        raise RuntimeError("Local CLIPVisionTower did not return tensor features.")
    info["raw_vision_feats_shape"] = tuple(feats.shape)

    base_model = getattr(getattr(model, "language_model", None), "model", None)
    if base_model is None or not hasattr(base_model, "mm_projector"):
        raise RuntimeError("Current language base model does not provide mm_projector.")
    mm_projector = getattr(base_model, "mm_projector")
    if mm_projector is None:
        raise RuntimeError("mm_projector is None.")
    projector_dtype = None
    try:
        projector_dtype = next(mm_projector.parameters()).dtype
    except StopIteration:
        projector_dtype = feats.dtype
    with torch.no_grad():
        feats = mm_projector(feats.to(dtype=projector_dtype))
    feats = feats.to(dtype=torch.float32)
    info["projected_vision_feats_shape"] = tuple(feats.shape)
    info["projected_hidden"] = int(feats.shape[-1])
    return feats, info


def _build_soft_mask(
    *,
    vision_tokens: torch.Tensor,
    geometric_tokens: torch.Tensor,
    language_queries: torch.Tensor,
    language_query_mask: Optional[torch.Tensor],
    soft_mask_num_heads: int,
    soft_mask_query_agg: str,
    soft_mask_head_agg: str,
    soft_mask_channel_mode: str,
    soft_mask_lambda: float,
    soft_mask_logit_scale: float,
    soft_mask_temperature: float,
    soft_mask_score_norm: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not (
        isinstance(vision_tokens, torch.Tensor)
        and isinstance(geometric_tokens, torch.Tensor)
        and isinstance(language_queries, torch.Tensor)
    ):
        raise ValueError("vision_tokens/geometric_tokens/language_queries must all be tensors.")
    if vision_tokens.ndim != 3 or geometric_tokens.ndim != 3 or language_queries.ndim != 3:
        raise ValueError("All token inputs must be [B, T, H].")
    if vision_tokens.shape[0] != geometric_tokens.shape[0] or vision_tokens.shape[0] != language_queries.shape[0]:
        raise ValueError("Batch size mismatch among tokens.")
    if vision_tokens.shape[2] != geometric_tokens.shape[2] or vision_tokens.shape[2] != language_queries.shape[2]:
        raise ValueError("Hidden size mismatch among tokens.")

    bsz = int(vision_tokens.shape[0])
    token_n = int(min(vision_tokens.shape[1], geometric_tokens.shape[1]))
    if token_n <= 0:
        raise ValueError("No valid tokens for soft-mask.")

    dtype = torch.float32
    device = language_queries.device
    q = language_queries.to(device=device, dtype=dtype)
    v = vision_tokens[:, :token_n, :].to(device=device, dtype=dtype)
    g = geometric_tokens[:, :token_n, :].to(device=device, dtype=dtype)

    if soft_mask_score_norm == "l2_only":
        q = F.normalize(q, dim=-1)
        v = F.normalize(v, dim=-1)
        g = F.normalize(g, dim=-1)

    query_mask = None
    if isinstance(language_query_mask, torch.Tensor):
        qm = language_query_mask.to(device=device, dtype=torch.bool)
        if qm.ndim == 2 and qm.shape[0] == bsz and qm.shape[1] == q.shape[1]:
            query_mask = qm

    if query_mask is None:
        query_weight = torch.full((bsz, q.shape[1]), 1.0 / max(int(q.shape[1]), 1), device=device, dtype=dtype)
    else:
        query_weight = query_mask.to(dtype=dtype)
        query_weight = query_weight / query_weight.sum(dim=1, keepdim=True).clamp_min(1.0)

    num_heads_cfg = max(1, int(soft_mask_num_heads))
    hidden = int(q.shape[-1])
    if hidden % num_heads_cfg != 0:
        num_heads = 1
    else:
        num_heads = num_heads_cfg
    head_dim = max(1, hidden // num_heads)

    def to_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(bsz, x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    qh, vh, gh = to_heads(q), to_heads(v), to_heads(g)
    temp = float(max(soft_mask_temperature, 1e-6))
    scale = float(max(soft_mask_logit_scale, 1e-6)) / temp
    if soft_mask_score_norm == "sqrt_only":
        scale = scale / max(math.sqrt(float(head_dim)), 1e-6)

    logits_vis = torch.einsum("bhqd,bhkd->bhqk", qh, vh) * scale
    logits_geo = torch.einsum("bhqd,bhkd->bhqk", qh, gh) * scale
    attn_vis = torch.softmax(logits_vis, dim=-1)
    attn_geo = torch.softmax(logits_geo, dim=-1)

    if soft_mask_query_agg == "max":
        if query_mask is not None:
            mask_expand = query_mask.unsqueeze(1).unsqueeze(-1)
            attn_vis_reduce = attn_vis.masked_fill(~mask_expand, 0.0)
            attn_geo_reduce = attn_geo.masked_fill(~mask_expand, 0.0)
        else:
            attn_vis_reduce = attn_vis
            attn_geo_reduce = attn_geo
        alpha_vis_head = attn_vis_reduce.max(dim=2).values
        alpha_geo_head = attn_geo_reduce.max(dim=2).values
    else:
        q_w = query_weight.unsqueeze(1).unsqueeze(-1)
        alpha_vis_head = torch.sum(attn_vis * q_w, dim=2)
        alpha_geo_head = torch.sum(attn_geo * q_w, dim=2)

    if soft_mask_head_agg == "max":
        alpha_vis = alpha_vis_head.max(dim=1).values
        alpha_geo = alpha_geo_head.max(dim=1).values
    else:
        alpha_vis = alpha_vis_head.mean(dim=1)
        alpha_geo = alpha_geo_head.mean(dim=1)

    lam = float(max(0.0, min(1.0, soft_mask_lambda)))
    if soft_mask_channel_mode == "vision_only":
        alpha_pre = alpha_vis
    elif soft_mask_channel_mode == "geo_only":
        alpha_pre = alpha_geo
    else:
        alpha_pre = (1.0 - lam) * alpha_vis + lam * alpha_geo
    alpha_pre = alpha_pre.clamp_min(0.0)
    alpha = alpha_pre / alpha_pre.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    with torch.no_grad():
        query_weight_f = query_weight.detach().float()
        attn_top1_vis_q = attn_vis.detach().float().amax(dim=-1)
        attn_top1_vis_mean = float(
            ((attn_top1_vis_q * query_weight_f.unsqueeze(1)).sum(dim=-1).mean(dim=1)).mean().item()
        )
        alpha_norm = alpha.detach().float()
        entropy = -torch.sum(alpha_norm * torch.log(alpha_norm.clamp_min(1e-9)), dim=-1).mean().item()
        alpha_max = alpha_norm.amax(dim=-1).mean().item()
        topk = min(int(token_n), 32)
        topk_mass = alpha_norm.topk(topk, dim=-1).values.sum(dim=-1).mean().item()

    stats = {
        "token_n": float(token_n),
        "attn_top1_mean_vis": float(attn_top1_vis_mean),
        "alpha_entropy": float(entropy),
        "alpha_max_mean": float(alpha_max),
        "topk_mass_32": float(topk_mass),
    }
    return alpha, stats


def _compare_alpha(alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> Dict[str, float]:
    a = alpha_a.detach().float().view(alpha_a.shape[0], -1)
    b = alpha_b.detach().float().view(alpha_b.shape[0], -1)
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    l1 = torch.mean(torch.abs(a - b)).item()
    l2 = torch.sqrt(torch.mean((a - b) ** 2)).item()
    return {"alpha_cosine_mean": float(cos), "alpha_l1_mean": float(l1), "alpha_l2_rms": float(l2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose vision-tower fallback impact on soft-mask.")
    parser.add_argument("--run-config", type=str, required=True, help="Path to training yaml config.")
    parser.add_argument("--image", type=str, required=True, help="Path to a test RGB image.")
    parser.add_argument("--instruction", type=str, required=True, help="Raw instruction text (without CoT template).")
    parser.add_argument(
        "--llava-repo",
        type=str,
        default="/Users/bazinga/code/my-starvla/LLaVA-3D",
        help="Path to local cloned LLaVA-3D repo for rebuilding CLIP vision tower path.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument("--output-json", type=str, default="", help="Optional output json path.")
    args = parser.parse_args()

    run_cfg = _to_namespace_dict(Path(args.run_config))
    ma_cfg = run_cfg["framework"]["mapanything_llava3d"]
    act_cfg = run_cfg["framework"]["action_model"]
    ds_cfg = run_cfg["datasets"]["vla_data"]
    base_vlm = str(ma_cfg["base_vlm"])
    cot_prompt = ds_cfg.get("CoT_prompt", None)

    if not Path(base_vlm).exists():
        raise FileNotFoundError(f"base_vlm does not exist: {base_vlm}")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"image does not exist: {args.image}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    with _suspend_meta_device():
        model = MapAnythingLlava3DForConditionalGeneration.from_pretrained(
            base_vlm,
            low_cpu_mem_usage=False,
            device_map=None,
            skip_language_model_preload=True,
            skip_geometric_model_preload=True,
        )
    model = model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model.config.language_model_name_or_path, trust_remote_code=True)
    processor = MapAnythingLlava3DProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        statistics=None,
        intrinsic_config=None,
        action_config=None,
        image_token_joiner="auto",
    )

    prompt, span = _build_prompt_and_span(args.instruction, cot_prompt)
    image = Image.open(args.image).convert("RGB")
    batch = processor(
        text=[prompt],
        images=[image],
        instruction_char_spans=[span],
        return_tensors="pt",
    ).data
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    with torch.no_grad():
        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

    lang_q = outputs.language_queries
    lang_q_mask = outputs.language_query_mask
    vis_curr = outputs.vision_hidden_states
    geo = outputs.geometric_hidden_states
    if not (isinstance(lang_q, torch.Tensor) and isinstance(vis_curr, torch.Tensor) and isinstance(geo, torch.Tensor)):
        raise RuntimeError(
            "language_queries/vision_hidden_states/geometric_hidden_states unavailable. "
            "Cannot build soft-mask for comparison."
        )

    mask_cfg = {
        "soft_mask_num_heads": int(act_cfg.get("soft_mask_num_heads", 4)),
        "soft_mask_query_agg": str(act_cfg.get("soft_mask_query_agg", "max")),
        "soft_mask_head_agg": str(act_cfg.get("soft_mask_head_agg", "max")),
        "soft_mask_channel_mode": str(act_cfg.get("soft_mask_channel_mode", "vision_only")),
        "soft_mask_lambda": float(act_cfg.get("soft_mask_lambda", 0.3)),
        "soft_mask_logit_scale": float(act_cfg.get("soft_mask_logit_scale", 16.0)),
        "soft_mask_temperature": float(act_cfg.get("soft_mask_temperature", 1.0)),
        "soft_mask_score_norm": str(act_cfg.get("soft_mask_score_norm", "l2_only")),
    }
    alpha_current, current_stats = _build_soft_mask(
        vision_tokens=vis_curr,
        geometric_tokens=geo,
        language_queries=lang_q,
        language_query_mask=lang_q_mask,
        **mask_cfg,
    )

    vision_tower_info = _inspect_llava_vision_tower(model)
    result = {
        "base_vlm": base_vlm,
        "device": str(device),
        "vision_tower_info": vision_tower_info,
        "current_mask_stats": current_stats,
        "mask_config": mask_cfg,
    }

    llava_alt_ok = False
    try:
        llava_vis = _encode_llava_vision_tokens(model, batch["pixel_values"])
        alpha_llava, llava_stats = _build_soft_mask(
            vision_tokens=llava_vis,
            geometric_tokens=geo,
            language_queries=lang_q,
            language_query_mask=lang_q_mask,
            **mask_cfg,
        )
        result["llava_vision_mask_stats"] = llava_stats
        result["current_vs_llava_alpha_diff"] = _compare_alpha(alpha_current, alpha_llava)
        llava_alt_ok = True
    except Exception as e:
        result["llava_vision_mask_stats"] = None
        result["current_vs_llava_alpha_diff"] = None
        result["llava_alt_error"] = str(e)

    if not llava_alt_ok:
        try:
            llava_repo = Path(args.llava_repo)
            llava_vis_rebuild, rebuild_info = _encode_llava_vision_tokens_with_local_repo(
                model=model,
                image=image,
                llava_repo=llava_repo,
                device=device,
            )
            alpha_llava_rebuild, llava_rebuild_stats = _build_soft_mask(
                vision_tokens=llava_vis_rebuild,
                geometric_tokens=geo,
                language_queries=lang_q,
                language_query_mask=lang_q_mask,
                **mask_cfg,
            )
            result["llava_rebuild_info"] = rebuild_info
            result["llava_vision_mask_stats"] = llava_rebuild_stats
            result["current_vs_llava_alpha_diff"] = _compare_alpha(alpha_current, alpha_llava_rebuild)
            result["llava_alt_error"] = None
            llava_alt_ok = True
        except Exception as e:
            result["llava_rebuild_info"] = None
            if "llava_alt_error" not in result:
                result["llava_alt_error"] = str(e)
            else:
                result["llava_alt_error_rebuild"] = str(e)

    result["llava_alt_available"] = llava_alt_ok
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
