#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
import types

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

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
    mm_proj, mm_proj_path = _find_mm_projector(base_model)
    out["has_mm_projector"] = bool(mm_proj is not None)
    out["mm_projector_attr_path"] = mm_proj_path
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


def _find_mm_projector(base_model) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    if base_model is None:
        return None, None
    candidates = [
        (base_model, "base_model"),
        (getattr(base_model, "model", None), "base_model.model"),
    ]
    if hasattr(base_model, "get_model"):
        try:
            candidates.append((base_model.get_model(), "base_model.get_model()"))
        except Exception:
            pass
    for obj, prefix in candidates:
        if obj is None:
            continue
        mm_proj = getattr(obj, "mm_projector", None)
        if mm_proj is not None:
            return mm_proj, f"{prefix}.mm_projector"
    return None, None


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
    llava_pretrained_path_override: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    info: Dict[str, object] = {"mode": "local_repo_rebuild"}
    llava_pretrained_path = (
        llava_pretrained_path_override
        if isinstance(llava_pretrained_path_override, str) and llava_pretrained_path_override
        else _resolve_llava_pretrained_path(model)
    )
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
    mm_projector, mm_proj_path = _find_mm_projector(base_model)
    info["mm_projector_attr_path"] = mm_proj_path
    if mm_projector is None:
        raise RuntimeError("Current language base model does not provide mm_projector.")
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


def _import_official_llava_modules(llava_repo: Path):
    if not llava_repo.exists():
        raise FileNotFoundError(f"LLaVA-3D repo does not exist: {llava_repo}")
    repo_str = str(llava_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    loaded_llava = sys.modules.get("llava", None)
    if loaded_llava is not None:
        loaded_file = getattr(loaded_llava, "__file__", "") or ""
        if loaded_file and not loaded_file.startswith(repo_str):
            drop_keys = [k for k in sys.modules.keys() if k == "llava" or k.startswith("llava.")]
            for k in drop_keys:
                sys.modules.pop(k, None)

    _ensure_torch_scatter_stub()

    from llava.constants import (  # noqa: WPS433
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import process_images, tokenizer_image_token  # noqa: WPS433

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IMAGE_PATCH_TOKEN": DEFAULT_IMAGE_PATCH_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "IGNORE_INDEX": IGNORE_INDEX,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
    }


def _ensure_torch_scatter_stub() -> None:
    if "torch_scatter" in sys.modules:
        return
    try:
        import torch_scatter  # noqa: F401,WPS433
        return
    except Exception:
        pass

    stub = types.ModuleType("torch_scatter")

    def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None):
        if not isinstance(src, torch.Tensor) or not isinstance(index, torch.Tensor):
            raise TypeError("scatter_mean expects tensor inputs.")
        if index.ndim != 1:
            index = index.reshape(-1)
        if dim < 0:
            dim += src.ndim
        if dim != 0:
            transposed = src.transpose(0, dim).contiguous()
            out_t = scatter_mean(transposed, index, dim=0, dim_size=dim_size)
            return out_t.transpose(0, dim).contiguous()
        if src.shape[0] != index.shape[0]:
            raise ValueError(f"scatter_mean shape mismatch: src0={src.shape[0]}, index={index.shape[0]}")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = (dim_size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
        if index.numel() == 0:
            return out
        expand_index = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
        out.scatter_add_(0, expand_index, src)
        cnt = torch.bincount(index, minlength=dim_size).to(device=src.device, dtype=src.dtype).clamp_min(1.0)
        if src.ndim > 1:
            cnt = cnt.view(dim_size, *([1] * (src.ndim - 1)))
        return out / cnt

    stub.scatter_mean = scatter_mean  # type: ignore[attr-defined]
    stub.__dict__["_is_stub"] = True
    sys.modules["torch_scatter"] = stub


def _build_language_queries_from_hidden(
    *,
    hidden_states: torch.Tensor,
    language_mask: torch.Tensor,
    select_mode: str,
    max_tokens: int,
    topk_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3 or language_mask.ndim != 2:
        raise ValueError("hidden_states must be [B,S,H], language_mask must be [B,S]")
    bsz, seq_len, hidden = hidden_states.shape
    if language_mask.shape[0] != bsz or language_mask.shape[1] != seq_len:
        raise ValueError("language_mask shape mismatch")

    max_tokens = max(1, int(max_tokens))
    topk_tokens = max(1, int(topk_tokens))
    select_mode = str(select_mode).strip().lower()
    if select_mode not in ("uniform", "topk_norm", "summary_topk"):
        select_mode = "summary_topk"

    query_lens = []
    valid_indices = []
    for bid in range(bsz):
        idx = torch.nonzero(language_mask[bid], as_tuple=False).flatten()
        if idx.numel() == 0:
            idx = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
        valid_indices.append(idx)
        if select_mode == "uniform":
            query_lens.append(min(max_tokens, int(idx.numel())))
        else:
            query_lens.append(min(min(max_tokens, topk_tokens), int(idx.numel())))
    qlen = max(1, max(query_lens))
    queries = hidden_states.new_zeros((bsz, qlen, hidden))
    qmask = torch.zeros((bsz, qlen), dtype=torch.bool, device=hidden_states.device)

    for bid in range(bsz):
        idx = valid_indices[bid]
        pool = hidden_states[bid, idx]
        if select_mode == "uniform":
            if idx.numel() > qlen:
                sample_pos = torch.linspace(0, float(idx.numel() - 1), steps=qlen, device=hidden_states.device).long()
                idx = idx[sample_pos]
            token_query = hidden_states[bid, idx]
        elif select_mode == "topk_norm":
            take_n = min(qlen, int(pool.shape[0]))
            score = pool.float().norm(dim=-1)
            sel = torch.topk(score, k=take_n, largest=True).indices
            token_query = pool.index_select(0, sel)
        else:
            take_n = min(qlen, int(pool.shape[0]))
            summary = pool.float().mean(dim=0, keepdim=True)
            score = torch.matmul(F.normalize(pool.float(), dim=-1), F.normalize(summary, dim=-1).squeeze(0))
            sel = torch.topk(score, k=take_n, largest=True).indices
            token_query = pool.index_select(0, sel)
        n_take = token_query.shape[0]
        queries[bid, :n_take] = token_query
        qmask[bid, :n_take] = True
    return queries, qmask


class _ZeroPositionalEncoder:
    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)

    def encode_pe(self, xyz: torch.Tensor) -> torch.Tensor:
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 3:
            raise ValueError("xyz must be a [B, N, 3] tensor.")
        return xyz.new_zeros((xyz.shape[0], xyz.shape[1], self.hidden_size))


class _ZeroPromptEncoder:
    def __init__(self, hidden_size: int):
        self.hidden_size = int(hidden_size)

    def __call__(self, clicks: torch.Tensor) -> torch.Tensor:
        if not isinstance(clicks, torch.Tensor):
            raise TypeError(f"clicks must be torch.Tensor, got: {type(clicks)}")
        if clicks.ndim == 1:
            clicks = clicks.unsqueeze(0)
        if clicks.ndim != 2:
            raise ValueError(f"clicks must be [N, D], got: {tuple(clicks.shape)}")
        return clicks.new_zeros((clicks.shape[0], self.hidden_size))


class _DummyVideoTower:
    def __init__(self, pe_hidden_size: int, prompt_hidden_size: int):
        self.is_loaded = True
        self.video_tower = _ZeroPositionalEncoder(pe_hidden_size)
        self.prompt_encoder = _ZeroPromptEncoder(prompt_hidden_size)
        self.video_processor = None

    def load_model(self, *args, **kwargs):
        return None

    def to(self, *args, **kwargs):
        return self


def _run_official_llava_branch(
    *,
    llava_repo: Path,
    llava_ckpt: Path,
    image: Image.Image,
    instruction: str,
    cot_prompt: Optional[str],
    device: torch.device,
    select_mode: str,
    max_tokens: int,
    topk_tokens: int,
) -> Dict[str, object]:
    mods = _import_official_llava_modules(llava_repo)
    llava_cfg = AutoConfig.from_pretrained(str(llava_ckpt), trust_remote_code=True)
    if hasattr(llava_cfg, "mm_video_tower"):
        setattr(llava_cfg, "mm_video_tower", None)
    if hasattr(llava_cfg, "video_tower"):
        setattr(llava_cfg, "video_tower", None)

    tokenizer = AutoTokenizer.from_pretrained(str(llava_ckpt), use_fast=False, trust_remote_code=True)
    llava_model = AutoModelForCausalLM.from_pretrained(
        str(llava_ckpt),
        config=llava_cfg,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
        device_map={"": str(device)} if str(device).startswith("cuda") else {"": "cpu"},
    )
    llava_model.eval()

    patch_info = {
        "video_tower_missing": False,
        "injected_dummy_video_tower": False,
        "dummy_video_pe_hidden_size": None,
        "dummy_prompt_hidden_size": None,
        "patched_encode_images": False,
        "patched_encode_prompts": False,
    }
    if hasattr(llava_model, "get_video_tower"):
        try:
            vt = llava_model.get_video_tower()
        except Exception:
            vt = None
        if vt is None:
            patch_info["video_tower_missing"] = True
            pe_hidden_size = int(getattr(llava_model.config, "mm_hidden_size", 1024))
            prompt_hidden_size = int(getattr(llava_model.config, "hidden_size", 4096))
            try:
                llava_model.get_model().video_tower = _DummyVideoTower(pe_hidden_size, prompt_hidden_size)
                patch_info["injected_dummy_video_tower"] = True
                patch_info["dummy_video_pe_hidden_size"] = float(pe_hidden_size)
                patch_info["dummy_prompt_hidden_size"] = float(prompt_hidden_size)
            except Exception:
                patch_info["injected_dummy_video_tower"] = False

            def _encode_images_no_video(self, images):
                image_features = self.get_model().get_vision_tower()(images)
                image_features = self.get_model().mm_projector(image_features)
                return image_features

            def _encode_prompts_no_video(self, clicks):
                if not isinstance(clicks, torch.Tensor):
                    raise TypeError(f"clicks must be torch.Tensor, got {type(clicks)}")
                hidden = int(getattr(self.config, "hidden_size", 4096))
                return clicks.new_zeros((clicks.shape[0], hidden))

            llava_model.encode_images = types.MethodType(_encode_images_no_video, llava_model)
            llava_model.encode_prompts = types.MethodType(_encode_prompts_no_video, llava_model)
            patch_info["patched_encode_images"] = True
            patch_info["patched_encode_prompts"] = True

    mm_use_im_start_end = bool(getattr(llava_model.config, "mm_use_im_start_end", False))
    mm_use_im_patch_token = bool(getattr(llava_model.config, "mm_use_im_patch_token", True))
    if mm_use_im_patch_token:
        tokenizer.add_tokens([mods["DEFAULT_IMAGE_PATCH_TOKEN"]], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([mods["DEFAULT_IM_START_TOKEN"], mods["DEFAULT_IM_END_TOKEN"]], special_tokens=True)
    llava_model.resize_token_embeddings(len(tokenizer))

    vision_tower = llava_model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("Official LLaVA model has no vision_tower.")
    if hasattr(vision_tower, "is_loaded") and (not vision_tower.is_loaded):
        vision_tower.load_model(device_map=str(device) if str(device).startswith("cuda") else "cpu")
    if str(device).startswith("cuda"):
        vision_tower.to(device=str(device), dtype=torch.float16)
    image_processor = vision_tower.image_processor

    image_token = (
        f"{mods['DEFAULT_IM_START_TOKEN']}{mods['DEFAULT_IMAGE_TOKEN']}{mods['DEFAULT_IM_END_TOKEN']}"
        if mm_use_im_start_end
        else mods["DEFAULT_IMAGE_TOKEN"]
    )
    prompt_core, _ = _build_prompt_and_span(instruction, cot_prompt)
    prompt = f"{image_token}\n{prompt_core}"

    input_ids = mods["tokenizer_image_token"](
        prompt,
        tokenizer,
        mods["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    labels = input_ids.clone()

    image_tensor = mods["process_images"]([image], image_processor, llava_model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)

    with torch.no_grad():
        (
            _,
            position_ids2,
            attn2,
            _,
            inputs_embeds2,
            labels2,
        ) = llava_model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=image_tensor,
            depths=None,
            poses=None,
            intrinsics=None,
            lengths=None,
            clicks=None,
            image_sizes=[list(image.size)],
        )
        out = llava_model(
            input_ids=None,
            attention_mask=attn2,
            position_ids=position_ids2,
            inputs_embeds=inputs_embeds2,
            labels=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_last = out.hidden_states[-1]
        vis_tokens = llava_model.encode_images(image_tensor)

    if not isinstance(hidden_last, torch.Tensor):
        raise RuntimeError("Official LLaVA hidden states unavailable.")
    if not isinstance(vis_tokens, torch.Tensor):
        raise RuntimeError("Official LLaVA vision tokens unavailable.")
    if attn2 is None:
        attn_mask_lang = torch.ones(hidden_last.shape[:2], dtype=torch.bool, device=hidden_last.device)
    else:
        attn_mask_lang = attn2.to(device=hidden_last.device, dtype=torch.bool)
    if labels2 is None:
        raise RuntimeError("labels2 is None, cannot build language mask for official branch.")
    ignore_index = int(mods["IGNORE_INDEX"])
    lang_mask = (labels2.to(device=hidden_last.device) != ignore_index) & attn_mask_lang
    q_official, qmask_official = _build_language_queries_from_hidden(
        hidden_states=hidden_last,
        language_mask=lang_mask,
        select_mode=select_mode,
        max_tokens=max_tokens,
        topk_tokens=topk_tokens,
    )

    model_name = str(getattr(llava_model.config, "_name_or_path", "")) or str(llava_ckpt)
    info = {
        "official_model_name": model_name,
        "official_mm_use_im_start_end": float(mm_use_im_start_end),
        "official_prompt_len": float(len(prompt)),
        "official_hidden_shape": tuple(hidden_last.shape),
        "official_vis_shape": tuple(vis_tokens.shape),
        "official_lang_tokens_mean": float(lang_mask.float().sum(dim=1).mean().item()),
        "official_query_count_mean": float(qmask_official.float().sum(dim=1).mean().item()),
        "official_patch_info": patch_info,
    }
    return {
        "vision_tokens": vis_tokens.to(dtype=torch.float32),
        "language_queries": q_official.to(dtype=torch.float32),
        "language_query_mask": qmask_official,
        "info": info,
    }


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
    parser.add_argument(
        "--official-llava-ckpt",
        type=str,
        default="",
        help="Path to official LLaVA-3D checkpoint for full language+vision branch comparison.",
    )
    parser.add_argument(
        "--llava-pretrained-path",
        type=str,
        default="",
        help="Optional override path to original LLaVA-3D checkpoint/config (contains mm_vision_tower).",
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
                llava_pretrained_path_override=args.llava_pretrained_path,
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

    official_ckpt = str(args.official_llava_ckpt).strip()
    if official_ckpt:
        try:
            official_branch = _run_official_llava_branch(
                llava_repo=Path(args.llava_repo),
                llava_ckpt=Path(official_ckpt),
                image=image,
                instruction=args.instruction,
                cot_prompt=cot_prompt,
                device=device,
                select_mode=str(ma_cfg.get("semantic_query_select_mode", "summary_topk")),
                max_tokens=int(ma_cfg.get("semantic_query_max_tokens", 8)),
                topk_tokens=int(ma_cfg.get("semantic_topk_tokens", 8)),
            )
            vis_official = official_branch["vision_tokens"].to(device=lang_q.device, dtype=lang_q.dtype)
            q_official = official_branch["language_queries"].to(device=lang_q.device, dtype=lang_q.dtype)
            qmask_official = official_branch["language_query_mask"].to(device=lang_q.device)

            geo_for_official = geo
            if not (isinstance(geo_for_official, torch.Tensor) and geo_for_official.shape[-1] == vis_official.shape[-1]):
                geo_for_official = vis_official

            alpha_official, official_mask_stats = _build_soft_mask(
                vision_tokens=vis_official,
                geometric_tokens=geo_for_official,
                language_queries=q_official,
                language_query_mask=qmask_official,
                **mask_cfg,
            )
            result["official_llava_branch_info"] = official_branch["info"]
            result["official_llava_mask_stats"] = official_mask_stats
            result["current_vs_official_alpha_diff"] = _compare_alpha(alpha_current, alpha_official)
        except Exception as e:
            result["official_llava_branch_info"] = None
            result["official_llava_mask_stats"] = None
            result["current_vs_official_alpha_diff"] = None
            result["official_llava_error"] = str(e)
            result["official_llava_error_traceback"] = traceback.format_exc()

    result["llava_alt_available"] = llava_alt_ok
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
