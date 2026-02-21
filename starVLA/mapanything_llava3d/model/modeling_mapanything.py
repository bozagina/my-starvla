#/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_LOCAL_MAPANYTHING_ROOT = os.path.join(os.path.dirname(__file__), "map-anything")
if _LOCAL_MAPANYTHING_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_MAPANYTHING_ROOT)

# If mapanything was already imported from elsewhere, force reload from local path.
if "mapanything" in sys.modules:
    mod = sys.modules["mapanything"]
    mod_file = getattr(mod, "__file__", "") or ""
    if not mod_file.startswith(_LOCAL_MAPANYTHING_ROOT):
        del sys.modules["mapanything"]

from mapanything.models.mapanything.model import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerInput


_REQUIRED_ACTIVE_PREFIXES = (
    "encoder.",
    "info_sharing.",
)


def _get_default_device() -> Optional[torch.device]:
    if hasattr(torch, "get_default_device"):
        return torch.get_default_device()
    if hasattr(torch, "_C") and hasattr(torch._C, "_get_default_device"):
        return torch._C._get_default_device()
    return None


@contextmanager
def _suspend_meta_device():
    """Ensure checkpoint load does not run under meta-device init context."""
    prev_device = _get_default_device()
    reset_device = prev_device is not None and str(prev_device) == "meta"
    try:
        if reset_device and hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
        try:
            from accelerate.utils import init_empty_weights
        except Exception:
            init_empty_weights = None
        prev_flag = None
        if init_empty_weights is not None and hasattr(init_empty_weights, "_is_enabled"):
            prev_flag = init_empty_weights._is_enabled
            init_empty_weights._is_enabled = False
        try:
            yield
        finally:
            if prev_flag is not None:
                init_empty_weights._is_enabled = prev_flag
    finally:
        if reset_device and hasattr(torch, "set_default_device"):
            torch.set_default_device(str(prev_device))


@contextmanager
def _force_cpu_device():
    """Force module construction on CPU even if outer context sets a meta device."""
    try:
        with torch.device("cpu"):
            yield
    except Exception:
        # Fallback for environments where torch.device context manager is unavailable.
        yield


def _meta_tensor_summary(module: nn.Module, max_items: int = 20) -> Tuple[int, int, list, list]:
    meta_params = []
    meta_buffers = []
    for name, p in module.named_parameters():
        if getattr(p, "is_meta", False):
            meta_params.append(name)
    for name, b in module.named_buffers():
        if getattr(b, "is_meta", False):
            meta_buffers.append(name)
    return len(meta_params), len(meta_buffers), meta_params[:max_items], meta_buffers[:max_items]


def _all_meta_param_names(module: nn.Module) -> list:
    return [name for name, p in module.named_parameters() if getattr(p, "is_meta", False)]


def _required_meta_param_names(module: nn.Module, required_prefixes=_REQUIRED_ACTIVE_PREFIXES) -> list:
    meta_names = _all_meta_param_names(module)
    return [
        name for name in meta_names
        if any(name.startswith(prefix) for prefix in required_prefixes)
    ]


def _materialize_module_if_meta(module: nn.Module):
    meta_names = _all_meta_param_names(module)
    if not meta_names:
        return
    if hasattr(module, "to_empty"):
        module.to_empty(device=torch.device("cpu"))


def _load_local_state_dict(local_dir: str) -> dict:
    model_dir = Path(local_dir)
    safetensor_single = model_dir / "model.safetensors"
    if safetensor_single.exists():
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(f"Failed to import safetensors.torch.load_file: {exc}")
        return load_safetensors_file(str(safetensor_single), device="cpu")

    pt_single = model_dir / "pytorch_model.bin"
    if pt_single.exists():
        ckpt = torch.load(str(pt_single), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if isinstance(ckpt, dict):
            return ckpt
        raise RuntimeError(f"Unexpected checkpoint format in {pt_single}")

    sf_index = model_dir / "model.safetensors.index.json"
    if sf_index.exists():
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(f"Failed to import safetensors.torch.load_file: {exc}")
        index = json.loads(sf_index.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing safetensor shard: {shard_path}")
            state_dict.update(load_safetensors_file(str(shard_path), device="cpu"))
        return state_dict

    pt_index = model_dir / "pytorch_model.bin.index.json"
    if pt_index.exists():
        index = json.loads(pt_index.read_text())
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        state_dict = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing pytorch shard: {shard_path}")
            ckpt = torch.load(str(shard_path), map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                raise RuntimeError(f"Unexpected shard format in {shard_path}")
            state_dict.update(ckpt)
        return state_dict

    raise FileNotFoundError(
        f"No supported local checkpoint file found in {local_dir}. "
        "Expected one of: model.safetensors, pytorch_model.bin, model.safetensors.index.json, pytorch_model.bin.index.json"
    )


def _assign_local_checkpoint_into_model(model: nn.Module, local_dir: str) -> Tuple[int, int]:
    state_dict = _load_local_state_dict(local_dir)
    return _assign_state_dict_into_model(model, state_dict)


def _load_state_dict_from_checkpoint_file(checkpoint_path: str) -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format for {checkpoint_path}")


def _assign_state_dict_into_model(model: nn.Module, state_dict: dict) -> Tuple[int, int]:
    incompatible = None
    try:
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        # Older torch versions may not support assign=True.
        incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", []) if incompatible is not None else []
    unexpected = getattr(incompatible, "unexpected_keys", []) if incompatible is not None else []
    return len(missing), len(unexpected)


def _build_mapanything_from_local_config(model_id: str) -> nn.Module:
    """
    Build MapAnything module structure from local config without loading weights.

    This is used when the outer VLM from_pretrained call is responsible for a single,
    unified state_dict load. It avoids nested from_pretrained calls in __init__.
    """
    model_dir = Path(model_id)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"MapAnything config.json not found under {model_id}. "
            "Cannot build geometry module skeleton for unified outer loading."
        )

    cfg = json.loads(config_file.read_text())
    sig = inspect.signature(MapAnything.__init__)
    valid_keys = {k for k in sig.parameters.keys() if k != "self"}

    candidates = []
    if isinstance(cfg, dict):
        candidates.append(cfg)
        for key in ("model_config", "class_init_args", "init_args"):
            nested = cfg.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)

    init_kwargs = None
    for candidate in candidates:
        filtered = {k: v for k, v in candidate.items() if k in valid_keys}
        required_keys = {
            "name",
            "encoder_config",
            "info_sharing_config",
            "pred_head_config",
            "geometric_input_config",
        }
        if required_keys.issubset(filtered.keys()):
            init_kwargs = filtered
            break

    if init_kwargs is None:
        raise RuntimeError(
            "Failed to infer MapAnything constructor kwargs from config.json. "
            f"model_id={model_id}, top_level_keys={list(cfg.keys())[:30]}"
        )

    return MapAnything(**init_kwargs)


def _load_mapanything_with_meta_recovery(model_id: str) -> nn.Module:
    """
    Load MapAnything weights robustly under distributed/meta-init environments.
    Retry with explicit CPU device scope if first attempt still leaves meta tensors.
    """
    attempts = [
        ("default", False),
        ("force_cpu", True),
    ]
    last_issue = None
    for tag, force_cpu in attempts:
        try:
            with _suspend_meta_device():
                if force_cpu:
                    with _force_cpu_device():
                        model = MapAnything.from_pretrained(model_id)
                else:
                    model = MapAnything.from_pretrained(model_id)
            meta_p_cnt, meta_b_cnt, _, _ = _meta_tensor_summary(model)
            if meta_p_cnt == 0 and meta_b_cnt == 0:
                return model
            # If constructed under meta context, allocate real CPU storage first.
            _materialize_module_if_meta(model)
            assign_fallback_info = ""
            if os.path.isdir(model_id):
                try:
                    missing_cnt, unexpected_cnt = _assign_local_checkpoint_into_model(model, model_id)
                    meta_p_cnt2, meta_b_cnt2, _, _ = _meta_tensor_summary(model)
                    if meta_p_cnt2 == 0 and meta_b_cnt2 == 0:
                        return model
                    assign_fallback_info = (
                        f", assign_fallback_missing={missing_cnt}, "
                        f"assign_fallback_unexpected={unexpected_cnt}, "
                        f"assign_fallback_meta_params_count={meta_p_cnt2}, "
                        f"assign_fallback_meta_buffers_count={meta_b_cnt2}"
                    )
                except Exception as assign_exc:
                    assign_fallback_info = f", assign_fallback_error={assign_exc}"
            if (
                "assign_fallback_meta_params_count=" in assign_fallback_info
                and getattr(model, "pretrained_checkpoint_path", None)
            ):
                extra_ckpt = str(getattr(model, "pretrained_checkpoint_path"))
                if os.path.isfile(extra_ckpt):
                    try:
                        extra_sd = _load_state_dict_from_checkpoint_file(extra_ckpt)
                        missing_cnt2, unexpected_cnt2 = _assign_state_dict_into_model(model, extra_sd)
                        meta_p_cnt3, meta_b_cnt3, _, _ = _meta_tensor_summary(model)
                        if meta_p_cnt3 == 0 and meta_b_cnt3 == 0:
                            return model
                        assign_fallback_info += (
                            f", extra_ckpt_assign_missing={missing_cnt2}, "
                            f"extra_ckpt_assign_unexpected={unexpected_cnt2}, "
                            f"extra_ckpt_assign_meta_params_count={meta_p_cnt3}, "
                            f"extra_ckpt_assign_meta_buffers_count={meta_b_cnt3}"
                        )
                    except Exception as extra_assign_exc:
                        assign_fallback_info += f", extra_ckpt_assign_error={extra_assign_exc}"
            required_meta = _required_meta_param_names(model)
            if not required_meta:
                return model
            req_head = required_meta[:20]
            last_issue = (
                f"attempt={tag}, meta_params_count={meta_p_cnt}, meta_buffers_count={meta_b_cnt}"
                f", required_meta_count={len(required_meta)}, required_meta_head={req_head}"
                f"{assign_fallback_info}"
            )
        except Exception as exc:
            last_issue = f"attempt={tag}, error={exc}"
    raise RuntimeError(
        "MapAnything load failed to materialize parameters after retry. "
        f"model_id={model_id}, last_issue={last_issue}"
    )


class MapAnythingWrapper(nn.Module):
    def __init__(self, config, load_pretrained: bool = True):
        super().__init__()
        if load_pretrained:
            self.map_anything_model = _load_mapanything_with_meta_recovery(
                config.mapanything_model_name_or_path
            )
        else:
            self.map_anything_model = _build_mapanything_from_local_config(
                config.mapanything_model_name_or_path
            )

        meta_p_cnt, meta_b_cnt, meta_p_head, meta_b_head = _meta_tensor_summary(self.map_anything_model)
        if load_pretrained:
            required_meta = _required_meta_param_names(self.map_anything_model)
            if required_meta:
                raise RuntimeError(
                    "MapAnything loaded with REQUIRED meta tensors (encoder/info_sharing); "
                    "checkpoint load is invalid for active geometric path. "
                    f"required_meta_count={len(required_meta)}, required_meta_head={required_meta[:20]}, "
                    f"meta_params_count={meta_p_cnt}, meta_buffers_count={meta_b_cnt}, "
                    f"meta_params_head={meta_p_head}, meta_buffers_head={meta_b_head}"
                )
            if meta_p_cnt > 0 or meta_b_cnt > 0:
                print(
                    "[mapanything] non-required meta tensors remain (likely optional heads), "
                    f"meta_params_count={meta_p_cnt}, meta_buffers_count={meta_b_cnt}"
                )
        else:
            # In unified outer from_pretrained loading, meta init is expected at this stage.
            if meta_p_cnt > 0 or meta_b_cnt > 0:
                print(
                    "[mapanything] built geometry skeleton without local preload; "
                    f"meta_params_count={meta_p_cnt}, meta_buffers_count={meta_b_cnt}"
                )
        enc_dim = getattr(self.map_anything_model.encoder, "enc_embed_dim", None)
        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.hidden_size = int(enc_dim) if enc_dim is not None else 1024

    @staticmethod
    def _unwrap_feature(feature):
        # Some map-anything versions return (feat, meta) tuples per view.
        if isinstance(feature, tuple):
            if len(feature) == 0:
                return feature
            return feature[0]
        if isinstance(feature, dict):
            for k in ("features", "feature", "x"):
                if k in feature:
                    return feature[k]
        return feature

    @staticmethod
    def _ensure_4d_feature(feature, view_idx: int | None = None):
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"map-anything view[{view_idx}] is not a Tensor: {type(feature)}")
        if feature.dim() == 4:
            return feature
        if feature.dim() == 3:
            b, c, l = feature.shape
            if l == 1:
                return feature.view(b, c, 1, 1)
            side = int(l**0.5)
            if side * side == l:
                return feature.view(b, c, side, side)
        raise ValueError(
            f"map-anything view[{view_idx}] has unsupported shape {tuple(feature.shape)}; "
            "expected 4D (N,C,H,W) or 3D with L=1 or perfect square."
        )

    def forward(self, pixel_values, intrinsics):
        views = []
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            b, v = pixel_values.shape[:2]
            for view_idx in range(v):
                view = {"img": pixel_values[:, view_idx], "data_norm_type": ["dinov2"]}
                view["img"] = view["img"].float().contiguous()
                if intrinsics is not None:
                    if isinstance(intrinsics, torch.Tensor) and intrinsics.dim() == 4:
                        view_intrinsics = intrinsics[:, view_idx]
                    else:
                        view_intrinsics = intrinsics
                    view["intrinsics"] = view_intrinsics.float().contiguous()
                views.append(view)
        else:
            view = {"img": pixel_values, "data_norm_type": ["dinov2"]}
            view["img"] = view["img"].float().contiguous()
            if intrinsics is not None:
                view["intrinsics"] = intrinsics.float().contiguous()
            views = [view]

        all_encoder_features = self.map_anything_model._encode_n_views(views)
        encoder_registers = None
        if isinstance(all_encoder_features, tuple) and len(all_encoder_features) == 2:
            # Newer map-anything returns (features, registers)
            # TODO: registers are currently ignored because MultiViewTransformerInput does not accept them.
            # Consider wiring them into additional_input_tokens(_per_view) if info_sharing starts using them.
            all_encoder_features, encoder_registers = all_encoder_features
        if not hasattr(self, "_debug_logged_encode_views"):
            try:
                print(f"[mapanything] _encode_n_views type: {type(all_encoder_features)}")
                if isinstance(all_encoder_features, (list, tuple)):
                    for i, f in enumerate(all_encoder_features[:3]):
                        print(f"[mapanything] view[{i}] type: {type(f)}")
                        if isinstance(f, torch.Tensor):
                            print(f"[mapanything] view[{i}] shape: {tuple(f.shape)} dtype: {f.dtype}")
                        elif isinstance(f, tuple) and len(f) > 0 and isinstance(f[0], torch.Tensor):
                            print(f"[mapanything] view[{i}] tuple[0] shape: {tuple(f[0].shape)} dtype: {f[0].dtype}")
                        elif isinstance(f, dict):
                            for k in ("features", "feature", "x"):
                                if k in f and isinstance(f[k], torch.Tensor):
                                    print(f"[mapanything] view[{i}] dict[{k}] shape: {tuple(f[k].shape)} dtype: {f[k].dtype}")
                                    break
                if encoder_registers is not None and isinstance(encoder_registers, (list, tuple)):
                    for i, f in enumerate(encoder_registers[:3]):
                        if isinstance(f, torch.Tensor):
                            print(f"[mapanything] register[{i}] shape: {tuple(f.shape)} dtype: {f.dtype}")
                self._debug_logged_encode_views = 1
            except Exception:
                self._debug_logged_encode_views = 1
        if isinstance(all_encoder_features, (list, tuple)):
            converted = []
            for i, f in enumerate(all_encoder_features):
                f = self._unwrap_feature(f)
                f = self._ensure_4d_feature(f, view_idx=i)
                converted.append(f)
            all_encoder_features = converted

        info_sharing_input = MultiViewTransformerInput(features=all_encoder_features)

        final_features, _ = self.map_anything_model.info_sharing(info_sharing_input)

        if len(final_features.features) == 1:
            geometric_features = final_features.features[0]
            if geometric_features.dim() == 4:
                b, c, h, w = geometric_features.shape
                geometric_features = geometric_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            seq_features = []
            for f in final_features.features:
                if f.dim() == 4:
                    b, c, h, w = f.shape
                    f = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
                elif f.dim() == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported feature shape: {tuple(f.shape)}")
                seq_features.append(f)
            geometric_features = torch.cat(seq_features, dim=1)

        class _Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return _Out(geometric_features)
