#/usr/bin/env python
# coding=utf-8

import os
import sys
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


class MapAnythingWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.map_anything_model = MapAnything.from_pretrained(config.mapanything_model_name_or_path)
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
