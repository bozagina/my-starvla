#/usr/bin/env python
# coding=utf-8

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "map-anything"))
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
