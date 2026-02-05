# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from LLaVA_3D.llava.constants import IGNORE_INDEX as LLAVA3D_IGNORE_INDEX
import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig
from LLaVA_3D.llava.constants import (
    IGNORE_INDEX as LLAVA3D_IGNORE_INDEX,
)

logger = logging.get_logger(__name__)

class SpatialVLAConfig(PretrainedConfig):
    model_type = "spatialvla"
    # 仅声明始终存在的子配置，避免在自动设置注意力实现时访问 None
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        vision_model_name_or_path="google/siglip-so400m-patch14-224",
        language_model_name_or_path=None,
        map_anything_model_name_or_path=None,
        vision_weight_source="siglip_official",
        spatialvla_vision_pretrained_path=None,
        ignore_index=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        vision_zoe_config=None,
        action_token_begin_idx=None,
        spatial_token_num=259,
        use_spatial_token=False,
        ego3d_patch_reso=4,
        n_freqs=8,
        use_vision_zoe=True,
        image_seq_length=None,
        **kwargs,
    ):
        if language_model_name_or_path is None and text_config is None:
            raise ValueError("Provide either `language_model_name_or_path` or a complete `text_config` for integrated checkpoints.")
        if map_anything_model_name_or_path is None:
            raise ValueError("Provide `map_anything_model_name_or_path` for MapAnything model.")
        
        self.vision_model_name_or_path = vision_model_name_or_path
        self.language_model_name_or_path = language_model_name_or_path
        self.mapanything_model_name_or_path = map_anything_model_name_or_path
        self.vision_weight_source = vision_weight_source
        self.spatialvla_vision_pretrained_path = spatialvla_vision_pretrained_path
        # 根据 LLaVA-3D 协议设置 ignore_index（可被覆盖）
        self._ignore_index = LLAVA3D_IGNORE_INDEX if ignore_index is None else ignore_index

        self.image_token_index = image_token_index
        self.is_encoder_decoder = False
        # 构建纯视觉 vision_config（遵循原版 SpatialVLA 语义）
        if isinstance(vision_config, dict):
            vc = dict(vision_config)
            vc["model_type"] = vc.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[vc["model_type"]](**vc)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        elif vision_config is None and vision_model_name_or_path is not None:
            raw_cfg = AutoConfig.from_pretrained(vision_model_name_or_path, trust_remote_code=True)
            self.vision_config = getattr(raw_cfg, "vision_config", raw_cfg)
        else:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )
        # 语言侧使用 LLaVA-3D 的配置（支持集成权重场景：直接用提供的 text_config）
        if isinstance(text_config, dict):
            mt = text_config.get("model_type")
            if mt is None:
                raise ValueError("`text_config` dict must include `model_type` for reconstruction.")
            self.text_config = CONFIG_MAPPING[mt](**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            self.text_config = AutoConfig.from_pretrained(language_model_name_or_path, trust_remote_code=True)
        # 计算图像补丁数，统一从纯视觉 config 读取；若 image_size 为 (H,W)，取 H
        image_size = getattr(self.vision_config, "image_size", None)
        patch_size = getattr(self.vision_config, "patch_size", None)
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        inferred_tokens = None
        if image_size is not None and patch_size is not None:
            inferred_tokens = (image_size // patch_size) ** 2
        # 优先使用外部传入的 image_seq_length，其次使用视觉配置推断
        if image_seq_length is not None:
            self.text_config.num_image_tokens = int(image_seq_length)
            self.image_seq_length = int(image_seq_length)
        elif inferred_tokens is not None:
            self.text_config.num_image_tokens = int(inferred_tokens)
            self.image_seq_length = int(inferred_tokens)
        else:
            # 延迟推断：首次前向通过视觉塔得到序列长度后写回
            self.text_config.num_image_tokens = getattr(self.text_config, "num_image_tokens", None)
            self.image_seq_length = getattr(self.text_config, "num_image_tokens", None)
        # projector 输出维度对齐语言侧 hidden_size
        self.vision_config.projection_dim = self.text_config.hidden_size

        # vision zoe config
        self.vision_zoe_config = vision_zoe_config
        if use_vision_zoe:
            pass
        else:
            pass
        
        # additional attributes
        self.action_token_begin_idx = action_token_begin_idx
        self.spatial_token_num = spatial_token_num
        self.use_spatial_token = use_spatial_token
        self.ego3d_patch_reso = ego3d_patch_reso
        self.n_freqs = n_freqs
        self.use_vision_zoe = use_vision_zoe
        self.hidden_size = self.text_config.hidden_size
        self._vocab_size = self.text_config.vocab_size
        self.projection_dim = self.vision_config.projection_dim
        # 统一存储：若上面尚未设置 image_seq_length，则与 text_config.num_image_tokens 对齐
        if getattr(self, "image_seq_length", None) is None and getattr(self.text_config, "num_image_tokens", None) is not None:
            self.image_seq_length = int(self.text_config.num_image_tokens)
        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output
