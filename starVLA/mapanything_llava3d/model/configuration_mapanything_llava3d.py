# coding=utf-8
# Copyright 2024 MapAnythingLlava3D Team. All rights reserved.
#
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

import os
from transformers.configuration_utils import PretrainedConfig
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
LLAVA3D_IGNORE_INDEX = -100

class MapAnythingLlava3DConfig(PretrainedConfig):
    model_type = "mapanything_llava3d"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        mapanything_config=None,
        vision_model_name_or_path="google/siglip-so400m-patch14-224",
        language_model_name_or_path=None,
        mapanything_model_name_or_path=None,
        ignore_index=LLAVA3D_IGNORE_INDEX,
        image_token_index=-200, # Default to LLaVA-3D standard
        projection_dim=2048,
        hidden_size=2048,
        action_token_begin_idx=None,
        spatial_token_num=259,
        use_spatial_token=False,
        use_geometric_branch=True,
        image_seq_length=None,
        action_expert_config=None,
        action_dim=14,
        action_horizon=1,
        expert_hidden_size=None,
        expert_num_layers=None,
        expert_mlp_ratio=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_model_name_or_path = vision_model_name_or_path
        self.language_model_name_or_path = language_model_name_or_path
        self.mapanything_model_name_or_path = mapanything_model_name_or_path
        
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        
        # --- Vision Config (SigLIP) ---
        if isinstance(vision_config, dict):
            vc = dict(vision_config)
            vc["model_type"] = vc.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[vc["model_type"]](**vc)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        elif vision_config is None and vision_model_name_or_path is not None:
            # Try to load config from path, otherwise use default
            try:
                raw_cfg = AutoConfig.from_pretrained(vision_model_name_or_path, trust_remote_code=True)
                self.vision_config = getattr(raw_cfg, "vision_config", raw_cfg)
            except Exception:
                logger.warning(f"Could not load vision config from {vision_model_name_or_path}, using default SigLIP config.")
                self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                    intermediate_size=4096,
                    hidden_size=1152,
                    patch_size=14,
                    image_size=224,
                    num_hidden_layers=27,
                    num_attention_heads=16,
                    vocab_size=32000, # Placeholder
                    vision_use_head=False,
                )
        else:
             self.vision_config = CONFIG_MAPPING["siglip_vision_model"]()

        # --- Text Config (LLaVA-3D LLM) ---
        if isinstance(text_config, dict):
            mt = text_config.get("model_type")
            if mt:
                self.text_config = CONFIG_MAPPING[mt](**text_config)
            else:
                # Fallback to AutoConfig if model_type missing but looks like config
                self.text_config = AutoConfig.for_model(**text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif text_config is None and language_model_name_or_path is not None:
             self.text_config = AutoConfig.from_pretrained(language_model_name_or_path, trust_remote_code=True)
        else:
            # Raise error if no text config provided, as we can't guess the LLM
            # Unless we are just initializing a skeleton
            self.text_config = None

        # --- MapAnything Config ---
        # MapAnything usually doesn't have a standard HF config, so we store parameters dict or minimal config
        if isinstance(mapanything_config, dict):
            self.mapanything_config = mapanything_config
        else:
            self.mapanything_config = {}
        
        # Additional attributes
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.action_token_begin_idx = action_token_begin_idx
        self.spatial_token_num = spatial_token_num
        self.use_spatial_token = use_spatial_token
        self.use_geometric_branch = use_geometric_branch
        self.image_seq_length = image_seq_length
        self.action_expert_config = action_expert_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.expert_hidden_size = expert_hidden_size
        self.expert_num_layers = expert_num_layers
        self.expert_mlp_ratio = expert_mlp_ratio

        # Propagate hidden size to root config if text_config exists
        if self.text_config:
            self.hidden_size = self.text_config.hidden_size
            self.vocab_size = self.text_config.vocab_size
