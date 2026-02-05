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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.utils import ModelOutput, logging
from transformers.generation import GenerationMixin
from .configuration_mapanything_llava3d import MapAnythingLlava3DConfig
from .modeling_mapanything import MapAnythingWrapper
from .modeling_flow_expert_dev import FlowMatchingActionExpert
from .modeling_llava3d_v2 import LLaVA3DForCausalLMV2
from .modeling_llava3d_v2_dev import LLaVA3DWithActionExpertModel
import torchvision.transforms.functional as TF

logger = logging.get_logger(__name__)

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

@dataclass
class MapAnythingLlava3DOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], object]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

class MapAnythingLlava3DPreTrainedModel(PreTrainedModel):
    config_class = MapAnythingLlava3DConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MapAnythingProjector"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class MapAnythingLlava3DForConditionalGeneration(MapAnythingLlava3DPreTrainedModel, GenerationMixin):
    def __init__(
        self, 
        config: MapAnythingLlava3DConfig, 
        vision_tower=None, 
        language_model=None, 
        mapanything_model=None,
        projector_model=None
    ):
        super().__init__(config)
        
        # 1. Vision Tower (SigLIP)
        # Allows separate loading: if instance provided, use it; else load from config
        if vision_tower is not None:
            self.vision_tower = vision_tower
        else:
            self.vision_tower = AutoModel.from_config(config.vision_config)
            # Note: We assume weights are loaded separately if using this path, 
            # or one calls .from_pretrained() on the wrapper.

        # 2. Language Model (LLaVA-3D / LLM)
        if language_model is not None:
            self.language_model = language_model
        elif config.language_model_name_or_path is not None:
            self.language_model = LLaVA3DForCausalLMV2.from_pretrained(
                config.language_model_name_or_path, 
                config=config.text_config
            )
        else:
            self.language_model = LLaVA3DForCausalLMV2(config.text_config)
        
        base_text_cfg = self.language_model.config
        base_hidden_size = base_text_cfg.hidden_size
        base_intermediate_size = getattr(base_text_cfg, "intermediate_size", base_hidden_size * 4)
        base_mlp_ratio = base_intermediate_size / base_hidden_size
        expert_hidden_size = getattr(config, "expert_hidden_size", None)
        if expert_hidden_size is None:
            expert_hidden_size = base_hidden_size
        expert_num_layers = getattr(config, "expert_num_layers", None)
        if expert_num_layers is None:
            expert_num_layers = base_text_cfg.num_hidden_layers
        expert_mlp_ratio = getattr(config, "expert_mlp_ratio", None)
        if expert_mlp_ratio is None:
            expert_mlp_ratio = base_mlp_ratio
        expert_cfg_dict = base_text_cfg.to_dict()
        expert_cfg_dict["hidden_size"] = expert_hidden_size
        expert_cfg_dict["intermediate_size"] = int(expert_hidden_size * expert_mlp_ratio)
        expert_cfg_dict["num_hidden_layers"] = expert_num_layers
        expert_cfg_dict["num_attention_heads"] = base_text_cfg.num_attention_heads
        expert_cfg_dict["num_key_value_heads"] = getattr(base_text_cfg, "num_key_value_heads", base_text_cfg.num_attention_heads)
        expert_config = base_text_cfg.__class__(**expert_cfg_dict)
        
        self.language_model_with_expert = LLaVA3DWithActionExpertModel(self.language_model, expert_config=expert_config)

        # 3. Geometric Model (MapAnything)
        if mapanything_model is not None:
            self.geometric_model = mapanything_model
        else:
            # Initialize wrapper from config
            self.geometric_model = MapAnythingWrapper(config)

        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_config.hidden_size
        self.vision_projector = projector_model or nn.Linear(self.vision_hidden_size, self.hidden_size)
    
        geom_dim = self._infer_geom_dim()
        self.geometric_projector = nn.Linear(geom_dim, self.hidden_size)
        self.fusion_projector = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 5. Action Expert (Optional) - Deep Fusion Version
        if getattr(config, "use_action_expert", False):
            self.action_expert = FlowMatchingActionExpert(
                llava_with_expert_model=self.language_model_with_expert,
                action_dim=getattr(config, "action_dim", 7),
                action_horizon=getattr(config, "action_horizon", 10),
                state_dim=getattr(config, "state_dim", None),
                use_state=getattr(config, "use_state", False),
                use_time_weight=getattr(config, "use_time_weight", True),
            )
        else:
            self.action_expert = None
        self.prefix_lang_dropout_prob = getattr(config, "prefix_lang_dropout_prob", 0.0)
        self.prefix_image_dropout_prob = getattr(config, "prefix_image_dropout_prob", 0.0)
        # 6. Spatial Tokens (Optional)
        if config.use_spatial_token:
            self.spatial_embed_tokens = nn.Embedding(config.spatial_token_num, self.hidden_size)
        else:
            self.spatial_embed_tokens = None
            
        # Post-init setup
        self.pad_token_id = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.text_config.vocab_size
        self.geom_feature_hook_enabled = False
        self.geom_feature_hook_max_steps = 100
        self.geom_feature_stats = []
        self._llava_vision_available = True

    def _build_joint_attention_inputs(
        self,
        prefix_embs: torch.Tensor,
        suffix_len: int,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ):
        """
        根据 prefix/suffix 长度构造联合 attention_mask 与 position_ids。
        - attention_mask: 加性 mask，shape [B, 1, 1, Lp+Ls]
        - position_ids:  连续位置编码，shape [B, Lp+Ls]
        """
        batch_size, prefix_len = prefix_embs.shape[:2]
        device = prefix_embs.device

        if attention_mask is None:
            prefix_pad = torch.ones((batch_size, prefix_len), device=device, dtype=torch.bool)
        else:
            prefix_pad = attention_mask.to(device=device)
            if prefix_pad.dtype != torch.bool:
                prefix_pad = prefix_pad != 0

        suffix_pad = torch.ones((batch_size, suffix_len), device=device, dtype=prefix_pad.dtype)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)

        # 构造类似 openpi.make_att_2d_masks 的 autoregressive mask：
        # - prefix tokens: 全 0（前缀内部完全双向、自身与视觉互相可见）
        # - suffix tokens: [1, 0, 0, ...]（前缀不会看到后缀，但后缀能看到前缀+后缀）
        att_list = [0] * prefix_len + ([1] + [0] * (suffix_len - 1) if suffix_len > 0 else [])
        att_masks = torch.tensor(att_list, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
        joint_2d = att_2d & pad_2d

        joint_mask = torch.zeros(
            (batch_size, 1, joint_2d.shape[1], joint_2d.shape[2]),
            device=device,
            dtype=prefix_embs.dtype,
        )
        mask_value = torch.finfo(prefix_embs.dtype).min
        joint_mask = joint_mask.masked_fill(~joint_2d[:, None, :, :], mask_value)

        # 与 openpi 一致：对非 padding 位置做 cumsum 生成连续 position_ids
        joint_position_ids = torch.cumsum(pad_masks.to(torch.long), dim=1) - 1

        return joint_mask, joint_position_ids, prefix_pad

    def _infer_geom_dim(self) -> int:
        # Prefer the multi-view transformer output dim defined in MapAnything config.
        mam = getattr(self.geometric_model, "map_anything_model", None)
        if mam is not None:
            info_sharing = getattr(mam, "info_sharing", None)
            if info_sharing is not None and hasattr(info_sharing, "dim"):
                return int(info_sharing.dim)
            encoder = getattr(mam, "encoder", None)
            if encoder is not None and hasattr(encoder, "enc_embed_dim"):
                return int(encoder.enc_embed_dim)
        geom_cfg = getattr(self.geometric_model, "config", None)
        if geom_cfg is not None and hasattr(geom_cfg, "hidden_size"):
            return int(geom_cfg.hidden_size)
        return self.vision_hidden_size

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
        
    def enable_geom_feature_hook(self, max_steps: int = 100):
        self.geom_feature_hook_enabled = True
        self.geom_feature_hook_max_steps = max_steps
        self.geom_feature_stats = []

    def disable_geom_feature_hook(self):
        self.geom_feature_hook_enabled = False

    def _record_geom_stats(self, tag: str, tensor: torch.Tensor):
        if not self.geom_feature_hook_enabled:
            return
        with torch.no_grad():
            t = tensor.detach()
            mean = t.mean().item()
            std = t.std().item()
            min_val = t.min().item()
            max_val = t.max().item()
            stats = {
                "tag": str(tag),
                "shape": tuple(t.shape),
                "mean": float(mean),
                "std": float(std),
                "min": float(min_val),
                "max": float(max_val),
            }
            self.geom_feature_stats.append(stats)
            if len(self.geom_feature_stats) > self.geom_feature_hook_max_steps:
                self.geom_feature_stats.pop(0)
        
    def enable_geom_grad_hook(self, max_steps: int = 100):
        self.geom_grad_hook_enabled = True
        self.geom_grad_hook_max_steps = max_steps
        self.geom_grad_stats = []
        handles = []
        modules = []
        if hasattr(self, "geometric_model") and self.geometric_model is not None:
            modules.append(("geometric_model", self.geometric_model))
        if hasattr(self, "geometric_projector") and self.geometric_projector is not None:
            modules.append(("geometric_projector", self.geometric_projector))
        if hasattr(self, "fusion_projector") and self.fusion_projector is not None:
            modules.append(("fusion_projector", self.fusion_projector))

        def make_hook(name):
            def _hook(module, grad_input, grad_output):
                if not getattr(self, "geom_grad_hook_enabled", False):
                    return
                with torch.no_grad():
                    total_sq = 0.0
                    count = 0
                    for p in module.parameters():
                        if p.grad is not None:
                            g = p.grad.detach()
                            total_sq += g.float().pow(2).sum().item()
                            count += g.numel()
                    if count == 0:
                        return
                    grad_l2 = total_sq ** 0.5
                    grad_rms = (total_sq / count) ** 0.5
                    stats = {
                        "module": str(name),
                        "grad_l2": float(grad_l2),
                        "grad_rms": float(grad_rms),
                    }
                    self.geom_grad_stats.append(stats)
                    max_steps = getattr(self, "geom_grad_hook_max_steps", 100)
                    if len(self.geom_grad_stats) > max_steps:
                        self.geom_grad_stats.pop(0)
            return _hook

        for name, module in modules:
            h = module.register_full_backward_hook(make_hook(name))
            handles.append(h)
        self._geom_grad_handles = handles

    def disable_geom_grad_hook(self):
        self.geom_grad_hook_enabled = False
        if hasattr(self, "_geom_grad_handles"):
            for h in self._geom_grad_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._geom_grad_handles = []
    
    def enable_vlm_vision_grad_hook(self, max_steps: int = 100):
        self.vlm_vision_grad_hook_enabled = True
        self.vlm_vision_grad_hook_max_steps = max_steps
        self.vlm_vision_grad_stats = []
        handles = []
        modules = []
        if hasattr(self, "language_model") and self.language_model is not None:
            modules.append(("language_model", self.language_model))
        if hasattr(self, "vision_tower") and self.vision_tower is not None:
            modules.append(("vision_tower", self.vision_tower))
        if hasattr(self, "vision_projector") and self.vision_projector is not None:
            modules.append(("vision_projector", self.vision_projector))

        def make_hook(name):
            def _hook(module, grad_input, grad_output):
                if not getattr(self, "vlm_vision_grad_hook_enabled", False):
                    return
                with torch.no_grad():
                    total_sq = 0.0
                    count = 0
                    for p in module.parameters():
                        if p.grad is not None:
                            g = p.grad.detach()
                            total_sq += g.float().pow(2).sum().item()
                            count += g.numel()
                    if count == 0:
                        return
                    grad_l2 = total_sq ** 0.5
                    grad_rms = (total_sq / count) ** 0.5
                    stats = {
                        "module": str(name),
                        "grad_l2": float(grad_l2),
                        "grad_rms": float(grad_rms),
                    }
                    self.vlm_vision_grad_stats.append(stats)
                    max_steps = getattr(self, "vlm_vision_grad_hook_max_steps", 100)
                    if len(self.vlm_vision_grad_stats) > max_steps:
                        self.vlm_vision_grad_stats.pop(0)
            return _hook

        for name, module in modules:
            h = module.register_full_backward_hook(make_hook(name))
            handles.append(h)
        self._vlm_vision_grad_handles = handles

    def disable_vlm_vision_grad_hook(self):
        self.vlm_vision_grad_hook_enabled = False
        if hasattr(self, "_vlm_vision_grad_handles"):
            for h in self._vlm_vision_grad_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._vlm_vision_grad_handles = []

    def record_core_weight_stats(self, step: Optional[int] = None, max_steps: int = 100):
        if not hasattr(self, "core_weight_stats"):
            self.core_weight_stats = []
            self.core_weight_stats_max_steps = max_steps
        modules = []
        if hasattr(self, "language_model") and self.language_model is not None:
            modules.append(("language_model", self.language_model))
        if hasattr(self, "vision_tower") and self.vision_tower is not None:
            modules.append(("vision_tower", self.vision_tower))
        if hasattr(self, "geometric_model") and self.geometric_model is not None:
            modules.append(("geometric_model", self.geometric_model))
        with torch.no_grad():
            for name, module in modules:
                total_sq = 0.0
                count = 0
                for p in module.parameters():
                    w = p.detach()
                    total_sq += w.float().pow(2).sum().item()
                    count += w.numel()
                if count == 0:
                    continue
                w_l2 = total_sq ** 0.5
                w_rms = (total_sq / count) ** 0.5
                stats = {
                    "module": str(name),
                    "step": int(step) if step is not None else None,
                    "w_l2": float(w_l2),
                    "w_rms": float(w_rms),
                }
                self.core_weight_stats.append(stats)
                max_len = getattr(self, "core_weight_stats_max_steps", max_steps)
                if len(self.core_weight_stats) > max_len:
                    self.core_weight_stats.pop(0)
        
    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        base_model = getattr(self.language_model, "model", None)
        vision_feats = None
        use_llava_encode = (
            self._llava_vision_available
            and base_model is not None
            and hasattr(base_model, "encode_images")
        )
        if use_llava_encode and hasattr(base_model, "get_vision_tower"):
            vision_tower = base_model.get_vision_tower()
            vt_path = getattr(vision_tower, "vision_tower", None)
            if vt_path is None or (isinstance(vt_path, str) and not vt_path):
                logger.info("LLaVA3D vision tower path is empty or None, using SigLIP vision tower instead.")
                use_llava_encode = False
                self._llava_vision_available = False
            elif hasattr(vision_tower, "is_loaded") and hasattr(vision_tower, "load_model") and not vision_tower.is_loaded:
                try:
                    vision_tower.load_model()
                except Exception as e:
                    logger.warning(f"Failed to load LLaVA3D vision tower ({e}), fallback to SigLIP path.")
                    use_llava_encode = False
                    self._llava_vision_available = False
        if use_llava_encode:
            try:
                vision_feats = base_model.encode_images(pixel_values)
            except Exception as e:
                logger.warning(f"LLaVA3D encode_images failed ({e}), fallback to SigLIP path.")
                vision_feats = None
                self._llava_vision_available = False
        if vision_feats is None:
            siglip_pixel_values = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
            siglip_pixel_values = siglip_pixel_values.float().contiguous()
            vision_outputs = self.vision_tower(siglip_pixel_values)
            vision_feats = vision_outputs.last_hidden_state
            if getattr(self, "geom_feature_hook_enabled", False):
                self._record_geom_stats("vision_raw", vision_feats)

        if vision_feats is not None and vision_feats.shape[-1] != self.hidden_size:
            vision_feats = self.vision_projector(vision_feats)
            if getattr(self, "geom_feature_hook_enabled", False):
                self._record_geom_stats("vision_proj", vision_feats)

        if self.training:
            p_img = getattr(self, "prefix_image_dropout_prob", 0.0)
            if p_img > 0.0 and vision_feats is not None:
                keep = torch.ones_like(vision_feats[..., 0], dtype=vision_feats.dtype, device=vision_feats.device)
                rand = torch.rand_like(keep)
                drop_mask = rand < p_img
                keep = keep.masked_fill(drop_mask, 0.0)
                scale_mask = ~drop_mask
                if scale_mask.any():
                    keep = keep.masked_fill(scale_mask, 1.0 / (1.0 - p_img))
                vision_feats = vision_feats * keep.unsqueeze(-1)

        geometric_out = self.geometric_model(pixel_values=pixel_values, intrinsics=intrinsic)
        geometric_features = geometric_out.last_hidden_state
        if geometric_features.dim() == 4:
            b, c, h, w = geometric_features.shape
            geom_seq = geometric_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            geom_seq = geometric_features

        geom_proj = self.geometric_projector(geom_seq).to(vision_feats.dtype)
        geom_global = geom_proj.mean(dim=1, keepdim=True)
        geom_broadcast = geom_global.expand(vision_feats.shape[0], vision_feats.shape[1], geom_global.shape[-1])
        fused_features_pre = torch.cat([vision_feats, geom_broadcast], dim=-1)
        final_features = self.fusion_projector(fused_features_pre)

        self._record_geom_stats("geom_proj", geom_proj)
        self._record_geom_stats("fused_pre", fused_features_pre)
        self._record_geom_stats("fused", final_features)

        return final_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        intrinsic: Optional[torch.Tensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_token_index: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, MapAnythingLlava3DOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- 1. Embedding Layer ---
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            embed = self.get_input_embeddings()
            vocab_size = embed.weight.shape[0]
            input_ids = input_ids.clamp(min=0, max=vocab_size - 1)
            inputs_embeds = embed(input_ids)

        # --- 2. Handle Image Features ---
        if pixel_values is not None:
            # We need to inject image features into inputs_embeds
            # Find image tokens
            target_image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
            
            image_features = self.get_image_features(pixel_values, intrinsic)
            
            # Mask replacement logic
            # Assuming one image per sequence for simplicity as per previous code, or strictly following masks
            # Create a mask for image tokens
            image_mask = (input_ids == target_image_token_index)
            
            if image_mask.any():
                # Verify shapes
                # Flatten features to match masked positions
                # This simple replacement assumes strict alignment between token count and feature count
                # or requires batch processing.
                
                # Check if batch size matches
                if image_features.shape[0] == inputs_embeds.shape[0]:
                    # Standard batching
                    # If we have multiple images per sample, this logic needs to be robust.
                    # Current assumption: 1 image, matching token count.
                    
                    # Safety check: do the counts match?
                    # num_img_tokens_in_input = image_mask.sum(dim=1)
                    # num_img_tokens_in_feat = image_features.shape[1]
                    
                    # We use masked_scatter_ or manual index replacement
                    # Flatten for replacement
                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
                else:
                    logger.warning("Batch size mismatch between input_ids and pixel_values. Skipping image injection.")

        # --- 3. Handle Spatial Tokens ---
        if self.config.use_spatial_token and self.spatial_embed_tokens is not None:
            # Assuming spatial tokens are in a specific range
            begin_idx = self.config.action_token_begin_idx
            if begin_idx is not None:
                spatial_mask = (input_ids >= begin_idx) & (input_ids < begin_idx + self.config.spatial_token_num)
                if spatial_mask.any():
                    spatial_ids = input_ids[spatial_mask] - begin_idx
                    inputs_embeds[spatial_mask] = self.spatial_embed_tokens(spatial_ids).to(inputs_embeds.dtype)

        if self.training:
            p_lang = getattr(self, "prefix_lang_dropout_prob", 0.0)
            if p_lang > 0.0 and input_ids is not None:
                if attention_mask is None:
                    active = torch.ones_like(input_ids, dtype=torch.bool, device=inputs_embeds.device)
                else:
                    active = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
                target_image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
                lang_token_mask = (input_ids != target_image_token_index) & active
                if lang_token_mask.any():
                    keep = torch.ones_like(input_ids, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                    rand = torch.rand_like(inputs_embeds[..., 0])
                    drop_lang = (rand < p_lang) & lang_token_mask
                    keep = keep.masked_fill(drop_lang, 0.0)
                    scale_lang_mask = lang_token_mask & (~drop_lang)
                    if scale_lang_mask.any():
                        keep = keep.masked_fill(scale_lang_mask, 1.0 / (1.0 - p_lang))
                    inputs_embeds = inputs_embeds * keep.unsqueeze(-1)

        # --- 4. Action Expert Training (Deep Fusion Flow Matching) ---
        # If actions are provided, use Deep Fusion path with action expert
        if actions is not None and self.action_expert is not None:
            # Use prefix embeddings (image + text) for Deep Fusion
            # Get state from kwargs if provided
            state = kwargs.get("state", None)

            include_state_token = (
                self.action_expert.use_state
                and self.action_expert.state_proj is not None
                and state is not None
            )
            suffix_len = self.action_expert.action_horizon + 1 + (1 if include_state_token else 0)
            joint_attention_mask, joint_position_ids, _ = self._build_joint_attention_inputs(
                prefix_embs=inputs_embeds,
                suffix_len=suffix_len,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
            # Compute Flow Matching loss using Deep Fusion
            # The expert will construct suffix_embs from actions and call WithExpert
            action_loss = self.action_expert.compute_loss(
                prefix_embs=inputs_embeds,
                actions=actions,
                state=state,
                attention_mask=joint_attention_mask,
                position_ids=joint_position_ids,
            )
            
            # Return action loss as main loss (action-only training)
            loss = action_loss
            
            # For action training, we don't need language model forward
            # Return minimal output
            return MapAnythingLlava3DOutput(
                loss=loss,
                logits=None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                image_hidden_states=image_features if pixel_values is not None else None
            )
        
        # --- 5. LLM Forward (Language Generation Path) ---
        # Only run this if no actions (pure language generation)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = outputs.loss

        if not return_dict:
            output = (outputs.logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MapAnythingLlava3DOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        inputs_embeds=None, 
        pixel_values=None, 
        intrinsic=None,
        attention_mask=None,
        **kwargs
    ):
        # Delegate to LLM, but ensure we keep pixel_values/intrinsic for the forward pass 
        # when we are at the first step (past_key_values is None or empty)
        
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Attach special inputs if they are needed for the first forward pass
        if past_key_values is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["intrinsic"] = intrinsic
            
        return model_inputs

    @torch.no_grad()
    def predict_action(
        self,
        model_inputs,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """
        Predict actions using Flow Matching with Deep Fusion.
        
        Args:
            model_inputs: Dictionary containing:
                - input_ids: [B, L] text token ids
                - pixel_values: [B, 3, H, W] images
                - intrinsic: [B, 3, 3] camera intrinsics
                - attention_mask: [B, L] attention mask
                - state: [B, state_dim] robot state (optional)
            num_steps: Number of Euler ODE steps for sampling (default: 20)
            
        Returns:
            predicted_actions: [B, action_horizon, action_dim]
        """
        def _move_field(k, v):
            if hasattr(v, "to"):
                if torch.is_floating_point(v):
                    if k in ("pixel_values", "intrinsic"):
                        v = v.to(dtype=torch.float32)
                    else:
                        v = v.to(dtype=torch.bfloat16)
                v = v.to(self.device)
            return v
        
        # Flow Matching inference with Deep Fusion
        if getattr(self, "action_expert", None) is not None:
            if isinstance(model_inputs, dict):
                model_inputs = {k: _move_field(k, v) for k, v in model_inputs.items()}
            else:
                model_inputs = _move_field("_", model_inputs)

            # --- 1. Construct prefix embeddings (image + text) ---
            input_ids = model_inputs.get("input_ids")
            pixel_values = model_inputs.get("pixel_values")
            intrinsic = model_inputs.get("intrinsic")
            attention_mask = model_inputs.get("attention_mask")
            image_token_index = model_inputs.get("image_token_index", self.config.image_token_index)
            state = model_inputs.get("state", None)
            
            # Get text embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            
            # Get image features and inject into text embeddings
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, intrinsic)
                image_mask = (input_ids == image_token_index)
                
                if image_mask.any():
                    inputs_embeds = inputs_embeds.clone()
                    inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1]).to(inputs_embeds.dtype)
            
            # Handle spatial tokens if needed
            if self.config.use_spatial_token and self.spatial_embed_tokens is not None:
                begin_idx = self.config.action_token_begin_idx
                if begin_idx is not None:
                    spatial_mask = (input_ids >= begin_idx) & (input_ids < begin_idx + self.config.spatial_token_num)
                    if spatial_mask.any():
                        spatial_ids = input_ids[spatial_mask] - begin_idx
                        inputs_embeds[spatial_mask] = self.spatial_embed_tokens(spatial_ids).to(inputs_embeds.dtype)
            
            # Now inputs_embeds is the prefix embeddings (image + text)
            prefix_embs = inputs_embeds
            
            include_state_token = (
                self.action_expert.use_state
                and self.action_expert.state_proj is not None
                and state is not None
            )
            suffix_len = self.action_expert.action_horizon + 1 + (1 if include_state_token else 0)
            joint_attention_mask, joint_position_ids, prefix_pad = self._build_joint_attention_inputs(
                prefix_embs=prefix_embs,
                suffix_len=suffix_len,
                attention_mask=attention_mask,
                position_ids=None,
            )

            prefix_position_ids = torch.cumsum(prefix_pad, dim=1).to(dtype=torch.long) - 1
            _, prefix_past_key_values = self.language_model_with_expert(
                attention_mask=prefix_pad,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            # --- 2. Sample actions using Euler ODE with Deep Fusion (suffix复用prefix cache) ---
            actions = self.action_expert.sample_actions(
                prefix_embs=prefix_embs,
                state=state,
                num_steps=num_steps,
                attention_mask=joint_attention_mask,
                position_ids=joint_position_ids,
                prefix_past_key_values=prefix_past_key_values,
            )
            
            return actions

        # Legacy 自回归动作推理：使用generate生成动作token，再由processor.decode_actions解析
        if isinstance(model_inputs, dict):
            model_inputs = {k: _move_field(k, v) for k, v in model_inputs.items()}
        else:
            model_inputs = _move_field("_", model_inputs)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False, use_cache=False)
        return generation_outputs[:, input_len:]
