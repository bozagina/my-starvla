# coding=utf-8
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from transformers.utils import ModelOutput, logging
from transformers.generation import GenerationMixin

from .configuration_mapanything_llava3d import MapAnythingLlava3DConfig
from .modeling_mapanything import MapAnythingWrapper
from .modeling_llava3d_v2 import LLaVA3DForCausalLMV2

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
        projector_model=None,
        skip_language_model_preload: bool = False,
    ):
        super().__init__(config)

        if vision_tower is not None:
            self.vision_tower = vision_tower
        else:
            self.vision_tower = AutoModel.from_config(config.vision_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            text_cfg = config.text_config
            if skip_language_model_preload and text_cfg is not None:
                text_cfg = copy.deepcopy(text_cfg)
                if hasattr(text_cfg, "llava3d_pretrained_path"):
                    setattr(text_cfg, "llava3d_pretrained_path", None)
                logger.info("Skip inner LLaVA preload in model __init__; rely on outer from_pretrained state_dict loading.")
            self.language_model = LLaVA3DForCausalLMV2(text_cfg)

        if mapanything_model is not None:
            self.geometric_model = mapanything_model
        else:
            self.geometric_model = MapAnythingWrapper(config)

        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_config.hidden_size
        self.vision_projector = projector_model or nn.Linear(self.vision_hidden_size, self.hidden_size)

        geom_dim = self._infer_geom_dim()
        self.geometric_projector = nn.Linear(geom_dim, self.hidden_size)
        self.fusion_projector = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if config.use_spatial_token:
            self.spatial_embed_tokens = nn.Embedding(config.spatial_token_num, self.hidden_size)
        else:
            self.spatial_embed_tokens = None

        self.pad_token_id = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.text_config.vocab_size
        self.geom_feature_hook_enabled = False
        self.geom_feature_hook_max_steps = 100
        self.geom_feature_stats = []
        self._llava_vision_available = True
        self.prefix_lang_dropout_prob = getattr(config, "prefix_lang_dropout_prob", 0.0)
        self.prefix_image_dropout_prob = getattr(config, "prefix_image_dropout_prob", 0.0)

    def _infer_geom_dim(self) -> int:
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

    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        base_model = getattr(self.language_model, "model", None)
        vision_feats = None
        multi_view = isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5
        if multi_view:
            b, v = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(b * v, *pixel_values.shape[2:])
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
            # `MapAnythingLlava3DProcessor` already applies SigLIP preprocessing.
            # Normalizing again here shifts the input distribution and harms visual encoding.
            siglip_pixel_values = pixel_values.float().contiguous()
            vision_outputs = self.vision_tower(siglip_pixel_values)
            vision_feats = vision_outputs.last_hidden_state
            if getattr(self, "geom_feature_hook_enabled", False):
                self._record_geom_stats("vision_raw", vision_feats)

        if vision_feats is not None and vision_feats.shape[-1] != self.hidden_size:
            vision_feats = self.vision_projector(vision_feats)
            if getattr(self, "geom_feature_hook_enabled", False):
                self._record_geom_stats("vision_proj", vision_feats)
        if multi_view and vision_feats is not None:
            vision_feats = vision_feats.view(b, v * vision_feats.shape[1], vision_feats.shape[2])
        if vision_feats is not None:
            print(f"[mapanything_llava3d] vision_feats.shape: {tuple(vision_feats.shape)}")

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

        use_geom = getattr(self.config, "use_geometric_branch", True)
        if not use_geom:
            self._last_image_features = vision_feats
            return vision_feats

        if multi_view:
            geom_pixel_values = pixel_values.view(b, v, *pixel_values.shape[1:])
        else:
            geom_pixel_values = pixel_values
        # Geometry branch expects image-like ranges before its own normalization logic.
        # Convert SigLIP-normalized [-1, 1] tensors back to [0, 1] when needed.
        if isinstance(geom_pixel_values, torch.Tensor):
            with torch.no_grad():
                min_v = float(geom_pixel_values.detach().amin().item())
            if min_v < -0.05:
                geom_pixel_values = (geom_pixel_values * 0.5 + 0.5).clamp(0.0, 1.0)
        geometric_out = self.geometric_model(pixel_values=geom_pixel_values, intrinsics=intrinsic)
        geometric_features = geometric_out.last_hidden_state
        if geometric_features.dim() == 4:
            b, c, h, w = geometric_features.shape
            geometric_features = geometric_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        print(f"[mapanything_llava3d] geom_seq.shape: {tuple(geometric_features.shape)}")

        final_features = self.fusion_module(geometric_features, vision_feats)

        self._last_image_features = final_features
        return final_features

    def fusion_module(self, geometric_features, vision_features):
        geometric_features = self.geometric_projector(geometric_features).to(vision_features.dtype)
        geometric_global = geometric_features.mean(dim=1, keepdim=True)  # TODO: revisit geometric pooling strategy
        geometric_broadcast = geometric_global.expand(vision_features.shape[0], vision_features.shape[1], geometric_global.shape[-1])
        fused_features_pre = torch.cat([vision_features, geometric_broadcast], dim=-1)
        final_features = self.fusion_projector(fused_features_pre)
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
        image_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, MapAnythingLlava3DOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            embed = self.get_input_embeddings()
            vocab_size = embed.weight.shape[0]
            input_ids = input_ids.clamp(max=vocab_size - 1)
            inputs_embeds = embed(input_ids)

        image_features = None
        if pixel_values is not None:
            spatial_img_id = None
            if image_token_id is not None:
                image_token_id = int(image_token_id)
                if image_token_id > vocab_size - 1:
                    image_token_id = vocab_size - 1
            if image_token_index is not None:
                image_token_index = int(image_token_index)
                if image_token_index > vocab_size - 1:
                    image_token_index = vocab_size - 1
            if image_token_id is not None:
                spatial_img_id = int(image_token_id)
            elif image_token_index is not None:
                spatial_img_id = int(image_token_index)
            else:
                spatial_img_id = getattr(self.config, "image_token_index", None)

            image_features = self.get_image_features(pixel_values, intrinsic)

            if spatial_img_id is not None:
                image_mask = input_ids == spatial_img_id
            else:
                image_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            try:
                self._debug_last_spatial_img_id = int(spatial_img_id) if spatial_img_id is not None else None
            except Exception:
                self._debug_last_spatial_img_id = None
            self._debug_last_image_token_index = image_token_index
            self._debug_last_image_token_id = image_token_id
            self._debug_last_has_pixel_values = True
            self._debug_last_image_mask_any = bool(image_mask.any().item())
            with torch.no_grad():
                self._debug_last_image_mask_sum = image_mask.sum(dim=1).detach().cpu()
                self._debug_last_input_ids_head = input_ids[:, :16].detach().cpu()
                self._debug_last_image_features_shape = tuple(image_features.shape)
                self._debug_last_inputs_embeds_shape = tuple(inputs_embeds.shape)

            if image_mask.any():
                if image_features.shape[0] == inputs_embeds.shape[0]:
                    b, l, h = inputs_embeds.shape
                    img_b, img_s, img_h = image_features.shape
                    if img_h != h:
                        image_features = image_features.to(inputs_embeds.dtype)
                    mask_per_sample = image_mask.sum(dim=1)
                    if (mask_per_sample == img_s).all():
                        mask_exp = image_mask.unsqueeze(-1).expand(-1, -1, h)
                        image_features_flat = image_features.reshape(-1)
                        zero_embeds = torch.zeros_like(inputs_embeds)
                        zero_embeds = zero_embeds.masked_scatter(
                            mask_exp, image_features_flat.to(zero_embeds.dtype)
                        )
                        inputs_embeds = torch.where(mask_exp, zero_embeds, inputs_embeds)
                    else:
                        logger.warning(
                            "Image token count per sample does not match image feature sequence length. "
                            "Skipping image injection."
                        )
                else:
                    logger.warning("Batch size mismatch between input_ids and pixel_values. Skipping image injection.")

        if self.config.use_spatial_token and self.spatial_embed_tokens is not None:
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
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        intrinsic=None,
        attention_mask=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        if past_key_values is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["intrinsic"] = intrinsic
        return model_inputs
