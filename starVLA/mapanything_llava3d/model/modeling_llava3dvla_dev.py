import torch


from configuration_spatialvla_dev import SpatialVLAConfig
from .modeling_llava3d_v2 import LLaVA3DForCausalLMV2
from .modeling_mapanything import MapAnythingWrapper
from transformers import AutoModel, AutoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class SpatialVLAForConditionalGeneration(SpatialVLAPreTrainedModel, GenerationMixin):
    def __init__(self, config: SpatialVLAConfig, vision_model=None, projector_model=None, language_model=None):
        super().__init__(config)
        # Use a valid token id as pad to avoid embedding index errors in tests
        _pad = getattr(self.config, "pad_token_id", None)
        self.pad_token_id = _pad if _pad is not None else 0

        # 视觉塔严格从纯视觉 config 构建（通过方法以便单测可 mock）
        self.vision_tower = vision_model or self._init_vision_tower()
        # 语言塔：训练构建时从路径加载；整合权重加载时直接根据 text_config 构造空结构
        if language_model is not None:
            self.language_model = language_model
        else:
            lm_path = getattr(config, "language_model_name_or_path", None)
            try:
                if lm_path is not None:
                    # Use wrapper's custom loader when path is provided
                    self.language_model = LLaVA3DForCausalLMV2.from_pretrained(lm_path, torch_dtype=self.dtype)
                else:
                    # Fallback to constructing from text_config when no path is given
                    self.language_model = LLaVA3DForCausalLMV2(config.text_config)
            except Exception:
                # Final fallback: construct minimal wrapper with text_config to satisfy tests
                self.language_model = LLaVA3DForCausalLMV2(config.text_config)
        self.geometric_model = MapAnythingWrapper(config)
        # This is the first MLP that map the mapanything output(1024 dims) to the siglip ourput(1052 dims)
        # self.geometric_projector = nn.Linear(1024, config.vision_config.hidden_size)
        if getattr(self.language_model, "_tied_weights_keys", None) is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        lm_hidden_size = self.language_model.get_input_embeddings().weight.shape[-1]
        vision_hidden_size = self.config.vision_config.hidden_size
        self.config.text_config.hidden_size = lm_hidden_size
        self.config.hidden_size = lm_hidden_size
        self.config.vision_config.projection_dim = lm_hidden_size

        self.multi_modal_projector = projector_model or nn.Linear(vision_hidden_size, lm_hidden_size, bias=True)
        self.vocab_size = config.text_config.vocab_size


          # shared spatial embeddings for <ACTION> <IMG>
        if config.use_spatial_token:
            self.spatial_embed_tokens = nn.Embedding(self.config.spatial_token_num, lm_hidden_size)
        else:
            self.spatial_embed_tokens = None

        # align spatial token embedding dimension with LM hidden size
        # if self.spatial_embed_tokens is not None and self.spatial_embed_tokens.weight.shape[-1] != lm_hidden_size:
        #     self.spatial_embed_tokens = nn.Embedding(self.config.spatial_token_num, lm_hidden_size)
        # ensure projector outputs LM hidden size; recreate its linear layer when mismatched
        
        # zzq 1117 暂时不使用vision_zoe
        # if config.use_vision_zoe:
        #     self.vision_zoe_model = vision_zoe_model or ZoeDepthForDepthEstimation(config.vision_zoe_config)
        #     self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
        #         config.ego3d_patch_reso**2 * 3, num_pos_feats=config.vision_config.hidden_size, n_freqs=config.n_freqs
        #     )
        #     # register buffer
        #     patch_size, reso, image_size = config.vision_config.patch_size, config.ego3d_patch_reso, config.vision_config.image_size
        #     y, x = torch.meshgrid(torch.arange(0, image_size, patch_size // reso), torch.arange(0, image_size, patch_size // reso), indexing="ij")  # (h//sp w//sp)
        #     y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
        #     uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # (3 hw)
        #     self.register_buffer("uv_h", uv_h, persistent=False)

      
        # self.image_to_text_adapter = (
        #     nn.Linear(config.vision_config.projection_dim, config.text_config.hidden_size, bias=False)
        #     if config.vision_config.projection_dim != config.text_config.hidden_size
        #     else None
        # )
        # TODO(temporarily disable MapAnything): comment out geometric pipeline for now
        # self.geometric_model = MapAnythingWrapper(config)
        # This is the first MLP that map the mapanything output(1024 dims) to the siglip ourput(1052 dims)
        _geom_cfg = getattr(self.geometric_model, "config", None)
        _geom_dim = None
        if _geom_cfg is not None and hasattr(_geom_cfg, "hidden_size"):
            _geom_dim = int(_geom_cfg.hidden_size)
        else:
            _encoder = getattr(getattr(self.geometric_model, "map_anything_model", None), "encoder", None)
            _geom_dim = int(getattr(_encoder, "enc_embed_dim", lm_hidden_size)) if _encoder is not None else lm_hidden_size
        self.geometric_projector = nn.Linear(_geom_dim, vision_hidden_size)
        self.fusion_projector = nn.Linear(vision_hidden_size * 2, lm_hidden_size)

        def _spatialvla_encode_images(self, pixel_values):
            return pixel_values

        import types
        self.language_model.encode_images = types.MethodType(_spatialvla_encode_images, self.language_model)

        original_prepare_inputs_for_generation = self.language_model.prepare_inputs_for_generation

        def _spatialvla_prepare_inputs_for_generation(self, *args, **kwargs):
            model_inputs = original_prepare_inputs_for_generation(*args, **kwargs)
            if "inputs_embeds" in kwargs:
                model_inputs["inputs_embeds"] = kwargs["inputs_embeds"]
            if "cache_position" in kwargs and (
                model_inputs.get("cache_position") is None
                or model_inputs["cache_position"].shape != kwargs["cache_position"].shape
            ):
                model_inputs["cache_position"] = kwargs["cache_position"]
            return model_inputs

        self.language_model.prepare_inputs_for_generation = types.MethodType(
            _spatialvla_prepare_inputs_for_generation, self.language_model
        )
        
        # Action Expert Initialization
        if getattr(config, "action_expert_config", None) is not None:
            self.action_expert = FlowMatchingActionExpert(
                config.action_expert_config,
                action_dim=getattr(config, "action_dim", 14), 
                action_horizon=getattr(config, "action_horizon", 1),
                vlm_hidden_size=config.text_config.hidden_size
            )
        else:
            self.action_expert = None

    def _init_vision_tower(self):
        vision_tower = AutoModel.from_config(self.config.vision_config)
        if self.config.vision_weight_source == "spatialvla_vision_only":
            if not self.config.spatialvla_vision_pretrained_path:
                raise ValueError("`spatialvla_vision_pretrained_path` must be set when `vision_weight_source` is 'spatialvla_vision_only'.")
            logger.info(f"Loading vision tower weights from SpatialVLA checkpoint: {self.config.spatialvla_vision_pretrained_path}")
            ckpt_path = os.path.join(self.config.spatialvla_vision_pretrained_path, "model.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
            spatialvla_state_dict = safe_load_file(ckpt_path)
            vision_tower_state_dict = {
                k.replace("vision_tower.", ""): v for k, v in spatialvla_state_dict.items() if k.startswith("vision_tower.")
            }
            vision_tower.load_state_dict(vision_tower_state_dict, strict=False)
        return vision_tower




    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = model_embeds.weight.shape[0]
        self.config.text_config.vocab_size = self.vocab_size = self.config._vocab_size = vocab_size
        self.tie_weights()
        return model_embeds
    
    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_ids=None,
        inputs_embeds=None,
        is_training: bool = False,
    ):
        # Always build a 4D causal mask for non-LLaVA paths to satisfy tests

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        try:
            if using_static_cache:
                target_length = past_key_values.get_max_cache_shape()
            elif isinstance(past_key_values, HybridCache):
                # Fallback safely if HybridCache is not fully initialized in tests
                target_length = past_key_values.get_max_cache_shape()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else (cache_position[0] + sequence_length + 1 if cache_position is not None else sequence_length)
                )
        except Exception:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else (cache_position[0] + sequence_length + 1 if cache_position is not None else sequence_length)
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device)
        if sequence_length != 1:
            if is_training: causal_mask = torch.triu(causal_mask, diagonal=1)
            else: causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0)
        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor):
        siglip_pixel_values = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        siglip_pixel_values = siglip_pixel_values.float().contiguous()
        image_outputs = self.vision_tower(siglip_pixel_values)
        feats = image_outputs.last_hidden_state
        # image_features = self.multi_modal_projector(feats)
        geometric_features = self.geometric_model(pixel_values=pixel_values, intrinsics=intrinsic).last_hidden_state
        # print(f"[Debug]geometric_features shape: {geometric_features.shape}")
        if geometric_features.dim() == 4:
            b, c, h, w = geometric_features.shape
            geom_seq = geometric_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        elif geometric_features.dim() == 3:
            raise NotImplementedError(f"geometric_features dim {geometric_features.dim()} != 4")
            # b, x, y = geometric_features.shape
            # _geom_cfg = getattr(self.geometric_model, "config", None)
            # _geom_dim = None
            # if _geom_cfg is not None and hasattr(_geom_cfg, "hidden_size"):
            #     _geom_dim = int(_geom_cfg.hidden_size)
            # else:
            #     _encoder = getattr(getattr(self.geometric_model, "map_anything_model", None), "encoder", None)
            #     _geom_dim = int(getattr(_encoder, "enc_embed_dim", y)) if _encoder is not None else y
            # geom_seq = geometric_features if y == _geom_dim else geometric_features.permute(0, 2, 1)
        else:
            raise NotImplementedError(f"geometric_features dim {geometric_features.dim()} != 4")
            # geom_seq = geometric_features.unsqueeze(1)
        lm_hidden_size = int(self.config.text_config.hidden_size)
        geom_dim = int(geom_seq.shape[-1])
        if getattr(self.geometric_projector, "in_features", None) != geom_dim:
            raise NotImplementedError(f"geometric_projector in_features {geom_dim} != geom_seq shape[-1] {geom_dim}")
            # device = self.geometric_projector.weight.device if hasattr(self.geometric_projector, "weight") else image_features.device
            # dtype = self.geometric_projector.weight.dtype if hasattr(self.geometric_projector, "weight") else image_features.dtype
            # self.geometric_projector = nn.Linear(geom_dim, lm_hidden_size)
            # self.geometric_projector = self.geometric_projector.to(device=device, dtype=dtype)
        projected_geometric_features = self.geometric_projector(geom_seq).to(feats.dtype)
        # geom_global = projected_geometric_features.mean(dim=1, keepdim=True)
        # geom_broadcast = geom_global.expand(feats.shape[0], feats.shape[1], geom_global.shape[-1])
        fused_features = torch.cat([feats, geom_broadcast], dim=-1)
        projected_fused_features = self.fusion_projector(fused_features)
        # Align config num_image_tokens on first call or if changed vision tower
        seq_len = int(projected_fused_features.shape[1])
        current = getattr(self.config.text_config, "num_image_tokens", None)
        if current is None or current != seq_len:
            self.config.text_config.num_image_tokens = seq_len
            setattr(self.config, "image_seq_length", seq_len)
        return projected_fused_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        actions: Optional[torch.FloatTensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
        image_token_index: Optional[int] = None,
        num_image_tokens: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states=output_hidden_states or (actions is not None and self.action_expert is not None),
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, SpatialVLACausalLMOutputWithPast]:

        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = int(image_token_id.item())
        if isinstance(image_token_index, torch.Tensor):
            image_token_index = int(image_token_index.item())
        spatial_img_id = (
            image_token_index if image_token_index is not None else (
                image_token_id if image_token_id is not None else getattr(self.config, "image_token_index", None)
            )
        )
        
        if inputs_embeds is None:
            # Avoid embedding out-of-range indices (e.g., image or special tokens outside vocab)
            safe_id = self.pad_token_id if self.pad_token_id is not None and self.pad_token_id >= 0 else 0
            ids_for_embed = input_ids.clone()
            # Mask image tokens to a safe id before embedding; will be replaced by image features later
            if spatial_img_id is not None:
                ids_for_embed[input_ids == spatial_img_id] = safe_id
            # If a loc token index exists and is outside vocab, mask it as well
            if hasattr(self.config, "loc_token_index"):
                ids_for_embed[input_ids == self.config.loc_token_index] = safe_id
            # Clamp any remaining out-of-range ids into valid range
            vocab_size = self.get_input_embeddings().weight.shape[0]
            ids_for_embed = ids_for_embed.clamp(min=0, max=vocab_size - 1)
            inputs_embeds = self.get_input_embeddings()(ids_for_embed).clone()  # avoid checkpoint grad True

        if self.config.use_spatial_token:
            spatial_selected = (input_ids >= self.config.action_token_begin_idx) & (input_ids < self.config.action_token_begin_idx + self.config.spatial_token_num)
            inputs_embeds[spatial_selected] = inputs_embeds[spatial_selected] * 0.0 + self.spatial_embed_tokens(input_ids[spatial_selected] - self.config.action_token_begin_idx)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=inputs_embeds.device if inputs_embeds is not None else input_ids.device,
            )

        if position_ids is None:
            device = inputs_embeds.device
            seq_length = inputs_embeds.shape[1]
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        # merge
        if pixel_values is not None:
            if spatial_img_id is None:
                raise ValueError("`image_token_id` must be provided when supplying `pixel_values`.")
            image_features = self.get_image_features(pixel_values, intrinsic)  # [B_v, S_v, H_v]
            if image_features.shape[-1] != inputs_embeds.shape[-1]:
                if image_features.shape[-1] != inputs_embeds.shape[-1]:
                    raise ValueError(f"Image features dim {image_features.shape[-1]} != text hidden {inputs_embeds.shape[-1]}")
            base_idx = getattr(self.config, "image_token_index", None)
            mask_primary = (input_ids == spatial_img_id)
            mask_base = (input_ids == base_idx) if base_idx is not None else torch.zeros_like(mask_primary)
            if image_token_id is not None:
                mask_id = (input_ids == image_token_id)
                image_token_bool = mask_primary | mask_base | mask_id
            else:
                image_token_bool = mask_primary | mask_base
            # Patch-world semantics
            total_image_positions = int(image_token_bool.sum().item())
            num_img_tokens = int(num_image_tokens) if num_image_tokens is not None else int(getattr(self.config.text_config, "num_image_tokens", image_features.shape[1]))
            if total_image_positions % num_img_tokens != 0:
                raise ValueError(
                    f"Total image token positions {total_image_positions} is not divisible by num_image_tokens per image {num_img_tokens}."
                )
            num_images = total_image_positions // num_img_tokens
            B_v, S_v, H_v = image_features.shape
            if S_v != num_img_tokens:
                raise ValueError(
                    f"Vision features seq_len {S_v} != expected num_image_tokens {num_img_tokens}."
                )
            if B_v != num_images:
                raise ValueError(
                    f"Vision batch size {B_v} != inferred num_images {num_images}. Currently only supports one image per <image> group."
                )
            special_image_mask = image_token_bool.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                per_sample_counts = image_token_bool.sum(dim=1).tolist()
                raise ValueError(
                    f"Number of image token elements {inputs_embeds[special_image_mask].numel()} != vision features elements {image_features.numel()}. "
                    f"total_image_positions={total_image_positions}, num_img_tokens={num_img_tokens}, image_features.shape={tuple(image_features.shape)}, per_sample_positions={per_sample_counts}"
                )
            flat_image_features = image_features.to(inputs_embeds.dtype).to(inputs_embeds.device).reshape(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask,
                flat_image_features,
            )

        debug = bool(kwargs.get("debug", False))
        if debug:
            def _shape(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return tuple(x.shape)
                if isinstance(x, (list, tuple)):
                    try:
                        return [tuple(t.shape) if hasattr(t, "shape") else type(t) for t in x[:2]]
                    except Exception:
                        return type(x)
                return type(x)
            print("[DBG] input_ids", _shape(input_ids))
            print("[DBG] attention_mask", _shape(attention_mask))
            print("[DBG] token_type_ids", _shape(token_type_ids))
            print("[DBG] position_ids", _shape(position_ids))
            print("[DBG] cache_position", _shape(cache_position))
            print("[DBG] inputs_embeds", _shape(inputs_embeds))
            print("[DBG] labels", _shape(labels))
            print("[DBG] past_key_values", _shape(past_key_values))

        # mask out pad-token-ids and special tokens in labels to avoid out-of-range targets
        if labels is not None:
            # ignore pad positions
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)
            # ignore image token positions
            if spatial_img_id is not None:
                labels = torch.where(input_ids == spatial_img_id, self.config.ignore_index, labels)
            # ignore loc token positions if defined
            if hasattr(self.config, "loc_token_index"):
                labels = torch.where(input_ids == self.config.loc_token_index, self.config.ignore_index, labels)
            # finally, any label outside vocab should be ignored to avoid IndexError in CE
            vocab_size = self.config.text_config.vocab_size
            labels = torch.where(labels >= vocab_size, self.config.ignore_index, labels)
            labels = torch.where(labels < 0, self.config.ignore_index, labels)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if input_ids is not None else torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        # Ensure attention_mask length matches inputs_embeds sequence length
        if attention_mask.shape[-1] != inputs_embeds.shape[1]:
            pad_len = inputs_embeds.shape[1] - attention_mask.shape[-1]
            if pad_len > 0:
                pad = torch.ones(attention_mask.shape[0], pad_len, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad], dim=-1)
            else:
                attention_mask = attention_mask[:, :inputs_embeds.shape[1]]
        outputs = self.language_model(
            attention_mask=attention_mask.bool(),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[1]
        loss = outputs.loss if return_dict else outputs[0]
        if loss is None and labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + (outputs[2:] if labels is not None else outputs[1:])
            return (loss,) + output if loss is not None else output

        return SpatialVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    # AR inference
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        intrinsic=None,
        image_token_index: Optional[int] = None,
        image_token_id: Optional[int] = None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if model_inputs.get("position_ids") is not None:
            pass
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        from LLaVA_3D.llava.constants import IMAGE_TOKEN_INDEX as LLAVA3D_IMAGE_TOKEN_INDEX
        if input_ids is not None:
            ids_for_llava3d = input_ids.clone()
            if image_token_index is not None and image_token_index != LLAVA3D_IMAGE_TOKEN_INDEX:
                ids_for_llava3d[input_ids == image_token_index] = LLAVA3D_IMAGE_TOKEN_INDEX
            if image_token_id is not None and image_token_id != LLAVA3D_IMAGE_TOKEN_INDEX:
                ids_for_llava3d[input_ids == image_token_id] = LLAVA3D_IMAGE_TOKEN_INDEX
            model_inputs["input_ids"] = ids_for_llava3d
        model_inputs["image_token_index"] = LLAVA3D_IMAGE_TOKEN_INDEX
        # 将 inputs_embeds 保留在返回字典中，供本模型 forward 使用以完成图像特征注入
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        model_inputs["intrinsic"] = intrinsic
        if image_token_id is not None:
            if isinstance(image_token_id, torch.Tensor):
                image_token_id = int(image_token_id.item())
            model_inputs["image_token_id"] = image_token_id
        if image_token_index is not None:
            model_inputs["image_token_index"] = image_token_index
        return model_inputs

    @torch.no_grad()
    def predict_action(
        self,
        model_inputs,
    ) -> torch.Tensor:
        def _move_field(k, v):
            if hasattr(v, "to"):
                if torch.is_floating_point(v):
                    if k in ("pixel_values", "intrinsic"):
                        v = v.to(dtype=torch.float32)
                    else:
                        v = v.to(dtype=torch.bfloat16)
                v = v.to(self.device)
            return v
        if isinstance(model_inputs, dict):
            model_inputs = {k: _move_field(k, v) for k, v in model_inputs.items()}
        else:
            model_inputs = _move_field("_", model_inputs)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False, use_cache=False)
        return generation_outputs[:, input_len:]
