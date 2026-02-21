# coding=utf-8
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
    task_hidden_states: Optional[torch.FloatTensor] = None


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
        semantic_heads = int(getattr(config, "semantic_num_heads", 8))
        if semantic_heads < 1 or (self.hidden_size % semantic_heads != 0):
            semantic_heads = 1
        self.semantic_num_heads = semantic_heads
        self.semantic_query_max_tokens = int(getattr(config, "semantic_query_max_tokens", 64))
        self.semantic_topk_tokens = int(getattr(config, "semantic_topk_tokens", 32))
        self.semantic_anchor_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.semantic_num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.semantic_anchor_norm = nn.LayerNorm(self.hidden_size)
        self.vision_semantic_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.semantic_num_heads,
            dropout=0.0,
            batch_first=True,
        )

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
        self.debug_health_trace_enabled = bool(getattr(config, "debug_health_trace_enabled", True))
        self.debug_health_trace_max_len = int(getattr(config, "debug_health_trace_max_len", 128))
        self.debug_health_trace: List[Dict[str, Any]] = []
        self.debug_first_nonfinite_stage: Optional[str] = None
        self.debug_first_nonfinite_record: Optional[Dict[str, float]] = None
        self.debug_shape_log_interval = int(getattr(config, "debug_shape_log_interval", 0))
        self._debug_shape_log_counter = 0
        self.debug_timing_enabled = bool(getattr(config, "debug_timing_enabled", True))
        self.debug_timing_cuda_sync = bool(getattr(config, "debug_timing_cuda_sync", True))
        self.debug_last_geom_model_forward_ms: float = 0.0
        self.debug_last_geom_feature_extract_ms: float = 0.0

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

    def _build_language_queries(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        image_token_index: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            return None, None
        bsz, seq_len, hidden = inputs_embeds.shape
        device = inputs_embeds.device
        if attention_mask is None:
            active = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        else:
            active = attention_mask.to(device=device, dtype=torch.bool)
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device=device)
            if image_token_index is not None:
                lang_mask = (input_ids != int(image_token_index)) & active
            else:
                lang_mask = active
        else:
            lang_mask = active

        max_tokens = max(1, int(self.semantic_query_max_tokens))
        max_valid = int(lang_mask.sum(dim=1).max().item()) if bsz > 0 else 0
        query_len = max(1, min(max_tokens, max_valid if max_valid > 0 else 1))
        queries = inputs_embeds.new_zeros((bsz, query_len, hidden))
        query_mask = torch.zeros((bsz, query_len), dtype=torch.bool, device=device)

        for bid in range(bsz):
            valid_idx = torch.nonzero(lang_mask[bid], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                valid_idx = torch.tensor([0], device=device, dtype=torch.long)
            if valid_idx.numel() > query_len:
                sample_pos = torch.linspace(
                    0,
                    float(valid_idx.numel() - 1),
                    steps=query_len,
                    device=device,
                ).long()
                valid_idx = valid_idx[sample_pos]
            token_query = inputs_embeds[bid, valid_idx]
            take_n = token_query.shape[0]
            queries[bid, :take_n] = token_query
            query_mask[bid, :take_n] = True
        return queries, query_mask

    @staticmethod
    def _tensor_health(tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            t = tensor.detach()
            if t.numel() == 0:
                return {
                    "shape": tuple(t.shape),
                    "dtype": str(t.dtype),
                    "finite_ratio": 1.0,
                    "nan_count": 0,
                    "inf_count": 0,
                    "absmax": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                }
            tf = t.float()
            finite = torch.isfinite(tf)
            nan_count = int(torch.isnan(tf).sum().item())
            inf_count = int(torch.isinf(tf).sum().item())
            return {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "finite_ratio": float(finite.float().mean().item()),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "absmax": float(tf.abs().max().item()),
                "mean": float(tf.mean().item()),
                "std": float(tf.std(unbiased=False).item()),
            }

    def _reset_health_trace(self):
        self.debug_health_trace = []
        self.debug_first_nonfinite_stage = None
        self.debug_first_nonfinite_record = None

    def _record_health(self, stage: str, tensor: Optional[torch.Tensor]):
        if (not self.debug_health_trace_enabled) or (tensor is None) or (not isinstance(tensor, torch.Tensor)):
            return
        record = {"stage": str(stage)}
        try:
            record.update(self._tensor_health(tensor))
        except Exception as e:
            record["error"] = str(e)
        self.debug_health_trace.append(record)
        if len(self.debug_health_trace) > self.debug_health_trace_max_len:
            self.debug_health_trace = self.debug_health_trace[-self.debug_health_trace_max_len :]
        if (
            self.debug_first_nonfinite_stage is None
            and float(record.get("finite_ratio", 1.0)) < 1.0
        ):
            self.debug_first_nonfinite_stage = str(stage)
            self.debug_first_nonfinite_record = {
                "stage": str(stage),
                "finite_ratio": float(record.get("finite_ratio", 1.0)),
                "nan_count": float(record.get("nan_count", 0.0)),
                "inf_count": float(record.get("inf_count", 0.0)),
                "absmax": float(record.get("absmax", 0.0)),
            }

    def _maybe_log_tensor_shape(self, tag: str, tensor: Optional[torch.Tensor]):
        if self.debug_shape_log_interval <= 0:
            return
        if not isinstance(tensor, torch.Tensor):
            return
        self._debug_shape_log_counter += 1
        if self._debug_shape_log_counter % self.debug_shape_log_interval != 0:
            return
        logger.info("[mapanything_llava3d] %s.shape=%s dtype=%s", tag, tuple(tensor.shape), tensor.dtype)

    def _maybe_cuda_synchronize(self, tensor: Optional[torch.Tensor] = None):
        if not self.debug_timing_enabled or not self.debug_timing_cuda_sync:
            return
        if not torch.cuda.is_available():
            return
        try:
            if isinstance(tensor, torch.Tensor):
                if tensor.device.type == "cuda":
                    torch.cuda.synchronize(device=tensor.device)
            else:
                torch.cuda.synchronize()
        except Exception:
            # Timing must never break training.
            pass

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        intrinsic: torch.FloatTensor,
        language_queries: Optional[torch.Tensor] = None,
        language_query_mask: Optional[torch.Tensor] = None,
    ):
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
        self._record_health("vision_feats", vision_feats)
        if multi_view and vision_feats is not None:
            vision_feats = vision_feats.view(b, v * vision_feats.shape[1], vision_feats.shape[2])
        self._maybe_log_tensor_shape("vision_feats", vision_feats)

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
            self.debug_last_geom_model_forward_ms = 0.0
            self.debug_last_geom_feature_extract_ms = 0.0
            self._last_image_features = vision_feats
            if isinstance(vision_feats, torch.Tensor):
                token_num = min(max(1, self.semantic_topk_tokens), vision_feats.shape[1])
                self._last_task_tokens = vision_feats[:, :token_num].contiguous()
            else:
                self._last_task_tokens = None
            self._record_health("image_features_output_vision_only", vision_feats)
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
        geom_extract_t0 = None
        geom_forward_t1 = None
        if self.debug_timing_enabled:
            self._maybe_cuda_synchronize(geom_pixel_values if isinstance(geom_pixel_values, torch.Tensor) else None)
            geom_extract_t0 = time.perf_counter()
        geometric_out = self.geometric_model(pixel_values=geom_pixel_values, intrinsics=intrinsic)
        if self.debug_timing_enabled and geom_extract_t0 is not None:
            self._maybe_cuda_synchronize(getattr(geometric_out, "last_hidden_state", None))
            geom_forward_t1 = time.perf_counter()
            self.debug_last_geom_model_forward_ms = float((geom_forward_t1 - geom_extract_t0) * 1000.0)
        geometric_features = geometric_out.last_hidden_state
        if geometric_features.dim() == 4:
            b, c, h, w = geometric_features.shape
            geometric_features = geometric_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.debug_timing_enabled and geom_extract_t0 is not None:
            self._maybe_cuda_synchronize(geometric_features)
            geom_extract_t1 = time.perf_counter()
            self.debug_last_geom_feature_extract_ms = float((geom_extract_t1 - geom_extract_t0) * 1000.0)
        self._record_health("geometric_features", geometric_features)
        self._maybe_log_tensor_shape("geom_seq", geometric_features)

        final_features, task_tokens = self.fusion_module(
            geometric_features,
            vision_feats,
            language_queries=language_queries,
            language_query_mask=language_query_mask,
        )
        self._record_health("image_features_output_fused", final_features)

        self._last_image_features = final_features
        self._last_task_tokens = task_tokens
        return final_features

    def fusion_module(
        self,
        geometric_features,
        vision_features,
        language_queries: Optional[torch.Tensor] = None,
        language_query_mask: Optional[torch.Tensor] = None,
    ):
        geometric_features = self.geometric_projector(geometric_features).to(vision_features.dtype)
        self._record_health("fusion/geometric_projected", geometric_features)
        if language_queries is None:
            query_len = min(max(1, self.semantic_query_max_tokens), vision_features.shape[1])
            semantic_query = vision_features[:, :query_len]
            semantic_query_mask = None
        else:
            semantic_query = language_queries.to(device=vision_features.device, dtype=vision_features.dtype)
            semantic_query_mask = language_query_mask
            if isinstance(semantic_query_mask, torch.Tensor):
                semantic_query_mask = semantic_query_mask.to(device=vision_features.device, dtype=torch.bool)
                semantic_query = semantic_query * semantic_query_mask.unsqueeze(-1).to(dtype=semantic_query.dtype)

        semantic_attn_out, semantic_attn_weights = self.semantic_anchor_attention(
            query=semantic_query,
            key=geometric_features,
            value=geometric_features,
            need_weights=True,
            average_attn_weights=False,
        )
        self._record_health("fusion/semantic_attn_out", semantic_attn_out)
        self._record_health("fusion/semantic_attn_weights", semantic_attn_weights)
        semantic_attn_out = self.semantic_anchor_norm(semantic_attn_out + semantic_query)
        self._record_health("fusion/semantic_attn_out_post_norm", semantic_attn_out)
        if semantic_attn_weights.dim() == 4:
            # [B, H, Nq, Ng] -> [B, Ng]
            token_scores = semantic_attn_weights.mean(dim=(1, 2))
        else:
            # [B, Nq, Ng] -> [B, Ng]
            token_scores = semantic_attn_weights.mean(dim=1)
        self._record_health("fusion/token_scores", token_scores)
        topk_num = min(max(1, self.semantic_topk_tokens), geometric_features.shape[1])
        topk_indices = torch.topk(token_scores, k=topk_num, dim=-1).indices
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, geometric_features.shape[-1])
        task_tokens = torch.gather(geometric_features, dim=1, index=gather_idx)
        semantic_summary = semantic_attn_out.mean(dim=1, keepdim=True)
        task_tokens = task_tokens + semantic_summary
        self._record_health("fusion/task_tokens", task_tokens)

        vision_context, _ = self.vision_semantic_attention(
            query=vision_features,
            key=task_tokens,
            value=task_tokens,
            need_weights=False,
        )
        fused_features_pre = torch.cat([vision_features, vision_context], dim=-1)
        self._record_health("fusion/fused_features_pre", fused_features_pre)
        final_features = self.fusion_projector(fused_features_pre)
        self._record_health("fusion/final_features", final_features)
        return final_features, task_tokens

        
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
        self._reset_health_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embed = self.get_input_embeddings()
        vocab_size = embed.weight.shape[0]
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            input_ids = input_ids.clamp(max=vocab_size - 1)
            inputs_embeds = embed(input_ids)
        self._record_health("forward/inputs_embeds_init", inputs_embeds)

        image_features = None
        task_tokens = None
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

            language_queries, language_query_mask = self._build_language_queries(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_token_index=spatial_img_id,
            )
            image_features = self.get_image_features(
                pixel_values,
                intrinsic,
                language_queries=language_queries,
                language_query_mask=language_query_mask,
            )
            task_tokens = getattr(self, "_last_task_tokens", None)
            self._record_health("forward/image_features", image_features)

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
                        self._record_health("forward/inputs_embeds_after_image_injection", inputs_embeds)
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
                    self._record_health("forward/inputs_embeds_after_lang_dropout", inputs_embeds)

        self._record_health("forward/inputs_embeds_before_language_model", inputs_embeds)

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
        try:
            if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                self._record_health("forward/lm_hidden_0", outputs.hidden_states[0])
                self._record_health("forward/lm_hidden_last", outputs.hidden_states[-1])
        except Exception:
            pass

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
            task_hidden_states=task_tokens if pixel_values is not None else None,
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
