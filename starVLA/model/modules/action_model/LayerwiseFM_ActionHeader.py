# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


DiTConfig = {"num_layers": 36, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32}

class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    use_concat_cross_context: bool = field(
        default=False,
        metadata={"help": "If true, use concat(vl_emb, hidden_states) as cross-attention context."},
    )
    cross_attention_assert_inputs: bool = field(
        default=True,
        metadata={"help": "If true, validate layerwise cross-attention input layer count and shape."},
    )
    cross_attention_debug_log_interval: int = field(
        default=0,
        metadata={"help": "If >0, log layerwise hidden-state stats every N calls."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)




DiTConfig = {"num_layers": 36, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32} # default for qwen2.5-vl


class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        num_vl_layers = global_config.framework.mapanything_llava3d.num_vl_layers
        cfg_num_layers = None
        try:
            if isinstance(diffusion_model_cfg, dict):
                cfg_num_layers = diffusion_model_cfg.get("num_layers", None)
            else:
                cfg_num_layers = getattr(diffusion_model_cfg, "num_layers", None)
        except Exception:
            cfg_num_layers = None
        if cfg_num_layers is None:
            effective_num_layers = num_vl_layers
        else:
            cfg_num_layers = int(cfg_num_layers)
            effective_num_layers = min(cfg_num_layers, num_vl_layers)

        DiTConfig["num_layers"] = effective_num_layers
        DiTConfig["input_embedding_dim"] = global_config.framework.mapanything_llava3d.vl_hidden_dim
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        diffusion_model_cfg.update(DiTConfig)
        diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"]
        self.input_embedding_dim = global_config.framework.mapanything_llava3d.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg)
        if isinstance(diffusion_model_cfg, dict):
            dit_output_dim = diffusion_model_cfg.get("output_dim", self.input_embedding_dim)
        else:
            dit_output_dim = getattr(diffusion_model_cfg, "output_dim", self.input_embedding_dim)
        self.dit_out_hidden_size = dit_output_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=action_config.state_dim,
            output_dim=self.input_embedding_dim,
        ) if action_config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.dit_out_hidden_size,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        self.use_concat_cross_context = bool(_cfg_get(action_config, "use_concat_cross_context", False))
        self.use_task_token_context = bool(_cfg_get(action_config, "use_task_token_context", True))
        self.cross_attention_assert_inputs = bool(_cfg_get(action_config, "cross_attention_assert_inputs", True))
        log_interval = _cfg_get(action_config, "cross_attention_debug_log_interval", 0)
        self.cross_attention_debug_log_interval = int(log_interval) if log_interval is not None else 0
        self._layerwise_forward_calls = 0
        self._last_dit_layer_means = []
        self._last_dit_layer_vars = []
        self._last_loss_breakdown = {}
        self._last_rrr = None
        self._logged_task_token_num_mismatch = False
        self.feedback_context_scale = float(_cfg_get(action_config, "feedback_context_scale", 1.0))

        self.world_hidden_dim = int(_cfg_get(action_config, "world_hidden_dim", 512))
        self.world_num_layers = int(_cfg_get(action_config, "world_num_layers", 4))
        self.world_num_heads = int(_cfg_get(action_config, "world_num_heads", 8))
        self.world_max_params = int(_cfg_get(action_config, "world_predictor_max_params", 30_000_000))
        self.num_task_tokens = int(_cfg_get(action_config, "num_task_tokens", 32))
        if self.world_num_heads < 1 or (self.world_hidden_dim % self.world_num_heads != 0):
            self.world_num_heads = 1
        state_dim = int(_cfg_get(action_config, "state_dim", 0) or 0)
        self.world_task_proj = nn.Linear(self.input_embedding_dim, self.world_hidden_dim)
        self.world_action_proj = nn.Linear(self.action_dim, self.world_hidden_dim)
        self.world_state_proj = nn.Linear(state_dim, self.world_hidden_dim) if state_dim > 0 else None
        world_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.world_hidden_dim,
            nhead=self.world_num_heads,
            dim_feedforward=self.world_hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.world_predictor = nn.TransformerEncoder(world_encoder_layer, num_layers=self.world_num_layers)
        self.world_delta_head = nn.Sequential(
            nn.LayerNorm(self.world_hidden_dim),
            nn.Linear(self.world_hidden_dim, self.world_hidden_dim),
            nn.GELU(),
            nn.Linear(self.world_hidden_dim, self.input_embedding_dim),
        )
        self.residual_proj = nn.Linear(self.input_embedding_dim, self.world_hidden_dim)
        self.temb_proj = nn.Linear(self.input_embedding_dim, self.world_hidden_dim)
        self.drift_head = nn.Sequential(
            nn.LayerNorm(self.world_hidden_dim * 2),
            nn.Linear(self.world_hidden_dim * 2, self.world_hidden_dim),
            nn.GELU(),
            nn.Linear(self.world_hidden_dim, self.action_dim),
        )
        self.geo_weight_diag = nn.Parameter(torch.ones(self.input_embedding_dim))

        self.loss_w_fm = float(_cfg_get(action_config, "loss_w_fm", 1.0))
        self.loss_w_dyn = float(_cfg_get(action_config, "loss_w_dyn", 0.2))
        self.loss_w_geo = float(_cfg_get(action_config, "loss_w_geo", 0.1))
        self.loss_w_reg = float(_cfg_get(action_config, "loss_w_reg", 1e-3))
        self.base_loss_w_fm = float(self.loss_w_fm)
        self.base_loss_w_dyn = float(self.loss_w_dyn)
        self.base_loss_w_geo = float(self.loss_w_geo)
        self.base_loss_w_reg = float(self.loss_w_reg)
        self._runtime_loss_weight_override: Optional[Dict[str, float]] = None
        self.drift_eta = float(_cfg_get(action_config, "drift_eta", 0.2))
        self.rrr_threshold = float(_cfg_get(action_config, "rrr_replan_threshold", 2.5))
        self.rrr_eps = float(_cfg_get(action_config, "rrr_eps", 1e-4))
        self.rrr_max_replans = int(_cfg_get(action_config, "rrr_max_replans", 1))
        self.world_action_source = str(_cfg_get(action_config, "world_action_source", "teacher")).lower()
        if self.world_action_source not in ("teacher", "noisy"):
            logger.warning(
                "Invalid world_action_source=%s, fallback to `teacher`.",
                self.world_action_source,
            )
            self.world_action_source = "teacher"

        world_modules = [
            self.world_task_proj,
            self.world_action_proj,
            self.world_predictor,
            self.world_delta_head,
            self.residual_proj,
            self.temb_proj,
            self.drift_head,
        ]
        if self.world_state_proj is not None:
            world_modules.append(self.world_state_proj)
        world_params = sum(p.numel() for module in world_modules for p in module.parameters())
        self.world_predictor_param_count = int(world_params)
        if world_params > self.world_max_params:
            logger.warning(
                "WorldPredictor parameter budget exceeded: params=%d > max=%d",
                world_params,
                self.world_max_params,
            )
        self._last_health_trace = []
        self._first_nonfinite_stage = None
        self._first_nonfinite_record = None
        logger.info(
            "LayerwiseFMActionHead initialized: use_concat_cross_context=%s, cross_attention_assert_inputs=%s, cross_attention_debug_log_interval=%d, world_params=%d, world_action_source=%s",
            self.use_concat_cross_context,
            self.cross_attention_assert_inputs,
            self.cross_attention_debug_log_interval,
            self.world_predictor_param_count,
            self.world_action_source,
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    @staticmethod
    def _is_main_process():
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _validate_cross_attention_inputs(self, saction_embs, vl_embs_list):
        if not self.cross_attention_assert_inputs:
            return
        if not isinstance(vl_embs_list, (list, tuple)) or len(vl_embs_list) == 0:
            raise ValueError("`vl_embs_list` must be a non-empty list/tuple of layerwise embeddings.")
        expected_layers = len(self.model.transformer_blocks)
        if len(vl_embs_list) != expected_layers:
            raise ValueError(
                f"Layerwise cross-attention mismatch: got {len(vl_embs_list)} vl layers, expected {expected_layers} DiT blocks."
            )

        expected_batch = saction_embs.shape[0]
        expected_hidden = saction_embs.shape[-1]
        for layer_idx, vl_emb in enumerate(vl_embs_list):
            if vl_emb.ndim != 3:
                raise ValueError(
                    f"vl_embs_list[{layer_idx}] must be 3D [B,S,D], got shape={tuple(vl_emb.shape)}."
                )
            if vl_emb.shape[0] != expected_batch:
                raise ValueError(
                    f"Batch mismatch at vl_embs_list[{layer_idx}]: got B={vl_emb.shape[0]}, expected B={expected_batch}."
                )
            if vl_emb.shape[-1] != expected_hidden:
                raise ValueError(
                    f"Hidden dim mismatch at vl_embs_list[{layer_idx}]: got D={vl_emb.shape[-1]}, expected D={expected_hidden}."
                )
            if vl_emb.device != saction_embs.device:
                raise ValueError(
                    f"Device mismatch at vl_embs_list[{layer_idx}]: got {vl_emb.device}, expected {saction_embs.device}."
                )

    def _maybe_log_layerwise_stats(self, log_context):
        if self.cross_attention_debug_log_interval <= 0:
            return
        if self._layerwise_forward_calls % self.cross_attention_debug_log_interval != 0:
            return
        if not self._is_main_process():
            return
        if not self._last_dit_layer_means or not self._last_dit_layer_vars:
            return
        logger.info(
            "LayerwiseCrossAttn[%s] call=%d concat_cross=%s layers=%d mean(first,last)=(%.6f,%.6f) var(first,last)=(%.6f,%.6f)",
            log_context,
            self._layerwise_forward_calls,
            self.use_concat_cross_context,
            len(self._last_dit_layer_means),
            self._last_dit_layer_means[0],
            self._last_dit_layer_means[-1],
            self._last_dit_layer_vars[0],
            self._last_dit_layer_vars[-1],
        )

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
            return {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "finite_ratio": float(finite.float().mean().item()),
                "nan_count": int(torch.isnan(tf).sum().item()),
                "inf_count": int(torch.isinf(tf).sum().item()),
                "absmax": float(tf.abs().max().item()),
                "mean": float(tf.mean().item()),
                "std": float(tf.std(unbiased=False).item()),
            }

    def _record_health(self, stage: str, tensor: torch.Tensor):
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        record = {"stage": str(stage)}
        try:
            record.update(self._tensor_health(tensor))
        except Exception as e:
            record["error"] = str(e)
        self._last_health_trace.append(record)
        if (
            self._first_nonfinite_stage is None
            and float(record.get("finite_ratio", 1.0)) < 1.0
        ):
            self._first_nonfinite_stage = str(stage)
            self._first_nonfinite_record = {
                "stage": str(stage),
                "finite_ratio": float(record.get("finite_ratio", 1.0)),
                "nan_count": float(record.get("nan_count", 0.0)),
                "inf_count": float(record.get("inf_count", 0.0)),
                "absmax": float(record.get("absmax", 0.0)),
            }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if values.ndim != 1:
            raise ValueError(f"`values` must be 1D [B], got shape={tuple(values.shape)}")
        if mask is None:
            return values.mean()
        if mask.ndim != 1:
            mask = mask.view(-1)
        if mask.shape[0] != values.shape[0]:
            raise ValueError(
                f"Mask batch mismatch: values.shape[0]={values.shape[0]}, mask.shape[0]={mask.shape[0]}"
            )
        # Important: avoid in-place ops on mask because the same mask can be
        # reused across multiple loss branches in one forward pass, which can
        # break autograd version tracking.
        mask = mask.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            return values.new_tensor(0.0)
        return torch.sum(values * mask) / denom

    def set_loss_weight_override(
        self,
        *,
        loss_w_fm: Optional[float] = None,
        loss_w_dyn: Optional[float] = None,
        loss_w_geo: Optional[float] = None,
        loss_w_reg: Optional[float] = None,
    ):
        override = {}
        if loss_w_fm is not None:
            override["loss_w_fm"] = float(loss_w_fm)
        if loss_w_dyn is not None:
            override["loss_w_dyn"] = float(loss_w_dyn)
        if loss_w_geo is not None:
            override["loss_w_geo"] = float(loss_w_geo)
        if loss_w_reg is not None:
            override["loss_w_reg"] = float(loss_w_reg)
        self._runtime_loss_weight_override = override if len(override) > 0 else None

    def clear_loss_weight_override(self):
        self._runtime_loss_weight_override = None

    def _resolve_loss_weights(
        self,
        loss_weight_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        weights = {
            "loss_w_fm": float(self.base_loss_w_fm),
            "loss_w_dyn": float(self.base_loss_w_dyn),
            "loss_w_geo": float(self.base_loss_w_geo),
            "loss_w_reg": float(self.base_loss_w_reg),
        }
        for override in (self._runtime_loss_weight_override, loss_weight_override):
            if not isinstance(override, dict):
                continue
            for key in ("loss_w_fm", "loss_w_dyn", "loss_w_geo", "loss_w_reg"):
                if key in override and override[key] is not None:
                    weights[key] = float(override[key])
        return weights

    def _apply_layerwise_cross_attention(
        self,
        saction_embs,
        vl_embs_list,
        temb,
        encoder_attention_mask=None,
        task_tokens: Optional[torch.Tensor] = None,
        log_context="train",
    ):
        """
        Apply layerwise cross-attention between state-action embeddings and vision-language embeddings.

        Args:
            saction_embs: Tensor of shape (B, seq_length, embedding_dim)
            vl_embs_list: List of tensors, each of shape (B, seq_length, embedding_dim)
            temb: Tensor of shape (B, embedding_dim)

        Returns:
            hidden_states: Tensor of shape (B, seq_length, embedding_dim)
        """
        self._validate_cross_attention_inputs(saction_embs, vl_embs_list)
        hidden_states = saction_embs
        if isinstance(encoder_attention_mask, torch.Tensor):
            encoder_attention_mask = encoder_attention_mask.to(device=saction_embs.device)
            if encoder_attention_mask.dtype != torch.bool:
                encoder_attention_mask = encoder_attention_mask != 0
        self._layerwise_forward_calls += 1
        self._last_dit_layer_means = []
        self._last_dit_layer_vars = []
        self._record_health("cross_attn/input_saction_embs", hidden_states)
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            if self.use_concat_cross_context:
                cross_context = torch.cat((vl_embs_list[layer_idx], hidden_states), dim=1)
                layer_encoder_attention_mask = None
                if isinstance(encoder_attention_mask, torch.Tensor):
                    hidden_tokens = torch.ones(
                        hidden_states.shape[:2],
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                    layer_encoder_attention_mask = torch.cat(
                        (encoder_attention_mask, hidden_tokens),
                        dim=1,
                    )
            else:
                cross_context = vl_embs_list[layer_idx]
                layer_encoder_attention_mask = encoder_attention_mask
            if self.use_task_token_context and isinstance(task_tokens, torch.Tensor):
                task_context = task_tokens.to(device=saction_embs.device, dtype=saction_embs.dtype)
                cross_context = torch.cat((cross_context, task_context), dim=1)
                if isinstance(layer_encoder_attention_mask, torch.Tensor):
                    task_mask = torch.ones(
                        task_context.shape[:2],
                        dtype=torch.bool,
                        device=task_context.device,
                    )
                    layer_encoder_attention_mask = torch.cat((layer_encoder_attention_mask, task_mask), dim=1)
            self._record_health(f"cross_attn/layer_{layer_idx}_context", cross_context)
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=cross_context,
                encoder_attention_mask=layer_encoder_attention_mask,
                temb=temb,
            )
            self._record_health(f"cross_attn/layer_{layer_idx}_hidden", hidden_states)
            stats = hidden_states.detach().float()
            self._last_dit_layer_means.append(stats.mean().item())
            self._last_dit_layer_vars.append(stats.var(unbiased=False).item())
        self._maybe_log_layerwise_stats(log_context=log_context)
        return hidden_states

    def _process_output(self, hidden_states, temb, actions_length):
        """
        Process the output of the transformer blocks.

        Args:
            hidden_states: Tensor of shape (B, seq_length, embedding_dim)
            temb: Tensor of shape (B, embedding_dim)
            actions_length: Length of the actions sequence (T)

        Returns:
            pred_velocity: Tensor of shape (B, T, action_dim)
        """
        conditioning = temb
        shift, scale = self.model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.model.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        action_features = self.model.proj_out_2(hidden_states)
        pred = self.action_decoder(action_features)
        pred_velocity = pred[:, -actions_length:]
        return pred_velocity

    def _normalize_task_tokens(
        self,
        task_tokens: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not isinstance(task_tokens, torch.Tensor):
            return None
        if task_tokens.ndim != 3:
            raise ValueError(f"`task_tokens` must be [B, Ns, D], got shape={tuple(task_tokens.shape)}")
        if task_tokens.shape[0] != batch_size:
            raise ValueError(
                f"`task_tokens` batch mismatch: got B={task_tokens.shape[0]}, expected B={batch_size}"
            )
        out = task_tokens.to(device=device, dtype=dtype)
        # Keep task token count from VLM as-is (fixed-K query design), and only
        # use config num_task_tokens as an informational target.
        if (
            out.shape[1] != self.num_task_tokens
            and self._is_main_process()
            and (not getattr(self, "_logged_task_token_num_mismatch", False))
        ):
            logger.info(
                "Task token count mismatch (allowed): VLM_K=%d, action_cfg.num_task_tokens=%d",
                int(out.shape[1]),
                int(self.num_task_tokens),
            )
            self._logged_task_token_num_mismatch = True
        return out

    def _normalize_feedback_tokens(
        self,
        feedback_tokens: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not isinstance(feedback_tokens, torch.Tensor):
            return None
        if feedback_tokens.ndim == 2:
            feedback_tokens = feedback_tokens.unsqueeze(1)
        if feedback_tokens.ndim != 3:
            raise ValueError(
                f"`feedback_tokens` must be [B, Kf, D] or [B, D], got shape={tuple(feedback_tokens.shape)}"
            )
        if feedback_tokens.shape[0] != batch_size:
            raise ValueError(
                f"`feedback_tokens` batch mismatch: got B={feedback_tokens.shape[0]}, expected B={batch_size}"
            )
        out = feedback_tokens.to(device=device, dtype=dtype)
        if out.shape[-1] != self.input_embedding_dim:
            raise ValueError(
                f"`feedback_tokens` hidden mismatch: got D={out.shape[-1]}, expected D={self.input_embedding_dim}"
            )
        return out

    def _predict_next_task_tokens(
        self,
        task_tokens: Optional[torch.Tensor],
        action_context: torch.Tensor,
        state_context: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if task_tokens is None:
            return None, None
        projected_task = self.world_task_proj(task_tokens)
        world_tokens = [self.world_action_proj(action_context).unsqueeze(1)]
        if self.world_state_proj is not None and isinstance(state_context, torch.Tensor):
            state_token = state_context
            if state_token.ndim == 3:
                state_token = state_token[:, -1, :]
            world_tokens.append(self.world_state_proj(state_token).unsqueeze(1))
        world_tokens.append(projected_task)
        world_input = torch.cat(world_tokens, dim=1)
        num_prefix = len(world_tokens) - 1
        world_out = self.world_predictor(world_input)
        task_out = world_out[:, num_prefix:, :]
        delta_task = self.world_delta_head(task_out)
        pred_next_task = task_tokens + delta_task
        return pred_next_task, delta_task

    def _compute_world_guidance(
        self,
        task_tokens: Optional[torch.Tensor],
        action_sequence: torch.Tensor,
        state_context: Optional[torch.Tensor],
        temb: torch.Tensor,
        timestep_bucket: torch.Tensor,
        observed_next_task_tokens: Optional[torch.Tensor] = None,
        valid_tk_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if task_tokens is None:
            zeros_b = action_sequence.new_zeros((action_sequence.shape[0], action_sequence.shape[1], self.action_dim))
            zeros_scalar = action_sequence.new_zeros((action_sequence.shape[0],))
            return {
                "drift": zeros_b,
                "pred_next_task_tokens": None,
                "loss_dyn": action_sequence.new_tensor(0.0),
                "loss_geo": action_sequence.new_tensor(0.0),
                "loss_reg": action_sequence.new_tensor(0.0),
                "rrr": zeros_scalar,
                "expected_change": zeros_scalar,
                "delta_z_norm": zeros_scalar,
                "geo_energy": zeros_scalar,
                "drift_action_cos": zeros_scalar,
                "drift_l2": zeros_scalar,
            }

        action_context = action_sequence[:, 0, :]
        pred_next_task, _ = self._predict_next_task_tokens(
            task_tokens=task_tokens,
            action_context=action_context,
            state_context=state_context,
        )
        if pred_next_task is None:
            zeros_b = action_sequence.new_zeros((action_sequence.shape[0], action_sequence.shape[1], self.action_dim))
            zeros_scalar = action_sequence.new_zeros((action_sequence.shape[0],))
            return {
                "drift": zeros_b,
                "pred_next_task_tokens": None,
                "loss_dyn": action_sequence.new_tensor(0.0),
                "loss_geo": action_sequence.new_tensor(0.0),
                "loss_reg": action_sequence.new_tensor(0.0),
                "rrr": zeros_scalar,
                "expected_change": zeros_scalar,
                "delta_z_norm": zeros_scalar,
                "geo_energy": zeros_scalar,
                "drift_action_cos": zeros_scalar,
                "drift_l2": zeros_scalar,
            }

        if isinstance(observed_next_task_tokens, torch.Tensor):
            target_next = observed_next_task_tokens.to(device=pred_next_task.device, dtype=pred_next_task.dtype)
            if target_next.shape[1] != pred_next_task.shape[1]:
                target_next = target_next[:, : pred_next_task.shape[1]]
        else:
            target_next = task_tokens.detach()
        mask = None
        if isinstance(valid_tk_mask, torch.Tensor):
            mask = valid_tk_mask.to(device=pred_next_task.device, dtype=pred_next_task.dtype).view(-1)
            if mask.shape[0] != pred_next_task.shape[0]:
                raise ValueError(
                    f"`valid_tk_mask` batch mismatch: got {mask.shape[0]}, expected {pred_next_task.shape[0]}"
                )
            mask = mask.clamp(0.0, 1.0)

        delta_z = target_next - pred_next_task
        pooled_residual = delta_z.mean(dim=1)
        geo_weight = F.softplus(self.geo_weight_diag).to(device=pooled_residual.device, dtype=pooled_residual.dtype)
        dyn_per_sample = (pred_next_task - target_next).pow(2).mean(dim=(1, 2))
        geo_per_sample = 0.5 * ((pooled_residual * geo_weight) * pooled_residual).sum(dim=-1)
        loss_dyn = self._masked_mean(dyn_per_sample, mask)
        loss_geo = self._masked_mean(geo_per_sample, mask)
        residual_world = self.residual_proj(pooled_residual)

        drift_input = torch.cat(
            [
                residual_world,
                self.temb_proj(temb),
            ],
            dim=-1,
        )
        drift_base = self.drift_head(drift_input).unsqueeze(1).expand(-1, action_sequence.shape[1], -1)
        t_ratio = timestep_bucket.float() / float(max(1, self.num_timestep_buckets))
        eta = (self.drift_eta * (1.0 - t_ratio)).clamp(min=0.0).view(-1, 1, 1)
        drift = eta * drift_base
        reg_per_sample = drift.pow(2).mean(dim=(1, 2))
        loss_reg = self._masked_mean(reg_per_sample, mask)

        expected_change = (pred_next_task - task_tokens).norm(dim=-1).mean(dim=-1)
        delta_z_norm = delta_z.norm(dim=-1).mean(dim=-1)
        rrr = delta_z_norm / expected_change.clamp_min(self.rrr_eps)
        drift_action = drift.mean(dim=1)
        drift_action_world = self.world_action_proj(drift_action)
        drift_action_cos = F.cosine_similarity(
            residual_world.float(),
            drift_action_world.float(),
            dim=-1,
            eps=1e-6,
        ).to(dtype=pred_next_task.dtype)
        drift_l2 = drift.pow(2).mean(dim=(1, 2)).sqrt()
        geo_energy = geo_per_sample
        if mask is not None:
            keep = mask > 0.5
            rrr = torch.where(keep, rrr, torch.zeros_like(rrr))
            expected_change = torch.where(keep, expected_change, torch.zeros_like(expected_change))
            delta_z_norm = torch.where(keep, delta_z_norm, torch.zeros_like(delta_z_norm))
            geo_energy = torch.where(keep, geo_energy, torch.zeros_like(geo_energy))
            drift_action_cos = torch.where(keep, drift_action_cos, torch.zeros_like(drift_action_cos))
            drift_l2 = torch.where(keep, drift_l2, torch.zeros_like(drift_l2))

        return {
            "drift": drift,
            "pred_next_task_tokens": pred_next_task,
            "loss_dyn": loss_dyn,
            "loss_geo": loss_geo,
            "loss_reg": loss_reg,
            "rrr": rrr,
            "expected_change": expected_change,
            "delta_z_norm": delta_z_norm,
            "geo_energy": geo_energy,
            "drift_action_cos": drift_action_cos,
            "drift_l2": drift_l2,
        }

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_tokens_next: Optional[torch.Tensor] = None,
        valid_tk: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        loss_weight_override: Optional[Dict[str, float]] = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        self._last_health_trace = []
        self._first_nonfinite_stage = None
        self._first_nonfinite_record = None
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        self._record_health("forward/actions", actions)
        self._record_health("forward/vl_embs_layer0", vl_embs_list[0])
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        self._record_health("forward/noise", noise)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        self._record_health("forward/noisy_trajectory", noisy_trajectory)
        self._record_health("forward/velocity_target", velocity)

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        t_discretized = t_discretized.clamp(0, max(self.num_timestep_buckets - 1, 0))
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        self._record_health("forward/action_features", action_features)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None
        if state_features is not None:
            self._record_health("forward/state_features", state_features)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
            self._record_health("forward/action_features_plus_pos", action_features)

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        task_tokens = self._normalize_task_tokens(
            task_tokens=task_tokens,
            batch_size=B,
            device=device,
            dtype=sa_embs.dtype,
        )
        task_tokens_next = self._normalize_task_tokens(
            task_tokens=task_tokens_next,
            batch_size=B,
            device=device,
            dtype=sa_embs.dtype,
        )
        feedback_tokens = self._normalize_feedback_tokens(
            feedback_tokens=feedback_tokens,
            batch_size=B,
            device=device,
            dtype=sa_embs.dtype,
        )
        context_task_tokens = task_tokens
        if isinstance(feedback_tokens, torch.Tensor):
            if self.feedback_context_scale != 1.0:
                feedback_tokens = feedback_tokens * float(self.feedback_context_scale)
            context_task_tokens = (
                feedback_tokens
                if context_task_tokens is None
                else torch.cat((feedback_tokens, context_task_tokens), dim=1)
            )
        valid_tk_mask = None
        if isinstance(valid_tk, torch.Tensor):
            valid_tk_mask = valid_tk.to(device=device, dtype=sa_embs.dtype).view(-1)
            if valid_tk_mask.shape[0] != B:
                raise ValueError(
                    f"`valid_tk` batch mismatch: got {valid_tk_mask.shape[0]}, expected {B}"
                )
            valid_tk_mask = valid_tk_mask.clamp(0.0, 1.0)
        self._record_health("forward/sa_embs", sa_embs)
        self._record_health("forward/task_tokens", task_tokens) if isinstance(task_tokens, torch.Tensor) else None
        self._record_health("forward/task_tokens_next", task_tokens_next) if isinstance(task_tokens_next, torch.Tensor) else None
        self._record_health("forward/feedback_tokens", feedback_tokens) if isinstance(feedback_tokens, torch.Tensor) else None
        self._record_health("forward/context_task_tokens", context_task_tokens) if isinstance(context_task_tokens, torch.Tensor) else None
        self._record_health("forward/valid_tk", valid_tk_mask) if isinstance(valid_tk_mask, torch.Tensor) else None
        temb = self.model.timestep_encoder(t_discretized)
        self._record_health("forward/temb", temb)
        hidden_states = self._apply_layerwise_cross_attention(
            sa_embs,
            vl_embs_list,
            temb,
            encoder_attention_mask=encoder_attention_mask,
            task_tokens=context_task_tokens,
            log_context="train",
        )
        self._record_health("forward/hidden_states", hidden_states)
        pred_velocity = self._process_output(hidden_states, temb, actions.shape[1])
        self._record_health("forward/pred_velocity_raw", pred_velocity)
        world_action_sequence = actions if self.world_action_source == "teacher" else noisy_trajectory
        world_guidance = self._compute_world_guidance(
            task_tokens=task_tokens,
            action_sequence=world_action_sequence,
            state_context=state,
            temb=temb,
            timestep_bucket=t_discretized,
            observed_next_task_tokens=task_tokens_next,
            valid_tk_mask=valid_tk_mask,
        )
        drift = world_guidance["drift"]
        if isinstance(drift, torch.Tensor):
            self._record_health("forward/drift", drift)
            pred_velocity = pred_velocity + drift
        self._record_health("forward/pred_velocity_corrected", pred_velocity)
        loss_cfm = ((pred_velocity - velocity) ** 2).mean()
        loss_dyn = world_guidance["loss_dyn"]
        loss_geo = world_guidance["loss_geo"]
        loss_reg = world_guidance["loss_reg"]
        loss_weights = self._resolve_loss_weights(loss_weight_override=loss_weight_override)
        total_loss = (
            loss_weights["loss_w_fm"] * loss_cfm
            + loss_weights["loss_w_dyn"] * loss_dyn
            + loss_weights["loss_w_geo"] * loss_geo
            + loss_weights["loss_w_reg"] * loss_reg
        )
        self._last_loss_breakdown = {
            "loss_cfm": float(loss_cfm.detach().item()),
            "loss_dyn": float(loss_dyn.detach().item()),
            "mse_dyn": float(loss_dyn.detach().item()),
            "loss_geo": float(loss_geo.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "loss_total": float(total_loss.detach().item()),
            "world_predictor_params": float(self.world_predictor_param_count),
            "weight_fm": float(loss_weights["loss_w_fm"]),
            "weight_dyn": float(loss_weights["loss_w_dyn"]),
            "weight_geo": float(loss_weights["loss_w_geo"]),
            "weight_reg": float(loss_weights["loss_w_reg"]),
        }
        if isinstance(valid_tk_mask, torch.Tensor):
            self._last_loss_breakdown["valid_tk_ratio"] = float((valid_tk_mask > 0.5).float().mean().item())
        if isinstance(feedback_tokens, torch.Tensor):
            self._last_loss_breakdown["feedback_token_num"] = float(feedback_tokens.shape[1])
            self._last_loss_breakdown["feedback_token_norm_mean"] = float(
                feedback_tokens.detach().norm(dim=-1).mean().item()
            )
        rrr = world_guidance.get("rrr")
        if isinstance(rrr, torch.Tensor) and rrr.numel() > 0:
            self._last_rrr = float(rrr.detach().mean().item())
            self._last_loss_breakdown["rrr_mean"] = self._last_rrr
        delta_z_norm = world_guidance.get("delta_z_norm")
        if isinstance(delta_z_norm, torch.Tensor) and delta_z_norm.numel() > 0:
            delta_z_norm_mean = self._masked_mean(delta_z_norm.view(-1), valid_tk_mask)
            self._last_loss_breakdown["delta_z_norm_mean"] = float(delta_z_norm_mean.detach().item())
            self._last_loss_breakdown["delta_z_norm_std"] = float(delta_z_norm.detach().float().std(unbiased=False).item())
        drift_action_cos = world_guidance.get("drift_action_cos")
        if isinstance(drift_action_cos, torch.Tensor) and drift_action_cos.numel() > 0:
            rho_da = self._masked_mean(drift_action_cos.view(-1), valid_tk_mask)
            self._last_loss_breakdown["rho_da"] = float(rho_da.detach().item())
            self._last_loss_breakdown["drift_action_cos_mean"] = float(rho_da.detach().item())
        geo_energy = world_guidance.get("geo_energy")
        if isinstance(geo_energy, torch.Tensor) and geo_energy.numel() > 0:
            geo_energy_mean = self._masked_mean(geo_energy.view(-1), valid_tk_mask)
            self._last_loss_breakdown["geo_energy_mean"] = float(geo_energy_mean.detach().item())
            self._last_loss_breakdown["E_geo"] = float(geo_energy_mean.detach().item())
        drift_l2 = world_guidance.get("drift_l2")
        if isinstance(drift_l2, torch.Tensor) and drift_l2.numel() > 0:
            drift_l2_mean = self._masked_mean(drift_l2.view(-1), valid_tk_mask)
            self._last_loss_breakdown["drift_l2_mean"] = float(drift_l2_mean.detach().item())
        self._record_health("forward/loss_total", total_loss.unsqueeze(0))
        return total_loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        noise_seed: int = None,
        encoder_attention_mask: torch.Tensor = None,
        task_tokens: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        force_replan: bool = False,
        rrr_threshold: Optional[float] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        self._last_health_trace = []
        self._first_nonfinite_stage = None
        self._first_nonfinite_record = None
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        use_cuda_timing = bool(device.type == "cuda" and torch.cuda.is_available())
        flow_start_event = flow_end_event = None
        flow_start_cpu = None
        per_step_events = []
        per_step_cpu_ms = []
        reflex_events = []
        reflex_cpu_ms = []
        if use_cuda_timing:
            flow_start_event = torch.cuda.Event(enable_timing=True)
            flow_end_event = torch.cuda.Event(enable_timing=True)
            flow_start_event.record()
        else:
            flow_start_cpu = time.perf_counter()
        generator = None
        if noise_seed is not None:
            try:
                gen_device = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
                generator = torch.Generator(device=gen_device)
            except Exception:
                generator = torch.Generator()
            generator.manual_seed(int(noise_seed))

        if generator is None:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs_list[0].dtype,
                device=device,
            )
        else:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs_list[0].dtype,
                device=device,
                generator=generator,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        threshold = float(self.rrr_threshold if rrr_threshold is None else rrr_threshold)
        task_tokens = self._normalize_task_tokens(
            task_tokens=task_tokens,
            batch_size=batch_size,
            device=device,
            dtype=vl_embs_list[0].dtype,
        )
        feedback_tokens = self._normalize_feedback_tokens(
            feedback_tokens=feedback_tokens,
            batch_size=batch_size,
            device=device,
            dtype=vl_embs_list[0].dtype,
        )
        context_task_tokens = task_tokens
        if isinstance(feedback_tokens, torch.Tensor):
            if self.feedback_context_scale != 1.0:
                feedback_tokens = feedback_tokens * float(self.feedback_context_scale)
            context_task_tokens = (
                feedback_tokens
                if context_task_tokens is None
                else torch.cat((feedback_tokens, context_task_tokens), dim=1)
            )
        replans = 0
        t = 0
        predicted_task_tokens = None
        expected_change = None
        rrr_trace = []
        delta_z_norm_trace = []
        geo_energy_trace = []
        drift_action_cos_trace = []
        while t < num_steps:
            if use_cuda_timing:
                step_start_event = torch.cuda.Event(enable_timing=True)
                step_end_event = torch.cuda.Event(enable_timing=True)
                step_start_event.record()
            else:
                step_start_cpu = time.perf_counter()

            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            temb = self.model.timestep_encoder(timesteps_tensor)
            hidden_states = self._apply_layerwise_cross_attention(
                sa_embs,
                vl_embs_list,
                temb,
                encoder_attention_mask=encoder_attention_mask,
                task_tokens=context_task_tokens,
                log_context="infer",
            )
            pred_velocity = self._process_output(hidden_states, temb, self.action_horizon)
            if use_cuda_timing:
                reflex_start_event = torch.cuda.Event(enable_timing=True)
                reflex_end_event = torch.cuda.Event(enable_timing=True)
                reflex_start_event.record()
            else:
                reflex_start_cpu = time.perf_counter()
            world_guidance = self._compute_world_guidance(
                task_tokens=task_tokens,
                action_sequence=actions,
                state_context=state,
                temb=temb,
                timestep_bucket=timesteps_tensor,
                observed_next_task_tokens=task_tokens,
            )
            drift = world_guidance["drift"]
            if isinstance(drift, torch.Tensor):
                pred_velocity = pred_velocity + drift
            if use_cuda_timing:
                reflex_end_event.record()
                reflex_events.append((reflex_start_event, reflex_end_event))
            else:
                reflex_cpu_ms.append(float((time.perf_counter() - reflex_start_cpu) * 1000.0))
            delta_z_norm = world_guidance.get("delta_z_norm")
            if isinstance(delta_z_norm, torch.Tensor) and delta_z_norm.numel() > 0:
                delta_z_norm_trace.append(float(delta_z_norm.detach().mean().item()))
            geo_energy = world_guidance.get("geo_energy")
            if isinstance(geo_energy, torch.Tensor) and geo_energy.numel() > 0:
                geo_energy_trace.append(float(geo_energy.detach().mean().item()))
            drift_action_cos = world_guidance.get("drift_action_cos")
            if isinstance(drift_action_cos, torch.Tensor) and drift_action_cos.numel() > 0:
                drift_action_cos_trace.append(float(drift_action_cos.detach().mean().item()))
            rrr = world_guidance.get("rrr")
            if isinstance(rrr, torch.Tensor) and rrr.numel() > 0:
                rrr_mean = float(rrr.detach().mean().item())
                rrr_trace.append(rrr_mean)
                self._last_rrr = rrr_mean
                if (force_replan or rrr_mean > threshold) and replans < self.rrr_max_replans:
                    replans += 1
                    if generator is None:
                        actions = torch.randn_like(actions)
                    else:
                        actions = torch.randn(
                            size=actions.shape,
                            dtype=actions.dtype,
                            device=actions.device,
                            generator=generator,
                        )
                    t = 0
                    force_replan = False
                    if use_cuda_timing:
                        step_end_event.record()
                        per_step_events.append((step_start_event, step_end_event))
                    else:
                        per_step_cpu_ms.append(float((time.perf_counter() - step_start_cpu) * 1000.0))
                    continue
            predicted_task_tokens = world_guidance.get("pred_next_task_tokens")
            expected_change = world_guidance.get("expected_change")
            actions = actions + dt * pred_velocity
            t += 1
            if use_cuda_timing:
                step_end_event.record()
                per_step_events.append((step_start_event, step_end_event))
            else:
                per_step_cpu_ms.append(float((time.perf_counter() - step_start_cpu) * 1000.0))

        if use_cuda_timing:
            flow_end_event.record()
            torch.cuda.synchronize(device=device)
            flow_total_ms = float(flow_start_event.elapsed_time(flow_end_event))
            step_ms_values = [float(s.elapsed_time(e)) for s, e in per_step_events]
            reflex_ms_values = [float(s.elapsed_time(e)) for s, e in reflex_events]
        else:
            flow_total_ms = float((time.perf_counter() - flow_start_cpu) * 1000.0)
            step_ms_values = per_step_cpu_ms
            reflex_ms_values = reflex_cpu_ms

        if len(step_ms_values) > 0:
            flow_step_ms_mean = float(sum(step_ms_values) / len(step_ms_values))
            flow_step_ms_max = float(max(step_ms_values))
        else:
            flow_step_ms_mean = 0.0
            flow_step_ms_max = 0.0

        reflex_total_ms = float(sum(reflex_ms_values)) if len(reflex_ms_values) > 0 else 0.0
        reflex_step_ms_mean = float(reflex_total_ms / len(reflex_ms_values)) if len(reflex_ms_values) > 0 else 0.0
        reflex_step_ms_max = float(max(reflex_ms_values)) if len(reflex_ms_values) > 0 else 0.0
        reflex_share = float(reflex_total_ms / max(flow_total_ms, 1e-6))
        drift_response_lag_ms = float(reflex_ms_values[0]) if len(reflex_ms_values) > 0 else 0.0
        effective_control_hz = float(1000.0 * len(reflex_ms_values) / max(flow_total_ms, 1e-6))

        if not return_info:
            return actions
        info = {
            "rrr_last": float(rrr_trace[-1]) if len(rrr_trace) > 0 else None,
            "rrr_max": float(max(rrr_trace)) if len(rrr_trace) > 0 else None,
            "rrr_trace": rrr_trace,
            "replan_count": int(replans),
            "predicted_task_tokens": predicted_task_tokens.detach() if isinstance(predicted_task_tokens, torch.Tensor) else None,
            "expected_change": expected_change.detach() if isinstance(expected_change, torch.Tensor) else None,
            "timing/flow_total_ms": flow_total_ms,
            "timing/flow_step_ms_mean": flow_step_ms_mean,
            "timing/flow_step_ms_max": flow_step_ms_max,
            "timing/flow_step_count": int(len(step_ms_values)),
            "timing/reflex_total_ms": reflex_total_ms,
            "timing/reflex_step_ms_mean": reflex_step_ms_mean,
            "timing/reflex_step_ms_max": reflex_step_ms_max,
            "timing/reflex_step_count": int(len(reflex_ms_values)),
            "timing/reflex_share": reflex_share,
            "timing/drift_response_lag_ms": drift_response_lag_ms,
            "timing/effective_control_hz": effective_control_hz,
            "geo/delta_z_norm_last": float(delta_z_norm_trace[-1]) if len(delta_z_norm_trace) > 0 else None,
            "geo/delta_z_norm_mean": float(sum(delta_z_norm_trace) / len(delta_z_norm_trace)) if len(delta_z_norm_trace) > 0 else None,
            "geo/energy_last": float(geo_energy_trace[-1]) if len(geo_energy_trace) > 0 else None,
            "geo/energy_mean": float(sum(geo_energy_trace) / len(geo_energy_trace)) if len(geo_energy_trace) > 0 else None,
            "geo/drift_action_cos_last": float(drift_action_cos_trace[-1]) if len(drift_action_cos_trace) > 0 else None,
            "geo/drift_action_cos_mean": float(sum(drift_action_cos_trace) / len(drift_action_cos_trace)) if len(drift_action_cos_trace) > 0 else None,
            "geo/delta_z_norm_trace": delta_z_norm_trace,
            "geo/energy_trace": geo_energy_trace,
            "geo/drift_action_cos_trace": drift_action_cos_trace,
            "feedback/token_num": int(feedback_tokens.shape[1]) if isinstance(feedback_tokens, torch.Tensor) else 0,
            "feedback/token_norm_mean": (
                float(feedback_tokens.detach().norm(dim=-1).mean().item())
                if isinstance(feedback_tokens, torch.Tensor)
                else 0.0
            ),
        }
        return actions, info

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(
        global_config=config
    )



if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
