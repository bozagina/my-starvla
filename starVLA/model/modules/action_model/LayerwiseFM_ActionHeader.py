# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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
    feedback_context_only: bool = field(
        default=False,
        metadata={"help": "If true, use only feedback tokens as task-token context (drop base task_tokens from context)."},
    )
    feedback_context_norm_type: str = field(
        default="none",
        metadata={"help": "Optional feedback token normalization before cross-attn context. One of: none|layernorm|rmsnorm."},
    )
    feedback_context_norm_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for feedback token normalization."},
    )
    feedback_context_alpha_mode: str = field(
        default="fixed",
        metadata={"help": "Feedback context scaling alpha mode. One of: fixed|learnable|schedule."},
    )
    feedback_context_alpha_init: float = field(
        default=1.0,
        metadata={"help": "Initial alpha for feedback token scaling."},
    )
    feedback_context_alpha_target: float = field(
        default=1.0,
        metadata={"help": "Target alpha for schedule mode."},
    )
    feedback_context_alpha_warmup_steps: int = field(
        default=0,
        metadata={"help": "Hold alpha at init for this many train forwards before schedule ramp."},
    )
    feedback_context_alpha_ramp_steps: int = field(
        default=1000,
        metadata={"help": "Linear ramp steps from alpha_init to alpha_target for schedule mode."},
    )
    feedback_probe_interval: int = field(
        default=0,
        metadata={"help": "If >0, run low-frequency no-feedback probe every N train forwards."},
    )
    feedback_probe_start_step: int = field(
        default=0,
        metadata={"help": "Do not run feedback probe before this train-forward step."},
    )
    feedback_in_context_enabled: bool = field(
        default=True,
        metadata={"help": "If false, do not inject feedback tokens into cross-attention context (keep task_tokens only)."},
    )
    feedback_delta_action_enabled: bool = field(
        default=False,
        metadata={"help": "Enable residual delta-action head from feedback tokens."},
    )
    feedback_delta_action_enable_inference: bool = field(
        default=True,
        metadata={"help": "Apply residual delta-action head during inference."},
    )
    feedback_delta_action_norm_type: str = field(
        default="layernorm",
        metadata={"help": "Normalization for pooled feedback vector: none|layernorm|rmsnorm."},
    )
    feedback_delta_action_norm_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for feedback delta-action normalization."},
    )
    feedback_delta_action_hidden_dim: int = field(
        default=0,
        metadata={"help": "Hidden dim for feedback delta-action MLP. <=0 means use input embedding dim."},
    )
    feedback_delta_action_last_layer_scale: float = field(
        default=1e-3,
        metadata={"help": "Scale factor for last linear layer init in delta-action head."},
    )
    feedback_delta_action_clip: float = field(
        default=0.05,
        metadata={"help": "Per-dimension tanh clip magnitude for delta-action."},
    )
    feedback_delta_action_use_valid_tk_gate: bool = field(
        default=True,
        metadata={"help": "Multiply delta-action gate by valid_tk mask when available."},
    )
    feedback_delta_action_alpha_mode: str = field(
        default="schedule",
        metadata={"help": "Delta-action alpha mode: fixed|learnable|schedule."},
    )
    feedback_delta_action_alpha_init: float = field(
        default=0.0,
        metadata={"help": "Initial alpha for delta-action gate."},
    )
    feedback_delta_action_alpha_target: float = field(
        default=0.1,
        metadata={"help": "Target alpha for delta-action schedule."},
    )
    feedback_delta_action_alpha_warmup_steps: int = field(
        default=2000,
        metadata={"help": "Warmup steps holding alpha_init before ramp for delta-action schedule."},
    )
    feedback_delta_action_alpha_ramp_steps: int = field(
        default=2000,
        metadata={"help": "Ramp steps from alpha_init to alpha_target for delta-action schedule."},
    )
    patha_residual_mode: str = field(
        default="pooled_delta_z",
        metadata={"help": "Path-A residual source: pooled_delta_z or token_delta_geo."},
    )
    soft_mask_enabled: bool = field(
        default=False,
        metadata={"help": "Enable v4-1-1 soft mask for token-level residual."},
    )
    soft_mask_lambda: float = field(
        default=0.3,
        metadata={"help": "Fusion weight between visual and geometric soft-mask channels."},
    )
    soft_mask_ema_beta: float = field(
        default=0.2,
        metadata={"help": "EMA beta for soft-mask smoothing."},
    )
    soft_mask_query_agg: str = field(
        default="mean",
        metadata={"help": "Soft-mask query aggregation: mean or max."},
    )
    soft_mask_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for soft-mask attention logits."},
    )
    soft_mask_use_ema_inference: bool = field(
        default=True,
        metadata={"help": "Apply soft-mask EMA smoothing during inference."},
    )
    soft_mask_use_ema_training: bool = field(
        default=False,
        metadata={"help": "Apply soft-mask EMA smoothing during training."},
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
        self.feedback_context_only = bool(_cfg_get(action_config, "feedback_context_only", False))
        self.feedback_context_norm_type = str(
            _cfg_get(action_config, "feedback_context_norm_type", "none")
        ).lower()
        self.feedback_context_norm_eps = float(_cfg_get(action_config, "feedback_context_norm_eps", 1e-6))
        self.feedback_context_norm = None
        self._feedback_context_manual_rmsnorm = False
        if self.feedback_context_norm_type not in ("none", "layernorm", "rmsnorm"):
            logger.warning(
                "Invalid feedback_context_norm_type=%s, fallback to `none`.",
                self.feedback_context_norm_type,
            )
            self.feedback_context_norm_type = "none"
        if self.feedback_context_norm_type == "layernorm":
            self.feedback_context_norm = nn.LayerNorm(
                self.input_embedding_dim,
                eps=self.feedback_context_norm_eps,
                elementwise_affine=False,
            )
        elif self.feedback_context_norm_type == "rmsnorm":
            if hasattr(nn, "RMSNorm"):
                self.feedback_context_norm = nn.RMSNorm(
                    self.input_embedding_dim,
                    eps=self.feedback_context_norm_eps,
                    elementwise_affine=False,
                )
            else:
                self._feedback_context_manual_rmsnorm = True
        self.feedback_context_alpha_mode = str(
            _cfg_get(action_config, "feedback_context_alpha_mode", "fixed")
        ).lower()
        self.feedback_context_alpha_init = float(_cfg_get(action_config, "feedback_context_alpha_init", 1.0))
        self.feedback_context_alpha_target = float(_cfg_get(action_config, "feedback_context_alpha_target", 1.0))
        self.feedback_context_alpha_warmup_steps = int(
            _cfg_get(action_config, "feedback_context_alpha_warmup_steps", 0)
        )
        self.feedback_context_alpha_ramp_steps = int(
            _cfg_get(action_config, "feedback_context_alpha_ramp_steps", 1000)
        )
        if self.feedback_context_alpha_warmup_steps < 0:
            self.feedback_context_alpha_warmup_steps = 0
        if self.feedback_context_alpha_ramp_steps < 1:
            self.feedback_context_alpha_ramp_steps = 1
        if self.feedback_context_alpha_mode not in ("fixed", "learnable", "schedule"):
            logger.warning(
                "Invalid feedback_context_alpha_mode=%s, fallback to `fixed`.",
                self.feedback_context_alpha_mode,
            )
            self.feedback_context_alpha_mode = "fixed"
        if self.feedback_context_alpha_mode == "learnable":
            self.feedback_context_alpha_param = nn.Parameter(
                torch.tensor(self.feedback_context_alpha_init, dtype=torch.float32)
            )
        else:
            self.feedback_context_alpha_param = None
        self.register_buffer(
            "_feedback_context_schedule_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._last_feedback_alpha_value = float(self.feedback_context_alpha_init)
        self.feedback_probe_interval = int(
            max(0, int(_cfg_get(action_config, "feedback_probe_interval", 0)))
        )
        self.feedback_probe_start_step = int(
            max(0, int(_cfg_get(action_config, "feedback_probe_start_step", 0)))
        )
        self._feedback_probe_step = 0
        self.feedback_in_context_enabled = bool(
            _cfg_get(action_config, "feedback_in_context_enabled", True)
        )

        self.feedback_delta_action_enabled = bool(
            _cfg_get(action_config, "feedback_delta_action_enabled", False)
        )
        self.feedback_delta_action_enable_inference = bool(
            _cfg_get(action_config, "feedback_delta_action_enable_inference", True)
        )
        self.feedback_delta_action_norm_type = str(
            _cfg_get(action_config, "feedback_delta_action_norm_type", "layernorm")
        ).lower()
        if self.feedback_delta_action_norm_type not in ("none", "layernorm", "rmsnorm"):
            logger.warning(
                "Invalid feedback_delta_action_norm_type=%s, fallback to `layernorm`.",
                self.feedback_delta_action_norm_type,
            )
            self.feedback_delta_action_norm_type = "layernorm"
        self.feedback_delta_action_norm_eps = float(
            _cfg_get(action_config, "feedback_delta_action_norm_eps", 1e-6)
        )
        hidden_dim_cfg = int(_cfg_get(action_config, "feedback_delta_action_hidden_dim", 0))
        self.feedback_delta_action_hidden_dim = (
            hidden_dim_cfg if hidden_dim_cfg > 0 else int(self.input_embedding_dim)
        )
        self.feedback_delta_action_last_layer_scale = float(
            _cfg_get(action_config, "feedback_delta_action_last_layer_scale", 1e-3)
        )
        self.feedback_delta_action_clip = float(
            _cfg_get(action_config, "feedback_delta_action_clip", 0.05)
        )
        self.feedback_delta_action_use_valid_tk_gate = bool(
            _cfg_get(action_config, "feedback_delta_action_use_valid_tk_gate", True)
        )
        self.feedback_delta_action_alpha_mode = str(
            _cfg_get(action_config, "feedback_delta_action_alpha_mode", "schedule")
        ).lower()
        if self.feedback_delta_action_alpha_mode not in ("fixed", "learnable", "schedule"):
            logger.warning(
                "Invalid feedback_delta_action_alpha_mode=%s, fallback to `schedule`.",
                self.feedback_delta_action_alpha_mode,
            )
            self.feedback_delta_action_alpha_mode = "schedule"
        self.feedback_delta_action_alpha_init = float(
            _cfg_get(action_config, "feedback_delta_action_alpha_init", 0.0)
        )
        self.feedback_delta_action_alpha_target = float(
            _cfg_get(action_config, "feedback_delta_action_alpha_target", 0.1)
        )
        self.feedback_delta_action_alpha_warmup_steps = int(
            _cfg_get(action_config, "feedback_delta_action_alpha_warmup_steps", 2000)
        )
        self.feedback_delta_action_alpha_ramp_steps = int(
            _cfg_get(action_config, "feedback_delta_action_alpha_ramp_steps", 2000)
        )
        if self.feedback_delta_action_alpha_warmup_steps < 0:
            self.feedback_delta_action_alpha_warmup_steps = 0
        if self.feedback_delta_action_alpha_ramp_steps < 1:
            self.feedback_delta_action_alpha_ramp_steps = 1
        if self.feedback_delta_action_alpha_mode == "learnable":
            self.feedback_delta_action_alpha_param = nn.Parameter(
                torch.tensor(self.feedback_delta_action_alpha_init, dtype=torch.float32)
            )
        else:
            self.feedback_delta_action_alpha_param = None
        self.register_buffer(
            "_feedback_delta_action_schedule_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._last_feedback_delta_action_alpha_value = float(
            self.feedback_delta_action_alpha_init
        )

        self.feedback_delta_action_norm = None
        self._feedback_delta_action_manual_rmsnorm = False
        if self.feedback_delta_action_enabled:
            if self.feedback_delta_action_norm_type == "layernorm":
                self.feedback_delta_action_norm = nn.LayerNorm(
                    self.input_embedding_dim,
                    eps=self.feedback_delta_action_norm_eps,
                    elementwise_affine=False,
                )
            elif self.feedback_delta_action_norm_type == "rmsnorm":
                if hasattr(nn, "RMSNorm"):
                    self.feedback_delta_action_norm = nn.RMSNorm(
                        self.input_embedding_dim,
                        eps=self.feedback_delta_action_norm_eps,
                        elementwise_affine=False,
                    )
                else:
                    self._feedback_delta_action_manual_rmsnorm = True

        self.feedback_delta_action_head = None
        if self.feedback_delta_action_enabled:
            self.feedback_delta_action_head = nn.Sequential(
                nn.Linear(self.input_embedding_dim, self.feedback_delta_action_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.feedback_delta_action_hidden_dim, self.action_dim),
            )
            # Keep initial residual near zero so base policy behavior is preserved.
            with torch.no_grad():
                last_linear = self.feedback_delta_action_head[-1]
                if isinstance(last_linear, nn.Linear):
                    nn.init.xavier_uniform_(last_linear.weight)
                    last_linear.weight.mul_(self.feedback_delta_action_last_layer_scale)
                    nn.init.zeros_(last_linear.bias)

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
        # Lightweight directional-alignment term:
        #   loss_align = 1 - cos(residual_world, drift_action_world)
        # This term is merged into loss_geo using `geo_align_weight`.
        self.geo_align_weight = float(_cfg_get(action_config, "geo_align_weight", 0.2))
        self.align_cos_eps = float(_cfg_get(action_config, "align_cos_eps", 1e-6))
        # Health check tolerance to avoid float-ratio false positives near 1.0.
        self.health_finite_ratio_tol = float(_cfg_get(action_config, "health_finite_ratio_tol", 1e-6))
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
        self.world_action_context_mode = str(
            _cfg_get(action_config, "world_action_context_mode", "prefix_mean")
        ).lower()
        self.world_action_prefix_len = int(
            _cfg_get(action_config, "world_action_prefix_len", 4)
        )
        if self.world_action_prefix_len < 1:
            self.world_action_prefix_len = 1
        if self.world_action_context_mode not in ("first", "mean", "prefix_mean", "mlp"):
            logger.warning(
                "Invalid world_action_context_mode=%s, fallback to `prefix_mean`.",
                self.world_action_context_mode,
            )
            self.world_action_context_mode = "prefix_mean"

        self.residual_pooling_mode = str(
            _cfg_get(action_config, "residual_pooling_mode", "slot_weighted")
        ).lower()
        self.residual_slot_temp = float(_cfg_get(action_config, "residual_slot_temp", 1.0))
        if self.residual_slot_temp <= 0.0:
            self.residual_slot_temp = 1.0
        if self.residual_pooling_mode not in ("mean", "slot_weighted"):
            logger.warning(
                "Invalid residual_pooling_mode=%s, fallback to `slot_weighted`.",
                self.residual_pooling_mode,
            )
            self.residual_pooling_mode = "slot_weighted"
        self.residual_slot_logits = nn.Parameter(torch.zeros(self.num_task_tokens))
        self.residual_slot_entropy_weight = float(
            _cfg_get(action_config, "residual_slot_entropy_weight", 0.0)
        )

        self.world_action_context_mlp = None
        if self.world_action_context_mode == "mlp":
            self.world_action_context_mlp = nn.Sequential(
                nn.LayerNorm(self.action_dim),
                nn.Linear(self.action_dim, self.action_dim),
                nn.GELU(),
                nn.Linear(self.action_dim, self.action_dim),
            )
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
            "LayerwiseFMActionHead initialized: use_concat_cross_context=%s, cross_attention_assert_inputs=%s, "
            "cross_attention_debug_log_interval=%d, world_params=%d, world_action_source=%s, "
            "world_action_context_mode=%s, residual_pooling_mode=%s, feedback_context_only=%s, "
            "feedback_context_norm_type=%s, feedback_context_alpha_mode=%s, "
            "feedback_in_context_enabled=%s, feedback_delta_action_enabled=%s, "
            "feedback_delta_action_alpha_mode=%s",
            self.use_concat_cross_context,
            self.cross_attention_assert_inputs,
            self.cross_attention_debug_log_interval,
            self.world_predictor_param_count,
            self.world_action_source,
            self.world_action_context_mode,
            self.residual_pooling_mode,
            self.feedback_context_only,
            self.feedback_context_norm_type,
            self.feedback_context_alpha_mode,
            self.feedback_in_context_enabled,
            self.feedback_delta_action_enabled,
            self.feedback_delta_action_alpha_mode,
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
        nan_count = float(record.get("nan_count", 0.0))
        inf_count = float(record.get("inf_count", 0.0))
        finite_ratio = float(record.get("finite_ratio", 1.0))
        has_nan_inf = (nan_count > 0.0) or (inf_count > 0.0)
        has_low_finite = finite_ratio < (1.0 - self.health_finite_ratio_tol)
        if (
            self._first_nonfinite_stage is None
            and (has_nan_inf or has_low_finite)
        ):
            self._first_nonfinite_stage = str(stage)
            self._first_nonfinite_record = {
                "stage": str(stage),
                "finite_ratio": finite_ratio,
                "nan_count": nan_count,
                "inf_count": inf_count,
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

    @staticmethod
    def _masked_percentiles(
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
        quantiles: Tuple[float, ...] = (0.5, 0.95),
    ) -> List[Optional[float]]:
        if values.ndim != 1:
            values = values.view(-1)
        v = values.detach().float()
        if mask is not None:
            m = mask.view(-1).to(device=v.device, dtype=torch.bool)
            if m.shape[0] != v.shape[0]:
                raise ValueError(
                    f"Mask batch mismatch for percentiles: values.shape[0]={v.shape[0]}, mask.shape[0]={m.shape[0]}"
                )
            v = v[m]
        if v.numel() == 0:
            return [None for _ in quantiles]
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return [None for _ in quantiles]
        out = []
        for q in quantiles:
            q = float(min(max(q, 0.0), 1.0))
            out.append(float(torch.quantile(v, q).item()))
        return out

    def _compute_slot_weights(
        self,
        *,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_slots < 1:
            raise ValueError(f"`num_slots` must be >=1, got {num_slots}")
        logits = self.residual_slot_logits
        if logits.numel() < num_slots:
            pad = logits.new_zeros((num_slots - logits.numel(),))
            logits = torch.cat((logits, pad), dim=0)
        else:
            logits = logits[:num_slots]
        weights = torch.softmax(logits / max(self.residual_slot_temp, 1e-6), dim=0)
        return weights.to(device=device, dtype=dtype)

    def pool_task_tokens(
        self,
        task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(task_tokens, torch.Tensor) or task_tokens.ndim != 3:
            raise ValueError(
                f"`task_tokens` must be tensor [B,K,H], got type={type(task_tokens)} "
                f"shape={None if not isinstance(task_tokens, torch.Tensor) else tuple(task_tokens.shape)}"
            )
        if task_tokens.shape[1] == 1:
            return task_tokens[:, 0, :]
        if self.residual_pooling_mode == "slot_weighted":
            weights = self._compute_slot_weights(
                num_slots=int(task_tokens.shape[1]),
                device=task_tokens.device,
                dtype=task_tokens.dtype,
            )
            return torch.sum(task_tokens * weights.view(1, -1, 1), dim=1)
        return task_tokens.mean(dim=1)

    def build_world_action_context(
        self,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(action_sequence, torch.Tensor) or action_sequence.ndim != 3:
            raise ValueError(
                f"`action_sequence` must be tensor [B,T,A], got type={type(action_sequence)} "
                f"shape={None if not isinstance(action_sequence, torch.Tensor) else tuple(action_sequence.shape)}"
            )
        if action_sequence.shape[1] < 1:
            raise ValueError(f"`action_sequence` has invalid horizon={action_sequence.shape[1]}")
        mode = self.world_action_context_mode
        if mode == "first":
            context = action_sequence[:, 0, :]
        elif mode == "prefix_mean":
            prefix_len = min(int(action_sequence.shape[1]), int(self.world_action_prefix_len))
            context = action_sequence[:, :prefix_len, :].mean(dim=1)
        else:
            context = action_sequence.mean(dim=1)
        if mode == "mlp" and self.world_action_context_mlp is not None:
            context = self.world_action_context_mlp(context)
        return context

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

    def _apply_feedback_context_norm(
        self,
        feedback_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.feedback_context_norm_type == "none":
            return feedback_tokens
        if self.feedback_context_norm is not None:
            return self.feedback_context_norm(feedback_tokens)
        if self._feedback_context_manual_rmsnorm:
            denom = feedback_tokens.pow(2).mean(dim=-1, keepdim=True).add(self.feedback_context_norm_eps).rsqrt()
            return feedback_tokens * denom
        return feedback_tokens

    def _compute_feedback_alpha(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, float]:
        mode = self.feedback_context_alpha_mode
        if mode == "learnable" and isinstance(self.feedback_context_alpha_param, torch.Tensor):
            alpha_tensor = self.feedback_context_alpha_param
            if alpha_tensor.device != device or alpha_tensor.dtype != dtype:
                alpha_tensor = alpha_tensor.to(device=device, dtype=dtype)
            alpha_value = float(self.feedback_context_alpha_param.detach().float().cpu().item())
            self._last_feedback_alpha_value = alpha_value
            return alpha_tensor, alpha_value
        if mode == "schedule":
            step = int(self._feedback_context_schedule_step.item())
            warmup = int(max(self.feedback_context_alpha_warmup_steps, 0))
            ramp = int(max(self.feedback_context_alpha_ramp_steps, 1))
            alpha_init = float(self.feedback_context_alpha_init)
            alpha_target = float(self.feedback_context_alpha_target)
            if step <= warmup:
                alpha_value = alpha_init
            else:
                progress = min(max((step - warmup) / float(ramp), 0.0), 1.0)
                alpha_value = alpha_init + (alpha_target - alpha_init) * progress
            self._last_feedback_alpha_value = alpha_value
            return torch.tensor(alpha_value, device=device, dtype=dtype), alpha_value
        alpha_value = float(self.feedback_context_alpha_init)
        self._last_feedback_alpha_value = alpha_value
        return torch.tensor(alpha_value, device=device, dtype=dtype), alpha_value

    def _maybe_advance_feedback_alpha_schedule(self):
        if self.training and self.feedback_context_alpha_mode == "schedule":
            self._feedback_context_schedule_step.add_(1)

    def _maybe_advance_feedback_delta_action_alpha_schedule(self):
        if self.training and self.feedback_delta_action_alpha_mode == "schedule":
            self._feedback_delta_action_schedule_step.add_(1)

    def _apply_feedback_delta_action_norm(
        self,
        feedback_vec: torch.Tensor,
    ) -> torch.Tensor:
        if self.feedback_delta_action_norm_type == "none":
            return feedback_vec
        if self.feedback_delta_action_norm is not None:
            return self.feedback_delta_action_norm(feedback_vec)
        if self._feedback_delta_action_manual_rmsnorm:
            denom = feedback_vec.pow(2).mean(dim=-1, keepdim=True).add(
                self.feedback_delta_action_norm_eps
            ).rsqrt()
            return feedback_vec * denom
        return feedback_vec

    def _compute_feedback_delta_action_alpha(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, float]:
        mode = self.feedback_delta_action_alpha_mode
        if mode == "learnable" and isinstance(self.feedback_delta_action_alpha_param, torch.Tensor):
            alpha_tensor = self.feedback_delta_action_alpha_param
            if alpha_tensor.device != device or alpha_tensor.dtype != dtype:
                alpha_tensor = alpha_tensor.to(device=device, dtype=dtype)
            alpha_value = float(self.feedback_delta_action_alpha_param.detach().float().cpu().item())
            self._last_feedback_delta_action_alpha_value = alpha_value
            return alpha_tensor, alpha_value
        if mode == "schedule":
            step = int(self._feedback_delta_action_schedule_step.item())
            warmup = int(max(self.feedback_delta_action_alpha_warmup_steps, 0))
            ramp = int(max(self.feedback_delta_action_alpha_ramp_steps, 1))
            alpha_init = float(self.feedback_delta_action_alpha_init)
            alpha_target = float(self.feedback_delta_action_alpha_target)
            if step <= warmup:
                alpha_value = alpha_init
            else:
                progress = min(max((step - warmup) / float(ramp), 0.0), 1.0)
                alpha_value = alpha_init + (alpha_target - alpha_init) * progress
            self._last_feedback_delta_action_alpha_value = alpha_value
            return torch.tensor(alpha_value, device=device, dtype=dtype), alpha_value
        alpha_value = float(self.feedback_delta_action_alpha_init)
        self._last_feedback_delta_action_alpha_value = alpha_value
        return torch.tensor(alpha_value, device=device, dtype=dtype), alpha_value

    def _apply_feedback_delta_action(
        self,
        *,
        pred_velocity_base: torch.Tensor,
        feedback_tokens: Optional[torch.Tensor],
        valid_tk_mask: Optional[torch.Tensor],
        enable_delta_action: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        """
        Apply post-hoc residual correction:
            a_out = a_base + g * clip(DeltaA(mean(feedback_tokens)))
        """
        metrics: Dict[str, float] = {
            "delta_action_enabled": 1.0 if self.feedback_delta_action_enabled else 0.0,
            "delta_action_applied": 0.0,
            # Clip diagnostics (minimal): pre-clip norm and saturation ratio.
            "delta_action_raw_norm_mean": 0.0,
            "delta_action_postclip_norm_mean": 0.0,
            "delta_action_post_over_raw_norm_mean": 0.0,
            "delta_action_clip_saturation_frac": 0.0,
            # Delta-head learning status (minimal): last linear layer weight RMS.
            "delta_action_last_layer_weight_rms": 0.0,
        }
        if isinstance(self.feedback_delta_action_head, nn.Sequential) and len(self.feedback_delta_action_head) > 0:
            last_linear = self.feedback_delta_action_head[-1]
            if isinstance(last_linear, nn.Linear):
                with torch.no_grad():
                    weight = last_linear.weight.detach().float()
                    metrics["delta_action_last_layer_weight_rms"] = float(
                        weight.pow(2).mean().sqrt().item()
                    )
        if (
            (not self.feedback_delta_action_enabled)
            or (not enable_delta_action)
            or self.feedback_delta_action_head is None
            or (not isinstance(feedback_tokens, torch.Tensor))
        ):
            metrics["delta_action_gate_mean"] = 0.0
            metrics["delta_action_alpha"] = float(self._last_feedback_delta_action_alpha_value)
            metrics["delta_action_norm_mean"] = 0.0
            metrics["delta_action_norm_p95"] = 0.0
            metrics["delta_action_effective_norm_mean"] = 0.0
            return pred_velocity_base, metrics, None

        if feedback_tokens.ndim != 3:
            metrics["delta_action_invalid_feedback_shape"] = 1.0
            metrics["delta_action_gate_mean"] = 0.0
            metrics["delta_action_alpha"] = float(self._last_feedback_delta_action_alpha_value)
            return pred_velocity_base, metrics, None

        # fb_vec: [B, H]
        feedback_vec = feedback_tokens.mean(dim=1)
        feedback_vec = self._apply_feedback_delta_action_norm(feedback_vec)
        raw_delta_action = self.feedback_delta_action_head(feedback_vec)  # [B, A]
        raw_delta_norm = raw_delta_action.detach().norm(dim=-1)

        clip_value = float(max(self.feedback_delta_action_clip, 0.0))
        if clip_value > 0.0:
            clipped_delta_action = clip_value * torch.tanh(
                raw_delta_action / max(clip_value, 1e-8)
            )
            clip_saturation_frac = float(
                (raw_delta_action.detach().abs() > clip_value).float().mean().item()
            )
        else:
            clipped_delta_action = raw_delta_action
            clip_saturation_frac = 0.0

        alpha_tensor, alpha_value = self._compute_feedback_delta_action_alpha(
            device=pred_velocity_base.device,
            dtype=pred_velocity_base.dtype,
        )
        gate = torch.ones(
            (pred_velocity_base.shape[0],),
            device=pred_velocity_base.device,
            dtype=pred_velocity_base.dtype,
        )
        if self.feedback_delta_action_use_valid_tk_gate and isinstance(valid_tk_mask, torch.Tensor):
            valid_gate = valid_tk_mask.to(device=gate.device, dtype=gate.dtype).view(-1)
            if valid_gate.shape[0] != gate.shape[0]:
                raise ValueError(
                    f"`valid_tk_mask` batch mismatch for delta-action gate: got {valid_gate.shape[0]}, expected {gate.shape[0]}"
                )
            gate = gate * valid_gate.clamp(0.0, 1.0)
        gate = gate * alpha_tensor

        effective_delta = gate[:, None] * clipped_delta_action
        pred_velocity = pred_velocity_base + effective_delta[:, None, :]

        delta_norm = clipped_delta_action.detach().norm(dim=-1)
        effective_delta_norm = effective_delta.detach().norm(dim=-1)
        post_over_raw = (delta_norm / raw_delta_norm.clamp_min(1e-8)).mean()
        p95_vals = self._masked_percentiles(
            delta_norm.view(-1),
            valid_tk_mask if isinstance(valid_tk_mask, torch.Tensor) else None,
            quantiles=(0.95,),
        )
        p95 = p95_vals[0] if len(p95_vals) > 0 else None

        metrics.update(
            {
                "delta_action_applied": 1.0,
                "delta_action_gate_mean": float(gate.detach().float().mean().item()),
                "delta_action_alpha": float(alpha_value),
                "delta_action_raw_norm_mean": float(raw_delta_norm.mean().item()),
                "delta_action_postclip_norm_mean": float(delta_norm.mean().item()),
                "delta_action_post_over_raw_norm_mean": float(post_over_raw.item()),
                "delta_action_clip_saturation_frac": float(clip_saturation_frac),
                "delta_action_norm_mean": float(delta_norm.mean().item()),
                "delta_action_norm_p95": float(p95 if p95 is not None else 0.0),
                "delta_action_effective_norm_mean": float(effective_delta_norm.mean().item()),
                "delta_action_feedback_vec_norm_mean": float(
                    feedback_vec.detach().norm(dim=-1).mean().item()
                ),
            }
        )
        return pred_velocity, metrics, effective_delta

    def _prepare_feedback_tokens_for_context(
        self,
        feedback_tokens: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        if not isinstance(feedback_tokens, torch.Tensor):
            return None, None
        context_feedback_tokens = self._apply_feedback_context_norm(feedback_tokens)
        alpha_tensor, alpha_value = self._compute_feedback_alpha(
            device=context_feedback_tokens.device,
            dtype=context_feedback_tokens.dtype,
        )
        context_feedback_tokens = context_feedback_tokens * alpha_tensor
        if self.feedback_context_scale != 1.0:
            context_feedback_tokens = context_feedback_tokens * float(self.feedback_context_scale)
        return context_feedback_tokens, alpha_value

    def _should_run_feedback_probe(self) -> bool:
        """
        Low-frequency diagnostics probe:
        compare normal branch vs no-feedback branch without extra VLM forward.
        """
        if not self.training:
            return False
        interval = int(max(self.feedback_probe_interval, 0))
        if interval <= 0:
            return False
        self._feedback_probe_step += 1
        if self._feedback_probe_step < int(max(self.feedback_probe_start_step, 0)):
            return False
        return (self._feedback_probe_step % interval) == 0

    @staticmethod
    def _build_no_feedback_context_tokens(
        *,
        task_tokens: Optional[torch.Tensor],
        zero_feedback_tokens_for_context: Optional[torch.Tensor],
        feedback_context_only: bool,
    ) -> Optional[torch.Tensor]:
        if feedback_context_only:
            return zero_feedback_tokens_for_context
        if isinstance(zero_feedback_tokens_for_context, torch.Tensor):
            return (
                zero_feedback_tokens_for_context
                if task_tokens is None
                else torch.cat((zero_feedback_tokens_for_context, task_tokens), dim=1)
            )
        return task_tokens

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
                "loss_align": action_sequence.new_tensor(0.0),
                "loss_reg": action_sequence.new_tensor(0.0),
                "rrr": zeros_scalar,
                "expected_change": zeros_scalar,
                "delta_z_norm": zeros_scalar,
                "geo_energy": zeros_scalar,
                "drift_action_cos": zeros_scalar,
                "drift_l2": zeros_scalar,
            }

        action_context = self.build_world_action_context(action_sequence)
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
                "loss_align": action_sequence.new_tensor(0.0),
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
        pooled_residual = self.pool_task_tokens(delta_z)
        geo_weight = F.softplus(self.geo_weight_diag).to(device=pooled_residual.device, dtype=pooled_residual.dtype)
        dyn_per_sample = (pred_next_task - target_next).pow(2).mean(dim=(1, 2))
        geo_per_sample = 0.5 * ((pooled_residual * geo_weight) * pooled_residual).sum(dim=-1)
        loss_dyn = self._masked_mean(dyn_per_sample, mask)
        loss_geo = self._masked_mean(geo_per_sample, mask)
        if self.residual_pooling_mode == "slot_weighted" and self.residual_slot_entropy_weight > 0.0:
            slot_w = self._compute_slot_weights(
                num_slots=int(delta_z.shape[1]),
                device=delta_z.device,
                dtype=delta_z.dtype,
            )
            slot_entropy = -(slot_w * torch.log(slot_w.clamp_min(1e-8))).sum()
            loss_geo = loss_geo - float(self.residual_slot_entropy_weight) * slot_entropy
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

        pooled_expected_change = self.pool_task_tokens(pred_next_task - task_tokens)
        expected_change = pooled_expected_change.norm(dim=-1)
        delta_z_norm = pooled_residual.norm(dim=-1)
        rrr = delta_z_norm / expected_change.clamp_min(self.rrr_eps)
        drift_action = drift.mean(dim=1)
        drift_action_world = self.world_action_proj(drift_action)
        drift_action_cos = F.cosine_similarity(
            residual_world.float(),
            drift_action_world.float(),
            dim=-1,
            eps=self.align_cos_eps,
        ).to(dtype=pred_next_task.dtype)
        drift_action_cos = drift_action_cos.clamp(-1.0, 1.0)
        align_per_sample = 1.0 - drift_action_cos
        loss_align = self._masked_mean(align_per_sample, mask)
        if self.geo_align_weight != 0.0:
            loss_geo = loss_geo + float(self.geo_align_weight) * loss_align
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
            "loss_align": loss_align,
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
        feedback_tokens_for_context, feedback_alpha_value = self._prepare_feedback_tokens_for_context(
            feedback_tokens=feedback_tokens,
        )
        context_task_tokens = task_tokens
        if self.feedback_in_context_enabled:
            if self.feedback_context_only:
                context_task_tokens = feedback_tokens_for_context
            elif isinstance(feedback_tokens_for_context, torch.Tensor):
                context_task_tokens = (
                    feedback_tokens_for_context
                    if context_task_tokens is None
                    else torch.cat((feedback_tokens_for_context, context_task_tokens), dim=1)
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
        self._record_health("forward/feedback_tokens_for_context", feedback_tokens_for_context) if isinstance(feedback_tokens_for_context, torch.Tensor) else None
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
        pred_velocity_base = pred_velocity
        pred_velocity, delta_action_metrics, _ = self._apply_feedback_delta_action(
            pred_velocity_base=pred_velocity_base,
            feedback_tokens=feedback_tokens,
            valid_tk_mask=valid_tk_mask,
            enable_delta_action=True,
        )
        self._record_health("forward/pred_velocity_corrected", pred_velocity)
        loss_cfm = ((pred_velocity - velocity) ** 2).mean()
        feedback_probe_metrics: Dict[str, float] = {
            "feedback_probe_enabled": 1.0 if self.feedback_probe_interval > 0 else 0.0,
            "feedback_probe_interval": float(self.feedback_probe_interval),
            "feedback_probe_step": float(self._feedback_probe_step),
            "feedback_probe_triggered": 0.0,
            "feedback_probe_mode_light": 0.0,
            "feedback_probe_mode_full": 0.0,
            "feedback_probe_delta_loss_per_effective": 0.0,
            "feedback_probe_effective_norm_with": 0.0,
        }
        if (
            self._should_run_feedback_probe()
            and isinstance(feedback_tokens, torch.Tensor)
        ):
            feedback_probe_metrics["feedback_probe_triggered"] = 1.0
            feedback_probe_metrics["feedback_probe_step"] = float(self._feedback_probe_step)
            try:
                with torch.no_grad():
                    zero_feedback = torch.zeros_like(feedback_tokens)
                    if self.feedback_in_context_enabled:
                        feedback_probe_metrics["feedback_probe_mode_full"] = 1.0
                        zero_feedback_tokens_for_context, _ = self._prepare_feedback_tokens_for_context(
                            feedback_tokens=zero_feedback
                        )
                        context_task_tokens_no_fb = self._build_no_feedback_context_tokens(
                            task_tokens=task_tokens,
                            zero_feedback_tokens_for_context=zero_feedback_tokens_for_context,
                            feedback_context_only=self.feedback_context_only,
                        )

                        saved_layer_means = list(self._last_dit_layer_means)
                        saved_layer_vars = list(self._last_dit_layer_vars)
                        saved_layer_calls = int(self._layerwise_forward_calls)
                        saved_health_trace = list(self._last_health_trace)
                        saved_first_nonfinite_stage = self._first_nonfinite_stage
                        saved_first_nonfinite_record = (
                            dict(self._first_nonfinite_record)
                            if isinstance(self._first_nonfinite_record, dict)
                            else self._first_nonfinite_record
                        )
                        saved_log_interval = int(self.cross_attention_debug_log_interval)
                        try:
                            self.cross_attention_debug_log_interval = 0
                            hidden_states_no_fb = self._apply_layerwise_cross_attention(
                                sa_embs,
                                vl_embs_list,
                                temb,
                                encoder_attention_mask=encoder_attention_mask,
                                task_tokens=context_task_tokens_no_fb,
                                log_context="probe_no_feedback",
                            )
                        finally:
                            self.cross_attention_debug_log_interval = saved_log_interval
                            self._last_dit_layer_means = saved_layer_means
                            self._last_dit_layer_vars = saved_layer_vars
                            self._layerwise_forward_calls = saved_layer_calls
                            self._last_health_trace = saved_health_trace
                            self._first_nonfinite_stage = saved_first_nonfinite_stage
                            self._first_nonfinite_record = saved_first_nonfinite_record

                        pred_velocity_no_fb = self._process_output(
                            hidden_states_no_fb, temb, actions.shape[1]
                        )
                        if isinstance(drift, torch.Tensor):
                            pred_velocity_no_fb = pred_velocity_no_fb + drift.detach()
                        pred_velocity_no_fb, _, _ = self._apply_feedback_delta_action(
                            pred_velocity_base=pred_velocity_no_fb,
                            feedback_tokens=zero_feedback,
                            valid_tk_mask=valid_tk_mask,
                            enable_delta_action=True,
                        )
                        loss_no_fb = ((pred_velocity_no_fb - velocity.detach()) ** 2).mean()

                        h_with = hidden_states.detach().float()
                        h_no = hidden_states_no_fb.detach().float()
                        denom = h_no.norm().clamp_min(1e-6)
                        perturb_ratio = (h_with - h_no).norm() / denom
                        feedback_probe_metrics["feedback_probe_hidden_perturb_ratio"] = float(
                            perturb_ratio.item()
                        )
                    else:
                        feedback_probe_metrics["feedback_probe_mode_light"] = 1.0
                        pred_velocity_no_fb, _, _ = self._apply_feedback_delta_action(
                            pred_velocity_base=pred_velocity_base.detach(),
                            feedback_tokens=zero_feedback,
                            valid_tk_mask=valid_tk_mask,
                            enable_delta_action=True,
                        )
                        loss_no_fb = ((pred_velocity_no_fb - velocity.detach()) ** 2).mean()
                        pred_with = pred_velocity.detach().float()
                        pred_no = pred_velocity_no_fb.detach().float()
                        denom = pred_no.norm().clamp_min(1e-6)
                        perturb_ratio = (pred_with - pred_no).norm() / denom
                        # Keep legacy key for compatibility with dashboards.
                        feedback_probe_metrics["feedback_probe_hidden_perturb_ratio"] = float(
                            perturb_ratio.item()
                        )

                    feedback_probe_metrics["feedback_probe_delta_loss_with_minus_no"] = float(
                        (loss_cfm.detach() - loss_no_fb.detach()).item()
                    )
                    feedback_probe_metrics["feedback_probe_loss_no"] = float(
                        loss_no_fb.detach().item()
                    )
                    eff_norm_with = float(delta_action_metrics.get("delta_action_effective_norm_mean", 0.0))
                    feedback_probe_metrics["feedback_probe_effective_norm_with"] = eff_norm_with
                    feedback_probe_metrics["feedback_probe_delta_loss_per_effective"] = float(
                        (loss_cfm.detach() - loss_no_fb.detach()).item() / max(eff_norm_with, 1e-8)
                    )
            except Exception:
                feedback_probe_metrics["feedback_probe_error"] = 1.0
        feedback_probe_metrics["feedback_probe_step"] = float(self._feedback_probe_step)
        loss_dyn = world_guidance["loss_dyn"]
        loss_geo = world_guidance["loss_geo"]
        loss_align = world_guidance.get("loss_align", loss_cfm.new_tensor(0.0))
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
            "loss_align": float(loss_align.detach().item()),
            "loss_reg": float(loss_reg.detach().item()),
            "loss_total": float(total_loss.detach().item()),
            "world_predictor_params": float(self.world_predictor_param_count),
            "weight_fm": float(loss_weights["loss_w_fm"]),
            "weight_dyn": float(loss_weights["loss_w_dyn"]),
            "weight_geo": float(loss_weights["loss_w_geo"]),
            "weight_reg": float(loss_weights["loss_w_reg"]),
            "geo_align_weight": float(self.geo_align_weight),
        }
        if isinstance(valid_tk_mask, torch.Tensor):
            self._last_loss_breakdown["valid_tk_ratio"] = float((valid_tk_mask > 0.5).float().mean().item())
        if isinstance(feedback_tokens, torch.Tensor):
            self._last_loss_breakdown["feedback_token_num"] = float(feedback_tokens.shape[1])
            self._last_loss_breakdown["feedback_token_norm_mean"] = float(
                feedback_tokens.detach().norm(dim=-1).mean().item()
            )
        if isinstance(feedback_tokens_for_context, torch.Tensor):
            self._last_loss_breakdown["feedback_context_token_num"] = float(feedback_tokens_for_context.shape[1])
            self._last_loss_breakdown["feedback_context_token_norm_mean"] = float(
                feedback_tokens_for_context.detach().norm(dim=-1).mean().item()
            )
        if feedback_alpha_value is not None:
            self._last_loss_breakdown["feedback_alpha"] = float(feedback_alpha_value)
        self._last_loss_breakdown["feedback_context_only"] = 1.0 if self.feedback_context_only else 0.0
        self._last_loss_breakdown["feedback_in_context_enabled"] = 1.0 if self.feedback_in_context_enabled else 0.0
        self._last_loss_breakdown["feedback_context_norm_enabled"] = 0.0 if self.feedback_context_norm_type == "none" else 1.0
        self._last_loss_breakdown["feedback_context_alpha_mode_fixed"] = 1.0 if self.feedback_context_alpha_mode == "fixed" else 0.0
        self._last_loss_breakdown["feedback_context_alpha_mode_learnable"] = 1.0 if self.feedback_context_alpha_mode == "learnable" else 0.0
        self._last_loss_breakdown["feedback_context_alpha_mode_schedule"] = 1.0 if self.feedback_context_alpha_mode == "schedule" else 0.0
        self._last_loss_breakdown["feedback_context_alpha_schedule_step"] = float(
            int(self._feedback_context_schedule_step.item())
        )
        self._last_loss_breakdown["feedback_delta_action_inference_enabled"] = (
            1.0 if self.feedback_delta_action_enable_inference else 0.0
        )
        self._last_loss_breakdown["feedback_delta_action_alpha_mode_fixed"] = (
            1.0 if self.feedback_delta_action_alpha_mode == "fixed" else 0.0
        )
        self._last_loss_breakdown["feedback_delta_action_alpha_mode_learnable"] = (
            1.0 if self.feedback_delta_action_alpha_mode == "learnable" else 0.0
        )
        self._last_loss_breakdown["feedback_delta_action_alpha_mode_schedule"] = (
            1.0 if self.feedback_delta_action_alpha_mode == "schedule" else 0.0
        )
        self._last_loss_breakdown["feedback_delta_action_alpha_schedule_step"] = float(
            int(self._feedback_delta_action_schedule_step.item())
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
            p50, p95 = self._masked_percentiles(delta_z_norm.view(-1), valid_tk_mask, quantiles=(0.5, 0.95))
            if p50 is not None:
                self._last_loss_breakdown["delta_z_norm_p50"] = float(p50)
            if p95 is not None:
                self._last_loss_breakdown["delta_z_norm_p95"] = float(p95)
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
            drift_p50, drift_p95 = self._masked_percentiles(drift_l2.view(-1), valid_tk_mask, quantiles=(0.5, 0.95))
            if drift_p50 is not None:
                self._last_loss_breakdown["drift_l2_p50"] = float(drift_p50)
            if drift_p95 is not None:
                self._last_loss_breakdown["drift_l2_p95"] = float(drift_p95)
                self._last_loss_breakdown["drift_norm_p95"] = float(drift_p95)
        self._last_loss_breakdown.update(delta_action_metrics)
        self._last_loss_breakdown.update(feedback_probe_metrics)
        self._maybe_advance_feedback_alpha_schedule()
        self._maybe_advance_feedback_delta_action_alpha_schedule()
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
        valid_tk: Optional[torch.Tensor] = None,
        num_inference_timesteps: Optional[int] = None,
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

        if num_inference_timesteps is None:
            num_steps = int(self.num_inference_timesteps)
        else:
            num_steps = max(1, int(num_inference_timesteps))
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        threshold = float(self.rrr_threshold if rrr_threshold is None else rrr_threshold)
        valid_tk_mask = None
        if isinstance(valid_tk, torch.Tensor):
            valid_tk_mask = valid_tk.to(device=device, dtype=vl_embs_list[0].dtype).view(-1)
            if valid_tk_mask.shape[0] != batch_size:
                raise ValueError(
                    f"`valid_tk` batch mismatch: got {valid_tk_mask.shape[0]}, expected {batch_size}"
                )
            valid_tk_mask = valid_tk_mask.clamp(0.0, 1.0)
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
        feedback_tokens_for_context, feedback_alpha_value = self._prepare_feedback_tokens_for_context(
            feedback_tokens=feedback_tokens,
        )
        context_task_tokens = task_tokens
        if self.feedback_in_context_enabled:
            if self.feedback_context_only:
                context_task_tokens = feedback_tokens_for_context
            elif isinstance(feedback_tokens_for_context, torch.Tensor):
                context_task_tokens = (
                    feedback_tokens_for_context
                    if context_task_tokens is None
                    else torch.cat((feedback_tokens_for_context, context_task_tokens), dim=1)
                )
        replans = 0
        t = 0
        predicted_task_tokens = None
        expected_change = None
        rrr_trace = []
        delta_z_norm_trace = []
        geo_energy_trace = []
        drift_action_cos_trace = []
        delta_action_gate_trace = []
        delta_action_effective_norm_trace = []
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
                valid_tk_mask=valid_tk_mask,
            )
            drift = world_guidance["drift"]
            if isinstance(drift, torch.Tensor):
                pred_velocity = pred_velocity + drift
            pred_velocity, delta_action_metrics, _ = self._apply_feedback_delta_action(
                pred_velocity_base=pred_velocity,
                feedback_tokens=feedback_tokens,
                valid_tk_mask=valid_tk_mask,
                enable_delta_action=bool(self.feedback_delta_action_enable_inference),
            )
            gate_val = delta_action_metrics.get("delta_action_gate_mean", None)
            eff_norm_val = delta_action_metrics.get("delta_action_effective_norm_mean", None)
            if isinstance(gate_val, (float, int)):
                delta_action_gate_trace.append(float(gate_val))
            if isinstance(eff_norm_val, (float, int)):
                delta_action_effective_norm_trace.append(float(eff_norm_val))
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
            "feedback/context_token_num": (
                int(feedback_tokens_for_context.shape[1])
                if isinstance(feedback_tokens_for_context, torch.Tensor)
                else 0
            ),
            "feedback/context_token_norm_mean": (
                float(feedback_tokens_for_context.detach().norm(dim=-1).mean().item())
                if isinstance(feedback_tokens_for_context, torch.Tensor)
                else 0.0
            ),
            "feedback/context_only": bool(self.feedback_context_only),
            "feedback/context_norm_type": str(self.feedback_context_norm_type),
            "feedback/alpha_mode": str(self.feedback_context_alpha_mode),
            "feedback/alpha": float(feedback_alpha_value) if feedback_alpha_value is not None else None,
            "feedback/alpha_schedule_step": int(self._feedback_context_schedule_step.item()),
            "feedback/in_context_enabled": bool(self.feedback_in_context_enabled),
            "feedback/delta_action_enabled": bool(self.feedback_delta_action_enabled),
            "feedback/delta_action_gate_mean": (
                float(sum(delta_action_gate_trace) / len(delta_action_gate_trace))
                if len(delta_action_gate_trace) > 0
                else 0.0
            ),
            "feedback/delta_action_effective_norm_mean": (
                float(sum(delta_action_effective_norm_trace) / len(delta_action_effective_norm_trace))
                if len(delta_action_effective_norm_trace) > 0
                else 0.0
            ),
            "feedback/valid_tk_ratio": (
                float((valid_tk_mask > 0.5).float().mean().item())
                if isinstance(valid_tk_mask, torch.Tensor)
                else 1.0
            ),
            "feedback/num_inference_timesteps_used": int(num_steps),
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
