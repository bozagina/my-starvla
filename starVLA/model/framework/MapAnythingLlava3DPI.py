from typing import Optional, List, Tuple, Dict
import math
import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    get_action_model,
    LayerwiseFlowmatchingActionHead,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
try:
    from deployment.model_server.tools.image_tools import to_pil_preserve
except Exception:  # pragma: no cover - fallback for training-only envs without deployment package
    def to_pil_preserve(x):
        """Best-effort PIL conversion used when deployment helpers are unavailable."""
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        if torch.is_tensor(x):
            t = x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError(f"Unsupported input type for to_pil_preserve fallback: {type(x)}")


logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("MapAnythingLlava3DPI")
class MapAnythingLlava3D_PI(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.mapanythingllava3d_vlm_interface = get_vlm_model(config=self.config)

        vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
        if vlm_core is not None and hasattr(vlm_core, "enable_geom_feature_hook"):
            try:
                vlm_core.enable_geom_feature_hook(max_steps=1000)
                
            except Exception:
                pass

        llm = self.mapanythingllava3d_vlm_interface.model.language_model.model
        num_vl_layers = getattr(llm.config, "num_hidden_layers", 32)
        llm_hidden_size = getattr(llm.config, "hidden_size", self.mapanythingllava3d_vlm_interface.model.hidden_size)
        self.config.framework.mapanything_llava3d.vl_hidden_dim = llm_hidden_size
        self.config.framework.mapanything_llava3d.num_vl_layers = num_vl_layers
        self.vl_layer_selection = getattr(self.config.framework.action_model, "vl_layer_selection", "last")
        self.normalize_vl_hidden = bool(
            getattr(self.config.framework.mapanything_llava3d, "normalize_vl_hidden", False)
        )
        action_cfg = self.config.framework.action_model
        self.enable_causal_feedback_token = bool(
            getattr(action_cfg, "enable_causal_feedback_token", False)
        )
        self.causal_feedback_token_num = max(
            1, int(getattr(action_cfg, "causal_feedback_token_num", 1))
        )
        self.causal_feedback_detach_delta = bool(
            getattr(action_cfg, "causal_feedback_detach_delta", True)
        )
        self.causal_feedback_detach_action = bool(
            getattr(action_cfg, "causal_feedback_detach_action", True)
        )
        self.causal_feedback_use_valid_mask = bool(
            getattr(action_cfg, "causal_feedback_use_valid_mask", True)
        )
        self.causal_feedback_scale = float(
            getattr(action_cfg, "causal_feedback_scale", 1.0)
        )
        self.causal_feedback_dropout_p = float(
            getattr(action_cfg, "causal_feedback_dropout", 0.0)
        )
        # Optional auxiliary supervision for Path-A feedback tokens.
        # Default is disabled to preserve old behavior.
        self.causal_feedback_aux_weight = float(
            getattr(action_cfg, "causal_feedback_aux_weight", 0.0)
        )
        self.causal_feedback_aux_detach_target = bool(
            getattr(action_cfg, "causal_feedback_aux_detach_target", True)
        )
        # Optional directional term for auxiliary supervision:
        #   loss_fb = mse(pred_delta, target_delta) + dir_w * (1 - cos(pred_delta, target_delta))
        self.causal_feedback_aux_dir_weight = float(
            getattr(action_cfg, "causal_feedback_aux_dir_weight", 0.0)
        )
        self.causal_feedback_aux_dir_eps = float(
            getattr(action_cfg, "causal_feedback_aux_dir_eps", 1e-6)
        )
        # v4-1-1 residual/soft-mask options.
        self.patha_residual_mode = str(
            getattr(action_cfg, "patha_residual_mode", "pooled_delta_z")
        ).strip().lower()
        if self.patha_residual_mode not in ("pooled_delta_z", "token_delta_geo"):
            logger.warning(
                "[causal_feedback] invalid patha_residual_mode=%s, fallback to pooled_delta_z",
                self.patha_residual_mode,
            )
            self.patha_residual_mode = "pooled_delta_z"
        self.soft_mask_enabled = bool(
            getattr(action_cfg, "soft_mask_enabled", False)
        )
        self.soft_mask_lambda = float(
            getattr(action_cfg, "soft_mask_lambda", 0.3)
        )
        self.soft_mask_ema_beta = float(
            getattr(action_cfg, "soft_mask_ema_beta", 0.2)
        )
        self.soft_mask_query_agg = str(
            getattr(action_cfg, "soft_mask_query_agg", "mean")
        ).strip().lower()
        if self.soft_mask_query_agg not in ("mean", "max"):
            logger.warning(
                "[causal_feedback] invalid soft_mask_query_agg=%s, fallback to mean",
                self.soft_mask_query_agg,
            )
            self.soft_mask_query_agg = "mean"
        self.soft_mask_temperature = float(
            getattr(action_cfg, "soft_mask_temperature", 1.0)
        )
        if self.soft_mask_temperature <= 0.0:
            self.soft_mask_temperature = 1.0
        self.soft_mask_use_ema_inference = bool(
            getattr(action_cfg, "soft_mask_use_ema_inference", True)
        )
        self.soft_mask_use_ema_training = bool(
            getattr(action_cfg, "soft_mask_use_ema_training", False)
        )
        # Low-frequency stability probe for Path-A residual direction:
        # cosine(mean(delta_t), mean(delta_{t-1})).
        self.causal_feedback_delta_cos_interval = max(
            1, int(getattr(action_cfg, "causal_feedback_delta_cos_interval", 100))
        )
        self._causal_feedback_delta_step = 0
        self._causal_feedback_prev_delta_mean = None
        action_dim = int(getattr(action_cfg, "action_dim", 0) or 0)
        self._causal_feedback_hidden_size = int(llm_hidden_size)
        self._causal_feedback_ready = bool(
            self.enable_causal_feedback_token
            and action_dim > 0
            and self._causal_feedback_hidden_size > 0
        )
        if self.enable_causal_feedback_token and not self._causal_feedback_ready:
            logger.warning(
                "[causal_feedback] disabled due to invalid dimensions: hidden=%s action_dim=%s",
                self._causal_feedback_hidden_size,
                action_dim,
            )
        if self._causal_feedback_ready:
            self.causal_feedback_delta_norm = nn.LayerNorm(self._causal_feedback_hidden_size)
            self.causal_feedback_action_proj = nn.Linear(action_dim, self._causal_feedback_hidden_size)
            self.causal_feedback_action_norm = nn.LayerNorm(self._causal_feedback_hidden_size)
            self.causal_feedback_fuse = nn.Sequential(
                nn.LayerNorm(self._causal_feedback_hidden_size * 2),
                nn.Linear(self._causal_feedback_hidden_size * 2, self._causal_feedback_hidden_size),
                nn.GELU(),
                nn.Linear(
                    self._causal_feedback_hidden_size,
                    self.causal_feedback_token_num * self._causal_feedback_hidden_size,
                ),
            )
            self.causal_feedback_dropout = nn.Dropout(self.causal_feedback_dropout_p)
            self.causal_feedback_recon_head = nn.Sequential(
                nn.LayerNorm(self._causal_feedback_hidden_size),
                nn.Linear(self._causal_feedback_hidden_size, self._causal_feedback_hidden_size),
                nn.GELU(),
                nn.Linear(self._causal_feedback_hidden_size, self._causal_feedback_hidden_size),
            )
        else:
            self.causal_feedback_delta_norm = None
            self.causal_feedback_action_proj = None
            self.causal_feedback_action_norm = None
            self.causal_feedback_fuse = None
            self.causal_feedback_dropout = None
            self.causal_feedback_recon_head = None

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.vlm_use_cache = False
        self.rrr_replan_threshold = float(
            getattr(self.config.framework.action_model, "rrr_replan_threshold", 2.5)
        )
        self.rrr_eps = float(getattr(self.config.framework.action_model, "rrr_eps", 1e-4))
        self.enable_causal_feedback_inference = bool(
            getattr(action_cfg, "enable_causal_feedback_inference", True)
        )
        self._rrr_prev_observed_task_tokens = None
        self._rrr_prev_predicted_task_tokens = None
        self._rrr_prev_expected_change = None
        self._patha_prev_task_tokens = None
        self._patha_prev_action_chunk = None
        self._patha_prev_geo_tokens = None
        self._patha_prev_vision_tokens = None
        self._patha_prev_language_queries = None
        self._patha_prev_language_query_mask = None
        self._patha_soft_mask_ema = None
        self._patha_prev_soft_mask_mean = None
        self._causal_feedback_prev_delta_mean = None
        self._causal_feedback_delta_step = 0
        self._debug_last_feedback_tokens_for_grad = None
        self._debug_last_task_tokens_for_grad = None
        self.reset_inference_state()
        self._configure_memory_optimizations()

    def reset_inference_state(self, **kwargs) -> None:
        """
        Reset per-session inference memories.

        This clears:
        - RRR-related rolling tensors.
        - Path-A rolling context used for causal feedback token construction.
        """
        self._rrr_prev_observed_task_tokens = None
        self._rrr_prev_predicted_task_tokens = None
        self._rrr_prev_expected_change = None
        self._patha_prev_task_tokens = None
        self._patha_prev_action_chunk = None
        self._patha_prev_geo_tokens = None
        self._patha_prev_vision_tokens = None
        self._patha_prev_language_queries = None
        self._patha_prev_language_query_mask = None
        self._patha_soft_mask_ema = None
        self._patha_prev_soft_mask_mean = None

    @staticmethod
    def _set_module_use_cache(module, use_cache: bool, module_name: str):
        if module is None:
            return
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            setattr(cfg, "use_cache", bool(use_cache))
            logger.info(f"[memory_opt] set {module_name}.config.use_cache={bool(use_cache)}")
        generation_cfg = getattr(module, "generation_config", None)
        if generation_cfg is not None and hasattr(generation_cfg, "use_cache"):
            setattr(generation_cfg, "use_cache", bool(use_cache))

    @staticmethod
    def _enable_gradient_checkpointing(module, module_name: str):
        if module is None:
            return False
        for fn_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    logger.info(f"[memory_opt] enabled gradient checkpointing on {module_name} via `{fn_name}`")
                    return True
                except TypeError:
                    try:
                        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
                        logger.info(f"[memory_opt] enabled gradient checkpointing on {module_name} via `{fn_name}` with kwargs")
                        return True
                    except Exception as e:
                        logger.warning(f"[memory_opt] failed enabling gradient checkpointing on {module_name}: {e}")
                except Exception as e:
                    logger.warning(f"[memory_opt] failed enabling gradient checkpointing on {module_name}: {e}")
        return False

    def _configure_memory_optimizations(self):
        fw_cfg = getattr(self.config, "framework", None)
        ma_cfg = getattr(fw_cfg, "mapanything_llava3d", None) if fw_cfg is not None else None
        if ma_cfg is not None and hasattr(ma_cfg, "use_cache"):
            self.vlm_use_cache = bool(getattr(ma_cfg, "use_cache"))
        else:
            self.vlm_use_cache = False

        vlm_interface = self.mapanythingllava3d_vlm_interface
        vlm_core = getattr(vlm_interface, "model", None)
        language_wrapper = getattr(vlm_core, "language_model", None) if vlm_core is not None else None
        llm_core = getattr(language_wrapper, "model", None) if language_wrapper is not None else None
        action_dit = getattr(self.action_model, "model", None)

        self._set_module_use_cache(vlm_core, self.vlm_use_cache, "vlm_core")
        self._set_module_use_cache(language_wrapper, self.vlm_use_cache, "language_wrapper")
        self._set_module_use_cache(llm_core, self.vlm_use_cache, "llm_core")

        enable_gc = bool(getattr(getattr(self.config, "trainer", None), "enable_gradient_checkpointing", False))
        if not enable_gc:
            logger.info("[memory_opt] gradient checkpointing disabled by config")
            return

        gc_enabled = False
        gc_enabled = self._enable_gradient_checkpointing(vlm_core, "vlm_core") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(language_wrapper, "language_wrapper") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(llm_core, "llm_core") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(action_dit, "action_dit") or gc_enabled
        if not gc_enabled:
            logger.warning("[memory_opt] no module accepted gradient checkpointing enable call")

    @staticmethod
    def _parse_bool_flag(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    @staticmethod
    def _resolve_feedback_ablation_mode(value) -> str:
        mode = "none" if value is None else str(value).strip().lower()
        if mode in ("0", "false", "off"):
            mode = "none"
        if mode not in ("none", "zero", "shuffle"):
            raise ValueError(
                f"Invalid `feedback_ablation_mode`={value!r}; expected one of ['none', 'zero', 'shuffle']."
            )
        return mode

    def _apply_feedback_ablation(
        self,
        feedback_tokens: Optional[torch.Tensor],
        *,
        mode: str = "none",
        seed: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        stats: Dict[str, float] = {
            "debug/causal_feedback/ablation_mode_none": 1.0 if mode == "none" else 0.0,
            "debug/causal_feedback/ablation_mode_zero": 1.0 if mode == "zero" else 0.0,
            "debug/causal_feedback/ablation_mode_shuffle": 1.0 if mode == "shuffle" else 0.0,
            "debug/causal_feedback/ablation_applied": 0.0,
        }
        if not isinstance(feedback_tokens, torch.Tensor):
            return feedback_tokens, stats
        if mode == "none":
            return feedback_tokens, stats

        ablated = feedback_tokens
        if mode == "zero":
            ablated = torch.zeros_like(feedback_tokens)
        elif mode == "shuffle":
            batch = int(feedback_tokens.shape[0])
            if batch > 1:
                try:
                    if seed is not None:
                        generator = torch.Generator(device=feedback_tokens.device)
                        generator.manual_seed(int(seed))
                        perm = torch.randperm(batch, device=feedback_tokens.device, generator=generator)
                    else:
                        perm = torch.randperm(batch, device=feedback_tokens.device)
                except Exception:
                    perm = torch.randperm(batch, device=feedback_tokens.device)
                ablated = feedback_tokens.index_select(0, perm)
                with torch.no_grad():
                    stats["debug/causal_feedback/ablation_shuffle_changed"] = float(
                        bool((perm != torch.arange(batch, device=perm.device)).any().item())
                    )
            else:
                stats["debug/causal_feedback/ablation_shuffle_changed"] = 0.0

        stats["debug/causal_feedback/ablation_applied"] = 1.0
        with torch.no_grad():
            stats["debug/causal_feedback/token_norm_mean_after_ablation"] = float(
                ablated.detach().norm(dim=-1).mean().item()
            )
            stats["debug/causal_feedback/token_absmax_after_ablation"] = float(
                ablated.detach().abs().max().item()
            )
        return ablated, stats

    @staticmethod
    def _safe_metric_key(text: str) -> str:
        if text is None:
            return "unknown"
        out = []
        for ch in str(text):
            if ch.isalnum() or ch in ("_", "-"):
                out.append(ch)
            else:
                out.append("_")
        key = "".join(out).strip("_")
        if not key:
            return "unknown"
        return key[:64]

    @staticmethod
    def _token_std_mean(tokens: Optional[torch.Tensor]) -> Optional[float]:
        """
        Mean std across token dimension:
            std(tokens, dim=1) -> mean(all dims).
        Used to detect token-level homogenization.
        """
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            return None
        if tokens.shape[1] <= 1:
            return 0.0
        with torch.no_grad():
            t = tokens.detach().float()
            return float(t.std(dim=1, unbiased=False).mean().item())

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if values.ndim != 1:
            values = values.view(-1)
        if mask is None:
            return values.mean()
        m = mask.view(-1).to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
        if m.shape[0] != values.shape[0]:
            raise ValueError(
                f"Mask batch mismatch: values.shape[0]={values.shape[0]}, mask.shape[0]={m.shape[0]}"
            )
        denom = m.sum()
        if float(denom.item()) <= 0.0:
            return values.new_tensor(0.0)
        return torch.sum(values * m) / denom

    @staticmethod
    def _masked_quantiles_1d(
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
        quantiles: Tuple[float, ...],
    ) -> List[Optional[float]]:
        if values.ndim != 1:
            values = values.view(-1)
        if mask is None:
            active = values
        else:
            m = mask.view(-1).to(device=values.device, dtype=values.dtype)
            if m.shape[0] != values.shape[0]:
                return [None for _ in quantiles]
            active = values[m > 0.5]
        if active.numel() == 0:
            return [None for _ in quantiles]
        try:
            q_tensor = torch.tensor(
                [float(max(0.0, min(1.0, q))) for q in quantiles],
                device=active.device,
                dtype=active.dtype,
            )
            out = torch.quantile(active, q_tensor)
            return [float(v.item()) for v in out]
        except Exception:
            sorted_vals, _ = torch.sort(active)
            n = int(sorted_vals.numel())
            out = []
            for q in quantiles:
                idx = int(max(0, min(n - 1, round(float(q) * max(n - 1, 0)))))
                out.append(float(sorted_vals[idx].item()))
            return out

    def _maybe_apply_soft_mask_ema(
        self,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(alpha, torch.Tensor) or alpha.ndim != 2:
            return alpha
        use_ema = bool(self.soft_mask_use_ema_training if self.training else self.soft_mask_use_ema_inference)
        if not use_ema:
            return alpha
        beta = float(max(0.0, min(1.0, self.soft_mask_ema_beta)))
        if beta <= 0.0:
            return alpha
        with torch.no_grad():
            cached = self._patha_soft_mask_ema
            alpha_detached = alpha.detach()
            if not isinstance(cached, torch.Tensor) or cached.shape != alpha_detached.shape:
                cached = alpha_detached.clone()
            else:
                cached = (1.0 - beta) * cached.to(device=alpha_detached.device, dtype=alpha_detached.dtype) + beta * alpha_detached
            self._patha_soft_mask_ema = cached.detach()
        return self._patha_soft_mask_ema.to(device=alpha.device, dtype=alpha.dtype)

    def _build_soft_mask(
        self,
        *,
        vision_tokens: Optional[torch.Tensor],
        geometric_tokens: Optional[torch.Tensor],
        language_queries: Optional[torch.Tensor],
        language_query_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        stats: Dict[str, float] = {
            "debug/causal_feedback/soft_mask_enabled": float(self.soft_mask_enabled),
            "debug/causal_feedback/soft_mask_applied": 0.0,
            "debug/causal_feedback/soft_mask_lambda": float(self.soft_mask_lambda),
            "debug/causal_feedback/soft_mask_temperature": float(self.soft_mask_temperature),
        }
        if not self.soft_mask_enabled:
            return None, stats
        if not (
            isinstance(vision_tokens, torch.Tensor)
            and isinstance(geometric_tokens, torch.Tensor)
            and isinstance(language_queries, torch.Tensor)
        ):
            stats["debug/causal_feedback/soft_mask_missing_inputs"] = 1.0
            return None, stats
        if vision_tokens.ndim != 3 or geometric_tokens.ndim != 3 or language_queries.ndim != 3:
            stats["debug/causal_feedback/soft_mask_invalid_ndim"] = 1.0
            return None, stats
        if vision_tokens.shape[0] != geometric_tokens.shape[0] or vision_tokens.shape[0] != language_queries.shape[0]:
            stats["debug/causal_feedback/soft_mask_batch_mismatch"] = 1.0
            return None, stats
        if vision_tokens.shape[2] != geometric_tokens.shape[2] or vision_tokens.shape[2] != language_queries.shape[2]:
            stats["debug/causal_feedback/soft_mask_hidden_mismatch"] = 1.0
            return None, stats
        bsz = int(vision_tokens.shape[0])
        token_n = int(min(vision_tokens.shape[1], geometric_tokens.shape[1]))
        if token_n <= 0:
            stats["debug/causal_feedback/soft_mask_empty_tokens"] = 1.0
            return None, stats
        vision_tokens = vision_tokens[:, :token_n, :]
        geometric_tokens = geometric_tokens[:, :token_n, :]

        dtype = language_queries.dtype
        device = language_queries.device
        temp = float(max(self.soft_mask_temperature, 1e-6))
        lam = float(max(0.0, min(1.0, self.soft_mask_lambda)))

        q = F.normalize(language_queries.to(dtype=dtype), dim=-1)
        v = F.normalize(vision_tokens.to(device=device, dtype=dtype), dim=-1)
        g = F.normalize(geometric_tokens.to(device=device, dtype=dtype), dim=-1)
        logits_vis = torch.einsum("blh,bnh->bln", q, v) / temp
        logits_geo = torch.einsum("blh,bnh->bln", q, g) / temp
        attn_vis = torch.softmax(logits_vis, dim=-1)
        attn_geo = torch.softmax(logits_geo, dim=-1)

        query_mask = None
        if isinstance(language_query_mask, torch.Tensor):
            qm = language_query_mask.to(device=device, dtype=torch.bool)
            if qm.ndim == 2 and qm.shape[0] == bsz and qm.shape[1] == language_queries.shape[1]:
                query_mask = qm

        if query_mask is None:
            query_weight = torch.full(
                (bsz, language_queries.shape[1]),
                1.0 / max(int(language_queries.shape[1]), 1),
                device=device,
                dtype=dtype,
            )
        else:
            query_weight = query_mask.to(dtype=dtype)
            query_weight = query_weight / query_weight.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.soft_mask_query_agg == "max":
            if query_mask is not None:
                mask_expand = query_mask.unsqueeze(-1)
                attn_vis = attn_vis.masked_fill(~mask_expand, 0.0)
                attn_geo = attn_geo.masked_fill(~mask_expand, 0.0)
            alpha_vis = attn_vis.max(dim=1).values
            alpha_geo = attn_geo.max(dim=1).values
        else:
            alpha_vis = torch.sum(attn_vis * query_weight.unsqueeze(-1), dim=1)
            alpha_geo = torch.sum(attn_geo * query_weight.unsqueeze(-1), dim=1)

        alpha = (1.0 - lam) * alpha_vis + lam * alpha_geo
        alpha = alpha.clamp_min(0.0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        alpha = self._maybe_apply_soft_mask_ema(alpha)

        with torch.no_grad():
            entropy = -torch.sum(alpha * torch.log(alpha.clamp_min(1e-9)), dim=-1)
            topk = min(int(token_n), 32)
            topk_mass = alpha.topk(topk, dim=-1).values.sum(dim=-1)
            stats["debug/causal_feedback/soft_mask_applied"] = 1.0
            stats["debug/causal_feedback/soft_mask_token_num"] = float(token_n)
            stats["debug/causal_feedback/soft_mask_entropy"] = float(entropy.mean().item())
            stats["debug/causal_feedback/soft_mask_topk_mass_32"] = float(topk_mass.mean().item())
            try:
                curr_mean = alpha.detach().float().mean(dim=0)
                prev_mean = self._patha_prev_soft_mask_mean
                if isinstance(prev_mean, torch.Tensor) and prev_mean.shape == curr_mean.shape:
                    cos_prev = F.cosine_similarity(
                        curr_mean.unsqueeze(0),
                        prev_mean.to(device=curr_mean.device).unsqueeze(0),
                        dim=-1,
                    )
                    stats["debug/causal_feedback/soft_mask_cos_prev"] = float(cos_prev.item())
                    stats["debug/causal_feedback/soft_mask_cos_prev_available"] = 1.0
                else:
                    stats["debug/causal_feedback/soft_mask_cos_prev_available"] = 0.0
                self._patha_prev_soft_mask_mean = curr_mean.detach().cpu()
            except Exception:
                stats["debug/causal_feedback/soft_mask_cos_prev_error"] = 1.0
        return alpha, stats

    def _pool_task_tokens_with_action_context(
        self,
        *,
        task_tokens: torch.Tensor,
        action_context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """
        Pool task tokens into [B, H] for Path-A feedback.

        Priority:
        1) Action-conditioned pooling (dynamic slot weights from action context).
        2) Action head slot/mean pooling (if available).
        3) Plain mean pooling fallback.
        """
        if not isinstance(task_tokens, torch.Tensor) or task_tokens.ndim != 3:
            raise ValueError(
                f"`task_tokens` must be tensor [B, K, H], got type={type(task_tokens)} "
                f"shape={None if not isinstance(task_tokens, torch.Tensor) else tuple(task_tokens.shape)}"
            )

        # Base fallback pooling to preserve behavior under any failure.
        if hasattr(self.action_model, "pool_task_tokens"):
            pooled_base = self.action_model.pool_task_tokens(task_tokens)
        else:
            pooled_base = task_tokens.mean(dim=1)

        if task_tokens.shape[1] == 1:
            one_w = task_tokens.new_ones((task_tokens.shape[0], 1))
            return pooled_base, one_w, False
        if (
            not self._causal_feedback_ready
            or self.causal_feedback_action_proj is None
            or not isinstance(action_context, torch.Tensor)
        ):
            return pooled_base, None, False

        try:
            proj_dtype = self.causal_feedback_action_proj.weight.dtype
            tokens_for_pool = task_tokens.to(dtype=proj_dtype)
            action_ctx = action_context.to(device=tokens_for_pool.device, dtype=proj_dtype)
            action_cond = self.causal_feedback_action_proj(action_ctx)  # [B, H]
            if action_cond.ndim != 2 or action_cond.shape[0] != tokens_for_pool.shape[0]:
                return pooled_base, None, False
            if action_cond.shape[1] != tokens_for_pool.shape[2]:
                return pooled_base, None, False

            # Dynamic slot weights conditioned on current action context.
            logits = torch.sum(tokens_for_pool * action_cond.unsqueeze(1), dim=-1)
            logits = logits / max(math.sqrt(float(tokens_for_pool.shape[-1])), 1.0)

            # If action head has static slot priors, use them as a stabilizing prior.
            if hasattr(self.action_model, "_compute_slot_weights"):
                static_w = self.action_model._compute_slot_weights(
                    num_slots=int(tokens_for_pool.shape[1]),
                    device=tokens_for_pool.device,
                    dtype=tokens_for_pool.dtype,
                )
                if isinstance(static_w, torch.Tensor) and static_w.numel() == tokens_for_pool.shape[1]:
                    logits = logits + torch.log(static_w.view(1, -1).clamp_min(1e-6))

            slot_w = torch.softmax(logits, dim=1)
            pooled = torch.sum(tokens_for_pool * slot_w.unsqueeze(-1), dim=1)
            return pooled.to(dtype=task_tokens.dtype), slot_w.to(dtype=task_tokens.dtype), True
        except Exception:
            return pooled_base, None, False

    @staticmethod
    def _synchronize_cuda_for_timing(reference: Optional[torch.Tensor] = None) -> None:
        if not torch.cuda.is_available():
            return
        try:
            if isinstance(reference, torch.Tensor) and reference.device.type == "cuda":
                torch.cuda.synchronize(device=reference.device)
            else:
                torch.cuda.synchronize()
        except Exception:
            # Timing should never break runtime.
            pass

    def _start_step_timer(self, device: Optional[torch.device] = None) -> dict:
        """Start a timing window with CUDA events when possible."""
        use_cuda = bool(torch.cuda.is_available())
        if use_cuda:
            timer_device = None
            if isinstance(device, torch.device) and device.type == "cuda":
                timer_device = device
            try:
                if timer_device is not None:
                    torch.cuda.synchronize(device=timer_device)
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return {
                "mode": "cuda",
                "start_event": start_event,
                "end_event": end_event,
                "device": timer_device,
            }
        return {"mode": "cpu", "start_time": time.perf_counter()}

    def _stop_step_timer(self, timer_ctx: dict, reference: Optional[torch.Tensor] = None) -> float:
        """Finish timing window and return milliseconds."""
        if not isinstance(timer_ctx, dict):
            return 0.0
        if timer_ctx.get("mode") == "cuda":
            start_event = timer_ctx.get("start_event", None)
            end_event = timer_ctx.get("end_event", None)
            if start_event is None or end_event is None:
                return 0.0
            end_event.record()
            try:
                if isinstance(reference, torch.Tensor) and reference.device.type == "cuda":
                    torch.cuda.synchronize(device=reference.device)
                else:
                    timer_device = timer_ctx.get("device", None)
                    if isinstance(timer_device, torch.device) and timer_device.type == "cuda":
                        torch.cuda.synchronize(device=timer_device)
                    else:
                        torch.cuda.synchronize()
            except Exception:
                pass
            return float(start_event.elapsed_time(end_event))
        if "start_time" not in timer_ctx:
            return 0.0
        return float((time.perf_counter() - float(timer_ctx["start_time"])) * 1000.0)

    def _infer_control_hz(self) -> float:
        action_cfg = getattr(getattr(self.config, "framework", None), "action_model", None)
        candidates = (
            "control_hz",
            "control_frequency_hz",
            "policy_hz",
            "execution_hz",
        )
        for key in candidates:
            try:
                value = getattr(action_cfg, key, None)
                if value is None:
                    continue
                value = float(value)
                if value > 0:
                    return value
            except Exception:
                continue
        # Keep a conservative default aligned with LIBERO-style control loops.
        return 20.0

    def _build_causal_feedback_tokens(
        self,
        *,
        task_tokens: Optional[torch.Tensor],
        task_tokens_next: Optional[torch.Tensor],
        action_chunk: Optional[torch.Tensor],
        valid_tk_mask: Optional[torch.Tensor],
        geometric_tokens: Optional[torch.Tensor] = None,
        geometric_tokens_next: Optional[torch.Tensor] = None,
        vision_tokens: Optional[torch.Tensor] = None,
        language_queries: Optional[torch.Tensor] = None,
        language_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float], Optional[torch.Tensor]]:
        stats: Dict[str, float] = {
            "debug/causal_feedback/enabled": float(self._causal_feedback_ready),
            "debug/causal_feedback/applied": 0.0,
            "debug/causal_feedback/residual_mode_token_delta_geo": 1.0
            if self.patha_residual_mode == "token_delta_geo"
            else 0.0,
            "debug/causal_feedback/residual_mode_pooled_delta_z": 1.0
            if self.patha_residual_mode == "pooled_delta_z"
            else 0.0,
            "debug/causal_feedback/residual_mode_used_token_delta_geo": 0.0,
            "debug/causal_feedback/residual_mode_used_pooled_delta_z": 0.0,
        }
        if not self._causal_feedback_ready:
            return None, stats, None
        if not isinstance(task_tokens, torch.Tensor):
            return None, stats, None
        if not isinstance(task_tokens_next, torch.Tensor):
            stats["debug/causal_feedback/missing_next_tokens"] = 1.0
            return None, stats, None
        if not isinstance(action_chunk, torch.Tensor):
            return None, stats, None
        if task_tokens.shape != task_tokens_next.shape:
            logger.warning(
                "[causal_feedback] skip due to shape mismatch task=%s next=%s",
                tuple(task_tokens.shape),
                tuple(task_tokens_next.shape),
            )
            stats["debug/causal_feedback/shape_mismatch"] = 1.0
            return None, stats, None
        if task_tokens.ndim != 3:
            logger.warning(
                "[causal_feedback] skip due to invalid task token ndim=%s",
                task_tokens.ndim,
            )
            stats["debug/causal_feedback/invalid_task_ndim"] = 1.0
            return None, stats, None
        batch_size, _, hidden = task_tokens.shape
        if hidden != self._causal_feedback_hidden_size:
            logger.warning(
                "[causal_feedback] skip due to hidden mismatch hidden=%s expected=%s",
                hidden,
                self._causal_feedback_hidden_size,
            )
            stats["debug/causal_feedback/hidden_mismatch"] = 1.0
            return None, stats, None
        if action_chunk.ndim != 3 or action_chunk.shape[0] != batch_size:
            logger.warning(
                "[causal_feedback] skip due to invalid action chunk shape=%s batch=%s",
                tuple(action_chunk.shape),
                batch_size,
            )
            stats["debug/causal_feedback/invalid_action_shape"] = 1.0
            return None, stats, None

        module_dtype = task_tokens.dtype
        try:
            if self.causal_feedback_delta_norm is not None:
                module_dtype = self.causal_feedback_delta_norm.weight.dtype
        except Exception:
            module_dtype = task_tokens.dtype

        if hasattr(self.action_model, "build_world_action_context"):
            action_context = self.action_model.build_world_action_context(action_chunk)
        else:
            action_context = action_chunk.mean(dim=1)
        action_context = action_context.to(device=task_tokens.device, dtype=module_dtype)
        if self.causal_feedback_detach_action:
            action_context = action_context.detach()

        residual_vec = None
        residual_vec_target = None
        delta_tokens_for_stats = None
        valid = None
        if (
            self.causal_feedback_use_valid_mask
            and isinstance(valid_tk_mask, torch.Tensor)
            and valid_tk_mask.numel() == batch_size
        ):
            valid = valid_tk_mask.to(device=task_tokens.device, dtype=module_dtype).view(-1, 1, 1).clamp(0.0, 1.0)
            stats["debug/causal_feedback/valid_tk_ratio"] = float((valid > 0.5).float().mean().item())

        use_token_delta_geo = bool(self.patha_residual_mode == "token_delta_geo")
        if use_token_delta_geo:
            if (
                isinstance(geometric_tokens, torch.Tensor)
                and isinstance(geometric_tokens_next, torch.Tensor)
                and geometric_tokens.ndim == 3
                and geometric_tokens_next.ndim == 3
                and geometric_tokens.shape[0] == geometric_tokens_next.shape[0] == batch_size
                and geometric_tokens.shape[2] == geometric_tokens_next.shape[2] == hidden
            ):
                token_n = min(int(geometric_tokens.shape[1]), int(geometric_tokens_next.shape[1]))
                if (
                    isinstance(vision_tokens, torch.Tensor)
                    and vision_tokens.ndim == 3
                    and vision_tokens.shape[0] == batch_size
                    and vision_tokens.shape[2] == hidden
                ):
                    token_n = min(token_n, int(vision_tokens.shape[1]))
                if token_n > 0:
                    geo_before = geometric_tokens[:, :token_n, :].to(device=task_tokens.device, dtype=module_dtype)
                    geo_after = geometric_tokens_next[:, :token_n, :].to(device=task_tokens.device, dtype=module_dtype)
                    delta_tokens = geo_after - geo_before
                    if self.causal_feedback_detach_delta:
                        delta_tokens = delta_tokens.detach()
                    alpha, soft_stats = self._build_soft_mask(
                        vision_tokens=vision_tokens[:, :token_n, :]
                        if isinstance(vision_tokens, torch.Tensor) and vision_tokens.ndim == 3
                        else None,
                        geometric_tokens=geo_before,
                        language_queries=language_queries,
                        language_query_mask=language_query_mask,
                    )
                    stats.update(soft_stats)
                    if not isinstance(alpha, torch.Tensor):
                        alpha = torch.full(
                            (batch_size, token_n),
                            1.0 / max(token_n, 1),
                            device=delta_tokens.device,
                            dtype=delta_tokens.dtype,
                        )
                    else:
                        alpha = alpha.to(device=delta_tokens.device, dtype=delta_tokens.dtype)
                        if alpha.shape[0] != batch_size or alpha.shape[1] != token_n:
                            alpha = torch.full(
                                (batch_size, token_n),
                                1.0 / max(token_n, 1),
                                device=delta_tokens.device,
                                dtype=delta_tokens.dtype,
                            )
                            stats["debug/causal_feedback/soft_mask_shape_mismatch"] = 1.0
                    if isinstance(valid, torch.Tensor):
                        delta_tokens = delta_tokens * valid
                    residual_tokens = delta_tokens * alpha.unsqueeze(-1)
                    residual_vec = residual_tokens.sum(dim=1)
                    residual_vec_target = residual_vec
                    delta_tokens_for_stats = delta_tokens
                    stats["debug/causal_feedback/residual_mode_used_token_delta_geo"] = 1.0
                    stats["debug/causal_feedback/residual_token_num"] = float(token_n)
                else:
                    stats["debug/causal_feedback/token_delta_geo_empty"] = 1.0
            else:
                stats["debug/causal_feedback/token_delta_geo_unavailable"] = 1.0

        if residual_vec is None:
            z_before, slot_w_before, used_cond_before = self._pool_task_tokens_with_action_context(
                task_tokens=task_tokens,
                action_context=action_context,
            )
            z_after, slot_w_after, used_cond_after = self._pool_task_tokens_with_action_context(
                task_tokens=task_tokens_next,
                action_context=action_context,
            )
            used_action_cond_pool = bool(used_cond_before and used_cond_after)
            stats["debug/causal_feedback/uses_action_slot_pooling"] = float(
                hasattr(self.action_model, "pool_task_tokens")
            )
            stats["debug/causal_feedback/action_conditioned_pooling"] = 1.0 if used_action_cond_pool else 0.0

            if isinstance(slot_w_before, torch.Tensor):
                with torch.no_grad():
                    entropy = -torch.sum(slot_w_before * torch.log(slot_w_before.clamp_min(1e-9)), dim=1)
                    stats["debug/causal_feedback/slot_entropy_before"] = float(entropy.mean().item())
            if isinstance(slot_w_after, torch.Tensor):
                with torch.no_grad():
                    entropy = -torch.sum(slot_w_after * torch.log(slot_w_after.clamp_min(1e-9)), dim=1)
                    stats["debug/causal_feedback/slot_entropy_after"] = float(entropy.mean().item())

            residual_vec = z_after - z_before
            if self.causal_feedback_detach_delta:
                residual_vec = residual_vec.detach()
            if isinstance(valid, torch.Tensor):
                residual_vec = residual_vec * valid.view(batch_size, 1)
            residual_vec_target = residual_vec
            stats["debug/causal_feedback/residual_mode_used_pooled_delta_z"] = 1.0

        # Residual direction stability probe: low-frequency cosine between
        # consecutive batch-mean deltas.
        with torch.no_grad():
            try:
                self._causal_feedback_delta_step += 1
                curr_delta_mean = residual_vec.detach().float().mean(dim=0)
                prev_delta_mean = self._causal_feedback_prev_delta_mean
                should_log_cos = (
                    self.causal_feedback_delta_cos_interval <= 1
                    or (self._causal_feedback_delta_step % self.causal_feedback_delta_cos_interval == 0)
                )
                if (
                    should_log_cos
                    and isinstance(prev_delta_mean, torch.Tensor)
                    and prev_delta_mean.shape == curr_delta_mean.shape
                ):
                    cos_prev = F.cosine_similarity(
                        curr_delta_mean.unsqueeze(0),
                        prev_delta_mean.to(device=curr_delta_mean.device).unsqueeze(0),
                        dim=-1,
                    )
                    stats["debug/causal_feedback/delta_mean_cos_prev"] = float(cos_prev.item())
                    stats["debug/causal_feedback/delta_mean_cos_prev_available"] = 1.0
                elif should_log_cos:
                    stats["debug/causal_feedback/delta_mean_cos_prev_available"] = 0.0
                self._causal_feedback_prev_delta_mean = curr_delta_mean.detach().cpu()
            except Exception:
                stats["debug/causal_feedback/delta_mean_cos_prev_error"] = 1.0
        # Feedback submodules may stay in fp32 while task tokens can be bf16/fp16
        # during inference eval. Run this branch in module parameter dtype to avoid
        # Float/BFloat16 mismatches (e.g., LayerNorm expected Float).
        residual_vec = residual_vec.to(dtype=module_dtype)
        residual_vec_target = residual_vec_target.to(dtype=module_dtype)

        delta_feat = self.causal_feedback_delta_norm(residual_vec)
        action_feat = self.causal_feedback_action_norm(self.causal_feedback_action_proj(action_context))
        fused = torch.cat((delta_feat, action_feat), dim=-1)
        feedback_flat = self.causal_feedback_fuse(fused)
        feedback_flat = self.causal_feedback_dropout(feedback_flat)
        feedback_tokens = feedback_flat.view(
            batch_size,
            self.causal_feedback_token_num,
            self._causal_feedback_hidden_size,
        )
        if self.causal_feedback_scale != 1.0:
            feedback_tokens = feedback_tokens * float(self.causal_feedback_scale)

        if isinstance(valid, torch.Tensor):
            valid = valid.to(device=feedback_tokens.device, dtype=feedback_tokens.dtype)
            feedback_tokens = feedback_tokens * valid
            residual_vec_target = residual_vec_target * valid.view(batch_size, 1)

        # Keep external contract: downstream branches expect feedback aligned to task token dtype.
        feedback_tokens = feedback_tokens.to(dtype=task_tokens.dtype)
        residual_vec_target = residual_vec_target.to(dtype=task_tokens.dtype)

        with torch.no_grad():
            residual_norm = residual_vec_target.detach().norm(dim=-1)
            p50, p95 = self._masked_quantiles_1d(
                residual_norm.view(-1),
                valid_tk_mask if isinstance(valid_tk_mask, torch.Tensor) else None,
                quantiles=(0.50, 0.95),
            )
            stats.update(
                {
                    "debug/causal_feedback/applied": 1.0,
                    "debug/causal_feedback/token_num": float(feedback_tokens.shape[1]),
                    "debug/causal_feedback/token_hidden": float(feedback_tokens.shape[2]),
                    "debug/causal_feedback/delta_z_norm_mean": float(
                        residual_norm.mean().item()
                    ),
                    "debug/causal_feedback/delta_z_norm_p50": float(p50 if p50 is not None else 0.0),
                    "debug/causal_feedback/delta_z_norm_p95": float(p95 if p95 is not None else 0.0),
                    "debug/causal_feedback/token_norm_mean": float(
                        feedback_tokens.detach().norm(dim=-1).mean().item()
                    ),
                    "debug/causal_feedback/token_absmax": float(
                        feedback_tokens.detach().abs().max().item()
                    ),
                }
            )
            if isinstance(delta_tokens_for_stats, torch.Tensor):
                token_delta_norm = delta_tokens_for_stats.detach().norm(dim=-1)
                stats["debug/causal_feedback/delta_geo_token_norm_mean"] = float(token_delta_norm.mean().item())
                token_p50, token_p95 = self._masked_quantiles_1d(
                    token_delta_norm.mean(dim=-1).view(-1),
                    valid_tk_mask if isinstance(valid_tk_mask, torch.Tensor) else None,
                    quantiles=(0.50, 0.95),
                )
                stats["debug/causal_feedback/delta_geo_token_norm_p50"] = float(
                    token_p50 if token_p50 is not None else 0.0
                )
                stats["debug/causal_feedback/delta_geo_token_norm_p95"] = float(
                    token_p95 if token_p95 is not None else 0.0
                )
        return feedback_tokens, stats, residual_vec_target

    def _compute_causal_feedback_aux_loss(
        self,
        *,
        feedback_tokens: Optional[torch.Tensor],
        task_tokens: Optional[torch.Tensor],
        task_tokens_next: Optional[torch.Tensor],
        valid_tk_mask: Optional[torch.Tensor],
        action_chunk: Optional[torch.Tensor] = None,
        residual_target: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        stats: Dict[str, float] = {
            "debug/causal_feedback/loss_fb_enabled": float(
                self._causal_feedback_ready and self.causal_feedback_aux_weight > 0.0
            ),
            "debug/causal_feedback/loss_fb_active": 0.0,
            "debug/causal_feedback/loss_fb_weight": float(self.causal_feedback_aux_weight),
            "debug/causal_feedback/loss_fb_dir_weight": float(self.causal_feedback_aux_dir_weight),
        }
        if (
            not self._causal_feedback_ready
            or self.causal_feedback_recon_head is None
            or self.causal_feedback_aux_weight <= 0.0
        ):
            return None, stats
        if not (
            isinstance(feedback_tokens, torch.Tensor)
            and isinstance(task_tokens, torch.Tensor)
            and isinstance(task_tokens_next, torch.Tensor)
        ):
            stats["debug/causal_feedback/loss_fb_missing_inputs"] = 1.0
            return None, stats
        if task_tokens.shape != task_tokens_next.shape:
            stats["debug/causal_feedback/loss_fb_shape_mismatch"] = 1.0
            return None, stats
        if feedback_tokens.ndim == 2:
            feedback_tokens = feedback_tokens.unsqueeze(1)
        if feedback_tokens.ndim != 3 or task_tokens.ndim != 3:
            stats["debug/causal_feedback/loss_fb_invalid_ndim"] = 1.0
            return None, stats
        if feedback_tokens.shape[0] != task_tokens.shape[0]:
            stats["debug/causal_feedback/loss_fb_batch_mismatch"] = 1.0
            return None, stats

        delta_z_target = None
        if isinstance(residual_target, torch.Tensor):
            delta_z_target = residual_target
            stats["debug/causal_feedback/loss_fb_target_from_residual"] = 1.0
        else:
            action_context = None
            if isinstance(action_chunk, torch.Tensor):
                try:
                    if action_chunk.ndim == 3 and action_chunk.shape[0] == task_tokens.shape[0]:
                        if hasattr(self.action_model, "build_world_action_context"):
                            action_context = self.action_model.build_world_action_context(action_chunk)
                        else:
                            action_context = action_chunk.mean(dim=1)
                        if self.causal_feedback_detach_action and isinstance(action_context, torch.Tensor):
                            action_context = action_context.detach()
                    else:
                        stats["debug/causal_feedback/loss_fb_action_shape_mismatch"] = 1.0
                except Exception:
                    stats["debug/causal_feedback/loss_fb_action_context_error"] = 1.0
                    action_context = None

            z_before, _, used_cond_before = self._pool_task_tokens_with_action_context(
                task_tokens=task_tokens,
                action_context=action_context,
            )
            z_after, _, used_cond_after = self._pool_task_tokens_with_action_context(
                task_tokens=task_tokens_next,
                action_context=action_context,
            )
            stats["debug/causal_feedback/loss_fb_action_conditioned_pooling"] = float(
                used_cond_before and used_cond_after
            )
            delta_z_target = z_after - z_before
            stats["debug/causal_feedback/loss_fb_target_from_residual"] = 0.0
        if self.causal_feedback_aux_detach_target:
            delta_z_target = delta_z_target.detach()

        feedback_summary = feedback_tokens.mean(dim=1)
        recon_dtype = feedback_summary.dtype
        try:
            recon_dtype = next(self.causal_feedback_recon_head.parameters()).dtype
        except Exception:
            recon_dtype = feedback_summary.dtype
        pred_delta = self.causal_feedback_recon_head(feedback_summary.to(dtype=recon_dtype))
        target_delta = delta_z_target.to(device=pred_delta.device, dtype=pred_delta.dtype)
        per_sample_mse = (pred_delta - target_delta).pow(2).mean(dim=-1)
        eps = max(float(self.causal_feedback_aux_dir_eps), 1e-12)
        pred_unit = F.normalize(pred_delta, dim=-1, eps=eps)
        target_unit = F.normalize(target_delta, dim=-1, eps=eps)
        per_sample_cos = torch.sum(pred_unit * target_unit, dim=-1).clamp(-1.0, 1.0)
        per_sample_dir = 1.0 - per_sample_cos

        mask = None
        if isinstance(valid_tk_mask, torch.Tensor):
            mask = valid_tk_mask.view(-1).to(device=pred_delta.device, dtype=pred_delta.dtype)
        loss_fb_mse = self._masked_mean(per_sample_mse, mask)
        loss_fb_dir = self._masked_mean(per_sample_dir, mask)
        loss_fb = loss_fb_mse + float(self.causal_feedback_aux_dir_weight) * loss_fb_dir

        with torch.no_grad():
            stats.update(
                {
                    "debug/causal_feedback/loss_fb_active": 1.0,
                    "debug/causal_feedback/loss_fb_mse": float(loss_fb_mse.detach().item()),
                    "debug/causal_feedback/loss_fb_dir": float(loss_fb_dir.detach().item()),
                    "debug/causal_feedback/loss_fb": float(loss_fb.detach().item()),
                    "debug/causal_feedback/cos_delta_mean": float(per_sample_cos.detach().mean().item()),
                    "debug/causal_feedback/delta_hat_norm_mean": float(
                        pred_delta.detach().norm(dim=-1).mean().item()
                    ),
                    "debug/causal_feedback/delta_target_norm_mean": float(
                        target_delta.detach().norm(dim=-1).mean().item()
                    ),
                }
            )
        return loss_fb, stats

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        self._debug_last_feedback_tokens_for_grad = None
        self._debug_last_task_tokens_for_grad = None
        feedback_ablation_mode = self._resolve_feedback_ablation_mode(
            kwargs.get("feedback_ablation_mode", "none")
        )
        feedback_disable_aux_loss = self._parse_bool_flag(
            kwargs.get("feedback_disable_aux_loss", False), default=False
        )
        feedback_ablation_seed = kwargs.get("feedback_ablation_seed", None)
        if feedback_ablation_seed is not None:
            try:
                feedback_ablation_seed = int(feedback_ablation_seed)
            except Exception:
                feedback_ablation_seed = None

        # Prefer explicit temporal keys when available; fall back to legacy keys.
        batch_images_t = [example.get("image_t", example["image"]) for example in examples]
        batch_images_tk = [
            example.get("image_tk", example.get("image_t", example["image"])) for example in examples
        ]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        if "state" in examples[0] or "state_t" in examples[0]:
            state = [example.get("state_t", example.get("state")) for example in examples]
        else:
            state = None
        has_temporal_images = any("image_tk" in example for example in examples)
        valid_tk_values = []
        for example in examples:
            try:
                valid_tk_values.append(float(example.get("valid_tk", 1.0)))
            except Exception:
                valid_tk_values.append(1.0)
        configured_task_token_num = None
        try:
            ma_cfg = getattr(getattr(self.config, "framework", None), "mapanything_llava3d", None)
            if ma_cfg is not None and hasattr(ma_cfg, "task_token_num"):
                configured_task_token_num = int(getattr(ma_cfg, "task_token_num"))
        except Exception:
            configured_task_token_num = None
        if configured_task_token_num is not None and configured_task_token_num < 1:
            configured_task_token_num = None
        vlm_t_ms = 0.0
        vlm_tk_ms = 0.0
        temporal_sync_ms = 0.0
        action_head_ms = 0.0
        forward_timer = self._start_step_timer()

        if not hasattr(self, "_debug_logged_instructions"):
            try:
                logger.info(f"[debug_instructions] batch_size={len(instructions)} example0={instructions[0]}")
            except Exception:
                pass
            self._debug_logged_instructions = 1
        elif self._debug_logged_instructions < 5:
            try:
                logger.info(f"[debug_instructions] batch_size={len(instructions)} example0={instructions[0]}")
            except Exception:
                pass
            self._debug_logged_instructions += 1

        vlm_inputs = self.mapanythingllava3d_vlm_interface.build_mapanythingllava3d_inputs(
            images=batch_images_t,
            instructions=instructions,
        )
        vlm_t_timer = self._start_step_timer()
        task_tokens_next = None
        valid_tk_tensor = None
        geometric_tokens = None
        geometric_tokens_next = None
        vision_tokens = None
        vision_tokens_next = None
        language_queries = None
        language_query_mask = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.mapanythingllava3d_vlm_interface(
                **vlm_inputs,
                use_cache=self.vlm_use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = vlm_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            total_layers = len(all_hidden) - 1
            if expected_layers > total_layers:
                raise ValueError(f"expected_layers={expected_layers} greater than available vlm layers={total_layers}")
            if self.vl_layer_selection == "first":
                indices = range(1, 1 + expected_layers)
            else:
                indices = range(len(all_hidden) - expected_layers, len(all_hidden))
            vl_embs_list = [all_hidden[i] for i in indices]
            if self.normalize_vl_hidden:
                vl_embs_list = [F.layer_norm(h, h.shape[-1:]) for h in vl_embs_list]
            base_hidden = vl_embs_list[-1]
            task_tokens = getattr(vlm_outputs, "task_hidden_states", None)
            if isinstance(task_tokens, torch.Tensor):
                task_tokens = task_tokens.to(device=base_hidden.device, dtype=base_hidden.dtype)
                if task_tokens.ndim != 3:
                    raise ValueError(
                        f"`task_hidden_states` must be [B,K,H], got shape={tuple(task_tokens.shape)}"
                    )
                if configured_task_token_num is not None and task_tokens.shape[1] != configured_task_token_num:
                    raise ValueError(
                        f"`task_hidden_states` token count mismatch: got K={task_tokens.shape[1]}, "
                        f"expected configured K={configured_task_token_num}"
                    )
            geom_out = getattr(vlm_outputs, "geometric_hidden_states", None)
            if isinstance(geom_out, torch.Tensor):
                geometric_tokens = geom_out.to(device=base_hidden.device, dtype=base_hidden.dtype)
            vis_out = getattr(vlm_outputs, "vision_hidden_states", None)
            if isinstance(vis_out, torch.Tensor):
                vision_tokens = vis_out.to(device=base_hidden.device, dtype=base_hidden.dtype)
            lang_q_out = getattr(vlm_outputs, "language_queries", None)
            if isinstance(lang_q_out, torch.Tensor):
                language_queries = lang_q_out.to(device=base_hidden.device, dtype=base_hidden.dtype)
            lang_q_mask_out = getattr(vlm_outputs, "language_query_mask", None)
            if isinstance(lang_q_mask_out, torch.Tensor):
                language_query_mask = lang_q_mask_out.to(device=base_hidden.device, dtype=torch.bool)
            attention_mask = vlm_inputs.get("attention_mask", None)
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device=base_hidden.device)
            valid_tk_tensor = torch.tensor(
                valid_tk_values,
                device=base_hidden.device,
                dtype=base_hidden.dtype,
            ).clamp(0.0, 1.0)
            vlm_t_ms = self._stop_step_timer(vlm_t_timer, reference=base_hidden)
            debug_metrics = {}
            try:
                try:
                    h = base_hidden.detach().float()
                    debug_metrics["debug/vl_norm/base_hidden_mean"] = float(h.mean().item())
                    debug_metrics["debug/vl_norm/base_hidden_std"] = float(h.std().item())
                    debug_metrics["debug/vl_norm/base_hidden_rms"] = float((h * h).mean().sqrt().item())
                except Exception:
                    debug_metrics = debug_metrics
                vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
                if geometric_tokens is None:
                    geo_cached = getattr(vlm_core, "_last_geometric_projected", None)
                    if isinstance(geo_cached, torch.Tensor):
                        geometric_tokens = geo_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
                if vision_tokens is None:
                    vis_cached = getattr(vlm_core, "_last_vision_features", None)
                    if isinstance(vis_cached, torch.Tensor):
                        vision_tokens = vis_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
                if language_queries is None:
                    lang_q_cached = getattr(vlm_core, "_last_language_queries", None)
                    if isinstance(lang_q_cached, torch.Tensor):
                        language_queries = lang_q_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
                if language_query_mask is None:
                    lang_qm_cached = getattr(vlm_core, "_last_language_query_mask", None)
                    if isinstance(lang_qm_cached, torch.Tensor):
                        language_query_mask = lang_qm_cached.to(device=base_hidden.device, dtype=torch.bool)
                geom_stats = getattr(vlm_core, "geom_feature_stats", None)
                geom_model_forward_ms = getattr(vlm_core, "debug_last_geom_model_forward_ms", None)
                geom_feature_extract_ms = getattr(vlm_core, "debug_last_geom_feature_extract_ms", None)
                vision_token_std_mean = self._token_std_mean(getattr(vlm_core, "_last_vision_features", None))
                if vision_token_std_mean is not None:
                    debug_metrics["debug/module1/vision_token_std_mean"] = vision_token_std_mean
                geo_token_std_mean = self._token_std_mean(getattr(vlm_core, "_last_geometric_projected", None))
                if geo_token_std_mean is not None:
                    debug_metrics["debug/module1/geo_token_std_mean"] = geo_token_std_mean
                if isinstance(geom_model_forward_ms, (int, float)):
                    debug_metrics["debug/timing/geom_model_forward_ms"] = float(geom_model_forward_ms)
                if isinstance(geom_feature_extract_ms, (int, float)):
                    debug_metrics["debug/timing/geom_feature_extract_ms"] = float(geom_feature_extract_ms)
                if isinstance(geom_model_forward_ms, (int, float)) or isinstance(geom_feature_extract_ms, (int, float)):
                    gmf = float(geom_model_forward_ms) if isinstance(geom_model_forward_ms, (int, float)) else 0.0
                    gfe = float(geom_feature_extract_ms) if isinstance(geom_feature_extract_ms, (int, float)) else 0.0
                    debug_metrics["debug/timing/perception_ms"] = float(gfe if gfe > 0.0 else gmf)
                if isinstance(geom_stats, list) and geom_stats:
                    by_tag = {}
                    for record in reversed(geom_stats):
                        tag = str(record.get("tag", ""))
                        if tag and tag not in by_tag:
                            by_tag[tag] = record
                        if len(by_tag) >= 3:
                            break
                    for tag, rec in by_tag.items():
                        prefix = f"debug/geom/{tag}"
                        debug_metrics[f"{prefix}_mean"] = float(rec.get("mean", 0.0))
                        debug_metrics[f"{prefix}_std"] = float(rec.get("std", 0.0))
                        debug_metrics[f"{prefix}_min"] = float(rec.get("min", 0.0))
                        debug_metrics[f"{prefix}_max"] = float(rec.get("max", 0.0))
                health_trace = getattr(vlm_core, "debug_health_trace", None)
                first_nonfinite = getattr(vlm_core, "debug_first_nonfinite_stage", None)
                first_nonfinite_record = getattr(vlm_core, "debug_first_nonfinite_record", None)
                debug_metrics["debug/health/has_nonfinite"] = 1.0 if first_nonfinite else 0.0
                debug_metrics["debug/health/first_nonfinite_present"] = (
                    1.0 if isinstance(first_nonfinite_record, dict) else 0.0
                )
                if isinstance(first_nonfinite_record, dict):
                    if "finite_ratio" in first_nonfinite_record:
                        debug_metrics["debug/health/first_nonfinite_finite_ratio"] = float(
                            first_nonfinite_record["finite_ratio"]
                        )
                    if "nan_count" in first_nonfinite_record:
                        debug_metrics["debug/health/first_nonfinite_nan_count"] = float(
                            first_nonfinite_record["nan_count"]
                        )
                    if "inf_count" in first_nonfinite_record:
                        debug_metrics["debug/health/first_nonfinite_inf_count"] = float(
                            first_nonfinite_record["inf_count"]
                        )
                    if "absmax" in first_nonfinite_record:
                        debug_metrics["debug/health/first_nonfinite_absmax"] = float(
                            first_nonfinite_record["absmax"]
                        )
                if isinstance(health_trace, list):
                    debug_metrics["debug/health/trace_len"] = float(len(health_trace))
                    if first_nonfinite:
                        first_idx = -1
                        for idx, rec in enumerate(health_trace):
                            if not isinstance(rec, dict):
                                continue
                            if str(rec.get("stage", "")) == str(first_nonfinite) and float(rec.get("finite_ratio", 1.0)) < 1.0:
                                first_idx = idx
                                break
                        if first_idx >= 0:
                            debug_metrics["debug/health/first_nonfinite_index"] = float(first_idx)
                    for local_idx, rec in enumerate(health_trace[-6:]):
                        if not isinstance(rec, dict):
                            continue
                        stage = self._safe_metric_key(rec.get("stage", f"stage_{local_idx}"))
                        prefix = f"debug/health/{local_idx}_{stage}"
                        if "finite_ratio" in rec:
                            debug_metrics[f"{prefix}_finite_ratio"] = float(rec["finite_ratio"])
                        if "absmax" in rec:
                            debug_metrics[f"{prefix}_absmax"] = float(rec["absmax"])
                first_indices = range(1, 1 + expected_layers)
                last_indices = range(len(all_hidden) - expected_layers, len(all_hidden))
                for rel_idx, layer_idx in enumerate(first_indices):
                    h = all_hidden[layer_idx].detach().float()
                    norm = h.view(-1).norm().item()
                    debug_metrics[f"debug/vlm_first/layer_{rel_idx}_norm"] = norm
                for rel_idx, layer_idx in enumerate(last_indices):
                    h = all_hidden[layer_idx].detach().float()
                    norm = h.view(-1).norm().item()
                    debug_metrics[f"debug/vlm_last/layer_{rel_idx}_norm"] = norm
                try:
                    input_ids = vlm_inputs.get("input_ids", None)
                    image_token_index = vlm_inputs.get("image_token_index", None)
                    if input_ids is not None and image_token_index is not None and input_ids.ndim == 2:
                        img_mask = input_ids == image_token_index
                        per_sample = img_mask.sum(dim=1)
                        debug_metrics["debug/input/num_image_tokens_total"] = float(per_sample.sum().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_mean"] = float(per_sample.float().mean().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_min"] = float(per_sample.min().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_max"] = float(per_sample.max().item())
                except Exception:
                    debug_metrics = debug_metrics
            except Exception:
                debug_metrics = debug_metrics

            if has_temporal_images:
                sync_timer = self._start_step_timer(device=base_hidden.device)
                tk_fastpath_used = 0.0
                try:
                    vlm_inputs_next = self.mapanythingllava3d_vlm_interface.build_mapanythingllava3d_inputs(
                        images=batch_images_tk,
                        instructions=instructions,
                    )
                    vlm_tk_timer = self._start_step_timer(device=base_hidden.device)
                    with torch.no_grad():
                        if hasattr(self.mapanythingllava3d_vlm_interface, "extract_task_tokens"):
                            try:
                                vlm_outputs_next = self.mapanythingllava3d_vlm_interface.extract_task_tokens(
                                    **vlm_inputs_next,
                                    return_dict=True,
                                )
                                tk_fastpath_used = 1.0
                            except Exception as fast_exc:
                                logger.warning(
                                    "[temporal_supervision] fast tk path failed (%s), fallback to full forward.",
                                    fast_exc,
                                )
                                vlm_outputs_next = self.mapanythingllava3d_vlm_interface(
                                    **vlm_inputs_next,
                                    use_cache=self.vlm_use_cache,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    return_dict=True,
                                )
                        else:
                            vlm_outputs_next = self.mapanythingllava3d_vlm_interface(
                                **vlm_inputs_next,
                                use_cache=self.vlm_use_cache,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                    vlm_tk_ms = self._stop_step_timer(vlm_tk_timer, reference=base_hidden)
                    task_tokens_next = getattr(vlm_outputs_next, "task_hidden_states", None)
                    geom_out_next = getattr(vlm_outputs_next, "geometric_hidden_states", None)
                    if isinstance(geom_out_next, torch.Tensor):
                        geometric_tokens_next = geom_out_next.to(
                            device=base_hidden.device,
                            dtype=base_hidden.dtype,
                        )
                    vis_out_next = getattr(vlm_outputs_next, "vision_hidden_states", None)
                    if isinstance(vis_out_next, torch.Tensor):
                        vision_tokens_next = vis_out_next.to(
                            device=base_hidden.device,
                            dtype=base_hidden.dtype,
                        )
                    if isinstance(task_tokens_next, torch.Tensor):
                        task_tokens_next = task_tokens_next.to(
                            device=base_hidden.device,
                            dtype=base_hidden.dtype,
                        )
                        if task_tokens_next.ndim != 3:
                            raise ValueError(
                                f"`task_tokens_next` must be [B,K,H], got shape={tuple(task_tokens_next.shape)}"
                            )
                        if isinstance(task_tokens, torch.Tensor) and task_tokens_next.shape[0] != task_tokens.shape[0]:
                            logger.warning(
                                f"[temporal_supervision] batch mismatch for task_tokens_next: "
                                f"{tuple(task_tokens_next.shape)} vs {tuple(task_tokens.shape)}; drop next tokens."
                            )
                            task_tokens_next = None
                            geometric_tokens_next = None
                            vision_tokens_next = None
                        if (
                            isinstance(task_tokens, torch.Tensor)
                            and isinstance(task_tokens_next, torch.Tensor)
                            and (
                                task_tokens_next.shape[1] != task_tokens.shape[1]
                                or task_tokens_next.shape[2] != task_tokens.shape[2]
                            )
                        ):
                            logger.warning(
                                "[temporal_supervision] task token shape mismatch t vs t+k: %s vs %s; drop next tokens.",
                                tuple(task_tokens.shape),
                                tuple(task_tokens_next.shape),
                            )
                            task_tokens_next = None
                            geometric_tokens_next = None
                            vision_tokens_next = None
                    if geometric_tokens_next is None:
                        geo_cached_next = getattr(self.mapanythingllava3d_vlm_interface.model, "_last_geometric_projected", None)
                        if isinstance(geo_cached_next, torch.Tensor):
                            geometric_tokens_next = geo_cached_next.to(
                                device=base_hidden.device,
                                dtype=base_hidden.dtype,
                            )
                    if vision_tokens_next is None:
                        vis_cached_next = getattr(self.mapanythingllava3d_vlm_interface.model, "_last_vision_features", None)
                        if isinstance(vis_cached_next, torch.Tensor):
                            vision_tokens_next = vis_cached_next.to(
                                device=base_hidden.device,
                                dtype=base_hidden.dtype,
                            )
                except Exception as exc:
                    logger.warning(f"[temporal_supervision] failed to build next-step task tokens: {exc}")
                    task_tokens_next = None
                    geometric_tokens_next = None
                    vision_tokens_next = None
                temporal_sync_ms = self._stop_step_timer(sync_timer, reference=base_hidden)
                debug_metrics["debug/timing/tk_fastpath_used"] = float(tk_fastpath_used)

            debug_metrics["debug/temporal/has_image_tk"] = float(has_temporal_images)
            debug_metrics["debug/temporal/valid_tk_ratio"] = float(valid_tk_tensor.mean().item())
            debug_metrics["debug/temporal/task_tokens_next_available"] = float(
                isinstance(task_tokens_next, torch.Tensor)
            )
            debug_metrics["debug/temporal/geo_tokens_available"] = float(
                isinstance(geometric_tokens, torch.Tensor)
            )
            debug_metrics["debug/temporal/geo_tokens_next_available"] = float(
                isinstance(geometric_tokens_next, torch.Tensor)
            )
            debug_metrics["debug/temporal/vision_tokens_available"] = float(
                isinstance(vision_tokens, torch.Tensor)
            )
            debug_metrics["debug/temporal/language_queries_available"] = float(
                isinstance(language_queries, torch.Tensor)
            )
            if isinstance(task_tokens, torch.Tensor):
                debug_metrics["debug/temporal/task_token_num"] = float(task_tokens.shape[1])
                debug_metrics["debug/temporal/task_token_hidden_size"] = float(task_tokens.shape[2])
            if configured_task_token_num is not None:
                debug_metrics["debug/temporal/task_token_num_config"] = float(configured_task_token_num)
            try:
                control_hz = float(self._infer_control_hz())
                action_horizon = int(getattr(self.config.framework.action_model, "action_horizon", self.future_action_window_size + 1))
                chunk_execution_ms = float(1000.0 * float(action_horizon) / max(control_hz, 1e-6))
                debug_metrics["debug/timing/control_hz"] = float(control_hz)
                debug_metrics["debug/timing/chunk_execution_ms"] = float(chunk_execution_ms)
                if "debug/timing/perception_ms" in debug_metrics:
                    debug_metrics["debug/timing/perception_over_chunk"] = float(
                        float(debug_metrics["debug/timing/perception_ms"]) / max(chunk_execution_ms, 1e-6)
                    )
            except Exception:
                pass

        action_head_timer = self._start_step_timer(device=base_hidden.device)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = 4
            if self.config is not None:
                try:
                    repeated_diffusion_steps = int(
                        getattr(self.config.framework.action_model, "repeated_diffusion_steps", repeated_diffusion_steps)
                    )
                except Exception:
                    repeated_diffusion_steps = repeated_diffusion_steps
                try:
                    if hasattr(self.config, "trainer") and hasattr(self.config.trainer, "repeated_diffusion_steps"):
                        repeated_diffusion_steps = int(self.config.trainer.repeated_diffusion_steps)
                except Exception:
                    repeated_diffusion_steps = repeated_diffusion_steps
            repeated_diffusion_steps = max(1, repeated_diffusion_steps)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            task_tokens_repeated = None
            if isinstance(task_tokens, torch.Tensor):
                task_tokens_repeated = task_tokens.repeat(repeated_diffusion_steps, 1, 1)
            task_tokens_next_repeated = None
            if isinstance(task_tokens_next, torch.Tensor):
                task_tokens_next_repeated = task_tokens_next.repeat(repeated_diffusion_steps, 1, 1)
            geometric_tokens_repeated = None
            if isinstance(geometric_tokens, torch.Tensor):
                geometric_tokens_repeated = geometric_tokens.repeat(repeated_diffusion_steps, 1, 1)
            geometric_tokens_next_repeated = None
            if isinstance(geometric_tokens_next, torch.Tensor):
                geometric_tokens_next_repeated = geometric_tokens_next.repeat(repeated_diffusion_steps, 1, 1)
            vision_tokens_repeated = None
            if isinstance(vision_tokens, torch.Tensor):
                vision_tokens_repeated = vision_tokens.repeat(repeated_diffusion_steps, 1, 1)
            language_queries_repeated = None
            if isinstance(language_queries, torch.Tensor):
                language_queries_repeated = language_queries.repeat(repeated_diffusion_steps, 1, 1)
            language_query_mask_repeated = None
            if isinstance(language_query_mask, torch.Tensor):
                language_query_mask_repeated = language_query_mask.repeat(repeated_diffusion_steps, 1)
            attention_mask_repeated = None
            if isinstance(attention_mask, torch.Tensor):
                attention_mask_repeated = attention_mask.repeat(repeated_diffusion_steps, 1)
            valid_tk_repeated = None
            if isinstance(valid_tk_tensor, torch.Tensor):
                valid_tk_repeated = valid_tk_tensor.repeat(repeated_diffusion_steps)
            if (
                isinstance(task_tokens_repeated, torch.Tensor)
                and isinstance(task_tokens_next_repeated, torch.Tensor)
                and isinstance(valid_tk_repeated, torch.Tensor)
            ):
                valid_mask = (valid_tk_repeated > 0.5).view(-1, 1, 1)
                task_tokens_next_repeated = torch.where(
                    valid_mask,
                    task_tokens_next_repeated,
                    task_tokens_repeated.detach(),
                )
                if (
                    isinstance(geometric_tokens_repeated, torch.Tensor)
                    and isinstance(geometric_tokens_next_repeated, torch.Tensor)
                    and geometric_tokens_repeated.shape == geometric_tokens_next_repeated.shape
                ):
                    geometric_tokens_next_repeated = torch.where(
                        valid_mask,
                        geometric_tokens_next_repeated,
                        geometric_tokens_repeated.detach(),
                    )
            feedback_tokens_repeated = None
            residual_target_repeated = None
            feedback_stats = {}
            feedback_aux_loss = None
            feedback_aux_stats = {}
            debug_metrics["debug/causal_feedback/ablation_mode_none"] = (
                1.0 if feedback_ablation_mode == "none" else 0.0
            )
            debug_metrics["debug/causal_feedback/ablation_mode_zero"] = (
                1.0 if feedback_ablation_mode == "zero" else 0.0
            )
            debug_metrics["debug/causal_feedback/ablation_mode_shuffle"] = (
                1.0 if feedback_ablation_mode == "shuffle" else 0.0
            )
            debug_metrics["debug/causal_feedback/ablation_disable_aux_loss"] = (
                1.0 if feedback_disable_aux_loss else 0.0
            )
            if self._causal_feedback_ready:
                feedback_tokens_repeated, feedback_stats, residual_target_repeated = self._build_causal_feedback_tokens(
                    task_tokens=task_tokens_repeated,
                    task_tokens_next=task_tokens_next_repeated,
                    action_chunk=actions_target_repeated,
                    valid_tk_mask=valid_tk_repeated,
                    geometric_tokens=geometric_tokens_repeated,
                    geometric_tokens_next=geometric_tokens_next_repeated,
                    vision_tokens=vision_tokens_repeated,
                    language_queries=language_queries_repeated,
                    language_query_mask=language_query_mask_repeated,
                )
                debug_metrics.update(feedback_stats)
                feedback_tokens_repeated, feedback_ablation_stats = self._apply_feedback_ablation(
                    feedback_tokens_repeated,
                    mode=feedback_ablation_mode,
                    seed=feedback_ablation_seed,
                )
                debug_metrics.update(feedback_ablation_stats)
                feedback_aux_loss, feedback_aux_stats = self._compute_causal_feedback_aux_loss(
                    feedback_tokens=feedback_tokens_repeated,
                    task_tokens=task_tokens_repeated,
                    task_tokens_next=task_tokens_next_repeated,
                    valid_tk_mask=valid_tk_repeated,
                    action_chunk=actions_target_repeated,
                    residual_target=residual_target_repeated,
                )
                debug_metrics.update(feedback_aux_stats)

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            try:
                if isinstance(task_tokens_repeated, torch.Tensor) and task_tokens_repeated.requires_grad:
                    task_tokens_repeated.retain_grad()
                    self._debug_last_task_tokens_for_grad = task_tokens_repeated
                if isinstance(feedback_tokens_repeated, torch.Tensor) and feedback_tokens_repeated.requires_grad:
                    feedback_tokens_repeated.retain_grad()
                    self._debug_last_feedback_tokens_for_grad = feedback_tokens_repeated
            except Exception:
                debug_metrics["debug/causal_feedback_grad/probe_retain_failed"] = 1.0

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=attention_mask_repeated,
                task_tokens=task_tokens_repeated,
                task_tokens_next=task_tokens_next_repeated,
                valid_tk=valid_tk_repeated,
                feedback_tokens=feedback_tokens_repeated,
            )
            if isinstance(feedback_aux_loss, torch.Tensor):
                debug_metrics["debug/causal_feedback/action_loss_base"] = float(action_loss.detach().item())
                if self.causal_feedback_aux_weight > 0.0 and not feedback_disable_aux_loss:
                    debug_metrics["debug/causal_feedback/loss_fb_weighted"] = float(
                        (float(self.causal_feedback_aux_weight) * feedback_aux_loss.detach()).item()
                    )
                    action_loss = action_loss + float(self.causal_feedback_aux_weight) * feedback_aux_loss
                    debug_metrics["debug/causal_feedback/action_loss_total"] = float(action_loss.detach().item())
                else:
                    debug_metrics["debug/causal_feedback/loss_fb_weighted"] = 0.0
                    debug_metrics["debug/causal_feedback/action_loss_total"] = float(action_loss.detach().item())
            try:
                layer_means = getattr(self.action_model, "_last_dit_layer_means", None)
                layer_vars = getattr(self.action_model, "_last_dit_layer_vars", None)
                loss_breakdown = getattr(self.action_model, "_last_loss_breakdown", None)
                if layer_means is not None:
                    for idx, m in enumerate(layer_means):
                        debug_metrics[f"debug/dit_layer/{idx}_mean"] = float(m)
                if layer_vars is not None:
                    for idx, v in enumerate(layer_vars):
                        debug_metrics[f"debug/dit_layer/{idx}_var"] = float(v)
                if isinstance(loss_breakdown, dict):
                    for k, v in loss_breakdown.items():
                        debug_metrics[f"debug/sagr/{k}"] = float(v)
                action_health_trace = getattr(self.action_model, "_last_health_trace", None)
                action_first_nonfinite = getattr(self.action_model, "_first_nonfinite_stage", None)
                action_first_nonfinite_record = getattr(self.action_model, "_first_nonfinite_record", None)
                debug_metrics["debug/action_health/has_nonfinite"] = 1.0 if action_first_nonfinite else 0.0
                debug_metrics["debug/action_health/first_nonfinite_present"] = (
                    1.0 if isinstance(action_first_nonfinite_record, dict) else 0.0
                )
                if isinstance(action_first_nonfinite_record, dict):
                    if "finite_ratio" in action_first_nonfinite_record:
                        debug_metrics["debug/action_health/first_nonfinite_finite_ratio"] = float(
                            action_first_nonfinite_record["finite_ratio"]
                        )
                    if "nan_count" in action_first_nonfinite_record:
                        debug_metrics["debug/action_health/first_nonfinite_nan_count"] = float(
                            action_first_nonfinite_record["nan_count"]
                        )
                    if "inf_count" in action_first_nonfinite_record:
                        debug_metrics["debug/action_health/first_nonfinite_inf_count"] = float(
                            action_first_nonfinite_record["inf_count"]
                        )
                    if "absmax" in action_first_nonfinite_record:
                        debug_metrics["debug/action_health/first_nonfinite_absmax"] = float(
                            action_first_nonfinite_record["absmax"]
                        )
                if isinstance(action_health_trace, list):
                    debug_metrics["debug/action_health/trace_len"] = float(len(action_health_trace))
                    if action_first_nonfinite:
                        first_idx = -1
                        for idx, rec in enumerate(action_health_trace):
                            if not isinstance(rec, dict):
                                continue
                            if str(rec.get("stage", "")) == str(action_first_nonfinite) and float(rec.get("finite_ratio", 1.0)) < 1.0:
                                first_idx = idx
                                break
                        if first_idx >= 0:
                            debug_metrics["debug/action_health/first_nonfinite_index"] = float(first_idx)
                    for local_idx, rec in enumerate(action_health_trace[-6:]):
                        if not isinstance(rec, dict):
                            continue
                        stage = self._safe_metric_key(rec.get("stage", f"stage_{local_idx}"))
                        prefix = f"debug/action_health/{local_idx}_{stage}"
                        if "finite_ratio" in rec:
                            debug_metrics[f"{prefix}_finite_ratio"] = float(rec["finite_ratio"])
                        if "absmax" in rec:
                            debug_metrics[f"{prefix}_absmax"] = float(rec["absmax"])
            except Exception:
                debug_metrics = debug_metrics
        action_head_ms = self._stop_step_timer(action_head_timer, reference=base_hidden)
        forward_total_ms = self._stop_step_timer(forward_timer, reference=base_hidden)
        vlm_total_ms = float(vlm_t_ms + vlm_tk_ms)
        debug_metrics["debug/timing/train_vlm_t_ms"] = float(vlm_t_ms)
        debug_metrics["debug/timing/train_vlm_tk_ms"] = float(vlm_tk_ms)
        debug_metrics["debug/timing/train_vlm_total_ms"] = float(vlm_total_ms)
        debug_metrics["debug/timing/train_temporal_sync_ms"] = float(temporal_sync_ms)
        debug_metrics["debug/timing/train_action_head_ms"] = float(action_head_ms)
        debug_metrics["debug/timing/train_forward_total_ms"] = float(forward_total_ms)
        debug_metrics["debug/timing/vlm_forward_ratio"] = float(vlm_total_ms / max(forward_total_ms, 1e-6))
        debug_metrics["debug/timing/temporal_sync_ratio"] = float(
            temporal_sync_ms / max(forward_total_ms, 1e-6)
        )
        debug_metrics["debug/timing/vlm_tk_over_vlm_t"] = float(vlm_tk_ms / max(vlm_t_ms, 1e-6))
        if "debug/timing/perception_ms" in debug_metrics:
            debug_metrics["debug/timing/geometric_latency_ms"] = float(
                debug_metrics["debug/timing/perception_ms"]
            )

        return {"action_loss": action_loss, "debug_metrics": debug_metrics}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        valid_tk_values = []
        for example in examples:
            try:
                valid_tk_values.append(float(example.get("valid_tk", 1.0)))
            except Exception:
                valid_tk_values.append(1.0)
        deterministic_seed = kwargs.get("deterministic_seed", None)
        return_debug_info = self._parse_bool_flag(kwargs.get("return_debug_info", False), default=False)
        return_feedback_tokens = self._parse_bool_flag(
            kwargs.get("return_feedback_tokens", False), default=False
        )
        requested_infer_steps = None
        if kwargs.get("num_ddim_steps", None) is not None:
            try:
                requested_infer_steps = max(1, int(kwargs.get("num_ddim_steps")))
            except Exception:
                requested_infer_steps = None
        request_reset_state = self._parse_bool_flag(kwargs.get("reset_inference_state", False), default=False)
        if request_reset_state:
            self.reset_inference_state()
        if deterministic_seed is not None:
            try:
                deterministic_seed = int(deterministic_seed)
            except Exception as exc:
                raise ValueError(f"`deterministic_seed` must be int-compatible, got {deterministic_seed!r}") from exc

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vlm_inputs = self.mapanythingllava3d_vlm_interface.build_mapanythingllava3d_inputs(images=batch_images, instructions=instructions)
        debug_info = None
        self._synchronize_cuda_for_timing()
        brain_t0 = time.perf_counter()
        self._synchronize_cuda_for_timing()
        vlm_t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.mapanythingllava3d_vlm_interface(
                **vlm_inputs,
                use_cache=self.vlm_use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = vlm_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            total_layers = len(all_hidden) - 1
            if expected_layers > total_layers:
                raise ValueError(f"expected_layers={expected_layers} greater than available vlm layers={total_layers}")
            if self.vl_layer_selection == "first":
                indices = range(1, 1 + expected_layers)
            else:
                indices = range(len(all_hidden) - expected_layers, len(all_hidden))
            vl_embs_list = [all_hidden[i] for i in indices]
            if self.normalize_vl_hidden:
                vl_embs_list = [F.layer_norm(h, h.shape[-1:]) for h in vl_embs_list]
            base_hidden = vl_embs_list[-1]
            task_tokens = getattr(vlm_outputs, "task_hidden_states", None)
            if isinstance(task_tokens, torch.Tensor):
                task_tokens = task_tokens.to(device=base_hidden.device, dtype=base_hidden.dtype)
            geometric_tokens = getattr(vlm_outputs, "geometric_hidden_states", None)
            if isinstance(geometric_tokens, torch.Tensor):
                geometric_tokens = geometric_tokens.to(device=base_hidden.device, dtype=base_hidden.dtype)
            vision_tokens = getattr(vlm_outputs, "vision_hidden_states", None)
            if isinstance(vision_tokens, torch.Tensor):
                vision_tokens = vision_tokens.to(device=base_hidden.device, dtype=base_hidden.dtype)
            language_queries = getattr(vlm_outputs, "language_queries", None)
            if isinstance(language_queries, torch.Tensor):
                language_queries = language_queries.to(device=base_hidden.device, dtype=base_hidden.dtype)
            language_query_mask = getattr(vlm_outputs, "language_query_mask", None)
            if isinstance(language_query_mask, torch.Tensor):
                language_query_mask = language_query_mask.to(device=base_hidden.device, dtype=torch.bool)
            vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
            if geometric_tokens is None:
                geo_cached = getattr(vlm_core, "_last_geometric_projected", None)
                if isinstance(geo_cached, torch.Tensor):
                    geometric_tokens = geo_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if vision_tokens is None:
                vis_cached = getattr(vlm_core, "_last_vision_features", None)
                if isinstance(vis_cached, torch.Tensor):
                    vision_tokens = vis_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if language_queries is None:
                lang_q_cached = getattr(vlm_core, "_last_language_queries", None)
                if isinstance(lang_q_cached, torch.Tensor):
                    language_queries = lang_q_cached.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if language_query_mask is None:
                lang_qm_cached = getattr(vlm_core, "_last_language_query_mask", None)
                if isinstance(lang_qm_cached, torch.Tensor):
                    language_query_mask = lang_qm_cached.to(device=base_hidden.device, dtype=torch.bool)
            force_replan = False
            rrr_entry = None
            if (
                isinstance(task_tokens, torch.Tensor)
                and isinstance(self._rrr_prev_predicted_task_tokens, torch.Tensor)
                and isinstance(self._rrr_prev_observed_task_tokens, torch.Tensor)
            ):
                prev_pred = self._rrr_prev_predicted_task_tokens.to(
                    device=task_tokens.device,
                    dtype=task_tokens.dtype,
                )
                prev_obs = self._rrr_prev_observed_task_tokens.to(
                    device=task_tokens.device,
                    dtype=task_tokens.dtype,
                )
                token_n = min(task_tokens.shape[1], prev_pred.shape[1], prev_obs.shape[1])
                if token_n > 0:
                    curr_task = task_tokens[:, :token_n]
                    pred_task = prev_pred[:, :token_n]
                    obs_task = prev_obs[:, :token_n]
                    residual = curr_task - pred_task
                    residual_norm = residual.norm(dim=-1).mean(dim=-1)
                    expected_change = self._rrr_prev_expected_change
                    if isinstance(expected_change, torch.Tensor):
                        expected_change = expected_change.to(
                            device=task_tokens.device,
                            dtype=task_tokens.dtype,
                        )
                        if expected_change.ndim == 0:
                            expected_change = expected_change.expand_as(residual_norm)
                        elif expected_change.shape[0] != residual_norm.shape[0]:
                            expected_change = expected_change.mean().expand_as(residual_norm)
                    else:
                        expected_change = (pred_task - obs_task).norm(dim=-1).mean(dim=-1)
                    rrr_tensor = residual_norm / expected_change.clamp_min(self.rrr_eps)
                    rrr_entry = float(rrr_tensor.mean().item())
                    force_replan = bool(rrr_entry > self.rrr_replan_threshold)
            if return_debug_info:
                debug_info = {
                    "deterministic_seed": deterministic_seed,
                    "instruction_preview": [str(x)[:256] for x in instructions[:2]],
                    "vl_layer_selection": str(self.vl_layer_selection),
                    "selected_vl_layer_indices": [int(i) for i in indices],
                    "vl_num_selected_layers": int(len(vl_embs_list)),
                    "rrr_threshold": float(self.rrr_replan_threshold),
                    "rrr_entry": rrr_entry,
                    "rrr_force_replan": bool(force_replan),
                }
                tokenizer = None
                processor = None
                try:
                    processor = getattr(self.mapanythingllava3d_vlm_interface, "processor", None)
                    tokenizer = getattr(processor, "tokenizer", None)
                except Exception:
                    processor = None
                    tokenizer = None

                def _safe_decode(token_ids):
                    if tokenizer is None:
                        return None
                    if not token_ids:
                        return ""
                    try:
                        return tokenizer.decode(
                            token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    except TypeError:
                        try:
                            return tokenizer.decode(token_ids, skip_special_tokens=False)
                        except Exception:
                            return None
                    except Exception:
                        return None

                def _decode_single_token(token_id):
                    if tokenizer is None:
                        return None
                    try:
                        return tokenizer.decode(
                            [int(token_id)],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    except TypeError:
                        try:
                            return tokenizer.decode([int(token_id)], skip_special_tokens=False)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        piece = tokenizer.convert_ids_to_tokens(int(token_id))
                        if isinstance(piece, str):
                            return piece
                    except Exception:
                        pass
                    return None

                def _safe_tokenize(text):
                    if tokenizer is None:
                        return None
                    if text is None:
                        return None
                    try:
                        out = tokenizer(
                            text,
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                        )
                        ids = None
                        if hasattr(out, "get"):
                            ids = out.get("input_ids", None)
                        if ids is None and hasattr(out, "input_ids"):
                            ids = out.input_ids
                        if ids is None:
                            return None
                        if isinstance(ids, torch.Tensor):
                            ids = ids.detach().to("cpu")
                            if ids.ndim >= 2:
                                ids = ids[0]
                            ids = ids.tolist()
                        elif isinstance(ids, np.ndarray):
                            ids = ids.tolist()
                        elif isinstance(ids, tuple):
                            ids = list(ids)
                        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                            ids = ids[0]
                        if not isinstance(ids, list):
                            return None
                        return [int(x) for x in ids]
                    except Exception:
                        try:
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            return [int(x) for x in ids]
                        except Exception:
                            return None

                def _infer_padding_layout(mask_row):
                    if not isinstance(mask_row, torch.Tensor) or mask_row.ndim != 1:
                        return "unknown", 0, 0
                    row = mask_row.detach().to("cpu").bool()
                    seq_len = int(row.numel())
                    if seq_len == 0:
                        return "unknown", 0, 0
                    active_idx = row.nonzero(as_tuple=False).flatten()
                    if active_idx.numel() == 0:
                        return "all_pad", seq_len, seq_len
                    first = int(active_idx[0].item())
                    last = int(active_idx[-1].item())
                    left_pad = first
                    right_pad = seq_len - 1 - last
                    contiguous = bool(row[first : last + 1].all().item())
                    if not contiguous:
                        return "mixed", left_pad, right_pad
                    if left_pad > 0 and right_pad == 0:
                        return "left", left_pad, right_pad
                    if left_pad == 0 and right_pad > 0:
                        return "right", left_pad, right_pad
                    if left_pad > 0 and right_pad > 0:
                        return "both", left_pad, right_pad
                    return "none", left_pad, right_pad

                cot_prompt = None
                try:
                    datasets_cfg = getattr(self.config, "datasets", None)
                    vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
                    if vla_cfg is not None and hasattr(vla_cfg, "CoT_prompt"):
                        cot_prompt = getattr(vla_cfg, "CoT_prompt")
                except Exception:
                    cot_prompt = None

                tokenizer_padding_side = getattr(tokenizer, "padding_side", None) if tokenizer is not None else None
                tokenizer_truncation_side = getattr(tokenizer, "truncation_side", None) if tokenizer is not None else None
                tokenizer_chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
                processor_chat_template = getattr(processor, "chat_template", None) if processor is not None else None

                image_token_text = None
                if tokenizer is not None:
                    image_token_text = getattr(tokenizer, "image_token", None)
                if image_token_text is None and processor is not None:
                    image_token_text = getattr(processor, "image_token", None)
                if image_token_text is None:
                    image_token_text = "<image>"

                tokenizer_image_token_id = getattr(tokenizer, "image_token_id", None) if tokenizer is not None else None
                processor_image_token_id = getattr(processor, "image_token_id", None) if processor is not None else None
                processor_image_token_index = getattr(processor, "image_token_index", None) if processor is not None else None
                image_token_probe_ids = _safe_tokenize(image_token_text)
                space_token_probe_ids = _safe_tokenize(" ")
                image_separator_token_id_set_probe = (
                    {int(x) for x in space_token_probe_ids}
                    if isinstance(space_token_probe_ids, list) and len(space_token_probe_ids) > 0
                    else set()
                )
                image_separator_token_id_set = set(image_separator_token_id_set_probe)
                expected_special_id = (
                    processor_image_token_id
                    if processor_image_token_id is not None
                    else tokenizer_image_token_id
                )
                if image_token_probe_ids is None:
                    image_probe_is_single = None
                else:
                    image_probe_is_single = bool(len(image_token_probe_ids) == 1)
                image_probe_matches_expected = None
                if image_probe_is_single and expected_special_id is not None:
                    try:
                        image_probe_matches_expected = bool(int(image_token_probe_ids[0]) == int(expected_special_id))
                    except Exception:
                        image_probe_matches_expected = None

                debug_info["tokenizer_padding_side"] = tokenizer_padding_side
                debug_info["tokenizer_truncation_side"] = tokenizer_truncation_side
                debug_info["tokenizer_chat_template_present"] = bool(tokenizer_chat_template)
                debug_info["tokenizer_chat_template_preview"] = (
                    str(tokenizer_chat_template)[:256] if tokenizer_chat_template else None
                )
                debug_info["processor_chat_template_present"] = bool(processor_chat_template)
                debug_info["processor_chat_template_preview"] = (
                    str(processor_chat_template)[:256] if processor_chat_template else None
                )
                debug_info["processor_image_token_joiner"] = (
                    getattr(processor, "image_token_joiner", None) if processor is not None else None
                )
                debug_info["processor_image_token_joiner_mode"] = (
                    getattr(processor, "image_token_joiner_mode", None) if processor is not None else None
                )
                debug_info["cot_prompt_present"] = bool(cot_prompt)
                debug_info["cot_prompt_preview"] = str(cot_prompt)[:256] if cot_prompt else None

                debug_info["image_token_text"] = str(image_token_text) if image_token_text is not None else None
                debug_info["tokenizer_image_token_id"] = (
                    int(tokenizer_image_token_id) if tokenizer_image_token_id is not None else None
                )
                debug_info["processor_image_token_id"] = (
                    int(processor_image_token_id) if processor_image_token_id is not None else None
                )
                debug_info["processor_image_token_index"] = (
                    int(processor_image_token_index) if processor_image_token_index is not None else None
                )
                debug_info["image_token_probe_ids"] = image_token_probe_ids
                debug_info["space_token_probe_ids"] = space_token_probe_ids
                debug_info["image_separator_token_ids"] = (
                    sorted([int(x) for x in image_separator_token_id_set]) if len(image_separator_token_id_set) > 0 else None
                )
                debug_info["image_separator_token_ids_probe"] = (
                    sorted([int(x) for x in image_separator_token_id_set_probe])
                    if len(image_separator_token_id_set_probe) > 0
                    else None
                )
                debug_info["image_separator_token_ids_inferred"] = None
                debug_info["image_token_probe_is_single_token"] = image_probe_is_single
                debug_info["image_token_probe_matches_expected_id"] = image_probe_matches_expected

                input_ids = vlm_inputs.get("input_ids", None)
                attention_mask = vlm_inputs.get("attention_mask", None)
                debug_info["action_head_uses_encoder_attention_mask"] = bool(isinstance(attention_mask, torch.Tensor))
                image_token_index = vlm_inputs.get("image_token_index", None)
                if isinstance(image_token_index, torch.Tensor):
                    image_token_index = int(image_token_index.detach().item())
                elif image_token_index is not None:
                    image_token_index = int(image_token_index)

                if isinstance(input_ids, torch.Tensor):
                    ids_cpu = input_ids.detach().to("cpu")
                    debug_info["input_ids_shape"] = [int(x) for x in ids_cpu.shape]
                    debug_info["input_ids_head"] = ids_cpu[:, :16].tolist()

                    if isinstance(attention_mask, torch.Tensor):
                        active_cpu = attention_mask.detach().to("cpu").bool()
                    else:
                        active_cpu = torch.ones_like(ids_cpu, dtype=torch.bool)

                    if image_token_index is not None:
                        image_mask_cpu = (ids_cpu == image_token_index) & active_cpu
                    else:
                        image_mask_cpu = torch.zeros_like(ids_cpu, dtype=torch.bool)
                    lang_mask_raw_cpu = (~image_mask_cpu) & active_cpu

                    image_left_neighbor = torch.zeros_like(image_mask_cpu, dtype=torch.bool)
                    image_right_neighbor = torch.zeros_like(image_mask_cpu, dtype=torch.bool)
                    image_left_neighbor[:, 1:] = image_mask_cpu[:, :-1]
                    image_right_neighbor[:, :-1] = image_mask_cpu[:, 1:]
                    between_two_images_cpu = image_left_neighbor & image_right_neighbor

                    # Infer separator token ids from `image - sep - image` patterns when tokenizer probing is unavailable.
                    inferred_separator_token_id_set = set()
                    between_non_image_cpu = active_cpu & (~image_mask_cpu) & between_two_images_cpu
                    if bool(between_non_image_cpu.any().item()):
                        inferred_ids = ids_cpu[between_non_image_cpu].to(torch.int64).tolist()
                        inferred_separator_token_id_set = {int(x) for x in inferred_ids}
                        image_separator_token_id_set |= inferred_separator_token_id_set

                    # Exclude separator tokens used by expanded image placeholders from language debug stats.
                    image_sep_mask_cpu = torch.zeros_like(ids_cpu, dtype=torch.bool)
                    if len(image_separator_token_id_set) > 0:
                        sep_mask_cpu = torch.zeros_like(ids_cpu, dtype=torch.bool)
                        for token_id in sorted(image_separator_token_id_set):
                            sep_mask_cpu |= ids_cpu == int(token_id)
                        image_sep_mask_cpu = sep_mask_cpu & active_cpu & between_two_images_cpu

                    lang_mask_cpu = lang_mask_raw_cpu & (~image_sep_mask_cpu)
                    debug_info["image_token_index"] = image_token_index
                    debug_info["image_separator_token_ids"] = (
                        sorted([int(x) for x in image_separator_token_id_set])
                        if len(image_separator_token_id_set) > 0
                        else None
                    )
                    debug_info["image_separator_token_ids_inferred"] = (
                        sorted([int(x) for x in inferred_separator_token_id_set])
                        if len(inferred_separator_token_id_set) > 0
                        else None
                    )
                    debug_info["active_tokens_per_sample"] = [int(x) for x in active_cpu.sum(dim=1).tolist()]
                    debug_info["image_tokens_per_sample"] = [int(x) for x in image_mask_cpu.sum(dim=1).tolist()]
                    debug_info["image_separator_tokens_per_sample"] = [int(x) for x in image_sep_mask_cpu.sum(dim=1).tolist()]
                    debug_info["image_placeholder_tokens_per_sample"] = [
                        int(x) for x in (image_mask_cpu | image_sep_mask_cpu).sum(dim=1).tolist()
                    ]
                    debug_info["lang_tokens_raw_per_sample"] = [int(x) for x in lang_mask_raw_cpu.sum(dim=1).tolist()]
                    debug_info["lang_tokens_per_sample"] = [int(x) for x in lang_mask_cpu.sum(dim=1).tolist()]
                    image_present_each = [int(x) > 0 for x in image_mask_cpu.sum(dim=1).tolist()]
                    debug_info["image_token_present_in_each_sample"] = image_present_each
                    debug_info["image_token_present_all_samples"] = bool(all(image_present_each))

                    padding_modes = []
                    left_pad_counts = []
                    right_pad_counts = []
                    for row_mask in active_cpu:
                        mode, n_left, n_right = _infer_padding_layout(row_mask)
                        padding_modes.append(mode)
                        left_pad_counts.append(int(n_left))
                        right_pad_counts.append(int(n_right))
                    debug_info["padding_mode_per_sample"] = padding_modes
                    debug_info["padding_left_count_per_sample"] = left_pad_counts
                    debug_info["padding_right_count_per_sample"] = right_pad_counts
                    debug_info["padding_right_present"] = bool(any(m == "right" for m in padding_modes))
                    debug_info["padding_mixed_present"] = bool(any(m in ("mixed", "both") for m in padding_modes))
                    debug_info["padding_modes_unique"] = sorted(list(set(padding_modes)))

                    token_signatures = []
                    lang_token_ids_full = []
                    lang_token_ids_head_128 = []
                    lang_token_ids_tail_128 = []
                    lang_decoded_text_full = []
                    lang_decoded_text_head_128 = []
                    lang_decoded_text_tail_128 = []
                    lang_whitespace_tokens_per_sample = []
                    lang_non_whitespace_tokens_per_sample = []
                    lang_non_whitespace_ratio_per_sample = []
                    whitespace_token_cache = {}

                    def _is_whitespace_token_id(token_id):
                        token_id_int = int(token_id)
                        if token_id_int in whitespace_token_cache:
                            return whitespace_token_cache[token_id_int]
                        if token_id_int in image_separator_token_id_set:
                            whitespace_token_cache[token_id_int] = True
                            return True
                        piece = _decode_single_token(token_id_int)
                        if piece is None:
                            whitespace_token_cache[token_id_int] = False
                            return False
                        is_whitespace = bool(len(piece) > 0 and piece.strip() == "")
                        whitespace_token_cache[token_id_int] = is_whitespace
                        return is_whitespace

                    for row_ids, row_lang_mask in zip(ids_cpu, lang_mask_cpu):
                        lang_ids = row_ids[row_lang_mask].to(torch.int64)
                        if lang_ids.numel() == 0:
                            token_signatures.append(0)
                            lang_token_ids_full.append([])
                            lang_token_ids_head_128.append([])
                            lang_token_ids_tail_128.append([])
                            lang_decoded_text_full.append(_safe_decode([]))
                            lang_decoded_text_head_128.append(_safe_decode([]))
                            lang_decoded_text_tail_128.append(_safe_decode([]))
                            lang_whitespace_tokens_per_sample.append(0)
                            lang_non_whitespace_tokens_per_sample.append(0)
                            lang_non_whitespace_ratio_per_sample.append(0.0)
                            continue
                        weights = torch.arange(1, lang_ids.numel() + 1, dtype=torch.int64)
                        signature = int((lang_ids * weights).sum().item() % 1000000007)
                        token_signatures.append(signature)
                        lang_list = lang_ids.tolist()
                        lang_token_ids_full.append(lang_list)
                        head_ids = lang_list[:128]
                        tail_ids = lang_list[-128:]
                        lang_token_ids_head_128.append(head_ids)
                        lang_token_ids_tail_128.append(tail_ids)
                        lang_decoded_text_full.append(_safe_decode(lang_list))
                        lang_decoded_text_head_128.append(_safe_decode(head_ids))
                        lang_decoded_text_tail_128.append(_safe_decode(tail_ids))
                        whitespace_count = int(sum(1 for tok in lang_list if _is_whitespace_token_id(tok)))
                        non_whitespace_count = int(len(lang_list) - whitespace_count)
                        lang_whitespace_tokens_per_sample.append(whitespace_count)
                        lang_non_whitespace_tokens_per_sample.append(non_whitespace_count)
                        lang_non_whitespace_ratio_per_sample.append(
                            float(non_whitespace_count / max(1, len(lang_list)))
                        )
                    debug_info["lang_token_signatures"] = token_signatures
                    debug_info["lang_token_ids_full"] = lang_token_ids_full
                    debug_info["lang_token_ids_head_128"] = lang_token_ids_head_128
                    debug_info["lang_token_ids_tail_128"] = lang_token_ids_tail_128
                    debug_info["lang_decoded_text_full"] = lang_decoded_text_full
                    debug_info["lang_decoded_text_head_128"] = lang_decoded_text_head_128
                    debug_info["lang_decoded_text_tail_128"] = lang_decoded_text_tail_128
                    debug_info["lang_whitespace_tokens_per_sample"] = lang_whitespace_tokens_per_sample
                    debug_info["lang_non_whitespace_tokens_per_sample"] = lang_non_whitespace_tokens_per_sample
                    debug_info["lang_non_whitespace_ratio_per_sample"] = lang_non_whitespace_ratio_per_sample
                    debug_info["lang_non_whitespace_ratio_mean"] = (
                        float(np.mean(lang_non_whitespace_ratio_per_sample))
                        if len(lang_non_whitespace_ratio_per_sample) > 0
                        else None
                    )
                    debug_info["lang_decode_available"] = tokenizer is not None
                    debug_info["image_token_special_token_integrity_ok"] = bool(
                        debug_info.get("image_token_present_all_samples", False)
                        and debug_info.get("image_token_probe_is_single_token", False)
                        and (
                            debug_info.get("image_token_probe_matches_expected_id") in (True, None)
                        )
                    )

                    image_mask_dev = image_mask_cpu.to(device=base_hidden.device)
                    lang_mask_dev = lang_mask_cpu.to(device=base_hidden.device)
                else:
                    image_mask_dev = None
                    lang_mask_dev = None
                    debug_info["input_ids_shape"] = None
                    debug_info["input_ids_head"] = None
                    debug_info["lang_decode_available"] = tokenizer is not None
                    debug_info["image_token_present_in_each_sample"] = None
                    debug_info["image_token_present_all_samples"] = None
                    debug_info["padding_mode_per_sample"] = None
                    debug_info["padding_left_count_per_sample"] = None
                    debug_info["padding_right_count_per_sample"] = None
                    debug_info["padding_right_present"] = None
                    debug_info["padding_mixed_present"] = None
                    debug_info["image_separator_tokens_per_sample"] = None
                    debug_info["image_placeholder_tokens_per_sample"] = None
                    debug_info["lang_tokens_raw_per_sample"] = None
                    debug_info["padding_modes_unique"] = None
                    debug_info["image_token_special_token_integrity_ok"] = None
                    debug_info["lang_token_ids_full"] = None
                    debug_info["lang_whitespace_tokens_per_sample"] = None
                    debug_info["lang_non_whitespace_tokens_per_sample"] = None
                    debug_info["lang_non_whitespace_ratio_per_sample"] = None
                    debug_info["lang_non_whitespace_ratio_mean"] = None

                vl_layer_mean = []
                vl_layer_std = []
                vl_layer_rms = []
                lang_layer_rms = []
                image_layer_rms = []
                for h in vl_embs_list:
                    hs = h.detach().float()
                    vl_layer_mean.append(float(hs.mean().item()))
                    vl_layer_std.append(float(hs.std(unbiased=False).item()))
                    vl_layer_rms.append(float((hs * hs).mean().sqrt().item()))

                    if lang_mask_dev is not None and bool(lang_mask_dev.any().item()):
                        lang_weight = lang_mask_dev.unsqueeze(-1).to(dtype=hs.dtype, device=hs.device)
                        lang_count = max(1.0, float(lang_mask_dev.sum().item()))
                        lang_rms = ((hs * hs * lang_weight).sum() / (lang_count * hs.shape[-1])).sqrt()
                        lang_layer_rms.append(float(lang_rms.item()))
                    else:
                        lang_layer_rms.append(None)

                    if image_mask_dev is not None and bool(image_mask_dev.any().item()):
                        image_weight = image_mask_dev.unsqueeze(-1).to(dtype=hs.dtype, device=hs.device)
                        image_count = max(1.0, float(image_mask_dev.sum().item()))
                        image_rms = ((hs * hs * image_weight).sum() / (image_count * hs.shape[-1])).sqrt()
                        image_layer_rms.append(float(image_rms.item()))
                    else:
                        image_layer_rms.append(None)

                debug_info["vl_layer_mean"] = vl_layer_mean
                debug_info["vl_layer_std"] = vl_layer_std
                debug_info["vl_layer_rms"] = vl_layer_rms
                debug_info["lang_layer_rms"] = lang_layer_rms
                debug_info["image_layer_rms"] = image_layer_rms
        self._synchronize_cuda_for_timing(base_hidden)
        vlm_forward_ms = float((time.perf_counter() - vlm_t0) * 1000.0)

        state_tensor = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        valid_tk_tensor = torch.tensor(
            valid_tk_values,
            device=base_hidden.device,
            dtype=base_hidden.dtype,
        ).view(-1)
        feedback_tokens = kwargs.get("feedback_tokens", None)
        if feedback_tokens is not None:
            if not isinstance(feedback_tokens, torch.Tensor):
                feedback_tokens = torch.tensor(np.array(feedback_tokens), device=base_hidden.device, dtype=base_hidden.dtype)
            else:
                feedback_tokens = feedback_tokens.to(device=base_hidden.device, dtype=base_hidden.dtype)
            if feedback_tokens.ndim == 2:
                feedback_tokens = feedback_tokens.unsqueeze(1)
            if feedback_tokens.ndim != 3:
                raise ValueError(
                    f"`feedback_tokens` must be [B, Kf, H] or [B, H], got shape={tuple(feedback_tokens.shape)}"
                )
            if feedback_tokens.shape[0] != base_hidden.shape[0]:
                if feedback_tokens.shape[0] == 1:
                    feedback_tokens = feedback_tokens.expand(base_hidden.shape[0], -1, -1)
                else:
                    raise ValueError(
                        f"`feedback_tokens` batch mismatch: got {feedback_tokens.shape[0]}, expected {base_hidden.shape[0]}"
                    )
        feedback_source = "external" if isinstance(feedback_tokens, torch.Tensor) else "none"
        patha_auto_stats: Dict[str, float] = {}
        if (
            feedback_tokens is None
            and self._causal_feedback_ready
            and self.enable_causal_feedback_inference
            and isinstance(task_tokens, torch.Tensor)
            and isinstance(self._patha_prev_task_tokens, torch.Tensor)
            and isinstance(self._patha_prev_action_chunk, torch.Tensor)
        ):
            prev_task = self._patha_prev_task_tokens.to(
                device=task_tokens.device,
                dtype=task_tokens.dtype,
            )
            prev_geo = None
            if isinstance(self._patha_prev_geo_tokens, torch.Tensor):
                prev_geo = self._patha_prev_geo_tokens.to(
                    device=task_tokens.device,
                    dtype=task_tokens.dtype,
                )
            prev_vis = None
            if isinstance(self._patha_prev_vision_tokens, torch.Tensor):
                prev_vis = self._patha_prev_vision_tokens.to(
                    device=task_tokens.device,
                    dtype=task_tokens.dtype,
                )
            prev_lang_q = None
            if isinstance(self._patha_prev_language_queries, torch.Tensor):
                prev_lang_q = self._patha_prev_language_queries.to(
                    device=task_tokens.device,
                    dtype=task_tokens.dtype,
                )
            prev_lang_q_mask = None
            if isinstance(self._patha_prev_language_query_mask, torch.Tensor):
                prev_lang_q_mask = self._patha_prev_language_query_mask.to(
                    device=task_tokens.device,
                    dtype=torch.bool,
                )
            prev_actions = self._patha_prev_action_chunk.to(
                device=task_tokens.device,
                dtype=task_tokens.dtype,
            )
            if prev_task.shape[0] != task_tokens.shape[0]:
                if prev_task.shape[0] == 1:
                    prev_task = prev_task.expand(task_tokens.shape[0], -1, -1)
                else:
                    prev_task = None
            if isinstance(prev_task, torch.Tensor) and prev_actions.shape[0] != task_tokens.shape[0]:
                if prev_actions.shape[0] == 1:
                    prev_actions = prev_actions.expand(task_tokens.shape[0], -1, -1)
                else:
                    prev_task = None
            if isinstance(prev_task, torch.Tensor) and isinstance(prev_geo, torch.Tensor) and prev_geo.shape[0] != task_tokens.shape[0]:
                if prev_geo.shape[0] == 1:
                    prev_geo = prev_geo.expand(task_tokens.shape[0], -1, -1)
                else:
                    prev_geo = None
            if isinstance(prev_task, torch.Tensor) and isinstance(prev_vis, torch.Tensor) and prev_vis.shape[0] != task_tokens.shape[0]:
                if prev_vis.shape[0] == 1:
                    prev_vis = prev_vis.expand(task_tokens.shape[0], -1, -1)
                else:
                    prev_vis = None
            if isinstance(prev_task, torch.Tensor) and isinstance(prev_lang_q, torch.Tensor) and prev_lang_q.shape[0] != task_tokens.shape[0]:
                if prev_lang_q.shape[0] == 1:
                    prev_lang_q = prev_lang_q.expand(task_tokens.shape[0], -1, -1)
                else:
                    prev_lang_q = None
            if (
                isinstance(prev_task, torch.Tensor)
                and isinstance(prev_lang_q_mask, torch.Tensor)
                and prev_lang_q_mask.shape[0] != task_tokens.shape[0]
            ):
                if prev_lang_q_mask.shape[0] == 1:
                    prev_lang_q_mask = prev_lang_q_mask.expand(task_tokens.shape[0], -1)
                else:
                    prev_lang_q_mask = None
            if isinstance(prev_task, torch.Tensor):
                token_n = min(task_tokens.shape[1], prev_task.shape[1])
                if token_n > 0:
                    if isinstance(valid_tk_tensor, torch.Tensor) and valid_tk_tensor.numel() == task_tokens.shape[0]:
                        valid_mask = valid_tk_tensor.to(
                            device=task_tokens.device,
                            dtype=task_tokens.dtype,
                        ).clamp(0.0, 1.0)
                    else:
                        valid_mask = torch.ones(
                            (task_tokens.shape[0],),
                            device=task_tokens.device,
                            dtype=task_tokens.dtype,
                        )
                    feedback_tokens, patha_auto_stats, _ = self._build_causal_feedback_tokens(
                        task_tokens=prev_task[:, :token_n, :],
                        task_tokens_next=task_tokens[:, :token_n, :],
                        action_chunk=prev_actions,
                        valid_tk_mask=valid_mask,
                        geometric_tokens=prev_geo,
                        geometric_tokens_next=geometric_tokens,
                        vision_tokens=prev_vis if isinstance(prev_vis, torch.Tensor) else vision_tokens,
                        language_queries=prev_lang_q if isinstance(prev_lang_q, torch.Tensor) else language_queries,
                        language_query_mask=prev_lang_q_mask if isinstance(prev_lang_q_mask, torch.Tensor) else language_query_mask,
                    )
                    if isinstance(feedback_tokens, torch.Tensor):
                        feedback_source = "auto_path_a"

        attention_mask = vlm_inputs.get("attention_mask", None)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device=base_hidden.device)

        # Keep inference inputs dtype-aligned with action head parameters to avoid
        # Float/BFloat16 mismatches under mixed precision server settings.
        action_device = self.action_model.device
        action_dtype = self.action_model.dtype
        vl_embs_list = [h.to(device=action_device, dtype=action_dtype) for h in vl_embs_list]
        if isinstance(task_tokens, torch.Tensor):
            task_tokens = task_tokens.to(device=action_device, dtype=action_dtype)
        if isinstance(feedback_tokens, torch.Tensor):
            feedback_tokens = feedback_tokens.to(device=action_device, dtype=action_dtype)
        if isinstance(state_tensor, torch.Tensor):
            state_tensor = state_tensor.to(device=action_device, dtype=action_dtype)
        if isinstance(valid_tk_tensor, torch.Tensor):
            valid_tk_tensor = valid_tk_tensor.to(device=action_device, dtype=action_dtype)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device=action_device)

        if action_device.type == "cuda" and action_dtype in (torch.float16, torch.bfloat16):
            action_autocast_ctx = torch.autocast("cuda", dtype=action_dtype)
        else:
            action_autocast_ctx = nullcontext()

        self._synchronize_cuda_for_timing(base_hidden)
        action_t0 = time.perf_counter()
        with action_autocast_ctx:
            pred_action_output = self.action_model.predict_action(
                vl_embs_list,
                state_tensor,
                noise_seed=deterministic_seed,
                encoder_attention_mask=attention_mask,
                task_tokens=task_tokens,
                feedback_tokens=feedback_tokens,
                valid_tk=valid_tk_tensor,
                num_inference_timesteps=requested_infer_steps,
                force_replan=force_replan,
                rrr_threshold=self.rrr_replan_threshold,
                return_info=True,
            )
            if isinstance(pred_action_output, tuple):
                pred_actions, planning_info = pred_action_output
            else:
                pred_actions = pred_action_output
                planning_info = {}
        self._synchronize_cuda_for_timing(pred_actions if isinstance(pred_actions, torch.Tensor) else base_hidden)
        action_head_wall_ms = float((time.perf_counter() - action_t0) * 1000.0)
        self._synchronize_cuda_for_timing(pred_actions if isinstance(pred_actions, torch.Tensor) else base_hidden)
        brain_total_ms = float((time.perf_counter() - brain_t0) * 1000.0)

        if isinstance(task_tokens, torch.Tensor):
            self._rrr_prev_observed_task_tokens = task_tokens.detach()
            self._patha_prev_task_tokens = task_tokens.detach()
        if isinstance(geometric_tokens, torch.Tensor):
            self._patha_prev_geo_tokens = geometric_tokens.detach()
        if isinstance(vision_tokens, torch.Tensor):
            self._patha_prev_vision_tokens = vision_tokens.detach()
        if isinstance(language_queries, torch.Tensor):
            self._patha_prev_language_queries = language_queries.detach()
        if isinstance(language_query_mask, torch.Tensor):
            self._patha_prev_language_query_mask = language_query_mask.detach()
        pred_task_tokens = planning_info.get("predicted_task_tokens", None) if isinstance(planning_info, dict) else None
        if isinstance(pred_task_tokens, torch.Tensor):
            self._rrr_prev_predicted_task_tokens = pred_task_tokens.detach()
        expected_change = planning_info.get("expected_change", None) if isinstance(planning_info, dict) else None
        if isinstance(expected_change, torch.Tensor):
            self._rrr_prev_expected_change = expected_change.detach()
        if isinstance(pred_actions, torch.Tensor):
            self._patha_prev_action_chunk = pred_actions.detach()

        normalized_actions = pred_actions.detach().cpu().numpy()
        flow_total_ms = 0.0
        flow_step_ms_mean = 0.0
        flow_step_ms_max = 0.0
        flow_step_count = 0
        reflex_total_ms = 0.0
        reflex_step_ms_mean = 0.0
        reflex_step_ms_max = 0.0
        reflex_step_count = 0
        reflex_share = 0.0
        if isinstance(planning_info, dict):
            try:
                flow_total_ms = float(planning_info.get("timing/flow_total_ms", flow_total_ms))
                flow_step_ms_mean = float(planning_info.get("timing/flow_step_ms_mean", flow_step_ms_mean))
                flow_step_ms_max = float(planning_info.get("timing/flow_step_ms_max", flow_step_ms_max))
                flow_step_count = int(planning_info.get("timing/flow_step_count", flow_step_count))
                reflex_total_ms = float(planning_info.get("timing/reflex_total_ms", reflex_total_ms))
                reflex_step_ms_mean = float(planning_info.get("timing/reflex_step_ms_mean", reflex_step_ms_mean))
                reflex_step_ms_max = float(planning_info.get("timing/reflex_step_ms_max", reflex_step_ms_max))
                reflex_step_count = int(planning_info.get("timing/reflex_step_count", reflex_step_count))
                reflex_share = float(planning_info.get("timing/reflex_share", reflex_share))
            except Exception:
                pass

        vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
        geom_model_forward_ms = getattr(vlm_core, "debug_last_geom_model_forward_ms", None)
        geom_feature_extract_ms = getattr(vlm_core, "debug_last_geom_feature_extract_ms", None)
        geom_model_forward_ms = float(geom_model_forward_ms) if isinstance(geom_model_forward_ms, (int, float)) else 0.0
        geom_feature_extract_ms = float(geom_feature_extract_ms) if isinstance(geom_feature_extract_ms, (int, float)) else 0.0
        perception_ms = geom_feature_extract_ms if geom_feature_extract_ms > 0.0 else geom_model_forward_ms

        action_cfg = getattr(getattr(self.config, "framework", None), "action_model", None)
        try:
            action_horizon = int(getattr(action_cfg, "action_horizon", int(normalized_actions.shape[1])))
        except Exception:
            action_horizon = int(normalized_actions.shape[1]) if normalized_actions.ndim >= 2 else 1
        control_hz = float(self._infer_control_hz())
        chunk_execution_ms = float(1000.0 * float(action_horizon) / max(control_hz, 1e-6))
        control_cycle_ms = float(1000.0 / max(control_hz, 1e-6))

        brain_from_parts_ms = float(vlm_forward_ms + (flow_total_ms if flow_total_ms > 0.0 else action_head_wall_ms))
        perception_over_chunk = float(perception_ms / max(chunk_execution_ms, 1e-6))
        brain_over_chunk = float(brain_total_ms / max(chunk_execution_ms, 1e-6))
        reflex_over_chunk = float(reflex_total_ms / max(chunk_execution_ms, 1e-6))
        reflex_step_over_cycle = float(reflex_step_ms_mean / max(control_cycle_ms, 1e-6))

        output = {"normalized_actions": normalized_actions}
        if return_feedback_tokens:
            if isinstance(feedback_tokens, torch.Tensor):
                output["feedback_tokens"] = (
                    feedback_tokens.detach().to(dtype=torch.float32).cpu().numpy()
                )
            else:
                output["feedback_tokens"] = None
        if return_debug_info:
            if debug_info is None:
                debug_info = {}
            debug_info["state_present"] = bool(state_tensor is not None)
            debug_info["state_shape"] = list(state_tensor.shape) if state_tensor is not None else None
            debug_info["normalized_action_mean"] = float(normalized_actions.mean())
            debug_info["normalized_action_std"] = float(normalized_actions.std())
            debug_info["normalized_action_absmax"] = float(np.abs(normalized_actions).max())
            layer_means = getattr(self.action_model, "_last_dit_layer_means", None)
            layer_vars = getattr(self.action_model, "_last_dit_layer_vars", None)
            if layer_means is not None:
                debug_info["dit_layer_means"] = [float(x) for x in layer_means]
            if layer_vars is not None:
                debug_info["dit_layer_vars"] = [float(x) for x in layer_vars]
            if isinstance(planning_info, dict):
                debug_info["rrr_last"] = planning_info.get("rrr_last")
                debug_info["rrr_max"] = planning_info.get("rrr_max")
                debug_info["rrr_trace"] = planning_info.get("rrr_trace")
                debug_info["rrr_replan_count"] = planning_info.get("replan_count")
                for k, v in planning_info.items():
                    if isinstance(k, str) and k.startswith("timing/") and isinstance(v, (int, float)):
                        debug_info[k] = float(v)
                    elif isinstance(k, str) and k.startswith("geo/"):
                        if isinstance(v, (int, float)):
                            debug_info[k] = float(v)
                        elif isinstance(v, list):
                            try:
                                debug_info[k] = [float(x) for x in v]
                            except Exception:
                                debug_info[k] = v
            debug_info["timing/vlm_forward_ms"] = float(vlm_forward_ms)
            debug_info["timing/action_head_wall_ms"] = float(action_head_wall_ms)
            debug_info["timing/brain_total_ms"] = float(brain_total_ms)
            debug_info["timing/brain_from_parts_ms"] = float(brain_from_parts_ms)
            debug_info["timing/perception_ms"] = float(perception_ms)
            debug_info["timing/perception_geom_model_forward_ms"] = float(geom_model_forward_ms)
            debug_info["timing/perception_geom_feature_extract_ms"] = float(geom_feature_extract_ms)
            debug_info["timing/chunk_execution_ms"] = float(chunk_execution_ms)
            debug_info["timing/control_cycle_ms"] = float(control_cycle_ms)
            debug_info["timing/action_horizon"] = int(action_horizon)
            debug_info["timing/control_hz"] = float(control_hz)
            debug_info["timing/perception_over_chunk"] = float(perception_over_chunk)
            debug_info["timing/brain_over_chunk"] = float(brain_over_chunk)
            debug_info["timing/reflex_over_chunk"] = float(reflex_over_chunk)
            debug_info["timing/reflex_step_over_cycle"] = float(reflex_step_over_cycle)
            debug_info["timing/perception_fits_chunk"] = float(perception_ms <= chunk_execution_ms)
            debug_info["timing/brain_fits_chunk"] = float(brain_total_ms <= chunk_execution_ms)
            debug_info["timing/reflex_step_fits_cycle"] = float(reflex_step_ms_mean <= control_cycle_ms)
            debug_info["path_a_feedback/source"] = str(feedback_source)
            debug_info["path_a_feedback/enabled"] = float(
                self._causal_feedback_ready and self.enable_causal_feedback_inference
            )
            debug_info["path_a_feedback/applied"] = 1.0 if isinstance(feedback_tokens, torch.Tensor) else 0.0
            if isinstance(feedback_tokens, torch.Tensor):
                debug_info["path_a_feedback/token_num"] = int(feedback_tokens.shape[1])
                debug_info["path_a_feedback/token_norm_mean"] = float(
                    feedback_tokens.detach().norm(dim=-1).mean().item()
                )
            else:
                debug_info["path_a_feedback/token_num"] = 0
            if isinstance(patha_auto_stats, dict):
                for k, v in patha_auto_stats.items():
                    if isinstance(k, str) and k.startswith("debug/causal_feedback/") and isinstance(v, (int, float)):
                        debug_info[f"path_a_feedback/{k.split('debug/causal_feedback/', 1)[-1]}"] = float(v)
            if isinstance(planning_info, dict):
                gate_mean = planning_info.get("feedback/delta_action_gate_mean", None)
                eff_norm = planning_info.get("feedback/delta_action_effective_norm_mean", None)
                valid_ratio = planning_info.get("feedback/valid_tk_ratio", None)
                infer_steps_used = planning_info.get("feedback/num_inference_timesteps_used", None)
                if isinstance(gate_mean, (int, float)):
                    debug_info["path_a_feedback/delta_action_gate_mean"] = float(gate_mean)
                if isinstance(eff_norm, (int, float)):
                    debug_info["path_a_feedback/delta_action_effective_norm_mean"] = float(eff_norm)
                if isinstance(valid_ratio, (int, float)):
                    debug_info["path_a_feedback/valid_tk_ratio"] = float(valid_ratio)
                if isinstance(infer_steps_used, (int, float)):
                    debug_info["timing/num_inference_timesteps_used"] = float(infer_steps_used)
            # Derived from action-head timing, exposed here for single-place monitoring.
            debug_info["timing/flow_total_ms_agg"] = float(flow_total_ms)
            debug_info["timing/flow_step_ms_mean_agg"] = float(flow_step_ms_mean)
            debug_info["timing/flow_step_ms_max_agg"] = float(flow_step_ms_max)
            debug_info["timing/flow_step_count_agg"] = int(flow_step_count)
            debug_info["timing/reflex_total_ms_agg"] = float(reflex_total_ms)
            debug_info["timing/reflex_step_ms_mean_agg"] = float(reflex_step_ms_mean)
            debug_info["timing/reflex_step_ms_max_agg"] = float(reflex_step_ms_max)
            debug_info["timing/reflex_step_count_agg"] = int(reflex_step_count)
            debug_info["timing/reflex_share_agg"] = float(reflex_share)

            # Closed-loop delay proxies (estimates). These are useful before
            # wiring true controller-side timestamps/indices.
            drift_response_lag_ms = float(
                debug_info.get("timing/drift_response_lag_ms", reflex_step_ms_mean)
            )
            drift_freshness_ms_est = float(perception_ms + drift_response_lag_ms)
            command_overlap_ratio_est = float(
                float(debug_info.get("timing/effective_control_hz", 0.0))
                * (chunk_execution_ms / 1000.0)
            )
            phase_shift_steps_est = float(
                drift_freshness_ms_est / max(control_cycle_ms, 1e-6)
            )
            debug_info["timing/drift_freshness_ms_est"] = drift_freshness_ms_est
            debug_info["timing/command_overlap_ratio_est"] = command_overlap_ratio_est
            debug_info["timing/phase_shift_steps_est"] = phase_shift_steps_est
            debug_info["timing/phase_shift_steps_est_rounded"] = float(
                round(phase_shift_steps_est)
            )
            debug_info["timing/drift_freshness_risky_est"] = float(
                drift_freshness_ms_est > 150.0
            )
            debug_info["timing/phase_shift_risky_est"] = float(
                phase_shift_steps_est > 2.0
            )
            debug_info["timing/command_overlap_insufficient_est"] = float(
                command_overlap_ratio_est < 1.0
            )
            output["debug_info"] = debug_info
        return output



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe_mapanything_llava3d.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.mapanything_llava3d.base_vlm = "/2025233147/zzq/SpatialVLA_llava3d/model_zoo/mapanythingllava3d_base_v3"

    

    model = MapAnythingLlava3D_PI(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
