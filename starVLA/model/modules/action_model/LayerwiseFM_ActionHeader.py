# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer


class TensorParallelContext:
    def __init__(self, tp_size, tp_rank, dp_size, dp_rank, tp_group):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.tp_group = tp_group


_TP_GROUP_CACHE = {}


def init_tensor_parallel(tp_size: int):
    if tp_size is None or tp_size <= 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    if world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} is not divisible by tp_size={tp_size}")
    rank = dist.get_rank()
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    tp_rank = rank % tp_size
    group_key = (world_size, tp_size, dp_rank)
    if group_key in _TP_GROUP_CACHE:
        tp_group = _TP_GROUP_CACHE[group_key]
    else:
        ranks = [dp_rank * tp_size + i for i in range(tp_size)]
        tp_group = dist.new_group(ranks=ranks)
        _TP_GROUP_CACHE[group_key] = tp_group
    if dist.get_rank() == 0:
        print(
            f"[TensorParallel] world_size={world_size}, tp_size={tp_size}, "
            f"dp_size={dp_size}"
        )
    return TensorParallelContext(tp_size=tp_size, tp_rank=tp_rank, dp_size=dp_size, dp_rank=dp_rank, tp_group=tp_group)


class TPLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_context: TensorParallelContext = None,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_context = tp_context
        self.is_tp = (
            tp_context is not None
            and tp_context.tp_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if not self.is_tp:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.tp_world_size = 1
            self.tp_rank = 0
        else:
            tp_world_size = tp_context.tp_size
            if out_features % tp_world_size != 0:
                raise ValueError(
                    f"out_features={out_features} must be divisible by tp_size={tp_world_size}"
                )
            self.tp_world_size = tp_world_size
            self.tp_rank = tp_context.tp_rank
            per_rank_out = out_features // tp_world_size
            self.weight = nn.Parameter(torch.empty(in_features, per_rank_out))
            if bias:
                self.bias = nn.Parameter(torch.empty(per_rank_out))
            else:
                self.bias = None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_tp:
            return self.linear(x)
        x_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        local_out = x_2d.matmul(self.weight)
        if self.bias is not None:
            local_out = local_out + self.bias
        if self.gather_output:
            tp_group = self.tp_context.tp_group
            outputs = [
                torch.empty_like(local_out) for _ in range(self.tp_world_size)
            ]
            dist.all_gather(outputs, local_out, group=tp_group)
            out_2d = torch.cat(outputs, dim=-1)
        else:
            out_2d = local_out
        out = out_2d.view(*x_shape[:-1], self.out_features)
        return out


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_context: TensorParallelContext = None,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_context = tp_context
        self.is_tp = (
            tp_context is not None
            and tp_context.tp_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if not self.is_tp:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.tp_world_size = 1
        else:
            tp_world_size = tp_context.tp_size
            if out_features % tp_world_size != 0:
                raise ValueError(
                    f"out_features={out_features} must be divisible by tp_size={tp_world_size}"
                )
            self.tp_world_size = tp_world_size
            per_rank_out = out_features // tp_world_size
            self.weight = nn.Parameter(torch.empty(in_features, per_rank_out))
            if bias:
                self.bias = nn.Parameter(torch.empty(per_rank_out))
            else:
                self.bias = None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_tp:
            return self.linear(x)
        x_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        local_out = x_2d.matmul(self.weight)
        if self.bias is not None:
            local_out = local_out + self.bias
        if self.gather_output:
            tp_group = self.tp_context.tp_group
            outputs = [
                torch.empty_like(local_out) for _ in range(self.tp_world_size)
            ]
            dist.all_gather(outputs, local_out, group=tp_group)
            out_2d = torch.cat(outputs, dim=-1)
        else:
            out_2d = local_out
        out = out_2d.view(*x_shape[:-1], self.out_features)
        return out


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_context: TensorParallelContext = None,
        reduce_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reduce_output = reduce_output
        self.tp_context = tp_context
        self.is_tp = (
            tp_context is not None
            and tp_context.tp_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        if not self.is_tp:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.tp_world_size = 1
        else:
            tp_world_size = tp_context.tp_size
            if in_features % tp_world_size != 0:
                raise ValueError(
                    f"in_features={in_features} must be divisible by tp_size={tp_world_size}"
                )
            self.tp_world_size = tp_world_size
            per_rank_in = in_features // tp_world_size
            self.weight = nn.Parameter(torch.empty(per_rank_in, out_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.bias = None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_tp:
            return self.linear(x)
        x_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        tp_world_size = self.tp_context.tp_size
        per_rank_in = self.in_features // tp_world_size
        rank = self.tp_context.tp_rank
        start = rank * per_rank_in
        end = start + per_rank_in
        x_local = x_2d[:, start:end]
        local_out = x_local.matmul(self.weight)
        if self.bias is not None:
            local_out = local_out + self.bias
        if self.reduce_output:
            tp_group = self.tp_context.tp_group
            dist.all_reduce(local_out, group=tp_group)
        out = local_out.view(*x_shape[:-1], self.out_features)
        return out

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

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
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        output_dim=2048,
        tp_context: TensorParallelContext = None,
    ):
        super().__init__()
        use_tp = tp_context is not None and dist.is_available() and dist.is_initialized()
        if use_tp and hidden_dim % tp_context.tp_size == 0:
            self.layer1 = TPLinear(input_dim, hidden_dim, tp_context=tp_context)
        else:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
        if use_tp and output_dim % tp_context.tp_size == 0:
            self.layer2 = TPLinear(hidden_dim, output_dim, tp_context=tp_context)
        else:
            self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, tp_context: TensorParallelContext = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = TPLinear(action_dim, hidden_size, tp_context=tp_context)
        self.layer2 = TPLinear(2 * hidden_size, hidden_size, tp_context=tp_context)
        self.layer3 = TPLinear(hidden_size, hidden_size, tp_context=tp_context)
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
        tp_size = getattr(action_config, "tp_size", 1)
        self.tp_context = init_tensor_parallel(tp_size)
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
        if isinstance(diffusion_model_cfg, dict):
            diffusion_model_cfg["cross_attention_dim"] = DiTConfig["input_embedding_dim"]
            diffusion_model_cfg["tp_size"] = tp_size
            diffusion_model_cfg["tp_enable_ffn"] = tp_size > 1
        else:
            diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"]
            diffusion_model_cfg.tp_size = tp_size
            diffusion_model_cfg.tp_enable_ffn = tp_size > 1
        self.input_embedding_dim = global_config.framework.mapanything_llava3d.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg)
        if self.tp_context is not None and self.tp_context.tp_size > 1 and dist.is_initialized():
            out1_dim = self.model.proj_out_1.out_features
            out2_dim = self.model.proj_out_2.out_features
            proj1 = TPLinear(
                self.model.inner_dim,
                out1_dim,
                bias=self.model.proj_out_1.bias is not None,
                tp_context=self.tp_context,
            )
            proj2 = TPLinear(
                self.model.inner_dim,
                out2_dim,
                bias=self.model.proj_out_2.bias is not None,
                tp_context=self.tp_context,
            )
            self.model.proj_out_1 = proj1
            self.model.proj_out_2 = proj2
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
            tp_context=self.tp_context,
        ) if action_config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
            tp_context=self.tp_context,
        )
        self.action_decoder = MLP(
            input_dim=self.dit_out_hidden_size,
            hidden_dim=1024,
            output_dim=self.action_dim,
            tp_context=self.tp_context,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if self.tp_context is not None and self.tp_context.tp_rank == 0:
            try:
                print("TensorParallel shard proj_out_1", tuple(self.model.proj_out_1.weight.shape))
                print("TensorParallel shard proj_out_2", tuple(self.model.proj_out_2.weight.shape))
                print("TensorParallel shard action_decoder.layer1", tuple(self.action_decoder.layer1.weight.shape))
                print("TensorParallel shard action_decoder.layer2", tuple(self.action_decoder.layer2.weight.shape))
            except Exception:
                pass

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def _apply_layerwise_cross_attention(self, saction_embs, vl_embs_list, temb):
        """
        Apply layerwise cross-attention between state-action embeddings and vision-language embeddings.

        Args:
            saction_embs: Tensor of shape (B, seq_length, embedding_dim)
            vl_embs_list: List of tensors, each of shape (B, seq_length, embedding_dim)
            temb: Tensor of shape (B, embedding_dim)

        Returns:
            hidden_states: Tensor of shape (B, seq_length, embedding_dim)
        """
        hidden_states = saction_embs
        self._last_dit_layer_means = []
        self._last_dit_layer_vars = []
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=vl_embs_list[layer_idx],
                temb=temb,
            )
            stats = hidden_states.detach().float()
            self._last_dit_layer_means.append(stats.mean().item())
            self._last_dit_layer_vars.append(stats.var(unbiased=False).item())
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

    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        
        temb = self.model.timestep_encoder(t_discretized)
        hidden_states = self._apply_layerwise_cross_attention(sa_embs, vl_embs_list, temb)
        pred_velocity = self._process_output(hidden_states, temb, actions.shape[1])
        loss = ((pred_velocity - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
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
            hidden_states = self._apply_layerwise_cross_attention(sa_embs, vl_embs_list, temb)
            pred_velocity = self._process_output(hidden_states, temb, self.action_horizon)
            actions = actions + dt * pred_velocity
        return actions

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
