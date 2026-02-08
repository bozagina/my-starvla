# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class DiTTPContext:
    def __init__(self, tp_size, tp_rank, tp_group):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group


_DIT_TP_GROUP_CACHE = {}


def init_dit_tensor_parallel(tp_size: int):
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
    key = (world_size, tp_size, dp_rank)
    if key in _DIT_TP_GROUP_CACHE:
        tp_group = _DIT_TP_GROUP_CACHE[key]
    else:
        ranks = [dp_rank * tp_size + i for i in range(tp_size)]
        tp_group = dist.new_group(ranks=ranks)
        _DIT_TP_GROUP_CACHE[key] = tp_group
    return DiTTPContext(tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_context: DiTTPContext = None,
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
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / fan_in**0.5 if fan_in > 0 else 0
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
        tp_context: DiTTPContext = None,
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
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / fan_in**0.5 if fan_in > 0 else 0
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


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class TPFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dropout: float,
        activation_fn: str,
        final_dropout: bool,
        bias: bool,
        tp_context: DiTTPContext,
    ):
        super().__init__()
        use_tp = (
            tp_context is not None
            and dist.is_available()
            and dist.is_initialized()
            and dim % tp_context.tp_size == 0
            and inner_dim % tp_context.tp_size == 0
        )
        if use_tp:
            self.w1 = ColumnParallelLinear(dim, inner_dim, bias=bias, tp_context=tp_context, gather_output=False)
            self.w2 = RowParallelLinear(inner_dim, dim, bias=bias, tp_context=tp_context, reduce_output=True)
        else:
            self.w1 = nn.Linear(dim, inner_dim, bias=bias)
            self.w2 = nn.Linear(inner_dim, dim, bias=bias)
        self.act = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=inner_dim,
            bias=bias,
        ).act
        self.dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        hidden = self.act(hidden)
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        out = self.w2(hidden)
        return out


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        tp_context: DiTTPContext = None,
        use_tp_ffn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type
        self.tp_context = tp_context
        self.use_tp_ffn = use_tp_ffn

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        inner_dim = ff_inner_dim if ff_inner_dim is not None else dim * 4
        if use_tp_ffn and tp_context is not None:
            self.ff = TPFeedForward(
                dim=dim,
                inner_dim=inner_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                bias=ff_bias,
                tp_context=tp_context,
            )
        else:
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=inner_dim,
                bias=ff_bias,
            )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1( 
            norm_hidden_states, 
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, #@JinhuiYE original attention_mask=attention_mask
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    # register_to_config 的作用是创建类的时候会自动把传入的参数注册到 config 中，这样后续调用的时候可以通过 self.config.xxx 调用 还不是 self.xxx
    @register_to_config # 去看一下这个的作用 --> 将传入的参数注册到配置中 TODO 改为我们的单例模式, 写一个 能够merge 的 @merge_pram_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        tp_size = getattr(self.config, "tp_size", 1)
        self.tp_context = init_dit_tensor_parallel(tp_size)
        use_tp_ffn = getattr(self.config, "tp_enable_ffn", False)

        # Timestep encoder
        #  self.config.compute_dtype 可能不存在，要提前处理
        compute_dtype = getattr(self.config, 'compute_dtype', torch.float32)
        self.timestep_encoder = TimestepEncoder( # TODO BUG, train 的时候 self.config.compute_dtype 不会报错， 但是 eval 的时候会
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    tp_context=self.tp_context,
                    use_tp_ffn=use_tp_ffn,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
