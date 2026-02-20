
import time
import random
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np
from peft import LoraConfig, get_peft_model
import csv

# Optional logging dependencies
try:
    import swanlab  # type: ignore
    SWANLAB_AVAILABLE = True
    SWANLAB_RUN = None  # type: ignore
except Exception:
    swanlab = None  # type: ignore
    SWANLAB_AVAILABLE = False
    SWANLAB_RUN = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

# Core OpenVLA components
from experiments.robot.openvla_utils import (
    get_processor,
    get_proprio_projector,
)

# Masks used to extract action-related hidden states
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)

# Constants
from prismatic.vla.constants import (
    NUM_ACTIONS_CHUNK,
    ACTION_DIM,
    PROPRIO_DIM,
)
from typing import Any

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from rl.utils import get_vla, compute_num_patches, prepare_inputs_batch, forward_vla


class ActorCritic(nn.Module):
    """
    Actor-Critic for OpenVLA-based continuous control.

    forward(inputs_batch) returns:
      - actions_all: sampled actions in (-1, 1), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)  [squashed Gaussian]
      - mu_all: mean actions from action_head.predict_action(...), shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - log_std_all: condition-independent log-std broadcast to all chunks, shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
      - value: state value estimate, shape (B,)
    """

    def __init__(self, cfg, torch_dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg

        # Device / dtype
        self.vla = get_vla(cfg, torch_dtype)
        self.device = self.vla.device
        self.model_dtype = torch_dtype
        self.vla = self.vla.to(dtype=self.model_dtype)
        # 计算有效的vocab范围
        """
        - 一些模型会将词表矩阵按 pad_to_multiple_of 对齐到某个倍数，导致实际矩阵行数大于真实词表大小
- 这里通过 vocab_size - pad_to_multiple_of 得到真实可用的词表上界，避免把对齐填充的“空位”误认为有效 token"""
        self.vocab_size = self.vla.config.text_config.vocab_size - self.vla.config.pad_to_multiple_of # ？
        self.n_action_bins = self.vla.config.n_action_bins
        self.action_vocab_start = self.vocab_size - self.n_action_bins
        
        # 原地替换lm_head为精简版本
        original_lm_head = self.vla.language_model.lm_head
        # 
        print(f"原始 lm_head 形状: weight={original_lm_head.weight.shape}, "
              f"bias={original_lm_head.bias.shape if original_lm_head.bias is not None else None}") # 原始 lm_head 形状: weight=torch.Size([32064, 4096]), bias=None
        
        # 提取权重和偏置的有效部分 [action_vocab_start:vocab_size, :]
        with torch.no_grad():
            action_weight = original_lm_head.weight[self.action_vocab_start:self.vocab_size, :].clone()
            if original_lm_head.bias is not None:
                action_bias = original_lm_head.bias[self.action_vocab_start:self.vocab_size].clone()
            else:
                action_bias = None
        
        # 创建新的精简lm_head并原地替换
        new_lm_head = nn.Linear(
            original_lm_head.in_features,
            self.n_action_bins,
            bias=(action_bias is not None)
        ).to(self.device).to(dtype=self.model_dtype)
        
        # 复制权重
        with torch.no_grad():
            new_lm_head.weight.copy_(action_weight)
            if action_bias is not None:
                new_lm_head.bias.copy_(action_bias)
        
        # 原地替换
        self.vla.language_model.lm_head = new_lm_head
        """
        - 原始 lm_head 是一个线性层，权重形状是 (vocab_size, hidden_dim) ，
        即 (32064, 4096) ，表示“把每个位置的隐藏向量（4096维）投影成对 32064 个词的 logits”。
        - 精简后把输出通道改为 n_action_bins （例如 256），权重形状变为 (256, 4096) 。
        含义是“把每个位置的隐藏向量（4096维）投影成对 256 个动作 bin 的 logits”。
        """
        print(f"精简后 lm_head 形状: weight={new_lm_head.weight.shape}, "
              f"bias={new_lm_head.bias.shape if new_lm_head.bias is not None else None}")
        print(f"lm_head 已从 ({original_lm_head.out_features}, {original_lm_head.in_features}) "
              f"精简为 ({self.n_action_bins}, {original_lm_head.in_features})")
        
        # 🔒 冻结 VLA 参数
        for param in self.vla.parameters():
            param.requires_grad = False
        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            self.vla = get_peft_model(self.vla, lora_config)
            print("lora_rank:", cfg.lora_rank)
            # 打印可训练Lora参数信息
            self.vla.print_trainable_parameters()
        self.vla.language_model: LlamaForCausalLM # 类型注释，
        # 手动解冻lm_head参数（保持全参量训练）
        for param in self.vla.language_model.lm_head.parameters():
            param.requires_grad = True
        """
        self.bins
        array([-1.000, -0.992, -0.984, -0.976, -0.969, -0.961, -0.953, -0.945,
            -0.937, -0.929, -0.922, -0.914, -0.906, -0.898, -0.890, -0.882,
            -0.875, -0.867, -0.859, -0.851, -0.843, -0.835, -0.827, -0.820,
            -0.812, -0.804, -0.796, -0.788, -0.780, -0.773, -0.765, -0.757,
            -0.749, -0.741, -0.733, -0.725, -0.718, -0.710, -0.702, -0.694,
            -0.686, -0.678, -0.671, -0.663, -0.655, -0.647, -0.639, -0.631,
            -0.624, -0.616, -0.608, -0.600, -0.592, -0.584, -0.576, -0.569,
            -0.561, -0.553, -0.545, -0.537, -0.529, -0.522, -0.514, -0.506,
            -0.498, -0.490, -0.482, -0.475, -0.467, -0.459, -0.451, -0.443,
            -0.435, -0.427, -0.420, -0.412, -0.404, -0.396, -0.388, -0.380,
            -0.373, -0.365, -0.357, -0.349, -0.341, -0.333, -0.325, -0.318,
            -0.310, -0.302, -0.294, -0.286, -0.278, -0.271, -0.263, -0.255,
            -0.247, -0.239, -0.231, -0.224, -0.216, -0.208, -0.200, -0.192,
            -0.184, -0.176, -0.169, -0.161, -0.153, -0.145, -0.137, -0.129,
            -0.122, -0.114, -0.106, -0.098, -0.090, -0.082, -0.075, -0.067,
            -0.059, -0.051, -0.043, -0.035, -0.027, -0.020, -0.012, -0.004,
            0.004, 0.012, 0.020, 0.027, 0.035, 0.043, 0.051, 0.059, 0.067,
            0.075, 0.082, 0.090, 0.098, 0.106, 0.114, 0.122, 0.129, 0.137,
            0.145, 0.153, 0.161, 0.169, 0.176, 0.184, 0.192, 0.200, 0.208,
            0.216, 0.224, 0.231, 0.239, 0.247, 0.255, 0.263, 0.271, 0.278,
            0.286, 0.294, 0.302, 0.310, 0.318, 0.325, 0.333, 0.341, 0.349,
            0.357, 0.365, 0.373, 0.380, 0.388, 0.396, 0.404, 0.412, 0.420,
            0.427, 0.435, 0.443, 0.451, 0.459, 0.467, 0.475, 0.482, 0.490,
            0.498, 0.506, 0.514, 0.522, 0.529, 0.537, 0.545, 0.553, 0.561,
            0.569, 0.576, 0.584, 0.592, 0.600, 0.608, 0.616, 0.624, 0.631,
            0.639, 0.647, 0.655, 0.663, 0.671, 0.678, 0.686, 0.694, 0.702,
            0.710, 0.718, 0.725, 0.733, 0.741, 0.749, 0.757, 0.765, 0.773,
            0.780, 0.788, 0.796, 0.804, 0.812, 0.820, 0.827, 0.835, 0.843,
            0.851, 0.859, 0.867, 0.875, 0.882, 0.890, 0.898, 0.906, 0.914,
            0.922, 0.929, 0.937, 0.945, 0.953, 0.961, 0.969, 0.976, 0.984,
            0.992, 1.000])
            self.bin_centers
            array([-0.996, -0.988, -0.980, -0.973, -0.965, -0.957, -0.949, -0.941,
            -0.933, -0.925, -0.918, -0.910, -0.902, -0.894, -0.886, -0.878,
            -0.871, -0.863, -0.855, -0.847, -0.839, -0.831, -0.824, -0.816,
            -0.808, -0.800, -0.792, -0.784, -0.776, -0.769, -0.761, -0.753,
            -0.745, -0.737, -0.729, -0.722, -0.714, -0.706, -0.698, -0.690,
            -0.682, -0.675, -0.667, -0.659, -0.651, -0.643, -0.635, -0.627,
            -0.620, -0.612, -0.604, -0.596, -0.588, -0.580, -0.573, -0.565,
            -0.557, -0.549, -0.541, -0.533, -0.525, -0.518, -0.510, -0.502,
            -0.494, -0.486, -0.478, -0.471, -0.463, -0.455, -0.447, -0.439,
            -0.431, -0.424, -0.416, -0.408, -0.400, -0.392, -0.384, -0.376,
            -0.369, -0.361, -0.353, -0.345, -0.337, -0.329, -0.322, -0.314,
            -0.306, -0.298, -0.290, -0.282, -0.275, -0.267, -0.259, -0.251,
            -0.243, -0.235, -0.227, -0.220, -0.212, -0.204, -0.196, -0.188,
            -0.180, -0.173, -0.165, -0.157, -0.149, -0.141, -0.133, -0.125,
            -0.118, -0.110, -0.102, -0.094, -0.086, -0.078, -0.071, -0.063,
            -0.055, -0.047, -0.039, -0.031, -0.024, -0.016, -0.008, 0.000,
            0.008, 0.016, 0.024, 0.031, 0.039, 0.047, 0.055, 0.063, 0.071,
            0.078, 0.086, 0.094, 0.102, 0.110, 0.118, 0.125, 0.133, 0.141,
            0.149, 0.157, 0.165, 0.173, 0.180, 0.188, 0.196, 0.204, 0.212,
            0.220, 0.227, 0.235, 0.243, 0.251, 0.259, 0.267, 0.275, 0.282,
            0.290, 0.298, 0.306, 0.314, 0.322, 0.329, 0.337, 0.345, 0.353,
            0.361, 0.369, 0.376, 0.384, 0.392, 0.400, 0.408, 0.416, 0.424,
            0.431, 0.439, 0.447, 0.455, 0.463, 0.471, 0.478, 0.486, 0.494,
            0.502, 0.510, 0.518, 0.525, 0.533, 0.541, 0.549, 0.557, 0.565,
            0.573, 0.580, 0.588, 0.596, 0.604, 0.612, 0.620, 0.627, 0.635,
            0.643, 0.651, 0.659, 0.667, 0.675, 0.682, 0.690, 0.698, 0.706,
            0.714, 0.722, 0.729, 0.737, 0.745, 0.753, 0.761, 0.769, 0.776,
            0.784, 0.792, 0.800, 0.808, 0.816, 0.824, 0.831, 0.839, 0.847,
            0.855, 0.863, 0.871, 0.878, 0.886, 0.894, 0.902, 0.910, 0.918,
            0.925, 0.933, 0.941, 0.949, 0.957, 0.965, 0.973, 0.980, 0.988,
            0.996])
            special variables:
            [0:255] : [-0.996078431372549, -0.9882352941176471, -0.9803921568627452, -0.9725490196078431, -0.9647058823529412, -0.9568627450980391, -0.9490196078431372, -0.9411764705882353, -0.9333333333333333, -0.9254901960784314, -0.9176470588235295, -0.9098039215686274, -0.9019607843137255, -0.8941176470588235, -0.8862745098039215, -0.8784313725490196, -0.8705882352941177, -0.8627450980392157, -0.8549019607843138, -0.8470588235294118, -0.8392156862745098, -0.8313725490196078, -0.8235294117647058, -0.8156862745098039, -0.807843137254902, -0.8, -0.7921568627450981, -0.7843137254901961, -0.7764705882352941, -0.7686274509803921, -0.7607843137254902, -0.7529411764705882, -0.7450980392156863, -0.7372549019607844, -0.7294117647058824, -0.7215686274509804, -0.7137254901960784, -0.7058823529411764, -0.6980392156862745, -0.6901960784313725, -0.6823529411764706, -0.6745098039215687, -0.6666666666666667, -0.6588235294117647, -0.6509803921568628, -0.6431372549019607, -0.6352941176470588, -0.6274509803921569, -0.6196078431372549, -0.611764705882353, -0.603921568627451, -0.596078431372549, -0.5882352941176471, -0.580392156862745, -0.5725490196078431, -0.5647058823529412, -0.5568627450980392, -0.5490196078431373, -0.5411764705882354, ...]
            dtype: dtype('float64')
            max: 0.996078431372549
            min: -0.996078431372549
            shape: (255,)
            size: 255
        """
    

        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Keep processor for external preparation
        self.processor = get_processor(cfg)
        """
        processor:图文处理器
        proprio_projector: 一个将本体感受状态（proprio）投影到语言模型嵌入维度的投影模块
        """
        self.proprio_projector = get_proprio_projector(
            cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )
        # 注意力池化层
        """
        - 为 Critic 提供一个可学习的“汇聚器”，把动作相关的隐状态集合 (B, N, D) 转成一个固定维度 (B, D) 的状态表示（而是对所有动作相关位置的隐藏向量做“加权汇聚”形成一个综合表示），
        再经 value_head 输出标量价值。
        - 本层为每个动作位置的隐藏向量打分，softmax 得到权重，再做加权和。这样 Critic 能更关注对价值判断更重要的动作位点，而不是简单平均。"""
        self.attn_pool = nn.Sequential(
            nn.Linear(self.vla.llm_dim, 1),
        ).to(self.device).to(dtype=self.model_dtype)

        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.vla.llm_dim),
            nn.Linear(self.vla.llm_dim, self.vla.llm_dim),
            nn.ReLU(),
            nn.Linear(self.vla.llm_dim, 1),
        )
        # self.safe_load_model("/cpfs01/liuwei_workspace/models/finetune_rl/agent_checkpoint_epoch_6000", strict=True)
        self.to(self.device).to(dtype=self.model_dtype)

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        将可训练参数分为 'policy' 和 'value' 两组。
        这对于为不同组件设置不同的学习率至关重要。
        """
        self.vla.language_model: LlamaForCausalLM 
        
        # 1. 收集所有可训练参数
        policy_params = list(self.proprio_projector.parameters())
        value_params = []
        
        # 2. 收集 LoRA 适配器参数 (policy)
        for name, param in self.vla.named_parameters():
            if param.requires_grad:
                policy_params.append(param)
        
        # 3. 收集 value head 参数 (value)
        value_params.extend(list(self.value_head.parameters()))
        # 添加注意力池化层参数到价值组
        value_params.extend(list(self.attn_pool.parameters()))

        # 4. 验证没有遗漏任何可训练参数
        all_trainable_params = set(filter(lambda p: p.requires_grad, self.parameters()))
        grouped_params = set(policy_params) | set(value_params)
        
        # 打印调试信息
        if all_trainable_params != grouped_params:
            missing_params = all_trainable_params - grouped_params
            print(f"警告: 发现 {len(missing_params)} 个未分组的可训练参数:")
            for p in missing_params:
                for n, param in self.named_parameters():
                    if param is p:
                        print(f"  - {n}")
                        break
            raise ValueError("参数分组不完整！请检查未分组的参数。")
        
        return [
            {"name": "policy", "params": policy_params},
            {"name": "value", "params": value_params},
        ]

    def _extract_actions_hidden(self, last_hidden_states: torch.Tensor, logits: torch.Tensor, labels, has_act_emb) -> torch.Tensor:
        """
        从 last_hidden_states 和 logits 中提取动作相关的部分。
        由于lm_head已经被精简为只输出n_action_bins，所以logits直接可用。
        
        返回:
          action_logits: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, n_action_bins)
          actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
        """

        """
        - B ：batch 大小
        - N ：动作相关位置的总数，等于 NUM_ACTIONS_CHUNK * ACTION_DIM；每个位置对应“某个时间步的某个动作维度”的隐藏向量
        - D ：语言模型隐藏维度 self.vla.llm_dim （例如 4096），是每个位置的特征向量维度
        """
        ground_truth_token_ids = labels[:, 1:].to(self.device)  # (B, text_len-1)
        # 标记“当前动作”的占位 token 位置
        current_action_mask = get_current_action_mask(ground_truth_token_ids)  # (B, text_len-1)
        # 标记“下一步动作”的占位 token 位置
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)      # (B, text_len-1)
        # 并集，表示“当前+下一步”所有动作位点
        action_mask = current_action_mask | next_actions_mask


        # 是序列最前面“非文本”部分的长度，用来定位文本段的起始位置。
        num_patches = self._compute_num_patches()
        if has_act_emb:
            num_patches += 1
        text_hidden_states = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)
        text_logits = logits[:, num_patches:-1]  # (B, text_len, n_action_bins) - 已经是精简后的

        B, _, D = text_hidden_states.shape
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
            .to(self.model_dtype)
        )
        
        # 提取动作对应的logits（已经是精简后的256维）
        action_logits = text_logits[action_mask].reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, self.n_action_bins)
        
        return action_logits, actions_hidden_states

    def _forward_vla(self, batch: Dict[str, torch.Tensor]):
        return forward_vla(self, batch)

    def _compute_value_from_hidden(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        使用注意力池化计算状态价值
        actions_hidden_states: (B, num_tokens, D)
        """

        
        # 1. 计算注意力分数
        """
        attn_pool可以理解成逐位置打分器，为每个动作位点（时间步 × 动作维度）计算一个可学习的“重要性分数”
        """
        scores = self.attn_pool(actions_hidden_states)  # (B, num_tokens, 1)
        
        
        # 2. 应用softmax获取注意力权重
        """
        沿序列维 dim=1 归一化，使每个样本的权重总和为 1，形成对所有动作位点的概率分布
        """
        weights = torch.softmax(scores, dim=1)  # (B, num_tokens, 1)
        
        # 3. 加权平均得到池化表示
        pooled = torch.sum(weights * actions_hidden_states, dim=1)  # (B, D)
        
        # 4. 通过价值头计算最终价值
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return value.to(torch.float32)

    def forward(self, inputs_batch: Dict[str, Any], return_vit_out=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          action_logits: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, n_action_bins)
          value:         (B,)
        """
        # Sanity checks
        for k in ("input_ids", "attention_mask", "pixel_values", "labels", "proprio"):
            if k not in inputs_batch:
                raise KeyError(f"inputs_batch missing key: {k}")

        # 1. VLA前向传播获取隐藏状态和logits
        output = self._forward_vla(inputs_batch)
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        """
        最后一层: last_hidden_states = output.hidden_states[-1] 取的是语言模型最后一个 Transformer Block 的输出，即“最终的上下文表征”
        """
        logits = output.logits  # (B, seq_len, n_action_bins) - 已经是精简后的

        """
        数据流: 先得到 last_hidden_states (B, seq_len, D) ，再用精简后的 lm_head: R^D -> R^256 投影为分类 logits (B, seq_len, 256) , logits是未经过softmax的分数
        """

        logits = output.logits
        action_logits, actions_hidden_states = self._extract_actions_hidden(last_hidden_states, logits, inputs_batch['labels'], has_act_emb=("this_act_emb" in inputs_batch))

        # 3. 计算价值函数
        value = self._compute_value_from_hidden(actions_hidden_states.detach())  # (B,)

        if return_vit_out:
            return action_logits, value.to(torch.float32), output.projector_features.to(torch.float32)
        else:
            return action_logits, value.to(torch.float32)

    def post_process(self, logits: torch.Tensor, deterministic: List[bool]) -> Tuple[torch.distributions.Categorical, torch.Tensor, np.ndarray]:
        """
        后处理logits以生成动作。
        注意：现在logits已经是精简后的 (B, num_dims, n_action_bins)，无需再截取。
        """
        # 创建分布并计算两种动作
        dist = torch.distributions.Categorical(logits=logits)
        stochastic_tokens = dist.sample()
        deterministic_tokens = torch.argmax(logits, dim=-1)
        """
        - stochastic_tokens 是按 Categorical(logits=logits) 在每个动作位点随机采样得到的 token id，形状 (B, N) ，对应每个样本、每个动作位点的随机选择。
    - deterministic_tokens 是对每个动作位点的 logits 做 argmax 得到的最大概率的 token id，形状 (B, N) ，对应贪心选择。
        """
        is_deterministic_tensor = torch.tensor(
            deterministic, dtype=torch.bool, device=logits.device
        )
        is_deterministic_tensor = is_deterministic_tensor.unsqueeze(1)
        action_token_ids = torch.where(
            is_deterministic_tensor, deterministic_tokens, stochastic_tokens
        )
        """
        action_token_ids: torch.Size([10, 56])
        56 的来源： N = NUM_ACTIONS_CHUNK * ACTION_DIM
        - 每个样本的 56 个位置表示：
        - 8 个时间步（open-loop 的动作序列长度， NUM_ACTIONS_CHUNK=8 ）
        - 每个时间步的 7 个动作维度（ ACTION_DIM=7 ）
        - 所以这 56 个位置就是“按时间步×维度展开”的所有动作位点
        - 值域： action_token_ids[b, t] ∈ [0, n_action_bins-1] 是第 b 个样本在第 t 个动作位点选择的离散 bin 索引（来自贪心或采样），随后会映射为归一化连续值。
        
        """

        # 将token ID转换为bin索引（注意：现在action_token_ids范围是0到n_action_bins-1）
        # 反向索引到分箱中心
        actions_from_tokens = self.n_action_bins - 1 - action_token_ids
        # 边界裁剪到中心索引范围
        discretized = np.clip(actions_from_tokens.cpu().numpy(), a_min=0, a_max=self.bin_centers.shape[0] - 1)
        # 查表得到连续值并整形
        normalized_actions = self.bin_centers[discretized]  # 形状 (B, NUM_ACTIONS_CHUNK * ACTION_DIM)
        normalized_actions = normalized_actions.reshape(
            normalized_actions.shape[0], NUM_ACTIONS_CHUNK, ACTION_DIM
        ) # (10, 8, 7)
        
        return dist, action_token_ids, normalized_actions

    def prepare_inputs_batch(self, inp, max_len=None):
        return prepare_inputs_batch(self, inp, max_len)

    def get_norm_stats(self):
        return self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]

    def _compute_num_patches(self):
        return compute_num_patches(self.vla, self.cfg)

    def save_model(self, save_path, epoch: int | None = None):
        """
        保存模型的 LoRA 权重和额外层
        """
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel
        import torch.distributed as dist
        import copy

        save_path = Path(save_path)
        # save_path.mkdir(parents=True, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else ""

        save_path = save_path/ f"agent_checkpoint{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)

        agent_lora_path = save_path / f"agent_lora"
        self.vla.save_pretrained(agent_lora_path)
        print(f"✓ Agent LoRA 权重已保存到: {agent_lora_path}")
        
        agent_extra_layers = {
            "value_head": self.value_head.state_dict(),
            "attn_pool": self.attn_pool.state_dict(),
            "lm_head": self.vla.language_model.lm_head.state_dict(),
        }
        # 额外补齐：保存 proprio_projector（因为它在 policy 里会训练）
        if hasattr(self, "proprio_projector") and self.proprio_projector is not None:
            agent_extra_layers["proprio_projector"] = self.proprio_projector.state_dict()

        agent_extra_path = save_path / f"agent_extra_layers.pt"
        torch.save(agent_extra_layers, agent_extra_path)
        print(f"✓ Agent 额外层已保存到: {agent_extra_path}")

    def safe_load_model(self, checkpoint_dir: str | Path, strict: bool = True):
        """
        加载模型的 LoRA 权重和额外层
        该方法不会修改传入的vla模型结构，仅加载权重
        """
        from peft import PeftModel
        from rl.utils import load_lora_inplace

        checkpoint_dir = Path(checkpoint_dir)

        # 1) 挂载 LoRA 适配器 -不可使用PeftModel.from_pretrained，会将原本的peftmodel结构的vla再次嵌套一层peft结构 
        lora_dir = checkpoint_dir / "agent_lora"
        if lora_dir.exists(): 
            assert isinstance(self.vla, PeftModel) 
            load_lora_inplace(self.vla, lora_dir) 
            print(f"✓ Agent LoRA 权重已安全加载") 
        else: 
            print(f"⚠️ 警告: 未找到 Agent LoRA 权重: {lora_dir}")

        # 2) 加载额外层
        extra_path = checkpoint_dir / "agent_extra_layers.pt"
        sd = torch.load(extra_path, map_location=self.device)

        self.vla.language_model.lm_head.to(self.device).to(self.model_dtype)
        self.value_head.to(self.device).to(self.model_dtype)
        self.attn_pool.to(self.device).to(self.model_dtype)
        if hasattr(self, "proprio_projector") and self.proprio_projector is not None:
            self.proprio_projector.to(self.device).to(self.model_dtype)

        self.vla.language_model.lm_head.load_state_dict(sd["lm_head"], strict=strict)
        self.value_head.load_state_dict(sd["value_head"], strict=strict)
        self.attn_pool.load_state_dict(sd["attn_pool"], strict=strict)
        if "proprio_projector" in sd and self.proprio_projector is not None:
            self.proprio_projector.load_state_dict(sd["proprio_projector"], strict=strict)

        print(f"✅ 已从 {checkpoint_dir} 加载 LoRA 与额外层")
    
    def load_lora_and_merge_for_eval(self, checkpoint_dir: str | Path, keep_dtype: torch.dtype, strict: bool = True):
        """
        <评估前> 合并总体模型，并输出两份产物：
        1) agent_merged_for_eval/   —— 合并后的骨干（剔除 lm_head）
        2) agent_extra_layers.pt    —— 外挂头 + 精简 lm_head（已合并后的普通 Linear）+ norm_stats
        """
        from peft import PeftModel
        from rl.utils import load_lora_inplace
        import torch, copy

        checkpoint_dir = Path(checkpoint_dir)

        # 1) 挂载 LoRA
        lora_dir = checkpoint_dir / "agent_lora"
        assert isinstance(self.vla, PeftModel), "self.vla 需要是 PeftModel 才能 merge_and_unload"
        load_lora_inplace(self.vla, lora_dir)
        print(f"✓ Agent LoRA 权重已加载")

        # 先把 norm_stats 取出来，避免 merge 时丢失自定义属性
        cached_norm_stats = copy.deepcopy(getattr(self.vla, "norm_stats", None))

        # 2) 合并 LoRA
        self.vla = self.vla.merge_and_unload()

        # 3) dtype/设备 & 解绑 tie
        self.vla = self.vla.to(device=self.device, dtype=keep_dtype)
        if hasattr(self.vla.config, "tie_word_embeddings"):
            self.vla.config.tie_word_embeddings = False
        if hasattr(self.vla, "tie_weights"):
            self.vla.tie_weights = lambda *a, **k: None

        # 把丢失的 norm_stats 放回去（如果原来有）
        if cached_norm_stats is not None:
            setattr(self.vla, "norm_stats", cached_norm_stats)
            print(f"✓ norm_stats: {cached_norm_stats} 放回")

        print(f"✓ Agent LoRA 权重已合并到基座模型")

        # 4) 保存：骨干（剔除 lm_head）
        save_path = checkpoint_dir / "agent_merged_for_eval"
        save_path.mkdir(parents=True, exist_ok=True)

        full_sd = self.vla.state_dict()
        filtered_sd = {k: v for k, v in full_sd.items()
                    if not k.endswith("lm_head.weight") and not k.endswith("lm_head.bias")}
        self.vla.save_pretrained(save_path, state_dict=filtered_sd)
        print(f"✓ 已保存合并后的骨干到: {save_path}(已剔除 lm_head)")

        # 5) 保存 extra：精简 lm_head（已合并，普通 Linear）+ 外挂头 + norm_stats
        extra_out = {}
        with torch.no_grad():
            extra_out["lm_head"] = {k: v.detach().cpu() for k, v in self.vla.language_model.lm_head.state_dict().items()}
            extra_out["value_head"] = {k: v.detach().cpu() for k, v in self.value_head.state_dict().items()}
            extra_out["attn_pool"]  = {k: v.detach().cpu() for k, v in self.attn_pool.state_dict().items()}
            if hasattr(self, "proprio_projector") and self.proprio_projector is not None:
                extra_out["proprio_projector"] = {k: v.detach().cpu() for k, v in self.proprio_projector.state_dict().items()}
            # 直接把 Python dict 存起来（torch.save 支持任意 Python 对象）
            extra_out["norm_stats"] = copy.deepcopy(getattr(self.vla, "norm_stats", None))

        extra_path = checkpoint_dir / "agent_extra_layers.pt"
        torch.save(extra_out, extra_path)
        print(f"✓ 已重写 extra 层到: {extra_path}（含精简 lm_head 与 norm_stats)")

    def load_merged_model_for_eval(
        self,
        path: str | Path,
        keep_dtype: torch.dtype,
        strict: bool = True,
        device: torch.device | str | None = None,
    ):
        from transformers import AutoModelForVision2Seq
        import os, copy, torch
        path = Path(path)
        device = device or self.device

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # 1) 定位目录
        if (path / "agent_merged_for_eval" / "config.json").exists():
            merged_dir = path / "agent_merged_for_eval"
            extra_path = path / "agent_extra_layers.pt"
        elif (path / "config.json").exists():
            merged_dir = path
            extra_path = path.parent / "agent_extra_layers.pt"
        else:
            raise FileNotFoundError(f"未找到合并模型的 config.json: {path}")

        print(f"[load] 读取合并主干: {merged_dir}", flush=True)

        # 2) CPU 上加载骨干（无 lm_head）
        self.vla = AutoModelForVision2Seq.from_pretrained(
            str(merged_dir),
            torch_dtype=None,            # 先不设 dtype
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            device_map="cpu",
        )
        self.vla.vision_backbone.set_num_images_in_input(2)
        print("[load] 主干加载完成", flush=True)

        # 3) 解绑 tie
        if hasattr(self.vla.config, "tie_word_embeddings"):
            self.vla.config.tie_word_embeddings = False
        if hasattr(self.vla, "tie_weights"):
            self.vla.tie_weights = lambda *a, **k: None

        # 4) 安装精简头 + 恢复外挂头 + 恢复 norm_stats（都先在 CPU）
        if extra_path.exists():
            print(f"[load] 读取 extra 层: {extra_path}", flush=True)
            sd = torch.load(extra_path, map_location="cpu")

            # lm_head
            old_head = self.vla.language_model.lm_head
            in_features = old_head.in_features
            slim_head = nn.Linear(in_features, self.n_action_bins, bias=(old_head.bias is not None))
            slim_head.load_state_dict(sd["lm_head"], strict=True)
            self.vla.language_model.lm_head = slim_head

            # 外挂头
            self.value_head.load_state_dict(sd["value_head"], strict=strict)
            self.attn_pool.load_state_dict(sd["attn_pool"], strict=strict)
            if "proprio_projector" in sd and getattr(self, "proprio_projector", None) is not None:
                self.proprio_projector.load_state_dict(sd["proprio_projector"], strict=strict)

            # norm_stats
            loaded_norm_stats = sd.get("norm_stats", None)
            if loaded_norm_stats is not None:
                setattr(self.vla, "norm_stats", loaded_norm_stats)
            print("[load] extra 层加载完成", flush=True)
        else:
            # 没有 extra 也要保证头尺寸正确（装空的精简头）
            old_head = self.vla.language_model.lm_head
            in_features = old_head.in_features
            self.vla.language_model.lm_head = nn.Linear(in_features, self.n_action_bins, bias=(old_head.bias is not None))
            print(f"[load] 未找到 extra 层文件（已安装空的精简 lm_head）: {extra_path}", flush=True)

        # 确保能返回 hidden_states
        if hasattr(self.vla.config, "output_hidden_states"):
            self.vla.config.output_hidden_states = True

        # 5) 统一搬到目标 device/dtype
        print(f"[load] 搬运到设备: {device}, dtype={keep_dtype}", flush=True)
        self.vla = self.vla.to(device=device, dtype=keep_dtype)
        self.value_head = self.value_head.to(device, dtype=keep_dtype)
        self.attn_pool  = self.attn_pool.to(device, dtype=keep_dtype)
        if hasattr(self, "proprio_projector") and self.proprio_projector is not None:
            self.proprio_projector = self.proprio_projector.to(device, dtype=keep_dtype)

        # 6) 兜底：如果 cfg.unnorm_key 不在 norm_stats，尽量做一次合理 fallback（并提示）
        ns = getattr(self.vla, "norm_stats", None)
        if ns is None or not isinstance(ns, dict):
            print("⚠️ 注意：模型内未发现 norm_stats，将创建空字典；可能影响动作反归一化。")
            setattr(self.vla, "norm_stats", {})
            ns = self.vla.norm_stats

        target_key = getattr(self.cfg, "unnorm_key", None)
        if target_key and target_key not in ns:
            # 尝试 fallback：优先 default / fallback 之类；否则取第一个 key
            cand = None
            for k in ("default", "libero_default", "fallback"):
                if k in ns:
                    cand = k
                    break
            if cand is None and len(ns) > 0:
                cand = next(iter(ns.keys()))
            if cand is not None:
                print(f"⚠️ unnorm_key='{target_key}' 不在 norm_stats 中，临时回退为 '{cand}'。可在 cfg 中改成此 key。")
                # 不直接改 cfg，保持仅运行时替代（你的 check_unnorm_key 会读 model.norm_stats）
                # 如果你强依赖 cfg.unnorm_key，且 check 是 assert，可以考虑：self.cfg.unnorm_key = cand
            else:
                # 没有任何可用项，给出详细提示
                ks = list(ns.keys())
                raise AssertionError(
                    f"Action un-norm key '{target_key}' 不在 VLA.norm_stats 中，且找不到任何可用项。"
                    f" 请确认训练时保存的 norm_stats 已随 extra 写入。当前可用 keys: {ks}"
                )

        self.vla.eval()
        print("✅ 合并模型与 extra 层加载完成，可用于评估/推理。", flush=True)

if __name__ == "__main__":
    import sys
    import numpy as np

    # Libero env wrapper and helpers
    from rl.libero_env import LiberoEnvWrapper
    from rl.utils import prepare_one_obs, check_unnorm_key
    from experiments.robot.libero.libero_utils import GenerateConfig, TaskSuite

    # Precision policy to match the example
    USE_BF16: bool = True
    TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    # 在这里设置要并行处理的环境数量
    ENVS_ID = list(range(10))
    envs_num = len(ENVS_ID)
    BENCHMARK = TaskSuite.LIBERO_SPATIAL
    unnorm_key = f"{BENCHMARK}_no_noops"
    checkpoint_base_dir = ""
    goal_checkpoint="/cpfs01/liuwei_workspace/models/finetune_im/goal_no_noops_resume+libero_goal_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state"
    object_checkpoint="/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_object_no_noops+b40+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--100000_chkpt"
    four_suites_checkpoint = "/cpfs01/liuwei_workspace/models/finetune_im/openvla-7b+libero_4_task_suites_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--discrete_acts--proprio_state--4tasks--70000_chkpt"
    
    # Instantiate config
    cfg = GenerateConfig(
        pretrained_checkpoint=four_suites_checkpoint,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
        device="cuda:4",
    )

    # 创建策略
    actor = ActorCritic(cfg, TORCH_DTYPE)

    # # #==保存与加载模型==
    # print("\n 模型初始化完成。开始测试 save_model ...")
    # # === 调用保存函数 ===
    # actor.save_model("./runs/rl_models", epoch=0)
    # print("\n save_model 测试完成！")

    # # #=== 调用加载函数 ===
    print("\n 模型保存完毕，开始测试 load_model ...")
    # lora_dir = ""
    # actor.safe_load_model("/cpfs01/liuwei_workspace/models/finetune_rl/agent_checkpoint_epoch_6000", strict=True)
    print("\n load model 测试完成！")

    # == 合并 LoRA 测试 ==
    # print("\n 开始测试 merge_and_unload_lora ...")
    # actor.load_lora_and_merge_for_eval(
    #     checkpoint_dir="./runs/rl_models/agent_lora_epoch_30000",
    #     keep_dtype=TORCH_DTYPE,
    #     strict=True,)
    # print("\n merge_and_unload_lora 测试完成！")

    # == 加载合并模型测试 ==
    # print("\n 开始测试 load_merged_model_for_eval ...")
    # actor.load_merged_model_for_eval(
    #     path="./runs/rl_models/agent_checkpoint_epoch_0",
    #     keep_dtype=TORCH_DTYPE,
    #     strict=True,
    #     device=actor.device,)
    # print("\n load_merged_model_for_eval 测试完成！")
 
    parameter_groups = actor.get_parameter_groups()
    check_unnorm_key(cfg, actor.vla)
    actor.eval()

    # 检查参数类型
    for key, value in actor.named_parameters():
        if value.dtype != TORCH_DTYPE:
            print(f"Warning: Parameter {key} has dtype {value.dtype}, expected {TORCH_DTYPE}.")
    print("策略初始化完成。")

    # ===== SwanLab & Excel setup =====
    # 识别预训练 checkpoint 名称与 LoRA 开关
    pretrained_str = str(cfg.pretrained_checkpoint)
    # 使用最后一级目录名以避免过长的项目名
    pretrained_name = os.path.basename(pretrained_str.rstrip("/")) or pretrained_str
    lora_flag = bool(getattr(cfg, "use_lora", False))
    lora_tag = "lora" if lora_flag else "nolora"

    # SwanLab 初始化（可选）
    if SWANLAB_AVAILABLE:
        try:
            # 使用简短的项目名：inference_{basename(pretrained_checkpoint)}_{lora}
            project_name = f"inference_{pretrained_name}_{lora_tag}"
            exp_name = f"libero_discrete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            SWANLAB_RUN = swanlab.init(
                project=project_name,
                experiment_name=exp_name,
                description=f"original_pretrained_checkpoint={pretrained_str}",
                config={
                    "pretrained_checkpoint": pretrained_str,
                    "use_lora": lora_flag,
                    "lora_rank": getattr(cfg, "lora_rank", None),
                    "benchmark": str(BENCHMARK),
                    "unnorm_key": str(cfg.unnorm_key),
                    "num_images_in_input": cfg.num_images_in_input,
                    "use_proprio": cfg.use_proprio,
                    "device": cfg.device,
                    "envs_num": envs_num,
                },
            )
        except Exception as e:
            print(f"SwanLab 初始化失败: {e}")

    # Excel/CSV 结果记录
    records: List[Dict[str, Any]] = []
    out_dir = Path("./swanlog")
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = out_dir / f"inference_{pretrained_name}_{lora_tag}.xlsx"
    csv_path = out_dir / f"inference_{pretrained_name}_{lora_tag}.csv"

    # 初始化环境
    print(f"正在初始化 {len(ENVS_ID)} 个并行的 Libero 环境...")
    envs = [
        LiberoEnvWrapper(
            benchmark_name=BENCHMARK,
            task_id=env_id,
            image_size=224,
            render_mode="rgb_array",
        )
        for env_id in ENVS_ID
    ]
    print("所有环境初始化完成。")

    # 全局统计
    total_episodes_finished = 0
    total_successes = 0

    from collections import deque

    # 初始化每个环境的动作队列
    env_queues = [deque() for _ in range(len(ENVS_ID))]  # ENVS_ID是环境ID列表

    # 主循环
    while True:
        # 初始化环境状态
        observations = []
        task_descriptions = []
        for i, env in enumerate(envs):
            obs, info = env.reset(seed=int(time.time()) + i)
            observations.append(obs)
            task_descriptions.append(env.task_description)
            print(f"环境 {i}: 任务 ID = {env.task_id}, 任务描述 = {env.task_description}")
            env_queues[i].clear()  # 重置该环境的动作队列

        # 跟踪变量
        active_envs = [True] * envs_num
        total_rewards = [0.0] * envs_num
        episode_steps = [0] * envs_num
        success_info = [False] * envs_num

        print(f"\n开始第 {total_episodes_finished // envs_num + 1} 轮并行执行...")

        # 环境执行循环
        while any(active_envs):
            # 1. 收集需要生成新动作的环境（队列为空且活跃的环境）
            need_generation_indices = []  # 需要生成新动作的环境索引
            inputs_t_list = []  # 需要生成新动作的环境输入
            
            for i in range(envs_num):
                if active_envs[i] and len(env_queues[i]) == 0:
                    inputs_t = prepare_one_obs(cfg, actor.processor, observations[i], task_descriptions[i], TORCH_DTYPE)
                    inputs_t_list.append(inputs_t)
                    need_generation_indices.append(i)
            
            # 2. 为需要生成新动作的环境批量生成动作
            """
            inputs_t:<class 'transformers.image_processing_utils.BatchFeature'>
            input_ids torch.Size([1, 92])
            attention_mask torch.Size([1, 92])
            pixel_values torch.Size([1, 12, 224, 224])
            labels torch.Size([1, 92])
            proprio (8,)
            """ 
            if inputs_t_list:
                inputs_batch = actor.prepare_inputs_batch(inputs_t_list)
                """
                input_ids torch.Size([10, 93])
                attention_mask torch.Size([10, 93])
                pixel_values torch.Size([10, 12, 224, 224])
                labels torch.Size([10, 93])
                proprio torch.Size([10, 8])
                """
                with torch.inference_mode():
                    action_logits, _ = actor.forward(inputs_batch)
                B = action_logits.size(0)
                deterministic_flags = [True] * B  # 若需贪心推理，改为 [True] * B
                _, _, normalized_actions = actor.post_process(action_logits, deterministic_flags)  # 形状 (B, 8, 7)
                                
                # 将生成的动作序列添加到对应环境的队列中
                for idx, env_idx in enumerate(need_generation_indices):
                    # 获取该环境生成的所有动作（8个）
                    action_sequence = normalized_actions[idx]  # 形状 (8, 7)
                    
                    # 将整个动作序列添加到队列
                    env_queues[env_idx].extend(action_sequence)  # 使用extend批量添加
            
            # 3. 执行动作（所有活跃环境）
            for i in range(envs_num):
                if not active_envs[i]:
                    continue  # 跳过非活跃环境
                    
                # 确保队列中有动作（如果没有，说明前面的生成动作步骤有问题）
                if len(env_queues[i]) == 0:
                    print(f"错误：环境 {i} 动作队列为空但未生成新动作")
                    continue
                    
                # 从队列中取出动作
                """
                归一化到 [-1, 1] 的 7 维向量
                """
                action_norm = env_queues[i].popleft()  # array([0.934, 0.469, -0.003, 0.000, 0.000, 0.000, 0.996])
                
                # 将归一化动作转换为环境动作
                """
                将归一化动作从 [-1, 1] 映射回环境期望的物理尺度/单位
                """
                action_env = actor.vla._unnormalize_actions(action_norm, cfg.unnorm_key)
                
                # 执行动作
                obs, reward, terminated, truncated, info = envs[i].step(action_env)
                
                # 更新状态
                observations[i] = obs
                """
                - 累积每个环境当前 episode 的总回报（return），而不是只记录本步奖励
                """
                total_rewards[i] += float(reward)
                episode_steps[i] += 1
                
                # 定期打印
                if episode_steps[i] % 50 == 0:
                    print(f"环境 {i}, Step: {episode_steps[i]}, 奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
                
                # 检查环境是否完成
                if terminated or truncated:
                    is_success = info.get('is_success', False)
                    total_successes += is_success
                    total_episodes_finished += 1
                    success_info[i] = is_success
                    
                    print("-" * 40)
                    print(f"环境 {i} 已完成 (任务: {envs[i].task_description[:50]}...)")
                    print(f"总步数: {episode_steps[i]}, 总奖励: {total_rewards[i]:.4f}, 是否成功: {is_success}")
                    print(f"成功率: {total_successes/total_episodes_finished:.3f}, 总回合数: {total_episodes_finished}")
                    print("-" * 40)
                    
                    # 重置环境
                    active_envs[i] = False
                    episode_steps[i] = 0
                    total_rewards[i] = 0
                    obs, info = envs[i].reset(seed=random.randint(0, 1000))
                    observations[i] = obs
                    env_queues[i].clear()  # 重置动作队列

        # 每轮结束后打印统计信息
        print("=" * 60)
        print(f"第 {total_episodes_finished // envs_num} 轮完成!")
        print(f"累计总回合数: {total_episodes_finished}, 成功次数: {total_successes}")
        print(f"总体成功率: {total_successes/total_episodes_finished:.3f}")
        print("=" * 60)

        # ===== 每5轮记录到 SwanLab 与 Excel/CSV =====
        round_idx = total_episodes_finished // envs_num
        # 当前轮的成功次数（基于 success_info）
        current_round_successes = sum(1 for s in success_info if s)
        current_round_success_rate = current_round_successes / float(envs_num)
        global_success_rate = total_successes / float(total_episodes_finished) if total_episodes_finished > 0 else 0.0

        if round_idx > 0 and round_idx % 5 == 0:
            # SwanLab 记录
            if SWANLAB_AVAILABLE and SWANLAB_RUN is not None:
                try:
                    SWANLAB_RUN.log({
                        "round_idx": round_idx,
                        "round_success_rate": current_round_success_rate,
                        "round_successes": current_round_successes,
                        "global_success_rate": global_success_rate,
                        "total_successes": total_successes,
                        "total_episodes": total_episodes_finished,
                    }, step=round_idx)
                except Exception as e:
                    print(f"SwanLab 记录失败: {e}")

            # 组装记录并写入 Excel/CSV
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "round_idx": round_idx,
                "round_success_rate": current_round_success_rate,
                "round_successes": current_round_successes,
                "global_success_rate": global_success_rate,
                "total_successes": total_successes,
                "total_episodes": total_episodes_finished,
                "pretrained_checkpoint": pretrained_str,
                "use_lora": lora_flag,
                "lora_rank": getattr(cfg, "lora_rank", None),
                "benchmark": str(BENCHMARK),
                "unnorm_key": str(cfg.unnorm_key),
                "envs_num": envs_num,
            }
            records.append(record)

            # 保存记录（优先 Excel，失败或无 pandas 则保存为 CSV）
            saved_path = None
            if PANDAS_AVAILABLE:
                try:
                    df = pd.DataFrame(records)
                    df.to_excel(excel_path, index=False)
                    saved_path = excel_path
                except Exception as e:
                    print(f"写入 Excel 失败，降级为 CSV: {e}")
            if saved_path is None:
                try:
                    write_header = not csv_path.exists()
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(record)
                    saved_path = csv_path
                except Exception as e:
                    print(f"保存记录失败: {e}")
            if saved_path is not None:
                print(f"✅ 记录已保存: {saved_path}")