# LLaVA3D Deep Fusion 动作建模改造方案（Dev 版）

## 🎯 当前进度总览（最后更新：2024-12-30）

### 整体完成度：✅ 100% 🎉

#### ✅ 已完成（步骤 0-4）全部完成！
- ✅ **步骤 0**：创建 dev 副本文件（`*_dev.py`）
- ✅ **步骤 1**：搭建 `LLaVA3DWithActionExpertModel` 最小版本框架
- ✅ **步骤 2**：实现双流联合注意力（Deep Fusion 核心） ⭐
- ✅ **步骤 3**：重写 `FlowMatchingActionExpert` 为 LLaVA3D-based 封装 ⭐
- ✅ **步骤 4**：改造 `MapAnythingLlava3DForConditionalGeneration` 接入新专家 ⭐

#### 📝 后续任务
- ⏳ 集成测试与端到端验证
- ⏳ 性能优化（KV cache、gradient checkpointing）
- ⏳ 单元测试编写

#### 🚀 下一阶段实现计划（对标 PI0，待落地）
- [ ] **联合 attention mask / position_ids**：在 wrapper 中根据 prefix_len 与 suffix_len 构造并下发 joint 2D/4D mask 与连续 position_ids，Flow Expert/WithExpert 全链路传递。
- [ ] **Suffix-only 推理复用 prefix KV cache**：先 prefix-only 缓存，再多步去噪时用 prefix KV + 当前 suffix 做注意力，避免重复算 prefix。
- [ ] **清理冗余日志**：移除 `get_image_features` 中的 debug print，保证训练/推理日志干净。
- [ ] （可选）**时间采样/数值策略对齐 PI0**：如需，改用 Beta(1.5,1.0) 采样 t，并调优 ODE 步长策略。
- [ ] （可选）**梯度检查点与单测**：为双流循环添加 checkpoint；补充最小单测覆盖三种前向和 mask 长度一致性。

---

## 一、背景与动机

- 当前系统结构：
  - 视觉：SigLIP 视觉塔。
  - 几何：MapAnythingWrapper 几何模型。
  - 语言：LLaVA3D（底层是 LlavaLlama 或 LlavaMistral），包装于 `LLaVA3DForCausalLMV2`。
  - 动作：Flow Matching 动作专家 `FlowMatchingActionExpert`（当前基于 Gemma）。
- 现状是 **Late Fusion**：
  - MapAnything wrapper 将图像和几何特征融合后注入 LLaVA3D；
  - 得到 VLM 最后一层 `last_hidden_state` 后，再送入 Gemma‑based Action Expert 做 Flow Matching。
- 目标是迁移到类似 OpenPI / PI0 的 **Deep Fusion** 设计：
  - 在 PI0 中，PaliGemma + Gemma Expert 共享同类 Transformer 底座，前缀（视觉+语言）和后缀（状态+动作+时间）在每层都有交互；
  - 在本项目中，希望用 **LLaVA3D 作为统一 Transformer 基座**，构造 `LLaVA3DWithActionExpertModel`，前缀是图像+几何+文本，后缀是状态+动作+时间，训练与推理都走同一 Deep Fusion 路径。
- 要求：
  - 不修改现有稳定实现，所有结构性改动都在 `*_dev.py` 副本中进行。

## 二、当前相关文件与角色

### 1. 多模态 + 几何 Wrapper

- 生产版  
  - 路径：`SpatialVLA_llava3d/model/modeling_mapanything_llava3d.py`  
  - 类：`MapAnythingLlava3DForConditionalGeneration`
  - 作用：
    - 管理 SigLIP Vision Tower：`self.vision_tower`。
    - 管理 MapAnything 几何模型：`self.geometric_model`。
    - 管理 LLaVA3D 文本模型：`self.language_model`（`LLaVA3DForCausalLMV2`）。
    - `get_image_features`：从 SigLIP 和 MapAnything 得到特征并融合为 `[B, S, H_llm]`。
    - `forward`：
      - 将 image token 特征注入到 `<image>` 位置；
      - 调用 `self.language_model(...)` 获得 logits/hidden_states；
      - 如 `actions is not None and self.action_expert is not None`：
        - 从 `outputs.hidden_states[-1]` 取 hidden；
        - 调用 `self.action_expert.compute_loss(last_hidden_state, actions)` 计算 `action_loss`，覆盖语言 loss。
    - `predict_action`：
      - 再走一遍 wrapper `self(...)` 得到 hidden_states；
      - 调用 `self.action_expert.sample_actions(last_hidden_state)` 预测动作；
      - 无 action_expert 时使用 legacy 自回归动作 token 生成。

- 开发版  
  - 路径：`SpatialVLA_llava3d/model/modeling_mapanything_llava3d_dev.py`  
  - 结构与生产版基本一致，是本次 Deep Fusion 改造的主要入口。

### 2. LLaVA3D 文本模型包装器

- 路径：`SpatialVLA_llava3d/model/modeling_llava3d_v2.py`  
- 类：`LLaVA3DForCausalLMV2`
- 作用：
  - 封装 LlavaLlama / LlavaMistral 为统一接口 `self.model`。
  - 提供 `forward`、`get_input_embeddings`、`prepare_inputs_for_generation` 等。
  - 目前仅支持 **单流** 输入：`input_ids` 或 `inputs_embeds`，尚无前缀/后缀双流联合逻辑。

### 3. Flow Matching 动作专家（Late Fusion 版）

- 路径：`SpatialVLA_llava3d/model/modeling_flow_expert.py`  
- 类：`FlowMatchingActionExpert(GemmaPreTrainedModel)`
- 作用：
  - 使用 `GemmaModel` 作为 Transformer 底座。
  - 输入：
    - `context_features`：来自 VLM 的 `[B, S_ctx, vlm_hidden_dim]`。
    - `actions`：`[B, H, action_dim]`。
    - `time`：`[B]`。
  - 将 `context_features` 映射到 Gemma hidden dim，再与 time embedding、action embedding 拼接送入 Gemma。
  - `compute_loss` 实现 Flow Matching：
    - 采样 `t ~ U(0,1)` 与噪声 `noise`；
    - 构造 `x_t = t * noise + (1 - t) * actions`；
    - 目标速度 `u_t = noise - actions`；
    - 预测速度 `pred_velocity` 并用 MSE 对齐。
  - `sample_actions` 实现 Euler 去噪。

### 4. PI0 / PaliGemma 深度交互实现（参考）

- PI0 主流程：  
  - 路径：`mapAnythingLlava3dPi0.5/openpi/models_pytorch/pi0_pytorch.py`  
  - 类：`PI0Pytorch`
  - 关键：
    - `embed_prefix`：经 `PaliGemmaWithExpertModel.embed_image` 与 `embed_language_tokens` 构造视觉+语言前缀 embedding；
    - `embed_suffix`：用 `state`、`noisy_actions`、`time` 构造动作后缀 embedding；
    - `forward`：
      - prefix_embs + suffix_embs 合并，构造 pad_masks、att_masks；
      - 调用 `PaliGemmaWithExpertModel.forward(inputs_embeds=[prefix_embs, suffix_embs], ...)`；
      - 从 suffix_out 中得到动作流隐藏表示，线性映射到动作速度。

- 深度交互底座：  
  - 路径：`mapAnythingLlava3dPi0.5/openpi/models_pytorch/gemma_pytorch.py`  
  - 类：`PaliGemmaWithExpertModel`
  - 关键逻辑：
    - 内部持有 `self.paligemma` 和 `self.gemma_expert`。
    - `forward(inputs_embeds=[prefix, suffix])` 时：
      - 若只有 prefix 或只有 suffix，则分别单流前向；
      - 若二者都有：
        - 对每层：
          - prefix 和 suffix 分别做 LN + q/k/v 投影；
          - 在 seq 维度拼接 q/k/v，统一做注意力；
          - 再按长度拆回两路，各自做 o_proj、FFN 与残差；
        - 最终分别做 final norm 得到 prefix_output 与 suffix_output。

这正是我们希望在 LLaVA3D 上复刻的双流 Deep Fusion 模式。

## 三、目标设计概述

总体目标：

1. 使用 **LLaVA3D 的 Transformer block 作为统一基座**（不再使用 Gemma 作为动作专家的底座）。
2. 构造 `LLaVA3DWithActionExpertModel`，功能类似 `PaliGemmaWithExpertModel`：
   - 前缀（prefix）：SigLIP 图像特征 + MapAnything 几何特征 + 文本 token。
   - 后缀（suffix）：状态（state）+ 动作（noisy actions）+ 时间（t）。
   - 在每一层都实现 prefix/suffix 的联合注意力（deep fusion）。
3. Flow Matching 的“算法层”（t、noise、x_t、u_t、Euler 迭代）保持不变，但不再自带 Gemma，而是调用 `LLaVA3DWithActionExpertModel` 进行前向。
4. 所有结构性改动仅在 dev 副本文件中进行：
   - `modeling_flow_expert_dev.py`
   - `modeling_llava3d_v2_dev.py`
   - `modeling_mapanything_llava3d_dev.py`

## 四、dev 文件的目标与关系

### 1. `modeling_llava3d_v2_dev.py` ✅

在当前 `LLaVA3DForCausalLMV2` 基础上新增：

- 新类：`LLaVA3DWithActionExpertModel`
- **当前实现状态**（✅ 步骤2已完成 - 2024-12-30）：
  - ✅ **已实现完整 Deep Fusion 逻辑**：
    - 接口为 `forward(attention_mask, position_ids, past_key_values, inputs_embeds=[prefix_embs, suffix_embs], use_cache, expert_cond, output_hidden_states)`；
    - **prefix-only 模式**：直接调用 `base_llava.model` 前向，返回 `prefix_output` 与可选 `past_key_values`，用于语言生成和构建前缀缓存；
    - **suffix-only 模式**：使用完整的 LLaVA3D 层结构处理 suffix，逐层前向（含 LayerNorm、Self-Attention、MLP、残差），返回 `suffix_output`；
    - **prefix+suffix 联合模式**（Deep Fusion 核心）：
      - 实现了 `_compute_layer_complete` 方法，类似 PI0 的 `compute_layer_complete`；
      - 对每一层：
        - prefix 和 suffix 分别做 `input_layernorm` 和 QKV 投影；
        - 在序列维度拼接 Q/K/V，统一应用 RoPE；
        - 计算联合注意力（prefix 和 suffix 互相可见）；
        - 按长度拆回两路，各自做 `o_proj`、第一残差、`post_attention_layernorm`、`mlp`、第二残差；
      - 支持 LLaMA 和 Mistral 两种架构；
      - 最终对两路分别做 final norm。
  - ⚠️ **部分实现（待优化）**：
    - suffix-only 模式目前未使用 prefix KV cache，而是独立前向；
    - 完整的 KV cache 管理（用于推理加速）待后续优化；
    - gradient checkpointing 支持待添加。
  
- **实现细节**：
  - 自动检测 LLaVA3D 的底层模型类型（LLaMA 或 Mistral）；
  - 复用 LLaVA3D 的层结构（`self.base_model.layers`）处理 prefix；
  - Expert 流共享相同的层权重（参数共享），但有独立的 final norm；
  - 支持可选的 expert 投影层（如果 expert_hidden_size 与 base 不同）；
  - 实现了模型类型无关的 RoPE 应用和注意力计算接口。

- **关系**：
  - 作为 Deep Fusion 底座，被 dev 版 Flow Expert 与 MapAnything wrapper 调用。
  - 原 `LLaVA3DForCausalLMV2` 保持不动，用于普通语言生成。

### 2. `modeling_flow_expert_dev.py` ✅

从 `modeling_flow_expert.py` 复制而来，**当前实现状态**（✅ 步骤3已完成 - 2024-12-30）：

- ✅ **已完成完整 LLaVA3D 集成**：
  - 删除了所有对 Gemma 的依赖（不再继承 `GemmaPreTrainedModel`）；
  - 改为调用 `LLaVA3DWithActionExpertModel` 进行网络前向；
  - 完整保留 Flow Matching 数学逻辑（`compute_loss`、`sample_actions`、Euler ODE）。
  
- **实现细节**：
  - **初始化**：
    - 接收 `llava_with_expert_model` 实例（`LLaVA3DWithActionExpertModel`）；
    - 配置 `action_dim`, `action_horizon`, `state_dim`, `use_state`；
    - 创建 suffix embedding 层：`state_proj`, `action_in_proj`, `time_mlp_in/out`, `action_out_proj`。
  
  - **核心方法**：
    - `_construct_suffix_embeddings(actions, time, state)`: 构造 suffix embeddings
      - 结构：[state_token?, action_tokens, time_token]
      - State token: 可选，通过 `state_proj` 投影
      - Action tokens: 每个 action step 投影为 hidden_size
      - Time token: sinusoidal embedding + MLP
    
    - `forward(prefix_embs, actions, time, state, attention_mask, position_ids)`: Deep Fusion 前向
      - 构造 suffix embeddings
      - 调用 `llava_with_expert(attention_mask, position_ids, [prefix_embs, suffix_embs])` 进行联合前向
      - 从 suffix_output 提取 action tokens
      - 通过 `action_out_proj` 预测 velocity
    
    - `compute_loss(prefix_embs, actions, state)`: Flow Matching 训练
      - 采样 t ~ Uniform(0, 1) 和 noise ~ N(0, I)
      - 构造 x_t = t * noise + (1 - t) * actions
      - 目标 u_t = noise - actions
      - 预测 v_t = forward(prefix_embs, x_t, t, state)
      - 计算 MSE(v_t, u_t)
    
    - `sample_actions(prefix_embs, state, num_steps)`: Euler ODE 采样
      - 初始化 x_t ~ N(0, I) (t=1)
      - 循环 num_steps 步：
        - 预测 v_t = forward(prefix_embs, x_t, t, state)
        - Euler 步：x_t = x_t + v_t * dt
      - 返回 x_0（clean actions）

- **Flow Matching 数学**：
  - 保留了完整的 Flow Matching 公式和 Euler ODE solver
  - 时间编码使用 sinusoidal position embedding
  - 支持可选的 proprioceptive state 输入

关系：

- 被 `modeling_mapanything_llava3d_dev.py` 使用。
- 与 `LLaVA3DWithActionExpertModel` 紧耦合（Deep Fusion）。
- 与具体视觉/几何模块解耦（仅接收 prefix_embs）。

### 3. `modeling_mapanything_llava3d_dev.py`

dev 版多模态+几何 wrapper 当前状态：

- 已从 `modeling_flow_expert_dev` 导入 `FlowMatchingActionExpert`，与生产版解耦；
- 在初始化中已构造：
  - `self.language_model`：`LLaVA3DForCausalLMV2` 实例；
  - `self.language_model_with_expert = LLaVA3DWithActionExpertModel(self.language_model)`；
- 目前仍使用 Gemma 版 Flow Expert 做 Late Fusion：
  - 从 `outputs.hidden_states[-1]` 取 VLM hidden 作为 `context_features`；
  - 调用 `self.action_expert.compute_loss(last_hidden_state, actions)` / `sample_actions(last_hidden_state)`。

dev 版多模态+几何 wrapper 的目标：

- 构造 prefix_embs：
  - 使用 `get_image_features(pixel_values, intrinsic)` 得到 `[B, S_v, H_llm]`；
  - 与文本 embedding 合并成统一的 prefix 序列（可按 `<image>` token 位置注入或直接拼接）。
- 构造 suffix_embs：
  - 使用 `state`、`x_t`（noisy actions）、`time` 构造后缀 token embedding；
  - 包括 state 投影、action 投影、time MLP 等。
- 调用 `LLaVA3DWithActionExpertModel`：
  - 训练时：走 Flow Matching 路径（prefix+suffix），由 FlowMatchingActionExpertDev 封装；
  - 推理时：先 prefix-only 建立缓存，再在 `predict_action` 里循环 suffix 去噪。

关系：

- 顶层多模态模型封装器；
- 向下依赖：
  - SigLIP Vision Tower 与 MapAnythingWrapper（构造视觉/几何前缀）。

---

## 五、小 expert + multi-expert attention 方案（无 adapter 路线）

这一节记录后续计划采用的方案与开发方向，用于指导“小 expert”与多专家联合注意力（multi-expert attention）的实现。

### 5.1 总体选择与约束

我们明确选择的路线是：

- **统一 LLaMA 架构家族**
  - base model：沿用当前 LLaVA3D 的 text backbone（Llama 风格）。
  - expert model：使用同一类 LLaMA block 作为结构模板，只是 width/层数更小。
- **无显式 adapter**
  - 不在 attention 前后额外挂 `Linear(D_expert -> 4096)` 这种 adapter；
  - 而是在多专家 attention 内部，通过各自的 qkv/o_proj 将不同宽度的 hidden 映射到同一个 head 空间。
- **统一 joint attention 头空间**
  - 由 base LLaVA3D 决定联合注意力的公共几何空间：
    - `hidden_dim_fusion = 4096`
    - `num_heads_fusion = 32`
    - `head_dim_fusion = 128`
    - `num_kv_heads_fusion = 32`
    - RoPE：与 base 完全一致（theta=10000 等）。
  - 所有参与 joint attention 的 expert（base + action expert）在 head 维度上必须与上述配置一致。

换句话说：

- expert 可以在「自身 hidden 宽度」和「层数」上变小；
- 但在注意力头的定义上，必须完全与 base 一致，保证 joint attention 时 Q/K 落在同一个几何空间。

### 5.2 小 expert 设计原则

小 expert 的角色是：

- 在相同的 LLaMA 结构配方下，作为 **更窄、更浅的动作/后缀专用分支**；
- 在每一层中与 base 前缀一起参与 joint attention；
- 不再单独承担语言建模任务。

设计原则：

- 结构家族
  - 直接沿用 LLaVA3D base 使用的 LLaMA block（RMSNorm、RoPE、自注意力、MLP 的组合形式）。
  - 不额外引入 TinyLlama/SmolLM2/Qwen 等异构实现，仅参考其「比例感」，不直接复用代码。
- 宽度与深度
  - `hidden_size_expert` < `hidden_size_base=4096`，例如可选：
    - 2048、1536 等作为候选宽度；
  - `num_layers_expert` 显著少于 base，例如：
    - base=32 层，小 expert=8–12 层。
- 注意力头
  - `num_attention_heads_expert = num_attention_heads_base = 32`
  - `head_dim_expert = head_dim_base = 128`
  - `num_kv_heads_expert = num_kv_heads_base = 32`
  - RoPE：使用与 base 完全一致的实现与参数。
- 初始化
  - 小 expert 权重可以从随机初始化开始；
  - 后续如需，可引入蒸馏或从 base 某些层派生初始化，但这不是当前阶段必须。

在这种设定下，小 expert 不需要显式 adapter，就可以通过 multi-expert attention 的 qkv/o_proj 投影接入统一的 head 空间。

### 5.3 multi-expert attention 目标形态（对标 openpi）

目标是将 `_compute_layer_complete` 进一步演化为真正的「多专家联合注意力」模块，形式上类似 openpi 的 `PaliGemmaWithExpertModel`：

- 对每一层 `layer_idx`：
  - 有两套（或多套）expert 参数：
    - expert 0：base LLaVA3D 的层参数（用于 prefix）；
    - expert 1：小 expert 的层参数（用于 suffix）。
  - 前向流程：
    1. 对每个 expert 的输入 hidden：
       - 先做各自的输入 norm（如 input_layernorm 或 gated RMSNorm）。
       - 用各自的 q_proj/k_proj/v_proj 将宽度为 `width_i` 的 hidden 映射到统一的 head 空间：
         - `width_i -> num_heads_fusion * head_dim_fusion`。
    2. 将所有 expert 的 Q/K/V 在序列维度 concat：
       - `joint_q = cat([q_prefix, q_suffix, ...], dim=seq)`
       - `joint_k / joint_v` 同理。
    3. 在 joint_q/joint_k 上统一应用 RoPE（使用 base 的实现）。
    4. 调用单次 attention kernel：
       - 例如复用当前 LLaMA 的 eager/flash attention 内核，或写一个专门的 `eager_attention_forward_multi_expert`。
    5. 将注意力输出在 seq 维拆回各 expert 段：
       - 各自通过自己的 `o_proj`（`num_heads_fusion * head_dim_fusion -> width_i`）映射回本 expert 的 hidden 宽度；
       - 再走各自的 post_attention_layernorm + MLP + 残差。

在实现上，这意味着：

- attention 计算阶段所有 expert 共享同一个 head 空间和 RoPE；
- qkv/o_proj 是 per-expert 的，负责把不同宽度的 hidden 与统一 head 空间对接；
- 不需要显式 `Linear(D_expert->4096)` adapter 层，因为这一步已经隐含在 qkv 投影中。

### 5.4 与当前实现的关系

当前 `LLaVA3DWithActionExpertModel` 的 `_compute_layer_complete` 已经具备「prefix/suffix 逐层联合 QKV 拼接 + 单次 attention + 再拆回」的整体形态，并且在 dev 版本中已经完成了同宽多 expert 的第一步改造：

- base 与 expert 分支分别持有独立的一套层参数（两套 LLaMA block），但配置保持一致（`hidden_size_expert = hidden_size_base = 4096`）；
- 在每一层 joint attention 中，prefix 使用 base_layer，suffix 使用 expert_layer，各自做 norm + qkv，再在统一的 head 空间中拼 Q/K/V 做一次 attention，最后拆回并用各自的 o_proj + MLP；
- attention 头空间由 `fusion_hidden_size/num_heads/head_dim` 显式控制，目前设置为与 base 一致，后续可以在保持几何不变的前提下进一步解耦。

后续改造方向：

- 在保持「per-layer 多 expert」结构不变的前提下，引入 `hidden_size_expert != hidden_size_base` 的能力；
- 为 expert 分支定义独立的 q_proj/k_proj/v_proj/o_proj/MLP，使其能够在内部使用更窄的 hidden，同时在 joint attention 时仍投影到统一的 head 空间。

### 5.5 开发计划（面向实现）

下面是面向实现的步骤规划，实际编码仍仅在 `*_dev.py` 中进行：

1. **定义 expert 配置结构**
   - 在 dev 配置或模型初始化中，增加 expert 相关字段：
     - `hidden_size_expert`（默认可先设为 4096，与 base 相同，方便平滑过渡）；
     - `num_layers_expert`（可先等于 base 层数，之后再减少）；
     - `mlp_ratio_expert`（例如与 base 一致，后续可独立调整）。
   - 先实现「同宽但不同参数」的多 expert 版本，再逐步引入 `hidden_size_expert < hidden_size_base`。

2. **重构 `LLaVA3DWithActionExpertModel` 使其支持 per-layer 多 expert**
   - 对 `self.base_model.layers` 的使用进行抽象，增加一套 expert 层参数存储：
     - base：`layers_base[layer_idx]`
     - expert：`layers_expert[layer_idx]`
   - 修改 `_compute_layer_complete`：
     - 不再假定 prefix/suffix 使用同一个 `layer` 实例；
     - 而是对 `[layer_base, layer_expert]` 进行类似 openpi 的 per-expert 处理：
       - 各自做 norm + qkv；
       - 拼接 Q/K/V，统一 attention；
       - 拆回后用各自 `o_proj + MLP`。

3. **在同宽场景下验证多 expert attention 的正确性**
   - 第一步不引入不同 `hidden_size_expert`，只做「参数不共享的 base/expert」：
     - `hidden_size_expert = hidden_size_base = 4096`；
     - `num_layers_expert = num_layers_base`；
   - 确保：
     - prefix-only / suffix-only / joint 三种模式依然工作正常；
     - suffix 分支可以更新自己的参数，而 base 保持冻结；
     - 不引入新的 NaN / OOM 问题。

4. **引入窄一点的 expert 宽度（真正小 expert）**
   - 将 `hidden_size_expert` 改为 2048 或其他小值（默认推荐 2048，`num_layers_expert = num_layers_base`，`mlp_ratio_expert = mlp_ratio_base`）；
   - 为 expert 层定义独立参数（仅在 `hidden_size_expert != hidden_size_base` 时启用）：
     - q_proj_expert/k_proj_expert/v_proj_expert（`hidden_size_expert -> fusion_num_heads * head_dim_fusion`）；
     - o_proj_expert（`fusion_num_heads * head_dim_fusion -> hidden_size_expert`）；
     - MLP_expert（例如 `hidden_size_expert -> 4*hidden_size_expert -> hidden_size_expert`，通过为 expert 单独构造一套 LLaMA block 实现）。
   - 更新 `_compute_layer_complete` 使其支持 base/expert 宽度不同：
     - prefix_hidden ∈ R^{B×L_p×hidden_size_base}；
     - suffix_hidden ∈ R^{B×L_s×hidden_size_expert}；
     - attention 内部统一在 `[B, L_total, fusion_num_heads, head_dim_fusion]` 空间上，prefix 继续使用 base 的 `self_attn` qkv/o_proj，suffix 则通过 per-layer expert_qkv/o_proj 映射进出该空间。

5. **训练策略与稳定性验证**
   - 初期训练策略：
     - 冻结 base LLaVA3D（包括其 RoPE、attention、MLP 等）；
     - 仅训练小 expert +动作头；
   - 验证内容：
     - 权重是否仍出现 NaN（重点关注 expert 的 RMSNorm、MLP）；
     - 相比「参数完全共享的大 expert」，显存与稳定性是否明显改善；
     - Flow Matching 性能与收敛速度情况。

6. **文档与测试**
   - 在 `STEP2_DEEP_FUSION_README.md` 中保持与此方案的高层描述一致；
   - 增补最小测试用例，覆盖：
     - base/expert 同宽多 expert attention；
     - base/expert 不同宽度多 expert attention；
     - prefix-only / suffix-only / joint 三种模式在多 expert 版本下的行为。

7. **对标 starVLA 的动作 expert 搭积木方式（DiT / Flow Matching 参考）**
   - starVLA 在 `starVLA/starVLA/model/modules/action_model` 中，采用高度模块化的方式构建动作专家：
     - 在 `DiT_modules/models.py` 中定义了通用的 DiT backbone：
       - `TimestepEmbedder` / `LabelEmbedder` / `ActionEmbedder` 将时间标量、条件 token、动作序列分别映射到统一的 token 空间；
       - `DiT` 使用标准自注意力 Transformer 对 `[cond_tokens, action_tokens]` 进行建模；
       - `DiTCrossAttn` / `DiTBlockCrossAttn` / `DiTBlockSelfAttn` 支持在 DiT 内部交替堆叠「self-attn + cross-attn」，并通过 `encoder_features` 注入上游 VLM 表示；
     - 在 `flow_matching_head` 子目录中，将 DiT backbone 封装为不同的 Flow Matching 头：
       - `action_encoder.py` 中的 `ActionEncoder`：将 `(actions, t)` 编码为动作 token 序列，形状统一为 `[B, T, hidden_size]`；
       - `cross_attention_dit.py` 中的 `DiT` / `SelfAttentionTransformer`：使用 diffusers 的 `BasicTransformerBlock` 作为可重用「一层 transformer（带可选 cross-attn）」模块，通过 `transformer_blocks = nn.ModuleList([...])` 实现任意深度堆叠；
       - `LayerwiseFM_ActionHeader.py` 中的 `LayerwiseFlowmatchingActionHead`：
         - 从全局 `global_config.framework.qwenvl` 读取 VLM hidden_size / num_layers 等，构造 `diffusion_model_cfg`，并用 `DiT(**diffusion_model_cfg)` 实例化一个与 VLM 层数对齐的动作 backbone；
         - 使用 `ActionEncoder` 将 noisy actions + time 编码到与 VLM hidden 一致的 token 维度；
         - 在 `forward` / `predict_action` 中，遍历 `self.model.transformer_blocks`，对每一层调用：
           - `layer(hidden_states=sa_embs, encoder_hidden_states=vl_embs_list[layer_idx], temb=temb)`；
           - 实现「逐层 cross-attention 到每一层 VLM 表示」的 layerwise Flow Matching；
         - 最后的 `action_decoder` 将 DiT 输出 token 映射回动作维度，并基于 `velocity = actions - noise` 定义 Flow Matching 损失。
   - 这一套设计的关键点在于：
     - 将「时间编码」「动作编码」「条件编码」「DiT backbone」「Flow Matching 损失」完全解耦为可组合的模块；
     - DiT 本身只关心「在统一 token 空间上的自注意力 / cross-attn 堆叠」，输入输出来自上游编码器（动作 / 状态 / VLM 特征）；
     - Flow Matching 头负责选择：
       - 使用哪一种 DiT 变体（纯 self-attn / 交替 self+cross）；
       - 以何种方式从 VLM 收集 layerwise 表示（`vl_embs_list`）并注入到 DiT；
       - 如何编码时间与动作（`ActionEncoder` / `MultiEmbodimentActionEncoder` 等）。
   - 对本项目的启发：
     - 可以将「小 expert + multi-expert attention」看作是类似的「搭积木」体系：
       - 小 expert 的每一层是「LLaMA 风格 DiT block」：支持自注意力、支持接收来自 base 的 encoder_features 做 cross-attn；
       - LLaVA3DWithActionExpertModel 在 `_compute_layer_complete` 中，相当于扮演 starVLA 中的 `LayerwiseFlowmatchingActionHead`：
         - 它对每一层维护 base_layer 和 expert_layer；
         - 通过统一的头空间和 RoPE，将 prefix/suffix/base/expert token 组织成 joint attention；
     - 后续在实现小 expert 时，可以借鉴 starVLA 的拆分方式：
       - 将「时间 + 动作 + 状态」的编码逻辑独立成 suffix encoder（类似 `ActionEncoder + state MLP`）；
       - 将「多层 LLaMA 小 expert」抽象为一个可配置的 ModuleList（类似 `self.transformer_blocks`），方便在 config 中按 depth/hidden_size/head 数控制结构；
       - `_compute_layer_complete` 只关心「如何在统一 head 空间拼/拆 QKV 与输出」，而不关心 expert 内部的时间/动作细节。

本节内容作为后续“小 expert + multi-expert attention”开发的统一参考，实际编码时应严格遵守「统一 head 空间 + 统一 RoPE」这一核心约束。
  - `LLaVA3DForCausalLMV2`（纯语言训练/推理）。
  - `LLaVA3DWithActionExpertModel`（动作 Deep Fusion）。
  - `FlowMatchingActionExpertDev`（Flow Matching 算法封装）。

### 7.1 starVLA DiT 动作头方案的深入分析与本项目对接思路（2026-01 补充）

在 starVLA 中，动作 expert 采用了高度模块化的 DiT 搭积木方式，结构上与本项目“LLaVA3D + 小 expert”存在天然对应关系。本小节对该方案做更细致的拆解，并给出在本项目中落地 DiT‑style expert 的建议实施路径。

#### 7.1.1 starVLA 动作 DiT 的组件拆解

1. 编码层（时间 / 条件 / 动作）
   - 时间编码：
     - 在 `DiT_modules/models.py` 中的 `TimestepEmbedder` 使用正弦位置编码 + 小 MLP 将标量时间 `t` 映射到 `hidden_size`，用于为 DiT 提供时间条件。
     - 在 `flow_matching_head/cross_attention_dit.py` 中，`TimestepEncoder` 复用 diffusers 的 `Timesteps + TimestepEmbedding`，输出维度与 DiT 内部 `inner_dim` 对齐。
   - 条件编码（LabelEmbedder）：
     - `LabelEmbedder` 将条件 token（例如 VLM 特征）线性映射到 `hidden_size`，并带有 classifier‑free guidance 风格的 token dropout，用于在训练中实现部分条件丢弃。
   - 动作编码（ActionEncoder）：
     - 在 `action_encoder.py` 与 `LayerwiseFM_ActionHeader.py` 中的 `ActionEncoder`，通过三层 MLP 和 sinusoidal 时间编码，将 `(actions, t)` 映射为 `[B, T, hidden_size]`，实现动作和时间的紧耦合表示。

2. DiT backbone 本体
   - 基础 DiT（不带 cross‑attn）：
     - `DiT_modules/models.py` 中定义的 `DiT` 是一串 `DiTBlock` 堆叠，每个 block 是“LN + self‑attn + 残差 + LN + MLP + 残差”，隐空间维度为 `token_size`，并带有可学习位置 embedding。
   - cross‑attn DiT（Flow Matching 版）：
     - `flow_matching_head/cross_attention_dit.py` 中的 `BasicTransformerBlock` 使用 diffusers 的 `Attention + FeedForward` 作为核心模块，通过 `cross_attention_dim` 指定是否启用 cross‑attn。
     - `AdaLayerNorm` 将时间 embedding `temb` 作为条件注入到 norm 中，从而实现时间条件化的注意力和 MLP。
     - `DiT`（flow head）内部维护 `transformer_blocks = nn.ModuleList([...])`，每层可以选择 self‑attn 或 cross‑attn，输出端使用 `LayerNorm + Linear` 将 hidden 映射到动作维度。

3. Layerwise Flow Matching 头
   - `LayerwiseFlowmatchingActionHead` 在 `LayerwiseFM_ActionHeader.py` 中实现：
     - 从全局 `global_config.framework.qwenvl` 读取 VLM 的隐藏维度 `vl_hidden_dim` 和层数 `num_vl_layers`，更新 `DiTConfig`：
       - `num_layers = num_vl_layers`；
       - `input_embedding_dim = vl_hidden_dim`；
       - `num_attention_heads = input_embedding_dim // attention_head_dim`。
     - 使用更新后的 `diffusion_model_cfg` 实例化 `self.model = DiT(**diffusion_model_cfg)`，保证 DiT 的层数与 VLM 对齐，hidden 维度与 VLM 完全一致。
     - `ActionEncoder` 将 `(noisy_actions, t)` 编码到 `input_embedding_dim`，`state_encoder` 将 state 编码到同一维度，构造 `[state_tokens?, future_tokens, action_tokens]` 序列。
     - 在 `forward` / `predict_action` 中，遍历 `self.model.transformer_blocks`，对每一层调用：
       - `layer(hidden_states=sa_embs, encoder_hidden_states=vl_embs_list[layer_idx], temb=temb)`；
       - 实现对每一层 VLM 表示的 layerwise cross‑attention。

4. Flow Matching 目标与采样
   - 训练：
     - 构造 `noisy_trajectory = (1 - t) * noise + t * actions`；
     - 目标速度 `velocity = actions - noise`；
     - DiT 输出的动作部分与 `velocity` 之间做 MSE，形成 Flow Matching loss。
   - 采样：
     - 初始化 `actions ~ N(0, I)`；
     - 迭代更新 `actions = actions + dt * pred_velocity`，使用 Euler 积分在时间维度推进。

#### 7.1.2 与当前 LLaVA3D Deep Fusion 方案的结构对应

从结构层面看，starVLA 的 LayerwiseFlowmatchingActionHead 与本项目的 Deep Fusion 有以下对应：

- VLM 层对齐：
  - starVLA：`num_layers = num_vl_layers`，DiT 的每个 block 对应一层 VLM hidden。
  - 本项目：`fusion_blocks = nn.ModuleList([...])`，每个 `DeepFusionBlock` 绑定一层 base_layer + 一层 expert_layer。
- hidden 维与 head 空间：
  - starVLA：`input_embedding_dim = vl_hidden_dim`，DiT 的 `inner_dim = num_heads * head_dim` 与 VLM 一致。
  - 本项目：`fusion_hidden_size = hidden_size`，`fusion_num_heads` 和 `fusion_head_dim` 与 LLaVA3D 完全一致。
- suffix 编码：
  - starVLA：`ActionEncoder + state_encoder` 输出与 VLM hidden 一致的 token。
  - 本项目：`FlowMatchingActionExpert._construct_suffix_embeddings` 输出 `expert_hidden_size` 维度的 `[state_token?, action_tokens, time_token]`。
- 每层与 VLM 表示的耦合方式：
  - starVLA：DiT block 对 `[state/future/action]` 做 self/cross‑attn，`encoder_hidden_states = vl_embs_list[layer_idx]`。
  - 本项目：DeepFusionBlock 通过 joint self‑attn 将 prefix/suffix 混合，然后拆回，两路共享一次 attention。

可以将 starVLA 视为“单流 DiT + layerwise cross‑attn”，而本项目是“双流 LLaMA block + per‑layer joint self‑attn”。两者都满足“动作流在每一层都直接看到 VLM 表示”的设计目标。

#### 7.1.3 在本项目中引入 DiT‑style expert 的建议实施路径

结合当前代码结构和训练脚本，建议按照由浅入深的三阶段路线引入 DiT 动作 expert。

1. 阶段 1：在现有架构外增加并联 DiT 动作头（最小侵入）
   - 目标：
     - 在不改动 `LLaVA3DWithActionExpertModel` / DeepFusionBlock 的前提下，增加一个 starVLA 风格的 DiT 动作头，用于与现有 LLaMA expert 对比；
   - 实施建议：
     - 在 `model/` 新增 dev 版 DiT 动作头（例如 `modeling_dit_action_head_dev.py`），包含：
       - 一个 `SuffixEncoder`：复用或改写 `FlowMatchingActionExpert._construct_suffix_embeddings`，输出维度对齐到 LLaVA3D hidden（必要时增加 Linear 适配）；
       - 一个 `DiTActionHead`：内部持有 `DiT(transformer_blocks)`，`num_layers = base_num_layers`，`input_embedding_dim = hidden_size`，`cross_attention_dim = hidden_size`，前向 API 为 `forward(vl_embs_list, actions, state)`；
     - 在 `MapAnythingLlava3DForConditionalGeneration` 中增加一个可选参数 `use_dit_action_head`：
       - 若开启，则在 forward 中调用 LLaVA3D（或 `language_model_with_expert`）获取 layerwise hidden 列表 `vl_embs_list`，传给 `DiTActionHead` 计算动作 loss；
       - 初期可以只开 DiT 头，不同时训练 Deep Fusion expert，便于单独对比。

2. 阶段 2：统一 suffix encoder，并在 Deep Fusion 与 DiT 之间共享
   - 目标：
     - 避免时间/动作/状态编码逻辑在 `FlowMatchingActionExpert` 与 DiT 动作头之间重复，实现编码模块的复用；
   - 实施建议：
     - 抽象出独立的 `SuffixEncoder` 模块：
       - 输入：`actions`、`state`、`time`；
       - 输出：`suffix_tokens`，维度可配置（expert_hidden_size 或 hidden_size）；
     - `FlowMatchingActionExpert` 与 DiT 动作头均通过该 encoder 构造 suffix token：
       - Deep Fusion 版本：`suffix_tokens` 送入 `LLaVA3DWithActionExpertModel` 的 suffix 流；
       - DiT 版本：`suffix_tokens` 作为 DiT 的 `hidden_states`，`encoder_hidden_states = vl_embs_list[layer_idx]`。

3. 阶段 3：探索 DiT expert 替代或补充 Deep Fusion expert
   - 目标：
     - 在数值稳定和训练经验的基础上，评估以下方案：
       - 完全用 DiT 动作头替代 LLaMA expert（LLaVA3D 只提供 layerwise features）；
       - 或采用混合方案：LLaMA expert 负责部分层的 joint attention，DiT 动作头在 final hidden 上做二次 refinement。
   - 实验建议：
     - 对比以下三种配置：
       - E1：当前 LLaMA 小 expert + FlowMatchingActionExpert（Deep Fusion）；
       - E2：仅 DiT 动作头（以 LLaVA3D layerwise hidden 为 encoder_features）； 
       - E3：Deep Fusion + DiT 混合（例如 Deep Fusion 输出 suffix hidden，再交给小深度 DiT 行为头）。
     - 使用与第 10 章相同的指标（loss、vel_cosine、grad_total_norm、grad_max_abs、`grad_guard` 触发频率）以及显存/速度，对比三种路线的收敛特性。

在工程权衡上：
- DiT 动作头的优势在于：完全解耦于 LLaVA3D 架构，可以较为轻松地迁移到其他 VLM，且可以直接复用 diffusers 的成熟实现；
- 现有 LLaMA expert Deep Fusion 的优势在于：更加贴近 openpi 的 PaliGemma+Gemma 多 expert 设计，RoPE 和 attention 几何空间与 LLaVA3D 完全对齐。

综上，建议先按照阶段 1 的方式将 DiT 动作头作为“外置 expert”引入，作为对照实验头验证其在 Libero/Flow Matching 下的表现，再根据结果决定是否进一步演化为主要 expert 路线或与 Deep Fusion expert 混合使用。

#### 7.1.4 供大模型实现的具体文件与接口规划（只在新副本上改动）

为方便后续让大模型自动实现 DiT‑style expert，本小节明确所有涉及的文件、副本命名和关键接口。约束是：**不修改当前已有代码文件，只通过复制‑重命名的 dev 版本来实现新功能**。

1. 顶层训练脚本副本
- 从现有脚本复制：
  - 源：`scripts/train_llava3d_deepfusion.py`
  - 目标：`scripts/train_llava3d_deepfusion_dit.py`
- 仅在新脚本中修改：
  - 保留原有 CLI 参数，新增：
    - `--use_dit_action_head`：启用 DiT 动作头；
    - `--dit_hidden_size`、`--dit_num_layers`（可选）：覆盖默认 DiT hidden 和层数。
  - 模型构建部分仍使用 `MapAnythingLlava3DConfig` 加载 `config.json`，并：
    - 设置 `config.use_dit_action_head = args.use_dit_action_head`；
    - 如果提供了 `--dit_hidden_size` / `--dit_num_layers`，写入 `config.dit_hidden_size` 与 `config.dit_num_layers`。

2. MapAnything wrapper 副本
- 从现有 dev 版复制：
  - 源：`model/modeling_mapanything_llava3d_dev.py`
  - 目标：`model/modeling_mapanything_llava3d_dit.py`
- 仅在 `_dit.py` 中修改：
  - `__init__` 中：
    - 保留原有 `self.action_expert = FlowMatchingActionExpert(...)`；
    - 当 `config.use_dit_action_head` 为 True 时，额外初始化：
      - `self.dit_action_head = DiTActionHead(..., hidden_size=self.hidden_size, action_dim=config.action_dim, action_horizon=config.action_horizon, state_dim=getattr(config, "state_dim", None))`；
    - 是否同时训练 Deep Fusion expert，可通过一个布尔 config（如 `config.enable_deepfusion_expert`）控制，初始建议先只训练 DiT 动作头以简化对比。
  - `forward` 中，在 `actions is not None` 分支下增加：
    - 若 `use_dit_action_head` 为 True：
      - 调用 LLaVA3D 获取 layerwise 前缀 hidden：
        - 将 `output_hidden_states=True` 传入 `self.language_model` 的 forward；
        - 把每层 `hidden_states[i]` 中对应 prefix 部分切出，组织成 `vl_embs_list: List[Tensor]`，形状约为 `[num_layers][B, L_p, H]`；
      - 调用 `dit_action_head.compute_loss(vl_embs_list, actions, state)` 计算动作损失；
      - 返回 `MapAnythingLlava3DOutput(loss=action_loss, logits=None, ...)`。
  - `predict_action` 中增加对应的 DiT 推理分支：
    - 构造与训练相同的 `vl_embs_list`；
    - 调用 `dit_action_head.predict_action(vl_embs_list, state, num_steps)` 返回 `[B, H, action_dim]`。

3. DiT 动作头与 suffix encoder 新文件
- 新增文件：`model/modeling_dit_action_head_dev.py`（不覆盖任何现有文件）。
- 推荐包含：
  - `SuffixEncoder`：
    - 输入：`actions: [B, H, action_dim]`、`time: [B]`、`state: Optional[B, state_dim]`；
    - 输出：`suffix_tokens: [B, L_s, hidden_size]`，其中 `hidden_size` 对齐 LLaVA3D hidden（如 4096）；
    - 实现可以直接借鉴 starVLA 的 `ActionEncoder`：线性层 + sinusoidal time embedding + MLP，支持可选 state token 和可选位置 embedding。
  - `DiTActionHead`：
    - 构造函数参数（示意）：
      - `hidden_size`（VLM hidden）、`num_layers`（默认等于 LLaVA3D 层数）、`num_heads`、`head_dim`；
      - `action_dim`、`action_horizon`、`state_dim`。
    - 内部成员：
      - `self.dit = DiT(num_attention_heads=num_heads, attention_head_dim=head_dim, output_dim=action_dim, num_layers=num_layers, cross_attention_dim=hidden_size, ...)`；
      - `self.state_encoder`（可选 MLP）：`state_dim -> hidden_size`；
      - `self.action_decoder`：`hidden_size -> action_dim`。
    - 核心方法：
      - `compute_loss(vl_embs_list, actions, state)`：
        - 采样 `t ~ Beta(alpha,beta)` 与 `noise ~ N(0,I)`；
        - 构造 `x_t` 与 `velocity`；
        - 用 `SuffixEncoder` 编码 `(x_t, t, state)`，得到 suffix tokens；
        - 使用 DiT 的每层 block 与 `vl_embs_list[layer_idx]` 做 cross‑attn；
        - 用 `action_decoder` 回归 `pred_velocity`，与 `velocity` 做 MSE；
        - 返回标量 loss。
      - `predict_action(vl_embs_list, state, num_steps)`：
        - 初始化 `actions ~ N(0,I)`；
        - 多步 Euler 更新 `actions += dt * pred_velocity`，每步调用与训练类似的编码和 DiT 前向。

4. 配置类副本
- 如需避免污染现有 `configuration_mapanything_llava3d.py`，可以复制为：
  - 源：`model/configuration_mapanything_llava3d.py`
  - 目标：`model/configuration_mapanything_llava3d_dit.py`
- 在 `_dit.py` 中增加字段：
  - `use_dit_action_head: bool = False`
  - `dit_hidden_size: Optional[int] = None`
  - `dit_num_layers: Optional[int] = None`
- 在 `train_llava3d_deepfusion_dit.py` 中导入 `_dit` 版配置，并将 CLI 参数写入上述字段。

5. 不修改现有代码的约束汇总
- 只能在以下新建/复制文件中编辑：
  - `scripts/train_llava3d_deepfusion_dit.py`
  - `model/modeling_mapanything_llava3d_dit.py`
  - `model/modeling_dit_action_head_dev.py`
  - `model/configuration_mapanything_llava3d_dit.py`（若创建）
- 不允许直接修改：
  - `scripts/train_llava3d_deepfusion.py`；
  - `model/modeling_mapanything_llava3d_dev.py`；
  - `model/modeling_llava3d_v2_dev.py`；
  - `model/modeling_flow_expert_dev.py` 等现有实现。

## 五、实施修改的具体步骤

### 步骤 0：创建 dev 副本（不动生产代码）

目标：保护现有稳定实现。

操作：

- 复制文件：
  - `modeling_flow_expert.py` → `modeling_flow_expert_dev.py`
  - `modeling_llava3d_v2.py` → `modeling_llava3d_v2_dev.py`
- 在 `modeling_mapanything_llava3d_dev.py` 中调整 import（指向 dev 版）：
  - `from .modeling_flow_expert import FlowMatchingActionExpert`  
    → `from .modeling_flow_expert_dev import FlowMatchingActionExpert`
  - 视后续需要，再将 LLaVA3D 改为 dev 版。

快速测试建议：

- 写一个简单脚本或测试函数：
  - 实例化 dev 版 `MapAnythingLlava3DForConditionalGeneration`；
  - 传入 dummy `input_ids`、`pixel_values` 等，跑通一次前向（不带 actions）。

### 步骤 1：在 `modeling_llava3d_v2_dev.py` 中搭建最小版 `LLaVA3DWithActionExpertModel`

目标：先有一个可用的壳，后续再填充真实双流逻辑。

1. 新增类骨架：

```python
class LLaVA3DWithActionExpertModel(nn.Module):
    def __init__(self, base_llava: LLaVA3DForCausalLMV2, expert_config):
        ...

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values=None,
        inputs_embeds=None,   # [prefix_embs, suffix_embs]
        use_cache=False,
        expert_cond=None,     # [cond_prefix, cond_suffix]
    ):
        ...
```

2. 最小实现：

- prefix-only：仅 `inputs_embeds[0]` 非 None 时：
  - 直接调用 `base_llava.model(...)`，返回 prefix_output 和 past_key_values。
- suffix-only：仅 `inputs_embeds[1]` 非 None 时：
  - 暂时用一套简化的 expert 流（例如一个小的 Transformer 或线性 + MLP），后续替换为真正的双流实现。
- prefix+suffix：临时方案可以先抛出 `NotImplementedError`，或者简单地独立跑两遍（仅用于接口测试）。

3. 测试：

- 构造随机 `prefix_embs`、`suffix_embs`、mask、position_ids`；
- 调用 prefix-only、suffix-only、prefix+suffix 三种模式；
- 确认形状正确，forward 不报错。

### 步骤 2：在 `LLaVA3DWithActionExpertModel` 中实现双流联合注意力 ✅ 已完成

**实施日期**：2024-12-30  
**状态**：✅ 核心功能已实现

目标：实现类似 `PaliGemmaWithExpertModel.compute_layer_complete` 的真正 Deep Fusion。

#### 已完成的实现：

1. ✅ **找到并访问 LLaVA3D 的单层结构**：
   - 通过 `self.base_llava.model.model` 访问底层 LlamaModel 或 MistralModel；
   - 使用 `self.base_model.layers[layer_idx]` 获取每一层；
   - 自动检测模型类型（LLaMA/Mistral）并适配不同的 API。

2. ✅ **Expert 流层结构**：
   - 采用参数共享策略：expert 流复用 prefix 的层权重（`self.expert_layers = self.base_model.layers`）；
   - 为 expert 创建独立的 final norm 层（`self.expert_norm`）；
   - 支持可选的投影层（当 expert_hidden_size ≠ base_hidden_size 时）。

3. ✅ **实现 `_compute_layer_complete` 逐层联合注意力**：
   - 对 prefix_hidden 和 suffix_hidden 分别做 `input_layernorm`；
   - 分别计算 Q/K/V 投影并 reshape 为 (batch, num_heads, seq, head_dim)；
   - 在序列维度拼接：`joint_q/k/v = cat([prefix_q/k/v, suffix_q/k/v], dim=2)`；
   - 统一应用 RoPE（通过 `_apply_rotary_pos_emb` 方法）；
   - 调用 `_compute_attention` 进行联合注意力计算（使用 `eager_attention_forward`）；
   - 根据 prefix_len 切分输出为 `prefix_attn_output` 和 `suffix_attn_output`；
   - 各自做 `o_proj` → 第一残差 → `post_attention_layernorm` → `mlp` → 第二残差；
   - 返回更新后的 prefix_hidden 和 suffix_hidden。

4. ✅ **实现三种前向模式**：
   - **Prefix-only**：调用 `self.llava_model` 标准前向，用于语言生成；
   - **Suffix-only**：逐层前向处理 suffix（未使用 prefix cache，待优化）；
   - **Prefix+Suffix**：调用 `_compute_layer_complete` 实现 Deep Fusion，最后分别做 final norm。

5. ✅ **实现辅助方法**：
   - `_create_norm_layer()`：根据模型类型创建对应的 RMSNorm；
   - `_apply_rotary_pos_emb()`：模型类型无关的 RoPE 应用接口；
   - `_compute_attention()`：模型类型无关的注意力计算接口。

#### 待优化项：

- ⚠️ **Suffix-only 模式的 KV cache 集成**：当前 suffix-only 独立前向，未使用 prefix 的 KV cache（推理加速优化）；
- ⚠️ **Gradient Checkpointing 支持**：参考 PI0 实现，为长序列训练添加梯度检查点；
- ⚠️ **单元测试**：需要添加测试验证三种模式的正确性。

#### 测试建议：

- 构造随机 prefix_embs 和 suffix_embs（small batch, small seq_len）；
- 测试三种模式的形状正确性；
- 检查 attention_mask 是否正确传播；
- Sanity check：当 prefix 和 suffix 输入相同时，两路输出应相似（在无 position bias 情况下）。

### 步骤 3：重写 `modeling_flow_expert_dev.py` 为 LLaVA3D‑based Flow Matching 封装 ✅ 已完成

**实施日期**：2024-12-30  
**状态**：✅ 核心功能已实现

目标：从 Gemma 解耦，专注 Flow Matching 数学和调用 WithExpert。

#### 已完成的实现：

1. ✅ **去除 Gemma 依赖**：
   - 删除了所有 `GemmaModel`, `GemmaPreTrainedModel`, `GemmaConfig` 相关依赖；
   - 类改为简单的 `nn.Module`，不再继承任何预训练模型基类；
   - 移除了内部 Transformer 模型，改为调用外部 `LLaVA3DWithActionExpertModel`。

2. ✅ **保留 Flow Matching 数学逻辑**：
   - ✅ 保留 `create_sinusoidal_pos_embedding`（时间编码）；
   - ✅ 保留 `sample_noise`（高斯噪声采样）；
   - ✅ 保留 `compute_loss` 中的完整 Flow Matching 公式：
     - t ~ Uniform(0, 1)
     - x_t = t * noise + (1 - t) * actions
     - u_t = noise - actions
     - v_t = model(x_t, t)
     - loss = MSE(v_t, u_t)
   - ✅ 保留 `sample_actions` 中的 Euler ODE 迭代框架：
     - x_t ~ N(0, I) at t=1
     - for t from 1 to 0: x_t = x_t + v_t * dt
     - return x_0

3. ✅ **重新设计接口（与 LLaVA3DWithActionExpert 集成）**：
   
   **初始化**：
   ```python
   def __init__(
       llava_with_expert_model,  # LLaVA3DWithActionExpertModel 实例
       action_dim=7,
       action_horizon=10,
       state_dim=None,
       use_state=False,
   )
   ```
   
   **核心方法**：
   
   a. `_construct_suffix_embeddings(actions, time, state)`：
   - 构造 suffix embeddings 序列
   - 结构：[state_token?, action_token_1, ..., action_token_H, time_token]
   - State: optional, projected by `state_proj`
   - Actions: each step projected to hidden_size by `action_in_proj`
   - Time: sinusoidal embedding -> MLP (time_mlp_in/out)
   
   b. `forward(prefix_embs, actions, time, state, ...)`：
   - 构造 suffix embeddings
   - 调用 `llava_with_expert([prefix_embs, suffix_embs])` (Deep Fusion Mode 3)
   - 从 suffix_output 提取 action tokens
   - 通过 `action_out_proj` 预测 velocity: [B, H, action_dim]
   
   c. `compute_loss(prefix_embs, actions, state, ...)`：
   - 采样 t 和 noise
   - 构造 noisy actions x_t
   - 调用 forward 预测 v_t
   - 计算 MSE(v_t, u_t)
   
   d. `sample_actions(prefix_embs, state, num_steps=10, ...)`：
   - 初始化 x_t ~ N(0, I)
   - Euler ODE 循环（t: 1 → 0）：
     - 预测 v_t = forward(prefix_embs, x_t, t, state)
     - 更新 x_t = x_t + v_t * dt
   - 返回 clean actions x_0

4. ✅ **新增功能**：
   - 支持可选的 proprioceptive state 输入（`use_state=True`）
   - 自动从 `llava_with_expert_model` 推断 `hidden_size`
   - 完整的类型注解和文档字符串
   - 支持 attention_mask 和 position_ids（用于 Deep Fusion）

#### 测试验证：

- ✅ 代码无 linter 错误
- ⚠️ 需要集成测试（等待步骤 4 完成后）
- ⚠️ 需要验证 Flow Matching 数学正确性（梯度流、loss 收敛等）

#### 关键改进：

相比原始实现：
- ❌ 删除了简单的 MLP 网络（`context_projector`, `dynamics_mlp`）
- ✅ 改为调用强大的 LLaVA3D Deep Fusion 模型
- ✅ Prefix 和 suffix 在每层都有交互（不再是简单的 global pooling + MLP）
- ✅ 更灵活的 suffix 结构（支持 state + actions + time）
- ✅ 更清晰的接口设计（prefix_embs 由 wrapper 提供）

### 步骤 4：改造 `modeling_mapanything_llava3d_dev.py` 接入新专家与前缀/后缀逻辑

目标：让 dev wrapper 完整走 Deep Fusion + Flow Matching。

1. 初始化阶段：

- 在 `__init__` 中：
  - 保留 `self.language_model`（用于纯语言任务）。
  - 新建 `self.language_model_with_expert = LLaVA3DWithActionExpertModel(...)`。
  - 使用 `FlowMatchingActionExpertDev`，把 WithExpert 实例注入进去。
  - 新建 suffix embedding 所需线性层：
    - `state_proj: state_dim -> hidden_size`；
    - `action_in_proj: action_dim -> hidden_size`；
    - `time_mlp_in/out: hidden_size -> hidden_size`。

2. ✅ **构造 prefix + attention mask**：

- 使用 `get_image_features` 得到 `[B, S_v, H_llm]`。
- 从 `input_ids` 得到文本 embedding。
- 通过 mask 将 image token 融合到文本，或者按顺序拼接 image + text。
- 通过 `_build_joint_attention_inputs` 基于 prefix 长度和 suffix_len 构造：
  - `joint_attention_mask: [B, 1, Lp+Ls, Lp+Ls]`
  - `joint_position_ids: [B, Lp+Ls]`

3. 构造 suffix：

- 根据 `state`、`x_t`（noisy actions）、`time`：
  - `state_proj(state)`，得到 state token（可选）；
  - `action_in_proj(x_t)` 得到 action token；
  - `create_sinusoidal_pos_embedding(time, hidden_size, ...)` + `time_mlp_in/out` 得到 time embedding；
  - 按设计拼接为 `[state_token?, action_time_tokens]`。
- 构建 `suffix_embs` 及 `suffix_pad_masks`、`suffix_att_masks`。

4. Flow Matching 训练路径：

- 在 `forward` 中，当 `actions is not None`：
  - 调用 FlowMatchingActionExpertDev 的 `compute_loss`：
    - 内部采样 t、噪声，构造 x_t、u_t；
    - 使用前缀/后缀 embedding + WithExpert 得到 v_t；
    - 计算 MSE。
  - 返回 `MapAnythingLlava3DOutput(loss=action_loss, logits=None, ...)` 或按需保留语言 logits。

5. 动作推理路径（`predict_action`）：

- 使用 prefix-only 模式调用 WithExpert 建立 prefix KV cache。
- 循环：
  - 对当前 `x_t` 与时间 t 构造 suffix_embs；
  - 调用 WithExpert 的 denoise_step 接口，得到 v_t；
  - 用 Euler 步更新 `x_t = x_t + dt * v_t`，时间递减；
- 最终返回 `x_t` 作为动作预测。

6. 集成测试：

- 构造小 hidden、小层数配置，在 CPU 上测试：
  - forward（无 actions）：纯语言路径；
  - forward（有 actions）：Flow Matching 路径；
  - predict_action：Euler 去噪路径；
  - 检查张量形状和梯度行为。

## 六、测试文件建议内容

为了验证 dev 方案，建议增加以下测试（pytest 或脚本均可）：

1. LLaVA3DWithActionExpertModel 单元测试：
   - prefix-only 前向：形状正确，与原 LLaVA3D 行为兼容。
   - suffix-only 前向：形状正确。
   - prefix+suffix 前向：形状正确，attention_mask 有效。

2. FlowMatchingActionExpertDev 单元测试：
   - 使用 mock WithExpert 测试 `compute_loss`：
     - 确认 t、x_t、u_t 构造正确且 loss 可反向传播。
   - 测试 `sample_actions`：
     - 小步数（例如 num_steps=2），确认输出 shape 正确且不报错。

3. MapAnythingLlava3DForConditionalGeneration_dev 集成测试：
   - forward：
     - 仅图像+文本（不带 actions），验证 logits/hidden_states 正常。
     - 带 actions，验证 loss 标量且返回结构完整。
   - predict_action：
     - 使用 dummy 图像、intrinsic、input_ids、state，检查动作形状 `[B, action_horizon, action_dim]`。

4. 回归 sanity：
   - 对同一输入，比较：
     - 原始 LLaVA3D 的 prefix-only hidden 与 WithExpert 的 prefix_output 是否一致（在无 suffix 情况下）。

通过以上步骤与测试，可以在不破坏现有稳定代码的前提下，逐步把动作建模从 Late Fusion 升级到基于 LLaVA3D 的 Deep Fusion Flow Matching 架构。

---

## 七、步骤 2 详细实现说明（2024-12-30，含后续升级规划）

### 7.1 实现概述

步骤 2 的核心目标是在 `LLaVA3DWithActionExpertModel` 中实现真正的双流联合注意力（Deep Fusion），使 prefix（视觉+语言）和 suffix（状态+动作+时间）在每一层都能互相感知和交互。

**初版关键设计决策（参数共享版，已实现）：**
1. **参数共享策略**：Expert 流复用 LLaVA3D 的层权重，而非创建独立的 expert 层，节省参数量并保持一致性。
2. **模型类型无关**：通过适配层自动检测并支持 LLaMA 和 Mistral 两种架构。
3. **三种前向模式**：支持 prefix-only（语言生成）、suffix-only（动作去噪）、prefix+suffix（联合训练）。

**升级版关键设计决策（独立 expert 骨架版，规划中）：**
1. 引入 `expert_model`：
   - 类型为 `LlamaModel` / `MistralModel`，config 与 base 模型一致但权重独立初始化；
   - suffix 流的 LayerNorm / QKV / MLP 全部来自这套 expert 模型。
2. Deep Fusion 联合注意力保持“QKV 拼接 + RoPE + 单一 attention 内核”的形式：
   - prefix 使用 `base_model.layers[layer_idx]` 的投影；
   - suffix 使用 `expert_model.layers[layer_idx]` 的投影；
   - Q/K/V 在序列维拼接后，统一送入 attention 内核（可复用 base 模型对应层的 `self_attn` 配置）。
3. 对外接口保持不变：
   - 仍然支持 prefix-only / suffix-only / prefix+suffix 三种模式；
   - 差异仅体现在内部是否为“共享层权重”或“独立 expert 骨架”。

### 7.2 核心方法详解

#### 7.2.1 `__init__` - 初始化

```python
def __init__(self, base_llava: LLaVA3DForCausalLMV2, expert_config=None)
```

**功能**：
- 访问 LLaVA3D 的底层模型结构（LlamaModel 或 MistralModel）
- 自动检测模型类型并提取配置（hidden_size, num_layers, num_heads 等）
- 创建 expert 专用的 final norm 层
- 可选：创建投影层（当 expert_hidden_size ≠ base_hidden_size 时）

**关键代码路径**：
```
base_llava.model               # LlavaLlamaForCausalLM / LlavaMistralForCausalLM
  └─ .model                     # LlamaModel / MistralModel (底层 Transformer)
      ├─ .layers[i]              # 各层 LlamaDecoderLayer / MistralDecoderLayer
      ├─ .norm                   # Final RMSNorm
      └─ .rotary_emb             # RoPE 模块
```

#### 7.2.2 `_compute_layer_complete` - 逐层联合注意力（核心）

```python
def _compute_layer_complete(
    layer_idx, prefix_hidden, suffix_hidden, attention_mask, position_ids
) -> (prefix_hidden, suffix_hidden)
```

**流程图**：

```
输入: prefix_hidden [B, L_p, H], suffix_hidden [B, L_s, H]
  │
  ├─ Step 1: 分别做 LayerNorm
  │   prefix_normed = layer.input_layernorm(prefix_hidden)
  │   suffix_normed = layer.input_layernorm(suffix_hidden)
  │
  ├─ Step 2: 分别计算 QKV
  │   prefix: Q_p, K_p, V_p = q_proj/k_proj/v_proj(prefix_normed)
  │   suffix: Q_s, K_s, V_s = q_proj/k_proj/v_proj(suffix_normed)
  │   reshape to [B, num_heads, seq, head_dim]
  │
  ├─ Step 3: 在序列维拼接 QKV
  │   joint_Q = concat([Q_p, Q_s], dim=2)  # [B, H, L_p+L_s, D]
  │   joint_K = concat([K_p, K_s], dim=2)
  │   joint_V = concat([V_p, V_s], dim=2)
  │
  ├─ Step 4: 统一应用 RoPE
  │   joint_Q, joint_K = apply_rotary_pos_emb(joint_Q, joint_K, position_ids)
  │
  ├─ Step 5: 计算联合注意力
  │   joint_attn_out = eager_attention_forward(...)
  │   # prefix 和 suffix 在这里互相感知！
  │
  ├─ Step 6: 拆分回两路
  │   prefix_attn_out = joint_attn_out[:, :L_p, :]
  │   suffix_attn_out = joint_attn_out[:, L_p:, :]
  │
  ├─ Step 7: 各自做 o_proj + 第一残差
  │   prefix_hidden = prefix_hidden + layer.self_attn.o_proj(prefix_attn_out)
  │   suffix_hidden = suffix_hidden + layer.self_attn.o_proj(suffix_attn_out)
  │
  └─ Step 8: 各自做 MLP + 第二残差
      prefix_normed = layer.post_attention_layernorm(prefix_hidden)
      prefix_hidden = prefix_hidden + layer.mlp(prefix_normed)
      suffix_normed = layer.post_attention_layernorm(suffix_hidden)
      suffix_hidden = suffix_hidden + layer.mlp(suffix_normed)
      
输出: 更新后的 prefix_hidden, suffix_hidden
```

**关键点**：
- **联合注意力**是 Deep Fusion 的核心：通过在序列维拼接 QKV，使得 prefix 的每个 token 都能 attend to suffix 的 token，反之亦然。
- **初版参数共享**：两路使用同一个 layer 的权重（q_proj, k_proj, v_proj, o_proj, mlp）。
- **升级版独立 expert**：prefix 和 suffix 在 QKV / MLP / LayerNorm 上使用不同的层（来自 base_model 与 expert_model），但仍然共享同一 attention 内核，实现“结构对齐但权重解耦”。 

#### 7.2.3 `forward` - 三种前向模式

**Mode 1: Prefix-only** (语言生成)
```python
if prefix_embs is not None and suffix_embs is None:
    # 直接调用 LLaVA3D 标准前向
    outputs = self.llava_model(inputs_embeds=prefix_embs, ...)
    return [prefix_output, None], past_key_values
```

**Mode 2: Suffix-only** (动作去噪)
```python
if prefix_embs is None and suffix_embs is not None:
    # 逐层前向处理 suffix
    for layer_idx in range(self.num_layers):
        suffix_hidden = self._process_suffix_layer(layer_idx, suffix_hidden, ...)
    suffix_output = self.expert_norm(suffix_hidden)
    return [None, suffix_output], None
```

**Mode 3: Prefix+Suffix** (Deep Fusion 训练)
```python
if prefix_embs is not None and suffix_embs is not None:
    # 逐层联合前向
    for layer_idx in range(self.num_layers):
        prefix_hidden, suffix_hidden = self._compute_layer_complete(
            layer_idx, prefix_hidden, suffix_hidden, ...
        )
    prefix_output = self.base_model.norm(prefix_hidden)
    suffix_output = self.expert_norm(suffix_hidden)
    return [prefix_output, suffix_output], None
```

### 7.3 辅助方法

#### `_apply_rotary_pos_emb(query, key, position_ids)`
- 模型类型无关的 RoPE 应用接口
- 自动适配 LLaMA 和 Mistral 的 RoPE 实现

#### `_compute_attention(layer, Q, K, V, mask)`
- 模型类型无关的注意力计算接口
- 使用 `eager_attention_forward`（支持 LLaMA/Mistral）

#### `_create_norm_layer()`
- 根据模型类型创建对应的 RMSNorm

### 7.4 与 PI0 实现的对比

| 特性 | PI0 (PaliGemma + Gemma) | LLaVA3D 初版 | LLaVA3D 升级版（规划） |
|------|-------------------------|-------------|-------------------------|
| **基础模型** | PaliGemma (prefix) + 独立 Gemma Expert (suffix) | LLaVA3D 统一处理两路 | LLaVA3D (prefix) + 独立 LLaMA/Mistral Expert (suffix) |
| **参数共享** | 否（两个独立模型） | 是（共享层权重） | 否（两套骨架，结构一致） |
| **联合注意力** | ✅ 逐层拼接 QKV | ✅ 逐层拼接 QKV | ✅ 逐层拼接 QKV |
| **RoPE 应用** | ✅ 统一应用 | ✅ 统一应用 | ✅ 统一应用 |
| **模型类型** | 仅 Gemma | LLaMA + Mistral | LLaMA + Mistral |
| **Gradient Checkpointing** | ✅ 支持 | ⚠️ 待添加 | ⚠️ 待添加 |
| **KV Cache** | ✅ Prefix cache | ⚠️ 待优化 | ⚠️ 待优化 |

### 7.5 已知限制与后续优化

#### 已知限制：
1. **Suffix-only 模式未使用 prefix KV cache**
   - 当前实现：suffix 独立前向，无法访问 prefix 的上下文
   - 影响：推理时无法充分利用 prefix 信息，且速度较慢
   - 计划：在步骤 4 中实现完整的 cache 管理

2. **缺少 Gradient Checkpointing**
   - 当前实现：所有层的激活都保留在内存中
   - 影响：大模型训练时显存占用高
   - 计划：参考 PI0 的实现添加 `torch.utils.checkpoint.checkpoint`

3. **缺少单元测试**
   - 当前实现：仅完成代码，未验证正确性
   - 计划：添加 pytest 测试（见第六节）

#### 性能优化方向：
- **Flash Attention 支持**：使用 Flash Attention 2 加速联合注意力
- **Mixed Precision**：自动混合精度训练（BF16/FP16）
- **Sequence Parallel**：超长序列的并行处理

### 7.6 使用示例

```python
# 初始化
base_llava = LLaVA3DForCausalLMV2(config)
model_with_expert = LLaVA3DWithActionExpertModel(base_llava)

# Mode 1: Prefix-only (语言生成)
prefix_embs = get_image_text_embeddings(...)  # [B, L_p, H]
outputs, cache = model_with_expert(
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
prefix_output = outputs[0]  # [B, L_p, H]

# Mode 2: Suffix-only (动作去噪)
suffix_embs = get_action_time_embeddings(...)  # [B, L_s, H]
outputs, _ = model_with_expert(
    inputs_embeds=[None, suffix_embs],
)
suffix_output = outputs[1]  # [B, L_s, H]

# Mode 3: Deep Fusion (联合训练)
outputs, _ = model_with_expert(
    inputs_embeds=[prefix_embs, suffix_embs],
)
prefix_output, suffix_output = outputs  # 两路都有输出
```

### 7.7 下一步（步骤 3）

步骤 2 已经完成了 Deep Fusion 的基础设施，下一步需要：

1. **改造 `FlowMatchingActionExpert`**：
   - 删除对 Gemma 的依赖
   - 将网络前向改为调用 `LLaVA3DWithActionExpertModel`
   - 保留 Flow Matching 数学逻辑（t, noise, x_t, u_t, Euler）

2. **在 `MapAnythingLlava3DForConditionalGeneration` 中集成**：
   - 构造 prefix_embs（image + geometric + text）
   - 构造 suffix_embs（state + noisy_actions + time）
   - 训练时调用 Mode 3（Deep Fusion）
   - 推理时调用 Mode 1（prefix cache）+ 循环 Mode 2（去噪）

---

**步骤 2 完成标志**：✅  
**代码文件**：`modeling_llava3d_v2_dev.py` (第 169-509 行)  
**实现日期**：2024-12-30  
**核心贡献**：实现了 LLaVA3D 的双流 Deep Fusion 架构，为后续 Flow Matching 集成奠定基础。

---

## 八、步骤 3 详细实现说明（2024-12-30）

### 8.1 实现概述

步骤 3 的核心目标是将 `FlowMatchingActionExpert` 从基于 Gemma 的 Late Fusion 架构改造为基于 `LLaVA3DWithActionExpertModel` 的 Deep Fusion 架构，同时完整保留 Flow Matching 的数学逻辑。

**关键设计决策**：
1. **完全删除 Gemma 依赖**：不再继承 `GemmaPreTrainedModel`，改为纯 `nn.Module`
2. **网络前向委托给 LLaVA3D**：调用 `LLaVA3DWithActionExpertModel` 进行 Deep Fusion
3. **专注于算法层**：FlowMatchingActionExpert 只负责 Flow Matching 数学和 suffix embedding 构造
4. **灵活的 suffix 结构**：支持 state + actions + time 的组合

### 8.2 核心方法详解

#### 8.2.1 `__init__` - 初始化

```python
def __init__(
    self,
    llava_with_expert_model,  # LLaVA3DWithActionExpertModel 实例
    action_dim: int = 7,
    action_horizon: int = 10,
    state_dim: Optional[int] = None,
    hidden_size: Optional[int] = None,
    use_state: bool = False,
)
```

**功能**：
- 接收 `LLaVA3DWithActionExpertModel` 实例（不再自己创建网络）
- 配置动作空间参数（`action_dim`, `action_horizon`）
- 可选：支持 proprioceptive state（机器人关节角度、速度等）
- 创建 suffix embedding 层：
  - `state_proj`: [state_dim] → [hidden_size]
  - `action_in_proj`: [action_dim] → [hidden_size]
  - `time_mlp_in/out`: [hidden_size] → [hidden_size]
  - `action_out_proj`: [hidden_size] → [action_dim]

**关键变化**：
- ❌ 删除：`context_projector`（不再需要投影 VLM features）
- ❌ 删除：`dynamics_mlp`（由 LLaVA3D 的 Deep Fusion 替代）
- ✅ 新增：`state_proj`（支持 proprioceptive state）
- ✅ 保留：`action_in_proj`, `time_mlp_in/out`, `action_out_proj`

#### 8.2.2 `_construct_suffix_embeddings` - 构造 Suffix Embeddings

```python
def _construct_suffix_embeddings(
    actions: torch.Tensor,  # [B, H, action_dim]
    time: torch.Tensor,     # [B]
    state: Optional[torch.Tensor] = None,  # [B, state_dim]
) -> torch.Tensor  # [B, suffix_seq_len, hidden_size]
```

**Suffix 序列结构**：

```
┌─────────────┬──────────────────┬────────────┐
│ state_token │ action_tokens    │ time_token │
│ (optional)  │ (H tokens)       │ (1 token)  │
├─────────────┼──────────────────┼────────────┤
│ [B, 1, H]   │ [B, H, H]        │ [B, 1, H]  │
└─────────────┴──────────────────┴────────────┘

Total length: (0 or 1) + H + 1 = H+1 or H+2
```

**构造流程**：

```
1. State Token (optional):
   state [B, state_dim] → state_proj → [B, 1, hidden_size]

2. Action Tokens:
   actions [B, H, action_dim] → action_in_proj → [B, H, hidden_size]

3. Time Token:
   time [B] → sinusoidal_embedding → [B, hidden_size]
           → time_mlp_in → SiLU → time_mlp_out → [B, 1, hidden_size]

4. Concatenate:
   suffix_embs = cat([state_token?, action_tokens, time_token], dim=1)
```

**关键点**：
- Time 使用 sinusoidal position embedding（频率从 min_period=4e-3 到 max_period=4.0）
- 每个 action step 独立 embedding（保留时序信息）
- State token 放在最前面（类似 [CLS] token 的位置）

#### 8.2.3 `forward` - Deep Fusion 前向（核心）

```python
def forward(
    prefix_embs: torch.Tensor,   # [B, L_p, H]
    actions: torch.Tensor,        # [B, H, action_dim]
    time: torch.Tensor,           # [B]
    state: Optional[torch.Tensor] = None,
    ...
) -> torch.Tensor  # [B, H, action_dim]
```

**流程图**：

```
输入:
  prefix_embs [B, L_p, H]  # 来自 wrapper: image + geo + text
  actions [B, H, action_dim]  # noisy actions x_t
  time [B]  # 当前时间步
  state [B, state_dim]  # 可选: 机器人状态

    ↓
Step 1: 构造 Suffix Embeddings
  suffix_embs [B, L_s, H] = _construct_suffix_embeddings(actions, time, state)

    ↓
Step 2: 调用 LLaVA3DWithActionExpertModel (Deep Fusion)
  outputs, _ = llava_with_expert(
      inputs_embeds=[prefix_embs, suffix_embs],
      attention_mask=joint_mask,
      position_ids=joint_pos_ids,
  )
  prefix_output, suffix_output = outputs  # [B, L_p, H], [B, L_s, H]
  
  # 注意: prefix 和 suffix 在每层都互相感知了！

    ↓
Step 3: 提取 Action Tokens
  if use_state:
      # suffix structure: [state_token, action_tokens, time_token]
      action_hidden = suffix_output[:, 1:1+H, :]  # [B, H, H]
  else:
      # suffix structure: [action_tokens, time_token]
      action_hidden = suffix_output[:, :H, :]  # [B, H, H]

    ↓
Step 4: 投影到 Action Velocity
  pred_velocity = action_out_proj(action_hidden)  # [B, H, action_dim]

输出:
  pred_velocity [B, H, action_dim]
```

**关键点**：
- 使用 Deep Fusion Mode 3：prefix 和 suffix 联合前向
- Prefix 提供视觉和语言上下文
- Suffix 在每层都能 attend to prefix（获得丰富的上下文信息）
- 只提取 action tokens（跳过 state 和 time tokens）

#### 8.2.4 `compute_loss` - Flow Matching 训练

```python
def compute_loss(
    prefix_embs: torch.Tensor,
    actions: torch.Tensor,  # [B, H, action_dim] ground truth
    state: Optional[torch.Tensor] = None,
    ...
) -> torch.Tensor  # scalar loss
```

**Flow Matching 公式**：

```
Given: clean actions a ∈ R^{H×action_dim}

Step 1: Sample time
  t ~ Uniform(0, 1)  ∈ R^B

Step 2: Sample noise
  ε ~ N(0, I)  ∈ R^{B×H×action_dim}

Step 3: Construct noisy actions (Flow Matching interpolation)
  x_t = t·ε + (1-t)·a

Step 4: Compute target velocity
  u_t = ε - a  (points from clean to noise)

Step 5: Predict velocity
  v_t = model(prefix_embs, x_t, t, state)

Step 6: Compute loss
  L = MSE(v_t, u_t) = ||v_t - u_t||²
```

**物理直觉**：
- Flow Matching 学习一个速度场 v_t(x, t)
- 该速度场引导从 噪声分布 (t=1) 流向 数据分布 (t=0)
- 训练时：随机采样 t 和 x_t，学习正确的速度方向
- 推理时：从噪声出发，沿速度场积分得到 clean action

#### 8.2.5 `sample_actions` - Euler ODE 采样

```python
@torch.no_grad()
def sample_actions(
    prefix_embs: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    num_steps: int = 10,
    ...
) -> torch.Tensor  # [B, H, action_dim]
```

**Euler ODE Solver**：

```
Given: prefix_embs (image + geo + text context)

Initialization:
  x_1 ~ N(0, I)  # Start from pure noise at t=1

ODE Integration (t: 1 → 0):
  dt = -1 / num_steps  # Negative step (backward in time)
  
  for step in range(num_steps):
      t_curr = 1 + step * dt  # t: 1.0, 0.9, 0.8, ..., 0.1
      
      # Predict velocity at current point
      v_t = model(prefix_embs, x_t, t_curr, state)
      
      # Euler step: move along velocity field
      x_{t+dt} = x_t + v_t · dt
  
Output:
  x_0  # Clean actions at t=0
```

**积分路径示意**：

```
t=1.0 (noise)                    t=0.0 (clean)
    ●                                 ★
    |                                 ↑
    | v_t ──→                         |
    ↓                                 |
    ●────→●────→●────→●────→●────→●────→●
   x_1   x_0.9 x_0.8 x_0.7  ...  x_0.1  x_0
   
每步: x_{t-0.1} = x_t + v_t * (-0.1)
```

### 8.3 与原始实现的对比

| 特性 | 原始实现 (简单 MLP) | 新实现 (Deep Fusion) |
|------|---------------------|----------------------|
| **网络结构** | context_projector + dynamics_mlp | LLaVA3D Deep Fusion |
| **上下文使用** | Global pooling (mean) | 每层联合注意力 |
| **参数量** | ~10M (独立 MLP) | 0 (复用 LLaVA3D) |
| **表达能力** | 弱（浅层 MLP） | 强（深层 Transformer） |
| **视觉-动作交互** | 无（仅最后一层） | 有（每层 cross-attention） |
| **State 支持** | 无 | 有（可选 state token） |
| **灵活性** | 低 | 高（可扩展 suffix 结构） |

### 8.4 使用示例

```python
# 初始化
base_llava = LLaVA3DForCausalLMV2(config)
llava_with_expert = LLaVA3DWithActionExpertModel(base_llava)
flow_expert = FlowMatchingActionExpert(
    llava_with_expert_model=llava_with_expert,
    action_dim=7,
    action_horizon=10,
    use_state=True,
    state_dim=14,
)

# 训练
prefix_embs = get_image_text_embeddings(...)  # [B, L_p, H]
actions = get_ground_truth_actions(...)  # [B, 10, 7]
state = get_robot_state(...)  # [B, 14]

loss = flow_expert.compute_loss(
    prefix_embs=prefix_embs,
    actions=actions,
    state=state,
)
loss.backward()

# 推理
with torch.no_grad():
    predicted_actions = flow_expert.sample_actions(
        prefix_embs=prefix_embs,
        state=state,
        num_steps=20,
    )  # [B, 10, 7]
```

### 8.5 下一步（步骤 4）

步骤 3 已经完成了 Flow Matching 算法与 Deep Fusion 的集成，接下来需要：

**步骤 4: 改造 `MapAnythingLlava3DForConditionalGeneration`**

1. **构造 prefix_embs**：
   - 使用 `get_image_features` 得到融合后的视觉+几何特征
   - 与文本 token embedding 合并（按 `<image>` token 位置注入）
   
2. **初始化 FlowMatchingActionExpert**：
   - 在 `__init__` 中创建 `self.action_expert`
   - 传入 `self.language_model_with_expert`
   
3. **训练路径 (`forward` with actions)**：
   - 构造 prefix_embs
   - 调用 `self.action_expert.compute_loss(prefix_embs, actions, state)`
   - 返回 action_loss
   
4. **推理路径 (`predict_action`)**：
   - 构造 prefix_embs
   - 调用 `self.action_expert.sample_actions(prefix_embs, state, num_steps)`
   - 返回预测的 actions

---

**步骤 3 完成标志**：✅  
**代码文件**：`modeling_flow_expert_dev.py` (完整重写)  
**实现日期**：2024-12-30  
**核心贡献**：将 Flow Matching 算法与 LLaVA3D Deep Fusion 完美集成，实现视觉-语言-动作的端到端深度交互。

---

## 九、训练策略与语言模型微调规划（2026-01 更新）

### 9.1 当前 Deep Fusion Flow Matching 训练策略

- 文本角色：
  - 在当前 Deep Fusion Flow Matching 路径中，**文本仅作为条件输入**（prompt + `<image>` 占位），不再对语言部分施加 cross-entropy 或 prefix-LM 损失；
  - 训练目标完全来自动作 Flow Matching（velocity MSE），语言模型充当“条件编码器”，类似 openpi 中的 prefix。
- 参数冻结策略：
  - 为提高数值稳定性并降低超参搜索成本，当前阶段采取 **大规模冻结骨干，仅训练动作相关模块** 的策略：
    - 冻结 LLaVA3D 主体（文本 + 视觉融合 Transformer）；
    - 冻结 SigLIP 视觉塔与 MapAnything 几何模型；
    - 仅训练 suffix expert（独立的 LLaMA/Mistral Expert）、FlowMatchingActionExpert 内部的 `state_proj` / `action_in_proj` / `time_mlp_in/out` / `action_out_proj` 等小型线性层。
- 数值精度策略：
  - 当前所有 Deep Fusion + Flow Matching 训练均在 **FP32 全精度** 下进行；
  - 文档中前面提到的 BF16/FP16 Mixed Precision 仍作为后续可选优化方向，暂不在不稳定阶段叠加。

### 9.2 与 openpi / PI0 的关系与对齐

- openpi 的两套 mask：
  - `token_ar_mask`：用于语言 prefix-LM 任务（前缀 + 后缀文本），约束后缀 token 只能看见前缀 + 自身左侧 token；
  - Flow Matching 侧的 `att_masks`：在 `embed_prefix` / `embed_suffix` 中根据 prefix_len / suffix_len 单独构造，用于控制动作流在联合注意力中的可见性。
- 本项目的选择：
  - 本项目复刻的是 **Flow Matching 路径的 mask 行为**，在 `modeling_mapanything_llava3d_dev.py::_build_joint_attention_inputs` 中实现了与 openpi 类似的 **自回归 `mask_ar`**：
    - prefix token 之间保持双向可见；
    - suffix token 只能看到 prefix + 自己之前的 suffix token；
    - 与 pad mask 结合后，形成 Deep Fusion 的最终联合注意力掩码。
  - 由于当前阶段 **不对语言模型做 prefix-LM 训练**，因此没有引入 openpi 专门用于语言任务的 `token_ar_mask`，也没有在训练中添加文本 cross-entropy loss。

### 9.3 语言模型微调的后续阶段规划

在完成“冻结骨干 + FP32” 的稳定性验证之后，如需要进一步提升性能或对齐特定机器人任务，可以考虑逐步引入 **轻量级的语言模型微调**。一个推荐的三阶段路线如下：

1. 阶段 A（已完成：冻结骨干 + FP32 稳定性验证）：
   - 冻结 LLaVA3D、SigLIP、MapAnything 等大模型骨干；
   - 仅训练 suffix expert + FlowMatchingActionExpert 内部小模块；
   - 结果：数值上稳定，无系统性 NaN/Inf 问题，但在当前数据与配置下，Flow Matching loss 通常停留在约 0.5–1.5 区间，难以进一步下降。

2. 阶段 B（进行中：轻量联合微调）：
   - 目标：在保持训练稳定的前提下，允许 LLaVA3D 顶部少量层对机器人动作数据做 **有限度适配**；
   - 当前策略与观察：
     - 在训练脚本中通过 `backbone_unfreeze_layers` 与 `backbone_lr_scale`，仅解冻 LLaVA3D 顶部若干层（例如最后 4 层），其余层和视觉塔继续冻结；
     - 对这些层采用 **明显小于** 动作 expert 的学习率，并与 `expert_lr_scale` 分组管理；
     - 现有实验表明：在相对保守的配置下，loss 曲线形状与完全冻结版本接近，说明仅靠轻量解冻和学习率调节很难带来质变。
   - 后续方向：
     - 在显存允许的前提下探索更多解冻层数或更高 `backbone_lr_scale` 的配置，并结合混合精度训练降低内存压力；
     - 将重点逐步转移到 Deep Fusion 架构本身和 Flow Matching 目标的优化上，而不是无限放大解冻强度。

3. 阶段 C（P2，可选，高级语言微调）：
   - 仅在需要显著提升语言理解或指令跟随能力时考虑；
   - 可能的方向：
     - 在保持 Deep Fusion 动作训练的同时，引入少量语言监督（如任务描述/对话的 CE loss），采用 prefix-LM 或 SFT；
     - 使用 LoRA / Adapter 等参数高效方法，仅在少量插入层上训练额外参数，而保持原有权重大体冻结；
   - 风险提示：该阶段风险较高，可能影响通用能力或导致数值更不稳定，建议在阶段 A/B 完全收敛并评估后再酌情尝试。

### 9.4 当前 TODO 摘要（与语言模型微调相关）

- [x] 在冻结骨干 + FP32 配置下，完成至少一个小规模数据集的端到端 Flow Matching 训练，对比 Late Fusion；
- [x] 记录训练稳定性（是否出现 NaN/Inf）、loss 曲线和动作质量，确认在现有架构与目标下 loss 难以降至 openpi/PI0 的 0.02 级别；
- [ ] 在阶段 B 设定下，系统探索不同 `backbone_unfreeze_layers`、`backbone_lr_scale` 与 `expert_lr_scale` 组合，评估其收益与显存开销；
- [ ] 在阶段 B 的基础上，启动 Deep Fusion 架构级改造（将 `_compute_layer_complete` 抽象为 per-block 的 `DeepFusionBlock`，为更原生 multi-expert 做准备）；
- [ ] 若未来需要语言行为显著增强，再设计阶段 C 的具体数据与损失权重（多任务：语言 + 动作）。
