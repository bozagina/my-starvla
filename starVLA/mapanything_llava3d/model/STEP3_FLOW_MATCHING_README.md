# 步骤 3: Flow Matching与 Deep Fusion 集成完成 ✅

**完成日期**: 2024-12-30  
**核心文件**: `modeling_flow_expert_dev.py`  
**实现**: 完整重写

---

## 📋 实现概述

步骤 3 成功将 **Flow Matching Action Expert** 从基于 Gemma 的 Late Fusion 架构改造为基于 `LLaVA3DWithActionExpertModel` 的 **Deep Fusion** 架构，实现了视觉-语言-动作的端到端深度交互。

### 核心特性

✅ **完全删除 Gemma 依赖**
- 不再继承 `GemmaPreTrainedModel`
- 改为纯 `nn.Module`
- 网络前向委托给 `LLaVA3DWithActionExpertModel`

✅ **完整保留 Flow Matching 数学**
- t ~ Uniform(0, 1) 时间采样
- x_t = t * noise + (1-t) * actions 插值
- u_t = noise - actions 速度目标
- Euler ODE solver 积分

✅ **灵活的 Suffix 结构**
- 支持 proprioceptive state (机器人状态)
- Action tokens: 每个 action step 独立 embedding
- Time token: sinusoidal embedding + MLP

✅ **Deep Fusion 集成**
- Prefix (vision+language) 和 Suffix (state+action+time) 联合前向
- 每层都有 cross-attention（不再是简单的 global pooling）

---

## 🏗️ 架构设计

### 类结构

```python
class FlowMatchingActionExpert(nn.Module):
    """
    Flow Matching with LLaVA3D Deep Fusion
    
    组件:
    - state_proj: [state_dim] → [hidden_size] (optional)
    - action_in_proj: [action_dim] → [hidden_size]
    - time_mlp_in/out: [hidden_size] → [hidden_size]
    - action_out_proj: [hidden_size] → [action_dim]
    - llava_with_expert: LLaVA3DWithActionExpertModel (外部)
    """
    
    def __init__(self, llava_with_expert_model, action_dim, action_horizon, ...)
    def _construct_suffix_embeddings(self, actions, time, state)
    def forward(self, prefix_embs, actions, time, state, ...)
    def compute_loss(self, prefix_embs, actions, state, ...)
    def sample_actions(self, prefix_embs, state, num_steps=10, ...)
```

### Suffix 序列结构

```
┌─────────────┬────────────────────────┬────────────┐
│ state_token │ action_token_1, ..., H │ time_token │
│ (optional)  │                        │            │
└─────────────┴────────────────────────┴────────────┘
    [B,1,H]         [B, H, H]           [B, 1, H]

Total length: (0 or 1) + H + 1 tokens
```

### Flow Matching 流程

**训练**:
```
Ground truth actions a ∈ R^{B×H×action_dim}
  ↓
Sample t ~ U(0,1), ε ~ N(0,I)
  ↓
Construct x_t = t·ε + (1-t)·a
  ↓
Target velocity u_t = ε - a
  ↓
Predict v_t = model(prefix_embs, x_t, t, state)
  ↓
Loss = MSE(v_t, u_t)
```

### 动作 Horizon 与数据对齐（与 openpi 保持一致）

- 在原生 openpi 训练中，`actions` 的形状由配置中的 `config.model.action_horizon` 决定：
  - DataLoader 通过 `create_torch_dataset(..., action_horizon=config.model.action_horizon, ...)` 调用 `LeRobotDataset`
  - `LeRobotDataset` 使用 `delta_timestamps` 和 `fps` 在时间轴上构造长度为 `H = action_horizon` 的动作序列
  - 测试中显式断言：`actions.shape == (batch_size, action_horizon, action_dim)`
- 在本项目的 Deep Fusion 集成中，我们对 Libero 走同样的路径：
  - 对于 `repo_id="physical-intelligence/libero"` 且设置了 `LIBERO_LOCAL_ROOT` 的情况，
    `create_torch_dataset` 会直接调用本地 `LeRobotDataset`，传入相同的 `action_horizon` 和 `delta_timestamps`
  - 这样从 DataLoader 输出的 `actions` 也满足：
    `actions.shape == (batch_size, config.model.action_horizon, config.model.action_dim)`
- FlowMatchingActionExpert 始终假定：
  - 输入 `actions` 的时间维 `H` 与配置的 `action_horizon` 一致
  - Flow Matching 中的 `x_t`, `u_t`, `pred_velocity`、以及 suffix 序列长度都围绕这个统一的 `H` 展开
  - 从而避免 `pred_velocity` 和 `target_velocity` 在时间维上出现广播式的 shape mismatch

**推理**:
```
Start: x_1 ~ N(0, I) (pure noise)
  ↓
For t from 1.0 to 0.0 (step=-1/num_steps):
  v_t = model(prefix_embs, x_t, t, state)
  x_t = x_t + v_t * dt
  ↓
End: x_0 (clean actions)
```

---

## 🔧 使用方法

### 初始化

```python
from modeling_llava3d_v2_dev import LLaVA3DForCausalLMV2, LLaVA3DWithActionExpertModel
from modeling_flow_expert_dev import FlowMatchingActionExpert

# 1. 加载 base LLaVA3D
base_llava = LLaVA3DForCausalLMV2.from_pretrained("path/to/llava3d")

# 2. 创建 Deep Fusion 模型
llava_with_expert = LLaVA3DWithActionExpertModel(base_llava)

# 3. 创建 Flow Matching Expert
flow_expert = FlowMatchingActionExpert(
    llava_with_expert_model=llava_with_expert,
    action_dim=7,              # 机器人自由度
    action_horizon=10,         # 预测步数
    state_dim=14,              # 关节角度+速度
    use_state=True,            # 使用 proprioceptive state
)
```

### 训练

```python
# 准备输入
prefix_embs = get_image_text_embeddings(...)  # [B, L_p, H]
actions = get_ground_truth_actions(...)       # [B, 10, 7]
state = get_robot_state(...)                  # [B, 14]

# 计算 Flow Matching loss
loss = flow_expert.compute_loss(
    prefix_embs=prefix_embs,
    actions=actions,
    state=state,
)

# 反向传播
loss.backward()
optimizer.step()
```

### 推理

```python
# 预测动作
with torch.no_grad():
    predicted_actions = flow_expert.sample_actions(
        prefix_embs=prefix_embs,
        state=state,
        num_steps=20,  # 更多步数 = 更精确
    )  # [B, 10, 7]

# 执行动作
robot.execute(predicted_actions[0])  # [10, 7]
```

---

## 📊 与原始实现的对比

| 维度 | Late Fusion (原始) | Deep Fusion (本实现) |
|------|-------------------|----------------------|
| **网络** | 独立 Gemma Expert | LLaVA3D 共享 Transformer |
| **上下文** | Global pooling (mean) | ✅ 每层 cross-attention |
| **参数量** | +3B (Gemma) | ✅ 0 (复用 LLaVA3D) |
| **视觉交互** | 仅最后一层 | ✅ 每层都交互 |
| **State 支持** | 无 | ✅ 可选 state token |
| **表达能力** | 中 (浅层融合) | ✅ 强 (深层融合) |

### 效果提升（理论预期）

- **更强的视觉理解**: Suffix 在每层都能 attend to prefix
- **更精确的动作预测**: 深层 Transformer 替代浅层 MLP
- **更少的参数**: 复用 LLaVA3D，无需额外 Gemma Expert
- **更好的泛化**: 视觉-语言-动作端到端训练

---

## 🔍 代码详解

### 核心方法 1: `_construct_suffix_embeddings`

**功能**: 将 state, actions, time 转换为 suffix token 序列

```python
def _construct_suffix_embeddings(self, actions, time, state):
    """
    输入:
    - actions: [B, H, action_dim] noisy or clean actions
    - time: [B] time values in [0, 1]
    - state: [B, state_dim] proprioceptive state (optional)
    
    输出:
    - suffix_embs: [B, L_s, hidden_size]
      where L_s = (0 or 1) + H + 1
    """
    suffix_tokens = []
    
    # 1. State token (optional)
    if self.use_state and state is not None:
        state_token = self.state_proj(state).unsqueeze(1)  # [B, 1, H]
        suffix_tokens.append(state_token)
    
    # 2. Action tokens
    action_tokens = self.action_in_proj(actions)  # [B, H, hidden_size]
    suffix_tokens.append(action_tokens)
    
    # 3. Time token
    time_embed = create_sinusoidal_pos_embedding(time, self.hidden_size, ...)
    time_embed = self.time_mlp_in(time_embed)
    time_embed = F.silu(time_embed)
    time_embed = self.time_mlp_out(time_embed)
    time_token = time_embed.unsqueeze(1)  # [B, 1, hidden_size]
    suffix_tokens.append(time_token)
    
    # Concatenate
    suffix_embs = torch.cat(suffix_tokens, dim=1)
    return suffix_embs
```

### 核心方法 2: `forward` (Deep Fusion)

**功能**: 通过 LLaVA3D Deep Fusion 预测 velocity

```python
def forward(self, prefix_embs, actions, time, state, ...):
    """
    输入:
    - prefix_embs: [B, L_p, H] vision + language
    - actions: [B, H, action_dim] noisy actions x_t
    - time: [B] current time t
    - state: [B, state_dim] robot state
    
    输出:
    - pred_velocity: [B, H, action_dim]
    """
    # Step 1: Construct suffix
    suffix_embs = self._construct_suffix_embeddings(actions, time, state)
    
    # Step 2: Deep Fusion forward
    outputs, _ = self.llava_with_expert(
        inputs_embeds=[prefix_embs, suffix_embs],  # Mode 3: joint
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    
    prefix_output, suffix_output = outputs
    # 注意: prefix 和 suffix 在每层都互相感知了！
    
    # Step 3: Extract action tokens
    if self.use_state:
        # Skip state_token and time_token
        action_hidden = suffix_output[:, 1:1+self.action_horizon, :]
    else:
        # Skip time_token
        action_hidden = suffix_output[:, :self.action_horizon, :]
    
    # Step 4: Project to velocity
    pred_velocity = self.action_out_proj(action_hidden)
    
    return pred_velocity
```

### 核心方法 3: `compute_loss` (Flow Matching Training)

**功能**: 实现 Flow Matching 训练算法

```python
def compute_loss(self, prefix_embs, actions, state, ...):
    """
    Flow Matching Loss:
    1. Sample t ~ U(0,1) and noise ~ N(0,I)
    2. Construct x_t = t*noise + (1-t)*actions
    3. Target u_t = noise - actions
    4. Predict v_t = forward(prefix_embs, x_t, t, state)
    5. Loss = MSE(v_t, u_t)
    """
    batch_size = actions.shape[0]
    device = actions.device
    
    # Sample time and noise
    t = torch.rand((batch_size,), device=device)
    noise = torch.randn_like(actions)
    
    # Construct noisy actions
    t_exp = t.view(batch_size, 1, 1)
    x_t = t_exp * noise + (1 - t_exp) * actions
    
    # Target velocity
    target_velocity = noise - actions
    
    # Predict velocity
    pred_velocity = self.forward(prefix_embs, x_t, t, state, ...)
    
    # MSE loss
    loss = F.mse_loss(pred_velocity, target_velocity)
    
    return loss
```

### 核心方法 4: `sample_actions` (Euler ODE Sampling)

**功能**: 使用 Euler ODE solver 从噪声生成 clean actions

```python
@torch.no_grad()
def sample_actions(self, prefix_embs, state, num_steps=10, ...):
    """
    Euler ODE Solver:
    - Start: x_1 ~ N(0, I)
    - Loop: x_t = x_t + v_t * dt (t: 1 → 0)
    - End: x_0 (clean actions)
    """
    batch_size = prefix_embs.shape[0]
    device = prefix_embs.device
    
    # Initialize with noise
    action_shape = (batch_size, self.action_horizon, self.action_dim)
    x_t = torch.randn(action_shape, device=device)
    
    # Time step
    dt = -1.0 / num_steps  # Negative (backward in time)
    
    # Euler integration
    for step in range(num_steps):
        t_curr = 1.0 + step * dt  # 1.0, 0.9, 0.8, ..., 0.1
        t_tensor = torch.full((batch_size,), t_curr, device=device)
        
        # Predict velocity
        v_t = self.forward(prefix_embs, x_t, t_tensor, state, ...)
        
        # Euler step
        x_t = x_t + v_t * dt
    
    return x_t  # x_0
```

---

## 📈 进度更新

```
步骤 0: 创建 dev 副本         ✅ 完成
步骤 1: 最小版本框架          ✅ 完成  
步骤 2: 双流联合注意力        ✅ 完成
步骤 3: Flow Matching 集成   ✅ 完成 ⭐ (本次)
步骤 4: Wrapper 集成         ⏳ 待实现
```

**整体完成度：~75%**

---

## 🚀 下一步（步骤 4）

步骤 3 已经完成了 Flow Matching 算法层，下一步需要在顶层 wrapper 中集成：

### 步骤 4: 改造 `MapAnythingLlava3DForConditionalGeneration`

**任务清单**:
1. **初始化 FlowMatchingActionExpert**
   ```python
   self.action_expert = FlowMatchingActionExpert(
       llava_with_expert_model=self.language_model_with_expert,
       action_dim=config.action_dim,
       action_horizon=config.action_horizon,
       ...
   )
   ```

2. **构造 prefix_embs** (在 `forward`)
   ```python
   # Get image + geometric features
   image_features = self.get_image_features(pixel_values, intrinsic)
   
   # Get text embeddings
   text_embeds = self.get_input_embeddings()(input_ids)
   
   # Inject image features at <image> token positions
   prefix_embs = inject_image_to_text(text_embeds, image_features, image_mask)
   ```

3. **训练路径** (在 `forward` with actions)
   ```python
   if actions is not None and self.action_expert is not None:
       action_loss = self.action_expert.compute_loss(
           prefix_embs=prefix_embs,
           actions=actions,
           state=state,
       )
       return MapAnythingLlava3DOutput(loss=action_loss, ...)
   ```

4. **推理路径** (在 `predict_action`)
   ```python
   # Construct prefix
   prefix_embs = ...
   
   # Sample actions
   predicted_actions = self.action_expert.sample_actions(
       prefix_embs=prefix_embs,
       state=state,
       num_steps=20,
   )
   
   return predicted_actions
   ```

---

## 📚 参考文档

- **实现方案**: `llava3d_deep_fusion_plan.md`（已更新步骤 3 状态）
- **步骤 2**: `STEP2_DEEP_FUSION_README.md`
- **PI0 参考**: `mapAnythingLlava3dPi0.5/openpi/models_pytorch/`

---

## ✨ 贡献者

**实现者**: AI Assistant  
**审核者**: 待定  
**日期**: 2024-12-30

---

## 📝 更新日志

### 2024-12-30
- ✅ 完全删除 Gemma 依赖
- ✅ 实现 `_construct_suffix_embeddings` (state + actions + time)
- ✅ 实现 `forward` (Deep Fusion 集成)
- ✅ 实现 `compute_loss` (Flow Matching 训练)
- ✅ 实现 `sample_actions` (Euler ODE 采样)
- ✅ 添加 proprioceptive state 支持
- ✅ 完整的文档和注释

---

**状态**: ✅ 步骤 3 完成  
**下一步**: 🚧 步骤 4 - 集成到 MapAnythingLlava3DForConditionalGeneration
