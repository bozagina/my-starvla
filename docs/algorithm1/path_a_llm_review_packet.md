# Path A 算法再评审资料包（给外部大模型）

## 1. 目标与上下文

本资料用于让外部大模型对当前 `Path A`（几何残差反馈 token）实现做系统复核。  
当前现象：`Path A` 已接入且训练稳定，但相较 baseline 没有表现出“更快收敛/更明显 loss 优势”。

你需要重点回答：

1. `Path A` 的信息构造是否有效（几何残差定义是否保真）。  
2. `Path A` 的注入方式是否足够强（模型是否真正使用了反馈信号）。  
3. 训练目标是否足够约束 `Path A`，避免其退化为弱条件。  
4. 如何做最小侵入优化，并用可验证指标证明有效。

---

## 2. 当前实现总览（简版）

### 2.1 VLM 侧任务 token（固定 K）

- 使用固定数量 learnable task queries，分别 cross-attn 到几何特征和视觉特征，再融合得到 `task_hidden_states`。  
- `task_hidden_states` 形状固定为 `[B, K, H]`。

关键实现：

- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:490`  
  `_build_fixed_task_tokens(...)`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:557`  
  `fusion_module(...)`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py:585`  
  `extract_task_tokens(...)`（tk fast path）

### 2.2 Path A 训练链路

1. `t` 帧跑主 VLM 前向，取 `task_tokens`。  
2. `t+k` 帧跑第二次前向（优先 `extract_task_tokens`），取 `task_tokens_next`。  
3. 用 action-conditioned pooling 把 `[B,K,H] -> [B,H]`，得到 `z_before/z_after`。  
4. `delta_z = z_after - z_before`；与动作上下文融合，生成 `feedback_tokens`。  
5. 将 `feedback_tokens` 拼到 action head 的 `context_task_tokens`。  
6. 额外辅助损失：`loss_fb`（重构 delta + 可选方向项）。

关键实现：

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:441`  
  `_build_causal_feedback_tokens(...)`
- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:587`  
  `_compute_causal_feedback_aux_loss(...)`
- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:700`  
  `forward(...)`（双时刻 token 获取 + Path A 训练组装）

### 2.3 Path A 推理链路

- 维护上一时刻 `task_tokens` 与上一动作 chunk。  
- 当前帧拿到新 `task_tokens` 后，自动构造 `feedback_tokens`（`auto_path_a`），传给 action head 做下一 chunk 预测。

关键实现：

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:1737`  
  推理中自动 Path A 构造逻辑
- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:1805`  
  `predict_action(..., feedback_tokens=...)`

### 2.4 Action head 中反馈注入

- 当前注入策略是：`context_task_tokens = cat([feedback_tokens, task_tokens], dim=1)`。  
- 属于“拼接式条件注入”，没有显式门控增益。

关键实现：

- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1115`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1139`  
  `_apply_layerwise_cross_attention(..., task_tokens=context_task_tokens)`

---

## 3. 关键配置项（Path A）

关键配置位于 `framework.action_model`：

- `enable_causal_feedback_token`
- `causal_feedback_token_num`
- `causal_feedback_detach_delta`
- `causal_feedback_detach_action`
- `causal_feedback_scale`
- `feedback_context_scale`
- `causal_feedback_aux_weight`
- `causal_feedback_aux_dir_weight`
- `causal_feedback_aux_detach_target`
- `world_action_context_mode`（建议 `prefix_mean`）
- `world_action_prefix_len`
- `residual_pooling_mode`（`mean` / `slot_weighted`）

配置读取入口：

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:71`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:374`

---

## 4. 当前已知瓶颈（请重点复核）

### B1. 注入强度可能不足

虽然 `feedback_tokens` 已拼接到 context，但模型可能在 cross-attn 中弱化该分支，导致对主 loss 贡献有限。

证据入口：

- `debug/causal_feedback/fb_weighted_over_base`
- `debug/causal_feedback/token_norm_mean`

### B2. 信息压缩可能过强

`delta_z` 先从 `[B,K,H]` 聚合成 `[B,H]`，再映射为少量 feedback tokens；关键 slot 差异可能被稀释。

证据入口：

- `debug/causal_feedback/slot_entropy_before`
- `debug/causal_feedback/slot_entropy_after`
- `debug/causal_feedback/delta_z_norm_mean`

### B3. detach 组合可能抑制可学习性

若 `causal_feedback_detach_delta=true`、`causal_feedback_aux_detach_target=true`，Path A 对上游表征与 pooling 的反向驱动受限。

代码入口：

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:77`
- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py:97`

### B4. 双前向成本较高，影响有效迭代速度

训练里 `t` 与 `t+k` 两次 VLM 前向会显著占时；若收益不足，整体看起来“收敛不更快”。

证据入口：

- `debug/timing/train_vlm_t_ms`
- `debug/timing/train_vlm_tk_ms`
- `debug/timing/vlm_forward_ratio`

---

## 5. 评审任务（请外部大模型完成）

请按下面顺序输出：

1. **机制诊断**  
   解释为什么 Path A 在当前实现下可能“稳定但不明显提速收敛”。

2. **最小侵入优化（按优先级）**  
   每条包含：
   - 改动点（函数/文件级）
   - 理论动机
   - 实施复杂度（低/中/高）
   - 风险
   - 成功判据（具体指标）

3. **Ablation 方案（至少 8 个）**  
   覆盖：
   - 注入强度（`feedback_context_scale`）
   - token 数（`causal_feedback_token_num`）
   - detach 策略
   - aux 损失策略（mse/dir）
   - pooling 策略（mean vs slot_weighted）
   - 注入结构（concat vs gated-add）

4. **两周可落地路线图**  
   先做配置级，再做小代码改，最后做结构改，给出回滚策略。

---

## 6. 建议给外部大模型的输入材料

最低需要提供：

- 代码文件：
  - `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
  - `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
  - `/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/MapAnythingLlava3D.py`
- 两组训练日志（baseline vs Path A 对齐 warmup/LR）
- 当前训练配置 yaml

---

## 7. 可直接复制的专业 Prompt（中文）

你是一个由以下角色组成的联合评审组：  
1) 多模态大模型/VLM 架构专家  
2) 机器人策略学习与时序建模专家  
3) 深度学习优化与训练稳定性专家  

你的任务：对我当前的 Path A（几何残差反馈 token）实现做严谨复核，并给出“可落地、可验证、低侵入”的优化方案。

### 背景与目标

我当前的系统中，Path A 设计目标是：  
利用两时刻任务 token 的几何残差 delta_z，构造 feedback tokens，作为下一轮动作生成的附加条件，从而提升动作去噪质量与收敛效率。

当前现象：  
训练稳定，但相比 baseline，主损失（action_dit_loss）没有出现预期的更快下降，Path A 增益不明显。

### 你必须先阅读并绑定以下代码

- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- `/Users/bazinga/code/my-starvla/starVLA/mapanything_llava3d/model/modeling_mapanything_llava3d_vlm.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/vlm/MapAnythingLlava3D.py`

重点关注函数：

- `MapAnythingLlava3DPI._build_causal_feedback_tokens`
- `MapAnythingLlava3DPI._compute_causal_feedback_aux_loss`
- `MapAnythingLlava3DPI.forward`
- `LayerwiseFM_ActionHeader.forward`
- `LayerwiseFM_ActionHeader._apply_layerwise_cross_attention`
- `LayerwiseFM_ActionHeader.build_world_action_context`
- `LayerwiseFM_ActionHeader.pool_task_tokens`
- `modeling_mapanything_llava3d_vlm._build_fixed_task_tokens`
- `modeling_mapanything_llava3d_vlm.extract_task_tokens`

### 输出要求（严格）

请按以下结构输出：

1. **机制级诊断（先做）**
   - 当前 Path A 信息流是否形成有效闭环？
   - 哪些环节可能导致“有信号但不被主任务使用”？
   - 是否存在信息瓶颈（delta_z 聚合、token 数、注入点）？
   - detach 与 auxiliary loss 的耦合是否阻碍收敛？

2. **优化清单（至少 12 条）**
   每条必须包含：
   - 具体改动点（到文件/函数）
   - 理论原因
   - 复杂度（低/中/高）
   - 风险
   - 验证指标（必须量化）

3. **最小可行实验矩阵（至少 8 组）**
   覆盖：
   - feedback 注入强度（scale）
   - feedback token 数
   - detach 策略
   - aux loss（mse + dir）
   - pooling（mean / slot-weighted）
   - 注入结构（concat / gated）
   对每组给出：
   - 唯一变量
   - 固定项
   - 预期方向
   - 判败条件

4. **两周落地路线图**
   - Week1：配置级优化与判据
   - Week2：低侵入代码改动
   - 每步包含回滚条件

### 指标约束

你的所有判断必须绑定指标，不允许“感觉会更好”：

- `action_dit_loss`
- `debug/causal_feedback/loss_fb`
- `debug/causal_feedback/cos_delta_mean`
- `debug/causal_feedback/fb_weighted_over_base`
- `debug/causal_feedback/slot_entropy_before/after`
- `debug/causal_feedback/token_norm_mean`
- `debug/timing/vlm_forward_ratio`

### 工程约束

- 不允许推翻现有框架重写。  
- 优先给“最小侵入”修改。  
- 兼容当前训练/推理输入输出协议。  
- 先保证稳定，再追求收益。

---

## 8. 我建议你让外部大模型额外回答的 5 个关键问题

1. Path A 当前是否属于“弱条件注入”，若是，最低成本强化方式是什么？  
2. delta_z 聚合是否应从全局聚合改为多 token 保留（例如 4-token feedback）？  
3. detach 策略的推荐调度曲线是什么（按 step 分段）？  
4. concat 注入是否应升级为 gated 注入？何时值得做？  
5. 在不增加太多训练时间的前提下，最优的三项改动是什么？

