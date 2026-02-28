# Path A 算法再评审资料包（给外部大模型）

## 1. 目标与上下文

本资料用于让外部大模型对当前 `Path A`（几何残差反馈 token）实现做系统复核。  
当前现象：`Path A` 已接入且训练稳定，但相较 baseline 没有表现出“更快收敛/更明显 loss 优势”。

版本口径（重要）：

- `v3/v4(旧)`：Path A residual 主要来自 pooled `delta_z`（`[B,K,H] -> [B,H]`）。
- `v4-1`：在 action 输出端增加 `Δa` 残差动作头（有 gate + clip 上界）。
- `v4-1-1`（本次新增方案）：residual 不再依赖旧的 pooled `delta_z` 主路径，改为 **token-level 几何残差 `Δgeo` + soft mask 过滤**，再接 `v4-1` 的 `Δa` 注入。

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

### 2.2 Path A 训练链路（旧 residual 主路径：v3/v4）

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

### 2.5 Path A 的输入/输出与形状定义（当前实现）

按训练前向中实际张量约定（`MapAnythingLlava3DPI.forward`）：

- `task_tokens`: `[B, K, H]`  
  - `B`: batch size  
  - `K`: task token 数（通常 32）  
  - `H`: token hidden dim（当前为 4096）
- `task_tokens_next`: `[B, K, H]`  
  - 来自 `t+k` 图像分支（优先 `extract_task_tokens`）
- `action_chunk`（训练里是 GT `actions_target`）: `[B, T, A]`  
  - `T`: 预测窗口长度（`future_action_window_size + 1`）  
  - `A`: action dim（当前为 7）
- `action_context`: `[B, A]`  
  - 由 `build_world_action_context` 从 `[B,T,A]` 汇聚得到（当前默认 `prefix_mean`）
- `z_before`, `z_after`: `[B, H]`  
  - 由 action-conditioned pooling 从 `[B,K,H]` 汇聚
- `delta_z = z_after - z_before`: `[B, H]`
- `feedback_tokens`: `[B, Kf, H]`  
  - `Kf = causal_feedback_token_num`（当前配置为 4）
- `context_task_tokens`: `[B, Kf + K, H]`  
  - 在 action head 内部由 `cat([feedback_tokens, task_tokens], dim=1)` 构造
- 辅助分支：  
  - `pred_delta`: `[B, H]`（由 `feedback_tokens.mean(dim=1)` 经 recon head 得到）  
  - `delta_z_target`: `[B, H]`  
  - `loss_fb = mse(pred_delta, delta_z_target) + dir_w * (1 - cos)`

补充：当前设置 `repeated_diffusion_steps > 1` 时，进入 action head 的有效 batch 变为 `B' = B * repeated_diffusion_steps`。

### 2.6 Path A 当前到底在训练什么模块（实操口径）

在 `enable_causal_feedback_token=true` 时，Path A 新增可学习模块包括：

- `causal_feedback_delta_norm`
- `causal_feedback_action_proj`
- `causal_feedback_action_norm`
- `causal_feedback_fuse`
- `causal_feedback_recon_head`（仅当 `causal_feedback_aux_weight > 0` 才有训练驱动）

同时，`feedback_tokens` 注入 action head 后，会通过主任务 loss 影响 action head 参数（DiT/cross-attn/action decoder）。

关键梯度路径说明（当前实现）：

- `t+k` 分支抽取 `task_tokens_next` 使用 `torch.no_grad()`，因此该分支本身不参与反传（见 `MapAnythingLlava3DPI.forward`）。
- 若 `causal_feedback_detach_delta=true`，则 `delta_z` 反传被切断，Path A 对上游 token/pooling 的驱动明显变弱。
- 若 `causal_feedback_aux_detach_target=true`，则 `loss_fb` 不反推 `delta_z_target` 分支，只训练预测侧。

### 2.7 v4-1（当前已落地）动作侧注入口径

`v4-1` 的核心是把 Path-A 从“强改 context”改成“有上界的后置纠偏”：

- `a_out = a_base + gate * clip(DeltaA(mean(feedback_tokens)))`
- `clip` 使用 `tanh` 限幅；`gate` 由 `alpha schedule * valid_tk` 构成。
- 该路径不新增 VLM forward，只增加 action head 轻量计算。

关键实现：

- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1253`  
  `_apply_feedback_delta_action(...)`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1303`  
  `feedback_vec = feedback_tokens.mean(dim=1)`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1309`  
  `clip_value * tanh(raw_delta/clip)`
- `/Users/bazinga/code/my-starvla/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:1337`  
  `effective_delta = gate * clipped_delta`

`v4-1` 已有关键监控（低成本）：

- `debug/sagr/delta_action_raw_norm_mean`
- `debug/sagr/delta_action_clip_saturation_frac`
- `debug/sagr/delta_action_last_layer_weight_rms`
- `debug/sagr/delta_action_gate_mean`
- `debug/sagr/delta_action_effective_norm_mean`

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

补充：这也是 `v4-1-1` 的直接改动动机，即把 residual 从 pooled 向量升级为 token-level `Δgeo`。

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

### B5. 量化证据显示 Path A“被启用但使用强度偏弱”

基于最新 run：

- 配置：  
  `/Users/bazinga/code/my-starvla/_remote_runs/1229_libero4in1_MapAnythingLlava3DPI_s42_20260226_005832/config.yaml`
- 指标：  
  `/Users/bazinga/code/my-starvla/_remote_runs/1229_libero4in1_MapAnythingLlava3DPI_s42_20260226_005832/metrics.jsonl`

对前 31 条 step 级日志统计结果：

- Path A 一直生效：  
  - `debug/causal_feedback/applied = 1.0`  
  - `debug/causal_feedback/action_conditioned_pooling = 1.0`
- 但 slot 选择性几乎没有形成：  
  - `slot_entropy_before/after ≈ 3.465735`  
  - 与 `ln(32)=3.465736` 几乎重合（接近均匀分配）
- feedback 对主 loss 的权重占比偏小：  
  - `loss_fb_weighted` 均值约 `0.0075`  
  - `action_loss_base` 均值约 `0.638`  
  - 推导 `fb_weighted_over_base = loss_fb_weighted / action_loss_base` 均值约 `1.7%`（早期约 `0.7%`）
- 训练阶段仍极早：  
  - 仅约 `epoch=0.02`  
  - lr 仍在 warmup 前段（`1e-8 -> 3.1e-7`）  
  - 短窗口下难观察到稳定增益
- 额外事实：  
  - `debug/timing/vlm_forward_ratio` 约 `0.74`，VLM 双前向成本占比高  
  - 当前注入是 concat，无门控，反馈 token 在大上下文中可能被弱化

由此，当前现象更符合“Path A 已接入且稳定，但属于弱条件注入，尚未形成明显主任务主导贡献”。

### B6. 为什么 `slot_entropy` 看起来几乎不变

当前配置中 `loss_w_geo=0`，而 `residual_slot_entropy_weight` 的作用点在 `loss_geo` 内部。  
这意味着即使设置了 `residual_slot_entropy_weight`，在 `loss_w_geo=0` 时这条正则对总损失无实际贡献，因此 `slot_entropy` 很可能保持在近均匀状态。

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

---

## 9. 新增方案：v4-1-1（token-level 几何残差 residual）

本节是当前建议的主线改造方案，作为 `v4-1` 的增量升级。

### 9.1 原理

旧方案的主要瓶颈在 residual 构造阶段：`[B,K,H] -> [B,H]` 的 pooled `delta_z` 容易丢失空间位置差异。  
`v4-1-1` 改为在 token 级别构造 residual，再由语言条件产生 soft mask 过滤，最后才汇聚成动作纠偏向量。

核心思想：

1. 先保留空间残差：`r_tokens = geo_after - geo_before`，形状 `[B, N, H]`。  
2. 再做条件筛选：`alpha`（soft mask，`[B,N]`）强调与语言相关的 token。  
3. 最后汇聚注入：`fb_vec = LN(sum(alpha * r_tokens))`，接 `v4-1` 的 `Δa` 头。

### 9.2 目标

`v4-1-1` 目标是同时解决两个问题：

1. 模块2（residual 构造）噪声大、信息被压缩。  
2. 模块3（注入与利用）容易“过强但不稳”或“过弱被忽略”。

具体期望：

- 提高 residual 的语义-空间对齐性（减少无关残差注入）。
- 保持注入有上界（继续沿用 `gate + clip`）。
- 在不增加额外 VLM forward 的前提下，提高 Path-A 的有效利用率。

### 9.3 想法（结构抽象）

把 Path-A 统一成三模块：

1. 模块1：`z_t`（融合表征）  
   来源于当前多模态链路（vision + geo + language），提供 query 与 token 特征。
2. 模块2：`r_t`（残差构造）  
   由 token-level `Δgeo` 构成：`r_tokens = geo_after - geo_before`。
3. 模块3：注入与利用  
   `a_out = a_base + gate * clip(DeltaA(fb_vec))`，其中 `fb_vec` 来自 masked residual。

关系说明：

- `v4-1` 解决模块3稳定性（注入方式）。
- `v4-1-1` 重点升级模块2质量（residual 来源与筛选）。

### 9.4 实现（建议落点与张量形状）

#### A. 输入张量（对齐口径）

- `vision_tokens`: `[B, N=512, H]`（2 视角，单视角 `16x16=256`，拼接后 512）。
- `geo_tokens_before`: `[B, 512, H]`
- `geo_tokens_after`: `[B, 512, H]`
- `language_queries`: `[B, Lq, H]`，`Lq <= semantic_query_max_tokens`（默认 64）。

注：`vision/geo` token 顺序必须保持同 patch 对齐（同一 index 对应同一视角同一 patch）。

#### B. soft mask 构造

双通道注意力（语言 query 到 token）：

- `alpha_vis`: 由 `Q=language_queries, K/V=vision_tokens` 得到 `[B,512]`
- `alpha_geo`: 由 `Q=language_queries, K/V=geo_tokens_before` 得到 `[B,512]`
- 融合：`alpha = normalize((1-lambda)*alpha_vis + lambda*alpha_geo)`
- 可选稳定：`alpha_ema = (1-beta)*alpha_ema + beta*alpha`

建议初值：

- `lambda = 0.3`
- `beta = 0.2`

#### C. residual 构造（替换旧 pooled 主路径）

- `r_tokens = geo_tokens_after - geo_tokens_before`，`[B,512,H]`
- `r_tokens_weighted = alpha[...,None] * r_tokens`
- `fb_vec = LN(sum(r_tokens_weighted, dim=1))`，`[B,H]`
- 与现有 action head 接口兼容：`feedback_tokens = fb_vec.unsqueeze(1)`（或 repeat 到 `Kf`）

#### D. 模块3注入（复用 v4-1）

保持现有：

- `raw_delta = DeltaA(fb_vec)`
- `delta = clip * tanh(raw_delta / clip)`
- `a_out = a_base + gate * delta`
- `gate = alpha_schedule * valid_tk`（先不引入复杂置信门控）

### 9.5 配置建议（v4-1-1）

建议新增/整理配置（命名可按工程风格微调）：

- `patha_residual_mode: token_delta_geo`（旧值可为 `pooled_delta_z`）
- `soft_mask_enabled: true`
- `soft_mask_lambda: 0.3`
- `soft_mask_ema_beta: 0.2`
- `soft_mask_query_agg: mean`
- `soft_mask_temperature: 1.0`

继续沿用 `v4-1` 关键项：

- `feedback_delta_action_enabled: true`
- `feedback_in_context_enabled: false`（建议先关闭 context 拼接，减少混杂变量）
- `feedback_delta_action_alpha_mode: schedule`
- `feedback_delta_action_alpha_init: 0.0`
- `feedback_delta_action_alpha_target: 0.05~0.2`（做 sweep）
- `feedback_delta_action_clip: 0.05`（先固定）

### 9.6 验证与验收

必须看三类证据：

1. 注入强度是否可控  
   - `delta_action_gate_mean`
   - `delta_action_effective_norm_mean`
   - `hidden_perturb_ratio`
2. 纠偏净贡献是否稳定  
   - `ΔL_probe_light = L_with_fb - L_zero_fb`
   - 统计“负值占比”与方差
3. residual 质量是否改善  
   - `delta_norm_p50/p95`（before/after mask）
   - `mask_entropy`
   - `topk_mass@32`
   - `mask_cos_prev`（时序稳定）

建议通过阈值（可按任务再调）：

- `ΔL_probe_light` 负值占比 > 70%
- `hidden_perturb_ratio` 不回到高扰动区（例如长期 > 0.6）
- `mse_score_patha` 不再长期劣于 base

### 9.7 回滚与兼容

为确保可快速回退，保留双路径开关：

- `token_delta_geo`（新）
- `pooled_delta_z`（旧）

并维持 action 注入路径一致（`v4-1` 不变）。  
这样可以把“residual 构造改动”的收益与风险单独归因，不和模块3改动混淆。
