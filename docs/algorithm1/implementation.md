# SA-GR Flow（Semantic-Anchored Geometric Residual Flow）算法实现规格说明书

> **定位**：SA-GR Flow 将 3D-VLA 任务重构为一个“**语义导航（Semantic Navigation）** + **几何纠偏（Geometric Correction）**”的耦合系统，用于提升任务相关表征选择能力与闭环纠错能力。

---

## 1. 算法全景架构（System Architecture）

### 1.1 语义锚定层（Semantic Anchor）

**目标**：消除空间感缺失，实现任务驱动的特征筛选。  
**插入点**：`modeling_..._vlm.py` 中的 `fusion_module`。

**输入**：
- MapAnything 稠密 Token 池：\( G \in \mathbb{R}^{B \times N_g \times D} \)
- 语言嵌入：\( L_{\text{embed}} \)（指令 tokens）

**核心逻辑**：
- 计算 Cross-Attention：
  - Query：来自 \( L_{\text{embed}} \) 的指令 Token
  - Key / Value：来自 \( G \)
- 输出任务相关稀疏 Token：
  - \( S_{\text{task}} \in \mathbb{R}^{B \times N_s \times D} \)
- 用 \( S_{\text{task}} \) **替代原有全局平均**或粗粒度池化表征。

---

### 1.2 几何潜在动态模型（World Predictor \(\hat{\Phi}\)）

**目标**：赋予模型物理预见性，预测动作导致的几何演变。  
**插入点**：`LayerwiseFM_ActionHeader.py` 内部。

**输入**：
- 当前任务稀疏几何 Token：\( S_{\text{task}, t} \)
- 当前状态：\( \text{state}_t \)
- Flow Matching 候选动作：\( a_t \)

**核心逻辑**：
- 使用 4–6 层轻量级 Transformer 预测下一时刻任务相关几何：
  \[
  \hat{S}_{\text{task}, t+1}
  =
  S_{\text{task}, t}
  +
  \text{MLP}\big(\text{Attn}(S_{\text{task}, t}, a_t)\big)
  \]

---

### 1.3 残差纠偏与漂移注入（Residual Guidance）

**目标**：修正 OOD 干扰及物理接触不准等误差。

**核心定义**：
- 几何残差（推理时为**观测残差**）：
  \[
  \Delta z = S_{\text{task}, t+1} - \hat{S}_{\text{task}, t+1}
  \]
- 引导漂移（对动作的纠偏项）：
  \[
  b = -\eta(t) \cdot \nabla_a \|\Delta z\|^2
  \]
- 在 Flow Matching 的 Euler 步中注入：
  \[
  v_{\text{final}} = v_{\text{pred}} + b
  \]

---

## 2. 训练策略（Training Strategy）

采用 **四阶段离线训练**，确保模块间因果关系明确、便于稳定收敛：

1. **Stage 1：语义对齐训练**  
   优化 Cross-Attention 参数，使语言 Query 能准确权重化目标物体的 3D Token。

2. **Stage 2：动力学自监督训练**  
   利用专家轨迹序列数据 \((S_t, a_t, S_{t+1})\) 训练 \(\hat{\Phi}\)，学习物理演变规律。

3. **Stage 3：基础策略模仿**  
   训练 Flow Matching Head，建立基础动作场 \(v_\theta\) 的生成能力。

4. **Stage 4：闭环纠偏优化**  
   联合微调 \(\hat{\Phi}\) 与 Drift 模块；引入微小随机扰动，强化模型通过 \(b\) 自我修正的能力。

---

## 3. 损失函数与梯度流（Loss & Gradient Flow）

### 3.1 综合损失函数定义

\[
\mathcal{L}_{\text{total}}
=
w_{fm}\mathcal{L}_{CFM}
+
w_{dyn}\mathcal{L}_{dyn}
+
w_{geo}\mathcal{L}_{geo}
+
w_{reg}\mathcal{L}_{reg}
\]

**各项定义**：

- **\(\mathcal{L}_{CFM}\)**（流匹配损耗）  
  \[
  \mathbb{E}\left\|v_\theta - (a_{\text{exp}} - z)\right\|^2
  \]

- **\(\mathcal{L}_{dyn}\)**（动力学损耗）  
  \[
  \left\|\hat{\Phi}(S_t, a_t) - S_{t+1}\right\|^2
  \]

- **\(\mathcal{L}_{geo}\)**（几何能量项）  
  \[
  \frac{1}{2} r^\top W r
  \]
  其中 \(r\) 为接触点几何残差。

- **\(\mathcal{L}_{reg}\)**（漂移正则）  
  \[
  \int \|b\|^2 \, dt
  \]
  用于防止修正动作过大。

---

### 3.2 梯度图逻辑

- **纠偏梯度（Geometric Correction Gradient）**  
  从 \(\mathcal{L}_{geo}\) 出发，通过 \(\Delta z\) 的雅可比矩阵反向作用于动作流变量（影响 \(b\) 与最终 \(v_{\text{final}}\)）。

- **语义梯度（Semantic Selection Gradient）**  
  从 \(\mathcal{L}_{CFM}\) 出发，通过 Cross-Attention 权重反向作用于语言特征选择层（影响 \(S_{\text{task}}\) 的形成）。

---

## 4. 推理期：动态重规划（Dynamic Replanning）

引入基于 **残差相对率（RRR）** 的触发机制：

**判定公式**：
\[
\text{RRR}
=
\frac{\|\Delta z\|}{\|\text{Expected Change}\|}
\]

**触发操作**：
- 若 \(\text{RRR} > \tau_{\text{replan}}\)（建议 \(\tau_{\text{replan}} = 2.5\)）：
  - 立即清空当前动作序列
  - 重启 Flow Matching 采样流程  
  以适应剧变环境（例如目标位置被强制移动、接触状态突变等）。

---

## 5. 指标与日志埋点（新增）

以下指标已接入训练/推理日志，便于评估“两次 VLM 代价”“纠偏有效性”“实时性”。

### 5.1 训练效率指标（`metrics.jsonl`）

- `debug/timing/train_vlm_t_ms`：第 \(t\) 帧 VLM 前向耗时（ms）
- `debug/timing/train_vlm_tk_ms`：第 \(t+k\) 帧 VLM 前向耗时（ms）
- `debug/timing/train_vlm_total_ms`：两次 VLM 合计耗时（ms）
- `debug/timing/train_temporal_sync_ms`：时序特征对齐总耗时（构建 `t+k` 输入 + 前向 + 对齐）
- `debug/timing/train_action_head_ms`：Action Head 前向耗时（ms）
- `debug/timing/train_forward_total_ms`：单步前向总耗时（ms）
- `debug/timing/vlm_forward_ratio`：VLM 总耗时占前向总耗时比例（\(T_{vlm}/T_{total}\)）
- `debug/timing/temporal_sync_ratio`：时序对齐耗时占比（\(T_{sync}/T_{total}\)）
- `debug/timing/vlm_tk_over_vlm_t`：第二次 VLM 与第一次 VLM 耗时比

### 5.2 几何纠偏能力指标（`debug/sagr/*`）

- `debug/sagr/mse_dyn`：World Predictor 动力学误差（\(MSE_{dyn}\)）
- `debug/sagr/delta_z_norm_mean`：几何残差范数均值（\(\|\Delta z\|\)）
- `debug/sagr/geo_energy_mean` / `debug/sagr/E_geo`：几何能量均值
- `debug/sagr/rho_da`：几何残差方向与 drift 方向相关性（余弦相似度）
- `debug/sagr/drift_l2_mean`：纠偏漂移幅值

### 5.3 训练梯度占比指标

- `debug/grad_norm/geo_group_l2`：几何分支参数组梯度范数
- `debug/grad_norm/cfm_group_l2`：CFM 主干参数组梯度范数
- `debug/grad_norm/geo_over_cfm_ratio_proxy`：几何梯度与 CFM 梯度占比（工程代理指标）

### 5.4 推理实时性指标（`predict_action(..., return_debug_info=True)`）

- `timing/perception_ms` / `timing/geometric_latency_ms`：几何感知时延
- `timing/drift_response_lag_ms`：首次纠偏响应时延
- `timing/effective_control_hz`：有效纠偏频率（每秒修正次数）
- `timing/drift_freshness_ms_est`：纠偏新鲜度估计（`perception_ms + drift_response_lag_ms`）
- `timing/command_overlap_ratio_est`：动作块覆盖期间可提供纠偏更新次数估计
- `timing/phase_shift_steps_est`：相位差估计（以 control step 为单位）
- `geo/delta_z_norm_trace`：动作块内几何残差演化曲线
- `geo/energy_trace`：动作块内几何能量演化曲线
- `geo/drift_action_cos_trace`：动作块内 drift 方向一致性曲线

---

## 6. Path A（执行汇报 Token）最小落地

为支持“上一块执行结果影响下一块动作生成”，当前实现新增了可选的 `causal feedback token` 通道，并保持原训练/推理接口兼容。

### 6.1 数据与张量定义

- `task_tokens`：\([B, K, H]\)，来自 `image_t` 的任务 token。
- `task_tokens_next`：\([B, K, H]\)，来自 `image_tk` 的任务 token。
- `actions_target`：\([B, T, A]\)，当前 chunk 的专家动作监督。
- `feedback_tokens`：\([B, Kf, H]\)，其中 `Kf=causal_feedback_token_num`（默认 1）。

构造过程：

1. `z_before = mean(task_tokens, dim=1)`，`z_after = mean(task_tokens_next, dim=1)`。
2. `delta_z = z_after - z_before`（可配置 `detach`）。
3. `a_ctx = mean(actions_target, dim=1)`（可配置 `detach`）。
4. `h_fb = CausalMLP([delta_z, a_ctx]) -> [B, Kf, H]`。
5. 将 `h_fb` 作为 `feedback_tokens` 透传给 Action Head，拼接到 cross-attention 的 task token 上下文。

### 6.2 代码落点

- `starVLA/model/framework/MapAnythingLlava3DPI.py`
  - 新增 `_build_causal_feedback_tokens(...)`
  - 训练 `forward` 中在进入 `action_model(...)` 前构造并传入 `feedback_tokens`
  - 推理 `predict_action` 支持可选 `feedback_tokens` 外部输入（不传则保持旧行为）
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
  - `forward(...)` / `predict_action(...)` 新增可选参数 `feedback_tokens`
  - 新增 `_normalize_feedback_tokens(...)`
  - cross-attention 条件从 `task_tokens` 扩展为 `cat(feedback_tokens, task_tokens)`（若 feedback 存在）

### 6.3 配置开关（默认关闭，兼容旧训练）

放在 `framework.action_model` 下：

- `enable_causal_feedback_token: false`
- `causal_feedback_token_num: 1`
- `causal_feedback_detach_delta: true`
- `causal_feedback_detach_action: true`
- `causal_feedback_use_valid_mask: true`
- `causal_feedback_scale: 1.0`
- `causal_feedback_dropout: 0.0`
- `feedback_context_scale: 1.0`

### 6.4 新增日志（便于排查）

- `debug/causal_feedback/enabled`
- `debug/causal_feedback/applied`
- `debug/causal_feedback/token_num`
- `debug/causal_feedback/delta_z_norm_mean`
- `debug/causal_feedback/token_norm_mean`
- `debug/sagr/feedback_token_num`
- `debug/sagr/feedback_token_norm_mean`

说明：该通道默认不改变旧行为；只有显式开启 `enable_causal_feedback_token` 后，才会在训练中注入反馈 token。
