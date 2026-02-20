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
