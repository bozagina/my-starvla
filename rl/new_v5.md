# 多维离散动作下的结构化 Advantage 设计（开发说明）

> 目标：在 **不改变 joint PPO 目标形式** 的前提下，为多维离散 / VLA 动作引入更加细粒度、结构化的优势函数，用于更好的 credit assignment 与诊断。

---

## 0. 问题设定与基础 PPO 形式

### 0.1 多维离散动作空间

- 状态：\(s_t \in \mathcal{S}\)
- 多维离散动作：
  \[
  a_t = (a_t^1, \dots, a_t^D), \quad a_t^i \in \mathcal{A}_i
  \]
- 联合动作空间：
  \[
  \mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_D
  \]

在 VLA 场景中，\(a_t\) 通常是一组 token（如末端位姿 bin、抓取开合、工具 id 等）的组合。

### 0.2 策略与 joint ratio

- 当前策略：\(\pi_\theta(a\mid s)\)
- 行为策略：\(\mu(a\mid s)\)

标准 importance ratio（joint）：

\[
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}
\]

假设策略是“多 head 因子化”的实现形式：

\[
\log \pi_\theta(a_t\mid s_t)
= \sum_{i=1}^D \log \pi_\theta^{(i)}(a_t^i\mid s_t)
\]

但是 **ratio 始终用 joint 概率比**（不拆维度）。

### 0.3 标准 joint PPO surrogate

给定一个 joint advantage 标量 \(\hat A_t\)（例如用 GAE 得到）：

\[
L^{\text{PPO}}(\theta)
= \mathbb{E}_t\Big[
\min\big(r_t(\theta)\,\hat A_t,\;
         \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t
\big)
\Big]
\]

我们的所有改动，都围绕“如何设计 `Advantage(s,a)`”展开，**保持 joint ratio 与 PPO 形式不变**。

---

## 1. 想法一：共享优势 + 每维局部优势（Shared + Local Advantage）

### 1.1 核心目标

当前工业实现通常是：**所有动作维度共享同一个 advantage 标量** \(\hat A_t\)。

我们希望：

- 保留一个 **共享 / 全局优势**：\(A_t^{\text{shared}}\)，用于表达 joint action 整体好坏；
- 为每个维度 \(i\) 引入一个 **局部残差优势**：\(\Delta A_t^{(i)}\)，用于 finer-grained credit；
- 保证整体 surrogate 在标量上仍然“等价于一个全局 advantage”。

### 1.2 参数化形式

设 advantage head 输出：

- 共享优势（仅依赖状态）：
  \[
  A_t^{\text{shared}} = A_\phi^{\text{shared}}(s_t)
  \]
- 每维 raw 局部残差：
  \[
  \Delta A_{t,\text{raw}}^{(i)}
  = \Delta A_\phi^{(i)}(s_t, a_t^i), \quad i = 1,\dots, D
  \]

**零和约束（中心化 / 去均值）**  
借鉴 BDQ 的“去均值优势”思想，对每个样本的局部残差做中心化：

\[
\Delta A_t^{(i)} 
= \Delta A_{t,\text{raw}}^{(i)} 
  - \frac{1}{D} \sum_{j=1}^D \Delta A_{t,\text{raw}}^{(j)}
\]

于是满足：

\[
\sum_{i=1}^D \Delta A_t^{(i)} = 0
\]

**每个维度的有效优势：**

\[
A_{t,i}^{\text{eff}}
= A_t^{\text{shared}} + \Delta A_t^{(i)}
\]

### 1.3 与 joint PPO surrogate 的关系

定义 per-sample 的 **平均有效优势**：

\[
\bar A_t^{\text{eff}}
:= \frac{1}{D} \sum_{i=1}^D A_{t,i}^{\text{eff}}
= \frac{1}{D} \sum_i \big(A_t^{\text{shared}} + \Delta A_t^{(i)}\big)
= A_t^{\text{shared}}
\]

因此，如果我们用 \(\bar A_t^{\text{eff}}\) 代替 \(\hat A_t\) 放进 PPO：

\[
L^{(1)}(\theta)
= \mathbb{E}_t\Big[
\min\big(r_t(\theta)\,\bar A_t^{\text{eff}},\;
         \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\bar A_t^{\text{eff}}\big)
\Big]
\]

只要训练时让：

\[
A_t^{\text{shared}} \approx \hat A_t
\]

那么标量 surrogate 就 **退化为标准 joint PPO**。  
也就是说：

> **在标量层面，目标函数不变；  
>  在梯度分布层面，每个维度 head 使用不同的 \(A_{t,i}^{\text{eff}}\)，实现 per-dim credit assignment。**

从梯度角度看（忽略 clip）：

\[
\nabla_\theta J(\theta)
\approx \mathbb{E}_t\Big[
r_t(\theta) \sum_{i=1}^D
\nabla_\theta \log \pi_\theta^{(i)}(a_t^i\mid s_t)\, A_{t,i}^{\text{eff}}
\Big]
\]

相比于“所有维度统一乘 \(\hat A_t\)”的实现，多了 per-dim 权重。

### 1.4 Advantage head 的训练目标

假设我们已有标准 GAE target \(\hat A_t\)（标量）：

- 共享优势拟合 \(\hat A_t\)：
  \[
  \mathcal{L}_{\text{shared}}(\phi)
  = \mathbb{E}_t \big(A_t^{\text{shared}} - \hat A_t\big)^2
  \]

- 局部残差的约束（借鉴 BDQ 的分支去均值思想）：

  - 强制零和（数值上已经通过中心化实现，可视作正则）：
    \[
    \Big(\sum_i \Delta A_t^{(i)}\Big)^2 = 0
    \]
  - 控制残差幅度，防止过拟合：
    \[
    \frac{1}{D} \sum_{i=1}^D \big(\Delta A_t^{(i)}\big)^2
    \]

  总体可写成：

  \[
  \mathcal{L}_{\text{local}}(\phi)
  = \lambda_1 \Big(\sum_i \Delta A_t^{(i)}\Big)^2
    + \lambda_2 \frac{1}{D}\sum_i \big(\Delta A_t^{(i)}\big)^2
  \]

- 综合 advantage loss：

  \[
  \mathcal{L}^{(1)}_{\text{adv}}(\phi)
  = \mathcal{L}_{\text{shared}} + \mathcal{L}_{\text{local}}
  \]

### 1.5 借鉴 BDQ 的元素

BDQ（Branching Dueling Q-Network）的结构为：

\[
Q_d(s,a_d)
= V(s) 
  + \Big(A_d(s,a_d) 
  - \frac{1}{n}\sum_{a'_d}A_d(s,a'_d)\Big)
\]

- **Shared component**：\(V(s)\) —— 对应我们的 \(A^{\text{shared}}(s)\)；
- **Branching advantages**：\(A_d(s,a_d)\) —— 对应我们的各维 \(\Delta A^{(i)}(s,a^i)\)；
- **去均值 / 零均值约束**：  
  防止分支优势吸收“全局偏移”，只表达“相对偏好”。

我们直接借鉴了这一思想：

- 用 \(A^{\text{shared}}(s)\) 表征 joint 层面的整体好坏；
- 用中心化的 \(\Delta A^{(i)}(s,a^i)\) 表征不同维度的相对偏好；
- 保证 per-sample \(\sum_i \Delta A^{(i)} = 0\)，从而保持标量 surrogate 与标准 PPO 对齐。

---

## 2. 想法二：结构化优势分解 PPO（Structured Advantage Decomposition PPO）

### 2.1 核心目标

想法一只做了“共享 + per-dim 残差”的 re-weighting，对于动作维度之间的**交互模式**（例如“维度 1 的某种选择需要配合维度 3 的某种选择”）没有显式建模。

想法二进一步将 advantage 分解为：

- 各维度单独的贡献；
- 维度之间的交互项。

### 2.2 数学形式：结构化 Aϕ

定义结构化 advantage：

\[
A_\phi(s,a)
= \sum_{i=1}^D A_\phi^{(i)}(s, a^i)
+ \sum_{i<j} A_\phi^{(ij)}(s, a^i, a^j)
+ \dots
\]

在实践中一般只保留：

- 一阶项：\(A_\phi^{(i)}(s,a^i)\)（per-dim）；
- 二阶项：\(A_\phi^{(ij)}(s,a^i,a^j)\)（pairwise interactions）。

> 直觉：  
> - \(A^{(i)}\)：第 i 维动作 token 自己带来的贡献；  
> - \(A^{(ij)}\)：动作维度 i、j 联合作用的好坏（协同或冲突）。

### 2.3 Advantage head 训练目标

仍然用一个 scalar GAE target \(\hat A_t\) 监督：

\[
\mathcal{L}^{(2)}_{\text{adv}}(\phi)
= \mathbb{E}_t \big(A_\phi(s_t,a_t) - \hat A_t\big)^2
+ \lambda_{\text{pair}}\,\Omega_{\text{pair}}(\{A_\phi^{(ij)}\})
\]

其中 \(\Omega_{\text{pair}}\) 是交互项的正则，例如：

- L2 正则：
  \[
  \Omega_{\text{pair}} = \mathbb{E}_t\left[ \sum_{i<j} \big(A_\phi^{(ij)}(s_t,a_t^i,a_t^j)\big)^2 \right]
  \]
- 或进一步做“去均值”约束，借鉴 BDQ，对交互项进行中心化，使其表达纯交互。

### 2.4 Actor 目标函数

在 PPO surrogate 中直接将黑盒 \(\hat A_t\) 替换为结构化 \(A_\phi(s_t,a_t)\)：

\[
L^{(2)}(\theta)
= \mathbb{E}_t\Big[
\min\big(r_t(\theta)\,A_\phi(s_t,a_t),\;
         \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,A_\phi(s_t,a_t)\big)
\Big]
\]

- ratio 始终是 joint 比率；
- 目标形式仍然是标准的 PPO surrogate，只是 advantage 模块更强、更结构化。

### 2.5 BDQ 的启发点

BDQ 里：

- 使用 **共享 state 价值 + 分支优势**；
- 分支优势做了 **去均值**，保证结构的稳定和可解释性。

在结构化 advantage 中：

- per-dim 项 \(A^{(i)}\) 类似 BDQ 的分支优势；
- 可以把一个额外的 shared term 类比 BDQ 的 \(V(s)\)：  
  \[
  A_\phi(s,a)
  = A^{\text{shared}}(s) 
    + \sum_i \tilde{A}^{(i)}(s,a^i)
    + \sum_{i<j}\tilde{A}^{(ij)}(s,a^i,a^j)
  \]
- 对 \(\tilde{A}^{(i)}, \tilde{A}^{(ij)}\) 做“去均值 / 中心化”或正则，保持 identifiability 和数值稳定：

  - 例如对每个样本，让：
    \[
    \sum_i \tilde{A}^{(i)}(s,a^i) \approx 0,\quad
    \sum_{i<j}\tilde{A}^{(ij)}(s,a^i,a^j) \approx 0
    \]
  - 或者只对 per-dim 项做去均值，把交互项作为纯 residual。

> 这样可以在论文里说：  
> “Our structured advantage head can be seen as a policy-gradient analogue of BDQ: instead of factorising Q-values, we factorise the advantage into shared, per-dimension and interaction terms with de-meaning constraints.”

---

## 3. 两种方案 + BDQ 借鉴点总结（开发视角）

### 3.1 方案 1：Shared + Local Advantage

**形式：**

- 优势分解：
  \[
  A_{t,i}^{\text{eff}}
  = A_t^{\text{shared}} + \Delta A_t^{(i)},\quad
  \sum_i \Delta A_t^{(i)} = 0
  \]
- PPO surrogate：
  \[
  \bar A_t^{\text{eff}} = \frac{1}{D}\sum_i A_{t,i}^{\text{eff}} = A_t^{\text{shared}}
  \]
  \[
  L^{(1)}(\theta)
  = \mathbb{E}_t\Big[
  \min\big(r_t(\theta)\,\bar A_t^{\text{eff}},\;\text{clip}(r_t)\,\bar A_t^{\text{eff}}\big)
  \Big]
  \]

**目标函数本质：**

- 标量 surrogate 与标准 joint PPO 等价（在 \(A^{\text{shared}} \approx \hat A_t\) 的前提下）；
- 实现和梯度分布上，对不同动作维度 head 施加了不同的优势权重。

**BDQ 借鉴：**

- Shared component + Branching residuals；
- 对各维 residual 做去均值（零和），防止结构不稳定；
- Advantage loss 里可加分支级正则。

---

### 3.2 方案 2：结构化优势分解 PPO

**形式：**

- 结构化优势：
  \[
  A_\phi(s,a)
  = \sum_i A_\phi^{(i)}(s,a^i)
  + \sum_{i<j} A_\phi^{(ij)}(s,a^i,a^j)
  \]
- Advantage loss：
  \[
  \mathcal{L}^{(2)}_{\text{adv}}(\phi)
  = \big(A_\phi(s_t,a_t) - \hat A_t\big)^2
  + \lambda_{\text{pair}}\,\Omega_{\text{pair}}(\{A^{(ij)}\})
  \]
- PPO surrogate：
  \[
  L^{(2)}(\theta)
  = \mathbb{E}_t\Big[
  \min\big(r_t(\theta)\,A_\phi(s_t,a_t),\;
           \text{clip}(r_t)\,A_\phi(s_t,a_t)\big)
  \Big]
  \]

**目标函数本质：**

- 从形式上仍是 standard PPO，只是 advantage approximation 从“黑盒 scalar”变成了“结构化分解”；
- 多了 per-dim、pairwise 可视化与 diagnostics（可以画 heatmap，看哪些维 / 维对贡献最大）。

**BDQ 借鉴：**

- 将 BDQ 的“共享 V(s) + 分支优势”转译为“共享 A(s) + per-dim advantage”；
- 可以对 per-dim 项做去均值，借鉴 BDQ 的 identifiability trick；
- 可以对 pairwise 项做中心化/正则，避免将整体趋势泄漏到交互项。

---

## 4. 开发建议（下一步代码实现）

1. **先实现方案 1（Shared + Local Advantage）作为 MVP：**
   - 新增 `AdvantageHeadSharedLocal` 模块：
     - 输出 `A_shared: [B]`；
     - 输出 `delta_raw: [B, D]`，内部做中心化得到 `delta: [B, D]`。
   - Actor loss：
     - 构造 `A_eff = A_shared.detach().unsqueeze(-1) + delta.detach()`；
     - 取 `A_eff_mean = A_eff.mean(dim=-1)` 作为 surrogate A；
     - 按标准 PPO 写 `ratio * A_eff_mean` + clip。
   - Critic/advantage loss：
     - `MSE(A_shared, A_GAE)` + `lambda * (delta^2 + zero_mean^2)`。

2. **验证方案 1 的基本诊断：**
   - 记录并可视化：
     - `delta` 各维平均值（应该在 0 附近）；
     - `delta` 方差、max/min（看哪些维有更大 local credit）。
   - 看是否在训练 early-stage 就能区分“关键维度”。

3. **在此基础上扩展方案 2：**
   - 在 `AdvantageHead` 里加入 pairwise 项：
     - per-dim embedding + 低秩双线性交互，输出 `pair_scores[i,j]`；
     - mask 上三角求和得到 `pair_sum`；
     - `A_total = A_per_dim_sum + pair_sum`。
   - 用 `A_total` 替代 \(\hat A_t\) 进入 PPO surrogate；
   - 对 `pair_scores` 做 L2 正则 / 中心化。

这样你就可以一边推进代码，一边用这套文档作为“方法章节骨架”和“实现约束说明”。
