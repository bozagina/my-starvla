# SA-GR Flow：基于几何残差引导的随机动力学 VLA 闭环进化方案

## 1. 背景与科学动机

在机器人操作中，传统的模仿学习（Imitation Learning, IL）模型在面对未见过的物体位移（OOD）时极度脆弱。其本质原因是模型缺乏 **“物理因果律”**：它往往只学会“看到什么就做什么”，却没有形成“做完动作后世界应该如何变化”的可验证预期。

**SA-GR Flow** 通过引入一个 **世界预测器（World Predictor）** 来构建这种因果律：模型不仅学习“该做什么动作”，还预测“做完动作后世界应该变成什么样”。当现实与预测不符时，产生的 **几何残差** $\Delta z$ 将作为反馈信号，通过“小脑反射”机制即时修正动作，从而形成闭环的物理一致性控制。

---

## 2. 核心算法架构解析

我们将系统的随机动力学定义为受控随机微分方程（Controlled SDE）：

$$
da_t
=
\underbrace{\left[
v_\theta(s,t)
+
b_\psi(\Delta z_t, s)
+
\eta_\phi(s)
\right]}_{f_{\text{total}}}
dt
+
\sigma(t)\,dw_t
$$

- $v_\theta(s,t)$：先验/主干 drift（可理解为“既有能力”或“流的主向量场”）
- $b_\psi(\Delta z_t, s)$：几何反射项（小脑），由残差驱动的即时纠偏
- $\eta_\phi(s)$：策略项（大脑），通过 RL 学到的长期最优修正
- $\sigma(t)dw_t$：随机扰动项（探索/随机动力学）

---

## 2.1 模块定义与实现细节

### (A) 世界预测器（World Predictor）$\hat{\Phi}$

**输入：**
- 当前任务相关 3D Tokens：$S_t \in \mathbb{R}^{N \times D}$
- 当前执行动作：$a_t$

**输出：**
- 预测下一帧 Token 的位移：$\Delta \hat{S}_{t+1}$

**实现建议：**
- 采用 **Residual Transformer**
- 将动作 $a_t$ 经 MLP 映射到 token 维度后，拼接/注入到每个 Token 上
- 通过 Transformer Block 输出每个 Token 的 3D 位移矢量

**形式化：**
$$
\hat{S}_{t+1} = S_t + \hat{\Phi}(S_t, a_t)
$$

---

### (B) 几何残差（Geometric Residual）$\Delta z$

**来源：**
- Rollout 过程中执行动作 $a_t$ 后观测到真实状态 $S_{\text{actual}}$

**定义：**
$$
\Delta z_t = S_{\text{actual}} - \hat{S}_{\text{predicted}}
$$

**物理意义：**
- $\Delta z$ 是“意外”的数学表达
- 若物体被推开/滑移，$\Delta z$ 会在对应 Token 区域出现明显的矢量偏差

---

### (C) 几何反射项（Geometric Reflex）$b_\psi$ —— “小脑”

**实现：**
- Cross-Attention 映射层

**核心逻辑：**
- 使用一个可学习的 **Action-Query** 去检索 $\Delta z$ 的空间结构
- 学习一种通用的物理映射：例如感知到物体向左偏 $5\text{cm}$，对应动作修正应包含向左平移/反向补偿

---

## 3. 损失计算与训练阶段

为保证系统稳定性，采用 **由本能到策略（Reflex to Strategy）** 的二阶段训练范式。

---

### 阶段 1：自监督“小脑”预热（Reflex Warmup）

在不引入 RL 奖励的情况下，仅利用 IL 数据集训练 $b_\psi$ 的纠偏能力。

**Hindsight Perturbation 损失** $\mathcal{L}_{\text{reflex}}$：

1. 从数据集中取真实样本 $(S_t, a_t^*, S_{t+1})$
2. 对动作注入扰动：$a'_t = a_t^* + \epsilon,\ \epsilon \sim \mathcal{N}(0,\nu)$
3. 世界预测器在扰动动作下推演：
   $$
   \hat{S}_{t+1} = \hat{\Phi}(S_t, a'_t)
   $$
4. 计算伪残差：
   $$
   \Delta z = S_{t+1} - \hat{S}_{t+1}
   $$
5. 目标：$b_\psi(\Delta z)$ 学习输出 $-\epsilon$ 抵消扰动

$$
\mathcal{L}_{\text{reflex}}
=
\mathbb{E}_\epsilon \left\| b_\psi(\Delta z) + \epsilon \right\|^2
$$

---

### 阶段 2：分布式 PPO 联合微调（Online RL）

在分布式框架（如 `ds_libero_ppo.py`）中进行在线强化学习。

**奖励重构：**
$$
R_t
=
\alpha \cdot r_{\text{task}}
-
\beta \|\Delta z_t\|^2
-
\gamma \|b_\psi(\Delta z_t)\|^2
$$

- $\|\Delta z_t\|^2$：惩罚预测失效，强迫模型理解物理
- $\|b_\psi(\Delta z_t)\|^2$：防止反射修正过大导致动作畸变

**PPO 策略损失** $\mathcal{L}_{\text{CLIP}}$：
- 使用 $f_{\text{total}}=v_\theta + b_\psi + \eta_\phi$ 作为动作分布的均值来计算 log-probability 并执行 PPO 更新

---

## 4. 算法伪代码（Pseudocode）

```python
# SA-GR Flow 训练循环
# 初始化: v_prior (冻结), b_reflex, world_predictor, eta_strategy, critic

# 阶段 1: 小脑自监督预热 (利用 IL 数据)
for batch in IL_Dataset:
    S_t, a_true, S_next = batch
    epsilon = sample_noise()
    a_perturbed = a_true + epsilon
    
    # 物理推演
    S_hat_next = world_predictor(S_t, a_perturbed)
    delta_z = S_next - S_hat_next  # 因扰动产生的残差
    
    # 学习纠偏本能
    loss_reflex = MSE(b_reflex(delta_z), -epsilon)
    update(b_reflex, loss_reflex)

# 阶段 2: PPO 在线微调
for iter in range(TRAIN_ITERS):
    # 分布式 Rollout (基于 SDE 采样)
    for step in env_steps:
        # 计算当前物理残差 (基于前一帧预测)
        delta_z = obs_actual - last_prediction
        
        # 合成 Drift 并采样
        mu = v_prior(s) + b_reflex(delta_z) + eta_strategy(s)
        action = mu + sigma * dw
        
        # 记录下一次预测
        last_prediction = world_predictor(obs_actual, action)
        
        # 记录 Experience 到 ReplayBuffer
        buffer.add(obs, action, reward, delta_z)
    
    # 异步训练周期
    for mini_batch in buffer:
        # 更新世界预测器 (练眼)
        loss_dyn = MSE(world_predictor(s, a), s_next)
        
        # PPO 更新 (练脑)
        # 计算优势 Adv, 并优化 eta_strategy 与 b_reflex
        loss_ppo = clipped_ppo_loss(mu_new, mu_old, Adv)
        update(eta_strategy, b_reflex, world_predictor, loss_ppo + loss_dyn)