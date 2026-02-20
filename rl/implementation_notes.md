# Success-as-a-Set Contrastive Bonus Phase (MLP PPO + V-trace + Replay)
**实现阶段：Part1–Part4（集成版）**  
**目标文件：** `ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py`

> 你现在的 Part4 文件是 **Part1+Part2+Part3+Part4 的集成版**：  
> - Part1：episode 元信息贯通 + replay 侧拼回完整 episode  
> - Part2：Paired-Queue（fail→Top-K success）+ divergence window + PairPack  
> - Part3：Bonus Time（加时赛）+ Loss A（success-as-a-set ranking）+ KL/Entropy guard（分布式同步）  
> - Part4：Success prototypes（在线聚类/原型）+ 跨簇采样 + 触发条件加入簇覆盖（防 mode collapse）

---

## 1. 目的与设计原则

### 1.1 目的（Why）
在“大 replay + off-policy 校正（V-trace）”框架下，**旧数据可复用**，但 replay 里 **成功/失败、新/旧** 被均匀对待会浪费结构信号。  
本方案不走 PER/改采样概率路线，而是引入 **新的、可控的梯度信号**：

- 把失败轨迹与相似成功轨迹进行结构化配对（Paired-Queue）
- 把成功当作 **集合/多模态混合分布**（success-as-a-set）
- 在主 PPO 更新之外，引入 Bonus Phase 进行额外小步更新（加时赛），用 Loss A 直接塑形 policy

### 1.2 设计原则（Engineering）
- **最小侵入**：不改 PPO+V-trace 主干张量路径，Bonus 是旁路。
- **异步隔离**：pair mining 在 replay actor 内做，失败只会 skip，不影响主训练。
- **稳定性护栏**：Bonus 必须有 KL/Entropy guard + 全局同步触发/早停，避免多卡死锁与崩溃。
- **逐步可升级**：v0（Top-K）→ v1（prototypes）→ v2（Φ shaping）。

---

## 2. 我们想要实现的内容（当前实现到哪里）

### 2.1 目标算法（你定义的核心）
在 PPO+V-trace + replay 上新增：
1) **Paired-Queue**：维护“失败 episode ↔ 成功集合”的配对样本（片段/窗口）
2) **Success-as-a-Set**：正例来自 **多条成功参照**，并在 v1 引入 **成功簇覆盖**
3) **Bonus Time**：当队列足够大且覆盖多模态时触发加时赛训练
4) **Loss A**（主线）：在 divergence window 内做 success-set vs fail-action 的 ranking（坏动作消除）

### 2.2 Part4 当前实现的状态
- ✅ 已实现 Loss A（success-as-a-set ranking，log-sum-exp positives）
- ✅ 已实现 Paired-Queue + pair mining（fail→Top-K success）
- ✅ 已实现 divergence window（共享前缀后首次偏离/最小相似度点）
- ✅ 已实现 Bonus Time 调度（全局同步触发/早停，KL/Entropy guard）
- ✅ 已实现 prototypes（在线 EMA 原型）+ cluster_id + 跨簇采样（v1）
- ❌ 尚未实现（后续）：DTW/soft alignment、更强的 embedding（MLP中间层）、Φ shaping（Loss B 接入 V-trace return）

### 2.3 最新改动（`ds_metaworld_ppo_mlp_add_vtrace_try_new_v2_with_param.py` 相对 `..._with_param.py`）
- Bonus 触发/批量参数可用环境变量调节：`BONUS_BATCH_PACKS`、`BONUS_TRIGGER_MIN_PAIRS`（默认 max(32, 2*batch)）、`BONUS_TRIGGER_MIN_CLUSTERS`、`BONUS_MAX_STEPS`。
- 新增 consume 开关与冷却：`BONUS_CONSUME_PACKS`（是否消费 paired pack）、`BONUS_COOLDOWN_STEPS`（默认 5000 步），`last_bonus_step` 记录上次触发步。
- Bonus 未触发原因日志：`bonus/not_triggered_reason_code`（1=pair 少，2=cluster 覆盖不足，3=无 pack，4=cooldown）及 `bonus/not_triggered_pairs`、`bonus/not_triggered_cov`、`bonus/not_triggered_cooldown`。
- Bonus 触发记录：`bonus/triggered` 与 `bonus/ran` 一致写出，触发后更新 `last_bonus_step`。
- Paired-Queue 取样支持 consume 开关：`sample_paired_packs(..., BONUS_CONSUME_PACKS, ...)`。
- EMA 最优策略广播：Trainer0 周期性导出 `get_ema_state_dict`（CPU 权重）并调用 InferenceActor `load_best_weights`，用于 best/perturbed_best rollout 冷启动提效。

---

## 3. 算法解释（公式 + 训练步骤）

### 3.1 Paired-Queue（fail→success-as-a-set）
当一个 episode 结束（done=True）：
1) 若 success：写入 success pool，并更新 success prototypes（Part4）
2) 若 failure：在 success pool 中按 embedding cosine 检索 Top-K success
3) 用 Top-1 success 定位 divergence 时刻 `t_d`（v0 简化）
4) 取 divergence window：`[t_d-w, t_d+w]`
5) 写入 PairPack：包含 `states`、`fail_actions`、`succ_actions[K]`、`zone_weight`、`succ_cluster_ids`

> **success-as-a-set 的关键**：正例不是一条成功轨迹，而是 `K` 条成功参照的动作集合。

---

### 3.2 Loss A：Success-as-a-Set Ranking（bad-action elimination）
在 divergence window 的每个状态 `s_t`：
- 负例：失败动作 `a^-`
- 正例集合：来自 K 条成功参照/多簇覆盖的动作集合 `{a^+}`

对 Multi-Discrete 动作（`ACTION_DIM` 个离散 token），实现为“维度内集合 soft 聚合”：

- 负例 logprob：
\[
\log p^{-}(s)=\sum_{d}\log \pi_d(a^{-}_d|s)
\]

- 正例集合 logprob（log-sum-exp over positives）：
\[
\log p^{+}(s)=\sum_{d}\log\sum_{a\in Pos_d(s)}\exp(\log \pi_d(a|s))
\]

- 排名型 loss（带温度 \(\tau\) 与窗权重 \(w_t\)）：
\[
L_A = -\, w_t\cdot \log \sigma\Big(\frac{\log p^{+}(s_t)-\log p^{-}(s_t)}{\tau}\Big)
\]

其中 `Pos_d(s)` 由 Top-K 成功参照在该时刻的 token 集合构造，并在 Part4 里尽量跨簇覆盖。

---

### 3.3 Bonus Time（加时赛）调度与护栏
触发条件（Part4）：
- `paired_queue.size >= BONUS_TRIGGER_MIN_PAIRS`
- `paired_queue.cluster_coverage >= BONUS_TRIGGER_MIN_CLUSTERS`
- 并做 **all-reduce(MIN)** 保证所有 rank 同步触发（避免死锁）

执行：
- 连续做 `BONUS_UPDATES` 次 bonus 更新（固定次数：多卡一致）
- 每次从 Paired-Queue 取 packs，展开为 step batch，计算 Loss A
- Guard：
  - **KL-guard**：KL 大于阈值 early stop
  - **Entropy floor**：entropy 小于阈值 early stop
  - early stop 同样全局一致

---

## 4. 代码结构与关键函数（按 Part1→Part4，对齐函数名与行号）

> 行号以 `ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py` 为准（可能随你后续编辑略有偏移）。

---

## Part1：episode 元信息贯通（为配对/bonus 提供 episode 边界）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:Experience (L151)]
- 新增字段：
  - `episode_id: int = -1`
  - `t: int = -1`
- 目的：replay 拼回 episode、对齐 step、定位 divergence window

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:BaseWorkerActor._reset_and_select_env (L844)]
- reset 时分配 `self.current_episode_id`  
- 目的：跨 worker 唯一的 episode 标识

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:RolloutWorkerActor.run (L882)]
- local_buffer 记录追加 `t_idx, episode_id`
- done 时提取 `info['success']` 与 `episode_return`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:RolloutWorkerActor._process_traj (L941)]
- 将 `episode_id/t` 写入每个 `Experience`
- 将 `(episode_id, episode_success, episode_return)` 传给 replay 的 `add_trajectory`

---

## Part2：Paired-Queue（失败-成功配对、divergence window、PairPack）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:PairPack (L167)]
- PairPack 字段：
  - `fail_ep_id`
  - `succ_ep_ids`
  - `succ_cluster_ids`（Part4 加入）
  - `divergence_t`, `window_start`, `window_end`
  - `states`, `fail_actions`, `succ_actions`, `zone_weight`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor.__init__ (L306)]
- 维护：
  - 训练用 buffer（不改主训练采样）
  - episode fragment 拼接：`_episode_frags`
  - episode store：`_episode_traj/_episode_emb/_episode_meta`
  - success/failure pools：`_success_eps/_failure_eps`
  - paired queue：`_paired_queue`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor.add_trajectory (L334)]
- **训练用**：`self.buffer.append((traj, done, last_obs))`（保持不变）
- **bonus 用**：按 episode_id 拼回完整 episode（done=True finalize）
- failure finalize 时调用 `_mine_pairs_for_failure(...)`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor._mine_pairs_for_failure (L556)]
- failure → success pool cosine 检索 Top-K
- divergence 定位：阈值跌落/最小相似
- window 切片，构造 `succ_actions[K,T,ACTION_DIM]`
- `zone_weight`（三角窗）
- 写入 `_paired_queue`
- Part4：优先跨簇覆盖挑选 Top-K success（避免同簇 Top-K）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor.sample_paired_packs (L473)]
- Bonus 取样入口：
  - 支持 `consume=True`（避免过拟合）
  - 支持 `diverse_clusters=True`（Part4：多簇优先）

---

## Part3：Bonus Time + Loss A + Guard（分布式同步）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:TrainerActor._should_run_bonus_globally (L1436)]
- 本地检查 paired_size/cluster_coverage
- all-reduce(MIN) 保证所有 rank 一致触发

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:TrainerActor._build_bonus_batch (L1450)]
- 将 PairPack 展开成 step batch：
  - `states [N, obs_dim]`
  - `fail_actions [N, ACTION_DIM]`
  - `pos_mask [N, ACTION_DIM, N_BINS]`（success-as-a-set）
  - `zone_w [N]`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:TrainerActor._lossA_set_ranking (L1508)]
- 实现 Loss A：
  - `logp_neg`：失败动作 logprob
  - `logp_pos`：正例集合 log-sum-exp 聚合
  - ranking loss（logistic / optional margin）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:TrainerActor._run_bonus_time_if_ready (L1558)]
- Bonus 主循环：
  - 固定 `BONUS_UPDATES` 次（多卡一致）
  - KL/Entropy guard + early stop

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:TrainerActor.run_training_epoch (L1633)]
- 在主 PPO 更新后插入：
  - `bonus_metrics = self._run_bonus_time_if_ready()`
- bonus 出错不会影响主训练（建议 try/except 保持）

---

## Part4：Success prototypes（在线聚类）+ 跨簇采样（防 mode collapse）

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor._ensure_prototypes (L404)]
- 初始化 prototypes / counts 的存储结构

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor._assign_success_cluster (L408)]
- success episode finalize 时：
  - init：前 K 个 success 填满 prototypes
  - 之后：分配最近原型（cosine）+ EMA 更新 prototype
- 将 `cluster_id` 写入 `self._episode_meta[ep]['cluster_id']`

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor.paired_cluster_coverage (L447)]
- 统计 Paired-Queue 中 `succ_cluster_ids` 覆盖簇数
- 作为 Bonus 触发条件之一

### [ds_metaworld_ppo_mlp_add_vatrace_try_new_part4.py:ReplayBufferActor.sample_paired_packs (L473)]
- `diverse_clusters=True` 时：
  - 选 pack 时优先覆盖更多 cluster（v1 的最小实现）

---

## 5. 注意事项与常见坑

### 5.1 默认阈值很大（训练 vs 单测）
默认超参（如 `PAIR_MIN_SUCCESS_POOL=32`, `BONUS_TRIGGER_MIN_PAIRS=256`）适合真实训练，不适合快速验证。  
测试/调试时建议临时调小（见测试脚本）。

### 5.2 分布式同步（最危险）
- Bonus 的触发与 early stop 必须全局一致，否则会出现 barrier 死锁或参数不同步。
- Part4 已通过 all-reduce(MIN) 做全局 gating；你后续任何新增条件都要保持同样风格。

### 5.3 mode collapse 风险与对策
- v0 仅 Top-K 不保证多模态覆盖（Top-K 可能都来自同簇）
- v1（Part4）通过 cluster/prototypes 与 diverse sampling 做最小防护
- 仍建议监控 entropy 与成功多样性指标

---

## 6. 下一步路线（从 Part4 往 v2）
- v2：加入 **Φ shaping（Loss B）** 并接入 V-trace return：
  - Φ(s) 多峰 log-sum-exp（基于 prototypes 距离）
  - stop-gradient 与 β 限制，防 reward hacking
- embedding 升级：
  - 用 policy 中间层/MLP embedding 替换 “mean(obs)” 的 episode embedding
- alignment 升级：
  - 加简化 DTW/soft alignment 或相似度滑窗对齐，提高 divergence window 可靠性

## 7. 最少必须做的 Ablation（建议你按 v0/v1 做）

- 强烈建议这几组（成本低、说服力强）：

 - Baseline：PPO + V-trace + Replay（无 bonus）

 - + PairedQueue 但无 LossA（只挖掘不训练）→ 证明提升来自“loss 注入”，不是数据结构

 - + LossA 但 Top-1 success（无 set）→ 对比 Top-K set，证明 set 的必要性

 - + LossA Top-K set（v0）

 - + Prototypes + 跨簇采样（v1 / Part4） → 证明防 collapse、多模态覆盖

 - 如果再加一组：

 - PER / prioritized sampling（只改采样概率）作为“相关但不同”的对照：
 - 你能说清楚：我们不是在“多抽某些样本”，而是在额外的梯度目标里注入“成功集合对比信号”。