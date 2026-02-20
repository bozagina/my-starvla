# 异步分布式强化学习框架代码审查

## 一、整体架构总览

### 1.1 系统组件角色

在连续版和离散版两个脚本中，整体架构基本一致，仅在动作空间和损失函数设计上存在差异。下面介绍各核心组件：

#### **TrainerActor**（训练器，Ray Actor + DeepSpeed）
- 多 GPU 上的策略/价值网络训练核心
- 使用 DeepSpeed ZeRO-2 进行数据并行与通信优化
- 通过异步后台协程 `_data_fetching_loop` 持续从 ReplayBuffer 拉取大批量数据，与前台 PPO 更新循环完全解耦

#### **InferenceActor**（推理器，Ray Actor + 单/多 GPU）
- 常驻在指定 GPU 上，仅负责前向推理，不参与反向传播
- 持有 ActorCritic 模型的一份副本
- 接收来自多个 RolloutWorker 的推理请求，通过异步队列与批处理实现高吞吐推理服务

#### **RolloutWorkerActor**（采样 Worker）
- 不加载完整大模型，仅持有 Processor 和环境实例
- 调用 InferenceActor 获取动作/logits/value，与环境交互、积累轨迹
- 在本地计算 GAE（广义优势估计）与回报，打包成 Experience 写入 ReplayBuffer

#### **ReplayBufferActor**（经验缓冲池）
- 轻量的远程 FIFO/随机采样缓冲区
- 接收来自众多 RolloutWorkers 的 `add_batch()` 调用
- 为 TrainerActor 提供 `sample(super_batch_size)` 采样接口

#### **StatsActor / EvaluationWorkerActor**
- **StatsActor**：聚合各环境的平均回报、轨迹长度、成功率等关键统计信息
- **EvaluationWorkerActor**（仅在离散版中）：使用当前策略进行评估 Rollout，不向 ReplayBuffer 写数据，用于在线性能评估

#### **ds_com 通信模块**
- `TrainerActorCom` / `InferenceActorCom` 抽象了训练器 → 推理器的权重广播机制
- 封装 `torch.distributed` 进程组初始化与 Broadcast 操作
- 支持 ZeRO-2 下的参数聚合后再广播

### 1.2 系统拓扑与数据流（逻辑时序）

整体拓扑的关键步骤如下：

1. **TrainerActor** 在多 GPU 上用 DeepSpeed 初始化训练进程组
2. **InferenceActor** 在独立 GPU 上加载同款 ActorCritic 模型
3. **TrainerActor** 周期性将最新权重通过 `TrainerActorCom.broadcast_weights` → NCCL 组 → `InferenceActorCom` 接收并更新
4. 大量 **RolloutWorkerActor** 持续调用 InferenceActor 的异步推理接口，从环境采样，写入 ReplayBuffer
5. **TrainerActor** 后台协程从 ReplayBuffer 拉取超级批次，前台协程执行多步梯度更新
6. **StatsActor / EvaluationWorkerActor** 异步记录指标、运行评估

**核心亮点**：Rollout、推理、训练三者通过 Ray + asyncio 完全解耦，在时间轴上高度重叠，实现了 GPU 的高利用率和典型 "异步式" 分布训练范式。

---

## 二、连续动作版：ds_libero_ppo.py 代码解构

### 2.1 系统规模 & 超参数设计

脚本开头定义了关键的超参数配置：

```python
NUM_TRAINER_GPUS = 4
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 40
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 8
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 1000
TRAIN_BATCH_SIZE = 20
ACCUMULATION_STEPS = 13
SUPER_BATCH_SIZE = 260
LOG_INTERVAL_SECONDS = 10
```

**资源利用设计意图**：

- **计算资源配置**
  - 4 张训练 GPU + 1 个推理 Actor + 40 个 rollout worker
  - 训练 GPU 通过 DeepSpeed ZeRO-2 进行数据并行
  - 推理与训练可分配到不同 GPU（如 `CUDA_VISIBLE_DEVICES="3,4,5,6,7"`），避免资源竞争
  - 40 个 rollout worker 充分打满 InferenceActor 的请求队列，隐藏推理/环境延迟

- **推理批处理机制**
  - InferenceActor 以 `INFERENCE_BATCH=8` 为单位进行批量前向传播
  - `INFERENCE_TIMEOUT_MS` 机制保证延迟与吞吐的平衡：短时间内请求不足 batch size 时，超时后强制执行小批次推理

- **训练梯度累计策略**
  - Trainer 从 ReplayBuffer 取 `SUPER_BATCH_SIZE=260` 轨迹样本
  - 细分为 `TRAIN_BATCH_SIZE=20` 的小批，配合 `ACCUMULATION_STEPS=13` 做梯度累计
  - 显存里仅放小 batch，逻辑上做大 batch 优化，兼顾稳定性与资源占用

- **系统优化**
  - `TMPDIR="/dev/shm"`：中间文件放内存盘，减轻 I/O 开销
  - `USE_BF16 + DeepSpeed ZeRO-2`：减显存与通信带宽

### 2.2 Experience & ReplayBufferActor

**连续版 Experience 数据结构**：

```python
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]      # CPU 端的观测字典
    action: np.ndarray                # 标准化动作（tanh 后，[-1, 1]）
    advantage: float                  # GAE 计算的优势估计
    behaviour_mu: np.ndarray          # 行为策略均值
    behaviour_log_std: np.ndarray     # 行为策略 log_std
    behaviour_value: float            # 行为策略价值估计 V(s)
```

**关键特性**：
- 观测 `obs` 已通过 `prepare_one_obs` 预处理为 CPU Tensor 字典，便于后续批量 pad/stack
- 动作信息 `(action, mu, log_std)` 存储完整，训练时可完全重建行为策略 log prob，用于 PPO ratio 计算

**ReplayBufferActor 设计**：
- 内部采用 `deque(maxlen=REPLAY_CAPACITY)` 实现
- `add_batch(batch)`：将 Experience 批次追加到缓冲池
- `sample(batch_size)`：随机采样并返回 `(obs_list, act, adv, mu_old, log_std_old, v_old)`
- 非阻塞设计：单线程 Ray actor 天然串行执行，但各 worker 异步发起远程调用，对上层是"异步管道的一环"

### 2.3 RolloutWorkerActor：环境交互 + GAE 计算

**初始化阶段**：
- 使用 `LiberoEnvWrapper` 包装 LIBERO 任务环境
- 每个 worker 持有一个 Processor（来自 OpenVLA），但不加载完整模型，显著减轻显存压力

**主循环 run() 的核心步骤**：

1. 构造单步输入：`prepare_one_obs + step_count`
2. 调用 `self.infer.request.remote(inputs_t)` → 同步等待获得：
   - `action_env`（已 unnormalize，可直接喂入环境）
   - `action_norm`（归一化动作，用于训练）
   - `mu, log_std, value`
3. 与环境交互，按 chunk 收集 `(obs, action_norm, reward_scaled, mu, log_std, value)` 进入 `self.local_buffer`

**缓冲区触发条件**：
- Episode 结束 → `_process_traj(self.local_buffer, bootstrap_val=0.0)` 清空缓冲
- Buffer 长度达到 `ROLLOUT_LOCAL_BUF + 1` → 用最后一个 value 做 bootstrap，执行 GAE 回传

**GAE 计算 _process_traj**：

逆序遍历轨迹，计算广义优势估计：

$$\delta_t = r_t + \gamma V_{t+1} - V_t$$
$$\text{GAE}_t = \delta_t + \gamma \lambda \text{GAE}_{t+1}$$

- 将优势 `adv` 存入 Experience；value 作为 `behaviour_value` 用于构造 value loss
- 通过 `self.replay.add_batch.remote(batch)` 异步写入 ReplayBufferActor

**设计亮点**：优势估计完全在 rollout 侧本地完成，Trainer 直接消费优势，减轻训练端计算负担，实现计算前移。

### 2.4 InferenceActor：异步批处理推理

**初始化**：
- 在 GPU 上加载 ActorCritic（连续版本）
- 配置 `batch_size = INFERENCE_BATCH` 和 `timeout_sec = INFERENCE_TIMEOUT_MS / 1000`
- 建立两个数据结构：
  - `self.requests: List[inputs_dict]`
  - `self.promises: List[asyncio.Future]`
- 创建后台任务：

```python
self._bg_task = loop.create_task(self._loop())
self._bg_task.add_done_callback(self._on_bg_task_done)
```

**前端接口 request()**：
- 为每个推理请求创建 `asyncio.Future` 放入 `promises`，inputs 放入 `requests`
- 直接返回 future，调用方可决定 `ray.get` 的等待时机
- RolloutWorker 调用 `infer.request.remote(...)`，等价于"RPC + Future"

**后台循环 _loop()（异步批处理核心）**：

持续监控，当满足条件时：
- `len(requests) >= batch_size` 或 `当前时间 - last_process_time > timeout_sec`

执行以下操作：

1. 弹出当前 requests 和 promises，形成一个批次
2. 用 `prepare_inputs_batch` 堆叠所有 inputs 为大 batch
3. GPU 前向传播，获得：标准化动作、mu、log_std、values
4. 标准化动作 clip 到 [-1,1]，再用 `vla._unnormalize_actions` 映射回环境动作
5. 将每条样本 `(action_env[i], actions_norm[i], mu[i], log_std[i], values[i])` 通过 `promise.set_result(...)` 返回

错误处理：捕获前向异常，打印堆栈，对所有未完成的 promises 调用 `set_exception`，避免上游死等。

**设计亮点**：
- 多个 rollout worker 的请求统一打包，最大化 GPU 利用率
- Timeout 机制在吞吐和延迟间取平衡
- 前后端用 Future 解耦，RolloutWorker 视角上只看到远端"推理服务"

### 2.5 TrainerActor：DeepSpeed + 异步数据加载 + PPO 训练

#### (1) DeepSpeed 初始化与轻量化微调

通过 `deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)`：

- model 是 ActorCritic，内部使用 LoRA 微调 OpenVLA 主干
- 仅少量参数可训练，显著减少显存占用
- 优化器参数分为 policy 和 value 两个 param group，支持不同的学习率调度
- DeepSpeed ZeRO-2 配置启用：
  - `overlap_comm=True`：通信与计算重叠
  - `reduce_scatter=True`：降低通信开销
- 脚本打印总参数量与可训练参数量，强调"主体参数冻结 + 少量可训练参数"的轻量特性

#### (2) 异步数据准备 _data_fetching_loop

启动时：`self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())`

_data_fetching_loop 逻辑：

1. 若 `self.next_ready_batch` 已有数据，则 `await asyncio.sleep(0.1)` 避免超前取数
2. 调用 `await self.replay_buffer.size.remote()` 检查 ReplayBuffer 大小
3. 不足 `SUPER_BATCH_SIZE` 则睡眠 3 秒重试
4. 数据足够时调用 `await self.replay_buffer.sample.remote(self.super_batch_size)` 拉取超级批次
5. 用 `base_model.prepare_inputs_batch(obs_list)` 做批量 pad/stack
6. 将 numpy 数据转换为 `torch.Tensor`，迁移到 `self.model` 的 device
7. 将 `(inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_old_t)` 塞进 `self.next_ready_batch`

**双缓冲管线**：
- 前台 `run_training_epoch` 消费 `current_batch` 时
- 后台 `_data_fetching_loop` 并行准备下一个 super batch

#### (3) PPO 训练逻辑 run_training_epoch

核心步骤：

1. 等待 `next_ready_batch` 准备完毕（首个周期会阻塞一次）
2. 拉出 `current_batch`，同时清空 `next_ready_batch` 让后台继续 fetch
3. 根据 `adv_t + v_old_t` 构造 `v_targ_t`，全局 all-reduce 标准化优势：
   - 本机求 `local_sum, local_sq_sum, local_count`
   - `distributed.all_reduce` 聚合到所有 trainer rank
   - 算出 `global_mean, global_std`，标准化优势
4. 将超级批次按 `TRAIN_BATCH_SIZE` 拆成小 batch，循环执行：
   - 前向得到新策略分布和 value
   - 计算 PPO 损失：policy loss + value loss + entropy regularization + KL penalty
   - 调用 `self.model.backward(loss) + 梯度裁剪 + self.model.step()`
   - 使用自定义 `_get_current_lr` 做 warmup + cosine decay，分别调整 policy、value 学习率

**设计特点**：训练过程是 synchronous data-parallel（all-reduce 梯度聚合），但与 ReplayBuffer / Rollout / 推理之间是异步解耦。

### 2.6 main()：训练循环与权重广播

主函数大致流程：

1. Ray 初始化，启动 TrainerActor、InferenceActor、ReplayBufferActor、RolloutWorkerActor、StatsActor 等
2. 训练 group 内使用 DeepSpeed / torch.distributed 进程组；推理广播组使用 ds_com 创建的独立 group
3. 通过 `get_broadcast_signature()` 比较两侧参数/缓冲区签名，确保结构一致
4. 初次广播：
   - `trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)`
   - `InferenceActor.receive_and_update_weights.remote(BROADCAST_GROUP_NAME)`
5. 启动 RolloutWorkers 和 EvaluationWorkers 的 `run.remote()`，后台无限循环
6. 等待 ReplayBuffer 填满一定量数据（warmup 阶段）
7. **主训练循环**：
   - 并行发起所有 trainer 的 `run_training_epoch.remote()`
   - `ray.get` 等待 epoch 完成，获得 loss / 统计信息 / global_step
   - 用 `broadcast_weights → receive_and_update_weights` 同步最新策略到所有推理器
   - 定期从 StatsActor 拉取统计，计算 steps/sec 和各环境平均回报/成功率，写 TensorBoard

**关键特点**：每个"epoch"内部同步（所有 TrainerActor 一起进退），但整体与 rollout 侧完全并行；rollout 永远在前台跑，trainer 只是后台周期性消费 buffer 中数据。

---

## 三、离散动作版：ds_libero_ppo_discrete.py 关键差异

离散版在整体架构上与连续版几乎一致，但有三个重要差异点：

### 3.1 Experience 结构与优势/回报

**离散版 Experience**：

```python
@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray           # 离散动作 token 向量
    advantage: float
    behaviour_logits: np.ndarray       # 行为策略 logits
    value_target: float               # V target（非 behaviour_value）
```

在 _process_traj 中同时计算：
- 逆序 GAE → `adv`
- `ret = adv + V` → `value_target`

Trainer 直接拿 `adv` 和 `value_target`，不再依赖 `behaviour_value`。

### 3.2 任务采样策略（轻量版 Curriculum Learning）

多任务 LIBERO 环境采样方式：

```python
failure_counts = np.array([sum(history) for history in self.env_outcome])
env_weights = failure_counts + 1
probabilities = env_weights / np.sum(env_weights)
self.current_env_idx = np.random.choice(self.num_tasks, p=probabilities)
```

- `env_outcome[i]` 记录该 task 最近若干 episode 的失败情况
- 失败越多 → 权重越高 → 更容易被采样
- 形成轻量级任务自适应调度，完全在 worker 端本地实现，不需额外调度服务

**卖点**：极低代价在 rollout 端实现动态任务采样，提高 sample efficiency，与异步架构兼容。

### 3.3 推理与训练细节差异

**推理端**：
- InferenceActor 使用离散动作版 ActorCritic
- 输出为离散 logits，使用 Categorical 采样

**Trainer 消费**：
- `action_token_np`：每个维度是一个离散 token
- `behaviour_logits`：原始行为策略 logits

**PPO 损失计算**：
- 行动概率来自 `softmax(logits)`
- 使用交叉熵或 log-prob 对比 `action_token`
- Ratio 基于行为策略 logits 计算的 old log-prob，与当前策略 log-prob 做剪切

**评估机制**：离散版增加 EvaluationWorkerActor，与 RolloutWorkerActor 类似但：
- 不向 ReplayBuffer 写数据
- 专门周期性评估当前策略的成功率
- 与 rollout worker 同样异步运行，对训练无阻塞

---

## 四、通信机制与异步特性总结

### 4.1 训练器 ↔ 推理器权重广播（ds_com）

**TrainerActorCom / InferenceActorCom 设计**：

使用单独的 collective process group（GLOO/NCCL 后端）做权重同步，与训练 all-reduce 组隔离。

**Trainer 端**：
- 用 `GatheredParameters` 在 ZeRO-2 下聚合完整参数到 rank 0
- 遍历 `module.named_parameters(recurse=True)` 和 `named_buffers` 打包为连贯 tensor 列表
- 调用 `dist.broadcast`

**Inference 端**：
- 用同样的顺序和 dtype 分配临时 buffer
- 接收广播结果，按顺序写回 model 参数/buffer

**get_broadcast_signature**：提供 (名称, 形状, dtype) 列表，用于初始化阶段对齐检查

**论文描述**：
- 训练通信（梯度 all-reduce）与推理广播通信使用不同进程组隔离
- 对推理侧采用 push-based 参数同步（非参数服务器拉取），简单轻量

### 4.2 数据流：从环境到梯度更新的异步管线

简化的流水线表示：

```
RolloutWorkerActor (CPU & 环境)
  ├─ 与环境交互，积累 (obs, action, reward, value)
  ├─ 本地 GAE → 优势/回报
  └─ replay.add_batch.remote(batch) [异步]
       ↓
ReplayBufferActor (单线程 Queue)
  ├─ 缓存 Experience
  └─ 支持随机采样
       ↓
TrainerActor (GPU，多进程)
  ├─ 后台协程 _data_fetching_loop 不断 sample(super_batch_size)
  └─ 前台协程 run_training_epoch 使用上一个 ready batch 做多步 PPO 更新
       ↕ (权重广播)
InferenceActor (GPU)
  ├─ 后台协程 _loop 聚合推理请求
  └─ 统一前向，RolloutWorker 视为调用"推理服务"
```

**Pipeline 特点**：

- 环境交互、推理前向、训练更新三条链路高度重叠
- 靠 Ray actor 抽象 + asyncio 协程串联
- 单个组件崩溃时打印详细堆栈，Ray 标记失败，便于 debug

---

## 五、代码层面的创新点 & 论文卖点

### 5.1 资源极致利用（Resource Efficiency）

- **推理端批处理 + 异步队列**
  - 通过 `INFERENCE_BATCH + INFERENCE_TIMEOUT_MS` 自适应 batch，尽量用满 GPU

- **训练端双缓冲超级批次**
  - `_data_fetching_loop` 与训练循环解耦，在 I/O/数据处理与反向传播间建立 pipeline

- **多级并行**
  - 环境并行（40 个 rollout worker）+ 推理批处理 + 多 GPU 数据并行（DeepSpeed ZeRO-2）

- **显存优化**
  - `bf16 + LoRA + ZeRO-2`：大部分参数冻结，仅 LoRA + value head 训练，显存/带宽开销大幅减小

### 5.2 轻量化设计（Lightweight System）

- 使用 Ray Actors + DeepSpeed + PyTorch 三件套，无额外复杂 RPC 框架或参数服务器
- ReplayBuffer 是极简结构（deque + random.sample），但通过 Ray 变成分布式可见服务
- 通信模块 ds_com 用少量代码封装广播逻辑，可重用于其他脚本
- Rollout 端不加载完整模型，仅需 processor，进一步降低单 worker 占用

### 5.3 异步式分布训练（Asynchronous Distributed RL）

- **Rollout ↔ Trainer 异步**
  - RolloutWorkers 永远在后台跑，不因训练慢而停
  - Trainer 仅偶尔从 ReplayBuffer 拉取一大批数据

- **推理服务异步**
  - InferenceActor 的 request/future 机制 + _loop 批处理
  - 标准的异步 RPC → batched inference 模式

- **训练内部异步数据预取**
  - `_data_fetching_loop` 与 `run_training_epoch` 分离
  - 典型的 producer–consumer 模式

- **同步保障**
  - 训练器内梯度同步仍为同步（DeepSpeed all-reduce）
  - 保证优化过程的理论稳定性与可分析性

### 5.4 策略裁剪：从硬 clip 到软 clip 的增强版

- **统一入口与配置**：命令行 `--clip-mode` 选择裁剪策略（默认 `clip`），可用 `--clip-config` 载入 YAML/JSON 覆写超参，便于实验切换。
- **软裁剪系列 (soft_clip)**：使用 `ratio` 的反向幅度 `(1/diff)^α` 作为平滑系数（α 可取 1/2），弱化过大比值的梯度但不置零，减少 dead gradient 与训练震荡。
- **SAPO sigmoid gate**：`tau_pos/tau_neg` 控制正负优势的压缩斜率，`gate = sigmoid(τ(r-1))*(4/τ)` 在极端 IS 权重处平滑抑制梯度，并记录饱和比例作为诊断。
- **CE-GPPO / CISPO 兼容**：提供 general-beta（`ce-gppo_clip`）与 IS-weight clip（`cispo`）两类变体，前向使用裁剪常数、反向保留原始梯度方向，兼顾稳定性与相对无偏性。
- **诊断指标**：训练期统计超出 clip 区间比例、dead-grad 比例、ESS 归一化等（并写入 TensorBoard/SwanLab），便于对比不同裁剪模式的收敛与稳定性。
- **复用范围**：同一 `compute_policy_surrogate` 被标准 PPO 与 V-trace 版共用，后者以 V-trace 的 `pg_adv` 作为优势输入，裁剪逻辑保持一致。
- **新增（本工作 soft clip α 版）**：在 `ds_metaworld_ppo_mlp_add_vtrace_with_param.py` 中引入可配置的软衰减指数 `soft_clip_alpha`，核心公式 `coeff=(1/diff)^α, diff=max(ratio, 1/ratio)`，在不做硬截断的前提下对极端 IS 权重进行平滑抑制（`clip_ratio=0`），α 越大抑制越强，用于与原始 PPO clip 与既有 soft clip 进行对比。

```228:241:rl/ds_metaworld_ppo_mlp_add_vtrace_with_param.py
    if clip_mode == "soft_clip" or clip_mode == "soft_clip_alpha-1" or clip_mode == "soft_clip_alpha-2":
        if clip_mode == "soft_clip_alpha-1":
            soft_clip_alpha = 1
        elif clip_mode == "soft_clip_alpha-2":
            soft_clip_alpha = 2
        else:
            soft_clip_alpha = clip_params.get("soft_clip_alpha", 1)
        diff = torch.maximum(ratio, 1.0 / ratio)
        coeff = (1.0 / diff).detach()
        coeff = coeff ** soft_clip_alpha
        surr_soft = surr1 * coeff
        policy_loss = -torch.mean(surr_soft)
        clip_ratio = 0.0  # 软裁剪无硬截断
        return policy_loss, clip_ratio
```

裁剪核心实现与调用示例：

```203:274:rl/ds_metaworld_ppo_mlp_with_param.py
def compute_policy_surrogate(
    clip_mode: str,
    ratio: torch.Tensor,
    adv_unsqueezed: torch.Tensor,
    clip_params: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    if clip_mode == "clip":
        ...
    if clip_mode == "soft_clip" or clip_mode == "soft_clip_alpha-1" or clip_mode == "soft_clip_alpha-2":
        ...
    if clip_mode in ("sapo_soft_clip", "sapo", "sapo_gate"):
        ...
    if clip_mode == "ce-gppo_clip":
        ...
    if clip_mode in ("cispo", "cispo_clip", "is_clip", "is_weight_clip"):
        ...
```

```1364:1387:rl/ds_metaworld_ppo_mlp_with_param.py
ratio = torch.exp(logp - logp_old)
adv_unsqueezed = normalized_adv.unsqueeze(dim=-1)
policy_loss, clip_ratio = compute_policy_surrogate(
    clip_mode=CLIP_MODE,
    ratio=ratio,
    adv_unsqueezed=adv_unsqueezed,
    clip_params=CLIP_PARAMS,
)
```
