## 一、整体架构总览（异步分布式训练框架）

### 1.1 系统组件角色

在连续版和离散版两个脚本中，整体架构基本一致，仅在动作空间和损失函数设计上存在差异。下面介绍各核心组件：

#### **TrainerActor**（训练器，Ray Actor + DeepSpeed）
- 多 GPU 上的策略/价值网络训练核心
- 使用 DeepSpeed ZeRO-2 进行数据并行与通信优化
- 通过异步后台协程 `_data_fetching_loop` 持续从 ReplayBuffer 拉取大批量数据（超级批次），与前台 PPO 更新循环完全解耦

#### **InferenceActor**（推理器，Ray Actor + 单/多 GPU）
- 常驻在指定 GPU 上，仅负责前向推理，不参与反向传播
- 持有 ActorCritic 模型的一份副本
- 接收来自多个 RolloutWorker 的推理请求，通过异步队列与批处理实现高吞吐推理服务

#### **RolloutWorkerActor**（采样 Worker）
- 不加载完整大模型，仅持有 Processor 和环境
- 调用 InferenceActor 获取动作/logits/value，与环境交互、积累轨迹
- 在本地计算 GAE（广义优势估计）与回报，然后打包成 Experience 写入 ReplayBuffer

#### **ReplayBufferActor**（经验池）
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

整体拓扑可以概括如下：

1. **TrainerActor** 在多 GPU 上用 DeepSpeed 初始化训练进程组
2. **InferenceActor** 在一块（或几块）独立 GPU 上加载同款 ActorCritic 模型
3. **TrainerActor** 周期性将最新权重通过 `TrainerActorCom.broadcast_weights` → NCCL 组 → `InferenceActorCom` 接收并更新
4. 大量 **RolloutWorkerActor** 持续调用 InferenceActor 的异步推理接口，从环境采样，写入 ReplayBuffer
5. **TrainerActor** 后台协程从 ReplayBuffer 拉取超级批次，前台协程执行多步梯度更新
6. **StatsActor / EvaluationWorkerActor** 异步记录指标、跑评估

**核心亮点**：Rollout、推理、训练三者通过 Ray + asyncio 解耦，在时间轴上高度重叠，实现了 GPU 的高利用率和典型 "异步式" 分布训练范式。

---



在脚本开头定义了一组关键超参数：

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


含义和资源利用意图：

4 张训练 GPU + 1 个推理 Actor + 40 个 rollout worker：

训练 GPU 通过 DeepSpeed ZeRO-2 进行 数据并行。

推理与训练物理上可以分 GPU（例如 CUDA_VISIBLE_DEVICES="3,4,5,6,7"，其中 3–6 训练，7 推理），避免推理/训练抢显存。

40 个 rollout worker 充分打满 InferenceActor 的请求队列，隐藏推理/环境延迟。

INFERENCE_BATCH + TIMEOUT 机制：

InferenceActor 以 INFERENCE_BATCH=8 为单位做 batched forward。

若短时间内请求不足 batch size，INFERENCE_TIMEOUT_MS 到期就用小批次强制推理，避免等待过久。

SUPER_BATCH_SIZE / TRAIN_BATCH_SIZE / ACCUMULATION_STEPS：

Trainer 每次从 ReplayBuffer 取一个 超级批次（260 轨迹样本），再细分成 TRAIN_BATCH_SIZE=20 的小批，配合梯度累计 ACCUMULATION_STEPS=13。

这套设计允许：显存里只放小 batch，但逻辑上做大 batch 优化，兼顾稳定性与资源占用。

再加上：

TMPDIR="/dev/shm"：中间文件放内存盘，减轻 I/O。

USE_BF16 + DeepSpeed ZeRO-2：减显存 + 减通信带宽。

→ 这为论文中“资源极致利用 / 轻量化”提供了非常具体的实现细节。

2.2 Experience & ReplayBufferActor（经验抽象与异步采样）

连续版的 Experience 结构：

@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]      # CPU 端的观测字典
    action: np.ndarray                # 标准化动作（tanh 之后，[-1, 1]）
    advantage: float                  # GAE 计算出的优势
    behaviour_mu: np.ndarray          # 行为策略的均值
    behaviour_log_std: np.ndarray     # 行为策略的 log_std
    behaviour_value: float            # 行为策略的 V(s)


观测 obs 是已经通过 prepare_one_obs 预处理后的 CPU Tensor 字典，方便后面批量 pad/stack。

动作信息以 (action, mu, log_std) 存在 ReplayBuffer 中，训练时可以完全重建行为策略 log prob，做 PPO ratio。

ReplayBufferActor 是一个非常轻量的 Ray actor：

内部就是 deque(maxlen=REPLAY_CAPACITY)。

add_batch(batch)：append 一批 Experience。

sample(batch_size)：

random.sample 从 buffer 选 batch。

返回 obs_list (list of dict)、act、adv、mu_old、log_std_old、v_old。

非阻塞 & 线程安全：作为单线程 Ray actor，本身天然串行执行，但各 worker 异步往它发远程调用；对上层来讲是“异步管道的一环”。

2.3 RolloutWorkerActor：环境交互 + GAE 计算

RolloutWorkerActor 的关键点：

环境与处理器初始化

使用 LiberoEnvWrapper 包装 LIBERO 任务。

每个 worker 持有一个 processor（来自 OpenVLA），但不加载完整模型，减轻显存压力。

主循环 run()

每一轮：

用 prepare_one_obs + step_count 构造单步输入。

调用 self.infer.request.remote(inputs_t) → 同步 ray.get 获得：

action_env（已 unnormalize，可直接喂环境）

action_norm（归一化动作，用于训练）

mu, log_std, value

对 action_env 逐步与环境交互，按 chunk 收集 (obs, action_norm, reward_scaled, mu, log_std, value) 进入 self.local_buffer。

一旦：

episode 结束 → _process_traj(self.local_buffer, bootstrap_val=0.0)，并清 buffer。

buffer 长度达到 ROLLOUT_LOCAL_BUF + 1 → 用最后一个 value 做 bootstrap，GAE 一次性回灌。

GAE 计算 _process_traj

逆序遍历轨迹，用

𝛿
𝑡
=
𝑟
𝑡
+
𝛾
𝑉
𝑡
+
1
−
𝑉
𝑡
,
GAE
𝑡
=
𝛿
𝑡
+
𝛾
𝜆
GAE
𝑡
+
1
δ
t
	​

=r
t
	​

+γV
t+1
	​

−V
t
	​

,GAE
t
	​

=δ
t
	​

+γλGAE
t+1
	​


将得到的优势 adv 存入 Experience；value 本身作为 behaviour_value，用于构造 value loss。

将整个 batch 通过 self.replay.add_batch.remote(batch) 异步写入 ReplayBufferActor。

这里一个重要卖点是：优势估计完全在 rollout 侧完成，Trainer 直接消费优势，从而减轻训练端的 compute load，把更多计算前移到 CPU 端 / rollout 端。

2.4 InferenceActor：异步批处理推理（异步性的核心）

InferenceActor 继承自 InferenceActorCom，关键结构：

初始化时：

在 GPU 上加载 ActorCritic（连续版本）。

记录 batch_size = INFERENCE_BATCH 和 timeout_sec = INFERENCE_TIMEOUT_MS / 1000。

建立两个队列：

self.requests: List[inputs_dict]

self.promises: List[asyncio.Future]

在当前事件循环里创建后台任务：

self._bg_task = loop.create_task(self._loop())
self._bg_task.add_done_callback(self._on_bg_task_done)


前端接口 request(self, inputs)

创建一个 asyncio.Future 放进 self.promises，inputs 放进 self.requests。

直接返回这个 future，由调用方决定 ray.get 等待结果。

RolloutWorker 调用 infer.request.remote(...)，等价于“RPC + Future”。

后台循环 _loop（异步批处理逻辑）

不断检查：

如果 len(requests) >= batch_size 或者 当前时间 - last_process_time > timeout_sec：

将当前 requests 和 promises 弹出，形成一个批次。

把所有 inputs 用 prepare_inputs_batch 堆叠成大 batch。

在 GPU 上前向一次，得到：

标准化动作

mu, log_std, values

将标准化动作 clip 到 [-1,1]，再用 vla._unnormalize_actions 映射回环境动作。

将每条样本的 (action_env[i], actions_norm[i], mu[i], log_std[i], values[i])
通过对应的 promise.set_result(...) 返回。

如果前向过程报错，会捕获异常，打印堆栈，并对所有未完成的 promises 调用 set_exception，避免上游死等。

这部分是整个框架 “异步推理 + 资源极致利用” 的核心：

多个 rollout worker 的请求被统一打包，最大化 GPU 利用率。

通过 timeout 机制在吞吐和延迟之间折中。

前后端用 Future 解耦，RolloutWorker 视角上只看到一个远端“推理服务”。

2.5 TrainerActor：DeepSpeed + 异步数据加载 + PPO 训练

TrainerActor 的亮点主要在：

(1) DeepSpeed 初始化 & 轻量化微调

通过 deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)：

model 是 ActorCritic，内部使用 LoRA 微调 OpenVLA 主干，仅少量参数可训练（显著减少显存占用）。

优化器参数分为 policy 和 value 两个 param group，可以使用不同的学习率调度。

DeepSpeed ZeRO-2 配置启用了：

overlap_comm=True，通信与计算重叠；

reduce_scatter=True 等，降低通信开销。

脚本中打印：

总参数量: n_total, 可训练参数量: n_trainable


强调“主体参数冻结 + 小量可训练参数”的轻量特性。

(2) 异步数据准备 _data_fetching_loop

启动时 self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())。

_data_fetching_loop 逻辑：

如果 self.next_ready_batch 已经有数据，则 await asyncio.sleep(0.1)，避免超前取数。

调用 await self.replay_buffer.size.remote() 检查 ReplayBuffer 大小，不足 SUPER_BATCH_SIZE 就睡 3 秒重试。

一旦数据足够，调用 await self.replay_buffer.sample.remote(self.super_batch_size) 拉取超级批次。

用 base_model.prepare_inputs_batch(obs_list) 做批量 pad/stack。

将 numpy 数据转换为 torch.Tensor，迁移到 self.model 的 device。

将 (inputs_batch, act_t, adv_t, mu_old_t, log_std_old_t, v_old_t) 塞进 self.next_ready_batch。

这实现了一个经典的 双缓冲 / 预取管线：

前台 run_training_epoch 正在吃 current_batch 时，

后台 _data_fetching_loop 已经在并行准备下一个 super batch。

(3) PPO 训练逻辑 run_training_epoch

核心步骤：

等待 next_ready_batch 准备好（首个周期会 block 一次）。

拉出 current_batch，同时清空 next_ready_batch，让后台继续 fetch。

根据 adv_t + v_old_t 构造 v_targ_t，并在全局范围做 advantage 的 all-reduce 标准化：

先在本机求 local_sum, local_sq_sum, local_count。

distributed.all_reduce 聚合到所有 trainer rank。

算出 global_mean, global_std，对优势做标准化。

将超级批次按 TRAIN_BATCH_SIZE 拆成多个小 batch，循环：

前向得到新策略分布和 value。

计算 PPO 损失：policy loss + value loss + entropy regularization + KL penalty 等。

调用 self.model.backward(loss) + 梯度裁剪 + self.model.step()。

使用自定义 _get_current_lr 做 warmup + cosine decay，分开对 policy、value 组调整学习率。

训练过程本身是典型的 synchronous data-parallel（用 all-reduce 做梯度聚合），
但与 ReplayBuffer / Rollout / 推理之间是 异步解耦 的。

2.6 main()：训练循环与权重广播

主函数大致流程：

Ray 初始化，启动多个 TrainerActor、InferenceActor、ReplayBufferActor、RolloutWorkerActor、StatsActor 等。

训练 group 内部使用 DeepSpeed / torch.distributed 自己的进程组；
推理广播组使用 ds_com 创建的额外 group（独立 master_port）。

通过 TrainerActorCom.get_broadcast_signature() 与 InferenceActorCom.get_broadcast_signature() 比较参数/缓冲区的签名，确保结构一致。

初次广播：

trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)。

InferenceActor.receive_and_update_weights.remote(BROADCAST_GROUP_NAME)。

启动 RolloutWorkers 和 EvaluationWorkers 的 run.remote()，它们在后台无限循环。

等待 ReplayBuffer 先填满一定量数据（warmup 阶段）。

主训练 loop：

并行发起所有 trainer 的 run_training_epoch.remote()。

使用 ray.get 等待一次 epoch 完成，拿到 loss / 统计信息 / global_step。

用 broadcast_weights → receive_and_update_weights 将最新策略同步到所有推理器。

定期从 StatsActor 拉取统计，计算训练速度 steps/sec 和各环境的平均回报/成功率，写 TensorBoard。

在这个粒度上，每个 “epoch” 仍然是同步的（所有 TrainerActor 一起进退），
但 和 rollout 侧是完全并行的：rollout 永远在前台跑，trainer 只是在后台周期性地消费 buffer 中的数据。

三、离散动作版：ds_libero_ppo_discrete.py 关键差异

离散版在整体架构上与连续版几乎一致，但有三个重要差异点，与论文 Methodology 中 “通用异步框架 + 多动作空间支持” 强相关：

3.1 Experience 结构与优势/回报

离散 Experience：

@dataclass
class Experience:
    obs: Dict[str, torch.Tensor]
    action_token: np.ndarray           # 离散动作 token 向量
    advantage: float
    behaviour_logits: np.ndarray       # 行为策略 logits
    value_target: float               # V target，而不是 behaviour_value


在 _process_traj 中同时计算：

逆序 GAE → adv；

ret = adv + V → value_target。

Trainer 在读取时直接拿 adv 和 value_target，不再依赖 behaviour_value。

3.2 RolloutWorkerActor 的 任务采样策略（轻量版 curriculum learning）

在离散版的 RolloutWorkerActor 中，多任务 LIBERO 环境通过以下方式选择：

failure_counts = np.array([sum(history) for history in self.env_outcome])
env_weights = failure_counts + 1
probabilities = env_weights / np.sum(env_weights)
self.current_env_idx = np.random.choice(self.num_tasks, p=probabilities)


env_outcome[i] 记录该 task 最近若干 episode 的失败情况。

失败越多 → 权重越高 → 更容易被采样。

这形成了一个 轻量级任务自适应调度策略，完全在 worker 端本地实现，不需要额外的调度服务。

卖点：利用极低代价在 rollout 端实现了动态任务采样，提高 sample efficiency，且与异步架构兼容。

3.3 推理与训练细节差异

推理端 InferenceActor 使用的是 ActorCritic 的 离散动作版本：

输出是一组离散 logits，使用 Categorical 采样。

Trainer 从 ReplayBuffer 拿到：

action_token_np：每个维度是一个离散 token。

behaviour_logits：原始行为策略 logits。

PPO 损失：

行动概率来自 softmax(logits)，使用交叉熵或 log-prob 对比 action_token。

ratio 基于行为策略 logits 计算的 old log-prob，与当前策略的 log-prob 做剪切。

此外，离散版增加了 EvaluationWorkerActor：

与 RolloutWorkerActor 类似，但：

不向 ReplayBuffer 写数据。

专门用于周期性评估当前策略的成功率。

评估 worker 与 rollout worker 同样异步运行，对训练无阻塞。

四、通信机制与异步特性总结（全局视角）
4.1 训练器 ↔ 推理器权重广播（ds_com）

TrainerActorCom / InferenceActorCom 封装了以下设计：

使用单独的 collective process group（可选 GLOO / NCCL 后端）做权重同步，与训练 all-reduce 组解耦。

Trainer 端：

用 GatheredParameters 在 ZeRO-2 下先聚合完整参数到 rank 0。

遍历 module.named_parameters(recurse=True) 和 named_buffers 打包为连贯 tensor 列表，调用 dist.broadcast。

Inference 端：

用同样的顺序和 dtype 分配临时 buffer。

接收广播结果，再按顺序写回 model 中的参数 / buffer。

get_broadcast_signature 提供（名称, 形状, dtype）列表，用于初始化阶段的对齐检查。

对论文而言，可以描述为：

训练通信（梯度 all-reduce）与推理广播通信使用不同进程组隔离；

对推理侧采用 push-based 参数同步（而非参数服务器拉取），简单轻量。

4.2 数据流：从环境到梯度更新的异步管线

可以用一个简化的流水线表示：

RolloutWorkerActor（CPU & 环境）

与环境交互，积累 (obs, action, reward, value)。

本地 GAE → 优势 / 回报。

replay.add_batch.remote(batch)（异步）。

ReplayBufferActor（单线程 Queue）

缓存 Experience，支持随机采样。

TrainerActor（GPU，多进程）

后台协程 _data_fetching_loop 不断 sample(super_batch_size)。

前台协程 run_training_epoch 使用上一个 ready batch 做多步 PPO 更新。

InferenceActor（GPU）

背景协程 _loop 聚合推理请求，统一 forward。

RolloutWorker 视为调用一个“推理服务”。

在这个 pipeline 中：

环境交互、推理前向、训练更新 三条链路高度重叠，靠 Ray 的 actor 抽象 + asyncio 协程串起来。

单个组件崩溃时，会打印详细堆栈并让 Ray 标记失败，便于 debug。

五、代码层面的创新点 & 论文可用卖点整理

结合你希望突出的三点，我帮你先从代码维度整理一个“可直接写进论文”的卖点列表，后续在 Methodology 可以展开：

5.1 资源极致利用（Resource Efficiency）

推理端批处理 + 异步队列

通过 INFERENCE_BATCH + INFERENCE_TIMEOUT_MS 做自适应 batch，尽量用满 GPU。

训练端双缓冲超级批次

_data_fetching_loop 与训练循环解耦，在 I/O / 数据处理与反向传播之间建立 pipeline。

多级并行

环境并行（40 个 rollout worker） + 推理批处理 + 多 GPU 数据并行（DeepSpeed ZeRO-2）。

bf16 + LoRA + ZeRO-2

大部分参数冻结，仅 LoRA + value head 训练，显存/带宽开销都大幅减小。

5.2 轻量化设计（Lightweight System）

使用 Ray Actors + DeepSpeed + PyTorch 三件套，没有额外复杂的 RPC 框架或参数服务器。

ReplayBuffer 是一个极简结构（deque + random.sample），但通过 Ray 变成分布式可见的服务。

通信模块 ds_com 用少量代码封装了广播逻辑，可重用于其他脚本（世界模型、GRPO 等）。

Rollout 端不加载完整模型，只需要 processor，进一步降低单 worker 占用。

5.3 异步式分布训练（Asynchronous Distributed RL）

Rollout ↔ Trainer 异步：

RolloutWorkers 永远在后台跑，不因训练慢而停；Trainer 只是偶尔从 ReplayBuffer 拉取一大批数据。

推理服务异步：

InferenceActor 的 request/future 机制 + _loop 批处理，是标准的异步 RPC → batched inference 模式。

训练内部异步数据预取：

_data_fetching_loop 与 run_training_epoch 的分离是典型的 producer–consumer 模式。

同时保持：

训练器内部的梯度同步仍然是同步的（通过 DeepSpeed all-reduce），

从而保证优化过程的理论稳定性与可分析性。