# RLinf 论文对比分析

## 一、研究问题与动机

### 1.1 核心问题

RLinf 从一开始就把问题定位在：**大模型时代 RL 训练的系统效率瓶颈**。

论文的核心观察是：现在各种 RL 工作流（RLHF、GRPO、Deep Research、具身强化学习等）非常"杂乱+动态"，包括：

- **多种 LLM**（actor、critic、reward、reference）
- **推理 / 生成 / 训练 / 模拟器 / 检索服务**等异构组件
- **不同组件用不同计算资源**（GPU/CPU、显存占用截然不同、并行策略截然不同）

### 1.2 主要问题

这导致两个直接问题：

#### 硬件利用率低

- LLM 训练需要大量显存存储梯度和优化器状态
- 生成/推理阶段经常因为响应长度分布长尾而导致 GPU 空转
- 模拟器、渲染等大量占用 CPU 或非 tensor 型 GPU pipeline

#### 单一执行模式不适配多样工作流

他们重点对比了两种典型系统模式：

- **Collocated execution**：所有阶段在同一批 GPU 上轮流跑（类似"阶段式大循环"），容易受长尾响应拖累，GPU 有大量 idle
- **Disaggregated pipelining**：不同阶段固定绑到不同 GPU 上，做流水线并行，缓解长尾，但又会产生显存和计算失衡（有的 stage 忙到爆，有的 stage 资源浪费）

论文指出：**没有一种单一的执行模式对于所有 RL 工作流都是最优的**，很多场景其实需要"混合调度"（部分 collocate，部分 pipeline），但现有系统很难在不改逻辑代码的前提下灵活切换这种执行模式。

### 1.3 核心动机

用一句话概括他们的动机：

> "RL 工作流太多样了，真正的瓶颈在于系统灵活性不足——不能够在不改算法代码的情况下切换和组合各种执行模式，从而错失了大量调度优化空间。"

## 二、研究现状与相关工作

论文在 Background + Related Work 里主要梳理了两条线：**通用分布式系统 / 大模型训练系统** 和 **RL 专用系统**。

### 2.1 通用大规模训练系统

他们引用了：

- **TensorFlow、Dryad、MapReduce** 等传统大规模分布式系统
- **现代大模型训练优化**：比如通信-计算 overlap、显存规划、碎片化管理等（如通信划分、spatio-temporal 显存规划的工作）

这些系统的特点是：偏静态 graph 或针对单一训练 workload 的调度优化，不太适合 RL 那种"多组件 + 多环路 + 动态 rollout 长度"的复杂工作流。

### 2.2 RL/LLM 专用系统

他们重点对比了几类：

- **RLHF / RL for LLM 系统**：比如 veRL（他们主要对标的开源 RLHF 系统）、OpenRLHF 等；这些系统通常实现了 GRPO / PPO 等算法，但执行模式比较单一（要么 collocated，要么固定 pipelining），可调度空间有限

- **具身 RL/VLA 训练框架**：比如 RL4VLA、SimpleVLA-RL 等，用于 OpenVLA/OpenVLA-OFT 这类 VLA 模型在 ManiSkill、LIBERO 上做 PPO/GRPO。SimpleVLA-RL 本质上是建立在 veRL 上的扩展

- **Ray 等 actor 模型框架**：Ray 提供了天然适合 RL 的 actor-based 编程模型，但默认还是"你怎么写 workflow graph，就怎么执行"，并没有自动进行复杂的资源调度、执行模式转换

### 2.3 论文定位

论文的定位是：**不是再造一个 RL 算法，而是提出一个新的 "RL 系统范式"——M2Flow**，专门补齐"逻辑 workflow" 与 "物理执行"之间这层灵活性断层。

## 三、框架设计与核心思想：RLinf & M2Flow

这一部分是整篇论文的灵魂，也是对你自己系统最有启发的部分。

### 3.1 核心思想：Macro-to-Micro Flow Transformation（M2Flow）

**关键理念**："开发者写的是宏观逻辑流程（macro logical flow），系统自动生成微观执行流程（micro execution flow）。"

#### 宏观逻辑流程（Macro logical flow）

开发者用一个很自然的、命令式的接口写 RL workflow：

```python
for data in data_iter:
    generate(...)
    inference(...)
    train(...)
    update_weight()
```

这里只描述"组件之间怎么通信、在逻辑上何时同步"，不关心各阶段具体放在哪张 GPU、用什么执行模式。

#### 微观执行流程（Micro execution flow）

RLinf 的调度器根据宏观 flow + profiling 信息，自动决定：

- 每个 worker 具体放在哪些卡（空间维度）
- 以什么粒度做 pipeline
- 哪些阶段在时间上 multiplex 同一批 GPU（通过 context switch）
- 使用 collocated / disaggregated / hybrid 哪种组合

**换句话说**：M2Flow 把 "写逻辑" 和 "调度执行" 解耦开了。

### 3.2 Worker 抽象与 Workflow 编程接口

RLinf 的体系结构大概这样：

- **顶层**：Procedural Programming Interface（命令式编程接口）
- **中间层**：M2Flow Transformation & Scheduling 模块（Scheduler、ExecFlow Manager、Worker Dispatcher、Profiler、Connection Manager）
- **数据面**：Adaptive Communication & Data Channel
- **底层**：Workers（封装 RL 组件：rollout、actor training、critic training、simulator 等）

#### Worker 抽象

有一个基类 Worker，提供：

- **通信接口**：`send/recv`
- **资源管理**：`onload/offload`（加载 / 卸载模型和缓存，用于 context switching）

每个具体 RL 组件（如 RolloutWorker, ActorWorker）继承 Worker，实现核心逻辑。

#### WorkflowRunner 接口

类似你现在自己写的 Ray-based 训练脚本，但更加规范化：

- 在 `__init__` 里创建 Cluster、launch 各类 worker group，创建 Channel
- `run()` 里用循环调用 worker 的方法，并通过 channel 传输数据

WorkerGroup 调用是天然异步的：返回一个 handle，开发者可以选择 `handle.wait()` 来显式同步，依此确定逻辑上的 barrier。

#### Data Channel & Device Lock

Data channel 作为数据平面，解耦 control flow 和 data flow。

一个重要点是 channel 上有 **distributed device_lock**：

- 用来控制哪些 worker 可以在同一时刻占用某个 device 的资源
- 是实现自动 context switching（时分复用 GPU）的关键原语

你可以把 RLinf 看成是：

> "用 Ray 起 worker，但在 Ray 之上再加了一层『workflow DSL + 调度器 + 通信抽象』，让系统可以重写你的执行策略。"

### 3.3 M2Flow 具体机制：Elastic Pipelining & Context Switching

M2Flow 的 transformation 主要靠两个机制：

#### Elastic Pipelining（弹性流水线）——空间维度调度

**核心思想**：允许每个 worker 动态选择处理数据的粒度（batch size）。

例如：

- 生成/推理可以对单个 query 或一批 query 做前向
- 训练 worker 则有 micro-batch / global-batch 概念

Execution Flow Manager 可以据此：

- 选择"更细粒度的 pipeline"：小 batch 快速往下游推进，保持流水线饱和
- 或选择大 batch 提高单次吞吐，减少通信开销

这形成了一个可调的 pipeline 粒度空间。

#### Automatic Context Switching（自动上下文切换）——时间维度调度

**动机**：有些 worker 不能共驻同一批 GPU（显存不够），但又都必须某种程度"共享"这些 GPU。

**实现方式**：

1. 使用 data channel 上的 `device_lock` 实现分布式锁
2. 当一个 worker 获得锁：
   - 通过 `onload()` 把自己的模型权重、KV cache 等加载到 GPU
   - 完成任务后 `offload()` 释放显存，交出锁

系统利用这个机制在时间轴上复用 GPU，实现多 worker 在有限 GPU 上的时分复用，即所谓 **temporal multiplexing**。

### 3.4 Profiling-Guided Scheduling Policy

M2Flow 之上还有一个 profiling-guided scheduler：

- **收集各 worker 的**：
  - 运行时延、显存占用、计算强度
  - 不同执行模式（collocated、pipelined、hybrid）下的 profile

- **根据 profile 搜索"合适的执行模式"和参数**（例如：
  - rollout/训练各用多少卡
  - pipeline 深度和 batch 粒度
  - 哪些 worker 共驻 / 需要 context switch）

这一套让 M2Flow 不只是"理念"，而且在实践中能自动探索调度空间，而不需要工程师手写大量 if-else。

## 四、实验设计：验证 M2Flow & RLinf 的效果

这一部分是你以后写自己实验章节时很值得借鉴的结构。

### 4.1 任务与模型

#### 推理 RL（Reasoning RL）

- **模型**：Qwen2.5-1.5B / 7B / 32B，都是用 DeepSeek-R1 蒸馏的版本
- **数据集**：AReaL-boba-Data，整合了 DeepScaleR、Open-Reasoner-Zero、Light-R1、DAPO、NuminaMath（AoPS/Olympiad 子集）、ZebraLogic 等多个难度较高的推理任务

#### 具身 RL / VLA（Embodied RL）

- **模型**：OpenVLA 和 OpenVLA-OFT（都经过 RL4VLA / SimpleVLA-RL SFT 预训练）
- **环境**：
  - ManiSkill 中的 pick-and-place 任务（用 OpenVLA）
  - LIBERO 基准（用 OpenVLA-OFT）

### 4.2 算法与超参

#### 推理 RL

使用 GRPO，增加两点改动：

- **Token-level loss**：对 response 的 token 平均，而不是 sequence 平均，防止极长答案主导训练（借鉴 DAPO）
- **minibatch early-stop**：对 importance ratio 过大的 mini-batch 直接丢弃，稳定训练

奖励采用简单的规则：答案正确 +5，错误 -5。

#### 具身模型

同时使用 PPO 和 GRPO，分别在 ManiSkill+OpenVLA、LIBERO+OpenVLA-OFT 上实验。

#### 实验规模（示例）

- **Reasoning RL**：
  - batch size 512，sequence length 28672，使用不同 tensor parallel 设置（例如 actor TP=2/4/8）

- **Embodied RL**：
  - ManiSkill：256 环境、每条 trajectory 80 steps
  - LIBERO：512 环境、每条 trajectory 64 steps

### 4.3 硬件 & Baselines

#### 硬件平台

- **H100 集群**：32 个节点，256 张 H100-80GB
- 节点间使用 8×400Gbps RoCE RDMA，节点内用 NVLink

#### Baselines

- **对于 LLM**：
  - veRL v0.5：当前 SOTA 开源 RLHF 系统（后端用 Megatron-LM + SGLang）

- **对于具身 VLA**：
  - RL4VLA：高效 PPO 训练 recipe，他们把算法直接集成进 RLinf 做对照
  - SimpleVLA-RL：在 veRL 上构建的具身 RL 框架，用于多环境并行渲染

#### 评价指标

- **推理 RL**：
  - RLHF throughput：每次 RL 迭代内，所有 prompt+response token 总数 / iteration 时间（tokens/sec）

- **具身 RL**：
  - batch throughput：每次迭代处理的 batch 数量 / iteration 时间（batches/sec）

- **模型质量**：在数学、逻辑推理和具身任务成功率上评估 RL 训练后的模型性能（论文有结果，但我们这里只关心系统层指标）

## 五、效果对比与消融实验

### 5.1 主结果：端到端吞吐量提升

#### 推理 RL（Reasoning RL）

在 Qwen2.5-1.5B / 7B / 32B 上，RLinf 相比 veRL 在 RLHF throughput 上提升：

- **1.10× ~ 1.58×**（随模型和集群规模变化）

还展示了 collocated vs disaggregated 的对比：

- 在长上下文（28,672 tokens）、group size 8 的设置下，disaggregated 模式比 collocated 提升 **1.17×~1.21×**

#### 具身 RL（Embodied RL）

- **ManiSkill（GPU-bound 环境）**：
  - 在 8 GPU、16 GPU、32 GPU 的不同规模下，RLinf 的 hybrid 模式相对 RL4VLA 提升大约 **1.61× ~ 1.88×**

- **LIBERO（CPU-bound 环境）**：
  - collocated 模式反而更优，达到了 **1.25× ~ 2.13×** 的加速

**重要观察**：不同任务（GPU-bound vs CPU-bound）最优执行模式完全不同，RLinf 通过 M2Flow + 调度策略找到了各自的高效模式。

### 5.2 性能拆解：哪部分改善了什么？

论文专门做了 latency breakdown：

#### 对 Qwen2.5-7B（对应某一组设置）

把时间划成：rollout、logprob 计算、训练 三个阶段；

对比 RLinf 和 veRL：

- veRL rollout 更慢（KV cache 分配受限，rollout engine 优化不足）
- veRL logprob/inference 也更慢，成为主要瓶颈
- RLinf 通过更好的执行模式（pipeline + hybrid +更好的 rollout 引擎）大幅减少 roll-out 和 logprob 的时间，从而提升整体吞吐

#### 对 collocated vs disaggregated 模式

- 即便 disaggregated 模式只给 rollout 分配了 64 张卡中的 40 张，rollout 时间只增加了 ≈14%
- 但因为训练和推理可以在其他 GPU 上与新的 rollout 并行，总体 iteration 时间反而减少，throughput 提升

#### 对 LIBERO

- breakdown 显示：这个场景完全被 CPU rollout 绑死，GPU 部分的优化空间有限
- collocated 模式把更多资源集中给 rollout，使得整体速度提升更明显

**结论**：这些拆解很好地支撑了论文的 claim：不同 workload 的瓶颈不同，系统需要足够灵活的执行模式去对症下药。

### 5.3 消融实验：组件贡献分析

严格意义上的"开关式消融"（比如"关闭 context switching 看效果"）在目前看到的文本中没有明显列出；但论文通过以下几种模式对比来间接完成消融：

#### 执行模式消融

同一套 RLinf 系统，在 collocated / disaggregated / hybrid 三种模式下对比吞吐量和 latency breakdown；

这相当于在"是否启用 elastic pipelining + context switching + 某种 placement 策略"之间做对比。

#### 基线系统对比

对比 veRL vs RLinf（LLM 场景）以及 SimpleVLA-RL / RL4VLA vs RLinf（具身场景），把"系统无 M2Flow + 固定执行模式"和"系统有 M2Flow + 调度器"当作整体模块的消融。

**总结**：RLinf 主要通过执行模式（collocated / pipeline / hybrid）的对比和 latency breakdown 来凸显 M2Flow 带来的调度空间和性能收益，而不是对单一机制做孤立的 ablation。

## 六、小结：这篇论文对你这篇论文的"启发点"

为后面阶段三做准备，我在这里提前帮你串一下和你自己的框架的对应关系（后面我们会系统写成 Methodology 对比）：

### 你现在的系统（ds_libero_ppo / _discrete）

**已经有**：

- Ray Actors for rollout / replay / trainer / inference
- 推理端异步批处理（有点像 RLinf 的 rollout worker + data channel）
- 训练端异步取数（super batch + 双缓冲）
- 支持多种算法（PPO、GRPO、世界模型）和多环境（LIBERO 等）

本质上是一个针对具身 VLA 场景的专用异步分布式 RL 框架。

### RLinf 提供了一个"更抽象的系统范式"

M2Flow 抽象出：

- **宏观 logical flow**（你现在基本已经遵守：rollout→replay→train）
- **微观执行 flow**（你部分通过 DeepSpeed 配置和 Ray actor placement 手动实现）

但 RLinf 有的"系统级自适应调度"和"worker 抽象 + data channel + device lock"的统一接口，是你可以借鉴并在论文里对标说明"我们在 VLA/具身场景下实现了一个更轻量、定制化的 M2Flow-like 框架"。

### 你想强调的三个卖点（极致资源利用 / 轻量化 / 异步式分布训练）

在 RLinf 里分别对应：

- **elastic pipelining + context switching + hybrid scheduling** → 极致资源利用
- **worker 抽象 + Ray 集成 + 丰富模型支持，但系统本身不绑定某一种算法** → 灵活且通用
- **rollout / inference / training 各自异步运行，通过 data channel 解耦** → 典型异步式 RL 系统

### 你这篇论文在结构上非常适合

- 先用 RLinf 作为"更通用、更大规模"的系统背景
- 然后强调：

> "我们在 VLA + 世界模型 + Libero 这类具身场景下，设计了一套更轻量、实现成本更低、但在 GPU 利用率和训练吞吐上同样极具优势的异步分布式训练框架。"