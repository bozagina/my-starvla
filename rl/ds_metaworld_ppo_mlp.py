import os
os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6,7"
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import numpy as np

import ray
import torch
import torch.distributions
from torch.distributions import kl
import deepspeed
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter
import swanlab

# MetaWorld 和 MLP Actor-Critic 组件
from rl.metaworld_env import MetaWorldWrapperDiscrete
from rl.policies.mlp_actor_critic import MLPActorCriticDiscrete
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port

# ================================================================
# 0. 超参数与配置 (已修改为单任务学习，并对齐 ds_meta 关键超参便于对比)
# ================================================================
# MetaWorld 单任务设置
METAWORLD_TASKS = ["reach-v3"]  # 单任务学习：只使用 reach-v3
BENCHMARK = "MetaWorld_reach_v3"

# 分布式系统参数（对齐 ds_meta：2 个 Trainer GPU、更多 rollout workers 等）
NUM_TRAINER_GPUS = 2
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 2  # 与 ds_meta 相同，便于对比采样效率
NUM_EVAL_WORKERS = 10
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 32       # 对齐 ds_meta 的推理批大小
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 50_000
TRAIN_BATCH_SIZE = 512
ACCUMULATION_STEPS = 4
TRAIN_ITERS = 100000       # 对齐 ds_meta 的训练总步数

# Checkpoint
CKPT_DIR = f"/cpfs01/qianfy_workspace/openvla_oft_rl/models/finetune_rl_metaworld"
CKPT_EVERY_STEPS = 2000   # 每 N 个训练步保存一次

# PPO（对齐 ds_meta 的大部分超参数）
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01   # 与 ds_meta 一致，保持相似的熵正则强度
KL_COEF = 0.1

# 奖励缩放
REWARD_SCALE = 0.01

# ================================================================
# 学习率调度参数（与 ds_meta 的 LR & warmup 大致对齐）
# ================================================================
VALUE_LR = 3e-5   # 对齐 ds_meta 的学习率量级
POLICY_LR = 3e-5  # 保持策略与价值网络相同量级
VALUE_WARMUP_STEPS = 1000
POLICY_WARMUP_STEPS = 1000
POLICY_TRAIN_START_STEP = 500 # 策略网络从第500个 *更新步* 开始训练

# 日志（滑动窗口长度对齐 ds_meta）
MOVING_AVG_WINDOW = 100
LOG_INTERVAL_SECONDS = 10

# 通信组
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"

# MLP Actor-Critic 配置
STATE_DIM = 39
ACTION_DIM = 4
N_ACTION_BINS = 256
TORCH_DTYPE = torch.float32  # MLP 模型使用 float32

# ================================================================
# 数据结构 更新经验数据结
# ================================================================
@dataclass
class Experience:
    obs: np.ndarray            # 状态向量 (state_dim,)
    action_token: np.ndarray                # 采样的离散动作 token (shape: [ACTION_DIM,])
    advantage: float
    behaviour_logits: np.ndarray            # 行为策略的 logits (shape: [ACTION_DIM, VOCAB_SIZE])
    value_target: float

# ================================================================
# 1.5. 统计模块 (StatsActor)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        self.stats = defaultdict(lambda: {
            "episode_returns": deque(maxlen=window_size),
            "step_times": deque(maxlen=window_size),
            "episode_lengths": deque(maxlen=window_size),
            "successes": deque(maxlen=window_size),
            "total_episodes_processed": 0,
            "total_env_steps": 0
        })
        self.timings = defaultdict(lambda: deque(maxlen=window_size))
        self.actor_last_active = {}
        self.active_window_seconds = 600
        self.total_samples_produced = 0

    def add_episode_return(
        self,
        env_name: str,
        ep_return: float,
        step_time: float,
        ep_length: int,
        success: float,
        actor_id: Optional[int] = None,
        step_num: int = 0,
    ):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["total_episodes_processed"] += 1
        env_stats["total_env_steps"] += ep_length
        if not env_name.startswith("eval_"):
            self.total_samples_produced += step_num
            if actor_id is not None:
                self.actor_last_active[actor_id] = time.time()

    def add_timing_metric(self, metric_name: str, value: float):
        """记录系统性能相关的计时指标"""
        self.timings[metric_name].append(value)

    def get_active_actor_count(self) -> int:
        current_time = time.time()
        cutoff = current_time - self.active_window_seconds
        return sum(1 for last_active in self.actor_last_active.values() if last_active >= cutoff)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        per_env_stats = {}
        all_returns, all_lengths, all_step_times = [], [], []
        total_episodes_processed = 0
        total_env_steps = 0

        eval_returns, eval_lengths, eval_step_times = [], [], []
        eval_total_episodes_processed = 0
        eval_total_env_steps = 0

        for env_name, env_data in self.stats.items():
            if not env_data["episode_returns"]:
                per_env_stats[env_name] = {
                    "avg_return": 0.0,
                    "avg_ep_len": 0.0,
                    "avg_success_rate": 0.0,
                    "num_episodes_in_avg": 0,
                    "total_episodes": env_data["total_episodes_processed"]}
                continue

            per_env_stats[env_name] = {
                "avg_return": np.mean(env_data["episode_returns"]),
                "avg_ep_len": np.mean(env_data["episode_lengths"]),
                "avg_success_rate": np.mean(env_data["successes"]),
                "num_episodes_in_avg": len(env_data["episode_returns"]),
                "total_episodes": env_data["total_episodes_processed"]
            }
            if env_name.startswith("eval_"):
                eval_total_episodes_processed += env_data["total_episodes_processed"]
                eval_total_env_steps += env_data["total_env_steps"]
                eval_returns.extend(env_data["episode_returns"])
                eval_lengths.extend(env_data["episode_lengths"])
                eval_step_times.extend(env_data["step_times"])
            else:
                total_episodes_processed += env_data["total_episodes_processed"]
                total_env_steps += env_data["total_env_steps"]
                all_returns.extend(env_data["episode_returns"])
                all_lengths.extend(env_data["episode_lengths"])
                all_step_times.extend(env_data["step_times"])

        per_env_stats["_global_rollout_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps,
            "total_samples_produced": self.total_samples_produced,
            "active_actor_count": self.get_active_actor_count()
        }
        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "total_episodes_processed": eval_total_episodes_processed,
            "total_env_steps": eval_total_env_steps
        }
        timing_stats = {}
        for name, deq in self.timings.items():
            timing_stats[name] = np.mean(deq) if deq else 0.0
        per_env_stats["_timings_"] = timing_stats
        return per_env_stats

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, batch: List[Experience]):
        self.buffer.extend(batch)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 现在是 numpy 数组，可以直接 stack
        obs_list = np.stack([b.obs for b in batch])  # (batch_size, state_dim)
        action_token = np.stack([b.action_token for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        logits_old = np.stack([b.behaviour_logits for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs_list, action_token, adv, logits_old, v_targ

class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.wid = wid

        # 单任务设置：只初始化一个环境
        self.task_name = METAWORLD_TASKS[0]  # 固定使用第一个（也是唯一的）任务
        print(f"BaseWorker {wid}: 正在初始化单任务 MetaWorld 环境: {self.task_name}")
        self.env = MetaWorldWrapperDiscrete(env_name=self.task_name, bins=N_ACTION_BINS)
        print(f"BaseWorker {wid}: 环境初始化完成。")

        self.task_description = "test: task_description"
        self.current_env_name = self.task_name

@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor):
        super().__init__(infer, replay, wid, stats_actor)
        # 单任务设置：移除环境选择的历史记录
        self.local_buffer = []

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        # 单任务设置：直接重置当前环境
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def run(self):
        try:
            current_seed = int(time.time() * 1000) + self.wid + os.getpid()
            obs, info = self._reset_and_select_env(seed=current_seed)
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            while True:
                # 直接使用状态向量作为模型输入
                action_env, action_token, logits, value = ray.get(self.infer.request.remote(obs, deterministic=False))

                # 修正: 传入 discrete token 给环境，因为 MetaWorldWrapperDiscrete.step 内部会进行从离散到连续的转换
                # 之前传入 action_env (已经转成连续值) 导致了双重转换，数值严重错误
                nxt, r, term, trunc, info = self.env.step(action_token)
                reward_sum += r
                chunk_reward = r * REWARD_SCALE
                step_count_total += 1
                done = term or trunc

                self.local_buffer.append((obs, action_token, chunk_reward, logits, value))
                obs = nxt

                if done:
                    step_time = (time.time() - time_start) / max(step_count_total, 1)
                    success = float(info.get('is_success', 0.0))
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name,
                        reward_sum,
                        step_time,
                        step_count_total,
                        success,
                        actor_id=self.wid,
                        step_num=step_count_total,
                    )
                    reward_sum = 0.0
                    if self.local_buffer: self._process_traj(self.local_buffer, 0.0)
                    self.local_buffer.clear()
                    current_seed = int(time.time() * 1000) + self.wid + os.getpid()
                    obs, info = self._reset_and_select_env(seed=current_seed)
                    time_start, step_count_total = time.time(), 0
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    _, _, _, _, bootstrap_val = self.local_buffer[-1]
                    self._process_traj(self.local_buffer[:-1], bootstrap_val)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e: import traceback; print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise

    def _process_traj(self, traj_segment, bootstrap_val):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        batch: List[Experience] = []
        for i, (s, a_token, _, logits, _) in enumerate(traj_segment):
            batch.append(
                Experience(
                    obs=s,  # 现在是 numpy 数组
                    action_token=a_token.astype(np.int64),
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                )
            )
        self.replay.add_batch.remote(batch)

@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor):
        super().__init__(infer, None, wid, stats_actor)
        print(f"EvaluationWorker {self.wid}: 环境初始化完成。")

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        # 单任务设置：直接重置当前环境
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def run(self):
        try:
            current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
            obs, info = self._reset_and_select_env(seed=current_seed)
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                while not done:
                    # 直接使用状态向量作为模型输入
                    action_env, action_token, _, _ = ray.get(self.infer.request.remote(obs, deterministic=True))

                    # 修正: 传入 discrete token，避免双重转换
                    obs, r, term, trunc, info = self.env.step(action_token)
                    reward_sum += r; step_count_total += 1
                    done = term or trunc

                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info.get('is_success', 0.0))
                self.stats_actor.add_episode_return.remote(
                    f"eval_{self.current_env_name}",
                    reward_sum,
                    step_time,
                    step_count_total,
                    success,
                    actor_id=None,
                    step_num=step_count_total,
                )
                current_seed = int(time.time() * 1000) + os.getpid() + random.randint(0, 10000)
                obs, info = self._reset_and_select_env(seed=current_seed)
        except Exception as e: import traceback; print(f"[ERROR] EvaluationWorker {self.wid} run() 崩溃: {e}", flush=True); traceback.print_exc(); raise


# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, stats_actor):
        super().__init__()
        self.actor_id = actor_id
        print(f"InferenceActor {actor_id}: 正在加载 MLP ActorCritic...")
        self.model = MLPActorCriticDiscrete(
            torch_dtype=TORCH_DTYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_action_bins=N_ACTION_BINS
        )
        self.model.cuda()
        self.model.eval()
        self.stats_actor = stats_actor

        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.requests, self.promises = [], []
        self.last_process_time = time.time()

        loop = asyncio.get_event_loop()
        self._bg_task = loop.create_task(self._loop())
        self._bg_task.add_done_callback(self._on_bg_task_done)
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。")
            return {}
        sd = self.model.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, obs: np.ndarray, deterministic: bool = False):
        """输入: numpy 数组状态向量，输出: 连续动作数组"""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((obs, deterministic))
        self.promises.append(fut)
        return await fut

    async def _loop(self):
        while True:
            should_process = self.requests and (
                len(self.requests) >= self.batch_size or
                time.time() - self.last_process_time > self.timeout_sec
            )
            if not should_process:
                await asyncio.sleep(0.0005)
                continue

            requests_to_process = self.requests
            promises_to_process = self.promises
            self.requests, self.promises = [], []
            self.last_process_time = time.time()

            obs_list = [r[0] for r in requests_to_process]
            deterministic_flags = [r[1] for r in requests_to_process]
            t_loop_start = time.time()
            try:
                # 将观测列表转换为批次输入
                inputs_batch = self.model.prepare_inputs_batch(obs_list)

                with torch.inference_mode():
                    # 前向传播获取 logits 和 value
                    action_logits, value = self.model(inputs_batch)

                    # 后处理以采样动作 tokens 和对应的离散动作
                    dist, action_tokens, discrete_actions = self.model.post_process(
                        action_logits, deterministic=deterministic_flags
                    )

                    # 将离散动作转换为连续动作（用于环境）
                    # MetaWorld 动作空间是4维连续动作，这里使用简单的线性映射
                    actions_env = []
                    for i in range(discrete_actions.shape[0]):
                        # 将每个维度的离散动作 [0, N_ACTION_BINS-1] 映射到连续动作 [-1, 1]
                        continuous_action = -1.0 + 2.0 * discrete_actions[i] / (N_ACTION_BINS - 1)
                        actions_env.append(continuous_action.astype(np.float32))

                    # 转换为numpy数组以便返回
                    actions_env = np.array(actions_env)
                    action_tokens = action_tokens.cpu().numpy()
                    logits = action_logits.cpu().numpy()
                    values = value.cpu().numpy()

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 连续环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i]                 # 价值估计
                    ))
                loop_duration = time.time() - t_loop_start
                self.stats_actor.add_timing_metric.remote("Inference/loop_time_s", loop_duration)
            except Exception as e:
                import traceback
                print(f"[ERROR] InferenceActor {self.actor_id} 批处理失败: {e}", flush=True)
                traceback.print_exc()
                for p in promises_to_process:
                    if not p.done():
                        p.set_exception(e)
                raise


# ================================================================
# 4. 训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor(TrainerActorCom):
    def __init__(self, rank, world_size, replay_buffer):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
        self.global_step = 0

        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_model_keys(self):
        if self.model is None:
            print("模型尚未初始化。请先调用 setup_deepspeed_group()。")
            return {}
        module = self.model.module if hasattr(self.model, "module") else self.model
        sd = module.state_dict()
        res = {k: float(v.abs().sum().item()) for k, v in sd.items()}
        return res

    def get_parameter_counts(self):
        if self.base_model is None:
            return 0, 0
        n_total = sum(p.numel() for p in self.base_model.parameters())
        n_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        return n_total, n_trainable

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def setup_deepspeed_group(self, master_addr, master_port):
        print(f"Trainer {self.rank}: 开始设置 DeepSpeed 环境变量...")
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        print(f"Trainer {self.rank}: 环境变量设置完成 - RANK={self.rank}, WORLD_SIZE={self.world_size}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

        print(f"Trainer {self.rank}: 初始化分布式后端 (nccl)...")
        deepspeed.init_distributed(dist_backend="nccl")
        print(f"Trainer {self.rank}: 分布式后端初始化完成")

        print(f"Trainer {self.rank}: 正在加载 MLP ActorCritic...")
        model = MLPActorCriticDiscrete(
            torch_dtype=TORCH_DTYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_action_bins=N_ACTION_BINS
        )
        self.base_model = model
        print(f"Trainer {self.rank}: MLP ActorCritic 模型创建完成")

        # 参数分组
        print(f"Trainer {self.rank}: 获取参数分组...")
        param_groups = self.base_model.get_parameter_groups()
        print(f"Trainer {self.rank}: 找到 {len(param_groups)} 个参数组")
        optimizer_params = [
            {"params": pg["params"], "name": pg["name"], "lr": POLICY_LR if pg["name"] == "policy" else VALUE_LR}
            for pg in param_groups
        ]
        print(f"Trainer {self.rank}: 参数分组和优化器参数准备完成")

        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "AdamW", "params": {}},
            "bf16": {"enabled": False},  # MLP 模型使用 float32
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
        }
        print(f"Trainer {self.rank}: DeepSpeed 配置准备完成")

        self.data_dtype = torch.float32  # MLP 使用 float32

        print(f"Trainer {self.rank}: 开始初始化 DeepSpeed...")
        try:
            print(f"model: {model}")
            print(f"ds_config: {ds_config}")
            print(f"optimizer_params: {optimizer_params}")
            self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
            print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        except Exception as e:
            print(f"Trainer {self.rank}: DeepSpeed 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")

    async def save_agent(self, ckpt_dir: str, step: int):
        """
        只在 rank-0 上调用。调用 MLPActorCritic 内部的 save_model
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        self.base_model.save_model(ckpt_dir, epoch=step)
        print(f"[Trainer {self.rank}] 已保存 checkpoint -> {ckpt_dir}/mlp_actor_critic_epoch_{step}.pt")

    def _get_current_lr(self, current_step: int, peak_lr: float, warmup_steps: int, total_steps: int, start_step: int = 0) -> float:
        if current_step < start_step: return 0.0
        effective_step = current_step - start_step
        if effective_step < warmup_steps: return peak_lr * (effective_step / warmup_steps)
        progress = (effective_step - warmup_steps) / (total_steps - start_step - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return peak_lr * cosine_decay

    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动 (超级批次大小: {self.super_batch_size})。")
        while True:
            try:
                if self.next_ready_batch is not None:
                    await asyncio.sleep(0.1)
                    continue

                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                obs_np, action_token_np, adv_np, logits_old_np, v_targ_np = \
                    await self.replay_buffer.sample.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                # obs_np 已经是 numpy 数组，直接转换为 torch tensor
                device = next(self.model.parameters()).device
                obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
                act_token_t = torch.tensor(action_token_np, dtype=torch.long, device=device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_old_np, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32, device=device)
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    'obs': obs_t,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)

    async def run_training_epoch(self) -> Tuple[float, float, float, float, float, Dict[str, float], int, float, float, Dict[str, float]]:
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.02)
            print(f"Trainer {self.rank}: 数据已收到，开始训练。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None

        obs_t = current_batch['obs']
        act_token_t = current_batch['act_token']
        adv_t = current_batch['advantage']
        logits_old_t = current_batch['logits_old']
        v_targ_t = current_batch['value_target']
        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # 修正std 归一化（消融1）
        local_sum = adv_t.sum()
        local_sq_sum = (adv_t * adv_t).sum()
        local_count = torch.tensor([adv_t.numel()], device=adv_t.device, dtype=torch.float32)

        distributed.all_reduce(local_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_sq_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_count, op=distributed.ReduceOp.SUM)

        global_mean = local_sum / torch.clamp(local_count, min=1.0)
        global_var = torch.clamp(local_sq_sum / torch.clamp(local_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
        epoch_ent, epoch_kl_divs = [], []

        num_updates_in_epoch = self.super_batch_size // TRAIN_BATCH_SIZE
        t_policy_train_start = time.time()

        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE; end = start + TRAIN_BATCH_SIZE
            mini_obs = obs_t[start:end]
            mini_act_token = act_token_t[start:end]
            mini_adv = adv_t[start:end]
            mini_logits_old = logits_old_t[start:end]
            mini_v_targ = v_targ_t[start:end]

            normalized_adv = (mini_adv - global_mean) / (global_std + 1e-8)

            # 前向
            action_logits, value = self.model(mini_obs)
            value = value.to(torch.float32)

            # 价值损失
            value_loss = VF_COEF * torch.mean((value - mini_v_targ) ** 2)

            if self.global_step < POLICY_TRAIN_START_STEP:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device)
                kl_div = 0.0
                ent = torch.tensor(0.0, device=loss.device)
            else:
                # 策略与熵损失 (离散版本)
                dist = torch.distributions.Categorical(logits=action_logits)
                logp = dist.log_prob(mini_act_token)

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)

                kl_div_tensor = kl.kl_divergence(dist_old, dist)
                kl_div = torch.mean(kl_div_tensor).item()
                kl_loss = KL_COEF * torch.mean(kl_div_tensor)

                ratio = torch.exp(logp - logp_old)
                adv_unsqueezed = normalized_adv.unsqueeze(dim=-1)
                surr1 = ratio * adv_unsqueezed
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_unsqueezed
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                ent = torch.mean(dist.entropy())
                ent_loss = -ENT_COEF * ent

                loss = policy_loss + value_loss + ent_loss + kl_loss

            self.model.backward(loss)
            self.model.step()
            epoch_losses.append(loss.item())
            epoch_p_losses.append(policy_loss.item())
            epoch_v_losses.append(value_loss.item())
            epoch_e_losses.append(ent_loss.item())
            epoch_kl_losses.append(kl_loss.item())
            epoch_ent.append(ent.item())
            epoch_kl_divs.append(kl_div)
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1

        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        avg_ent = np.mean(epoch_ent)
        avg_kl_div = np.mean(epoch_kl_divs)

        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": time.time() - t_policy_train_start
        }

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, perf_metrics


# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    print("=== 开始 MetaWorld MLP PPO 训练脚本 ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"TORCH_DISTRIBUTED_DEBUG: {os.environ.get('TORCH_DISTRIBUTED_DEBUG', 'Not set')}")

    # 设置调试环境变量
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用 IB，如果网络有问题

    # 检查 GPU 状态
    if torch.cuda.is_available():
        print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"GPU {i} 检查失败: {e}")
    else:
        print("CUDA 不可用")

    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm', include_dashboard=False)

    log_dir = f"runs/MetaWorld/{BENCHMARK}/MLP_DS_PPO_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    # 初始化 SwanLab
    swanlab_config = {
        "METAWORLD_TASKS": METAWORLD_TASKS,
        "BENCHMARK": BENCHMARK,
        "NUM_TRAINER_GPUS": NUM_TRAINER_GPUS,
        "NUM_INFERENCE_ACTORS": NUM_INFERENCE_ACTORS,
        "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
        "NUM_EVAL_WORKERS": NUM_EVAL_WORKERS,
        "ROLLOUT_LOCAL_BUF": ROLLOUT_LOCAL_BUF,
        "INFERENCE_BATCH": INFERENCE_BATCH,
        "INFERENCE_TIMEOUT_MS": INFERENCE_TIMEOUT_MS,
        "REPLAY_CAPACITY": REPLAY_CAPACITY,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "TRAIN_ITERS": TRAIN_ITERS,
        "CKPT_DIR": CKPT_DIR,
        "CKPT_EVERY_STEPS": CKPT_EVERY_STEPS,
        "GAMMA": GAMMA,
        "LAMBDA": LAMBDA,
        "CLIP_EPS": CLIP_EPS,
        "VF_COEF": VF_COEF,
        "ENT_COEF": ENT_COEF,
        "KL_COEF": KL_COEF,
        "REWARD_SCALE": REWARD_SCALE,
        "VALUE_LR": VALUE_LR,
        "POLICY_LR": POLICY_LR,
        "VALUE_WARMUP_STEPS": VALUE_WARMUP_STEPS,
        "POLICY_WARMUP_STEPS": POLICY_WARMUP_STEPS,
        "POLICY_TRAIN_START_STEP": POLICY_TRAIN_START_STEP,
        "MOVING_AVG_WINDOW": MOVING_AVG_WINDOW,
        "LOG_INTERVAL_SECONDS": LOG_INTERVAL_SECONDS,
        "STATE_DIM": STATE_DIM,
        "ACTION_DIM": ACTION_DIM,
        "N_ACTION_BINS": N_ACTION_BINS,
    }
    swanlab.init(
        project="MetaWorld-PPO-Benchmark",
        experiment_name=f"MLP_DS_PPO_{int(time.time())}",
        description=f"MetaWorld MLP DeepSpeed PPO Training - MetaWorld_reach_v3",
        config=swanlab_config,
    )

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [
        TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i])
        for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, stats_actor=stats_actor) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor,
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]
    eval_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS], f"eval_{i}", stats_actor
        ) for i in range(NUM_EVAL_WORKERS)
    ]
    print(f"已创建 {NUM_ROLLOUT_WORKERS} 个 Rollout workers 和 {NUM_EVAL_WORKERS} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    print("正在查找空闲端口...")
    train_group_port = find_free_port()
    print(f"训练组端口: {train_group_port}")

    broadcast_group_port = find_free_port()
    while broadcast_group_port == train_group_port:
        broadcast_group_port = find_free_port()
    print(f"广播组端口: {broadcast_group_port}")

    print("获取训练器主节点地址...")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    print(f"训练器主节点地址: {trainer_master_addr}")

    print("开始初始化 DeepSpeed 训练组...")
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    print(f"创建了 {len(train_setup_tasks)} 个训练器设置任务")

    try:
        print("等待 DeepSpeed 初始化完成...")
        ray.get(train_setup_tasks, timeout=30)  # 5分钟超时
        print("DeepSpeed 训练组建立完成。")

        # 获取并记录参数量
        n_total_params, n_trainable_params = ray.get(trainer_group[0].get_parameter_counts.remote())
        print(f"模型总参数量: {n_total_params:,}, 可训练参数量: {n_trainable_params:,}")
        try:
            swanlab.config.update({
                "total_params": n_total_params,
                "trainable_params": n_trainable_params
            })
        except Exception as e:
            print(f"SwanLab config update failed: {e}")

    except Exception as e:
        print(f"DeepSpeed 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n--- 步骤 3: 建立共享广播组 ({BROADCAST_GROUP_NAME}) ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=broadcast_group_port,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("共享广播组建立完成。")

    inf_keys = ray.get(inference_pool[0].get_model_keys.remote())
    trainer_keys = ray.get(trainer_group[0].get_model_keys.remote())
    for key in inf_keys:
        if key not in trainer_keys:
            print(f"警告: 推理器中缺少训练器的键: {key}")
    for key in trainer_keys:
        if key not in inf_keys:
            print(f"警告: 训练器中缺少推理器的键: {key}")
    train_sig = ray.get(trainer_group[0].get_broadcast_signature.remote())
    infer_sig = ray.get(inference_pool[0].get_broadcast_signature.remote())
    if len(train_sig) != len(infer_sig):
        raise RuntimeError(f"训练器与推理器的广播签名长度不匹配: {len(train_sig)} vs {len(infer_sig)}")
    for i, (a, b) in enumerate(zip(train_sig, infer_sig)):
        if a != b:
            raise RuntimeError(f"First mismatch at idx: {i}, trainer: {a}, inference: {b}")
    print("广播签名验证通过。")

    broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
    receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
    ray.get([broadcast_task] + receive_tasks)
    print("初始权重已广播到所有推理器。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers: w.run.remote()
    for w in eval_workers: w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _ = results[0]
        train_time = time.time() - t_train_start

        t_sync_start = time.time()
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        sync_time = time.time() - t_sync_start

        if global_step > 0 and global_step % CKPT_EVERY_STEPS == 0:
            ray.get(trainer_group[0].save_agent.remote(CKPT_DIR, global_step))

        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            all_stats = ray.get(stats_actor.get_stats.remote())

            elapsed_log_time = current_time - last_log_time
            steps_since_last_log = global_step - last_log_global_step
            training_speed_steps_per_sec = steps_since_last_log / elapsed_log_time if elapsed_log_time > 0 else 0.0

            timing_stats = all_stats.pop("_timings_", {})
            global_stats = all_stats.pop("_global_rollout_")
            eval_stats = all_stats.pop("_global_eval_")
            avg_return = global_stats["avg_return"]
            avg_ep_len = global_stats["avg_ep_len"]
            total_episodes = global_stats["total_episodes_processed"]
            total_env_steps = global_stats["total_env_steps"]
            avg_step_time = global_stats["avg_step_time"]

            eval_avg_return = eval_stats["avg_return"]
            eval_avg_ep_len = eval_stats["avg_ep_len"]
            eval_total_episodes = eval_stats["total_episodes_processed"]
            eval_env_steps = eval_stats["total_env_steps"]
            eval_avg_step_time = eval_stats["avg_step_time"]

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, perf_metrics_list = zip(*results)
            current_lrs = lrs_list[0]

            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"更新步 {global_step}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"全局平均奖励: {avg_return:.2f} | 全局平均幕长: {avg_ep_len:.1f} | Eval奖励: {eval_avg_return:.2f} | "
                  f"value loss: {np.mean(v_losses):.4f} | LR(V/P): {current_lrs['value']:.7f}/{current_lrs['policy']:.7f} | "
                  f"Episodes数量: {total_episodes:,} | Step平均时间: {avg_step_time:.3f}s")

            log_metrics = {}
            log_metrics['Train/Learning_Rate/Value'] = current_lrs['value']
            log_metrics['Train/Learning_Rate/Policy'] = current_lrs['policy']
            log_metrics['Loss/Total'] = np.mean(total_losses)
            log_metrics['Loss/Policy'] = np.mean(p_losses)
            log_metrics['Loss/Value'] = np.mean(v_losses)
            log_metrics['Loss/Entropy'] = np.mean(e_losses)
            log_metrics['Loss/KL'] = np.mean(kl_losses)

            log_metrics['Metrics/Entropy'] = np.mean(ents)
            log_metrics['Metrics/KL_Divergence'] = np.mean(avg_kl_divs)
            log_metrics['Metrics/Training_Speed_Steps_per_Sec'] = training_speed_steps_per_sec
            
            for metric_name, metric_value in timing_stats.items():
                log_metrics[f'Performance/{metric_name}'] = metric_value

            avg_policy_sample_time = np.mean([pm["policy_sample_time"] for pm in perf_metrics_list])
            avg_policy_prep_time = np.mean([pm["policy_prep_time"] for pm in perf_metrics_list])
            avg_policy_train_time = np.mean([pm["policy_train_time"] for pm in perf_metrics_list])
            
            log_metrics['Performance/policy_sample_time'] = avg_policy_sample_time
            log_metrics['Performance/policy_prep_time'] = avg_policy_prep_time
            log_metrics['Performance/policy_train_time'] = avg_policy_train_time
            log_metrics['Performance/train_time'] = train_time
            log_metrics['Performance/sync_time'] = sync_time
            log_metrics['Performance/train_time_total'] = time.time() - t_train_start

            log_metrics['Rollout/Average_Return'] = avg_return
            log_metrics['Rollout/Average_Episode_Length'] = avg_ep_len
            log_metrics['Eval/Average_Return'] = eval_avg_return
            log_metrics['Eval/Average_Episode_Length'] = eval_avg_ep_len

            log_metrics['System/Replay_Buffer_Size_Total'] = total_buffer_size
            log_metrics['System/Total_Episodes_Processed'] = total_episodes
            log_metrics['System/Total_Env_Steps'] = total_env_steps
            log_metrics['System/Avg_Step_Time'] = avg_step_time
            log_metrics['System/Eval_Total_Episodes_Processed'] = eval_total_episodes
            log_metrics['System/Eval_Total_Env_Steps'] = eval_env_steps
            log_metrics['System/Eval_Avg_Step_Time'] = eval_avg_step_time
            log_metrics['System/Active_Rollout_Actors'] = global_stats.get("active_actor_count", 0)
            log_metrics['System/Total_Samples_Produced'] = global_stats.get("total_samples_produced", 0)

            for env_name, env_stats in all_stats.items():
                if env_name.startswith("eval_"):
                    tag_prefix = f"Eval/{env_name.replace('eval_', '')}"
                    log_metrics[f'{tag_prefix}/Average_Return'] = env_stats['avg_return']
                    log_metrics[f'{tag_prefix}/Average_Episode_Length'] = env_stats['avg_ep_len']
                    log_metrics[f'{tag_prefix}/Success_Rate'] = env_stats['avg_success_rate']
                    log_metrics[f'{tag_prefix}/Total_Episodes'] = env_stats['total_episodes']
                else:
                    tag_prefix = f"Rollout/{env_name}"
                    log_metrics[f'{tag_prefix}/Average_Return'] = env_stats['avg_return']
                    log_metrics[f'{tag_prefix}/Average_Episode_Length'] = env_stats['avg_ep_len']
                    log_metrics[f'{tag_prefix}/Success_Rate'] = env_stats['avg_success_rate']
                    log_metrics[f'{tag_prefix}/Total_Episodes'] = env_stats['total_episodes']

            for k, v in log_metrics.items():
                writer.add_scalar(k, v, global_step)
            swanlab.log(log_metrics, step=global_step)

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
