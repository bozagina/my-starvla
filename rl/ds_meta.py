import os
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 设置MuJoCo的渲染后端为osmesa，用于无头服务器渲染
os.environ["MUJOCO_GL"] = "osmesa"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# Ray 会根据 @ray.remote(num_gpus=1) 的请求来为每个 Actor 分配和隔离 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6,7" 

import time
import random
import asyncio
from collections import deque
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass

# --- 新增: 引入新环境所需的库 ---
import gymnasium
import metaworld
import numpy as np
from gymnasium.spaces import Box

import ray
import torch
import torch.nn as nn
import torchvision.models as models
import torch.distributed as dist
from torch.distributed import Backend
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import deepspeed
from torch.utils.tensorboard import SummaryWriter
import swanlab

# ================================================================
# 0. 超参数 (已为 MetaWorld reach-v3 调整)
# ================================================================
# --- MetaWorld 环境参数 ---
OBS_SHAPE = (64, 64, 3)    # 图像观测空间 (H, W, C)
ACT_DIM = 4                # 动作空间维度

# --- 分布式系统参数 ---
NUM_TRAINER_GPUS = 2       # Trainer使用GPU数量
NUM_INFERENCE_ACTORS = 1   # 推理Actor数量
NUM_ROLLOUT_WORKERS = 2  # 数据收集Worker数量
NUM_EVAL_WORKERS = 10      # 评估Worker数量
ROLLOUT_LOCAL_BUF = 64     # 每个Rollout Worker的本地缓冲区大小
INFERENCE_BATCH = 32       # 推理服务的批处理大小
INFERENCE_TIMEOUT_MS = 300 # 推理请求的批次级超时时间 (毫秒)
REPLAY_CAPACITY = 50_000   # 每个经验池的容量 (总容量 = 此值 * NUM_TRAINER_GPUS)
TRAIN_BATCH_SIZE = 512    # PPO的训练批次大小 (视觉任务通常需要更小的批次)
ACCUMULATION_STEPS = 4
TRAIN_ITERS = 100000      # 总训练迭代次数

# --- PPO 算法参数 ---
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-5
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# --- 奖励归一化参数 ---
REWARD_SCALE = 0.01  # 设置奖励缩放因子

# --- 学习率调度器参数 ---
WARMUP_STEPS = 1000 # 线性预热的步数

# --- 日志和统计参数 ---
MOVING_AVG_WINDOW = 100    # 计算滑动平均奖励的窗口大小
LOG_INTERVAL_SECONDS = 10  # 每隔多少秒打印一次日志并写入TensorBoard

# --- 通信组端口和名称 ---
TRAIN_GROUP_PORT = 29501
BROADCAST_GROUP_NAME = "trainer_to_inference_broadcast"
BROADCAST_GROUP_PORT = 29502

# ================================================================
# 0.5. 环境封装器与数据类
# ================================================================

class MetaWorldWrapper(gymnasium.Wrapper):
    """
    一个封装器，用于处理 MetaWorld 环境的图像观测。
    它将环境的 render() 输出作为观测。
    """
    def __init__(self, env, shape, sparse=False):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.sparse = sparse
        self.instruction = '指令：请移动到指定位置'

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        image = self.env.render()
        info["life_loss"] = False
        return image, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        image = self.env.render()
        info["life_loss"] = False
        if self.sparse:
            reward = info['success']
        return image, reward, done, truncated, info

@dataclass
class Experience:
    """用于在ReplayBuffer中存储一条经验的数据结构。"""
    obs: np.ndarray          # 观测 (64, 64, 3), uint8
    action: np.ndarray       # 动作
    advantage: float         # 优势值
    behaviour_mu: np.ndarray # 产生动作的策略均值
    value_target: float      # 价值目标 (GAE + V)

# ================================================================
# 1. 通信模块 (与之前代码相同)
# ================================================================
def init_custom_process_group(
    backend=None, init_method=None, timeout=None, world_size=-1, rank=-1,
    store=None, group_name=None, pg_options=None,):
    from torch.distributed.distributed_c10d import (
        Backend, PrefixStore, _new_process_group_helper, _world,
        default_pg_timeout, rendezvous)
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("nccl")
    if timeout is None:
        from datetime import timedelta
        timeout = timedelta(minutes=30)
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store, group_name=group_name,
        **{pg_options_param_name: pg_options}, timeout=timeout)
    if _world:
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg

class GroupManager:
    def __init__(self):
        self._name_group_map = {}
    def create_collective_group(self, backend, world_size, rank, master_addr: str, master_port: int, group_name):
        init_method = f"tcp://{master_addr}:{master_port}"
        pg_handle = init_custom_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank, group_name=group_name)
        self._name_group_map[group_name] = pg_handle
        return pg_handle
    def is_group_exist(self, group_name):
        return group_name in self._name_group_map
    def get_group_by_name(self, group_name):
        if not self.is_group_exist(group_name):
            print(f"警告: 通信组 '{group_name}' 未初始化。")
            return None
        return self._name_group_map[group_name]

_group_mgr = GroupManager()

def init_collective_group(
    world_size: int, rank: int, master_addr: str, master_port: int,
    backend: Union[str, Backend] = "nccl", group_name: str = "default"):
    global _group_mgr
    if not group_name: raise ValueError(f"group_name '{group_name}' 必须是一个非空字符串。")
    if _group_mgr.is_group_exist(group_name): return
    _group_mgr.create_collective_group(backend, world_size, rank, master_addr, master_port, group_name)

def broadcast(tensor, src_rank: int = 0, group_name: str = "default"):
    group_handle = _group_mgr.get_group_by_name(group_name)
    dist.broadcast(tensor, src=src_rank, group=group_handle)

# ================================================================
# 1.5. 统计模块 (StatsActor) (与之前代码相同)
# ================================================================
@ray.remote
class StatsActor:
    def __init__(self, window_size=MOVING_AVG_WINDOW):
        # 训练统计
        self.train_returns = deque(maxlen=window_size)
        self.train_step_times = deque(maxlen=window_size)
        self.train_lengths = deque(maxlen=window_size)
        self.total_train_episodes = 0
        
        # 评估统计
        self.eval_returns = deque(maxlen=window_size)
        self.eval_step_times = deque(maxlen=window_size)
        self.eval_lengths = deque(maxlen=window_size)
        self.total_eval_episodes = 0

    def add_episode_return(self, ep_return: float, step_time: float, ep_length: int, is_eval: bool = False):
        if is_eval:
            self.eval_returns.append(ep_return)
            self.eval_step_times.append(step_time)
            self.eval_lengths.append(ep_length)
            self.total_eval_episodes += 1
        else:
            self.train_returns.append(ep_return)
            self.train_step_times.append(step_time)
            self.train_lengths.append(ep_length)
            self.total_train_episodes += 1

    def get_stats(self) -> Dict[str, float]:
        # 计算训练统计
        if not self.train_returns:
            train_stats = {"avg_return": 0.0, "num_episodes_in_avg": 0, "total_episodes": self.total_train_episodes, "avg_step_time": 0.0}
        else:
            train_stats = {
                "avg_return": np.mean(self.train_returns),
                "num_episodes_in_avg": len(self.train_returns),
                "total_episodes": self.total_train_episodes,
                "avg_step_time": np.mean(self.train_step_times)
            }
            
        # 计算评估统计
        if not self.eval_returns:
            eval_stats = {"eval_avg_return": 0.0, "eval_total_episodes": self.total_eval_episodes}
        else:
            eval_stats = {
                "eval_avg_return": np.mean(self.eval_returns),
                "eval_total_episodes": self.total_eval_episodes
            }
            
        return {**train_stats, **eval_stats}

# ================================================================
# 2. 模型、经验回放、数据收集器
# ================================================================

class ActorCritic(nn.Module):
    """
    用于处理图像输入的 Actor-Critic 模型。
    使用 CNN 作为主干网络来提取特征。
    """
    def __init__(self, input_shape=OBS_SHAPE, output_dim=ACT_DIM, hidden_dim=512):
        super().__init__()
        c, h, w = input_shape
        
        # CNN 主干网络
        self.backbone = nn.Sequential(
            # 输入: (B, 3, 64, 64)
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=0), # -> (B, 32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), # -> (B, 64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # -> (B, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(), # -> (B, 64 * 4 * 4) = (B, 1024)
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头和价值头
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"模型参数量: {num_params:,} (CNN ActorCritic for MetaWorld)")

    def forward(self, x):
        # x 的预期形状: (B, C, H, W)
        feat = self.backbone(x)
        mu = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return mu, self.log_std, value

@ray.remote
class ReplayBufferActor:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    def add_batch(self, batch):
        self.buffer.extend(batch)
    def size(self):
        return len(self.buffer)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # obs 是 (B, 64, 64, 3), uint8
        obs = np.stack([b.obs for b in batch]) 
        act = np.stack([b.action for b in batch])
        adv = np.asarray([b.advantage for b in batch], np.float32)
        mu_old = np.stack([b.behaviour_mu for b in batch])
        v_targ = np.asarray([b.value_target for b in batch], np.float32)
        return obs, act, adv, mu_old, v_targ

@ray.remote
class RolloutWorkerActor:
    def __init__(self, infer, replay, wid, stats_actor):
        self.infer, self.replay = infer, replay
        self.stats_actor = stats_actor
        
        # --- 初始化 MetaWorld 环境 ---
        import metaworld  # 虽然没有显式调用，但是make的时候会用到
        env = gymnasium.make('Meta-World/MT1', env_name="reach-v3", render_mode="rgb_array", camera_name="corner", width=OBS_SHAPE[0], height=OBS_SHAPE[1])
        self.env = MetaWorldWrapper(env, shape=OBS_SHAPE)
        
        self.wid = wid
        self.local_buffer = []

    def run(self):
        obs, _ = self.env.reset(seed=self.wid)
        reward_sum = 0.0
        ep_len = 0
        time_start = time.time()
        step_count = 0
        while True:
            # obs 是 (64, 64, 3) 的 uint8 numpy 数组
            action, mu, value = ray.get(self.infer.request.remote(obs))
            nxt, r, term, trunc, _ = self.env.step(action)
            reward_sum += r
            r *= REWARD_SCALE  # 奖励归一化
            ep_len += 1
            self.local_buffer.append((obs, action, r, mu, value))
            obs = nxt
            step_count += 1
            if term or trunc:
                step_time = (time.time() - time_start) / step_count
                self.stats_actor.add_episode_return.remote(reward_sum, step_time, ep_len)
                reward_sum = 0.0
                ep_len = 0
                if self.local_buffer:
                    self._process_traj(self.local_buffer, 0.0)
                self.local_buffer.clear()
                obs, _ = self.env.reset()
            elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                _, _, _, _, bootstrap_val = self.local_buffer[-1]
                self._process_traj(self.local_buffer[:-1], bootstrap_val)
                self.local_buffer = [self.local_buffer[-1]]

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
        advs_np = (advs_np - np.mean(advs_np)) / (np.std(advs_np) + 1e-8)
        batch = [Experience(s, a, advs_np[i], mu, rets[i]) for i, (s, a, _, mu, _) in enumerate(traj_segment)]
        self.replay.add_batch.remote(batch)

@ray.remote
class EvaluationWorkerActor:
    def __init__(self, infer, wid, stats_actor):
        self.infer = infer
        self.stats_actor = stats_actor
        
        # --- 初始化 MetaWorld 环境 ---
        import metaworld  # 虽然没有显式调用，但是make的时候会用到
        env = gymnasium.make('Meta-World/MT1', env_name="reach-v3", render_mode="rgb_array", camera_name="corner", width=OBS_SHAPE[0], height=OBS_SHAPE[1])
        self.env = MetaWorldWrapper(env, shape=OBS_SHAPE)
        
        self.wid = wid
       

    def run(self):
        obs, _ = self.env.reset(seed=self.wid)
        reward_sum = 0.0
        ep_len = 0
        time_start = time.time()
        step_count = 0
        while True:
            # obs 是 (64, 64, 3) 的 uint8 numpy 数组
            # 评估时使用 deterministic=True
            action, mu, value = ray.get(self.infer.request.remote(obs, deterministic=True))
            nxt, r, term, trunc, _ = self.env.step(action)
            reward_sum += r
            r *= REWARD_SCALE  # 奖励归一化
            ep_len += 1
            obs = nxt
            step_count += 1
            if term or trunc:
                step_time = (time.time() - time_start) / step_count
                # 标记 is_eval=True
                self.stats_actor.add_episode_return.remote(reward_sum, step_time, ep_len, is_eval=True)
                reward_sum = 0.0
                ep_len = 0
                obs, _ = self.env.reset()
                time_start = time.time() # 重置计时器
                step_count = 0
            
                

# ================================================================
# 3. 融合后的推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_id):
        self.actor_id = actor_id
        # --- 修改: 使用新的 CNN 模型 ---
        self.model = ActorCritic(input_shape=(3, OBS_SHAPE[0], OBS_SHAPE[1]), output_dim=ACT_DIM).cuda().eval()
        self.batch_size = INFERENCE_BATCH
        self.timeout_sec = INFERENCE_TIMEOUT_MS / 1000.0
        self.states, self.promises = [], []
        # last_process_time 记录了上次处理批次的时间点
        self.last_process_time = time.time()
        asyncio.get_event_loop().create_task(self._loop())
        print(f"InferenceActor {self.actor_id} 初始化于 GPU: {ray.get_gpu_ids()} (批次超时: {INFERENCE_TIMEOUT_MS}ms)")

    async def request(self, obs, deterministic=False):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.states.append((obs, deterministic))
        self.promises.append(fut)
        return await fut
    
    async def _loop(self):
        while True:
            # 检查是否应该处理当前批次：
            # 1. 队列中必须有请求 (self.states)。
            # 2. 满足以下任一条件：
            #    a. 队列长度达到或超过批次大小。
            #    b. 自上次处理以来，已经超过了超时时间。
            should_process = self.states and (
                len(self.states) >= self.batch_size or 
                time.time() - self.last_process_time > self.timeout_sec
            )
            if should_process:
                # --- 锁定当前要处理的批次 ---
                # 立即从主队列中取出所有待处理的请求，这样新的请求可以继续被添加进来。
                states_to_process = self.states
                promises_to_process = self.promises
                self.states, self.promises = [], []
                # --- 重置计时器 ---
                self.last_process_time = time.time()
                # --- 准备数据并进行推理 (与之前相同) ---
                obs_list = [s[0] for s in states_to_process]
                det_flags = [s[1] for s in states_to_process]
                
                obs_batch_np = np.stack(obs_list)
                obs_processed = obs_batch_np.astype(np.float32) / 255.0
                obs_tensor = torch.tensor(obs_processed, dtype=torch.float32).permute(0, 3, 1, 2).cuda()

                with torch.no_grad():
                    mu, log_std, values = self.model(obs_tensor)
                    std = torch.exp(log_std)
                    base_dist = Normal(mu, std)
                    dist = TransformedDistribution(base_dist, TanhTransform())
                    
                    # 采样动作 (Stochastic)
                    actions_sampled = dist.sample()
                    # 确定性动作 (Deterministic) -> tanh(mu)
                    actions_det = torch.tanh(mu)
                
                actions_sampled_np = actions_sampled.cpu().numpy()
                actions_det_np = actions_det.cpu().numpy()
                mu_np = mu.cpu().numpy()
                values_np = values.cpu().numpy()
                
                # 为每个请求设置其对应的结果
                for i in range(len(promises_to_process)):
                    if det_flags[i]:
                        act = actions_det_np[i]
                    else:
                        act = actions_sampled_np[i]
                    promises_to_process[i].set_result((act, mu_np[i], values_np[i]))
            else:
                await asyncio.sleep(0.0005)

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"InferenceActor {self.actor_id}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    def receive_and_update_weights(self, group_name):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            received_tensor = torch.empty_like(state_dict[key], device='cuda')
            broadcast(received_tensor, src_rank=0, group_name=group_name)
            state_dict[key] = received_tensor
        self.model.load_state_dict(state_dict)

# ================================================================
# 4. 融合后的训练器 (TrainerActor)
# ================================================================
@ray.remote(num_gpus=1)
class TrainerActor:
    def __init__(self, rank, world_size, replay_buffer):
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        self.model = None
        self.training_batch: Optional[Tuple[torch.Tensor, ...]] = None
        self.data_fetching_task = None
        print(f"TrainerActor Rank {self.rank} 初始化于 GPU: {ray.get_gpu_ids()}")

    def get_node_ip(self):
        return ray.util.get_node_ip_address()
    
    async def _data_fetching_loop(self):
        print(f"Trainer {self.rank}: 后台数据准备循环已启动。")
        while True:
            try:
                # 1. 异步获取 NumPy 数据。obs_np 是 (B, H, W, C), uint8
                obs_np, act_np, adv_np, mu_old_np, v_targ_np = await self.replay_buffer.sample.remote(TRAIN_BATCH_SIZE)
                
                # 2. 预处理: uint8 -> float32, 归一化, 维度转换 (B,H,W,C) -> (B,C,H,W)
                obs_processed_np = obs_np.astype(np.float32) / 255.0
                obs_t = torch.tensor(obs_processed_np, dtype=self.data_dtype).permute(0, 3, 1, 2).to(self.model.device)
                
                # 其他数据转换为 Tensor
                act_t = torch.tensor(act_np, dtype=self.data_dtype).to(self.model.device)
                adv_t = torch.tensor(adv_np, dtype=torch.float32).to(self.model.device)
                mu_old_t = torch.tensor(mu_old_np, dtype=torch.float32).to(self.model.device)
                v_targ_t = torch.tensor(v_targ_np, dtype=torch.float32).to(self.model.device)
                
                # 3. 原子地更新训练批次。
                self.training_batch = (obs_t, act_t, adv_t, mu_old_t, v_targ_t)
            except Exception as e:
                if self.training_batch is None:
                    print(f"Trainer {self.rank}: 数据采样失败 (可能是经验池为空)，将在1秒后重试。错误: {e}")
                await asyncio.sleep(3)

    def setup_deepspeed_group(self, master_addr, master_port):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        deepspeed.init_distributed(dist_backend="nccl")
        # --- 修改: 使用新的 CNN 模型 ---
        model = ActorCritic(input_shape=(3, OBS_SHAPE[0], OBS_SHAPE[1]), output_dim=ACT_DIM)
        ds_config = {
            "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "optimizer": {"type": "Adam", "params": {"lr": LR}},
            "scheduler": {
                "type": "WarmupCosineLR", "params": {
                    "total_num_steps": TRAIN_ITERS, "warmup_num_steps": WARMUP_STEPS,
                    "warmup_type": "linear", "warmup_min_ratio": 0.0, "cos_min_ratio": 0.0,
                }
            },
            "fp16": {"enabled": False},
            "zero_optimization": {
                "stage": 2, "allgather_partitions": True, "allgather_bucket_size": 2e8,
                "reduce_scatter": True, "reduce_bucket_size": 2e8, "overlap_comm": False,
                "contiguous_gradients": True
            }
        }
        if ds_config.get("fp16", {}).get("enabled", False): self.data_dtype = torch.float16
        elif ds_config.get("bf16", {}).get("enabled", False): self.data_dtype = torch.bfloat16
        else: self.data_dtype = torch.float32
        self.model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())

    def setup_broadcast_group(self, master_addr, master_port, group_name, group_world_size, my_rank_in_group):
        init_collective_group(
            world_size=group_world_size, rank=my_rank_in_group, master_addr=master_addr,
            master_port=master_port, group_name=group_name)
        print(f"TrainerActor Rank {self.rank}: 已作为 rank {my_rank_in_group} 加入广播组 '{group_name}'。")

    async def train_step(self) -> Tuple[float, float, float, float, float]:
        if self.training_batch is None:
            print(f"Trainer {self.rank}: 首次训练，等待初始数据批次...")
            while self.training_batch is None:
                await asyncio.sleep(0.02) # zzq1204 减少等待时间
            print(f"Trainer {self.rank}: 初始数据已收到，开始训练。")
        
        obs_t, act_t, adv_t, mu_old_t, v_targ_t = self.training_batch
        
        mu, log_std, value = self.model(obs_t)
        std = torch.exp(log_std.expand_as(mu))
        base_dist = Normal(mu, std)
        dist = TransformedDistribution(base_dist, TanhTransform())
        
        with torch.no_grad():
            std_old = std
            base_dist_old = Normal(mu_old_t, std_old)
            dist_old = TransformedDistribution(base_dist_old, TanhTransform())
            logp_old = dist_old.log_prob(act_t).sum(axis=-1)
            
        # --- 修改: 裁剪动作以防止在log_prob计算中出现极端值 ---
        # TanhTransform的输出理论上在(-1, 1)内，但浮点数精度可能导致等于1或-1
        # 将其裁剪到稍小的范围内可以避免log_prob计算中出现-inf
        epsilon = 1e-6
        clipped_act_t = torch.clamp(act_t, -1.0 + epsilon, 1.0 - epsilon)
        logp = dist.log_prob(clipped_act_t).sum(axis=-1)
            
        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = VF_COEF * torch.mean((value - v_targ_t)**2)
        ent_loss = -ENT_COEF * torch.mean(base_dist.entropy().sum(axis=-1))
        loss = policy_loss + value_loss + ent_loss
        
        self.model.backward(loss)
        self.model.step()
        
        current_lr = self.model.get_lr()[0]
        
        return loss.item(), policy_loss.item(), value_loss.item(), ent_loss.item(), current_lr
    
    def broadcast_weights(self, group_name):
        with deepspeed.zero.GatheredParameters(self.model.parameters(), modifier_rank=0):
            if self.rank == 0:
                state_dict = self.model.state_dict()
                for tensor in state_dict.values():
                    tensor_gpu = tensor.to(self.model.device)
                    broadcast(tensor_gpu, src_rank=0, group_name=group_name)

# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm')

    # --- 修改: 更新日志目录名 ---
    log_dir = "outputs/runs/MetaWorld/DS_PPO_simpleCNN_250f3c7_" + str(int(time.time()))
    
    # --- SwanLab Init ---
    # 临时实例化模型以计算参数量
    temp_model = ActorCritic(input_shape=(3, OBS_SHAPE[0], OBS_SHAPE[1]), output_dim=ACT_DIM)
    total_params = sum(p.numel() for p in temp_model.parameters())
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model  # 释放内存

    swanlab_config = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "OBS_SHAPE": OBS_SHAPE,
        "ACT_DIM": ACT_DIM,
        "NUM_TRAINER_GPUS": NUM_TRAINER_GPUS,
        "NUM_INFERENCE_ACTORS": NUM_INFERENCE_ACTORS,
        "NUM_ROLLOUT_WORKERS": NUM_ROLLOUT_WORKERS,
        "ROLLOUT_LOCAL_BUF": ROLLOUT_LOCAL_BUF,
        "INFERENCE_BATCH": INFERENCE_BATCH,
        "REPLAY_CAPACITY": REPLAY_CAPACITY,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
        "TRAIN_ITERS": TRAIN_ITERS,
        "GAMMA": GAMMA,
        "LAMBDA": LAMBDA,
        "LR": LR,
        "CLIP_EPS": CLIP_EPS,
        "VF_COEF": VF_COEF,
        "ENT_COEF": ENT_COEF,
        "REWARD_SCALE": REWARD_SCALE,
        "WARMUP_STEPS": WARMUP_STEPS,
    }
    swanlab.init(
        project="MetaWorld-PPO-Benchmark",
        name=f"CNN_DS_PPO_{int(time.time())}",
        description=f"MetaWorld CNN DeepSpeed PPO Training - MetaWorld_reach_v3",
        config=swanlab_config,
    )

    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    print(f"TensorBoard 日志将保存在: {log_dir}")

    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i]) for i in range(NUM_TRAINER_GPUS)]
    inference_pool = [InferenceActor.remote(actor_id=i) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS], 
            replay_buffers[i % NUM_TRAINER_GPUS],
            i, 
            stats_actor
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]
    evaluation_workers = [
        EvaluationWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            i,
            stats_actor
        ) for i in range(NUM_EVAL_WORKERS)
    ]
    
    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, TRAIN_GROUP_PORT) for actor in trainer_group]
    ray.get(train_setup_tasks)
    print("DeepSpeed 训练组建立完成。")

    print(f"\n--- 步骤 3: 建立共享广播组 ({BROADCAST_GROUP_NAME}) ---")
    broadcast_participants = [trainer_group[0]] + inference_pool
    broadcast_group_world_size = len(broadcast_participants)
    broadcast_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    broadcast_setup_tasks = [
        actor.setup_broadcast_group.remote(
            master_addr=broadcast_master_addr, master_port=BROADCAST_GROUP_PORT,
            group_name=BROADCAST_GROUP_NAME, group_world_size=broadcast_group_world_size,
            my_rank_in_group=rank) for rank, actor in enumerate(broadcast_participants)
    ]
    ray.get(broadcast_setup_tasks)
    print("共享广播组建立完成。")

    print("\n--- 步骤 4: 启动 Rollout Workers 进行数据收集 ---")
    for w in rollout_workers:
        w.run.remote()
    for w in evaluation_workers:
        w.run.remote()
    
    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    while not all(size >= TRAIN_BATCH_SIZE for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {TRAIN_BATCH_SIZE})... (当前大小: {sizes})")
        time.sleep(2)
    print("远程经验池已准备好，训练器将按需获取数据。")
    
    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    
    for i in range(TRAIN_ITERS):
        train_tasks = [trainer.train_step.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        
        current_time = time.time()
        if current_time - last_log_time > LOG_INTERVAL_SECONDS:
            stats = ray.get(stats_actor.get_stats.remote())
            avg_return = stats["avg_return"]
            eval_avg_return = stats.get("eval_avg_return", 0.0)
            
            total_losses, p_losses, v_losses, e_losses, lrs = zip(*results)
            current_lr = lrs[0]
            
            elapsed_time = current_time - start_time
            total_buffer_size = sum(ray.get([rb.size.remote() for rb in replay_buffers]))

            print(f"迭代 {i+1}/{TRAIN_ITERS} | 时间: {elapsed_time:.1f}s | "
                  f"平均奖励: {avg_return:.2f} | 评估奖励: {eval_avg_return:.2f} | "
                  f"总损失: {np.mean(total_losses):.3f} | "
                  f"学习率: {current_lr:.7f} | "
                  f"经验池总大小: {total_buffer_size:,} | "
                  f"Step平均时间: {stats['avg_step_time']:.3f}s | ")
            
            writer.add_scalar('Train/Learning_Rate', current_lr, i)
            writer.add_scalar('Rollout/Average_Return', avg_return, i)
            writer.add_scalar('Eval/Average_Return', eval_avg_return, i)
            writer.add_scalar('Loss/Total', np.mean(total_losses), i)
            writer.add_scalar('Loss/Policy', np.mean(p_losses), i)
            writer.add_scalar('Loss/Value', np.mean(v_losses), i)
            writer.add_scalar('Loss/Entropy', np.mean(e_losses), i)
            writer.add_scalar('System/Replay_Buffer_Size_Total', total_buffer_size, i)
            writer.add_scalar('System/Total_Episodes_Processed', stats["total_episodes"], i)
            writer.add_scalar('System/Avg_Step_Time', stats["avg_step_time"], i)
            
            swanlab.log({
                'Train/Learning_Rate': current_lr,
                'Rollout/Average_Return': avg_return,
                'Eval/Average_Return': eval_avg_return,
                'Loss/Total': np.mean(total_losses),
                'Loss/Policy': np.mean(p_losses),
                'Loss/Value': np.mean(v_losses),
                'Loss/Entropy': np.mean(e_losses),
                'System/Replay_Buffer_Size_Total': total_buffer_size,
                'System/Total_Episodes_Processed': stats["total_episodes"],
                'System/Avg_Step_Time': stats["avg_step_time"],
            }, step=i)
            
            last_log_time = current_time

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
