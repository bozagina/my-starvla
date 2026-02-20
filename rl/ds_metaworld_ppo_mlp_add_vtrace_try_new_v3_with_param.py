import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# os.environ["MUJOCO_GL"] = "osmesa"           # 强制软件渲染
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 保险起见，给 PyOpenGL 也指明
# 设置临时文件目录，避免磁盘I/O瓶颈
os.environ["TMPDIR"] = "/dev/shm"
# 为了让 Ray 能看到所有可用的 GPU，我们在脚本开头设置。
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# NCCL/编译相关的兜底设置，尽量避免初始化卡死
os.environ["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/dev/shm/torch_ext")
os.environ.setdefault("DS_BUILD_FUSED_ADAM", "0")
# 防止 transformers 库的 tokenizer 并行化警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import asyncio
import argparse
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import math
import json

import numpy as np

import ray
import torch
import torch.distributions
from torch.distributions import kl
import deepspeed
import torch.distributed as distributed
from torch.utils.tensorboard import SummaryWriter
import swanlab

try:
    import yaml  # PyYAML 解析配置
except ImportError:
    yaml = None

# MetaWorld 和 MLP Actor-Critic 组件
# zzq 1219 如果success提前终止回合。仅在我提出的新算法上使用，其他时候使用效果一般
# from rl.metaworld_env_success_early_termination import MetaWorldWrapperDiscrete
from rl.metaworld_env import MetaWorldWrapperDiscrete
from rl.policies.mlp_actor_critic import MLPActorCriticDiscrete
# 训练/推理通信（保持接口不变）
from ds_com import TrainerActorCom, InferenceActorCom
from rl.com_utils import find_free_port

IS_SUCCESS_EARLY_TERMINATION = False
# ================================================================
# 0. 超参数与配置 (已修改为单任务学习，并对齐 ds_meta 关键超参便于对比)
# ================================================================
# # MetaWorld 单任务设置
# METAWORLD_TASKS = ["reach-v3"]  # 单任务学习：只使用 reach-v3
# BENCHMARK = "MetaWorld_reach_v3"

# 分布式系统参数（对齐 ds_meta：2 个 Trainer GPU、更多 rollout workers 等）
NUM_TRAINER_GPUS = 1
NUM_INFERENCE_ACTORS = 1
NUM_ROLLOUT_WORKERS = 1  # 与 ds_meta 相同，便于对比采样效率
NUM_EVAL_WORKERS = 5
ROLLOUT_LOCAL_BUF = 64
INFERENCE_BATCH = 32       # 对齐 ds_meta 的推理批大小
INFERENCE_TIMEOUT_MS = 300
REPLAY_CAPACITY = 50_000
TRAIN_BATCH_SIZE = 512
ACCUMULATION_STEPS = 4
TRAIN_ITERS = 100000       # 对齐 ds_meta 的训练总步数

# ===== Success-as-a-Set / Paired-Queue (v0) 超参数 =====
PAIR_TOPK_SUCCESS = 4              # Top-K 最相似 success episodes
PAIR_MIN_SUCCESS_POOL = 4         # success 池太小则不挖掘 pair
PAIR_MAX_SUCC_CANDIDATES = 2048    # 为了效率，从 success 池随机抽样候选上限
PAIR_SIM_THRESH = 0.95             # divergence 判定的 obs cosine 相似度阈值
PAIR_MIN_PREFIX = 5                # 至少共享前缀长度（步）
PAIR_DIVERGENCE_W = 4              # divergence window 半径 w
PAIRED_QUEUE_MAXLEN = 4096         # Paired-Queue 最大容量（pair pack 个数）
PAIRED_MIN_LEN = 8                 # window 有效最短长度（太短跳过）

# ===== Success-as-a-Set (v1) prototypes / clusters 超参数 =====
PROTO_K = 8                         # 成功模式簇数（prototypes 数量）
PROTO_EMA = 0.9                     # prototype EMA 更新系数（越大越稳定）
PROTO_MIN_INIT = PROTO_K            # 至少收集多少 success episodes 后再进行稳定分配（简单版：等于 K）
PAIR_DIVERSE_BY_CLUSTER = True      # pair mining 选 Top-K success 时优先跨簇覆盖

# ===== Bonus Time（Loss A）触发与训练参数 =====
# 通过环境变量便捷调参
BONUS_BATCH_PACKS = int(os.environ.get("BONUS_BATCH_PACKS", "8"))
# 触发阈值默认随 batch 调整：max(32, 2 * BONUS_BATCH_PACKS)
BONUS_TRIGGER_MIN_PAIRS = int(os.environ.get("BONUS_TRIGGER_MIN_PAIRS", str(max(32, 2 * BONUS_BATCH_PACKS))))
BONUS_TRIGGER_MIN_CLUSTERS = int(os.environ.get("BONUS_TRIGGER_MIN_CLUSTERS", "2"))
BONUS_MAX_STEPS = int(os.environ.get("BONUS_MAX_STEPS", "4096"))
# consume: True 则消费 paired pack，False 不消费（便于观察/复用）
BONUS_CONSUME_PACKS = int(os.environ.get("BONUS_CONSUME_PACKS", "1")) == 1
# 冷却步数：防止每个 train step 都触发
BONUS_COOLDOWN_STEPS = int(os.environ.get("BONUS_COOLDOWN_STEPS", "1000"))
BONUS_UPDATES = 4                  # bonus 额外更新次数（固定，避免多卡不同步）
BONUS_LAMBDA = 0.5                 # bonus loss 权重（相对 PPO 主损失）
BONUS_TAU = 0.5                    # ranking 温度 τ
BONUS_MARGIN = 0.0                 # 可选：margin ranking（0 表示关闭）
BONUS_KL_MAX = 0.05                # KL-guard：超过则 early stop（全局平均 KL）
BONUS_ENTROPY_MIN = 0.2            # entropy floor：低于则 early stop（全局平均 entropy）
BONUS_LOG_EVERY = 1                # bonus 内每几步打印一次（debug）


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

# 裁剪默认配置（可扩展）
DEFAULT_CLIP_CONFIG = {
    "clip": {},
}

# 奖励缩放
REWARD_SCALE = 0.01

# 诊断绘图频率
DIAG_EVERY_STEPS = int(os.environ.get("DIAG_EVERY_STEPS", "200"))

# ===== 冷启动自举与策略扰动超参 =====
# Elite 自举（将“近成功”轨迹视为正例，喂给 prototypes / pair）
ELITE_ENABLE = True
ELITE_WINDOW_N = int(os.environ.get("ELITE_WINDOW_N", "500"))
ELITE_TOP_PCT = float(os.environ.get("ELITE_TOP_PCT", "0.1"))
ELITE_MIN_GAP = float(os.environ.get("ELITE_MIN_GAP", "0.0"))  # P90-P50 gap
ELITE_USE_VALUE_FALLBACK = True

# 最佳策略采样（增加成功概率）
P_BEST_ROLLOUT = float(os.environ.get("P_BEST_ROLLOUT", "0.4"))
BEST_POLICY_MODE = os.environ.get("BEST_POLICY_MODE", "ema")  # latest / ema
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))

# Perturbed-best 失败生成（制造可对比失败）
P_PERTURB_BEST = float(os.environ.get("P_PERTURB_BEST", "0.3"))
PERTURB_START_FRAC_MIN = float(os.environ.get("PERTURB_START_FRAC_MIN", "0.3"))
PERTURB_START_FRAC_MAX = float(os.environ.get("PERTURB_START_FRAC_MAX", "0.7"))
PERTURB_TOKEN_PROB = float(os.environ.get("PERTURB_TOKEN_PROB", "0.15"))
PERTURB_TEMPERATURE = float(os.environ.get("PERTURB_TEMPERATURE", "1.5"))
# EMA 权重广播频率（按训练更新步计数）
EMA_BROADCAST_EVERY_UPDATES = int(os.environ.get("EMA_BROADCAST_EVERY_UPDATES", "100"))


PAIR_BACKFILL_MAX_ATTEMPTS = int(os.environ.get("PAIR_BACKFILL_MAX_ATTEMPTS", "16"))
PAIR_BACKFILL_RECENT_MULT  = int(os.environ.get("PAIR_BACKFILL_RECENT_MULT", "4"))
PAIR_BACKFILL_USE_ANCHOR_SIM = int(os.environ.get("PAIR_BACKFILL_USE_ANCHOR_SIM", "1")) == 1
PAIR_BACKFILL_FORCE_ANCHOR = int(os.environ.get("PAIR_BACKFILL_FORCE_ANCHOR", "1")) == 1

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
# 随机种子配置 - 新增部分
# ================================================================
SEED = 42  # 全局固定随机种子，便于复现

def set_seed(seed: int):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 保证可复现性（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(args: Optional[List[str]] = None):
    """
    解析命令行参数，允许动态指定关键超参与设备配置。
    仅暴露与需求对齐的选项：benchmark、CUDA_VISIBLE_DEVICES、训练/采样并行规模、batch size、随机种子与策略裁剪模式。
    """
    parser = argparse.ArgumentParser(description="MetaWorld MLP DeepSpeed PPO 训练脚本")
    parser.add_argument("--task_name", type=str, required=True, help="MetaWorld 任务名称")
    parser.add_argument("--cuda-visible-devices", dest="cuda_visible_devices", type=str,
                        default=os.environ.get("CUDA_VISIBLE_DEVICES", "6,7"),
                        help="CUDA_VISIBLE_DEVICES 环境变量设置")
    parser.add_argument("--num-trainer-gpus", dest="num_trainer_gpus", type=int,
                        default=NUM_TRAINER_GPUS, help="TrainerActor 的 GPU 数量")
    parser.add_argument("--num-rollout-workers", dest="num_rollout_workers", type=int,
                        default=NUM_ROLLOUT_WORKERS, help="Rollout worker 数量")
    parser.add_argument("--num-eval-workers", dest="num_eval_workers", type=int,
                        default=NUM_EVAL_WORKERS, help="Evaluation worker 数量")
    parser.add_argument("--train-batch-size", dest="train_batch_size", type=int,
                        default=TRAIN_BATCH_SIZE, help="训练微批大小")
    parser.add_argument("--seed", dest="seed", type=int, default=SEED, help="全局随机种子")
    parser.add_argument("--clip-mode", dest="clip_mode", type=str, default="clip",
                        help=f"策略裁剪模式（支持 clip/soft_clip，或配置文件中的自定义模式）")
    parser.add_argument("--clip-config", dest="clip_config", type=str, default=None,
                        help="裁剪配置文件路径（YAML/JSON），按 clip_mode 选择对应超参")
    parser.add_argument("--is-success-early-termination", dest="is_success_early_termination", type=bool, default=IS_SUCCESS_EARLY_TERMINATION, help="是否提前终止回合")
    return parser.parse_args(args=args)


def load_clip_config(config_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    加载裁剪模式配置，支持 YAML/JSON。
    返回格式: {clip_mode: {hyperparam: value, ...}, ...}
    """
    default_config = DEFAULT_CLIP_CONFIG
    if config_path is None:
        return default_config

    try:
        with open(config_path, "r") as f:
            if yaml is not None:
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        if not isinstance(cfg, dict):
            print(f"[Warn] clip_config {config_path} 内容不是 dict，使用默认配置。")
            return default_config
        # 合并默认，允许覆盖或新增模式
        merged = {**default_config, **cfg}
        return merged
    except Exception as e:
        print(f"[Warn] 加载 clip_config 失败: {e}，回退默认配置。")
        return default_config


def select_clip_params(clip_mode: str, clip_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    基于 clip_mode 选择对应超参，返回该模式的完整配置 dict。
    后续新增模式时，只需在配置里添加对应键与超参。
    """
    params = clip_config.get(clip_mode, {})
    if clip_mode not in clip_config:
        print(f"[Warn] 未在配置中找到 clip_mode={clip_mode}，将使用默认/空配置。")
    return params


def debug_log(msg: str):
    """统一调试日志格式，带时间戳并立即刷新。"""
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def compute_policy_surrogate(
    clip_mode: str,
    ratio: torch.Tensor,
    adv_unsqueezed: torch.Tensor,
    clip_params: Dict[str, Any],
) -> Tuple[torch.Tensor, float]:
    """
    根据裁剪模式计算策略损失与 clip 比例。
    clip: PPO 原版 clip
    soft_clip: 软衰减版本（参考 ds_metaworld_ppo_mlp_add_vatrace_soft_clip.py 实现）
    后续如需新增模式，可在此函数中扩展。
    """
    surr1 = ratio * adv_unsqueezed

    if clip_mode == "clip":
        ratio_clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        surr2 = ratio_clipped * adv_unsqueezed
        surr_min = torch.min(surr1, surr2)
        policy_loss = -torch.mean(surr_min)
        is_clipped = ~torch.isclose(surr_min, surr1, atol=1e-8)
        clip_ratio = is_clipped.float().mean().item()
        return policy_loss, clip_ratio

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

    # ===== 新增：SAPO soft clip（sigmoid gate）=====
    if clip_mode in ("sapo_soft_clip", "sapo", "sapo_gate"):
        # τ 的非对称设置：通常 τ_neg > τ_pos（负优势更“硬”一点）
        tau_pos = float(clip_params.get("tau_pos", 1.0))
        tau_neg = float(clip_params.get("tau_neg", 2.0))
        if tau_pos <= 0 or tau_neg <= 0:
            raise ValueError(f"tau_pos/tau_neg must be > 0, got {tau_pos}, {tau_neg}")

        # 数值稳定：避免 ratio 极端导致 inf（可按需调大/关掉）
        ratio_min = float(clip_params.get("ratio_min", 1e-6))
        ratio_max = float(clip_params.get("ratio_max", 1e6))
        r = ratio.clamp(ratio_min, ratio_max)

        tau_pos_t = torch.full_like(adv_unsqueezed, tau_pos)
        tau_neg_t = torch.full_like(adv_unsqueezed, tau_neg)
        tau = torch.where(adv_unsqueezed > 0, tau_pos_t, tau_neg_t)

        # gate(r) = (4/τ) * sigmoid( τ*(r-1) )
        x = tau * (r - 1.0)
        gate = torch.sigmoid(x) * (4.0 / tau)

        # surrogate = gate * A   （注意：这里不再是 r*A）
        surr_sapo = gate * adv_unsqueezed
        policy_loss = -torch.mean(surr_sapo)

        # “clip_ratio”在 SAPO 里没有硬裁剪；这里给一个可选的“饱和比例”指标
        # 当 w = 4*p*(1-p) 很小，说明 sigmoid 饱和、更新被强烈抑制（类似“被clip了”）
        p = torch.sigmoid(x)
        w = 4.0 * p * (1.0 - p)
        w_thresh = float(clip_params.get("sapo_w_thresh", 0.05))
        clip_ratio = (w < w_thresh).float().mean().item()

        return policy_loss, clip_ratio

    if clip_mode == "ce-gppo_clip":
        # 对齐 CE-GPPO/GPPO repo 的 general_beta 写法：三种 case
        # low_mask: (ratio < 1-eps) & (A < 0)
        # high_mask:(ratio > 1+eps) & (A > 0)
        beta1 = float(clip_params.get("beta1", 0.75))
        beta2 = float(clip_params.get("beta2", 1.0))

        eps = float(clip_params.get("clip_eps", CLIP_EPS))
        low = 1.0 - eps
        high = 1.0 + eps

        # 只统计“会被 PPO clip 的两类 token”
        low_mask = (ratio < low) & (adv_unsqueezed < 0)
        high_mask = (ratio > high) & (adv_unsqueezed > 0)
        other_mask = ~(low_mask | high_mask)

        ratio_det = ratio.detach().clamp_min(1e-8)

        eff_ratio = torch.empty_like(ratio)
        # 关键：forward = 常数(beta*(1±eps))，backward 通过 ratio/ratio_det 保留梯度
        eff_ratio[low_mask] = beta1 * low / ratio_det[low_mask] * ratio[low_mask]
        eff_ratio[high_mask] = beta2 * high / ratio_det[high_mask] * ratio[high_mask]
        eff_ratio[other_mask] = ratio[other_mask]

        surr = eff_ratio * adv_unsqueezed
        policy_loss = -torch.mean(surr)

        clip_ratio = (low_mask | high_mask).float().mean().item()
        return policy_loss, clip_ratio
    # ===== 新增：CISPO（MiniMax-M1）IS-weight clip（非 PPO 的 token clip/mask）=====
    # 论文形式： r_hat = clip(r, 1-eps_low^IS, 1+eps_high^IS), 目标 ~ sg(r_hat) * A * log pi
    # 这里在只有 ratio 的实现中，用 “forward=常数，backward=ratio” 的 trick 达到同样梯度：
    # eff_ratio = (r_hat.detach() / ratio.detach()) * ratio  =>  grad ~ r_hat.detach() * A * ∇logπ
    if clip_mode in ("cispo", "cispo_clip", "is_clip", "is_weight_clip"):
        eps_is_low = clip_params.get("eps_is_low", None)   # None 表示不做下界（等价 low≈0）
        eps_is_high = float(clip_params.get("eps_is_high", clip_params.get("clip_eps", CLIP_EPS)))

        if eps_is_high < 0:
            raise ValueError(f"eps_is_high must be >= 0, got {eps_is_high}")
        if eps_is_low is not None and float(eps_is_low) < 0:
            raise ValueError(f"eps_is_low must be >= 0, got {eps_is_low}")

        # 非对称区间： [1-eps_is_low, 1+eps_is_high]
        high = 1.0 + eps_is_high
        if eps_is_low is None:
            low = 0.0  # ratio>0，基本等价于“只裁上界”
        else:
            low = 1.0 - float(eps_is_low)
            low = max(0.0, low)

        # 数值稳定（可选）
        ratio_min = float(clip_params.get("ratio_min", 1e-8))
        ratio_max = float(clip_params.get("ratio_max", 1e8))
        r = ratio.clamp(ratio_min, ratio_max)

        # r_hat: forward 里就是被裁剪的 IS 权重
        r_hat = r.clamp(low, high)

        # “forward 常数 + backward 走 ratio”的 CISPO trick
        r_det = r.detach().clamp_min(ratio_min)
        eff_ratio = (r_hat.detach() / r_det) * r

        surr = eff_ratio * adv_unsqueezed
        policy_loss = -torch.mean(surr)

        # 这里的 clip_ratio 表示“有多少样本的 IS 权重被截到了边界”（并不代表无梯度）
        is_clipped = (r < low) | (r > high)
        clip_ratio = is_clipped.float().mean().item()
        return policy_loss, clip_ratio
    raise ValueError(f"Unsupported clip mode: {clip_mode}")

def make_policy_prob_diag_image(
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    max_points: int = 20000,
    title: str = "Old vs New Policy Probability",
) -> Tuple[np.ndarray | None, float]:
    """
    绘制旧/新策略在相同动作上的概率散点图。
    x 轴: p_old = exp(logp_old)
    y 轴: p_new = exp(logp)
    颜色: |p_new - p_old|
    """
    with torch.no_grad():
        po = p_old.detach().reshape(-1)
        pn = p_new.detach().reshape(-1)
        n = po.numel()
        if n == 0:
            return None, float("nan")
        if n > max_points:
            idx = torch.randperm(n, device=po.device)[:max_points]
            po = po[idx]
            pn = pn[idx]
        po_np = po.float().cpu().numpy()
        pn_np = pn.float().cpu().numpy()

    diff = np.abs(pn_np - po_np)
    corr = float(np.corrcoef(po_np, pn_np)[0, 1]) if len(po_np) >= 2 else float("nan")

    fig = plt.figure(figsize=(6.5, 5), dpi=160)
    ax = fig.add_subplot(111)
    sc = ax.scatter(po_np, pn_np, c=diff, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    ax.set_xlabel("Old Policy Probability")
    ax.set_ylabel("New Policy Probability")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="|p_new - p_old|")
    ax.legend(loc="upper left")
    ax.text(
        0.60, 0.05, f"Correlation: {corr:.6f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black")
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = mpimg.imread(buf)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_uint8, corr

def make_old_new_prob_scatter_image(
    p_old: torch.Tensor,
    p_new: torch.Tensor,
    max_points: int = 20000,
    title: str = "Old vs New Action Probability",
) -> Tuple[np.ndarray, float]:
    """
    输入: p_old/p_new 为 torch Tensor (任意 shape)，会 flatten。
    输出: (RGB uint8 图像数组, corr)
    """
    po = p_old.detach().float().reshape(-1).cpu().numpy()
    pn = p_new.detach().float().reshape(-1).cpu().numpy()

    n = min(len(po), len(pn))
    po, pn = po[:n], pn[:n]
    if n == 0:
        return None, float("nan")

    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        po, pn = po[idx], pn[idx]

    diff = np.abs(pn - po)
    corr = float(np.corrcoef(po, pn)[0, 1]) if len(po) >= 2 else float("nan")

    fig = plt.figure(figsize=(6.5, 5), dpi=160)
    ax = fig.add_subplot(111)
    sc = ax.scatter(po, pn, c=diff, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    ax.set_xlabel("Old Policy Probability")
    ax.set_ylabel("New Policy Probability")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="|p_new - p_old|")
    ax.legend(loc="upper left")
    ax.text(
        0.60, 0.05, f"Correlation: {corr:.6f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="black")
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    img = mpimg.imread(buf)  # float [0,1], shape [H,W,4] or [H,W,3]
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img_uint8, corr

# ================================================================
# 数据结构 更新经验数据结
# ================================================================
@dataclass
class Experience:
    obs: np.ndarray            # 状态向量 (state_dim,)
    action_token: np.ndarray   # 采样的离散动作 token (shape: [ACTION_DIM,])
    advantage: float = 0.0
    behaviour_logits: np.ndarray | None = None  # 行为策略的 logits (shape: [ACTION_DIM, VOCAB_SIZE])
    value_target: float = 0.0
    reward: float = 0.0           # r_t (已经 * REWARD_SCALE)
    discount: float = 1.0         # gamma * (1 - done_t)
    behaviour_logp: np.ndarray = None  # log μ(a_t | s_t), shape: [ACTION_DIM]
    # --- 新增：episode 级元信息（用于 Paired-Queue / bonus time）---
    episode_id: int = -1         # 同一 episode 的唯一标识  # (Part1) episode 元信息：用于 episode 拼接/配对/bonus
    t: int = -1                  # step index (从 0 开始)  # (Part1) step index：用于 divergence window 对齐
    policy_source: int = 0       # 0=train,1=best,2=perturbed_best


# ===== Paired-Queue 数据结构（v0：不聚类，Top-K success 参照）=====
@dataclass
class PairPack:
    fail_ep_id: int
    succ_ep_ids: List[int]
    # divergence window [window_start, window_end) in aligned time index
    window_start: int
    window_end: int
    divergence_t: int
    divergence_score: float

    # window 内数据
    states: np.ndarray          # [T, obs_dim]
    fail_actions: np.ndarray    # [T, ACTION_DIM] (int tokens)
    succ_actions: np.ndarray    # [K, T, ACTION_DIM] (int tokens)

    # 权重（可用于后续 Loss A 的 zone_weight）
    zone_weight: np.ndarray     # [T] float32

    # meta
    fail_return: float
    succ_returns: List[float]
    succ_cluster_ids: Optional[List[int]] = None  # v0 为空，v1 才用 prototypes


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
            "success_steps": deque(maxlen=window_size),
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
        success_step: Optional[int] = None,
        actor_id: Optional[int] = None,
        step_num: int = 0,
    ):
        env_stats = self.stats[env_name]
        env_stats["episode_returns"].append(ep_return)
        env_stats["step_times"].append(step_time)
        env_stats["episode_lengths"].append(ep_length)
        env_stats["successes"].append(success)
        env_stats["success_steps"].append(int(success_step) if (success_step is not None) else -1)
        env_stats["total_episodes_processed"] += 1
        env_stats["total_env_steps"] += ep_length
        self.total_samples_produced += ep_length
        if not env_name.startswith("eval_"):
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

        all_successes = []
        eval_successes = []
        all_success_steps = []
        eval_success_steps = []
        all_rps = []
        eval_rps = []
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
                eval_successes.extend(env_data["successes"])
                eval_success_steps.extend(env_data.get("success_steps", []))
                eval_rps.extend([(r / (l + 1e-8)) for r, l in zip(env_data["episode_returns"], env_data["episode_lengths"])])
            else:
                total_episodes_processed += env_data["total_episodes_processed"]
                total_env_steps += env_data["total_env_steps"]
                all_returns.extend(env_data["episode_returns"])
                all_lengths.extend(env_data["episode_lengths"])
                all_step_times.extend(env_data["step_times"])
                all_successes.extend(env_data["successes"])
                all_success_steps.extend(env_data.get("success_steps", []))
                all_rps.extend([(r / (l + 1e-8)) for r, l in zip(env_data["episode_returns"], env_data["episode_lengths"])])

        # 辅助：避免对空列表 np.mean([]) 报 warning
        def _safe_mean(xs: List[float], default: float = 0.0) -> float:
            return float(np.mean(xs)) if xs else float(default)

        per_env_stats["_global_rollout_"] = {
            "avg_return": np.mean(all_returns) if all_returns else 0.0,
            "avg_ep_len": np.mean(all_lengths) if all_lengths else 0.0,
            "avg_step_time": np.mean(all_step_times) if all_step_times else 0.0,
            "avg_success_rate": np.mean(all_successes) if 'all_successes' in locals() and len(all_successes) > 0 else 0.0,
            "avg_reward_per_step": _safe_mean(all_rps, 0.0),
            "avg_return_success": _safe_mean([r for r, s in zip(all_returns, all_successes) if s > 0.5], 0.0),
            "avg_return_fail": _safe_mean([r for r, s in zip(all_returns, all_successes) if s <= 0.5], 0.0),
            "avg_ep_len_success": _safe_mean([l for l, s in zip(all_lengths, all_successes) if s > 0.5], 0.0),
            "avg_ep_len_fail": _safe_mean([l for l, s in zip(all_lengths, all_successes) if s <= 0.5], 0.0),
            "avg_success_step": _safe_mean([t for t, s in zip(all_success_steps, all_successes) if s > 0.5 and int(t) >= 0], 0.0),
            "total_episodes_processed": total_episodes_processed,
            "total_env_steps": total_env_steps,
            "total_samples_produced": self.total_samples_produced,
            "active_actor_count": self.get_active_actor_count()
        }
        per_env_stats["_global_eval_"] = {
            "avg_return": np.mean(eval_returns) if eval_returns else 0.0,
            "avg_ep_len": np.mean(eval_lengths) if eval_lengths else 0.0,
            "avg_step_time": np.mean(eval_step_times) if eval_step_times else 0.0,
            "avg_success_rate": np.mean(eval_successes) if 'eval_successes' in locals() and len(eval_successes) > 0 else 0.0,
            "avg_reward_per_step": _safe_mean(eval_rps, 0.0),
            "avg_return_success": _safe_mean([r for r, s in zip(eval_returns, eval_successes) if s > 0.5], 0.0),
            "avg_return_fail": _safe_mean([r for r, s in zip(eval_returns, eval_successes) if s <= 0.5], 0.0),
            "avg_ep_len_success": _safe_mean([l for l, s in zip(eval_lengths, eval_successes) if s > 0.5], 0.0),
            "avg_ep_len_fail": _safe_mean([l for l, s in zip(eval_lengths, eval_successes) if s <= 0.5], 0.0),
            "avg_success_step": _safe_mean([t for t, s in zip(eval_success_steps, eval_successes) if s > 0.5 and int(t) >= 0], 0.0),
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
    def __init__(self, capacity=REPLAY_CAPACITY, seed=SEED):
        # 设置随机种子
        random.seed(seed + id(self) % 1000)  # 为每个buffer添加不同偏移
        np.random.seed(seed + id(self) % 1000)
        
        # 存的是 trajectory 列表，每个元素为 (traj: List[Experience], done: bool)
        self.buffer = deque(maxlen=capacity)

        # ====== 额外：episode 级缓存（用于 Paired-Queue / bonus time）======
        # 将被分段写入的 traj 重新拼回完整 episode
        self._episode_frags: Dict[int, List[Experience]] = {}
        self._episode_traj: Dict[int, List[Experience]] = {}
        self._episode_emb: Dict[int, np.ndarray] = {}
        self._episode_meta: Dict[int, Dict[str, float]] = {}
        self._episode_order = deque()  # episode_id FIFO
        self._episode_store_capacity = 10000
        self._success_eps = deque(maxlen=5000)
        self._failure_eps = deque(maxlen=5000)
        self._hard_success_count = 0
        self._elite_count = 0
        self._paired_queue = deque(maxlen=PAIRED_QUEUE_MAXLEN)  # (Part2) Paired-Queue：失败-成功配对样本队列

        # 防止同一个失败 episode 被重复挖矿（否则 success 到来后反复 remine 会抖动/浪费）
        self._pair_mined_fail_ids = set()

        self._paired_queue_stats = {
            'mined': 0,
            'skipped': 0,
            'skipped_no_pospool': 0,
            'skipped_short': 0,
            'skipped_invalid_window': 0,
            'last_fail_ep': -1,
            'window_len_sum': 0.0,
            'td_sum': 0.0,
        }

        # ====== Success-as-a-Set v1: prototypes / clusters（成功多模态集合建模）======
        self._proto_k = int(PROTO_K)
        self._prototypes = None  # np.ndarray [K, D], L2-normalized
        self._proto_counts = np.zeros((self._proto_k,), dtype=np.int64)
        self._proto_inited = False
        self._ep_cluster: Dict[int, int] = {}  # episode_id -> cluster_id（success episodes 才会写入）

        # pair mining 开销统计
        self._pair_mine_time_stats = {"total_ms": 0.0, "count": 0, "last_ms": 0.0}

        # Elite 自举 & policy source 统计
        self._elite_eps = deque(maxlen=5000)
        self._elite_scores = deque(maxlen=ELITE_WINDOW_N)
        self._policy_source_counts = defaultdict(int)  # 0 train / 1 best / 2 perturbed
        self._cluster_update_called = 0
        self._cluster_input_n = 0
        self._elite_new_added = 0


    def _maybe_mark_elite(self, episode_id: int, score: float, succ: float) -> bool:
        """判定是否将失败 episode 作为 elite 正例，用于冷启动。
        规则：在最近 ELITE_WINDOW_N 内取 top pct；可选 gap 限制；score>0。
        """
        if not ELITE_ENABLE:
            return False
        if succ > 0.5:
            return False
        self._elite_scores.append((episode_id, float(score)))
        scores = [s for _, s in self._elite_scores]
        if len(scores) < max(10, int(ELITE_WINDOW_N * 0.2)):
            return False
        scores_sorted = sorted(scores)
        idx_thr = max(0, int(len(scores_sorted) * (1 - ELITE_TOP_PCT)) - 1)
        thr = scores_sorted[idx_thr]
        p90 = scores_sorted[int(len(scores_sorted) * 0.9)]
        p50 = scores_sorted[int(len(scores_sorted) * 0.5)]
        gap_ok = (p90 - p50) >= ELITE_MIN_GAP
        if score >= thr and score > 0 and (ELITE_MIN_GAP <= 0 or gap_ok):
            self._elite_eps.append(episode_id)
            return True
        return False

    def add_trajectory(self, traj: List[Experience], done: bool, last_obs: np.ndarray,
                     episode_id: int = -1, episode_success: Optional[float] = None, episode_return: Optional[float] = None):  # (Part1) episode 元信息：用于 episode 拼接/配对/bonus
        # 训练用：依旧写入原 buffer（不改现有采样/训练逻辑）
        self.buffer.append((traj, done, last_obs))
        try:
            src = int(getattr(traj[0], "policy_source", 0))
            self._policy_source_counts[src] += 1
        except Exception:
            pass

        # bonus 用：拼回完整 episode（需要 episode_id/t 字段）
        if episode_id is None or int(episode_id) < 0:
            return

        episode_id = int(episode_id)
        frag = self._episode_frags.get(episode_id)
        if frag is None:
            frag = []
            self._episode_frags[episode_id] = frag
        frag.extend(traj)

        if not done:
            return

        # episode 结束：整理、做一个轻量 episode embedding（v0：mean state embedding）
        frag.sort(key=lambda e: e.t)
        try:
            ep_obs = np.stack([e.obs for e in frag], axis=0).astype(np.float32)
            emb = ep_obs.mean(axis=0)
        except Exception:
            # 极端情况下回退：用最后一个状态
            emb = np.array(frag[-1].obs, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        succ = float(episode_success) if episode_success is not None else 0.0
        ret = float(episode_return) if episode_return is not None else 0.0

        # 存储
        self._episode_traj[episode_id] = frag
        self._episode_emb[episode_id] = emb
        # 提取 value 近似（仅在奖励稀疏时 fallback 为 score）
        mean_v = float(np.mean([e.value_target for e in frag])) if len(frag) > 0 else 0.0
        self._episode_meta[episode_id] = {'success': succ, 'return': ret, 'len': float(len(frag)), 'cluster_id': -1, 'score_v': mean_v}

        # (Part4) Success-as-a-Set v1：为成功 episode 分配 cluster_id，并在线更新 prototypes
        elite = False
        if succ > 0.5:
            try:
                cid = int(self._assign_success_cluster(episode_id, emb))
                self._episode_meta[episode_id]['cluster_id'] = cid
            except Exception:
                self._episode_meta[episode_id]['cluster_id'] = -1
            self._episode_meta[episode_id]['elite'] = 0
            self._hard_success_count += 1
            self._cluster_input_n += len(frag)
        else:
            # Elite 自举：用 return 或 value 作为 score，填充正例池
            score = ret if ret > 0 else (mean_v if ELITE_USE_VALUE_FALLBACK else 0.0)
            elite = self._maybe_mark_elite(episode_id, score, succ)
            if elite:
                try:
                    cid = int(self._assign_success_cluster(episode_id, emb))
                    self._episode_meta[episode_id]['cluster_id'] = cid
                except Exception:
                    self._episode_meta[episode_id]['cluster_id'] = -1
                self._episode_meta[episode_id]['elite'] = 1
                self._elite_count += 1
                self._elite_new_added += 1
                self._cluster_input_n += len(frag)
            else:
                self._episode_meta[episode_id]['elite'] = 0
        self._episode_order.append(episode_id)
        if succ > 0.5 or elite:
            self._success_eps.append(episode_id)

            # ====== 必改2：补偿机制 ======
            # 成功出现后，回头从 failure pool 挑一些"最近失败"补挖 pair
            # 解决冷启动：前期失败太多/首次成功很晚 => fail 时挖不到 pair
            self._mine_pairs_from_failure_pool(max_attempts=16)

        if (succ <= 0.5) and (not elite):  # 失败且非 elite 的 episode 才进入失败池
            self._failure_eps.append(episode_id)
            # v0: 失败 episode 结束时尝试挖掘 pair pack（不影响主训练）
            try:
                self._mine_pairs_for_failure(int(episode_id))
            except Exception:
                self._paired_queue_stats['skipped'] += 1

        # 清理分段缓存
        if episode_id in self._episode_frags:
            del self._episode_frags[episode_id]

        # 驱逐旧 episode（控制内存）
        while len(self._episode_order) > self._episode_store_capacity:
            old = self._episode_order.popleft()
            self._episode_traj.pop(old, None)
            self._episode_emb.pop(old, None)
            self._episode_meta.pop(old, None)
    # ===========================
    # Success-as-a-Set v1: prototypes / clusters
    # ===========================
    def _ensure_prototypes(self, emb_dim: int):
        """Lazy init prototype buffers."""
        if self._prototypes is None:
            self._prototypes = np.zeros((self._proto_k, emb_dim), dtype=np.float32)

    def _assign_success_cluster(self, episode_id: int, emb: np.ndarray) -> int:
        """为成功 episode 分配 cluster，并用 EMA 更新对应 prototype（在线、轻量）。
        emb 必须已 L2-normalize。
        """
        self._cluster_update_called += 1
        episode_id = int(episode_id)
        emb = emb.astype(np.float32, copy=False)
        emb_dim = int(emb.shape[0])
        self._ensure_prototypes(emb_dim)

        # 1) init 阶段：前 K 个 success 直接填充 prototypes
        if not self._proto_inited:
            k = int(self._proto_counts.sum())
            if k < self._proto_k:
                cid = k
                self._prototypes[cid] = emb
                self._proto_counts[cid] += 1
                self._ep_cluster[episode_id] = int(cid)
                if int(self._proto_counts.sum()) >= int(PROTO_MIN_INIT):
                    self._proto_inited = True
                return int(cid)
            else:
                self._proto_inited = True

        # 2) 已初始化：cosine 最近原型分配
        protos = self._prototypes
        # 归一化（数值安全；init 阶段可能有 0 向量）
        norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
        protos_n = protos / norms
        sims = protos_n @ emb  # [K]
        cid = int(np.argmax(sims))

        # 3) EMA 更新 prototype 并重新归一化
        ema = float(PROTO_EMA)
        protos[cid] = ema * protos[cid] + (1.0 - ema) * emb
        protos[cid] = protos[cid] / (np.linalg.norm(protos[cid]) + 1e-8)
        self._proto_counts[cid] += 1
        self._ep_cluster[episode_id] = int(cid)
        return int(cid)

    def paired_cluster_coverage(self) -> int:
        """Paired-Queue 中 success cluster 覆盖数量（v1 触发 bonus 的条件之一）。"""
        if len(self._paired_queue) == 0:
            return 0
        clusters = set()
        for pack in self._paired_queue:
            cids = pack.get('succ_cluster_ids')
            if cids is None:
                continue
            for c in cids:
                if c is None:
                    continue
                ci = int(c)
                if ci >= 0:
                    clusters.add(ci)
        return int(len(clusters))

    def cluster_entropy(self) -> Dict[str, float]:
        """成功 episode 的簇分布熵/有效簇数（用于多模态覆盖监控）。"""
        from math import log, exp
        counts = defaultdict(int)
        for cid in self._ep_cluster.values():
            if cid is None or cid < 0:
                continue
            counts[int(cid)] += 1
        total = sum(counts.values())
        if total == 0:
            return {"entropy": float("nan"), "effective_k": float("nan"), "num_clusters": 0, "total_success": 0}
        ps = [c / total for c in counts.values()]
        ent = -sum(p * log(max(p, 1e-12)) for p in ps)
        eff_k = exp(ent)
        return {
            "entropy": float(ent),
            "effective_k": float(eff_k),
            "num_clusters": len(counts),
            "total_success": total,
        }

    def prototype_recent_coverage(self, last_n: int = 128) -> int:
        """最近 N 个成功 episode 覆盖的簇数（用于监控多模态覆盖）。"""
        if len(self._success_eps) == 0:
            return 0
        ids = list(self._success_eps)[-int(last_n):]
        clusters = set()
        for eid in ids:
            cid = self._episode_meta.get(int(eid), {}).get('cluster_id', -1)
            if cid is not None and int(cid) >= 0:
                clusters.add(int(cid))
        return int(len(clusters))

    def get_elite_stats(self) -> Dict[str, float]:
        scores = [s for _, s in self._elite_scores]
        top_score = max(scores) if scores else 0.0
        return {
            "elite_enabled": 1.0 if ELITE_ENABLE else 0.0,
            "elite_count": float(len(self._elite_eps)),
            "elite_top_score": float(top_score),
            "hard_success_count": float(self._hard_success_count),
            "total_positive": float(self._hard_success_count + self._elite_count),
        }

    def get_policy_source_stats(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self._policy_source_counts.items()}

    def get_success_stats(self) -> Dict[str, float]:
        """返回成功/elite 计数，用于日志与触发判断。"""
        return {
            "hard_success_count": float(self._hard_success_count),
            "elite_count": float(self._elite_count),
            "total_positive": float(self._hard_success_count + self._elite_count),
            "success_pool_size": float(len(self._success_eps)),
            "failure_pool_size": float(len(self._failure_eps)),
        }

    def get_positive_stats(self) -> Dict[str, float]:
        total_eps = max(1, len(self._episode_order))
        total_pos = float(self._hard_success_count + self._elite_count)
        return {
            "positive_total": total_pos,
            "positive_rate": total_pos / float(total_eps),
            "hard_success_count": float(self._hard_success_count),
            "elite_count": float(self._elite_count),
            "elite_buffer_size": float(len(self._elite_eps)),
            "elite_new_additions": float(self._elite_new_added),
            "cluster_update_called": float(self._cluster_update_called),
            "cluster_input_n": float(self._cluster_input_n),
        }

    def get_pair_mining_stats(self) -> Dict[str, float]:
        """返回配对挖掘的耗时/数量指标。"""
        c = max(1, int(self._pair_mine_time_stats["count"]))
        avg_ms = float(self._pair_mine_time_stats["total_ms"] / c)
        mined = float(self._paired_queue_stats.get("mined", 0))
        window_mean = float(self._paired_queue_stats["window_len_sum"] / max(1.0, mined))
        td_mean = float(self._paired_queue_stats["td_sum"] / max(1.0, mined))
        return {
            "avg_ms": avg_ms,
            "last_ms": float(self._pair_mine_time_stats["last_ms"]),
            "count": float(self._pair_mine_time_stats["count"]),
            "paired_size": float(len(self._paired_queue)),
            "paired_coverage": float(self.paired_cluster_coverage()),
            "mined": float(self._paired_queue_stats.get("mined", 0)),
            "skipped": float(self._paired_queue_stats.get("skipped", 0)),
            "skipped_no_pospool": float(self._paired_queue_stats.get("skipped_no_pospool", 0)),
            "skipped_short": float(self._paired_queue_stats.get("skipped_short", 0)),
            "skipped_invalid_window": float(self._paired_queue_stats.get("skipped_invalid_window", 0)),
            "success_pool_size": float(len(self._success_eps)),
            "failure_pool_size": float(len(self._failure_eps)),
            "proto_recent_coverage": float(self.prototype_recent_coverage()),
            "window_len_mean": window_mean,
            "divergence_t_mean": td_mean,
        }

# ===========================
    # Paired-Queue APIs (v0)
    # ===========================
    def paired_size(self) -> int:
        return len(self._paired_queue)
    
    def get_paired_stats(self) -> Dict[str, int]:
        return dict(self._paired_queue_stats)
    
    def sample_paired_packs(self, batch_size: int, consume: bool = True, diverse_clusters: bool = True) -> List[Dict[str, Any]]:
        """从 Paired-Queue 取出若干 pair packs（Ray 序列化友好：返回 dict 列表）。

        Args:
            batch_size: 取出 pack 数量
            consume: True 则会从队列弹出（推荐，避免反复对同一 pack 过拟合）；False 仅拷贝
            diverse_clusters: True 则优先选择来自不同 success cluster 的 packs（v1 防 mode collapse）
        """
        batch_size = int(batch_size)
        if batch_size <= 0 or len(self._paired_queue) == 0:
            return []

        n = min(batch_size, len(self._paired_queue))

        def pack_clusters(pack: Dict[str, Any]) -> set:
            cids = pack.get('succ_cluster_ids')
            if cids is None:
                return set()
            out = set()
            for c in cids:
                if c is None:
                    continue
                ci = int(c)
                if ci >= 0:
                    out.add(ci)
            return out

        if not diverse_clusters:
            if consume:
                return [self._paired_queue.popleft() for _ in range(n)]
            else:
                return list(self._paired_queue)[:n]

        # --- cluster-diverse sampling ---
        if not consume:
            packs = list(self._paired_queue)
            seen = set()
            chosen = []
            rest = []
            for p in packs:
                if len(chosen) >= n:
                    rest.append(p)
                    continue
                pc = pack_clusters(p)
                if len(pc - seen) > 0:
                    chosen.append(p)
                    seen |= pc
                else:
                    rest.append(p)
            # 补齐
            if len(chosen) < n:
                chosen.extend(rest[: (n - len(chosen))])
            return chosen[:n]

        # consume=True: 我们会临时弹出，按“先覆盖簇再补齐”的策略选择，然后把未选中的放回队列尾部
        seen = set()
        chosen = []
        stash = []
        total = len(self._paired_queue)
        for _ in range(total):
            p = self._paired_queue.popleft()
            if len(chosen) < n:
                pc = pack_clusters(p)
                if len(pc - seen) > 0:
                    chosen.append(p)
                    seen |= pc
                else:
                    stash.append(p)
            else:
                stash.append(p)

        # 补齐剩余
        if len(chosen) < n:
            need = n - len(chosen)
            chosen.extend(stash[:need])
            stash = stash[need:]

        # 未被消费的 pack 放回队列（保持 FIFO 近似）
        for p in stash:
            self._paired_queue.append(p)

        return chosen[:n]
    
    def _mine_pairs_from_failure_pool(self, max_attempts: int = 8, anchor_succ_id: Optional[int] = None):
        """当出现新的 success 后，从 failure pool 回头补挖一些 pair（偏向最近失败，可选：按 anchor 相似度排序）。

        兼容：episode_emb 缺失时自动调用 _compute_episode_embedding() 并缓存。
        统计：
          - 记录 anchor 排序是否真正改变候选顺序（reorder / avg rank change）
          - 记录 anchor 排序是否提高 backfill mined 成功率（仅在尝试 anchor 排序时分组：used vs fallback）
        """
        if max_attempts <= 0:
            return
        if len(self._success_eps) == 0:
            return
        if len(self._failure_eps) == 0:
            return

        s = self._paired_queue_stats

        # ---- init stats keys (safe) ----
        s.setdefault("backfill_tried", 0.0)
        s.setdefault("backfill_mined", 0.0)

        # anchor-sort diagnostics (cumulative)
        s.setdefault("backfill_anchor_sort_attempts", 0.0)       # 进入过 anchor-sort 分支的次数
        s.setdefault("backfill_anchor_sort_used", 0.0)           # 实际采用 anchor-sort 结果的次数（scored>0）
        s.setdefault("backfill_anchor_emb_missing", 0.0)         # anchor emb 不可得次数（无法排序）
        s.setdefault("backfill_anchor_scored_failures", 0.0)     # 累计参与打分的 failure 数
        s.setdefault("backfill_anchor_scored_events", 0.0)       # 累计发生 anchor 打分的事件数（用于算均值）
        s.setdefault("backfill_anchor_reorder_events", 0.0)      # 排序后顺序确实发生变化的事件数
        s.setdefault("backfill_anchor_rank_change_sum", 0.0)     # 平均 rank change 的累加
        s.setdefault("backfill_anchor_rank_change_events", 0.0)  # rank change 统计事件数

        # success-rate by group (cumulative) -- only meaningful when anchor-sort attempted
        s.setdefault("backfill_anchor_used_events", 0.0)
        s.setdefault("backfill_anchor_used_tried", 0.0)
        s.setdefault("backfill_anchor_used_mined", 0.0)
        s.setdefault("backfill_anchor_fallback_events", 0.0)
        s.setdefault("backfill_anchor_fallback_tried", 0.0)
        s.setdefault("backfill_anchor_fallback_mined", 0.0)

        # optional: when anchor-sort is disabled or no anchor provided
        s.setdefault("backfill_no_anchor_sort_events", 0.0)
        s.setdefault("backfill_no_anchor_sort_tried", 0.0)
        s.setdefault("backfill_no_anchor_sort_mined", 0.0)

        # 1) 最近失败优先（更可能与近期 success 共享 prefix）
        recent_k = max_attempts * int(PAIR_BACKFILL_RECENT_MULT)
        recent_fail_ids = list(self._failure_eps)[-recent_k:]
        orig_recent_fail_ids = [int(x) for x in recent_fail_ids]

        anchor_sort_attempted = (anchor_succ_id is not None and bool(PAIR_BACKFILL_USE_ANCHOR_SIM))

        # 2) 可选：用 anchor success 对 recent failures 再排序（提升挖到有效 divergence 的概率）
        used_anchor_sort = False
        scored_count = 0
        avg_rank_change = None
        reorder_changed = False

        if anchor_sort_attempted:
            s["backfill_anchor_sort_attempts"] += 1.0
            anchor_succ_id = int(anchor_succ_id)

            # anchor emb 缺失则现场算 + 缓存
            anchor_emb = self._episode_emb.get(anchor_succ_id, None)
            if anchor_emb is None:
                tr = self._episode_traj.get(anchor_succ_id, None)
                if tr is not None:
                    anchor_emb = self._compute_episode_embedding(
                        episode_id=anchor_succ_id,
                        traj=tr,
                        meta=self._episode_meta.get(anchor_succ_id),
                    )
                    if anchor_emb is not None:
                        self._episode_emb[anchor_succ_id] = anchor_emb

            if anchor_emb is None:
                s["backfill_anchor_emb_missing"] += 1.0
            else:
                scored = []
                for fid in recent_fail_ids:
                    fid = int(fid)

                    # 失败轨迹不存在（可能被清理/未落库），跳过
                    ftraj = self._episode_traj.get(fid, None)
                    if ftraj is None:
                        continue

                    # failure emb 缺失则现场算 + 缓存
                    femb = self._episode_emb.get(fid, None)
                    if femb is None:
                        femb = self._compute_episode_embedding(
                            episode_id=fid,
                            traj=ftraj,
                            meta=self._episode_meta.get(fid),
                        )
                        if femb is not None:
                            self._episode_emb[fid] = femb

                    if femb is None:
                        continue

                    scored.append((float(np.dot(femb, anchor_emb)), fid))

                scored_count = len(scored)
                s["backfill_anchor_scored_failures"] += float(scored_count)
                s["backfill_anchor_scored_events"] += 1.0

                if scored_count > 0:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    sorted_ids = [fid for _, fid in scored]

                    # 采用 anchor 排序结果
                    recent_fail_ids = sorted_ids
                    used_anchor_sort = True
                    s["backfill_anchor_sort_used"] += 1.0

                    # ---- diagnose reorder effect ----
                    min_len = min(len(orig_recent_fail_ids), len(sorted_ids))
                    reorder_changed = (orig_recent_fail_ids[:min_len] != sorted_ids[:min_len])
                    if reorder_changed:
                        s["backfill_anchor_reorder_events"] += 1.0

                    # avg |new_rank - old_rank|
                    old_rank = {fid: i for i, fid in enumerate(orig_recent_fail_ids)}
                    diffs = []
                    for new_i, fid in enumerate(sorted_ids):
                        if fid in old_rank:
                            diffs.append(abs(new_i - old_rank[fid]))
                    if len(diffs) > 0:
                        avg_rank_change = float(np.mean(diffs))
                        s["backfill_anchor_rank_change_sum"] += avg_rank_change
                        s["backfill_anchor_rank_change_events"] += 1.0

        tried = 0
        mined = 0

        # 3) 逐个失败挖 pair（最多 max_attempts 次）
        for fail_id in recent_fail_ids:
            fail_id = int(fail_id)
            if fail_id in self._pair_mined_fail_ids:
                continue

            # 没有轨迹就没法挖
            if fail_id not in self._episode_traj:
                continue

            before = int(s.get("mined", 0))
            self._mine_pairs_for_failure(fail_id, anchor_succ_id=anchor_succ_id)
            after = int(s.get("mined", 0))

            tried += 1
            if after > before:
                mined += 1

            if tried >= max_attempts:
                break

        # ---- overall backfill ----
        s["backfill_tried"] += float(tried)
        s["backfill_mined"] += float(mined)

        # ---- group-wise success rate ----
        if anchor_sort_attempted:
            if used_anchor_sort:
                s["backfill_anchor_used_events"] += 1.0
                s["backfill_anchor_used_tried"] += float(tried)
                s["backfill_anchor_used_mined"] += float(mined)
            else:
                s["backfill_anchor_fallback_events"] += 1.0
                s["backfill_anchor_fallback_tried"] += float(tried)
                s["backfill_anchor_fallback_mined"] += float(mined)
        else:
            s["backfill_no_anchor_sort_events"] += 1.0
            s["backfill_no_anchor_sort_tried"] += float(tried)
            s["backfill_no_anchor_sort_mined"] += float(mined)

        # ---- last snapshot (debug-friendly, non-cumulative) ----
        s["backfill_last_anchor_sort_attempted"] = float(1.0 if anchor_sort_attempted else 0.0)
        s["backfill_last_anchor_sort_used"] = float(1.0 if used_anchor_sort else 0.0)
        s["backfill_last_anchor_scored"] = float(scored_count)
        s["backfill_last_anchor_reordered"] = float(1.0 if reorder_changed else 0.0)
        s["backfill_last_anchor_avg_rank_change"] = float(avg_rank_change if avg_rank_change is not None else -1.0)
        s["backfill_last_tried"] = float(tried)
        s["backfill_last_mined"] = float(mined)

    def _mine_pairs_for_failure(self, fail_ep_id: int):  # (Part2) failure finalize 时挖掘 pair pack
        """失败 episode 结束时挖掘 pair pack（v0：Top-K success 参照，简化 divergence window）。
        该函数必须保证：任何异常都不会影响主训练（调用方有 try/except）。
        """
        # 防止重复挖矿
        if fail_ep_id in self._pair_mined_fail_ids:
            return

        _t0 = time.time()
        fail_ep_id = int(fail_ep_id)
        if fail_ep_id not in self._episode_emb or fail_ep_id not in self._episode_traj:
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_no_pospool'] += 1
            self._pair_mine_time_stats["last_ms"] = (time.time() - _t0) * 1000
            self._pair_mine_time_stats["total_ms"] += self._pair_mine_time_stats["last_ms"]
            self._pair_mine_time_stats["count"] += 1
            return
        if len(self._success_eps) < PAIR_MIN_SUCCESS_POOL:
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_no_pospool'] += 1
            self._pair_mine_time_stats["last_ms"] = (time.time() - _t0) * 1000
            self._pair_mine_time_stats["total_ms"] += self._pair_mine_time_stats["last_ms"]
            self._pair_mine_time_stats["count"] += 1
            return
    
        fail_emb = self._episode_emb[fail_ep_id]  # already L2-normalized
        # 1) Top-K success 检索（为效率可随机抽样候选）
        cand_ids = list(self._success_eps)
        if len(cand_ids) > PAIR_MAX_SUCC_CANDIDATES:
            cand_ids = random.sample(cand_ids, PAIR_MAX_SUCC_CANDIDATES)
    
        # cosine = dot (emb 已归一化)
        sims = []
        for sid in cand_ids:
            emb = self._episode_emb.get(int(sid))
            if emb is None: 
                continue
            sims.append((float(np.dot(fail_emb, emb)), int(sid)))
        if len(sims) == 0:
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_no_pospool'] += 1
            return
        sims.sort(key=lambda x: x[0], reverse=True)

        # (Part4) v1：Top-K success 选择时优先跨簇覆盖（success-as-a-set 多模态）
        succ_ids: List[int] = []
        succ_cids: List[int] = []
        seen_c = set()
        for _, sid in sims:
            if len(succ_ids) >= int(PAIR_TOPK_SUCCESS):
                break
            cid = int(self._episode_meta.get(int(sid), {}).get('cluster_id', -1))
            if bool(PAIR_DIVERSE_BY_CLUSTER) and cid >= 0:
                if cid in seen_c:
                    continue
                seen_c.add(cid)
            succ_ids.append(int(sid))
            succ_cids.append(int(cid))

        # 若跨簇筛选导致不足，回退补齐（允许重复簇）
        if len(succ_ids) < int(PAIR_TOPK_SUCCESS):
            for _, sid in sims:
                if len(succ_ids) >= int(PAIR_TOPK_SUCCESS):
                    break
                if int(sid) in succ_ids:
                    continue
                cid = int(self._episode_meta.get(int(sid), {}).get('cluster_id', -1))
                succ_ids.append(int(sid))
                succ_cids.append(int(cid))
    
        # 2) divergence window：用 Top-1 success 做对齐并找 divergence_t
        best_succ = succ_ids[0]
        ftraj = self._episode_traj[fail_ep_id]
        straj = self._episode_traj.get(best_succ)
        if straj is None:
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_no_pospool'] += 1
            return
    
        # 对齐到 min_len
        ftraj = sorted(ftraj, key=lambda e: e.t)
        straj = sorted(straj, key=lambda e: e.t)
        Tf, Ts = len(ftraj), len(straj)
        min_len = min(Tf, Ts)
        if min_len <= max(PAIR_MIN_PREFIX + 1, PAIRED_MIN_LEN):
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_short'] += 1
            return
    
        f_obs = np.stack([e.obs for e in ftraj[:min_len]], axis=0).astype(np.float32)
        s_obs = np.stack([e.obs for e in straj[:min_len]], axis=0).astype(np.float32)
    
        # normalize per-step for cosine
        f_norm = f_obs / (np.linalg.norm(f_obs, axis=1, keepdims=True) + 1e-8)
        s_norm = s_obs / (np.linalg.norm(s_obs, axis=1, keepdims=True) + 1e-8)
        cos = np.sum(f_norm * s_norm, axis=1)  # [min_len]
    
        # divergence: first below threshold after shared prefix; fallback to argmin after prefix
        td = None
        for t in range(PAIR_MIN_PREFIX, min_len):
            if cos[t] < PAIR_SIM_THRESH:
                td = t
                break
        if td is None:
            td = int(np.argmin(cos[PAIR_MIN_PREFIX:]) + PAIR_MIN_PREFIX)
    
        divergence_score = float(max(0.0, 1.0 - float(cos[td])))
    
        w = int(PAIR_DIVERGENCE_W)
        ws = max(0, td - w)
        we = min(min_len, td + w + 1)
        T = we - ws
        if T < PAIRED_MIN_LEN:
            self._paired_queue_stats['skipped'] += 1
            self._paired_queue_stats['skipped_invalid_window'] += 1
            return
    
        # 3) 构造 window 内的 states/actions
        states = f_obs[ws:we].astype(np.float32)                      # [T, obs_dim]
        fail_actions = np.stack([e.action_token for e in ftraj[ws:we]], axis=0).astype(np.int64)  # [T, ACTION_DIM]
    
        # success-as-a-set：为每个 succ 提供对齐窗口内动作（长度不足则跳过该 succ）
        succ_actions_list = []
        succ_returns = []
        succ_cluster_ids = []
        for sid in succ_ids:
            tr = self._episode_traj.get(int(sid))
            if tr is None:
                continue
            tr = sorted(tr, key=lambda e: e.t)
            if len(tr) < we:
                continue
            sa = np.stack([e.action_token for e in tr[ws:we]], axis=0).astype(np.int64)  # [T, ACTION_DIM]
            succ_actions_list.append(sa)
            succ_returns.append(float(self._episode_meta.get(int(sid), {}).get('return', 0.0)))
            succ_cluster_ids.append(int(self._episode_meta.get(int(sid), {}).get('cluster_id', -1)))
        if len(succ_actions_list) == 0:
            self._paired_queue_stats['skipped'] += 1
            self._pair_mine_time_stats["last_ms"] = (time.time() - _t0) * 1000
            self._pair_mine_time_stats["total_ms"] += self._pair_mine_time_stats["last_ms"]
            self._pair_mine_time_stats["count"] += 1
            return
        succ_actions = np.stack(succ_actions_list, axis=0)            # [K', T, ACTION_DIM]
    
        # 4) zone_weight：以 divergence 为中心的三角窗（可乘 divergence_score）
        center = td - ws
        idxs = np.arange(T, dtype=np.float32)
        tri = 1.0 - (np.abs(idxs - float(center)) / float(w + 1))
        tri = np.clip(tri, 0.0, 1.0).astype(np.float32)
        zone_weight = (tri * max(0.1, divergence_score)).astype(np.float32)
    
        pack = {
            'fail_ep_id': int(fail_ep_id),
            'succ_ep_ids': [int(x) for x in succ_ids],
            'window_start': int(ws),
            'window_end': int(we),
            'divergence_t': int(td),
            'divergence_score': float(divergence_score),
            'states': states,
            'fail_actions': fail_actions,
            'succ_actions': succ_actions,
            'zone_weight': zone_weight,
            'fail_return': float(self._episode_meta.get(fail_ep_id, {}).get('return', 0.0)),
            'succ_returns': [float(x) for x in succ_returns],
            'succ_cluster_ids': [int(x) for x in succ_cluster_ids],
        }
    
        self._paired_queue.append(pack)
        self._paired_queue_stats['mined'] += 1

        # 标记该失败 episode 已经成功产出过 pair（避免被 success 补偿机制反复 remine）
        self._pair_mined_fail_ids.add(fail_ep_id)

        self._paired_queue_stats['last_fail_ep'] = int(fail_ep_id)
        self._paired_queue_stats['window_len_sum'] += float(T)
        self._paired_queue_stats['td_sum'] += float(td)
        self._pair_mine_time_stats["last_ms"] = (time.time() - _t0) * 1000
        self._pair_mine_time_stats["total_ms"] += self._pair_mine_time_stats["last_ms"]
        self._pair_mine_time_stats["count"] += 1
    
    
    
    def size(self):
        # 返回总步数，保持语义近似
        return sum(len(traj) - 1 for traj, _, _ in self.buffer)
    

    def sample_sequences(self, min_total_steps: int):
        """
        采样若干条轨迹，使得它们的总步数 >= min_total_steps。
        """
        current_size = self.size()
        if current_size < min_total_steps:
            raise ValueError(f"Buffer total steps {current_size} < requested {min_total_steps}")
            
        # 随机打乱整个 buffer 的索引（或者简单地随机采样直到满足条件）
        # 为了效率，我们随机选择起始点，然后连续取，或者随机取索引。
        # 鉴于 buffer 是 deque，随机访问可能慢，但在 Ray Actor 中这通常是内存对象。
        # 这里采用随机抽样索引的方式。
        
        # 注意：self.buffer 是 deque，不支持切片或随机索引。先转为 list 引用会比较快，
        # 但如果 buffer 很大，每次转 list 开销大。
        # 优化方案：random.sample 直接从 deque 采一个较大的预估数量，然后截断？
        # 不，最准确的方法是维护一个 list 或者 accept 随机访问。
        # 鉴于 REPLAY_CAPACITY=50000，转 list 很快。
        
        buffer_list = list(self.buffer)
        # 随机打乱列表 (全量 shuffle 可能慢，不如随机选索引)
        indices = list(range(len(buffer_list)))
        random.shuffle(indices)
        
        traj_batch = []
        total_steps_collected = 0
        
        for idx in indices:
            traj, done, last_obs = buffer_list[idx]
            traj_batch.append((traj, done, last_obs))
            total_steps_collected += len(traj) - 1 # 最后一步不参与vtrace计算，bootstrap
            
            if total_steps_collected >= min_total_steps:
                break
        
        # 直接返回列表，不做 Padding 和 Stack，由 Trainer 自行处理
        # 为了传输效率，我们将同一字段的数据归拢到一起
        
        batch_obs_list, batch_act_list, batch_rew_list = [], [], []
        batch_disc_list, batch_logp_list = [], []
        batch_adv_list, batch_logits_list, batch_vtarg_list = [], [], []
        batch_done_list, batch_last_obs_list = [], []

        for traj, done, last_obs in traj_batch:
            # 提取基础数据
            batch_obs_list.append(np.stack([e.obs for e in traj]))
            batch_act_list.append(np.stack([e.action_token for e in traj]))
            batch_rew_list.append(np.array([e.reward for e in traj], dtype=np.float32))
            batch_disc_list.append(np.array([e.discount for e in traj], dtype=np.float32))
            batch_logp_list.append(np.stack([e.behaviour_logp for e in traj]))  # [T, ACTION_DIM]
            
            # 提取旧字段（兼容 PPO）
            batch_adv_list.append(np.array([e.advantage for e in traj], dtype=np.float32))
            batch_logits_list.append(np.stack([e.behaviour_logits for e in traj]))
            batch_vtarg_list.append(np.array([e.value_target for e in traj], dtype=np.float32))
            
            batch_done_list.append(float(done))
            batch_last_obs_list.append(last_obs)

        return (
            batch_obs_list,      # List[np.ndarray(T, D)]
            batch_act_list,
            batch_rew_list,
            batch_disc_list,
            batch_logp_list,
            batch_adv_list,
            batch_logits_list,
            batch_vtarg_list,
            np.array(batch_done_list, dtype=np.float32), # [B]
            np.stack(batch_last_obs_list)                # [B, D]
        )

# ================================================================
# 2. 经验回放与 Rollout
# ================================================================
class BaseWorkerActor:
    """rollout 和 eval worker 的共享逻辑。"""
    def __init__(self, infer, replay, wid, stats_actor, seed=SEED):
        self.infer = infer
        self.replay = replay
        self.stats_actor = stats_actor
        self.wid = wid
        
        # 设置每个worker的随机种子
        # 如果 wid 是字符串（如 "eval_0"），需要转换为整数
        if isinstance(wid, str):
            try:
                # 尝试从字符串中提取数字部分，如 "eval_0" -> 0
                wid_num = int(wid.split('_')[-1])
            except (ValueError, IndexError):
                # 如果无法提取，使用哈希值
                wid_num = hash(wid) % 10000
        else:
            wid_num = wid
            
        self.worker_seed = seed + wid_num * 1000
        random.seed(self.worker_seed)
        np.random.seed(self.worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.worker_seed)

        # 单任务设置：只初始化一个环境
        self.task_name = TASK_NAME  # 固定使用第一个（也是唯一的）任务
        print(f"BaseWorker {wid}: 正在初始化单任务 MetaWorld 环境: {self.task_name}")
        
        # MetaWorldWrapperDiscrete 可能不支持 seed 参数，需要检查其构造函数
        # 如果构造函数不支持 seed，我们需要在 reset 方法中设置
        try:
            # 尝试传入 seed 参数
            self.env = MetaWorldWrapperDiscrete(env_name=self.task_name, bins=N_ACTION_BINS, seed=self.worker_seed)
        except TypeError:
            # 如果构造函数不支持 seed 参数，创建一个没有 seed 的环境
            print(f"BaseWorker {wid}: MetaWorldWrapperDiscrete 不支持 seed 参数，将在 reset 时设置")
            self.env = MetaWorldWrapperDiscrete(env_name=self.task_name, bins=N_ACTION_BINS)
            # 存储种子，在第一次 reset 时使用
            self._pending_seed = self.worker_seed
        else:
            # 如果构造函数支持 seed 参数，清空 pending_seed
            self._pending_seed = None
            
        print(f"BaseWorker {wid}: 环境初始化完成。")

        self.task_description = "test: task_description"
        self.current_env_name = self.task_name
        self.episode_counter = 0  # 用于生成每个episode的唯一种子

    def _reset_and_select_env(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境，支持传入特定种子"""
        # 单任务设置：直接重置当前环境
        # 使用基础种子 + worker_id * 1000 + episode_counter
        if seed is None:
            episode_seed = self.worker_seed + self.episode_counter * 10000
        else:
            episode_seed = seed
        # 分配 episode_id，并递增计数器（与 seed 解耦，保证唯一性）
        ep_idx = self.episode_counter
        self.episode_counter += 1
        self.current_episode_id = (int(self.wid) << 32) + int(ep_idx)
            
        # 如果环境在初始化时没有设置种子，在 reset 时设置
        if self._pending_seed is not None:
            # 第一次 reset 时使用 pending_seed
            obs, info = self.env.reset(seed=self._pending_seed)
            self._pending_seed = None  # 只在第一次使用
        else:
            obs, info = self.env.reset(seed=episode_seed)
            
        return obs, info


@ray.remote
class RolloutWorkerActor(BaseWorkerActor):
    def __init__(self, infer, replay, wid, stats_actor):
        # wid 应该是整数，但为了安全起见，确保它是整数
        if isinstance(wid, str) and wid.startswith("rollout_"):
            try:
                wid = int(wid.split('_')[1])
            except (ValueError, IndexError):
                pass
                
        super().__init__(infer, replay, wid, stats_actor)
        # 单任务设置：移除环境选择的历史记录
        self.local_buffer = []
        self.policy_source = 0  # 0 train / 1 best / 2 perturbed_best
        self.perturb_active = False
        self.perturb_start_step = 0

    def run(self):
        try:
            debug_log(f"RolloutWorker {self.wid} run() 启动，等待环境重置")
            # 使用确定性种子
            obs, info = self._reset_and_select_env()
            reward_sum, time_start, step_count_total = 0.0, time.time(), 0
            while True:
                # 选择本 episode 的行为策略（train/best/perturbed best）
                if step_count_total == 0:
                    self.perturb_active = False
                    self.policy_source = 0
                    policy_mode = "train"
                    deterministic_flag = False
                    if random.random() < P_BEST_ROLLOUT:
                        policy_mode = "best"
                        deterministic_flag = True
                        self.policy_source = 1
                        if random.random() < P_PERTURB_BEST:
                            self.perturb_active = True
                            self.policy_source = 2
                            frac = random.uniform(PERTURB_START_FRAC_MIN, PERTURB_START_FRAC_MAX)
                            # 粗略设定：至少 10 步，避免过早扰动
                            self.perturb_start_step = max(1, int(frac * 50))
                    self.current_policy_mode = policy_mode
                    self.current_deterministic = deterministic_flag

                # 直接使用状态向量作为模型输入
                perturb_temp = PERTURB_TEMPERATURE if self.perturb_active else None
                action_env, action_token, logits, value, logp_mu = ray.get(
                    self.infer.request.remote(
                        obs, deterministic=self.current_deterministic, policy_mode=self.current_policy_mode,
                        perturb_temperature=perturb_temp if step_count_total >= self.perturb_start_step else None
                    )
                )
                # infer 可能返回只读的 numpy 视图，这里复制一份可写数组，便于扰动
                action_token = np.array(action_token, copy=True)

                # 若处于扰动阶段，对 token 做随机替换，重新计算 logp 与 action_env
                if self.perturb_active and step_count_total >= self.perturb_start_step:
                    logits_t = torch.tensor(logits)
                    probs = torch.softmax(logits_t, dim=-1)
                    for d in range(ACTION_DIM):
                        if random.random() < PERTURB_TOKEN_PROB:
                            action_token[d] = random.randrange(N_ACTION_BINS)
                    # 重新计算 logp_mu
                    logp_re = []
                    for d in range(ACTION_DIM):
                        logp_re.append(float(torch.log(probs[d, action_token[d]] + 1e-8)))
                    logp_mu = np.array(logp_re, dtype=np.float32)
                    # 重新映射 env action
                    continuous_action = -1.0 + 2.0 * action_token / (N_ACTION_BINS - 1)
                    action_env = continuous_action.astype(np.float32)

                # 修正: 传入 discrete token 给环境
                nxt, r, term, trunc, info = self.env.step(action_token)
                reward_sum += r
                chunk_reward = r * REWARD_SCALE
                step_count_total += 1
                done = term or trunc
                
                # 计算 discount
                step_discount = GAMMA * (0.0 if done else 1.0)

                # step index (从 0 开始) + episode_id（用于 Paired-Queue / bonus time）
                t_idx = step_count_total - 1
                self.local_buffer.append((obs, action_token, chunk_reward, logits, value, logp_mu, step_discount, t_idx, self.current_episode_id, self.policy_source))
                obs = nxt

                if done:
                    step_time = (time.time() - time_start) / max(step_count_total, 1)
                    success = float(info['success'])
                    self.stats_actor.add_episode_return.remote(
                        self.current_env_name,
                        reward_sum,
                        step_time,
                        step_count_total,
                        success,
                        success_step=(step_count_total if success > 0.5 else None),
                        actor_id=self.wid,
                        step_num=step_count_total,
                    )
                    ep_return = float(reward_sum)
                    reward_sum = 0.0
                    if self.local_buffer: 
                        bootstrap_val = 0.0  # episode 结束时 bootstrap value 为 0
                        self._process_traj(self.local_buffer, done=True, bootstrap_val=bootstrap_val, last_obs=obs,
                                         episode_id=self.current_episode_id, episode_success=success, episode_return=ep_return, policy_source=self.policy_source)
                    self.local_buffer.clear()
                    # 使用确定性种子重置
                    obs, info = self._reset_and_select_env()
                    time_start, step_count_total = time.time(), 0
                elif len(self.local_buffer) == ROLLOUT_LOCAL_BUF + 1:
                    # local_buffer 元素现包含 10 项（末尾多 policy_source），展开时也要对齐，避免解包报错导致 rollout 崩溃
                    last_state, _, _, _, last_value, _, _, _, _, _ = self.local_buffer[-1]
                    last_obs = last_state
                    bootstrap_val = last_value  # 使用最后一个状态的 value 作为 bootstrap
                    self._process_traj(self.local_buffer[:-1], done=False, bootstrap_val=bootstrap_val, last_obs=last_obs,
                                         episode_id=self.current_episode_id, episode_success=None, episode_return=None, policy_source=self.policy_source)
                    self.local_buffer = [self.local_buffer[-1]]
        except Exception as e: 
            import traceback
            print(f"[ERROR] RolloutWorker {self.wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise

    def _process_traj(self, traj_segment, done: bool, bootstrap_val: float, last_obs: np.ndarray,
                     episode_id: int, episode_success: Optional[float] = None, episode_return: Optional[float] = None,
                     policy_source: int = 0):
        rets, advs = [], []
        gae = 0.0
        for i in reversed(range(len(traj_segment))):
            _, _, r, _, v, _, _, _, _, _ = traj_segment[i]
            nv = bootstrap_val if i == len(traj_segment) - 1 else traj_segment[i+1][4]
            delta = r + GAMMA * nv - v
            gae = delta + GAMMA * LAMBDA * gae
            advs.append(gae)
            rets.append(gae + v)
        advs.reverse(); rets.reverse()
        advs_np = np.array(advs, dtype=np.float32)

        traj: List[Experience] = []
        for i, (s, a_token, r_val, logits, _v_unused, logp_mu, step_discount, t_idx, ep_id, _policy_source) in enumerate(traj_segment):
            traj.append(
                Experience(
                    obs=s,  # 现在是 numpy 数组
                    action_token=a_token.astype(np.int64),
                    advantage=float(advs_np[i]),
                    behaviour_logits=logits.astype(np.float32),
                    value_target=float(rets[i]),
                    reward=float(r_val),
                    discount=float(step_discount),
                    behaviour_logp=logp_mu.astype(np.float32),  # [ACTION_DIM]
                    episode_id=int(ep_id),
                    t=int(t_idx),
                    policy_source=int(_policy_source),
                )
            )
        self.replay.add_trajectory.remote(traj, done, last_obs, episode_id, episode_success, episode_return)


@ray.remote
class EvaluationWorkerActor(BaseWorkerActor):
    def __init__(self, infer, wid, stats_actor):
        # 确保 wid 是整数，eval_workers 创建时传入的是字符串 "eval_{i}"
        # 我们需要将字符串转换为整数用于种子计算
        if isinstance(wid, str) and wid.startswith("eval_"):
            try:
                # 提取数字部分，如 "eval_0" -> 0
                wid_num = int(wid.split('_')[1])
                # 使用提取的数字作为 wid 传递给父类
                super().__init__(infer, None, wid_num, stats_actor)
                # 但保留原始 wid 字符串用于标识
                self.original_wid = wid
            except (ValueError, IndexError):
                # 如果提取失败，使用哈希值
                wid_num = hash(wid) % 10000
                super().__init__(infer, None, wid_num, stats_actor)
                self.original_wid = wid
        else:
            super().__init__(infer, None, wid, stats_actor)
            self.original_wid = wid
            
        print(f"EvaluationWorker {self.original_wid}: 环境初始化完成。")

    def run(self):
        try:
            # 使用确定性种子
            obs, info = self._reset_and_select_env()
            while True:
                reward_sum, time_start, step_count_total, done = 0.0, time.time(), 0, False
                while not done:
                    # 直接使用状态向量作为模型输入
                    action_env, action_token, _, _, _ = ray.get(self.infer.request.remote(obs, deterministic=True))

                    # 修正: 传入 discrete token，避免双重转换
                    obs, r, term, trunc, info = self.env.step(action_token)
                    reward_sum += r
                    step_count_total += 1
                    done = term or trunc

                step_time = (time.time() - time_start) / max(step_count_total, 1)
                success = float(info["success"])
                self.stats_actor.add_episode_return.remote(
                    f"eval_{self.current_env_name}",
                    reward_sum,
                    step_time,
                    step_count_total,
                    success,
                    success_step=(step_count_total if success > 0.5 else None),
                    actor_id=self.original_wid,  # 使用原始字符串ID
                    step_num=step_count_total,
                )
                # 使用确定性种子重置
                obs, info = self._reset_and_select_env()
        except Exception as e: 
            import traceback
            print(f"[ERROR] EvaluationWorker {self.original_wid} run() 崩溃: {e}", flush=True)
            traceback.print_exc()
            raise
# ================================================================
# 3. 推理器 (InferenceActor)
# ================================================================
@ray.remote(num_gpus=1)
class InferenceActor(InferenceActorCom):
    def __init__(self, actor_id, stats_actor, seed=SEED):
        super().__init__()
        self.actor_id = actor_id
        
        # 设置推理器的随机种子
        self.inference_seed = seed + actor_id * 100
        random.seed(self.inference_seed)
        np.random.seed(self.inference_seed)
        torch.manual_seed(self.inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.inference_seed)
            torch.cuda.manual_seed_all(self.inference_seed)
        
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

    def load_best_weights(self, state_dict: Dict[str, torch.Tensor]):
        """加载最佳/EMA 策略权重，用于提升成功率。"""
        if state_dict is None or len(state_dict) == 0:
            return
        if not hasattr(self, "model_best") or self.model_best is None:
            self.model_best = MLPActorCriticDiscrete(
                torch_dtype=TORCH_DTYPE,
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                n_action_bins=N_ACTION_BINS
            ).cuda()
            self.model_best.eval()
        self.model_best.load_state_dict(state_dict)

    def _on_bg_task_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            import traceback
            print(f"[ERROR] InferenceActor {self.actor_id} 后台任务异常: {e}", flush=True)
            traceback.print_exc()

    async def request(self, obs: np.ndarray, deterministic: bool = False, policy_mode: str = "train", perturb_temperature: Optional[float] = None):
        """输入: numpy 数组状态向量，输出: 连续动作数组
        policy_mode: train / best
        perturb_temperature: 若提供则在采样时应用温度（用于扰动 best）
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self.requests.append((obs, deterministic, policy_mode, perturb_temperature))
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
            policy_modes = [r[2] for r in requests_to_process]
            perturb_temps = [r[3] for r in requests_to_process]
            t_loop_start = time.time()
            try:
                # 将观测列表转换为批次输入
                inputs_batch = self.model.prepare_inputs_batch(obs_list)

                with torch.inference_mode():
                    # 选择策略：训练/最佳（EMA）
                    model_use = self.model
                    # 提前构建 best logits 需要逐请求区分，这里先用 train，再在 post_process 前按需调整
                    action_logits_train, value_train = model_use(inputs_batch)
                    action_logits_best, value_best = (None, None)
                    if any(pm != "train" for pm in policy_modes) and hasattr(self, "model_best") and self.model_best is not None:
                        action_logits_best, value_best = self.model_best(inputs_batch)

                    # 后处理以采样动作 tokens 和对应的离散动作
                    logits_use_list = []
                    values_use_list = []
                    for i, pm in enumerate(policy_modes):
                        if pm != "train" and action_logits_best is not None:
                            logits_use = action_logits_best[i:i+1]
                            val_use = value_best[i:i+1]
                        else:
                            logits_use = action_logits_train[i:i+1]
                            val_use = value_train[i:i+1]
                        temp = perturb_temps[i]
                        if temp is not None and temp > 1e-6:
                            logits_use = logits_use / float(temp)
                        logits_use_list.append(logits_use)
                        values_use_list.append(val_use if val_use is not None else value_train[i:i+1])
                    action_logits = torch.cat(logits_use_list, dim=0)
                    value = torch.cat(values_use_list, dim=0)

                    dist, action_tokens, discrete_actions = self.model.post_process(
                        action_logits, deterministic=deterministic_flags
                    )
                    
                    # 计算行为策略 log-prob (每个动作维度的 log-prob，不求和)
                    logp_mu = dist.log_prob(action_tokens)  # [B, ACTION_DIM]

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
                    logp_mu_np = logp_mu.cpu().numpy()

                for i in range(len(promises_to_process)):
                    promises_to_process[i].set_result((
                        actions_env[i],           # 连续环境动作
                        action_tokens[i],         # 离散动作 token
                        logits[i],                # 对应的 logits
                        values[i],                # 价值估计
                        logp_mu_np[i]             # 行为策略 log-prob
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
    def __init__(self, rank, world_size, replay_buffer, seed=SEED):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.replay_buffer = replay_buffer
        
        # 设置训练器的随机种子（每个rank不同的种子）
        self.trainer_seed = seed + rank * 100
        random.seed(self.trainer_seed)
        np.random.seed(self.trainer_seed)
        torch.manual_seed(self.trainer_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.trainer_seed)
            torch.cuda.manual_seed_all(self.trainer_seed)
        self.diag_every = int(os.environ.get("DIAG_EVERY_STEPS", DIAG_EVERY_STEPS))
        
        self.model = None
        self.optimizer = None
        self.base_model = None
        self.data_dtype = None
        self.next_ready_batch: Optional[Tuple] = None
        self.data_fetching_task = None
        self.super_batch_size = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
        self.global_step = 0
        self.last_bonus_step = -10**9
        self.ema_state = None

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

    @torch.no_grad()
    def vtrace_from_log_rhos(self, log_rhos, discounts, rewards, values, bootstrap_value,
                             clip_rho=1.0, clip_c=1.0, clip_pg_rho=1.0):
        """
        V-trace 计算函数
        
        Args:
            log_rhos: [T] - log(π(a|s) / μ(a|s))，单个序列
            discounts: [T] - γ * (1 - done)
            rewards: [T] - 奖励
            values: [T] - V(s) 当前策略的价值估计
            bootstrap_value: scalar - 序列末尾的状态价值 V(s_T)
        
        Returns:
            vs: [T] - value targets for critic
            pg_adv: [T] - policy gradient advantages
        """
        if values.ndim == 0:
            values = values.unsqueeze(0)
        if log_rhos.ndim == 0:
            log_rhos = log_rhos.unsqueeze(0)
        if discounts.ndim == 0:
            discounts = discounts.unsqueeze(0)
        if rewards.ndim == 0:
            rewards = rewards.unsqueeze(0)
        bootstrap_value = bootstrap_value.to(values.device).to(values.dtype)
        if values.numel() == 0:
            return values, values
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=clip_rho)
        cs = torch.clamp(rhos, max=clip_c)

        print(f"[Debug] values shape: {values.shape}")
        print(f"[Debug] bootstrap_value shape: {bootstrap_value.shape}")
        # values_tp1 = concat([v_{t+1} for t in 0..T-1], bootstrap_value)
        values_tp1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

        # deltas_t = clipped_rho_t * (r_t + γ_{t+1} * v_{t+1} - v_t)
        deltas = clipped_rhos * (rewards + discounts * values_tp1 - values)  # [T]

        # 从后往前计算 value targets
        acc = torch.zeros_like(bootstrap_value)  # scalar，初始化为 0
        vs_minus_v = []
        for t in reversed(range(values.shape[0])):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            vs_minus_v.append(acc)
        vs_minus_v = torch.stack(list(reversed(vs_minus_v)), dim=0)
        vs = vs_minus_v + values

        # 计算 policy gradient advantages
        vs_tp1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_rhos = torch.clamp(rhos, max=clip_pg_rho)
        pg_adv = pg_rhos * (rewards + discounts * vs_tp1 - values)

        return vs, pg_adv

    def setup_deepspeed_group(self, master_addr, master_port):
        print(f"[DS-setup][rank {self.rank}] 进入 setup_deepspeed_group")
        print(f"Trainer {self.rank}: 开始设置 DeepSpeed 环境变量...")
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = "0"
        print(f"Trainer {self.rank}: 环境变量设置完成 - RANK={self.rank}, WORLD_SIZE={self.world_size}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

        print(f"[DS-setup][rank {self.rank}] before init_distributed (backend=nccl)")
        print(f"Trainer {self.rank}: 初始化分布式后端 (nccl)...")
        deepspeed.init_distributed(dist_backend="nccl")
        print(f"[DS-setup][rank {self.rank}] after init_distributed")
        print(f"Trainer {self.rank}: 分布式后端初始化完成")

        print(f"[DS-setup][rank {self.rank}] before create model")
        print(f"Trainer {self.rank}: 正在加载 MLP ActorCritic...")
        model = MLPActorCriticDiscrete(
            torch_dtype=TORCH_DTYPE,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_action_bins=N_ACTION_BINS
        )
        self.base_model = model
        print(f"Trainer {self.rank}: MLP ActorCritic 模型创建完成")
        # 初始化 EMA 权重（用于最佳策略采样）
        self.ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

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

        print(f"[DS-setup][rank {self.rank}] before deepspeed.initialize (world_size={self.world_size}, rank={self.rank})")
        print(f"Trainer {self.rank}: 开始初始化 DeepSpeed...")
        try:
            # print(f"model: {model}")
            # print(f"ds_config: {ds_config}")
            # print(f"optimizer_params: {optimizer_params}")
            self.model, self.optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=optimizer_params)
            print(f"[DS-setup][rank {self.rank}] after deepspeed.initialize")
            print(f"TrainerActor Rank {self.rank}: DeepSpeed 训练组 (ZeRO-2) 初始化完成。")
        except Exception as e:
            print(f"Trainer {self.rank}: DeepSpeed 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[DS-setup][rank {self.rank}] before create data_fetching_task")
        self.data_fetching_task = asyncio.get_event_loop().create_task(self._data_fetching_loop())
        print(f"[DS-setup][rank {self.rank}] after create data_fetching_task")

        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {n_total:,}, 可训练参数量: {n_trainable:,}")
        print(f"[DS-setup][rank {self.rank}] setup_deepspeed_group done")

    def _update_ema(self):
        if self.ema_state is None:
            return
        decay = float(EMA_DECAY)
        with torch.no_grad():
            for name, param in self.base_model.state_dict().items():
                if name not in self.ema_state:
                    self.ema_state[name] = param.detach().clone().float()
                    continue
                self.ema_state[name].mul_(decay).add_(param.detach().float(), alpha=1.0 - decay)

    def get_ema_state_dict(self):
        if self.ema_state is None:
            return {}
        # 返回 CPU 上的权重，避免跨进程 GPU Tensor 传输造成阻塞/错误
        return {k: v.detach().cpu() for k, v in self.ema_state.items()}

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

                # V-trace 采样：由于结束轨迹不损失步数，截断轨迹只损失1步
                # 采样时只需要稍微多采样一些即可（考虑到部分轨迹可能是截断的）
                # 简化逻辑：直接采样 super_batch_size，如果不足会在训练循环中动态调整
                while await self.replay_buffer.size.remote() < self.super_batch_size:
                    print(f"Trainer {self.rank} (BG): 等待 ReplayBuffer 填充至 {self.super_batch_size}...")
                    await asyncio.sleep(3)

                t_sample_start = time.time()
                # 精确采样模式：采样 super_batch_size 步
                # ReplayBuffer 会自动凑够轨迹来满足这个要求。
                (obs_seq_list, act_seq_list, rew_seq_list, disc_seq_list, logp_seq_list, 
                 adv_seq_list, logits_seq_list, vtarg_seq_list, done_seq, last_obs_seq) = \
                    await self.replay_buffer.sample_sequences.remote(self.super_batch_size)
                sample_time = time.time() - t_sample_start

                t_prep_start = time.time()
                
                # 1. Flatten for GAE PPO (Compatibility Mode)
                # 直接将所有轨迹拼接起来，形成一个巨大的散乱 batch
                obs_flat = np.concatenate(obs_seq_list, axis=0)
                act_flat = np.concatenate(act_seq_list, axis=0)
                adv_flat = np.concatenate(adv_seq_list, axis=0)
                logits_flat = np.concatenate(logits_seq_list, axis=0)
                vtarg_flat = np.concatenate(vtarg_seq_list, axis=0)

                device = next(self.model.parameters()).device
                
                # Flat tensors for current PPO
                obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
                act_token_t = torch.tensor(act_flat, dtype=torch.long, device=device)
                adv_t = torch.tensor(adv_flat, dtype=torch.float32, device=device)
                logits_old_t = torch.tensor(logits_flat, dtype=torch.float32, device=device)
                v_targ_t = torch.tensor(vtarg_flat, dtype=torch.float32, device=device)
                
                # 2. Sequence Tensors for V-trace (Future)
                # 目前先不处理 Padding/Stacking，因为还没用到 V-trace。
                # 如果需要用，可以在这里做 Pad + Stack，或者留给训练循环做。
                # 为了避免不必要的开销，我们暂时只保留引用。
                
                prep_time = time.time() - t_prep_start

                self.next_ready_batch = {
                    # GAE 兼容数据（flatten）
                    'obs': obs_t,
                    'act_token': act_token_t,
                    'advantage': adv_t,
                    'logits_old': logits_old_t,
                    'value_target': v_targ_t,
                    
                    # V-trace 序列数据（列表形式，每个元素是一个序列）
                    'obs_seq_list': obs_seq_list,
                    'act_seq_list': act_seq_list,
                    'rew_seq_list': rew_seq_list,
                    'disc_seq_list': disc_seq_list,
                    'logp_mu_seq_list': logp_seq_list,
                    'logits_seq_list': logits_seq_list,  # 行为策略的 logits，用于 KL 散度计算
                    'done_seq': done_seq,
                    'last_obs_seq': last_obs_seq,
                    
                    'sample_time': sample_time,
                    'prep_time': prep_time
                }

            except Exception as e:
                print(f"Trainer {self.rank}: 数据采样失败: {e}。将在3秒后重试。")
                await asyncio.sleep(3)
                import traceback
                traceback.print_exc()# ===========================
# Bonus Time (Loss A) helpers
    # ===========================
    def _allreduce_mean_scalar(self, x: torch.Tensor) -> float:
        """在多卡下做全局均值（返回 python float）。"""
        if (not distributed.is_available()) or (not distributed.is_initialized()):
            return float(x.item())
        y = x.detach().clone()
        distributed.all_reduce(y, op=distributed.ReduceOp.SUM)
        y = y / float(self.world_size)
        return float(y.item())
    
    async def _should_run_bonus_globally(self) -> Tuple[bool, int, Dict[str, float]]:
        """避免多卡不同步：只有当所有 rank 都满足触发条件时才跑 bonus。
        reason_code 规范：
            0 = trigger gate 通过
            1 = pairs < th_pairs
            2 = cov < th_cov
            3 = cooldown
        """
        device = next(self.model.parameters()).device
        # 当前状态
        local_pairs = int(await self.replay_buffer.paired_size.remote())
        local_cov = int(await self.replay_buffer.paired_cluster_coverage.remote())
        th_pairs = int(BONUS_TRIGGER_MIN_PAIRS)
        th_cov = int(BONUS_TRIGGER_MIN_CLUSTERS)
        cooldown_left = int(BONUS_COOLDOWN_STEPS) - (self.global_step - self.last_bonus_step)
        cooldown_left = int(max(0, cooldown_left))

        # all-reduce 取全局最小，排查“某 rank 无数据导致永不触发”
        pairs_t = torch.tensor([local_pairs], device=device, dtype=torch.int64)
        cov_t = torch.tensor([local_cov], device=device, dtype=torch.int64)
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(pairs_t, op=distributed.ReduceOp.MIN)
            distributed.all_reduce(cov_t, op=distributed.ReduceOp.MIN)
        pairs_min_all = int(pairs_t.item())
        cov_min_all = int(cov_t.item())

        reason_code = 0
        if cooldown_left > 0:
            reason_code = 3
        elif pairs_min_all < th_pairs:
            reason_code = 1
        elif cov_min_all < th_cov:
            reason_code = 2

        can_run = (reason_code == 0)
        debug_info = {
            "local_pairs": float(local_pairs),
            "local_cov": float(local_cov),
            "pairs_min_allranks": float(pairs_min_all),
            "cov_min_allranks": float(cov_min_all),
            "th_pairs": float(th_pairs),
            "th_cov": float(th_cov),
            "cooldown_left": float(cooldown_left),
        }
        return can_run, reason_code, debug_info
    
    def _build_bonus_batch(self, packs: List[Dict[str, Any]], max_steps: int):
        """把若干 PairPack(dict) 展开成 step-level batch。
        输出：states [N, obs_dim], fail_actions [N, ACTION_DIM], pos_mask [N, ACTION_DIM, N_BINS], zone_w [N]
        """
        if len(packs) == 0:
            return None
    
        states_list = []
        fail_a_list = []
        pos_mask_list = []
        zone_list = []
    
        for p in packs:
            s = p['states']           # [T, obs_dim]
            a_fail = p['fail_actions']# [T, ACTION_DIM]
            a_succ = p['succ_actions']# [K, T, ACTION_DIM]
            zw = p['zone_weight']     # [T]
            if s is None or a_fail is None or a_succ is None:
                continue
            T = s.shape[0]
            K = a_succ.shape[0]
            # 生成正例集合 mask：每个 step、每个 action 维度上，标记 Top-K success 的 token union
            # pos_mask[t, d, token] = True if any succ has that token at (t,d)
            pos_mask = np.zeros((T, ACTION_DIM, N_ACTION_BINS), dtype=np.bool_)
            for k in range(K):
                for d in range(ACTION_DIM):
                    tok = a_succ[k, :, d]  # [T]
                    pos_mask[np.arange(T), d, tok] = True
    
            states_list.append(s)
            fail_a_list.append(a_fail)
            pos_mask_list.append(pos_mask)
            zone_list.append(zw)
    
        if len(states_list) == 0:
            return None
    
        states = np.concatenate(states_list, axis=0).astype(np.float32)         # [N, obs_dim]
        fail_actions = np.concatenate(fail_a_list, axis=0).astype(np.int64)     # [N, ACTION_DIM]
        pos_mask = np.concatenate(pos_mask_list, axis=0)                        # [N, ACTION_DIM, N_BINS]
        zone_w = np.concatenate(zone_list, axis=0).astype(np.float32)           # [N]
    
        # 如果太大：随机子采样（保证多卡步数一致：每个 rank 都裁到 max_steps）
        N = states.shape[0]
        if N > max_steps:
            idx = np.random.choice(N, size=max_steps, replace=False)
            states = states[idx]
            fail_actions = fail_actions[idx]
            pos_mask = pos_mask[idx]
            zone_w = zone_w[idx]
    
        device = next(self.model.parameters()).device
        states_t = torch.from_numpy(states).to(device=device, dtype=TORCH_DTYPE)
        fail_a_t = torch.from_numpy(fail_actions).to(device=device, dtype=torch.long)
        pos_mask_t = torch.from_numpy(pos_mask.astype(np.bool_)).to(device=device)
        zone_w_t = torch.from_numpy(zone_w).to(device=device, dtype=torch.float32)
        return states_t, fail_a_t, pos_mask_t, zone_w_t
    
    def _lossA_set_ranking(self, logits: torch.Tensor, fail_actions: torch.Tensor, pos_mask: torch.Tensor, zone_w: torch.Tensor):
        """Loss A：success-as-a-set ranking loss。
        logits: [N, ACTION_DIM, N_BINS]
        fail_actions: [N, ACTION_DIM]
        pos_mask: [N, ACTION_DIM, N_BINS] bool
        zone_w: [N]
        """
        # log_probs: [N, ACTION_DIM, N_BINS]
        log_probs = torch.log_softmax(logits, dim=-1)
    
        # negative: log π(a_fail|s) summed over action dims
        logp_neg = log_probs.gather(-1, fail_actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # [N]  # 负例：失败动作 logprob
    
        # positive set: sum_d logsumexp_{a in Pos_d(s)} log π_d(a|s)
        # if some dim has empty Pos set -> skip that dim by treating as -inf; we enforce at least one True in pos_mask via mining
        masked = torch.where(pos_mask, log_probs, torch.full_like(log_probs, -1e9))
        logp_pos_per_dim = torch.logsumexp(masked, dim=-1)  # [N, ACTION_DIM]  # 正例集合：log-sum-exp over positives (success-as-a-set)
        logp_pos = logp_pos_per_dim.sum(dim=-1)             # [N]  # 正例集合：log-sum-exp over positives (success-as-a-set)
    
        # ranking score
        tau = float(BONUS_TAU)
        score = (logp_pos - logp_neg) / max(1e-6, tau)  # 负例：失败动作 logprob  # 正例集合：log-sum-exp over positives (success-as-a-set)
        margin_mean = (logp_pos - logp_neg).mean()
    
        if BONUS_MARGIN and BONUS_MARGIN > 0.0:
            # margin ranking: max(0, m - (logp_pos - logp_neg))
            m = float(BONUS_MARGIN)
            loss = torch.relu(m - (logp_pos - logp_neg))  # 负例：失败动作 logprob  # 正例集合：log-sum-exp over positives (success-as-a-set)
        else:
            # logistic ranking: -log σ(score)
            loss = -torch.nn.functional.logsigmoid(score)
    
        # weighted mean
        w = torch.clamp(zone_w, min=0.0)
        loss = (loss * w).sum() / torch.clamp(w.sum(), min=1.0)
        return loss, margin_mean
    
    def _kl_categorical(self, ref_logits: torch.Tensor, cur_logits: torch.Tensor) -> torch.Tensor:
        """KL(ref || cur) for categorical per action-dim; returns scalar mean over batch & dims."""
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        cur_logp = torch.log_softmax(cur_logits, dim=-1)
        ref_p = torch.softmax(ref_logits, dim=-1)
        kl = (ref_p * (ref_logp - cur_logp)).sum(dim=-1)  # [N, ACTION_DIM]
        return kl.mean()
    
    def _entropy_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        ent = -(p * logp).sum(dim=-1)  # [N, ACTION_DIM]
        return ent.mean()
    
    async def _run_bonus_time_if_ready(self) -> Dict[str, float]:  # (Part3) Bonus Time：从 Paired-Queue 取 pack 做 Loss A 更新
        """Bonus Time: 取 paired packs -> Loss A 更新若干次；带 KL/entropy guard，且多卡同步 early-stop。"""
        metrics: Dict[str, float] = {}
        can_run, reason_code, debug_info = await self._should_run_bonus_globally()

        # 基础状态指标（每次检查都写）
        pairs_now = debug_info.get("local_pairs", 0.0)
        cov_now = debug_info.get("local_cov", 0.0)
        pairs_min_all = debug_info.get("pairs_min_allranks", pairs_now)
        cov_min_all = debug_info.get("cov_min_allranks", cov_now)
        th_pairs = debug_info.get("th_pairs", float(BONUS_TRIGGER_MIN_PAIRS))
        th_cov = debug_info.get("th_cov", float(BONUS_TRIGGER_MIN_CLUSTERS))
        cooldown_left = debug_info.get("cooldown_left", float(max(0, int(BONUS_COOLDOWN_STEPS) - (self.global_step - self.last_bonus_step))))

        metrics['bonus/pairs_now'] = float(pairs_now)
        metrics['bonus/cov_now'] = float(cov_now)
        metrics['bonus/pairs_min_allranks'] = float(pairs_min_all)
        metrics['bonus/cov_min_allranks'] = float(cov_min_all)
        metrics['bonus/th_pairs'] = float(th_pairs)
        metrics['bonus/th_cov'] = float(th_cov)
        metrics['bonus/cooldown_left'] = float(cooldown_left)
        metrics['bonus/not_triggered_pairs_gap'] = float(max(0.0, th_pairs - pairs_now))
        metrics['bonus/not_triggered_cov_gap'] = float(max(0.0, th_cov - cov_now))
        metrics['bonus/reason_code'] = float(reason_code)

        if not can_run:
            metrics['bonus/ran'] = 0.0
            metrics['bonus/triggered'] = 0.0
            return metrics

        # 取 paired packs（consume 可配置；默认 True）
        packs = await self.replay_buffer.sample_paired_packs.remote(BONUS_BATCH_PACKS, BONUS_CONSUME_PACKS, True)
        batch = self._build_bonus_batch(packs, BONUS_MAX_STEPS)
        if batch is None:
            metrics['bonus/ran'] = 0.0
            metrics['bonus/triggered'] = 0.0
            metrics['bonus/reason_code'] = 4.0  # sample_empty
            return metrics
        states_t, fail_a_t, pos_mask_t, zone_w_t = batch
    
        # ref logits snapshot（用于 KL-guard）
        with torch.no_grad():
            ref_logits, _ = self.model(states_t)  # [N, ACTION_DIM, N_BINS]
            ref_logits = ref_logits.detach()
    
        ran_updates = 0
        last_loss = None
        stop_flag = False
        last_kl = None
        last_ent = None
        last_margin = None
    
        for u in range(int(BONUS_UPDATES)):
            # 前向
            cur_logits, _ = self.model(states_t)
    
            # Loss A
            lossA, margin_mean = self._lossA_set_ranking(cur_logits, fail_a_t, pos_mask_t, zone_w_t)
            loss = BONUS_LAMBDA * lossA
            last_margin = float(margin_mean.item())
    
            # guard metrics (global mean)
            with torch.no_grad():
                kl = self._kl_categorical(ref_logits, cur_logits)
                ent = self._entropy_categorical(cur_logits)
                g_kl = self._allreduce_mean_scalar(kl)
                g_ent = self._allreduce_mean_scalar(ent)
                last_kl = float(g_kl)
                last_ent = float(g_ent)
    
            # early stop decision must be synchronized across ranks
            local_continue = 1
            if g_kl > float(BONUS_KL_MAX) or g_ent < float(BONUS_ENTROPY_MIN):
                local_continue = 0
            if (not distributed.is_available()) or (not distributed.is_initialized()):
                global_continue = local_continue
            else:
                t = torch.tensor([local_continue], device=states_t.device, dtype=torch.int32)
                distributed.all_reduce(t, op=distributed.ReduceOp.MIN)
                global_continue = int(t.item())
    
            if global_continue == 0:
                stop_flag = True
                if u == 0:
                    # 第 0 步就触发 guard：直接不更新，避免主训练崩
                    metrics['bonus/guard_stop_immediate'] = 1.0
                break
    
            # update
            self.model.backward(loss)
            self.model.step()
            ran_updates += 1
            last_loss = float(lossA.item())
    
            if u % int(BONUS_LOG_EVERY) == 0 and self.rank == 0:
                print(f"[Bonus] update={u} lossA={last_loss:.4f} KL={g_kl:.4f} Ent={g_ent:.4f} N={states_t.shape[0]}")
    
        metrics['bonus/ran'] = 1.0
        metrics['bonus/triggered'] = 1.0
        metrics['bonus/reason_code'] = 0.0
        metrics['bonus/updates'] = float(ran_updates)
        if last_loss is not None:
            metrics['bonus/lossA'] = float(last_loss)
        if last_kl is not None:
            metrics['bonus/kl_ref'] = float(last_kl)
        if last_ent is not None:
            metrics['bonus/entropy'] = float(last_ent)
        if last_margin is not None:
            metrics['bonus/margin_mean'] = float(last_margin)
        metrics['bonus/stopped_by_guard'] = 1.0 if stop_flag else 0.0
        metrics['bonus/early_stop'] = 1.0 if stop_flag else 0.0
        metrics['bonus/paired_size_after'] = float(await self.replay_buffer.paired_size.remote())
        # 记录本次 bonus 的触发步，启用 cooldown
        self.last_bonus_step = int(self.global_step)
        return metrics
    
    
    
    async def run_training_epoch(self) -> Tuple[float, float, float, float, float, Dict[str, float], int, float, float, float, Dict[str, float]]:
        if self.next_ready_batch is None:
            print(f"Trainer {self.rank}: 等待超级批次...")
            while self.next_ready_batch is None:
                await asyncio.sleep(0.02)
            print(f"Trainer {self.rank}: 数据已收到，开始训练。")

        current_lrs = {}
        value_lr = self._get_current_lr(self.global_step, VALUE_LR, VALUE_WARMUP_STEPS, TRAIN_ITERS)
        policy_lr = self._get_current_lr(self.global_step, POLICY_LR, POLICY_WARMUP_STEPS, TRAIN_ITERS, start_step=POLICY_TRAIN_START_STEP)
        diag_eps = float(CLIP_PARAMS.get("clip_eps", CLIP_EPS)) if CLIP_PARAMS else CLIP_EPS
        diag_outside_clip_ratio = None
        diag_dead_grad_ratio = None
        diag_ess_norm = None
        diag_image = None
        diag_image_corr = None
        diag_every = self.diag_every
        do_diag = False

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'value': param_group['lr'] = value_lr; current_lrs['value'] = value_lr
            elif param_group['name'] == 'policy': param_group['lr'] = policy_lr; current_lrs['policy'] = policy_lr

        current_batch = self.next_ready_batch
        self.next_ready_batch = None

        policy_sample_time = current_batch['sample_time']
        policy_prep_time = current_batch['prep_time']

        # =================================================================
        # V-trace 计算：单独处理每个序列
        # =================================================================
        device = next(self.model.parameters()).device
        
        # 获取序列数据
        obs_seq_list = current_batch['obs_seq_list']
        act_seq_list = current_batch['act_seq_list']
        rew_seq_list = current_batch['rew_seq_list']
        disc_seq_list = current_batch['disc_seq_list']
        logp_mu_seq_list = current_batch['logp_mu_seq_list']
        logits_seq_list = current_batch['logits_seq_list']  # 行为策略的 logits
        done_seq = current_batch['done_seq']
        last_obs_seq = current_batch['last_obs_seq']

        # 对每个序列单独计算 V-trace（不 padding，单独处理）
        all_obs_flat, all_act_flat, all_pg_adv_flat, all_vs_flat, all_logits_old_flat = [], [], [], [], []
        rho_values, rho_clipped_values = [], []
        rho_gt_clip_counts, rho_counts = 0.0, 0.0
        
        for seq_idx in range(len(obs_seq_list)):
            # 获取当前序列数据
            obs_seq = torch.tensor(obs_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, state_dim]
            act_seq = torch.tensor(act_seq_list[seq_idx], dtype=torch.long, device=device)    # [T, ACTION_DIM]
            rew_seq = torch.tensor(rew_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T]
            disc_seq = torch.tensor(disc_seq_list[seq_idx], dtype=torch.float32, device=device) # [T]
            logp_mu_seq = torch.tensor(logp_mu_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, ACTION_DIM]
            
            T = obs_seq.shape[0]
            if T <= 1:
                continue
            
            # 用当前模型计算当前策略的输出（前 T-1 步）
            action_logits, value = self.model(obs_seq[:-1])  # logits: [T-1, ACTION_DIM, n_bins], value: [T-1, 1]
            value = value.squeeze(-1)  # [T-1]
            
            # 计算 bootstrap value（最后一个状态）
            # print(f"[Debug] done_seq shape: {done_seq.shape}")
            # print(f"[Debug] done_seq: {done_seq}")

            # print(f"[Debug] seq_idx: {seq_idx}")
            # raise ValueError("Debug")
            if done_seq[seq_idx] > 0.5: # done shape: [B, ]
                bootstrap_value = torch.tensor(0.0, device=device)
            else:
                _, bootstrap_value = self.model(obs_seq[-1:])  # [1, 1]
                bootstrap_value = bootstrap_value.squeeze(-1).squeeze(0)  # scalar
            
            # 计算当前策略的 log-prob（每个动作维度）
            dist = torch.distributions.Categorical(logits=action_logits)  # batch_shape [T-1, ACTION_DIM]
            logp_pi = dist.log_prob(act_seq[:-1])  # [T-1, ACTION_DIM]
            
            # 计算 log_rhos = log π(a|s) - log μ(a|s)
            # V-trace 需要标量的 log_rhos，所以对 ACTION_DIM 求和得到联合 log-prob
            log_rhos = logp_pi.sum(dim=-1) - logp_mu_seq[:-1].sum(dim=-1)  # [T-1]
            
            # V-trace 计算
            vs, pg_adv = self.vtrace_from_log_rhos(
                log_rhos,      # [T-1]
                disc_seq[:-1], # [T-1]
                rew_seq[:-1],  # [T-1]
                value,         # [T-1]
                bootstrap_value  # scalar
            )
            rhos = torch.exp(log_rhos.detach())
            rho_values.append(rhos)
            rho_clipped = torch.clamp(rhos, max=1.0)
            rho_clipped_values.append(rho_clipped)
            rho_gt_clip_counts += float((rhos > 1.0).float().sum().item())
            rho_counts += float(rhos.numel())
            
            # 收集结果（只收集前 T-1 步，因为 V-trace 需要 bootstrap）
            all_obs_flat.append(obs_seq[:-1])      # [T-1, state_dim]
            all_act_flat.append(act_seq[:-1])      # [T-1, ACTION_DIM]
            all_pg_adv_flat.append(pg_adv)         # [T-1]
            all_vs_flat.append(vs)                 # [T-1]
            # 保存行为策略的 logits，用于后续计算 KL 散度
            logits_old_seq = torch.tensor(logits_seq_list[seq_idx], dtype=torch.float32, device=device)  # [T, ACTION_DIM, n_bins]
            all_logits_old_flat.append(logits_old_seq[:-1])  # [T-1, ACTION_DIM, n_bins]
        
        # 拼接所有序列的结果
        obs_t = torch.cat(all_obs_flat, dim=0)     # [total_steps, state_dim]
        act_token_t = torch.cat(all_act_flat, dim=0)  # [total_steps, ACTION_DIM]
        pg_adv_t = torch.cat(all_pg_adv_flat, dim=0)  # [total_steps]
        vs_t = torch.cat(all_vs_flat, dim=0)       # [total_steps]
        logits_old_t = torch.cat(all_logits_old_flat, dim=0)  # [total_steps, ACTION_DIM, n_bins]

        # V-trace 重要性权重统计
        if rho_counts > 0:
            all_rho = torch.cat(rho_values, dim=0)
            all_rho_clipped = torch.cat(rho_clipped_values, dim=0)
            rho_gt1_rate = rho_gt_clip_counts / rho_counts
            rho_clipped_mean = float(all_rho_clipped.mean().item())
            rho_mean = float(all_rho.mean().item())
        else:
            rho_gt1_rate = 0.0
            rho_clipped_mean = 0.0
            rho_mean = 0.0
        
        # 调试信息：显示采样统计
        num_traj_sampled = len(obs_seq_list)
        total_steps_sampled = sum(len(obs_seq_list[i]) for i in range(num_traj_sampled))
        actual_usable_steps = obs_t.shape[0]  # V-trace 后实际可用步数
        steps_lost = total_steps_sampled - actual_usable_steps  # 损失的步数（应该等于轨迹数）
        if self.rank == 0:  # 只在 rank 0 打印，避免重复日志
            print(f"[V-trace采样统计] Rank {self.rank}: 采样了 {num_traj_sampled} 条轨迹，"
                  f"总步数 {total_steps_sampled}，实际可用步数 {actual_usable_steps}，"
                  f"损失步数 {steps_lost} (目标: {self.super_batch_size})")
        
        # =================================================================
        # 关键修正: 对 Flatten 后的序列数据进行 Shuffle
        # =================================================================
        current_batch_size = obs_t.shape[0]
        # 使用固定的随机种子进行shuffle以确保可复现性
        generator = torch.Generator(device=obs_t.device)
        generator.manual_seed(self.trainer_seed + self.global_step * 1000)
        indices = torch.randperm(current_batch_size, device=obs_t.device, generator=generator)
        
        obs_t = obs_t[indices]
        act_token_t = act_token_t[indices]
        pg_adv_t = pg_adv_t[indices]
        vs_t = vs_t[indices]
        logits_old_t = logits_old_t[indices]  # 也需要 shuffle logits_old_t

        # 全局 advantage 归一化（使用 V-trace 的 pg_adv）
        local_sum = pg_adv_t.sum()
        local_sq_sum = (pg_adv_t * pg_adv_t).sum()
        local_count = torch.tensor([pg_adv_t.numel()], device=pg_adv_t.device, dtype=torch.float32)

        distributed.all_reduce(local_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_sq_sum, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(local_count, op=distributed.ReduceOp.SUM)

        global_mean = local_sum / torch.clamp(local_count, min=1.0)
        global_var = torch.clamp(local_sq_sum / torch.clamp(local_count, min=1.0) - global_mean * global_mean, min=1e-12)
        global_std = torch.sqrt(global_var)

        epoch_losses, epoch_p_losses, epoch_v_losses, epoch_e_losses, epoch_kl_losses = [], [], [], [], []
        epoch_ent, epoch_kl_divs, epoch_clip_ratios = [], [], []

        # Update based on actual batch size (which may vary slightly due to trajectory lengths)
        current_batch_size = obs_t.shape[0]
        
        # V-trace 验证：实际可用步数 = 采样总步数 - 轨迹数量（每条轨迹损失1步用于bootstrap）
        # 如果不足，动态调整 update 次数（但至少保证能完成一次更新）
        if current_batch_size < self.super_batch_size:
            max_possible_updates = current_batch_size // TRAIN_BATCH_SIZE
            num_updates_in_epoch = max(1, min(ACCUMULATION_STEPS, max_possible_updates))
            print(f"[Warning] Rank {self.rank}: V-trace 后实际可用步数 {current_batch_size} < 目标 {self.super_batch_size}。"
                  f"将使用 {num_updates_in_epoch} 次更新（而不是 {ACCUMULATION_STEPS} 次）。")
        else:
            # 强制固定 update 次数，防止多卡死锁
            num_updates_in_epoch = ACCUMULATION_STEPS
        
        if current_batch_size < TRAIN_BATCH_SIZE:
            print(f"[Error] Rank {self.rank}: 数据严重不足! {current_batch_size} 步不足以进行一次更新（需要至少 {TRAIN_BATCH_SIZE} 步）。")
            # 返回零损失，避免崩溃
            current_lrs = {}
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'value': current_lrs['value'] = param_group['lr']
                elif param_group['name'] == 'policy': current_lrs['policy'] = param_group['lr']
            return 0.0, 0.0, 0.0, 0.0, 0.0, current_lrs, self.global_step, 0.0, 0.0, {}

        t_policy_train_start = time.time()

        for i in range(num_updates_in_epoch):
            start = i * TRAIN_BATCH_SIZE
            end = min(start + TRAIN_BATCH_SIZE, current_batch_size)  # 防止越界
            mini_obs = obs_t[start:end]
            mini_act_token = act_token_t[start:end]
            mini_pg_adv = pg_adv_t[start:end]
            mini_vs = vs_t[start:end]
            mini_logits_old = logits_old_t[start:end]  # [B, ACTION_DIM, n_bins]

            normalized_adv = (mini_pg_adv - global_mean) / (global_std + 1e-8)

            # 前向（重新计算，因为需要当前策略的输出）
            action_logits, value = self.model(mini_obs)
            value = value.to(torch.float32)

            # V-trace value loss: (value - vs)^2
            value_loss = VF_COEF * torch.mean((value - mini_vs) ** 2)

            if self.global_step < POLICY_TRAIN_START_STEP:
                loss = value_loss
                policy_loss = torch.tensor(0.0, device=loss.device)
                ent_loss = torch.tensor(0.0, device=loss.device)
                kl_loss = torch.tensor(0.0, device=loss.device)
                kl_div = 0.0
                ent = torch.tensor(0.0, device=loss.device)
                clip_ratio = 0.0  # 策略未开始训练时，clip ratio 为 0
            else:
                # 策略与熵损失 (离散版本)
                dist = torch.distributions.Categorical(logits=action_logits)
                logp = dist.log_prob(mini_act_token)  # [B, ACTION_DIM] - 每个动作维度的 log-prob

                with torch.no_grad():
                    dist_old = torch.distributions.Categorical(logits=mini_logits_old)
                    logp_old = dist_old.log_prob(mini_act_token)  # [B, ACTION_DIM]

                kl_div_tensor = kl.kl_divergence(dist_old, dist)  # [B, ACTION_DIM]
                kl_div = torch.mean(kl_div_tensor).item()
                kl_loss = KL_COEF * torch.mean(kl_div_tensor)

                ratio = torch.exp(logp - logp_old)  # [B, ACTION_DIM]
                adv_unsqueezed = normalized_adv.unsqueeze(dim=-1)  # [B, 1]
                if diag_outside_clip_ratio is None:
                    with torch.no_grad():
                        adv_expand = adv_unsqueezed.expand_as(ratio)
                        outside = ((ratio < 1 - diag_eps) | (ratio > 1 + diag_eps)).float().mean()
                        dead_grad = (((ratio > 1 + diag_eps) & (adv_expand > 0)) | ((ratio < 1 - diag_eps) & (adv_expand < 0))).float().mean()
                        w = ratio.clamp(1 - diag_eps, 1 + diag_eps)
                        ess = (w.sum() ** 2) / (w.pow(2).sum() + 1e-8)
                        diag_outside_clip_ratio = outside.item()
                        diag_dead_grad_ratio = dead_grad.item()
                        diag_ess_norm = (ess / w.numel()).item()
                        do_diag = (self.rank == 0) and (self.global_step >= POLICY_TRAIN_START_STEP) and (self.global_step % diag_every == 0)
                        if do_diag:
                            p_old = torch.exp(logp_old)
                            p_new = torch.exp(logp)
                            diag_image, diag_image_corr = make_policy_prob_diag_image(p_old, p_new, max_points=20000)
                policy_loss, clip_ratio = compute_policy_surrogate(
                    clip_mode=CLIP_MODE,
                    ratio=ratio,
                    adv_unsqueezed=adv_unsqueezed,
                    clip_params=CLIP_PARAMS,
                )          
                ent = torch.mean(dist.entropy())  # [B, ACTION_DIM] -> scalar (对所有元素求平均)
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
            epoch_clip_ratios.append(clip_ratio)
            if self.model.is_gradient_accumulation_boundary():
                self.global_step += 1
        # ===========================
        # Bonus Time (Loss A) stage
        # ===========================
        bonus_metrics = {}
        try:
            # 仅在 policy 已开始训练后才允许 bonus（避免 early instability）
            if self.global_step >= POLICY_TRAIN_START_STEP:
                bonus_metrics = await self._run_bonus_time_if_ready()  # (Part3) 在主 PPO 更新后插入 bonus 阶段
        except Exception as e:
            # bonus 不允许影响主训练稳定性
            bonus_metrics = {'bonus/ran': 0.0, 'bonus/error': 1.0, 'bonus/reason_code': 5.0}
            if self.rank == 0:
                print(f"[Bonus] skipped due to error: {e}")

        avg_loss = np.mean(epoch_losses)
        avg_p_loss = np.mean(epoch_p_losses)
        avg_v_loss = np.mean(epoch_v_losses)
        avg_e_loss = np.mean(epoch_e_losses)
        avg_kl_loss = np.mean(epoch_kl_losses)
        avg_ent = np.mean(epoch_ent)
        avg_kl_div = np.mean(epoch_kl_divs)
        avg_clip_ratio = np.mean(epoch_clip_ratios) if epoch_clip_ratios else 0.0

        perf_metrics = {
            "policy_sample_time": policy_sample_time,
            "policy_prep_time": policy_prep_time,
            "policy_train_time": time.time() - t_policy_train_start,
            "vtrace/rho_gt1_rate": float(rho_gt1_rate),
            "vtrace/rho_clipped_mean": float(rho_clipped_mean),
            "vtrace/rho_mean": float(rho_mean),
        }

        # append bonus metrics (if any)
        if isinstance(bonus_metrics, dict) and len(bonus_metrics) > 0:
            for k, v in bonus_metrics.items():
                try:
                    perf_metrics[k] = float(v)
                except Exception:
                    pass

        # ---- pair/backfill diagnostics (cheap scalars only) ----
        try:
            s = getattr(self, "_paired_queue_stats", {}) or {}
            attempts = float(s.get("backfill_anchor_sort_attempts", 0.0))
            used = float(s.get("backfill_anchor_sort_used", 0.0))
            scored_events = float(s.get("backfill_anchor_scored_events", 0.0))
            rankchg_events = float(s.get("backfill_anchor_rank_change_events", 0.0))
            reorder_events = float(s.get("backfill_anchor_reorder_events", 0.0))

            used_tried = float(s.get("backfill_anchor_used_tried", 0.0))
            used_mined = float(s.get("backfill_anchor_used_mined", 0.0))
            fb_tried = float(s.get("backfill_anchor_fallback_tried", 0.0))
            fb_mined = float(s.get("backfill_anchor_fallback_mined", 0.0))

            no_anchor_tried = float(s.get("backfill_no_anchor_sort_tried", 0.0))
            no_anchor_mined = float(s.get("backfill_no_anchor_sort_mined", 0.0))

            bf_tried = float(s.get("backfill_tried", 0.0))
            bf_mined = float(s.get("backfill_mined", 0.0))

            perf_metrics["pair/backfill_mined_rate"] = float(bf_mined / max(1.0, bf_tried))
            perf_metrics["pair/backfill_anchor_sort_used_rate"] = float(used / max(1.0, attempts))
            perf_metrics["pair/backfill_anchor_sort_reorder_rate"] = float(reorder_events / max(1.0, used))
            perf_metrics["pair/backfill_anchor_avg_scored_failures"] = float(s.get("backfill_anchor_scored_failures", 0.0) / max(1.0, scored_events))
            perf_metrics["pair/backfill_anchor_avg_rank_change"] = float(s.get("backfill_anchor_rank_change_sum", 0.0) / max(1.0, rankchg_events))

            # success rate comparison (only meaningful when anchor-sort attempted)
            perf_metrics["pair/backfill_used_mined_rate"] = float(used_mined / max(1.0, used_tried))
            perf_metrics["pair/backfill_fallback_mined_rate"] = float(fb_mined / max(1.0, fb_tried))
            perf_metrics["pair/backfill_no_anchor_sort_mined_rate"] = float(no_anchor_mined / max(1.0, no_anchor_tried))

            # last-event snapshots (debug)
            perf_metrics["pair/backfill_last_anchor_sort_attempted"] = float(s.get("backfill_last_anchor_sort_attempted", 0.0))
            perf_metrics["pair/backfill_last_anchor_sort_used"] = float(s.get("backfill_last_anchor_sort_used", 0.0))
            perf_metrics["pair/backfill_last_anchor_scored"] = float(s.get("backfill_last_anchor_scored", 0.0))
            perf_metrics["pair/backfill_last_anchor_reordered"] = float(s.get("backfill_last_anchor_reordered", 0.0))
            perf_metrics["pair/backfill_last_anchor_avg_rank_change"] = float(s.get("backfill_last_anchor_avg_rank_change", -1.0))
            perf_metrics["pair/backfill_last_tried"] = float(s.get("backfill_last_tried", 0.0))
            perf_metrics["pair/backfill_last_mined"] = float(s.get("backfill_last_mined", 0.0))

            # pool/queue sizes (useful)
            perf_metrics["pair/paired_queue_size"] = float(len(getattr(self, "_paired_queue", [])))
            perf_metrics["pair/success_pool_size"] = float(len(getattr(self, "_success_eps", [])))
            perf_metrics["pair/failure_pool_size"] = float(len(getattr(self, "_failure_eps", [])))
        except Exception:
            pass

        perf_metrics["diag_outside_clip_ratio"] = diag_outside_clip_ratio
        perf_metrics["diag_dead_grad_ratio"] = diag_dead_grad_ratio
        perf_metrics["diag_ess_norm"] = diag_ess_norm
        perf_metrics["diag_every_steps"] = self.diag_every
        if diag_image is not None:
            perf_metrics["diag_old_new_prob_image"] = diag_image
            perf_metrics["diag_old_new_prob_corr"] = diag_image_corr

        # 更新 EMA（最佳策略）
        self._update_ema()

        return avg_loss, avg_p_loss, avg_v_loss, avg_e_loss, avg_kl_loss, current_lrs, self.global_step, avg_ent, avg_kl_div, avg_clip_ratio, perf_metrics



# ================================================================
# 5. 主逻辑
# ================================================================
def main():
    args = parse_args()
    global TASK_NAME, BENCHMARK, NUM_TRAINER_GPUS, NUM_ROLLOUT_WORKERS, NUM_EVAL_WORKERS, TRAIN_BATCH_SIZE, SEED, CLIP_MODE, CLIP_PARAMS
    TASK_NAME = args.task_name
    BENCHMARK = "MetaWorld_" + TASK_NAME.replace("-", "_")
    debug_log("进入 main()")
    debug_log(f"收到命令行参数: {args}")
    debug_log(f"----------------------------------------------------")
    debug_log(f"目前训练metaworld任务为: {TASK_NAME}")
    debug_log(f"----------------------------------------------------")
    NUM_TRAINER_GPUS = args.num_trainer_gpus
    NUM_ROLLOUT_WORKERS = args.num_rollout_workers
    NUM_EVAL_WORKERS = args.num_eval_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    SEED = args.seed
    CLIP_MODE = args.clip_mode

    # 加载裁剪配置并应用对应超参
    clip_config = load_clip_config(args.clip_config)
    CLIP_PARAMS = select_clip_params(CLIP_MODE, clip_config)

    debug_log(f"[Clip] 使用模式: {CLIP_MODE}, 配置: {CLIP_PARAMS}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    debug_log("=== 开始 MetaWorld MLP PPO 训练脚本 ===")
    debug_log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    debug_log(f"TORCH_DISTRIBUTED_DEBUG: {os.environ.get('TORCH_DISTRIBUTED_DEBUG', 'Not set')}")
    
    # 设置全局随机种子 - 新增
    debug_log(f"设置全局随机种子: {SEED}")
    set_seed(SEED)

    # 设置调试环境变量
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用 IB，如果网络有问题

    # 检查 GPU 状态
    if torch.cuda.is_available():
        debug_log(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                debug_log(f"GPU {i}: {torch.cuda.get_device_name(i)}, 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
                torch.cuda.empty_cache()
            except Exception as e:
                debug_log(f"GPU {i} 检查失败: {e}")
    else:
        debug_log("CUDA 不可用")

    os.environ["RAY_DEDUP_LOGS"] = "0"
    debug_log("即将调用 ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm', include_dashboard=False)")
    _t_ray_init = time.time()
    ray.init(ignore_reinit_error=True, _temp_dir='/dev/shm', include_dashboard=False)
    debug_log(f"ray.init 完成，用时 {time.time() - _t_ray_init:.2f} 秒")
    try:
        debug_log(f"Ray 集群资源: {ray.cluster_resources()}")
        debug_log(f"Ray 当前可用资源: {ray.available_resources()}")
    except Exception as e:
        debug_log(f"获取 Ray 资源信息失败: {e}")

    log_dir = f"runs/MetaWorld/{BENCHMARK}/MLP_DS_PPO_{int(time.time())}_seed{SEED}"
    writer = SummaryWriter(log_dir)
    stats_actor = StatsActor.remote(window_size=MOVING_AVG_WINDOW)
    debug_log(f"TensorBoard 日志将保存在: {log_dir}")

    # 初始化 SwanLab
    swanlab_config = {
        "SEED": SEED,  # 新增：记录种子
        "TASK_NAME": TASK_NAME,
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
        "CLIP_MODE": CLIP_MODE,
        "CLIP_PARAMS": CLIP_PARAMS,
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
        "IS_SUCCESS_EARLY_TERMINATION": IS_SUCCESS_EARLY_TERMINATION,
        "BONUS_BATCH_PACKS": BONUS_BATCH_PACKS,
        "BONUS_COOLDOWN_STEPS": BONUS_COOLDOWN_STEPS,
    }
    swanlab.init(
        project="MetaWorld-PPO-Reuse-success-fail-traj-Benchmark",
        experiment_name=f"{BENCHMARK}_MLP_DS_PPO_add_vtrace_{CLIP_MODE}_{int(time.time())}_seed{SEED}",
        description=f"MetaWorld MLP DeepSpeed PPO Training with vtrace - {BENCHMARK} (Seed: {SEED})",
        config=swanlab_config,
    )
    # zzq1211 保存当前运行代码
    def save_and_log_code_to_swanlab(log_dir: str, seed: int = None):
        """
        保存当前运行的代码到指定目录并上传到SwanLab
        
        Args:
            log_dir: 日志目录路径
            seed: 随机种子，用于文件名
        """
        import os
        import shutil
        from datetime import datetime
        
        try:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 获取当前脚本的绝对路径
            current_script = os.path.abspath(__file__)
            script_name = os.path.basename(current_script)
            
            # 生成带时间戳和种子的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed_suffix = f"_seed{seed}" if seed is not None else ""
            code_txt_path = os.path.join(log_dir, f"code_{timestamp}{seed_suffix}.py")
            
            # 读取当前脚本内容
            with open(current_script, "r", encoding="utf-8") as f:
                code_content = f.read()
            
            # 写入到日志目录
            with open(code_txt_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            print(f"✅ 已保存源代码到: {code_txt_path}")
            
            # 尝试上传到SwanLab
            try:
                # 先检查swanlab是否已初始化
                import swanlab
                if hasattr(swanlab, 'run') and swanlab.run is not None:
                    # 创建一个更美观的代码文件格式
                    code_py_path = os.path.join(log_dir, f"run_code{seed_suffix}.py")
                    with open(code_py_path, "w", encoding="utf-8") as f:
                        # 添加文件头注释
                        f.write(f"# Code snapshot for experiment (Seed: {seed if seed else 'N/A'})\n")
                        f.write(f"# Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"# Original file: {script_name}\n")
                        f.write("#" * 80 + "\n\n")
                        f.write(code_content)
                    
                    # 上传到SwanLab
                    swanlab.log_artifact(
                        code_py_path,
                        name=f"source_code_seed{seed}" if seed else "source_code",
                        type="code"
                    )
                    print(f"✅ 源代码已作为artifact上传至SwanLab: {code_py_path}")
                    
                    # 可选：删除临时文件
                    os.remove(code_py_path)
                else:
                    print("⚠️  SwanLab未初始化，跳过artifact上传")
            except ImportError:
                print("⚠️  SwanLab未安装，跳过artifact上传")
            except Exception as e:
                print(f"⚠️  SwanLab上传失败: {e}")
                
        except FileNotFoundError:
            print(f"❌ 找不到脚本文件: {current_script}")
        except PermissionError:
            print(f"❌ 权限不足，无法写入目录: {log_dir}")
        except Exception as e:
            print(f"❌ 保存代码失败: {e}")
            import traceback
            traceback.print_exc()
    # ✅ 新增：保存并上传当前代码
    try:
        save_and_log_code_to_swanlab(log_dir, seed=SEED)
    except Exception as e:
        print(f"⚠️ 保存代码时出现警告: {e}")
    print("--- 步骤 1: 创建 Actors ---")
    replay_buffers = [ReplayBufferActor.remote(capacity=REPLAY_CAPACITY, seed=SEED) for _ in range(NUM_TRAINER_GPUS)]
    trainer_group = [
        TrainerActor.remote(rank=i, world_size=NUM_TRAINER_GPUS, replay_buffer=replay_buffers[i], seed=SEED)
        for i in range(NUM_TRAINER_GPUS)
    ]
    inference_pool = [InferenceActor.remote(actor_id=i, stats_actor=stats_actor, seed=SEED) for i in range(NUM_INFERENCE_ACTORS)]
    rollout_workers = [
        RolloutWorkerActor.remote(
            inference_pool[i % NUM_INFERENCE_ACTORS],
            replay_buffers[i % NUM_TRAINER_GPUS], i, stats_actor,
        ) for i in range(NUM_ROLLOUT_WORKERS)
    ]
    # eval_workers = [
    #     EvaluationWorkerActor.remote(
    #         inference_pool[i % NUM_INFERENCE_ACTORS], f"eval_{i}", stats_actor
    #     ) for i in range(NUM_EVAL_WORKERS)
    # ]
    eval_workers = [
    EvaluationWorkerActor.remote(
        inference_pool[i % NUM_INFERENCE_ACTORS], i, stats_actor  # 传递整数 i
    ) for i in range(NUM_EVAL_WORKERS)
]
    print(f"已创建 {NUM_ROLLOUT_WORKERS} 个 Rollout workers 和 {NUM_EVAL_WORKERS} 个 Evaluation workers。")

    print("\n--- 步骤 2: 建立独立的 DeepSpeed 训练组 ---")
    print("正在查找空闲端口...")
    train_group_port = find_free_port(base_port=29500)
    print(f"训练组端口: {train_group_port}")

    # 使用不同的 base_port 查找第二个端口，避免冲突
    # 如果 train_group_port 在 29500-29599 范围内，使用 29600 作为 base_port
    # 否则使用 train_group_port + 100 作为 base_port
    if 29500 <= train_group_port < 29600:
        broadcast_base_port = 29600
    else:
        broadcast_base_port = train_group_port + 100
    
    broadcast_group_port = find_free_port(base_port=broadcast_base_port)
    max_retries = 10
    try_times = 0
    while broadcast_group_port == train_group_port and try_times < max_retries:
        try_times += 1
        print(f"尝试 {try_times}/{max_retries} 次，广播组端口与训练组端口相同 ({broadcast_group_port})，重新查找")
        # 每次尝试使用不同的 base_port
        broadcast_base_port = broadcast_base_port + 100
        broadcast_group_port = find_free_port(base_port=broadcast_base_port)
    
    if broadcast_group_port == train_group_port:
        raise RuntimeError(f"无法找到与训练组端口不同的广播组端口（已重试 {max_retries} 次）。训练组端口: {train_group_port}")
    
    print(f"广播组端口: {broadcast_group_port}")

    print("获取训练器主节点地址...")
    trainer_master_addr = ray.get(trainer_group[0].get_node_ip.remote())
    print(f"训练器主节点地址: {trainer_master_addr}")

    print("开始初始化 DeepSpeed 训练组...")
    train_setup_tasks = [actor.setup_deepspeed_group.remote(trainer_master_addr, train_group_port) for actor in trainer_group]
    print(f"创建了 {len(train_setup_tasks)} 个训练器设置任务")

    try:
        print("等待 DeepSpeed 初始化完成...")
        ray.get(train_setup_tasks, timeout=300)  # 5分钟超时
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
    debug_log("广播签名校验通过，准备推送初始权重到推理器。")

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
    for idx, w in enumerate(rollout_workers):
        debug_log(f"启动 RolloutWorker {idx}")
        w.run.remote()
    for idx, w in enumerate(eval_workers):
        debug_log(f"启动 EvaluationWorker {idx}")
        w.run.remote()

    print("\n--- 步骤 5: 等待远程经验池填充初始数据 ---")
    min_buffer_size_for_start = TRAIN_BATCH_SIZE * ACCUMULATION_STEPS
    assert min_buffer_size_for_start < REPLAY_CAPACITY, "初始填充量必须小于回放池总容量"
    while not all(size >= min_buffer_size_for_start for size in ray.get([rb.size.remote() for rb in replay_buffers])):
        sizes = ray.get([rb.size.remote() for rb in replay_buffers])
        print(f"等待所有经验池填充初始数据 (目标: {min_buffer_size_for_start})... (当前大小: {sizes})")
        time.sleep(5)
        # 额外调试：周期性打印经验池大小，便于确认是否在正常采样
        try:
            debug_log(f"[调试] 当前经验池大小: {sizes}")
        except Exception:
            pass
    print("远程经验池已准备好，训练器将按需获取数据。")

    print("\n--- 步骤 6: 开始主训练与同步循环 ---")
    start_time = time.time()
    last_log_time = time.time()
    last_log_global_step = 0
    global_step = 0
    time_to_x = {0.6: None, 0.8: None, 0.9: None}
    while global_step < TRAIN_ITERS:
        t_train_start = time.time()
        train_tasks = [trainer.run_training_epoch.remote() for trainer in trainer_group]
        results = ray.get(train_tasks)
        _, _, _, _, _, _, global_step, _, _, _, _ = results[0]
        train_time = time.time() - t_train_start

        t_sync_start = time.time()
        broadcast_task = trainer_group[0].broadcast_weights.remote(BROADCAST_GROUP_NAME)
        receive_tasks = [inf.receive_and_update_weights.remote(BROADCAST_GROUP_NAME) for inf in inference_pool]
        ray.get([broadcast_task] + receive_tasks)
        sync_time = time.time() - t_sync_start
        # 额外广播 EMA/best 权重（用于 best rollout），仅由 trainer0 周期性广播
        if (
            P_BEST_ROLLOUT > 0
            and BEST_POLICY_MODE == "ema"
            and global_step > 0
            and global_step % EMA_BROADCAST_EVERY_UPDATES == 0
        ):
            try:
                ema_sd = ray.get(trainer_group[0].get_ema_state_dict.remote())
                load_tasks = [inf.load_best_weights.remote(ema_sd) for inf in inference_pool]
                ray.get(load_tasks)
            except Exception as e:
                print(f"[Warn] 广播 EMA 权重失败: {e}")

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
            rollout_success_rate = global_stats.get("avg_success_rate", float("nan"))
            eval_success_rate = eval_stats.get("avg_success_rate", float("nan"))

            # 额外指标获取（replay 侧）
            cluster_stats = ray.get(replay_buffers[0].cluster_entropy.remote())
            pair_stats = ray.get(replay_buffers[0].get_pair_mining_stats.remote())
            elite_stats = ray.get(replay_buffers[0].get_elite_stats.remote())
            policy_src_stats = ray.get(replay_buffers[0].get_policy_source_stats.remote())
            success_stats = ray.get(replay_buffers[0].get_success_stats.remote())
            positive_stats = ray.get(replay_buffers[0].get_positive_stats.remote())

            total_losses, p_losses, v_losses, e_losses, kl_losses, lrs_list, _, ents, avg_kl_divs, clip_ratios, perf_metrics_list = zip(*results)
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
            log_metrics['Metrics/Ineffective_Data_Ratio'] = np.mean(clip_ratios)  # 无效数据比例（梯度为 0 的样本比例，在 backward 后检查 ratio 的梯度）
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
            diag_outside = perf_metrics_list[0].get("diag_outside_clip_ratio", None)
            diag_dead = perf_metrics_list[0].get("diag_dead_grad_ratio", None)
            diag_ess = perf_metrics_list[0].get("diag_ess_norm", None)
            diag_every = perf_metrics_list[0].get("diag_every_steps", DIAG_EVERY_STEPS)
            if diag_outside is not None:
                log_metrics['Diag/Outside_Clip_Ratio'] = diag_outside
            if diag_dead is not None:
                log_metrics['Diag/Dead_Grad_Ratio'] = diag_dead
            if diag_ess is not None:
                log_metrics['Diag/ESS_Norm'] = diag_ess
            if diag_every is not None:
                log_metrics['Diag/Diag_Every_Steps'] = diag_every

            # --- New: success/fail split + reward-per-step (helps diagnose reward hacking) ---
            rollout_avg_rps = global_stats.get("avg_reward_per_step", float("nan"))
            rollout_return_succ = global_stats.get("avg_return_success", float("nan"))
            rollout_return_fail = global_stats.get("avg_return_fail", float("nan"))
            rollout_len_succ = global_stats.get("avg_ep_len_success", float("nan"))
            rollout_len_fail = global_stats.get("avg_ep_len_fail", float("nan"))
            rollout_success_step = global_stats.get("avg_success_step", float("nan"))

            eval_avg_rps = eval_stats.get("avg_reward_per_step", float("nan"))
            eval_return_succ = eval_stats.get("avg_return_success", float("nan"))
            eval_return_fail = eval_stats.get("avg_return_fail", float("nan"))
            eval_len_succ = eval_stats.get("avg_ep_len_success", float("nan"))
            eval_len_fail = eval_stats.get("avg_ep_len_fail", float("nan"))
            eval_success_step = eval_stats.get("avg_success_step", float("nan"))

            log_metrics['Rollout/Average_Return'] = avg_return
            log_metrics['Rollout/Average_Episode_Length'] = avg_ep_len
            log_metrics['Rollout/Success_Rate'] = rollout_success_rate
            log_metrics['Eval/Average_Return'] = eval_avg_return
            log_metrics['Eval/Average_Episode_Length'] = eval_avg_ep_len
            log_metrics['Eval/Success_Rate'] = eval_success_rate

            log_metrics['Eval/Reward_Per_Step'] = eval_avg_rps
            log_metrics['Eval/Return_Success_Mean'] = eval_return_succ
            log_metrics['Eval/Return_Fail_Mean'] = eval_return_fail
            log_metrics['Eval/Episode_Length_Success_Mean'] = eval_len_succ
            log_metrics['Eval/Episode_Length_Fail_Mean'] = eval_len_fail
            log_metrics['Eval/Success_Step_Mean'] = eval_success_step

            log_metrics['Rollout/Reward_Per_Step'] = rollout_avg_rps
            log_metrics['Rollout/Return_Success_Mean'] = rollout_return_succ
            log_metrics['Rollout/Return_Fail_Mean'] = rollout_return_fail
            log_metrics['Rollout/Episode_Length_Success_Mean'] = rollout_len_succ
            log_metrics['Rollout/Episode_Length_Fail_Mean'] = rollout_len_fail
            log_metrics['Rollout/Success_Step_Mean'] = rollout_success_step

            log_metrics['System/Replay_Buffer_Size_Total'] = total_buffer_size
            log_metrics['System/Total_Episodes_Processed'] = total_episodes
            log_metrics['System/Total_Env_Steps'] = total_env_steps
            log_metrics['System/Avg_Step_Time'] = avg_step_time
            log_metrics['System/Eval_Total_Episodes_Processed'] = eval_total_episodes
            log_metrics['System/Eval_Total_Env_Steps'] = eval_env_steps
            log_metrics['System/Eval_Avg_Step_Time'] = eval_avg_step_time
            log_metrics['System/Active_Rollout_Actors'] = global_stats.get("active_actor_count", 0)
            log_metrics['System/Total_Samples_Produced'] = global_stats.get("total_samples_produced", 0)

            # bonus/vtrace 直接转发
            pm0 = perf_metrics_list[0]
            for k, v in pm0.items():
                if isinstance(v, (int, float)) and (k.startswith("bonus/") or k.startswith("vtrace/") or k.startswith("pair/")):
                    log_metrics[k] = v
            # 友好别名
            if 'bonus/ran' in log_metrics:
                log_metrics['Bonus/triggered'] = log_metrics['bonus/ran']
            if 'bonus/lossA' in log_metrics:
                log_metrics['Bonus/lossA'] = log_metrics['bonus/lossA']
            if 'bonus/kl_ref' in log_metrics:
                log_metrics['Bonus/kl'] = log_metrics['bonus/kl_ref']
            if 'bonus/entropy' in log_metrics:
                log_metrics['Bonus/entropy'] = log_metrics['bonus/entropy']
            if 'bonus/stopped_by_guard' in log_metrics:
                log_metrics['Bonus/early_stop'] = log_metrics['bonus/stopped_by_guard']
            if 'bonus/margin_mean' in log_metrics:
                log_metrics['LossA/margin_mean'] = log_metrics['bonus/margin_mean']

            # cluster 覆盖/熵
            log_metrics['Cluster/Entropy'] = cluster_stats.get("entropy", 0.0)
            log_metrics['Cluster/Effective_K'] = cluster_stats.get("effective_k", 0.0)
            log_metrics['Cluster/Num_Clusters'] = cluster_stats.get("num_clusters", 0.0)
            log_metrics['Cluster/Total_Success'] = cluster_stats.get("total_success", 0.0)
            log_metrics['Cluster/Total_Positive'] = elite_stats.get("total_positive", 0.0)
            log_metrics['Cluster/Update_Called'] = positive_stats.get("cluster_update_called", 0.0)
            log_metrics['Cluster/Input_N'] = positive_stats.get("cluster_input_n", 0.0)
            log_metrics['HardSuccess/Count'] = success_stats.get("hard_success_count", 0.0)
            log_metrics['Elite/Count'] = success_stats.get("elite_count", 0.0)
            log_metrics['Elite/Buffer_Size'] = positive_stats.get("elite_buffer_size", 0.0)
            log_metrics['Elite/New_Additions'] = positive_stats.get("elite_new_additions", 0.0)
            log_metrics['Pos/Positive_Total'] = positive_stats.get("positive_total", 0.0)
            log_metrics['Pos/Positive_Rate'] = positive_stats.get("positive_rate", 0.0)
            log_metrics['Pos/Positive_Count_In_Buffer'] = positive_stats.get("positive_total", 0.0)

            # paired queue & mining
            log_metrics['Pair/Size'] = pair_stats.get("paired_size", 0.0)
            log_metrics['Pair/Coverage'] = pair_stats.get("paired_coverage", 0.0)
            log_metrics['Pair/Mined'] = pair_stats.get("mined", 0.0)
            log_metrics['Pair/Skipped'] = pair_stats.get("skipped", 0.0)
            log_metrics['Pair/Skipped_NoPosPool'] = pair_stats.get("skipped_no_pospool", 0.0)
            log_metrics['Pair/Skipped_Short'] = pair_stats.get("skipped_short", 0.0)
            log_metrics['Pair/Skipped_InvalidWindow'] = pair_stats.get("skipped_invalid_window", 0.0)
            log_metrics['Pair/Mine_Avg_ms'] = pair_stats.get("avg_ms", 0.0)
            log_metrics['Pair/Mine_Last_ms'] = pair_stats.get("last_ms", 0.0)
            log_metrics['Pair/Mine_Count'] = pair_stats.get("count", 0.0)
            log_metrics['PairedQueue/mined'] = pair_stats.get("mined", 0.0)
            log_metrics['PairedQueue/skipped'] = pair_stats.get("skipped", 0.0)
            log_metrics['PairedQueue/size'] = pair_stats.get("paired_size", 0.0)
            log_metrics['SuccessPool/size'] = pair_stats.get("success_pool_size", 0.0)
            log_metrics['FailurePool/size'] = pair_stats.get("failure_pool_size", 0.0)
            log_metrics['Prototypes/coverage_recent'] = pair_stats.get("proto_recent_coverage", 0.0)
            log_metrics['Prototypes/entropy'] = cluster_stats.get("entropy", 0.0)
            log_metrics['Divergence/window_len_mean'] = pair_stats.get("window_len_mean", 0.0)
            log_metrics['Divergence/t_d_mean'] = pair_stats.get("divergence_t_mean", 0.0)
            log_metrics['Elite/Enabled'] = elite_stats.get("elite_enabled", 0.0)
            log_metrics['Elite/Count'] = elite_stats.get("elite_count", 0.0)
            log_metrics['Elite/TopScore'] = elite_stats.get("elite_top_score", 0.0)
            log_metrics['HardSuccess/Count'] = float(len(replay_buffers)) * 0.0 + cluster_stats.get("total_success", 0.0)
            # 行为策略来源分布
            log_metrics['Rollout/PolicySource_Train'] = policy_src_stats.get("0", 0.0)
            log_metrics['Rollout/PolicySource_Best'] = policy_src_stats.get("1", 0.0)
            log_metrics['Rollout/PolicySource_PerturbedBest'] = policy_src_stats.get("2", 0.0)

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

            # Time-to-X（基于 eval 成功率均值）
            eval_success_list = [env_stats['avg_success_rate'] for env_name, env_stats in all_stats.items() if env_name.startswith("eval_")]
            mean_eval_success = float(np.mean(eval_success_list)) if len(eval_success_list) > 0 else 0.0
            for thr in (0.6, 0.8, 0.9):
                if time_to_x[thr] is None and mean_eval_success >= thr:
                    time_to_x[thr] = {
                        "steps": total_env_steps,
                        "seconds": time.time() - start_time
                    }
                if time_to_x[thr] is not None:
                    log_metrics[f'TimeToX/Success_{int(thr*100)}_Steps'] = time_to_x[thr]["steps"]
                    log_metrics[f'TimeToX/Success_{int(thr*100)}_Seconds'] = time_to_x[thr]["seconds"]

            for k, v in log_metrics.items():
                writer.add_scalar(k, v, global_step)
            swanlab.log(log_metrics, step=global_step)
            diag_img = perf_metrics_list[0].get("diag_old_new_prob_image", None)
            if diag_img is not None:
                diag_corr = perf_metrics_list[0].get("diag_old_new_prob_corr", float("nan"))
                try:
                    swanlab.log({
                        "Diag/Old_vs_New_Prob_Scatter": swanlab.Image(
                            diag_img,
                            caption=f"step={global_step}, corr={diag_corr:.4f}"
                        )
                    }, step=global_step)
                except Exception as e:
                    print(f"[Warn] SwanLab 图像记录失败: {e}")

            last_log_time = current_time
            last_log_global_step = global_step

    print(f"\n成功完成 {TRAIN_ITERS} 次训练与同步循环！")
    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
