#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/ds_metaworld_ppo_mlp_with_param_more_stats.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# --- Default Parameters for Single Run ---
DEFAULT_TASK_NAME="sweep-into-v3"
DEFAULT_CLIP_MODE="sapo_soft_clip"
DEFAULT_SEED=42
DEFAULT_CUDA_DEVICES="6,7"
DEFAULT_NUM_TRAINER_GPUS=1
DEFAULT_NUM_ROLLOUT_WORKERS=1
DEFAULT_NUM_EVAL_WORKERS=5
DEFAULT_TRAIN_BATCH_SIZE=512
DEFAULT_REPLAY_RECENT_FRAC=1.0
DEFAULT_REPLAY_MAX_VERSION_GAP="inf"

# --- Batch Experiment Configuration ---
# 根据你的注释区整理的任务列表
TASKS=(
  "sweep-into-v3"
  "drawer-open-v3"
  "door-open-v3"
  "button-press-topdown-v3"
  "handle-press-v3"
  "push-v3"
  "peg-insert-side-v3"
  "pick-place-v3"
  "plate-slide-v3"
  "coffee-button-v3"
)

# 根据你的注释区整理的裁剪模式列表
CLIP_MODES=(
  "clip"                # 对应注释中的 "ppo" (PPO standard hard clip)
  "soft_clip_alpha-1"   # 对应注释中的 "soft clip(alpha=1)"
  "soft_clip_alpha-2"   # 对应注释中的 "soft clip(alpha=2)"
  "sapo_soft_clip"      # 对应注释中的 "sapo"
)

# --- Core Training Function ---
run_training() {
  echo "===================================================================="
  echo "Starting training:"
  echo "  -> Task:                 ${TASK_NAME}"
  echo "  -> Clip Mode:            ${CLIP_MODE}"
  echo "  -> Seed:                 ${SEED}"
  echo "  -> CUDA:                 ${CUDA_VISIBLE_DEVICES}"
  echo "  -> Replay Recent Frac:   ${REPLAY_RECENT_FRAC}"
  echo "  -> Replay Max Version Gap: ${REPLAY_MAX_VERSION_GAP}"
  echo "===================================================================="

  # 显式导出 CUDA 可见设备，便于底层库读取
  export CUDA_VISIBLE_DEVICES

  # 执行 Python 训练脚本
  ${PYTHON_BIN} "${PYTHON_SCRIPT}" \
    --task_name "${TASK_NAME}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --num-trainer-gpus "${NUM_TRAINER_GPUS}" \
    --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
    --num-eval-workers "${NUM_EVAL_WORKERS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --seed "${SEED}" \
    --clip-mode "${CLIP_MODE}" \
    --clip-config "${CLIP_CONFIG}" \
    --replay-recent-frac "${REPLAY_RECENT_FRAC}" \
    --replay-max-version-gap "${REPLAY_MAX_VERSION_GAP}"
  
  echo "--- Finished training for Task: ${TASK_NAME}, Clip Mode: ${CLIP_MODE} ---"
  echo
}

# --- Main Script Logic ---
# 如果第一个参数是 "batch"，则运行批处理实验
if [[ "${1:-}" == "batch" ]]; then
  echo "===== Starting Batch Experiment Run ====="
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      # 为 run_training 函数设置环境变量
      # 允许从外部覆盖，否则使用默认值
      TASK_NAME="${task}"
      CLIP_MODE="${clip_mode}"
      SEED="${SEED:-${DEFAULT_SEED}}"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
      NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
      NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
      NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
      CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
      REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
      REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"

      run_training
    done
  done
  echo "===== Batch Experiment Run Finished ====="
else
  # 否则，像以前一样运行单次实验
  echo "===== Starting Single Experiment Run ====="
  # 使用环境变量或默认值
  TASK_NAME="${TASK_NAME:-${DEFAULT_TASK_NAME}}"
  CLIP_MODE="${CLIP_MODE:-${DEFAULT_CLIP_MODE}}"
  SEED="${SEED:-${DEFAULT_SEED}}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
  NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
  NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
  NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
  CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
  REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
  REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"

  run_training
fi