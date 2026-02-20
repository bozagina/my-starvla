#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="ds_metaworld_ppo_mlp_add_vtrace_try_new_v4_with_param.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# --- Default Parameters for Single Run ---
DEFAULT_TASK_NAME="sweep-into-v3"
DEFAULT_CLIP_MODE="clip"
DEFAULT_SEED=42
DEFAULT_CUDA_DEVICES="6,7"
DEFAULT_NUM_TRAINER_GPUS=1
DEFAULT_NUM_ROLLOUT_WORKERS=16
DEFAULT_NUM_EVAL_WORKERS=32
DEFAULT_TRAIN_BATCH_SIZE=512
DEFAULT_TRAIN_ITERS=100000
# 新增：采样过滤参数（用于 Recency Window Ablation）
DEFAULT_REPLAY_RECENT_FRAC=1.0        # 最新比例窗口：只从最新 f% 数据采样
DEFAULT_REPLAY_MAX_VERSION_GAP="inf"  # 版本差窗口：只采样 version_gap <= max_gap 的数据

# --- Structured Advantage & Credit Assignment Config ---
DEFAULT_ENABLE_OUTCOME_MODEL=0
DEFAULT_ENABLE_STRUCTURED_ADV=0
DEFAULT_ENABLE_CREDIT_UPDATE=0
DEFAULT_STRUCT_TOPK_K=8
DEFAULT_STRUCT_LAM_PAIR="1e-3"
DEFAULT_STRUCT_USE_GAUGE_LOSS=0
DEFAULT_STRUCT_LAM_GAUGE="1e-3"
DEFAULT_OUTCOME_LR="3e-5"
DEFAULT_ADV_LR="3e-5"
DEFAULT_OUTCOME_BATCH_SIZE=1024
DEFAULT_STRUCT_MIN_SUCCESS_EP=8
DEFAULT_STRUCT_MIN_FAILURE_EP=8

# --- 并行执行配置 ---
MAX_PARALLEL_JOBS=4  # 最大并行任务数（默认6个）
# 多 GPU 配置：每个并行任务使用的 GPU 组合
# 注意：多个任务可以共享同一组 GPU（前提是显存足够大）
# 格式：每行一个 GPU 配置（逗号分隔的 GPU ID）
GPU_CONFIGS=(
  "6,7"  # 第1个任务使用 GPU 6,7
  "6,7"  # 第2个任务使用 GPU 6,7
  "6,7"  # 第3个任务使用 GPU 6,7
  "6,7"  # 第4个任务使用 GPU 6,7
  # "6,7"  # 第5个任务使用 GPU 6,7
  # "6,7"  # 第6个任务使用 GPU 6,7
)

# 启动延迟（秒）：避免多个任务同时初始化导致资源竞争和端口冲突
# ⚠️ 重要：这个延迟必须足够长，让前一个任务完全启动并占用端口
STARTUP_DELAY=90  # 每个新任务启动前等待的秒数（推荐 15-25 秒）

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
  "soft_clip_alpha-0"   # 对应注释中的 "soft clip(alpha=0)"
  "sapo_soft_clip"      # 对应注释中的 "sapo"
  "log_gauss_clip"      # 对应注释中的 "log gauss clip"
  "soft_clip_alpha-0-5" # 对应注释中的 "soft clip(alpha=0.5)"
  "soft_clip_alpha-1"   # 对应注释中的 "soft clip(alpha=1)"
  "soft_clip_alpha-2"   # 对应注释中的 "soft clip(alpha=2)"

)

# --- Core Training Function ---
run_training() {
  local log_prefix="${JOB_PREFIX:-}"
  
  echo "===================================================================="
  echo "${log_prefix}Starting training:"
  echo "${log_prefix}  -> Task:                 ${TASK_NAME}"
  echo "${log_prefix}  -> Clip Mode:            ${CLIP_MODE}"
  echo "${log_prefix}  -> Seed:                 ${SEED}"
  echo "${log_prefix}  -> CUDA:                 ${CUDA_VISIBLE_DEVICES}"
  echo "${log_prefix}  -> Replay Recent Frac:   ${REPLAY_RECENT_FRAC}"
  echo "${log_prefix}  -> Replay Max Version Gap: ${REPLAY_MAX_VERSION_GAP}"
  echo "${log_prefix}  -> Outcome Model:        ${ENABLE_OUTCOME_MODEL}"
  echo "${log_prefix}  -> Structured Adv:       ${ENABLE_STRUCTURED_ADV}"
  echo "${log_prefix}  -> Credit Update:        ${ENABLE_CREDIT_UPDATE}"
  echo "===================================================================="

  # 显式导出 CUDA 可见设备，便于底层库读取
  export CUDA_VISIBLE_DEVICES
  
  # 导出 Structured Advantage 相关环境变量
  export ENABLE_OUTCOME_MODEL
  export ENABLE_STRUCTURED_ADV
  export ENABLE_CREDIT_UPDATE
  export STRUCT_TOPK_K
  export STRUCT_LAM_PAIR
  export STRUCT_USE_GAUGE_LOSS
  export STRUCT_LAM_GAUGE
  export OUTCOME_LR
  export ADV_LR
  export OUTCOME_BATCH_SIZE
  export STRUCT_MIN_SUCCESS_EP
  export STRUCT_MIN_FAILURE_EP

  # 执行 Python 训练脚本（将输出重定向到日志文件，在并行模式下）
  if [[ -n "${LOG_FILE:-}" ]]; then
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
      --replay-max-version-gap "${REPLAY_MAX_VERSION_GAP}" \
      --train-iters "${TRAIN_ITERS}" \
      > "${LOG_FILE}" 2>&1
  else
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
      --replay-max-version-gap "${REPLAY_MAX_VERSION_GAP}" \
      --train-iters "${TRAIN_ITERS}" 
  fi
  
  local exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    echo "${log_prefix}--- ✓ Finished training for Task: ${TASK_NAME}, Clip Mode: ${CLIP_MODE} ---"
  else
    echo "${log_prefix}--- ✗ Failed training for Task: ${TASK_NAME}, Clip Mode: ${CLIP_MODE} (exit code: ${exit_code}) ---"
  fi
  echo
  
  return ${exit_code}
}

# --- Helper Function: Wait for Available GPU Slot ---
wait_for_slot() {
  # 等待直到当前运行的后台任务数量小于最大并行数
  local running_jobs=$(jobs -r | wc -l)
  while [[ $running_jobs -ge ${MAX_PARALLEL_JOBS} ]]; do
    echo "当前有 $running_jobs 个任务在运行，等待空闲槽位..."
    sleep 5
    running_jobs=$(jobs -r | wc -l)
  done
  echo "检测到空闲槽位，当前运行任务数: $running_jobs"
}

# --- Helper Function: Check GPU Memory ---
check_gpu_memory() {
  if command -v nvidia-smi &> /dev/null; then
    echo
    echo "========== GPU 显存状态 =========="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
      awk -F', ' '{printf "GPU %s (%s): %d MB / %d MB (%.1f%% used)\n", $1, $2, $3, $4, ($3/$4)*100}'
    echo "=================================="
    echo
  fi
}

# --- Helper: Print running background jobs (PID + command) ---
print_running_jobs() {
  local running_jobs
  running_jobs=$(jobs -r | wc -l)
  echo "当前后台任务数: ${running_jobs}"
  if [[ ${running_jobs} -gt 0 ]]; then
    jobs -r -l
  fi
}

# --- Helper: Cleanup on exit (e.g., Ctrl+C) ---
cleanup_and_kill_jobs() {
  echo "捕获到退出信号，正在终止后台任务..."
  local pids
  pids=$(jobs -p)
  if [[ -n "${pids}" ]]; then
    echo "将杀死以下后台进程: ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap cleanup_and_kill_jobs INT TERM

# --- Main Script Logic ---
# 如果第一个参数是 "batch"，则运行批处理实验
if [[ "${1:-}" == "batch" ]]; then
  echo "===== Starting Batch Experiment Run (Parallel Mode) ====="
  echo "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
  echo "GPU configurations: ${GPU_CONFIGS[@]}"
  echo "Startup delay between jobs: ${STARTUP_DELAY}s"
  echo "⚠️  注意：多个任务共享同一组GPU，请确保显存足够！"
  
  # 显示当前GPU状态
  check_gpu_memory
  echo
  
  # 创建日志目录
  LOG_DIR="logs/parallel_runs_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LOG_DIR}"
  echo "训练日志将保存到: ${LOG_DIR}"
  echo
  
  job_idx=0  # 任务索引，用于分配 GPU
  total_jobs=$((${#TASKS[@]} * ${#CLIP_MODES[@]}))
  
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      # 等待有空闲的执行槽位
      wait_for_slot
      
      # 为当前任务分配 GPU（循环使用 GPU_CONFIGS）
      gpu_idx=$((job_idx % ${#GPU_CONFIGS[@]}))
      assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
      
      # 生成日志文件名
      log_file="${LOG_DIR}/job_${job_idx}_${task}_${clip_mode}.log"
      
      echo "========================================"
      echo "[Job $((job_idx+1))/${total_jobs}] Starting: Task=${task}, Clip=${clip_mode}, GPU=${assigned_gpu}"
      echo "[Job $((job_idx+1))/${total_jobs}] Log file: ${log_file}"
      echo "========================================"
      
      # 在后台启动训练任务（使用子 shell 避免环境变量污染）
      (
        # 为 run_training 函数设置环境变量
        TASK_NAME="${task}"
        CLIP_MODE="${clip_mode}"
        SEED="${SEED:-${DEFAULT_SEED}}"
        CUDA_VISIBLE_DEVICES="${assigned_gpu}"  # 使用分配的 GPU
        NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
        NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
        NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
        CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
        REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
        REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
        TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
        
        # Structured Advantage Config (Defaults if not set)
        ENABLE_OUTCOME_MODEL="${ENABLE_OUTCOME_MODEL:-${DEFAULT_ENABLE_OUTCOME_MODEL}}"
        ENABLE_STRUCTURED_ADV="${ENABLE_STRUCTURED_ADV:-${DEFAULT_ENABLE_STRUCTURED_ADV}}"
        ENABLE_CREDIT_UPDATE="${ENABLE_CREDIT_UPDATE:-${DEFAULT_ENABLE_CREDIT_UPDATE}}"
        STRUCT_TOPK_K="${STRUCT_TOPK_K:-${DEFAULT_STRUCT_TOPK_K}}"
        STRUCT_LAM_PAIR="${STRUCT_LAM_PAIR:-${DEFAULT_STRUCT_LAM_PAIR}}"
        STRUCT_USE_GAUGE_LOSS="${STRUCT_USE_GAUGE_LOSS:-${DEFAULT_STRUCT_USE_GAUGE_LOSS}}"
        STRUCT_LAM_GAUGE="${STRUCT_LAM_GAUGE:-${DEFAULT_STRUCT_LAM_GAUGE}}"
        OUTCOME_LR="${OUTCOME_LR:-${DEFAULT_OUTCOME_LR}}"
        ADV_LR="${ADV_LR:-${DEFAULT_ADV_LR}}"
        OUTCOME_BATCH_SIZE="${OUTCOME_BATCH_SIZE:-${DEFAULT_OUTCOME_BATCH_SIZE}}"
        STRUCT_MIN_SUCCESS_EP="${STRUCT_MIN_SUCCESS_EP:-${DEFAULT_STRUCT_MIN_SUCCESS_EP}}"
        STRUCT_MIN_FAILURE_EP="${STRUCT_MIN_FAILURE_EP:-${DEFAULT_STRUCT_MIN_FAILURE_EP}}"

        LOG_FILE="${log_file}"
        JOB_PREFIX="[Job $((job_idx+1))] "
        
        run_training
        
        echo "[Job $((job_idx+1))] Completed: Task=${task}, Clip=${clip_mode}"
      ) &  # 在后台运行
      
      # 获取刚启动的任务的 PID
      last_pid=$!
      echo "[Job $((job_idx+1))] 任务已启动，PID=${last_pid}"
      echo "[Job $((job_idx+1))] 当前后台任务列表："
      print_running_jobs
      
      job_idx=$((job_idx + 1))
      
      # 启动延迟：给每个任务足够的时间完成初始化
      # 重要：必须等待足够长的时间，让任务完全启动并被 jobs -r 识别
      if [[ $job_idx -lt ${total_jobs} ]]; then
        echo ""
        echo "等待 ${STARTUP_DELAY}s，让任务完全启动后再启动下一个..."
        echo "当前后台任务数: $(jobs -r | wc -l)"
        sleep ${STARTUP_DELAY}
        echo ""
      fi
    done
  done
  
  # 等待所有后台任务完成
  echo
  echo "所有任务已启动，等待完成..."
  echo "您可以使用以下命令监控各个任务的进度："
  echo "  tail -f ${LOG_DIR}/job_*.log"
  echo
  
  wait
  
  echo
  echo "===== Batch Experiment Run Finished ====="
  echo "所有日志保存在: ${LOG_DIR}/"
  
  # 显示最终GPU状态
  check_gpu_memory
elif [[ "${1:-}" == "batch-serial" ]]; then
  # 串行模式（保留原始的串行执行逻辑）
  echo "===== Starting Batch Experiment Run (Serial Mode) ====="
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      # 为 run_training 函数设置环境变量
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
      TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
      
      # Structured Advantage Config
      ENABLE_OUTCOME_MODEL="${ENABLE_OUTCOME_MODEL:-${DEFAULT_ENABLE_OUTCOME_MODEL}}"
      ENABLE_STRUCTURED_ADV="${ENABLE_STRUCTURED_ADV:-${DEFAULT_ENABLE_STRUCTURED_ADV}}"
      ENABLE_CREDIT_UPDATE="${ENABLE_CREDIT_UPDATE:-${DEFAULT_ENABLE_CREDIT_UPDATE}}"
      STRUCT_TOPK_K="${STRUCT_TOPK_K:-${DEFAULT_STRUCT_TOPK_K}}"
      STRUCT_LAM_PAIR="${STRUCT_LAM_PAIR:-${DEFAULT_STRUCT_LAM_PAIR}}"
      STRUCT_USE_GAUGE_LOSS="${STRUCT_USE_GAUGE_LOSS:-${DEFAULT_STRUCT_USE_GAUGE_LOSS}}"
      STRUCT_LAM_GAUGE="${STRUCT_LAM_GAUGE:-${DEFAULT_STRUCT_LAM_GAUGE}}"
      OUTCOME_LR="${OUTCOME_LR:-${DEFAULT_OUTCOME_LR}}"
      ADV_LR="${ADV_LR:-${DEFAULT_ADV_LR}}"
      OUTCOME_BATCH_SIZE="${OUTCOME_BATCH_SIZE:-${DEFAULT_OUTCOME_BATCH_SIZE}}"
      STRUCT_MIN_SUCCESS_EP="${STRUCT_MIN_SUCCESS_EP:-${DEFAULT_STRUCT_MIN_SUCCESS_EP}}"
      STRUCT_MIN_FAILURE_EP="${STRUCT_MIN_FAILURE_EP:-${DEFAULT_STRUCT_MIN_FAILURE_EP}}"

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