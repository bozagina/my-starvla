#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MetaWorld PPO Training Script with Structured Advantage & AuxAdv Support
# =============================================================================
# 
# Usage Examples:
# 
# 1. Single run (default parameters):
#    ./run_metaworld_ppo_more_stats_try_new_v5.sh
# 
# 2. Single run with Structured Advantage via CLI:
#    ADV_STRUCT_MODE=state_softmax ADV_WEIGHT_TEMP=0.5 ./run_metaworld_ppo_more_stats_try_new_v5.sh
# 
# 3. Single run with Part 4 (AuxAdv) enabled:
#    AUX_ADV_ENABLE=1 AUX_ADV_COEF=0.1 ./run_metaworld_ppo_more_stats_try_new_v5.sh
# 
# 4. Single run with both Part 3 and Part 4:
#    ADV_STRUCT_MODE=state_softmax AUX_ADV_ENABLE=1 ./run_metaworld_ppo_more_stats_try_new_v5.sh
# 
# 5. Single run with environment variables (100% guaranteed to work):
#    export ADV_STRUCT_MODE=state_softmax
#    export ADV_WEIGHT_TEMP=0.5
#    export ADV_WEIGHT_REG=0.01
#    export AUX_ADV_ENABLE=1
#    export AUX_ADV_COEF=0.15
#    ./run_metaworld_ppo_more_stats_try_new_v5.sh
# 
# 6. Batch parallel mode:
#    ./run_metaworld_ppo_more_stats_try_new_v5.sh batch
# 
# 7. Batch serial mode:
#    ./run_metaworld_ppo_more_stats_try_new_v5.sh batch-serial
#
# 8. Ablation study mode (test Part 3 & Part 4 combinations):
#    ./run_metaworld_ppo_more_stats_try_new_v5.sh ablation
#    # This will run 4 experiments:
#    # - Baseline (no Part 3, no Part 4)
#    # - Part 3 only (state_softmax)
#    # - Part 4 only (AuxAdv)
#    # - Part 3 + Part 4 (both enabled)
# 
# Note: The script supports both CLI arguments and environment variables for
#       Structured Advantage and AuxAdv parameters. Environment variables take 
#       precedence if both are set.
# =============================================================================

# --- Configuration ---
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/ds_metaworld_ppo_mlp_with_param_more_stats_try_new_v5.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# --- Default Parameters for Single Run ---
DEFAULT_TASK_NAME="sweep-into-v3"
DEFAULT_CLIP_MODE="clip"
DEFAULT_SEED=42
DEFAULT_CUDA_DEVICES="0,1"
DEFAULT_NUM_TRAINER_GPUS=1
DEFAULT_NUM_ROLLOUT_WORKERS=16
DEFAULT_NUM_EVAL_WORKERS=32
DEFAULT_TRAIN_BATCH_SIZE=512
DEFAULT_TRAIN_ITERS=100000
# log_gauss_clip 专用：sigma 列表（空格分隔多值自动展开）
DEFAULT_LOG_GAUSS_SIGMAS="0.5 1 2"
# 新增：采样过滤参数（用于 Recency Window Ablation）
DEFAULT_REPLAY_RECENT_FRAC=1.0        # 最新比例窗口：只从最新 f% 数据采样
DEFAULT_REPLAY_MAX_VERSION_GAP="inf"  # 版本差窗口：只采样 version_gap <= max_gap 的数据
# 新增：Structured Advantage 参数
DEFAULT_ADV_STRUCT_MODE="none"        # none | state_softmax | state_action_softmax
DEFAULT_ADV_WEIGHT_TEMP=0.5           # 温度参数（越小越尖锐，促进权重分化）
DEFAULT_ADV_WEIGHT_REG=0.0            # 正则系数 (w-1)^2

# Part 3: Anti-collapse 参数
DEFAULT_ADV_W_MIN=-1.0                # 权重下限（< 0 表示不启用）
DEFAULT_ADV_W_MAX=-1.0                # 权重上限（< 0 表示不启用）
DEFAULT_ADV_W_ENT_REG=0.0             # 熵下限正则化系数
DEFAULT_ADV_W_ENT_FLOOR_FRAC=0.3      # 熵下限阈值：H_min = frac * ln(D)

# AdvStruct 新增参数：Late-start, Alpha, Separate LR
DEFAULT_ADV_STRUCT_LATE_START_FRAC=0.0  # AdvStruct 延迟启动比例（0.0 立即启动，0.2 训练 20% 后启动）
DEFAULT_ADV_STRUCT_ALPHA=1.0            # 权重收缩系数：w=1+α*(w_norm-1)，α=0 完全禁用，α=1 原样使用
DEFAULT_ADV_STRUCT_LR_MULT=1.0          # AdvStruct 权重网络学习率倍数（相对于 policy_lr）

# Part 4: Auxiliary Advantage Decomposition 参数
DEFAULT_AUX_ADV_ENABLE=0              # 是否启用（默认关闭）
DEFAULT_AUX_ADV_COEF=0.1              # MSE loss 系数
DEFAULT_AUX_ADV_LR=3e-5               # 独立学习率
DEFAULT_AUX_ADV_EMB_DIM=32            # Token embedding 维度
DEFAULT_AUX_ADV_HID=128               # State encoder hidden 维度
DEFAULT_AUX_ADV_START_STEP=0          # 从第几步开始训练
DEFAULT_AUX_ADV_PAIR_L1=0.0           # Pairwise 项 L1 正则

# --- 并行执行配置 ---
MAX_PARALLEL_JOBS=4  # 最大并行任务数（默认6个）
# 多 GPU 配置：每个并行任务使用的 GPU 组合
# 注意：多个任务可以共享同一组 GPU（前提是显存足够大）
# 格式：每行一个 GPU 配置（逗号分隔的 GPU ID）
GPU_CONFIGS=(
  "0,1"  # 第1个任务使用 GPU 0,1
  "0,1"  # 第2个任务使用 GPU 6,7
  "0,1"  # 第3个任务使用 GPU 0,1
  "0,1"  # 第4个任务使用 GPU 0,1
  # "0,1"  # 第5个任务使用 GPU 0,1
  # "0,1"  # 第6个任务使用 GPU 0,1
)

# 启动延迟（秒）：避免多个任务同时初始化导致资源竞争和端口冲突
# ⚠️ 重要：这个延迟必须足够长，让前一个任务完全启动并占用端口
STARTUP_DELAY=90  # 每个新任务启动前等待的秒数（推荐 15-25 秒）

# --- Batch Experiment Configuration ---
# 根据你的注释区整理的任务列表
TASKS=(
  "sweep-into-v3"
  "drawer-open-v3"
  # "door-open-v3"
  # "button-press-topdown-v3"
  # "handle-press-v3"
  # "push-v3"
  # "peg-insert-side-v3"
  # "pick-place-v3"
  # "plate-slide-v3"
  # "coffee-button-v3"
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
# --- Helper: generate clip config for log_gauss_clip ---
make_log_gauss_clip_config() {
  local sigma="$1"
  local tmp_cfg
  tmp_cfg=$(mktemp /tmp/clip_log_gauss_XXXX.yml)
  cat > "${tmp_cfg}" <<EOF
log_gauss_clip:
  # coeff = exp(-0.5 * (log(r+eps)/sigma)^2)
  sigma: ${sigma}
  eps: 1e-9
EOF
  echo "${tmp_cfg}"
}
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
  echo "${log_prefix}  -> Adv Struct Mode:      ${ADV_STRUCT_MODE}"
  echo "${log_prefix}  -> Adv Weight Temp:      ${ADV_WEIGHT_TEMP}"
  echo "${log_prefix}  -> Adv Weight Reg:       ${ADV_WEIGHT_REG}"
  if [[ "${ADV_STRUCT_MODE}" != "none" ]]; then
    echo "${log_prefix}  -> Adv Late Start Frac:  ${ADV_STRUCT_LATE_START_FRAC}"
    echo "${log_prefix}  -> Adv Struct Alpha:     ${ADV_STRUCT_ALPHA}"
    echo "${log_prefix}  -> Adv Struct LR Mult:   ${ADV_STRUCT_LR_MULT}"
    echo "${log_prefix}  -> Adv W Min:            ${ADV_W_MIN}"
    echo "${log_prefix}  -> Adv W Max:            ${ADV_W_MAX}"
  fi
  echo "${log_prefix}  -> Aux Adv Enable:       ${AUX_ADV_ENABLE}"
  if [[ "${AUX_ADV_ENABLE}" == "1" ]]; then
    echo "${log_prefix}  -> Aux Adv Coef:         ${AUX_ADV_COEF}"
    echo "${log_prefix}  -> Aux Adv LR:           ${AUX_ADV_LR}"
    echo "${log_prefix}  -> Aux Adv Emb Dim:      ${AUX_ADV_EMB_DIM}"
    echo "${log_prefix}  -> Aux Adv Hid:          ${AUX_ADV_HID}"
    echo "${log_prefix}  -> Aux Adv Start Step:   ${AUX_ADV_START_STEP}"
    echo "${log_prefix}  -> Aux Adv Pair L1:      ${AUX_ADV_PAIR_L1}"
  fi
  if [[ "${CLIP_MODE}" == "log_gauss_clip" ]]; then
    echo "${log_prefix}  -> Log Gauss Sigma:      ${LOG_GAUSS_SIGMA}"
    echo "${log_prefix}  -> Clip Config:          ${CLIP_CONFIG}"
  fi
  echo "===================================================================="

  # 显式导出 CUDA 可见设备，便于底层库读取
  export CUDA_VISIBLE_DEVICES

  # 执行 Python 训练脚本（将输出重定向到日志文件，在并行模式下）
  if [[ -n "${LOG_FILE:-}" ]]; then
    # 在日志文件第一行记录进程信息
    {
      echo "[PROCESS_INFO] PID=$$ | Task=${TASK_NAME} | ClipMode=${CLIP_MODE} | Seed=${SEED} | CUDA=${CUDA_VISIBLE_DEVICES} | ReplayRecentFrac=${REPLAY_RECENT_FRAC} | ReplayMaxVersionGap=${REPLAY_MAX_VERSION_GAP} | AdvStructMode=${ADV_STRUCT_MODE} | AdvWeightTemp=${ADV_WEIGHT_TEMP} | AdvWeightReg=${ADV_WEIGHT_REG} | AdvLateStartFrac=${ADV_STRUCT_LATE_START_FRAC} | AdvAlpha=${ADV_STRUCT_ALPHA} | AdvLRMult=${ADV_STRUCT_LR_MULT} | AdvWMin=${ADV_W_MIN} | AdvWMax=${ADV_W_MAX} | AuxAdvEnable=${AUX_ADV_ENABLE} | TrainIters=${TRAIN_ITERS}${LOG_GAUSS_SIGMA:+ | LogGaussSigma=${LOG_GAUSS_SIGMA}} | StartTime=$(date '+%Y-%m-%d %H:%M:%S')"
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
        --adv-struct-mode "${ADV_STRUCT_MODE}" \
        --adv-weight-temp "${ADV_WEIGHT_TEMP}" \
        --adv-weight-reg "${ADV_WEIGHT_REG}" \
        --adv-w-min "${ADV_W_MIN}" \
        --adv-w-max "${ADV_W_MAX}" \
        --adv-w-ent-reg "${ADV_W_ENT_REG}" \
        --adv-w-ent-floor-frac "${ADV_W_ENT_FLOOR_FRAC}" \
        --adv-struct-late-start-frac "${ADV_STRUCT_LATE_START_FRAC}" \
        --adv-struct-alpha "${ADV_STRUCT_ALPHA}" \
        --adv-struct-lr-mult "${ADV_STRUCT_LR_MULT}" \
        --aux-adv-enable "${AUX_ADV_ENABLE}" \
        --aux-adv-coef "${AUX_ADV_COEF}" \
        --aux-adv-lr "${AUX_ADV_LR}" \
        --aux-adv-emb-dim "${AUX_ADV_EMB_DIM}" \
        --aux-adv-hid "${AUX_ADV_HID}" \
        --aux-adv-start-step "${AUX_ADV_START_STEP}" \
        --aux-adv-pair-l1 "${AUX_ADV_PAIR_L1}" \
        2>&1
    } > "${LOG_FILE}"
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
      --train-iters "${TRAIN_ITERS}" \
      --adv-struct-mode "${ADV_STRUCT_MODE}" \
      --adv-weight-temp "${ADV_WEIGHT_TEMP}" \
      --adv-weight-reg "${ADV_WEIGHT_REG}" \
      --adv-w-min "${ADV_W_MIN}" \
      --adv-w-max "${ADV_W_MAX}" \
      --adv-w-ent-reg "${ADV_W_ENT_REG}" \
      --adv-w-ent-floor-frac "${ADV_W_ENT_FLOOR_FRAC}" \
      --adv-struct-late-start-frac "${ADV_STRUCT_LATE_START_FRAC}" \
      --adv-struct-alpha "${ADV_STRUCT_ALPHA}" \
      --adv-struct-lr-mult "${ADV_STRUCT_LR_MULT}" \
      --aux-adv-enable "${AUX_ADV_ENABLE}" \
      --aux-adv-coef "${AUX_ADV_COEF}" \
      --aux-adv-lr "${AUX_ADV_LR}" \
      --aux-adv-emb-dim "${AUX_ADV_EMB_DIM}" \
      --aux-adv-hid "${AUX_ADV_HID}" \
      --aux-adv-start-step "${AUX_ADV_START_STEP}" \
      --aux-adv-pair-l1 "${AUX_ADV_PAIR_L1}"
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
  # 解析 log_gauss sigma 列表（空格分隔）
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"

  echo "===== Starting Batch Experiment Run (Parallel Mode) ====="
  echo "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
  echo "GPU configurations: ${GPU_CONFIGS[@]}"
  echo "Startup delay between jobs: ${STARTUP_DELAY}s"
  echo "⚠️  注意：多个任务共享同一组GPU，请确保显存足够！"
  
  # 显示当前GPU状态
  check_gpu_memory
  echo
  
  # 创建日志目录
  LOG_DIR="logs/parallel_runs_try_new_v5_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LOG_DIR}"
  echo "训练日志将保存到: ${LOG_DIR}"
  echo
  
  job_idx=0  # 任务索引，用于分配 GPU
  total_jobs=0
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        total_jobs=$((total_jobs + ${#LOG_GAUSS_SIGMAS[@]}))
      else
        total_jobs=$((total_jobs + 1))
      fi
    done
  done
  
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        for sigma in "${LOG_GAUSS_SIGMAS[@]}"; do
          wait_for_slot
          gpu_idx=$((job_idx % ${#GPU_CONFIGS[@]}))
          assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
          log_file="${LOG_DIR}/job_${job_idx}_${task}_${clip_mode}_sigma-${sigma}.log"
          clip_cfg_path=$(make_log_gauss_clip_config "${sigma}")

          echo "========================================"
          echo "[Job $((job_idx+1))/${total_jobs}] Starting: Task=${task}, Clip=${clip_mode}, Sigma=${sigma}, GPU=${assigned_gpu}"
          echo "[Job $((job_idx+1))/${total_jobs}] Log file: ${log_file}"
          echo "========================================"

          (
            TASK_NAME="${task}"
            CLIP_MODE="${clip_mode}"
            LOG_GAUSS_SIGMA="${sigma}"
            SEED="${SEED:-${DEFAULT_SEED}}"
            CUDA_VISIBLE_DEVICES="${assigned_gpu}"
            NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
            NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
            NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
            TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
            CLIP_CONFIG="${clip_cfg_path}"
            REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
            REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
            TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
            ADV_STRUCT_MODE="${ADV_STRUCT_MODE:-${DEFAULT_ADV_STRUCT_MODE}}"
            ADV_WEIGHT_TEMP="${ADV_WEIGHT_TEMP:-${DEFAULT_ADV_WEIGHT_TEMP}}"
            ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
            ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
            ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
            ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
            ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
            AUX_ADV_ENABLE="${AUX_ADV_ENABLE:-${DEFAULT_AUX_ADV_ENABLE}}"
            AUX_ADV_COEF="${AUX_ADV_COEF:-${DEFAULT_AUX_ADV_COEF}}"
            AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
            AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
            AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
            AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
            AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
            LOG_FILE="${log_file}"
            JOB_PREFIX="[Job $((job_idx+1))] "

            run_training

            echo "[Job $((job_idx+1))] Completed: Task=${task}, Clip=${clip_mode}, Sigma=${sigma}"
          ) &

          last_pid=$!
          echo "[Job $((job_idx+1))] 任务已启动，PID=${last_pid}"
          job_idx=$((job_idx + 1))

          if [[ $job_idx -lt ${total_jobs} ]]; then
            echo ""
            echo "等待 ${STARTUP_DELAY}s，让任务完全启动后再启动下一个..."
            echo "当前后台任务数: $(jobs -r | wc -l)"
            sleep ${STARTUP_DELAY}
            echo ""
          fi
        done
      else
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
        ADV_STRUCT_MODE="${ADV_STRUCT_MODE:-${DEFAULT_ADV_STRUCT_MODE}}"
        ADV_WEIGHT_TEMP="${ADV_WEIGHT_TEMP:-${DEFAULT_ADV_WEIGHT_TEMP}}"
        ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
        ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
        ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
        ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
        ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
        AUX_ADV_ENABLE="${AUX_ADV_ENABLE:-${DEFAULT_AUX_ADV_ENABLE}}"
        AUX_ADV_COEF="${AUX_ADV_COEF:-${DEFAULT_AUX_ADV_COEF}}"
        AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
        AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
        AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
        AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
        AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
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
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"

  # 串行模式（保留原始的串行执行逻辑）
  echo "===== Starting Batch Experiment Run (Serial Mode) ====="
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      if [[ "${clip_mode}" == "log_gauss_clip" ]]; then
        for sigma in "${LOG_GAUSS_SIGMAS[@]}"; do
          TASK_NAME="${task}"
          CLIP_MODE="${clip_mode}"
          LOG_GAUSS_SIGMA="${sigma}"
          CLIP_CONFIG="$(make_log_gauss_clip_config "${sigma}")"
          SEED="${SEED:-${DEFAULT_SEED}}"
          CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
          NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
          NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
          NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
          TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
          REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
          REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
          TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
          ADV_STRUCT_MODE="${ADV_STRUCT_MODE:-${DEFAULT_ADV_STRUCT_MODE}}"
          ADV_WEIGHT_TEMP="${ADV_WEIGHT_TEMP:-${DEFAULT_ADV_WEIGHT_TEMP}}"
          ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
          ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
          ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
          ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
          ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
          AUX_ADV_ENABLE="${AUX_ADV_ENABLE:-${DEFAULT_AUX_ADV_ENABLE}}"
          AUX_ADV_COEF="${AUX_ADV_COEF:-${DEFAULT_AUX_ADV_COEF}}"
          AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
          AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
          AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
          AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
          AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
          run_training
        done
      else
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
      ADV_STRUCT_MODE="${ADV_STRUCT_MODE:-${DEFAULT_ADV_STRUCT_MODE}}"
      ADV_WEIGHT_TEMP="${ADV_WEIGHT_TEMP:-${DEFAULT_ADV_WEIGHT_TEMP}}"
      ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
      ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
      ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
      ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
      ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
      AUX_ADV_ENABLE="${AUX_ADV_ENABLE:-${DEFAULT_AUX_ADV_ENABLE}}"
      AUX_ADV_COEF="${AUX_ADV_COEF:-${DEFAULT_AUX_ADV_COEF}}"
      AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
      AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
      AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
      AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
      AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
      run_training
      fi
    done
  done
  echo "===== Batch Experiment Run Finished ====="
elif [[ "${1:-}" == "ablation" ]]; then
  # Ablation study mode: test Part 3 & Part 4 combinations
  echo "===== Starting Ablation Study (Part 3 & Part 4 Combinations) ====="
  echo "Maximum parallel jobs: ${MAX_PARALLEL_JOBS}"
  echo "GPU configurations: ${GPU_CONFIGS[@]}"
  echo "Startup delay between jobs: ${STARTUP_DELAY}s"
  echo
  
  # 创建日志目录
  LOG_DIR="logs/ablation_part3_part4_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${LOG_DIR}"
  echo "训练日志将保存到: ${LOG_DIR}"
  echo
  
  # 定义 Part 3 & Part 4 组合
  # 格式：label|adv_struct_mode|adv_weight_temp|aux_adv_enable|aux_adv_coef
  declare -a COMBINATIONS=(
    "baseline|none|1.0|0|0.0"
    "part3_only|state_softmax|0.5|0|0.0"
    "part4_only|none|1.0|1|0.1"
    "part3_part4|state_softmax|0.5|1|0.1"
  )
  
  # 如果未指定任务列表，使用单个默认任务
  if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${DEFAULT_TASK_NAME}")
  fi
  
  # 如果未指定裁剪模式，使用默认的 clip
  if [[ ${#CLIP_MODES[@]} -eq 0 ]]; then
    CLIP_MODES=("clip")
  fi
  
  job_idx=0
  total_jobs=$((${#TASKS[@]} * ${#CLIP_MODES[@]} * ${#COMBINATIONS[@]}))
  
  echo "将运行 ${total_jobs} 个实验："
  echo "  - 任务数: ${#TASKS[@]}"
  echo "  - 裁剪模式数: ${#CLIP_MODES[@]}"
  echo "  - 组合数: ${#COMBINATIONS[@]} (baseline, part3_only, part4_only, part3_part4)"
  echo
  
  for task in "${TASKS[@]}"; do
    for clip_mode in "${CLIP_MODES[@]}"; do
      for combo in "${COMBINATIONS[@]}"; do
        wait_for_slot
        
        # 解析组合配置
        IFS='|' read -r label adv_mode adv_temp aux_enable aux_coef <<< "${combo}"
        
        # 分配 GPU
        gpu_idx=$((job_idx % ${#GPU_CONFIGS[@]}))
        assigned_gpu="${GPU_CONFIGS[$gpu_idx]}"
        
        # 生成日志文件名
        log_file="${LOG_DIR}/job_${job_idx}_${task}_${clip_mode}_${label}.log"
        
        echo "========================================"
        echo "[Job $((job_idx+1))/${total_jobs}] Starting Ablation: ${label}"
        echo "[Job $((job_idx+1))/${total_jobs}]   Task: ${task}"
        echo "[Job $((job_idx+1))/${total_jobs}]   Clip: ${clip_mode}"
        echo "[Job $((job_idx+1))/${total_jobs}]   GPU: ${assigned_gpu}"
        echo "[Job $((job_idx+1))/${total_jobs}]   Part 3 (AdvStruct): ${adv_mode} (temp=${adv_temp})"
        echo "[Job $((job_idx+1))/${total_jobs}]   Part 4 (AuxAdv): enable=${aux_enable} (coef=${aux_coef})"
        echo "[Job $((job_idx+1))/${total_jobs}]   Log: ${log_file}"
        echo "========================================"
        
        (
          TASK_NAME="${task}"
          CLIP_MODE="${clip_mode}"
          SEED="${SEED:-${DEFAULT_SEED}}"
          CUDA_VISIBLE_DEVICES="${assigned_gpu}"
          NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS:-${DEFAULT_NUM_TRAINER_GPUS}}"
          NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-${DEFAULT_NUM_ROLLOUT_WORKERS}}"
          NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${DEFAULT_NUM_EVAL_WORKERS}}"
          TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
          CLIP_CONFIG="${CLIP_CONFIG:-${CLIP_CONFIG_PATH}}"
          REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
          REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
          TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
          
          # Part 3 配置（根据 combo 设置）
          ADV_STRUCT_MODE="${adv_mode}"
          ADV_WEIGHT_TEMP="${adv_temp}"
          ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
          ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
          ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
          ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
          ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
          
          # Part 4 配置（根据 combo 设置）
          AUX_ADV_ENABLE="${aux_enable}"
          AUX_ADV_COEF="${aux_coef}"
          AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
          AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
          AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
          AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
          AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
          
          LOG_FILE="${log_file}"
          JOB_PREFIX="[Job $((job_idx+1)) ${label}] "
          
          run_training
          
          echo "[Job $((job_idx+1))] Completed: ${label} - ${task} - ${clip_mode}"
        ) &
        
        last_pid=$!
        echo "[Job $((job_idx+1))] 任务已启动，PID=${last_pid}"
        job_idx=$((job_idx + 1))
        
        if [[ $job_idx -lt ${total_jobs} ]]; then
          echo ""
          echo "等待 ${STARTUP_DELAY}s，让任务完全启动后再启动下一个..."
          echo "当前后台任务数: $(jobs -r | wc -l)"
          sleep ${STARTUP_DELAY}
          echo ""
        fi
      done
    done
  done
  
  # 等待所有后台任务完成
  echo
  echo "所有 ablation 任务已启动，等待完成..."
  echo "您可以使用以下命令监控各个任务的进度："
  echo "  tail -f ${LOG_DIR}/job_*.log"
  echo
  
  wait
  
  echo
  echo "===== Ablation Study Finished ====="
  echo "所有日志保存在: ${LOG_DIR}/"
  echo
  echo "实验结果对比："
  echo "  - baseline:    logs/*_baseline.log"
  echo "  - part3_only:  logs/*_part3_only.log"
  echo "  - part4_only:  logs/*_part4_only.log"
  echo "  - part3_part4: logs/*_part3_part4.log"
  echo
  check_gpu_memory
else
  # 否则，像以前一样运行单次实验
  echo "===== Starting Single Experiment Run ====="
  read -ra LOG_GAUSS_SIGMAS <<< "${LOG_GAUSS_SIGMAS:-${DEFAULT_LOG_GAUSS_SIGMAS}}"
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
  if [[ "${CLIP_MODE}" == "log_gauss_clip" ]]; then
    LOG_GAUSS_SIGMA="${LOG_GAUSS_SIGMAS[0]}"  # 选第一个 sigma 作为单次运行默认
    CLIP_CONFIG="$(make_log_gauss_clip_config "${LOG_GAUSS_SIGMA}")"
  fi
  REPLAY_RECENT_FRAC="${REPLAY_RECENT_FRAC:-${DEFAULT_REPLAY_RECENT_FRAC}}"
  REPLAY_MAX_VERSION_GAP="${REPLAY_MAX_VERSION_GAP:-${DEFAULT_REPLAY_MAX_VERSION_GAP}}"
  TRAIN_ITERS="${TRAIN_ITERS:-${DEFAULT_TRAIN_ITERS}}"
  ADV_STRUCT_MODE="${ADV_STRUCT_MODE:-${DEFAULT_ADV_STRUCT_MODE}}"
  ADV_WEIGHT_TEMP="${ADV_WEIGHT_TEMP:-${DEFAULT_ADV_WEIGHT_TEMP}}"
  ADV_WEIGHT_REG="${ADV_WEIGHT_REG:-${DEFAULT_ADV_WEIGHT_REG}}"
  ADV_W_MIN="${ADV_W_MIN:-${DEFAULT_ADV_W_MIN}}"
  ADV_W_MAX="${ADV_W_MAX:-${DEFAULT_ADV_W_MAX}}"
  ADV_W_ENT_REG="${ADV_W_ENT_REG:-${DEFAULT_ADV_W_ENT_REG}}"
  ADV_W_ENT_FLOOR_FRAC="${ADV_W_ENT_FLOOR_FRAC:-${DEFAULT_ADV_W_ENT_FLOOR_FRAC}}"
  ADV_STRUCT_LATE_START_FRAC="${ADV_STRUCT_LATE_START_FRAC:-${DEFAULT_ADV_STRUCT_LATE_START_FRAC}}"
  ADV_STRUCT_ALPHA="${ADV_STRUCT_ALPHA:-${DEFAULT_ADV_STRUCT_ALPHA}}"
  ADV_STRUCT_LR_MULT="${ADV_STRUCT_LR_MULT:-${DEFAULT_ADV_STRUCT_LR_MULT}}"
  AUX_ADV_ENABLE="${AUX_ADV_ENABLE:-${DEFAULT_AUX_ADV_ENABLE}}"
  AUX_ADV_COEF="${AUX_ADV_COEF:-${DEFAULT_AUX_ADV_COEF}}"
  AUX_ADV_LR="${AUX_ADV_LR:-${DEFAULT_AUX_ADV_LR}}"
  AUX_ADV_EMB_DIM="${AUX_ADV_EMB_DIM:-${DEFAULT_AUX_ADV_EMB_DIM}}"
  AUX_ADV_HID="${AUX_ADV_HID:-${DEFAULT_AUX_ADV_HID}}"
  AUX_ADV_START_STEP="${AUX_ADV_START_STEP:-${DEFAULT_AUX_ADV_START_STEP}}"
  AUX_ADV_PAIR_L1="${AUX_ADV_PAIR_L1:-${DEFAULT_AUX_ADV_PAIR_L1}}"
  run_training
fi