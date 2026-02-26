#!/bin/bash

cd /2025233147/zzq/SpatialVLA_llava3d/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/2025233147/zzq/LIBERO
export LIBERO_CONFIG_PATH=/2025233147/zzq/LIBERO
export LIBERO_Python=$(which python)

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

host="${HOST:-127.0.0.1}"
base_port="${PORT:-5694}"
unnorm_key="franka"
your_ckpt="${CKPT:-./results/Checkpoints/1229_libero4in1_MapAnythingLlava3DPI_s42_20260213_155123/checkpoints/steps_10000_pytorch_model.pt}"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_libero.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

task_suite_name="${TASK_SUITE:-libero_goal}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-5}"
video_out_path="${EVAL_VIDEO_OUT_PATH:-results/${task_suite_name}/${folder_name}}"
enable_rt_metrics="${EVAL_ENABLE_RT_METRICS:-True}"
rt_metrics_filename="${EVAL_RT_METRICS_FILENAME:-rt_metrics.jsonl}"
request_policy_debug_info="${EVAL_REQUEST_POLICY_DEBUG_INFO:-False}"

use_state=true
expected_state_dim=8
auto_pad_state_to_expected_dim=false
log_payload_every_n_steps=1
repeat_infer_debug_times="${EVAL_REPEAT_INFER_DEBUG_TIMES:-1}"
rotate_180="${EVAL_ROTATE_180:-True}"
action_chunk_size_override="${EVAL_ACTION_CHUNK_OVERRIDE:--1}"

# Normalize shell bool-like strings to 1/0.
# Accept: true/false, True/False, 1/0, yes/no, y/n, on/off.
bool_to_01() {
    local v
    v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        1|true|yes|y|on) echo "1" ;;
        0|false|no|n|off|"") echo "0" ;;
        *)
            echo "[eval_libero.sh][WARN] Unrecognized bool value '$1', fallback to 0" >&2
            echo "0"
            ;;
    esac
}

extra_args=()
if [ "$use_state" = true ]; then
    extra_args+=(--args.use-state)
fi
if [ "$auto_pad_state_to_expected_dim" = true ]; then
    extra_args+=(--args.auto-pad-state-to-expected-dim)
fi
extra_args+=(--args.expected-state-dim "$expected_state_dim")
extra_args+=(--args.log-payload-every-n-steps "$log_payload_every_n_steps")
extra_args+=(--args.repeat-infer-debug-times "$repeat_infer_debug_times")
extra_args+=(--args.rt-metrics-filename "$rt_metrics_filename")
extra_args+=(--args.action-chunk-size-override "$action_chunk_size_override")

if [ "$(bool_to_01 "$enable_rt_metrics")" = "1" ]; then
    extra_args+=(--args.enable-rt-metrics)
else
    extra_args+=(--no-args.enable-rt-metrics)
fi
if [ "$(bool_to_01 "$request_policy_debug_info")" = "1" ]; then
    extra_args+=(--args.request-policy-debug-info)
else
    extra_args+=(--no-args.request-policy-debug-info)
fi
if [ "$(bool_to_01 "$rotate_180")" = "1" ]; then
    extra_args+=(--args.rotate-180)
else
    extra_args+=(--no-args.rotate-180)
fi

echo "Using host=$host"
echo "Using base_port=$base_port"
echo "Using task_suite_name=$task_suite_name"
echo "Using num_trials_per_task=$num_trials_per_task"
echo "Using video_out_path=$video_out_path"
echo "Using your_ckpt=$your_ckpt"
echo "Using use_state=$use_state"
echo "Using expected_state_dim=$expected_state_dim"
echo "Using auto_pad_state_to_expected_dim=$auto_pad_state_to_expected_dim"
echo "Using log_payload_every_n_steps=$log_payload_every_n_steps"
echo "Using repeat_infer_debug_times=$repeat_infer_debug_times"
echo "Using enable_rt_metrics=$enable_rt_metrics"
echo "Using rt_metrics_filename=$rt_metrics_filename"
echo "Using request_policy_debug_info=$request_policy_debug_info"
echo "Using rotate_180=$rotate_180"
echo "Using action_chunk_size_override=$action_chunk_size_override"
echo "Logs will be saved to ${LOG_FILE}"

"${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    "${extra_args[@]}"
