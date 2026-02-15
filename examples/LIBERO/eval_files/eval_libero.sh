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

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=./results/Checkpoints/1229_libero4in1_MapAnythingLlava3DPI_s42_20260213_155123/checkpoints/steps_10000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_libero.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

task_suite_name=libero_goal
num_trials_per_task=5
video_out_path="results/${task_suite_name}/${folder_name}"

use_state=true
expected_state_dim=8
auto_pad_state_to_expected_dim=false
log_payload_every_n_steps=1
repeat_infer_debug_times=3

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
echo "Logs will be saved to ${LOG_FILE}"

"${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    "${extra_args[@]}"
