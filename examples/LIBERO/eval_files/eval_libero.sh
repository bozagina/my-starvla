#!/bin/bash

cd /2025233147/zzq/SpatialVLA_llava3d/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/2025233147/zzq/LIBERO
export LIBERO_CONFIG_PATH=/2025233147/zzq/LIBERO
export LIBERO_Python=$(which python)

# 让 Python 能找到 LIBERO 和 starVLA 这两个包
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
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
