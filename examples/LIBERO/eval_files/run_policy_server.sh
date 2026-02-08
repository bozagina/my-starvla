#!/bin/bash
# bash examples/LIBERO/eval_files/run_policy_server.sh
cd /2025233147/zzq/SpatialVLA_llava3d/starVLA
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=$(which python)
your_ckpt=/2025233147/zzq/SpatialVLA_llava3d/model_zoo/qwen3vlPI/checkpoints/steps_100000_pytorch_model.pt
gpu_id=${GPU_ID:-1}
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
