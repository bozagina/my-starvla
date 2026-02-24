#!/bin/bash

cd /2025233147/zzq/SpatialVLA_llava3d/starVLA
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=$(which python)

# Accept runtime overrides from orchestrator; keep a fallback for manual runs.
your_ckpt="${CKPT:-./results/Checkpoints/1229_libero4in1_MapAnythingLlava3DPI_s42_20260213_155123/checkpoints/steps_10000_pytorch_model.pt}"
gpu_id="${GPU_ID:-1}"
port="${PORT:-5694}"
################# star Policy Server ######################

# export DEBUG=true
echo "[run_policy_server] CKPT=${your_ckpt}"
echo "[run_policy_server] GPU_ID=${gpu_id}"
echo "[run_policy_server] PORT=${port}"
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
