#!/bin/bash

cd /2025233147/zzq/SpatialVLA_llava3d/starVLA
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=$(which python)

# Accept runtime overrides from orchestrator; keep a fallback for manual runs.
your_ckpt="${CKPT:-./results/Checkpoints/1229_libero4in1_MapAnythingLlava3DPI_s42_20260213_155123/checkpoints/steps_10000_pytorch_model.pt}"
gpu_id="${GPU_ID:-1}"
port="${PORT:-5694}"
disable_patha_infer="${DISABLE_PATHA_INFER:-0}"
enable_patha_infer="${ENABLE_PATHA_INFER:-0}"
patha_feedback_scale="${PATHA_FEEDBACK_SCALE:-}"
################# star Policy Server ######################

# export DEBUG=true
echo "[run_policy_server] CKPT=${your_ckpt}"
echo "[run_policy_server] GPU_ID=${gpu_id}"
echo "[run_policy_server] PORT=${port}"
echo "[run_policy_server] DISABLE_PATHA_INFER=${disable_patha_infer}"
echo "[run_policy_server] ENABLE_PATHA_INFER=${enable_patha_infer}"
echo "[run_policy_server] PATHA_FEEDBACK_SCALE=${patha_feedback_scale}"

${star_vla_python} - <<'PY'
import importlib
import sys

required = ["torch", "omegaconf", "accelerate"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as e:
        missing.append((name, str(e)))

if missing:
    print("[run_policy_server][ERROR] Missing required Python packages in current env:")
    for name, err in missing:
        print(f"  - {name}: {err}")
    print("[run_policy_server][ERROR] Please switch to the training/eval env or install these deps, then retry.")
    sys.exit(2)

try:
    importlib.import_module("starVLA.model.framework.MapAnythingLlava3DPI")
except Exception as e:
    print("[run_policy_server][ERROR] Failed to import target framework module:")
    print(f"  - starVLA.model.framework.MapAnythingLlava3DPI: {e}")
    print("[run_policy_server][ERROR] Current env is not compatible with this checkpoint.")
    sys.exit(3)
PY

extra_args=()
if [ "${disable_patha_infer}" = "1" ]; then
  extra_args+=(--disable_path_a_inference)
fi
if [ "${enable_patha_infer}" = "1" ]; then
  extra_args+=(--enable_path_a_inference)
fi
if [ -n "${patha_feedback_scale}" ]; then
  extra_args+=(--path_a_feedback_scale "${patha_feedback_scale}")
fi

CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    "${extra_args[@]}"

# #################################
