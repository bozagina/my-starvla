#!/usr/bin/env bash
set -euo pipefail

HOST="myserver"
REMOTE_CKPT="/2025233147/zzq/SpatialVLA_llava3d/starVLA/results/Checkpoints"
LOCAL_OUT="${1:-$PWD/_remote_runs}"

mkdir -p "$LOCAL_OUT"

echo "[fetch] Remote checkpoints: $REMOTE_CKPT"

RUNS=$(
  ssh "$HOST" "ls -1d $REMOTE_CKPT/*/ 2>/dev/null | xargs -n1 basename" || true
)

if [ -z "${RUNS:-}" ]; then
  echo "[fetch] ERROR: No run directories found under $REMOTE_CKPT"
  echo "[fetch] Remote listing (top 50):"
  ssh "$HOST" "ls -lah $REMOTE_CKPT | head -n 50" || true
  exit 2
fi

LATEST_REL=$(
  printf "%s\n" "$RUNS" \
    | grep -E "_20[0-9]{6}_[0-9]{6}$" \
    | sort \
    | tail -n 1
)

if [ -z "${LATEST_REL:-}" ]; then
  LATEST_REL=$(
    printf "%s\n" "$RUNS" | sort | tail -n 1
  )
fi

echo "[fetch] Latest run: $LATEST_REL"
mkdir -p "$LOCAL_OUT/$LATEST_REL"

echo "[fetch] Pulling: config.yaml metrics.jsonl train.log"
rsync -av \
  "$HOST:$REMOTE_CKPT/$LATEST_REL/config.yaml" \
  "$HOST:$REMOTE_CKPT/$LATEST_REL/metrics.jsonl" \
  "$HOST:$REMOTE_CKPT/$LATEST_REL/train.log" \
  "$LOCAL_OUT/$LATEST_REL/" || true

echo "[fetch] Local path: $LOCAL_OUT/$LATEST_REL"
ls -lh "$LOCAL_OUT/$LATEST_REL" || true
