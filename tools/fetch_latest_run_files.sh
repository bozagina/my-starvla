#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-myserver}"
REMOTE_CKPT="${REMOTE_CKPT:-/2025233147/zzq/SpatialVLA_llava3d/starVLA/results/Checkpoints}"
LOCAL_OUT="${1:-$PWD/_remote_runs}"
RUN_ID="${RUN_ID:-}"
RUN_FILTER="${RUN_FILTER:-}"
ALLOW_LOCAL_FALLBACK="${ALLOW_LOCAL_FALLBACK:-0}"

mkdir -p "$LOCAL_OUT"

echo "[fetch] Source host: $HOST"
echo "[fetch] Remote checkpoints: $REMOTE_CKPT"

USE_SSH=1
if [ -z "${HOST}" ] || [ "${HOST}" = "local" ] || [ "${HOST}" = "localhost" ] || [ "${HOST}" = "127.0.0.1" ]; then
  USE_SSH=0
fi
if [ "$USE_SSH" -eq 1 ]; then
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "echo ok" >/dev/null 2>&1; then
    if [ -d "$REMOTE_CKPT" ] && [ "$ALLOW_LOCAL_FALLBACK" = "1" ]; then
      echo "[fetch] WARN: ssh to ${HOST} failed; using local path ${REMOTE_CKPT} (ALLOW_LOCAL_FALLBACK=1)"
      USE_SSH=0
    else
      echo "[fetch] ERROR: ssh to ${HOST} failed."
      echo "[fetch] Check first: ssh ${HOST}"
      echo "[fetch] Hint: run this script on your local machine to pull from server '${HOST}'."
      echo "[fetch] If you intentionally want local-path copy on this machine, set ALLOW_LOCAL_FALLBACK=1."
      exit 3
    fi
  fi
fi

if [ "$USE_SSH" -eq 1 ]; then
  RUNS=$(
    ssh "$HOST" "ls -1d $REMOTE_CKPT/*/ 2>/dev/null | xargs -n1 basename" || true
  )
else
  RUNS=$(
    ls -1d "$REMOTE_CKPT"/*/ 2>/dev/null | xargs -n1 basename || true
  )
fi

if [ -z "${RUNS:-}" ]; then
  echo "[fetch] ERROR: No run directories found under $REMOTE_CKPT"
  echo "[fetch] Remote listing (top 50):"
  if [ "$USE_SSH" -eq 1 ]; then
    ssh "$HOST" "ls -lah $REMOTE_CKPT | head -n 50" || true
  else
    ls -lah "$REMOTE_CKPT" | head -n 50 || true
  fi
  exit 2
fi

if [ -n "$RUN_FILTER" ]; then
  FILTERED_RUNS=$(
    printf "%s\n" "$RUNS" | grep -E "$RUN_FILTER" || true
  )
  if [ -z "${FILTERED_RUNS:-}" ]; then
    echo "[fetch] WARN: RUN_FILTER='$RUN_FILTER' matched nothing; fallback to all runs."
  else
    RUNS="$FILTERED_RUNS"
  fi
fi

if [ -n "$RUN_ID" ]; then
  MATCHED_RUNS=$(
    printf "%s\n" "$RUNS" \
      | awk -v rid="$RUN_ID" '
          $0 == rid { print; next }
          index($0, rid "__") == 1 { print }
        ' \
      || true
  )
  if [ -n "${MATCHED_RUNS:-}" ]; then
    LATEST_REL="$(printf "%s\n" "$MATCHED_RUNS" | sort | tail -n 1)"
    if [ "$LATEST_REL" != "$RUN_ID" ]; then
      echo "[fetch] INFO: RUN_ID matched by prefix, selected: $LATEST_REL"
    fi
  else
    echo "[fetch] WARN: RUN_ID '$RUN_ID' not found in run directories; using it as-is."
    LATEST_REL="$RUN_ID"
  fi
else
  LATEST_REL=$(
    printf "%s\n" "$RUNS" \
      | awk '
          {
            run = $0
            # Allow run dir suffixes like "__ALG1-MASK-20260301-002-OC".
            base = run
            sub(/__.*/, "", base)
            if (match(base, /_20[0-9]{6}_[0-9]{6}$/)) {
              ts = substr(base, RSTART + 1, RLENGTH - 1)
              gsub(/_/, "", ts)
              print ts "\t" run
            }
          }
        ' \
      | sort -k1,1 \
      | tail -n 1 \
      | cut -f2
  )

  if [ -z "${LATEST_REL:-}" ]; then
    LATEST_REL=$(
      printf "%s\n" "$RUNS" | sort | tail -n 1
    )
  fi
fi

echo "[fetch] Latest run: $LATEST_REL"
ABS_LOCAL_OUT="$(cd "$LOCAL_OUT" && pwd)"
LOCAL_RUN_PATH="$ABS_LOCAL_OUT/$LATEST_REL"
mkdir -p "$LOCAL_RUN_PATH"

echo "[fetch] Pulling: config.yaml metrics.jsonl summary.jsonl train.log train.raw.log"
FILES=("config.yaml" "metrics.jsonl" "summary.jsonl" "train.log" "train.raw.log")
for f in "${FILES[@]}"; do
  REMOTE_FILE="$REMOTE_CKPT/$LATEST_REL/$f"
  if [ "$USE_SSH" -eq 1 ]; then
    if ssh "$HOST" "test -f '$REMOTE_FILE'" >/dev/null 2>&1; then
      rsync -av "$HOST:$REMOTE_FILE" "$LOCAL_RUN_PATH/"
    else
      echo "[fetch] SKIP missing: $f"
    fi
  else
    if [ -f "$REMOTE_FILE" ]; then
      rsync -av "$REMOTE_FILE" "$LOCAL_RUN_PATH/"
    else
      echo "[fetch] SKIP missing: $f"
    fi
  fi
done

echo "[fetch] Saved to: $LOCAL_RUN_PATH"
ln -sfn "$LOCAL_RUN_PATH" "$ABS_LOCAL_OUT/latest"
echo "[fetch] Latest symlink: $ABS_LOCAL_OUT/latest"
ls -lh "$LOCAL_RUN_PATH" || true

if [ ! -f "$LOCAL_RUN_PATH/train.log" ] && [ -f "$LOCAL_RUN_PATH/train.raw.log" ]; then
  tr '\r' '\n' < "$LOCAL_RUN_PATH/train.raw.log" > "$LOCAL_RUN_PATH/train.log"
  echo "[fetch] NOTE: train.log missing on source; generated locally from train.raw.log"
fi

# Try to extract experiment identity from run config for later analysis handoff.
CONFIG_FILE="$LOCAL_RUN_PATH/config.yaml"
RUN_META_FILE="$LOCAL_RUN_PATH/run_identity.txt"
if [ -f "$CONFIG_FILE" ]; then
  RUN_ID_LINE="$(grep -E '^run_id:' "$CONFIG_FILE" | head -n 1 || true)"
  RUN_ID_VALUE="$(printf "%s" "$RUN_ID_LINE" | sed -E 's/^run_id:[[:space:]]*//; s/[[:space:]]+$//; s/^"//; s/"$//')"
  EXP_ID_VALUE="$(printf "%s" "$RUN_ID_VALUE" | grep -oE 'ALG1-[A-Z]+-[0-9]{8}-[0-9]{3}-[A-Z0-9_]+' | head -n 1 || true)"
  SHORT_TAG_VALUE="$(printf "%s" "$RUN_ID_VALUE" | grep -oE '[A-Z]+-[0-9]{4}-[0-9]{3}' | head -n 1 || true)"
  {
    echo "run_dir=$LATEST_REL"
    echo "run_id=$RUN_ID_VALUE"
    echo "exp_id=$EXP_ID_VALUE"
    echo "run_tag=$SHORT_TAG_VALUE"
  } > "$RUN_META_FILE"
  echo "[fetch] Run identity saved: $RUN_META_FILE"
  if [ -n "$EXP_ID_VALUE" ]; then
    echo "[fetch] Parsed EXP_ID: $EXP_ID_VALUE"
  elif [ -n "$SHORT_TAG_VALUE" ]; then
    echo "[fetch] Parsed run tag: $SHORT_TAG_VALUE"
  else
    echo "[fetch] WARN: No EXP_ID/run tag found in run_id. Recommend appending '__ALG1-...'."
  fi
fi
