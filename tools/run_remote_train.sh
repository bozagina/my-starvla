#!/usr/bin/env bash
set -euo pipefail

HOST="myserver"
REMOTE_REPO="/2025233147/zzq/SpatialVLA_llava3d/starVLA"
TRAIN_CMD="bash examples/LIBERO/train_files/run_libero_train.sh"

# session name: train_YYYYmmdd_HHMMSS
SESSION="train_$(date +%Y%m%d_%H%M%S)"

ssh "$HOST" bash -lc "'
set -euo pipefail
cd \"$REMOTE_REPO\"

# ensure tmux exists
command -v tmux >/dev/null 2>&1 || { echo \"tmux not found\"; exit 1; }

mkdir -p results/Checkpoints/_launch_logs
LAUNCH_LOG=results/Checkpoints/_launch_logs/launch_${SESSION}.log

# Create detached session and run training inside it (with tee)
tmux new-session -d -s \"$SESSION\" \"cd '$REMOTE_REPO' && $TRAIN_CMD 2>&1 | tee -a '\$LAUNCH_LOG' \"

echo \"Started in tmux session: $SESSION\"
echo \"Attach: tmux attach -t $SESSION\"
echo \"Launch log: $REMOTE_REPO/\$LAUNCH_LOG\"
'"
