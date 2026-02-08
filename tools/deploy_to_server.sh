#!/usr/bin/env bash
set -euo pipefail

HOST="myserver"
REMOTE_REPO="/2025233147/zzq/SpatialVLA_llava3d/starVLA"

# push current branch
git push

# pull on server
ssh "$HOST" bash -lc "'
set -euo pipefail
cd \"$REMOTE_REPO\"
git pull
git status -sb
'"
