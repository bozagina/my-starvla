#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/bazinga/code/my-starvla"
PROMPT_DOC="$ROOT/docs/algorithm1/handoff/new_chat_bootstrap_command.md"
PROGRESS_TOOL="$ROOT/tools/handoff/new_progress_entry.py"

usage() {
  cat <<'EOF'
Usage:
  bootstrap_session.sh prompt
      Print the copy-paste startup instruction for a new chat.

  bootstrap_session.sh start --module <MASK|FBLOSS|CFG|DIAG|DATA|INFRA> [--owner <TAG>] [--title <TEXT>]
      Create a standardized IN_PROGRESS progress entry (EXP_ID) for this session.

  bootstrap_session.sh help
EOF
}

print_prompt() {
  if [[ ! -f "$PROMPT_DOC" ]]; then
    echo "Prompt doc not found: $PROMPT_DOC" >&2
    exit 1
  fi
  awk '/^```text$/{flag=1;next}/^```$/{if(flag){exit}}flag' "$PROMPT_DOC"
}

start_session() {
  local module=""
  local owner="OC"
  local title="New session task"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --module)
        module="${2:-}"
        shift 2
        ;;
      --owner)
        owner="${2:-}"
        shift 2
        ;;
      --title)
        title="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$module" ]]; then
    echo "Missing required --module" >&2
    usage
    exit 1
  fi

  python "$PROGRESS_TOOL" --module "$module" --owner "$owner" --title "$title"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  prompt)
    print_prompt
    ;;
  start)
    start_session "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
