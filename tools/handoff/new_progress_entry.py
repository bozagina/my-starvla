#!/usr/bin/env python3
"""
Create a standardized progress entry with a unique EXP_ID.

Example:
python /Users/bazinga/code/my-starvla/tools/handoff/new_progress_entry.py \
  --module MASK \
  --owner OC \
  --title "Tune feedback mask contrast schedule"
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


DEFAULT_PROGRESS_FILE = Path(
    "/Users/bazinga/code/my-starvla/docs/algorithm1/handoff/progress_live.md"
)
MODULE_CHOICES = ("MASK", "FBLOSS", "CFG", "DIAG", "DATA", "INFRA")
STATUS_CHOICES = ("IN_PROGRESS", "DONE", "BLOCKED", "ROLLED_BACK")


def _format_ts(now: dt.datetime) -> str:
    offset = now.strftime("%z")
    if len(offset) == 5:
        offset = f"{offset[:3]}:{offset[3:]}"
    return now.strftime("%Y-%m-%d %H:%M:%S ") + offset


def _next_seq(text: str, module: str, ymd: str) -> int:
    pattern = re.compile(rf"ALG1-{re.escape(module)}-{re.escape(ymd)}-(\d{{3}})-[A-Z0-9_]+")
    seqs = [int(m.group(1)) for m in pattern.finditer(text)]
    if not seqs:
        return 1
    return max(seqs) + 1


def _sanitize_owner(owner: str) -> str:
    owner = owner.strip().upper()
    owner = re.sub(r"[^A-Z0-9_]", "", owner)
    return owner or "UNKNOWN"


def _build_entry(timestamp: str, exp_id: str, title: str, owner: str, status: str) -> str:
    safe_title = title.strip() or "Untitled task"
    return (
        f"\n## [{timestamp}] {exp_id} {safe_title}\n\n"
        f"- Owner: {owner}\n"
        f"- Status: {status}\n"
        f"- Objective:\n"
        f"  - ...\n"
        f"- Changes:\n"
        f"  - Files:\n"
        f"    - `...`\n"
        f"  - Code/Config summary:\n"
        f"    - ...\n"
        f"- Evidence:\n"
        f"  - Commands:\n"
        f"    - `...`\n"
        f"  - Key outputs/metrics:\n"
        f"    - ...\n"
        f"- Decision:\n"
        f"  - ...\n"
        f"- Risks/Notes:\n"
        f"  - ...\n"
        f"- Next step:\n"
        f"  - ...\n"
        f"- Commit message:\n"
        f"  - `[{exp_id}] {safe_title}`\n"
    )


def _exp_short(exp_id: str) -> str:
    parts = exp_id.split("-")
    if len(parts) < 5:
        return exp_id
    module = parts[1]
    ymd = parts[2]
    seq = parts[3]
    return f"{module}-{ymd[4:]}-{seq}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a standardized progress entry with unique EXP_ID.")
    parser.add_argument("--module", required=True, choices=MODULE_CHOICES)
    parser.add_argument("--owner", default="OC")
    parser.add_argument("--title", required=True)
    parser.add_argument("--status", default="IN_PROGRESS", choices=STATUS_CHOICES)
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    progress_file: Path = args.progress_file
    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")

    text = progress_file.read_text(encoding="utf-8")
    owner = _sanitize_owner(args.owner)
    now = dt.datetime.now().astimezone()
    ymd = now.strftime("%Y%m%d")
    seq = _next_seq(text=text, module=args.module, ymd=ymd)
    exp_id = f"ALG1-{args.module}-{ymd}-{seq:03d}-{owner}"
    timestamp = _format_ts(now)
    entry = _build_entry(timestamp=timestamp, exp_id=exp_id, title=args.title, owner=owner, status=args.status)
    suggested_commit = f"[{exp_id}] {args.title.strip()}"
    run_suffix = _exp_short(exp_id)

    if args.dry_run:
        print(exp_id)
        print(entry.strip("\n"))
        print(f"SUGGESTED_COMMIT_MESSAGE={suggested_commit}")
        print(f"SUGGESTED_RUN_ID_SUFFIX={run_suffix}")
        return

    if not text.endswith("\n"):
        text += "\n"
    text += entry
    progress_file.write_text(text, encoding="utf-8")
    print(exp_id)
    print(f"Appended entry to: {progress_file}")
    print(f"SUGGESTED_COMMIT_MESSAGE={suggested_commit}")
    print(f"SUGGESTED_RUN_ID_SUFFIX={run_suffix}")


if __name__ == "__main__":
    main()
