#!/usr/bin/env python3
"""
Stamp training YAML run_id with EXP_ID for traceable remote logs.

Example:
python /Users/bazinga/code/my-starvla/tools/handoff/stamp_run_id_with_exp.py \
  --config /path/to/train.yaml \
  --exp-id ALG1-MASK-20260301-006-OC \
  --in-place
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


EXP_RE = re.compile(r"^ALG1-[A-Z]+-[0-9]{8}-[0-9]{3}-[A-Z0-9_]+$")


def _short_tag(exp_id: str) -> str:
    parts = exp_id.split("-")
    module = parts[1]
    ymd = parts[2]
    seq = parts[3]
    return f"{module}-{ymd[4:]}-{seq}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append EXP_ID to run_id in training YAML.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--use-short-tag", action="store_true", help="Append short tag (e.g. MASK-0301-003).")
    parser.add_argument("--output", type=Path, default=None, help="Output path when not in-place.")
    args = parser.parse_args()

    exp_id = args.exp_id.strip().upper()
    if not EXP_RE.match(exp_id):
        raise ValueError(f"Invalid EXP_ID format: {exp_id}")

    cfg_path: Path = args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")

    run_id = data.get("run_id", "")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Config missing valid top-level `run_id` string.")
    run_id = run_id.strip()

    tag = _short_tag(exp_id) if args.use_short_tag else exp_id
    suffix = f"__{tag}"
    if run_id.endswith(suffix):
        new_run_id = run_id
    else:
        # Remove previous EXP suffix if present.
        new_run_id = re.sub(r"__ALG1-[A-Z]+-[0-9]{8}-[0-9]{3}-[A-Z0-9_]+$", "", run_id)
        new_run_id = re.sub(r"__[A-Z]+-[0-9]{4}-[0-9]{3}$", "", new_run_id)
        new_run_id = f"{new_run_id}{suffix}"
    data["run_id"] = new_run_id

    out_path = cfg_path if args.in_place else (args.output or cfg_path.with_name(f"{cfg_path.stem}.exp.yaml"))
    out_text = yaml.safe_dump(data, allow_unicode=False, sort_keys=False)
    out_path.write_text(out_text, encoding="utf-8")

    print(f"CONFIG={cfg_path}")
    print(f"OUTPUT={out_path}")
    print(f"EXP_ID={exp_id}")
    print(f"RUN_ID_OLD={run_id}")
    print(f"RUN_ID_NEW={new_run_id}")


if __name__ == "__main__":
    main()
