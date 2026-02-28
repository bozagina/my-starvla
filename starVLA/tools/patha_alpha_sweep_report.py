#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RunSummary:
    run_dir: str
    alpha_target: Optional[float]
    probe_count: int
    delta_neg_ratio: Optional[float]
    delta_mean: Optional[float]
    delta_abs_mean: Optional[float]
    hpr_mean: Optional[float]
    hpr_max: Optional[float]
    mse_pair_count: int
    patha_minus_base_mean: Optional[float]
    patha_worse_ratio: Optional[float]


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "None"
    return f"{v:.{digits}f}"


def _load_alpha_target(config_path: str) -> Optional[float]:
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("feedback_delta_action_alpha_target:"):
                    value = s.split(":", 1)[1].strip()
                    return float(value)
    except Exception:
        return None
    return None


def summarize_run(run_dir: str) -> RunSummary:
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    config_path = os.path.join(run_dir, "config.yaml")
    alpha_target = _load_alpha_target(config_path)

    probe_delta: List[float] = []
    probe_hpr: List[float] = []
    mse_gap: List[float] = []
    mse_worse_flags: List[float] = []

    if not os.path.exists(metrics_path):
        return RunSummary(
            run_dir=run_dir,
            alpha_target=alpha_target,
            probe_count=0,
            delta_neg_ratio=None,
            delta_mean=None,
            delta_abs_mean=None,
            hpr_mean=None,
            hpr_max=None,
            mse_pair_count=0,
            patha_minus_base_mean=None,
            patha_worse_ratio=None,
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, object] = json.loads(line)
            except Exception:
                continue

            trig = obj.get("debug/sagr/feedback_probe_triggered")
            if float(trig) == 1.0:
                d = obj.get("debug/sagr/feedback_probe_delta_loss_with_minus_no")
                h = obj.get("debug/sagr/feedback_probe_hidden_perturb_ratio")
                if isinstance(d, (int, float)):
                    probe_delta.append(float(d))
                if isinstance(h, (int, float)):
                    probe_hpr.append(float(h))

            mse = obj.get("mse_score")
            mse_patha = obj.get("mse_score_patha")
            if isinstance(mse, (int, float)) and isinstance(mse_patha, (int, float)):
                gap = float(mse_patha) - float(mse)
                mse_gap.append(gap)
                mse_worse_flags.append(1.0 if gap > 0.0 else 0.0)

    delta_neg_ratio = None
    if probe_delta:
        delta_neg_ratio = float(sum(1.0 for x in probe_delta if x < 0.0) / len(probe_delta))

    return RunSummary(
        run_dir=run_dir,
        alpha_target=alpha_target,
        probe_count=len(probe_delta),
        delta_neg_ratio=delta_neg_ratio,
        delta_mean=_safe_mean(probe_delta),
        delta_abs_mean=_safe_mean([abs(x) for x in probe_delta]),
        hpr_mean=_safe_mean(probe_hpr),
        hpr_max=max(probe_hpr) if probe_hpr else None,
        mse_pair_count=len(mse_gap),
        patha_minus_base_mean=_safe_mean(mse_gap),
        patha_worse_ratio=_safe_mean(mse_worse_flags),
    )


def verdict(summary: RunSummary) -> List[str]:
    notes: List[str] = []
    if summary.probe_count == 0:
        notes.append("probe_missing")
        return notes

    if summary.delta_neg_ratio is not None and summary.delta_neg_ratio >= 0.7:
        notes.append("delta_sign_ok")
    else:
        notes.append("delta_sign_weak")

    if summary.hpr_max is not None and summary.hpr_max < 0.1:
        notes.append("hpr_guard_ok")
    else:
        notes.append("hpr_too_high")

    # "不能持续变差" -> use both mean gap and worse ratio as a simple proxy.
    if summary.mse_pair_count == 0:
        notes.append("mse_pair_missing")
    else:
        mean_gap = summary.patha_minus_base_mean if summary.patha_minus_base_mean is not None else 0.0
        worse_ratio = summary.patha_worse_ratio if summary.patha_worse_ratio is not None else 1.0
        if mean_gap <= 0.0 and worse_ratio < 0.8:
            notes.append("mse_guard_ok")
        else:
            notes.append("mse_guard_risky")
    return notes


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Step-2 alpha sweep acceptance metrics from training metrics.jsonl."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories that contain config.yaml and metrics.jsonl",
    )
    args = parser.parse_args()

    summaries = [summarize_run(run_dir=os.path.abspath(p)) for p in args.runs]
    summaries.sort(key=lambda x: (x.alpha_target is None, x.alpha_target))

    header = (
        "run_dir\talpha_target\tprobe_count\tdelta_neg_ratio\tdelta_mean\tdelta_abs_mean\t"
        "hpr_mean\thpr_max\tmse_pair_count\tpatha_minus_base_mean\tpatha_worse_ratio\tverdict"
    )
    print(header)
    for s in summaries:
        v = ",".join(verdict(s))
        print(
            "\t".join(
                [
                    s.run_dir,
                    _fmt(s.alpha_target, 4),
                    str(s.probe_count),
                    _fmt(s.delta_neg_ratio, 4),
                    _fmt(s.delta_mean, 6),
                    _fmt(s.delta_abs_mean, 6),
                    _fmt(s.hpr_mean, 6),
                    _fmt(s.hpr_max, 6),
                    str(s.mse_pair_count),
                    _fmt(s.patha_minus_base_mean, 6),
                    _fmt(s.patha_worse_ratio, 4),
                    v,
                ]
            )
        )


if __name__ == "__main__":
    main()
