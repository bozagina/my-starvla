#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _extract_step(record: Dict, fallback_idx: int) -> int:
    keys = [
        "debug/sagr/feedback_mask_contrast_step",
        "step",
        "global_step",
        "trainer/step",
    ]
    for key in keys:
        val = record.get(key, None)
        if _is_number(val):
            try:
                return int(val)
            except Exception:
                pass
    return int(fallback_idx)


def _load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _collect(records: List[Dict], key: str) -> List[float]:
    vals = []
    for r in records:
        v = r.get(key, None)
        if _is_number(v):
            vals.append(float(v))
    return vals


def _safe_mean(vals: List[float]) -> Optional[float]:
    return mean(vals) if vals else None


def _fmt(v: Optional[float], ndigits: int = 6) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def _check(name: str, value: Optional[float], cmp: str, threshold: float) -> Tuple[str, str]:
    if value is None:
        return name, "SKIP"
    if cmp == ">=":
        ok = value >= threshold
    elif cmp == "<=":
        ok = value <= threshold
    else:
        raise ValueError(f"unsupported cmp: {cmp}")
    return name, "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate m_clip reliability from metrics.jsonl")
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--tail", type=int, default=0, help="Use only the last N filtered records. 0 means all.")
    parser.add_argument("--min-active-ratio", type=float, default=0.7)
    parser.add_argument("--max-entropy-ratio", type=float, default=0.999)
    parser.add_argument("--min-topk-ratio", type=float, default=1.05)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--min-neg-kl", type=float, default=0.01)
    parser.add_argument("--min-neg-top1-gap", type=float, default=0.0005)
    parser.add_argument("--min-alpha-cos", type=float, default=0.05)
    parser.add_argument("--require-neg-control", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any required check fails.")
    args = parser.parse_args()

    records_raw = _load_jsonl(args.metrics_jsonl)
    if not records_raw:
        print("No valid records found.")
        return 1

    records = []
    for idx, r in enumerate(records_raw):
        step = _extract_step(r, idx)
        if args.min_step is not None and step < args.min_step:
            continue
        if args.max_step is not None and step > args.max_step:
            continue
        x = dict(r)
        x["__step"] = step
        records.append(x)

    if args.tail and args.tail > 0:
        records = records[-int(args.tail) :]

    if not records:
        print("No records after step filtering.")
        return 1

    total_n = len(records)
    active_flags = [
        float(r.get("debug/causal_feedback/soft_mask_teacher_active", 0.0)) > 0.5
        for r in records
    ]
    active_records = [r for r, a in zip(records, active_flags) if a]
    active_ratio = float(len(active_records)) / float(max(total_n, 1))

    token_num_vals = _collect(active_records, "debug/causal_feedback/soft_mask_teacher_token_num")
    token_num_mean = _safe_mean(token_num_vals)
    entropy_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_entropy"))
    topk32_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_topk_mass_32"))
    confidence_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_confidence_mean"))
    alpha_cos_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_alpha_cos"))
    teacher_kl_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_kl"))
    neg_kl_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_neg_shuffle_kl"))
    neg_top1_gap_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_neg_top1_gap"))
    neg_entropy_gap_mean = _safe_mean(_collect(active_records, "debug/causal_feedback/soft_mask_teacher_neg_entropy_gap"))

    entropy_ratio = None
    topk_ratio = None
    if token_num_mean is not None and token_num_mean > 1:
        uniform_entropy = math.log(token_num_mean)
        if entropy_mean is not None and uniform_entropy > 0:
            entropy_ratio = entropy_mean / uniform_entropy
        uniform_topk = min(32.0 / token_num_mean, 1.0)
        if topk32_mean is not None and uniform_topk > 0:
            topk_ratio = topk32_mean / uniform_topk

    step_min = min(r["__step"] for r in records)
    step_max = max(r["__step"] for r in records)

    print("=== m_clip Reliability Report ===")
    print(f"metrics: {args.metrics_jsonl}")
    print(f"window: steps [{step_min}, {step_max}], records={total_n}, active={len(active_records)}")
    print(f"teacher_active_ratio={active_ratio:.4f}")
    print(f"teacher_kl_mean={_fmt(teacher_kl_mean)}")
    print(f"teacher_entropy_mean={_fmt(entropy_mean)}")
    print(f"teacher_entropy_ratio_to_uniform={_fmt(entropy_ratio)}")
    print(f"teacher_topk_mass_32_mean={_fmt(topk32_mean)}")
    print(f"teacher_topk_ratio_to_uniform={_fmt(topk_ratio)}")
    print(f"teacher_confidence_mean={_fmt(confidence_mean)}")
    print(f"teacher_alpha_cos_mean={_fmt(alpha_cos_mean)}")
    print(f"teacher_neg_shuffle_kl_mean={_fmt(neg_kl_mean)}")
    print(f"teacher_neg_top1_gap_mean={_fmt(neg_top1_gap_mean)}")
    print(f"teacher_neg_entropy_gap_mean={_fmt(neg_entropy_gap_mean)}")

    checks: List[Tuple[str, str, str]] = []
    checks.append((*_check("active_ratio", active_ratio, ">=", args.min_active_ratio), f">= {args.min_active_ratio}"))
    checks.append((*_check("entropy_ratio", entropy_ratio, "<=", args.max_entropy_ratio), f"<= {args.max_entropy_ratio}"))
    checks.append((*_check("topk_ratio", topk_ratio, ">=", args.min_topk_ratio), f">= {args.min_topk_ratio}"))
    checks.append((*_check("confidence_mean", confidence_mean, ">=", args.min_confidence), f">= {args.min_confidence}"))
    checks.append((*_check("alpha_cos_mean", alpha_cos_mean, ">=", args.min_alpha_cos), f">= {args.min_alpha_cos}"))

    if args.require_neg_control:
        checks.append((*_check("neg_shuffle_kl", neg_kl_mean, ">=", args.min_neg_kl), f">= {args.min_neg_kl}"))
        checks.append((*_check("neg_top1_gap", neg_top1_gap_mean, ">=", args.min_neg_top1_gap), f">= {args.min_neg_top1_gap}"))

    print("\nChecks:")
    failed_required = False
    for name, status, threshold in checks:
        print(f"- {name}: {status} (target {threshold})")
        if status == "FAIL":
            failed_required = True

    if args.strict and failed_required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
