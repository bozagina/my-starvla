#!/usr/bin/env python3
# Copyright 2026 starVLA community.
# Licensed under the MIT License, Version 1.0.

"""
Checkpoint-only Path-A sanity checker.

This script runs four checks without continuing training:
1) Inference ablation on feedback tokens (none / zero / shuffle)
2) Teacher-forced ablation on feedback tokens
3) Gradient attribution share on feedback tokens vs task tokens
4) Compact conclusion from aggregated metrics

Usage example:
  torchrun --nproc_per_node=4 starVLA/tools/patha_checkpoint_ablation.py \
    --config_yaml /path/to/config.yaml \
    --checkpoint /path/to/steps_2000_pytorch_model.pt \
    --num_batches 8
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework import build_framework
from starVLA.training.train_starvla import (
    VLATrainer,
    build_accelerator,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

logger = get_logger(__name__)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _main_print(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _ensure_output_dir(cfg) -> str:
    run_root = str(getattr(cfg, "run_root_dir", "./results/Checkpoints"))
    run_id = str(getattr(cfg, "run_id", f"patha_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    output_dir = str(getattr(cfg, "output_dir", os.path.join(run_root, run_id)))
    os.makedirs(output_dir, exist_ok=True)
    cfg.output_dir = output_dir
    return output_dir


def _build_dataloader(cfg) -> DataLoader:
    dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=dataset_cfg)
    num_workers = int(getattr(dataset_cfg, "num_workers", 4))
    batch_size = int(getattr(dataset_cfg, "per_device_batch_size", 1))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
    )
    return dataloader


def _load_checkpoint_state_dict(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, object]:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is empty.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    return {
        "checkpoint_path": checkpoint_path,
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "missing_keys_preview": list(missing_keys[:20]),
        "unexpected_keys_preview": list(unexpected_keys[:20]),
    }


def _autocast_context() -> contextlib.AbstractContextManager:
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _run_grad_probe(trainer: VLATrainer, examples: List[dict]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    was_training = bool(getattr(trainer.model, "training", False))
    trainer.model.train()
    trainer.optimizer.zero_grad()
    try:
        with _autocast_context():
            output_dict = trainer.model.forward(copy.deepcopy(examples))
            action_loss = output_dict["action_loss"]
        trainer.accelerator.backward(action_loss)
        metrics["debug/causal_feedback_grad/probe_action_loss"] = float(action_loss.detach().float().cpu().item())
        trainer._collect_feedback_grad_attribution(metrics)
    finally:
        trainer.optimizer.zero_grad()
        if not was_training:
            trainer.model.eval()
    return metrics


def _numeric_items(metrics: Dict[str, object]) -> Iterable[Tuple[str, float]]:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            val = float(value)
            if np.isfinite(val):
                yield key, val


def _collect_key_union(local_keys: List[str]) -> List[str]:
    key_set = set(local_keys)
    if dist.is_initialized():
        gathered: List[List[str]] = [None for _ in range(_world_size())]  # type: ignore
        dist.all_gather_object(gathered, list(key_set))
        for ks in gathered:
            if ks is None:
                continue
            key_set.update(ks)
    return sorted(key_set)


def _reduce_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _aggregate_metrics(
    local_sums: Dict[str, float],
    local_counts: Dict[str, int],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    all_keys = _collect_key_union(list(local_sums.keys()) + list(local_counts.keys()))
    aggregated: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        global_sum = _reduce_scalar(local_sums.get(key, 0.0), device=device)
        global_count = _reduce_scalar(float(local_counts.get(key, 0)), device=device)
        if global_count <= 0:
            continue
        aggregated[key] = {
            "mean": float(global_sum / max(global_count, 1.0)),
            "count": float(global_count),
        }
    return aggregated


def _pick_mean(metrics: Dict[str, Dict[str, float]], key: str) -> Optional[float]:
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    mean = item.get("mean")
    if mean is None:
        return None
    return float(mean)


def _build_conclusion(aggregated: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    zero_rel = _pick_mean(aggregated, "debug/patha_sanity/mse_rel_delta_zero_vs_ext_none")
    shuffle_rel = _pick_mean(aggregated, "debug/patha_sanity/mse_rel_delta_shuffle_vs_ext_none")
    tf_zero_rel = _pick_mean(aggregated, "debug/patha_sanity/tf_action_loss_rel_delta_zero")
    tf_shuffle_rel = _pick_mean(aggregated, "debug/patha_sanity/tf_action_loss_rel_delta_shuffle")
    fb_grad_share = _pick_mean(aggregated, "debug/causal_feedback_grad/fb_grad_share_l2")
    fb_available = _pick_mean(aggregated, "debug/patha_sanity/feedback_token_available")

    notes: List[str] = []
    status = "undetermined"
    severity = "info"

    if fb_available is not None and fb_available < 0.5:
        status = "feedback_not_available"
        severity = "high"
        notes.append("Inference output did not expose feedback_tokens on most evaluated batches.")
    else:
        weak_infer = (
            zero_rel is not None
            and shuffle_rel is not None
            and abs(zero_rel) < 0.01
            and abs(shuffle_rel) < 0.01
        )
        strong_infer = (
            zero_rel is not None
            and shuffle_rel is not None
            and (zero_rel > 0.05 or shuffle_rel > 0.05)
        )

        weak_tf = (
            tf_zero_rel is not None
            and tf_shuffle_rel is not None
            and abs(tf_zero_rel) < 0.01
            and abs(tf_shuffle_rel) < 0.01
        )
        strong_tf = (
            tf_zero_rel is not None
            and tf_shuffle_rel is not None
            and (tf_zero_rel > 0.02 or tf_shuffle_rel > 0.02)
        )

        low_grad = fb_grad_share is not None and fb_grad_share < 0.05
        high_grad = fb_grad_share is not None and fb_grad_share > 0.15

        if strong_infer or strong_tf:
            status = "feedback_used"
            severity = "info"
            notes.append("Ablation degrades performance, indicating Path-A signal is used.")
            if low_grad:
                notes.append("Gradient share on feedback tokens is low despite ablation effect; inspect scale/alignment.")
        elif weak_infer and weak_tf and low_grad:
            status = "feedback_likely_ignored"
            severity = "high"
            notes.append("Ablation is almost invariant and feedback gradient share is near zero.")
        else:
            status = "feedback_weak_or_unstable"
            severity = "medium"
            notes.append("Some probes are inconclusive or disagree; run more batches and inspect raw metrics.")

        if zero_rel is not None and zero_rel < 0:
            notes.append("Zero-feedback ablation improved MSE; this may indicate noisy/misaligned feedback tokens.")
        if shuffle_rel is not None and shuffle_rel < 0:
            notes.append("Shuffled-feedback ablation improved MSE; check token semantics and scaling.")
        if high_grad:
            notes.append("Feedback gradient share is high; training depends on feedback path.")

    return {
        "status": status,
        "severity": severity,
        "notes": notes,
        "key_means": {
            "mse_rel_delta_zero_vs_ext_none": zero_rel,
            "mse_rel_delta_shuffle_vs_ext_none": shuffle_rel,
            "tf_action_loss_rel_delta_zero": tf_zero_rel,
            "tf_action_loss_rel_delta_shuffle": tf_shuffle_rel,
            "fb_grad_share_l2": fb_grad_share,
            "feedback_token_available": fb_available,
        },
    }


def run(args, cfg) -> Dict[str, object]:
    output_dir = _ensure_output_dir(cfg)
    accelerator = build_accelerator(cfg)

    rank = _rank()
    seed = int(getattr(cfg, "seed", 3047)) + rank
    set_seed(seed)

    _main_print(f"[PathA-Ablation] Output dir: {output_dir}")
    _main_print(f"[PathA-Ablation] World size: {_world_size()}, rank: {rank}")

    model = build_framework(cfg)
    checkpoint_path = str(args.checkpoint or getattr(cfg.trainer, "pretrained_checkpoint", ""))
    if not checkpoint_path:
        raise ValueError("Please provide checkpoint path via --checkpoint or trainer.pretrained_checkpoint in config.")
    load_info = _load_checkpoint_state_dict(model, checkpoint_path)
    _main_print(
        "[PathA-Ablation] Checkpoint loaded: "
        f"missing={load_info['missing_keys_count']}, unexpected={load_info['unexpected_keys_count']}"
    )

    dataloader = _build_dataloader(cfg)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.logger_backend = "none"

    freeze_modules = (
        cfg.trainer.freeze_modules if (cfg and hasattr(cfg.trainer, "freeze_modules")) else None
    )
    trainer.model = trainer.freeze_backbones(trainer.model, freeze_modules=freeze_modules)

    trainer.model, trainer.optimizer, trainer.vla_train_dataloader = trainer.setup_distributed_training(
        trainer.accelerator,
        trainer.model,
        trainer.optimizer,
        trainer.vla_train_dataloader,
    )
    trainer._patch_deepspeed_no_sync_if_needed()
    trainer._register_grad_hooks(trainer.accelerator.unwrap_model(trainer.model))

    run_teacher_forced = not bool(args.skip_teacher_forced)
    run_infer_ablation = not bool(args.skip_inference_ablation)
    run_grad_probe = not bool(args.skip_grad_probe)

    num_ddim_steps = int(getattr(cfg.trainer, "eval_num_ddim_steps", 20))
    local_sums: Dict[str, float] = defaultdict(float)
    local_counts: Dict[str, int] = defaultdict(int)
    raw_batch_metrics: List[Dict[str, float]] = []

    processed_batches = 0
    for batch_idx, examples in enumerate(trainer.vla_train_dataloader):
        if batch_idx >= int(args.num_batches):
            break
        processed_batches += 1
        batch_metrics: Dict[str, float] = {}

        actions_tk = np.asarray([example.get("action_tk", example["action"]) for example in examples], dtype=np.float32)
        valid_tk = np.asarray([float(example.get("valid_tk", 1.0)) for example in examples], dtype=np.float32)

        trainer.model.eval()
        if run_infer_ablation:
            infer_metrics = trainer._run_inference_feedback_sanity(
                examples=copy.deepcopy(examples),
                actions_tk=actions_tk,
                valid_tk=valid_tk,
                num_ddim_steps=num_ddim_steps,
            )
            batch_metrics.update(infer_metrics)

        if run_teacher_forced:
            tf_metrics = trainer._run_teacher_forced_feedback_ablation(
                examples=copy.deepcopy(examples),
            )
            batch_metrics.update(tf_metrics)

        if run_grad_probe:
            grad_metrics = _run_grad_probe(trainer, examples=examples)
            batch_metrics.update(grad_metrics)

        batch_metrics["debug/patha_sanity/batch_index"] = float(batch_idx)
        for key, value in _numeric_items(batch_metrics):
            local_sums[key] += value
            local_counts[key] += 1

        if len(raw_batch_metrics) < int(args.keep_raw_batches):
            raw_batch_metrics.append({k: v for k, v in _numeric_items(batch_metrics)})

        if _rank() == 0:
            _main_print(
                f"[PathA-Ablation] Batch {batch_idx + 1}/{args.num_batches}: "
                f"keys={len(batch_metrics)}"
            )

    local_processed_batches = float(processed_batches)
    aggregated = _aggregate_metrics(local_sums, local_counts, device=trainer.accelerator.device)
    conclusion = _build_conclusion(aggregated)

    global_processed_batches = _reduce_scalar(local_processed_batches, trainer.accelerator.device)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_yaml": args.config_yaml,
        "checkpoint": checkpoint_path,
        "world_size": _world_size(),
        "num_batches_requested": int(args.num_batches),
        "num_batches_processed_sum_over_ranks": float(global_processed_batches),
        "num_ddim_steps_arg": int(num_ddim_steps),
        "load_info": load_info,
        "aggregated_metrics": aggregated,
        "conclusion": conclusion,
        "raw_batch_metrics_preview": raw_batch_metrics,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pt path; overrides cfg.trainer.pretrained_checkpoint")
    parser.add_argument("--num_batches", type=int, default=8, help="Number of eval batches per rank")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output report path. Default: <cfg.output_dir>/patha_ablation_report.json",
    )
    parser.add_argument("--keep_raw_batches", type=int, default=4, help="Keep first N raw batch metrics in report")
    parser.add_argument("--skip_teacher_forced", action="store_true", help="Skip teacher-forced ablation")
    parser.add_argument("--skip_inference_ablation", action="store_true", help="Skip inference ablation")
    parser.add_argument("--skip_grad_probe", action="store_true", help="Skip gradient attribution probe")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if args.checkpoint is not None:
        cfg.trainer.pretrained_checkpoint = args.checkpoint

    report = run(args, cfg)

    if _rank() == 0:
        output_json = args.output_json
        if not output_json:
            output_json = os.path.join(str(cfg.output_dir), "patha_ablation_report.json")
        output_parent = os.path.dirname(output_json)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        conclusion = report.get("conclusion", {})
        key_means = conclusion.get("key_means", {})
        _main_print(f"[PathA-Ablation] Report saved: {output_json}")
        _main_print(f"[PathA-Ablation] status={conclusion.get('status')} severity={conclusion.get('severity')}")
        _main_print(
            "[PathA-Ablation] key means: "
            f"mse_zero_rel={key_means.get('mse_rel_delta_zero_vs_ext_none')}, "
            f"mse_shuffle_rel={key_means.get('mse_rel_delta_shuffle_vs_ext_none')}, "
            f"tf_zero_rel={key_means.get('tf_action_loss_rel_delta_zero')}, "
            f"tf_shuffle_rel={key_means.get('tf_action_loss_rel_delta_shuffle')}, "
            f"fb_grad_share_l2={key_means.get('fb_grad_share_l2')}"
        )
        notes = conclusion.get("notes", [])
        for idx, note in enumerate(notes, start=1):
            _main_print(f"[PathA-Ablation] note{idx}: {note}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
