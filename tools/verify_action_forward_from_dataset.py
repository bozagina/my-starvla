#!/usr/bin/env python3
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework


def _setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("verify_action_forward")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _load_dataset(cfg, dataset_py: str, logger: logging.Logger, data_mix_override: str | None = None):
    if dataset_py != "lerobot_datasets":
        raise ValueError(
            f"Only dataset_py=lerobot_datasets is supported for now, got {dataset_py}."
        )
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

    vla_cfg = cfg.datasets.vla_data
    if data_mix_override:
        vla_cfg.data_mix = data_mix_override
        logger.info(f"Override data_mix -> {data_mix_override}")
    data_mix = getattr(vla_cfg, "data_mix", None)
    if data_mix not in DATASET_NAMED_MIXTURES:
        available = ", ".join(sorted(DATASET_NAMED_MIXTURES.keys()))
        raise KeyError(
            f"data_mix '{data_mix}' not in DATASET_NAMED_MIXTURES. Available: {available}"
        )
    dataset = get_vla_dataset(data_cfg=vla_cfg)
    logger.info(f"Loaded dataset {type(dataset).__name__} with length {len(dataset)}")
    return dataset


def _select_sample(dataset, index: int | None, seed: int, logger: logging.Logger):
    if len(dataset) <= 0:
        raise RuntimeError("Dataset is empty; cannot sample.")
    if index is None:
        random.seed(seed)
        index = random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    logger.info(f"Selected sample index: {index}")
    return index, sample


def _extract_instruction(sample: dict) -> str:
    if "lang" in sample:
        return sample["lang"]
    if "language" in sample:
        return sample["language"]
    if "instruction" in sample:
        return sample["instruction"]
    raise KeyError("Sample does not contain lang/language/instruction field.")


def _stat_tensor(x: np.ndarray | torch.Tensor):
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = x.detach().cpu().numpy()
    return {
        "shape": list(arr.shape),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--base-vlm", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-py", default=None)
    parser.add_argument("--data-mix", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", default="tools/verify_action_forward_from_dataset.log")
    parser.add_argument("--output-json", default="tools/verify_action_forward_from_dataset.json")
    args = parser.parse_args()

    _ensure_dir(args.log_file)
    _ensure_dir(args.output_json)
    logger = _setup_logging(args.log_file)

    cfg = OmegaConf.load(args.config_yaml)
    if args.base_vlm is not None:
        cfg.framework.mapanything_llava3d.base_vlm = args.base_vlm

    dataset_py = args.dataset_py or cfg.datasets.vla_data.get("dataset_py", "lerobot_datasets")
    logger.info(f"Using dataset_py={dataset_py}")

    dataset = _load_dataset(cfg, dataset_py, logger, data_mix_override=args.data_mix)
    index, sample = _select_sample(dataset, args.index, args.seed, logger)

    instruction = _extract_instruction(sample)
    images = sample.get("image")
    if images is None:
        raise KeyError("Sample does not contain image field.")
    if not isinstance(images, list):
        images = [images]

    action = sample.get("action")
    state = sample.get("state", None)

    logger.info(f"Instruction: {instruction}")
    logger.info(f"Num images: {len(images)}")

    model = build_framework(cfg)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Optional checkpoint load
    missing, unexpected = None, None
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        load_res = model.load_state_dict(state_dict, strict=False)
        missing = list(load_res.missing_keys)
        unexpected = list(load_res.unexpected_keys)
        logger.info(f"Checkpoint loaded. missing={len(missing)} unexpected={len(unexpected)}")

    example = {"image": images, "lang": instruction, "action": action}
    if state is not None:
        example["state"] = state

    # Forward for action loss
    with torch.no_grad():
        out = model.forward([example])
    action_loss = out.get("action_loss")
    action_loss_val = float(action_loss.item()) if action_loss is not None else None
    logger.info(f"action_loss: {action_loss_val}")

    # Predict action for stats
    with torch.no_grad():
        pred = model.predict_action([example])
    pred_actions = pred.get("normalized_actions")

    # Stats
    action_stats = _stat_tensor(np.array(action)) if action is not None else None
    pred_stats = _stat_tensor(pred_actions) if pred_actions is not None else None
    state_stats = _stat_tensor(np.array(state)) if state is not None else None

    cfg_action_dim = int(getattr(cfg.framework.action_model, "action_dim", -1))
    cfg_state_dim = int(getattr(cfg.framework.action_model, "state_dim", -1))
    action_dim = int(np.array(action).shape[-1]) if action is not None else None
    state_dim = int(np.array(state).shape[-1]) if state is not None else None

    logger.info(f"action_dim (cfg/actual): {cfg_action_dim} / {action_dim}")
    logger.info(f"state_dim (cfg/actual): {cfg_state_dim} / {state_dim}")

    result = {
        "config_yaml": args.config_yaml,
        "base_vlm": cfg.framework.mapanything_llava3d.base_vlm,
        "checkpoint": args.checkpoint,
        "dataset_py": dataset_py,
        "data_mix": getattr(cfg.datasets.vla_data, "data_mix", None),
        "sample_index": index,
        "instruction": instruction,
        "num_images": len(images),
        "image_sizes": [getattr(img, "size", None) for img in images],
        "action_loss": action_loss_val,
        "action_stats": action_stats,
        "pred_action_stats": pred_stats,
        "state_stats": state_stats,
        "cfg_action_dim": cfg_action_dim,
        "cfg_state_dim": cfg_state_dim,
        "actual_action_dim": action_dim,
        "actual_state_dim": state_dim,
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
