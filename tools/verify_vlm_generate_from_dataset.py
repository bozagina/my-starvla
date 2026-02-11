#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.vlm import get_vlm_model


def _setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("verify_vlm_generate")
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


def _ensure_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--base-vlm", default=None)
    parser.add_argument("--dataset-py", default=None)
    parser.add_argument("--data-mix", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--log-file", default="tools/verify_vlm_generate_from_dataset.log")
    parser.add_argument("--output-json", default="tools/verify_vlm_generate_from_dataset.json")
    parser.add_argument("--save-image", default=None)
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

    # Optional: save the first image for inspection
    saved_image_path = None
    if args.save_image:
        _ensure_dir(args.save_image)
        try:
            images[0].save(args.save_image)
            saved_image_path = args.save_image
        except Exception as e:
            logger.warning(f"Failed to save image to {args.save_image}: {e}")

    logger.info(f"Instruction: {instruction}")
    logger.info(f"Num images: {len(images)}")

    # Build VLM
    vlm = get_vlm_model(config=cfg)
    model = vlm.model
    processor = vlm.processor

    # Build inputs
    inputs = vlm.build_mapanythingllava3d_inputs(images=[images], instructions=[instruction])
    input_len = int(inputs["input_ids"].shape[1])
    logger.info(f"input_ids_len: {input_len}")

    # Generate
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = gen[:, input_len:]
    decoded_new = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    logger.info(f"decoded_new: {decoded_new}")

    # Action token check
    action_begin = getattr(model.config, "action_token_begin_idx", None)
    spatial_token_num = getattr(model.config, "spatial_token_num", None)
    action_token_info = {
        "action_token_begin_idx": action_begin,
        "spatial_token_num": spatial_token_num,
        "action_token_count": 0,
        "action_token_ids_sample": [],
        "decoded_actions": None,
    }
    if action_begin is not None and spatial_token_num is not None:
        token_ids = new_tokens[0].detach().cpu().tolist()
        in_range = [tid for tid in token_ids if action_begin <= tid < action_begin + spatial_token_num]
        action_token_info["action_token_count"] = len(in_range)
        action_token_info["action_token_ids_sample"] = in_range[:20]
        logger.info(
            f"action_token_range=[{action_begin}, {action_begin + spatial_token_num}) count={len(in_range)}"
        )

        # Try decode actions if tokenizer available
        if getattr(processor, "action_tokenizer", None) is not None:
            try:
                decoded_actions = processor.decode_actions(new_tokens, unnorm_key=None)
                action_token_info["decoded_actions"] = {
                    "action_ids": decoded_actions.get("action_ids").tolist(),
                    "actions": decoded_actions.get("actions").tolist(),
                }
            except Exception as e:
                logger.warning(f"decode_actions failed: {e}")
    else:
        logger.info("action_token_begin_idx is None or spatial_token_num missing; action token check skipped.")

    # Save JSON
    result = {
        "config_yaml": args.config_yaml,
        "base_vlm": cfg.framework.mapanything_llava3d.base_vlm,
        "dataset_py": dataset_py,
        "sample_index": index,
        "instruction": instruction,
        "num_images": len(images),
        "image_sizes": [getattr(img, "size", None) for img in images],
        "saved_image": saved_image_path,
        "input_ids_len": input_len,
        "max_new_tokens": args.max_new_tokens,
        "decoded_new": decoded_new,
        "action_token_info": action_token_info,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
