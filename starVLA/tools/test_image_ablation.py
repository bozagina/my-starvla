import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.dataloader import build_dataloader


def _ensure_dist_get_rank_safe():
    try:
        _ = dist.get_rank()
    except Exception:
        def _fake_get_rank():
            return 0

        dist.get_rank = _fake_get_rank


def load_config(config_yaml: str, dot_args):
    cfg = OmegaConf.load(config_yaml)
    dotlist = normalize_dotlist_args(dot_args)
    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    if not hasattr(cfg, "output_dir"):
        cfg.output_dir = "./results/debug_image_ablation"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def build_model_and_data(cfg, checkpoint_path: str | None = None):
    _ensure_dist_get_rank_safe()
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    model = build_framework(cfg)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()
    return model, dataloader


@torch.inference_mode()
def run_image_ablation(model, dataloader, num_batches: int = 1):
    device = next(model.parameters()).device
    batch_iter = iter(dataloader)

    for step in range(num_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break

        examples = batch
        examples_with = copy.deepcopy(examples)
        examples_without = copy.deepcopy(examples)

        for ex in examples_with:
            img = ex.get("image", None)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                ex["image"] = img[0]
        for ex in examples_without:
            img = ex.get("image", None)
            if img is None:
                continue
            if isinstance(img, np.ndarray) and img.ndim == 4:
                img = img[0]
            if isinstance(img, np.ndarray):
                ex["image"] = np.zeros_like(img)
            else:
                arr = np.array(img)
                ex["image"] = np.zeros_like(arr)

        out_with = model.predict_action(examples_with)
        out_without = model.predict_action(examples_without)

        act_with = out_with["normalized_actions"]
        act_without = out_without["normalized_actions"]

        act_with_t = torch.from_numpy(act_with).to(device=device, dtype=torch.float32)
        act_without_t = torch.from_numpy(act_without).to(device=device, dtype=torch.float32)

        diff = act_with_t - act_without_t
        rms_diff = diff.pow(2).mean().sqrt().item()
        rms_with = act_with_t.pow(2).mean().sqrt().item()

        per_sample_diff = diff.pow(2).mean(dim=(1, 2)).sqrt()
        per_sample_diff = per_sample_diff.cpu().tolist()

        print(f"[batch {step}] action RMS with image: {rms_with:.6f}")
        print(f"[batch {step}] action RMS difference (with vs without image): {rms_diff:.6f}")
        print(f"[batch {step}] per-sample RMS diff: {per_sample_diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to training config YAML (e.g. starvla_train_oxe_mapanything_llava3d.yaml)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to model state_dict (pytorch_model.pt) for the trained checkpoint",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="Number of batches to evaluate",
    )

    args, clipargs = parser.parse_known_args()

    cfg = load_config(args.config_yaml, clipargs)
    model, dataloader = build_model_and_data(cfg, checkpoint_path=args.checkpoint_path)
    run_image_ablation(model, dataloader, num_batches=args.num_batches)


if __name__ == "__main__":
    main()
