import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _ensure_dist_get_rank_safe():
    try:
        if dist.is_available() and dist.is_initialized():
            _ = dist.get_rank()
            return
    except Exception:
        pass

    def _fake_get_rank():
        return 0

    dist.get_rank = _fake_get_rank


def load_config(config_yaml: str, dot_args):
    cfg = OmegaConf.load(config_yaml)
    dotlist = normalize_dotlist_args(dot_args)
    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def print_module_grad_stats(name: str, module: torch.nn.Module):
    if module is None:
        print(f"[{name}] module is None")
        return
    n_params = sum(p.numel() for p in module.parameters())
    n_req = sum(p.numel() for p in module.parameters() if p.requires_grad)
    any_grad = any(p.grad is not None for p in module.parameters() if p.requires_grad)
    nonzero_grad = any(
        p.grad is not None and torch.any(p.grad != 0) for p in module.parameters() if p.requires_grad
    )
    print(f"[{name}] n_params={n_params}, requires_grad_params={n_req}")
    print(f"[{name}] any_grad={any_grad}, any_nonzero_grad={nonzero_grad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=1,
        help="Number of batches to run for debug",
    )
    args, clipargs = parser.parse_known_args()

    cfg = load_config(args.config_yaml, clipargs)

    _ensure_dist_get_rank_safe()

    # 单卡 debug，不使用 accelerate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    model = build_framework(cfg)
    model.to(device)
    model.train()

    batch_iter = iter(dataloader)
    for step in range(args.max_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break

        model.zero_grad()
        out = model.forward(batch)
        loss = out["action_loss"]
        print(f"[step {step}] action_loss={loss.item():.6f}")

        base_model = model
        vlm_interface = getattr(base_model, "mapanythingllava3d_vlm_interface", None)
        vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None

        image_features = getattr(vlm_model, "_last_image_features", None) if vlm_model is not None else None
        if image_features is not None and image_features.requires_grad:
            image_features.retain_grad()

        loss.backward()

        if vlm_model is None:
            print("vlm_model is None, cannot inspect grads")
            return

        debug_spatial_img_id = getattr(vlm_model, "_debug_last_spatial_img_id", None)
        debug_image_token_index = getattr(vlm_model, "_debug_last_image_token_index", None)
        debug_image_token_id = getattr(vlm_model, "_debug_last_image_token_id", None)
        debug_has_pixel_values = getattr(vlm_model, "_debug_last_has_pixel_values", None)
        debug_image_mask_any = getattr(vlm_model, "_debug_last_image_mask_any", None)
        debug_image_mask_sum = getattr(vlm_model, "_debug_last_image_mask_sum", None)
        debug_input_ids_head = getattr(vlm_model, "_debug_last_input_ids_head", None)
        debug_image_feat_shape = getattr(vlm_model, "_debug_last_image_features_shape", None)
        debug_inputs_embeds_shape = getattr(vlm_model, "_debug_last_inputs_embeds_shape", None)

        print("[vlm_debug] has_pixel_values:", debug_has_pixel_values)
        print("[vlm_debug] spatial_img_id:", debug_spatial_img_id)
        print("[vlm_debug] image_token_index:", debug_image_token_index)
        print("[vlm_debug] image_token_id:", debug_image_token_id)
        print("[vlm_debug] image_mask_any:", debug_image_mask_any)
        if debug_image_mask_sum is not None:
            try:
                print("[vlm_debug] image_mask_sum_per_sample:", debug_image_mask_sum.tolist())
            except Exception:
                print("[vlm_debug] image_mask_sum_per_sample: <unavailable>")
        if debug_input_ids_head is not None:
            try:
                print("[vlm_debug] input_ids_head[0]:", debug_input_ids_head[0].tolist())
            except Exception:
                print("[vlm_debug] input_ids_head: <unavailable>")
        print("[vlm_debug] image_features_shape:", debug_image_feat_shape)
        print("[vlm_debug] inputs_embeds_shape:", debug_inputs_embeds_shape)

        if image_features is not None:
            grad_exists = image_features.grad is not None
            grad_norm = (
                float(image_features.grad.detach().float().norm().item())
                if image_features.grad is not None
                else 0.0
            )
            print(
                f"[image_features] requires_grad={image_features.requires_grad}, "
                f"grad_exists={grad_exists}, grad_norm={grad_norm}"
            )

        print_module_grad_stats("vision_tower", getattr(vlm_model, "vision_tower", None))
        print_module_grad_stats("vision_projector", getattr(vlm_model, "vision_projector", None))
        print_module_grad_stats("geometric_model", getattr(vlm_model, "geometric_model", None))
        print_module_grad_stats(
            "geometric_projector", getattr(vlm_model, "geometric_projector", None)
        )
        print_module_grad_stats("fusion_projector", getattr(vlm_model, "fusion_projector", None))
        print_module_grad_stats("language_model", getattr(vlm_model, "language_model", None))
        break


if __name__ == "__main__":
    main()
