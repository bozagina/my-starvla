import argparse
from omegaconf import OmegaConf
import torch
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework


def _flatten_geom(geom_feat):
    if geom_feat.dim() == 4:
        b, c, h, w = geom_feat.shape
        return geom_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    return geom_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--dataset_py", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fallback_random", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.dataset_py:
        cfg.datasets.vla_data.dataset_py = args.dataset_py
    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "pp"):
        cfg.trainer.pp.enabled = False
        cfg.trainer.pp.distributed = False

    model = build_framework(cfg)
    vlm = model.mapanythingllava3d_vlm_interface
    geom_dim_infer = None
    try:
        geom_dim_infer = int(vlm.model._infer_geom_dim())
    except Exception:
        geom_dim_infer = None
    pixel_values = None
    intrinsic = None
    try:
        dl = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        batch = next(iter(dl))
        images = [e["image"] for e in batch]
        instrs = [e["lang"] for e in batch]
        inputs = vlm.build_mapanythingllava3d_inputs(images=images, instructions=instrs)
        pixel_values = inputs["pixel_values"]
        intrinsic = inputs.get("intrinsic", None)
    except FileNotFoundError as e:
        if not args.fallback_random:
            raise e
        b = 1
        v = max(1, args.num_views)
        pixel_values = torch.randn(b, v, 3, args.image_size, args.image_size)
        intrinsic = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1)

    pixel_single = pixel_values[:, 0] if pixel_values.dim() == 5 else pixel_values
    intrinsic_single = intrinsic[:, 0] if (intrinsic is not None and intrinsic.dim() == 4) else intrinsic

    with torch.no_grad():
        geom_out_single = vlm.model.geometric_model(pixel_values=pixel_single, intrinsics=intrinsic_single)
        geom_feat_single = _flatten_geom(geom_out_single.last_hidden_state)
        geom_dim_single = int(geom_feat_single.shape[-1])

        geom_out_multi = vlm.model.geometric_model(pixel_values=pixel_values, intrinsics=intrinsic)
        geom_feat_multi = _flatten_geom(geom_out_multi.last_hidden_state)
        geom_dim_multi = int(geom_feat_multi.shape[-1])

    print(f"geom_dim_infer={geom_dim_infer}")
    print(f"geom_dim_single={geom_dim_single} shape={tuple(geom_feat_single.shape)}")
    print(f"geom_dim_multi={geom_dim_multi} shape={tuple(geom_feat_multi.shape)}")
    print(f"dim_equal={geom_dim_single == geom_dim_multi}")


if __name__ == "__main__":
    main()
