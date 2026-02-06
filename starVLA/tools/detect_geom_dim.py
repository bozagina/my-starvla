import argparse
import json
import os
from omegaconf import OmegaConf
import torch
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--dataset_py", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.dataset_py:
        cfg.datasets.vla_data.dataset_py = args.dataset_py
    dl = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    batch = []
    itr = iter(dl)
    for _ in range(max(1, args.samples)):
        try:
            batch.extend(next(itr))
        except StopIteration:
            break
    if not batch:
        raise RuntimeError("No samples available for detection")

    model = build_framework(cfg)
    vlm = model.mapanythingllava3d_vlm_interface

    images = [e["image"] for e in batch]
    instrs = [e["lang"] for e in batch]
    inputs = vlm.build_mapanythingllava3d_inputs(images=images, instructions=instrs)

    pixel_values = inputs["pixel_values"]
    intrinsic = inputs.get("intrinsic", None)
    with torch.no_grad():
        geom_out = vlm.model.geometric_model(pixel_values=pixel_values, intrinsics=intrinsic)
        geom_feat = geom_out.last_hidden_state
        if geom_feat.dim() == 4:
            b, c, h, w = geom_feat.shape
            geom_feat = geom_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        geom_dim = int(geom_feat.shape[-1])

    print(f"Detected geometric feature dim: {geom_dim}")
    cfg.framework.mapanything_llava3d.geom_dim_override = geom_dim
    out_json = args.output_json or os.path.join(os.path.dirname(args.config_yaml), "geom_dim_detected.json")
    with open(out_json, "w") as f:
        json.dump({"geom_dim_override": geom_dim}, f, indent=2)
    print(f"Wrote geom_dim_override={geom_dim} to {out_json}. Please update your YAML config accordingly.")

if __name__ == "__main__":
    main()
