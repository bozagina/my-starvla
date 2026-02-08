from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoModel


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava3d-path", required=True)
    parser.add_argument("--llava3d-model-type", default="llama", choices=["llama", "mistral"])
    parser.add_argument("--siglip-path", required=True)
    parser.add_argument("--mapanything-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-geometric-branch", action="store_true")
    parser.add_argument("--safe-serialization", action="store_true")
    return parser.parse_args()


def _has_prefix(state_dict: dict, prefix: str) -> bool:
    for k in state_dict.keys():
        if k.startswith(prefix):
            return True
    return False


def build_base_checkpoint(args) -> None:
    _ensure_repo_root_on_path()

    from starVLA.mapanything_llava3d.model.configuration_mapanything_llava3d import (
        MapAnythingLlava3DConfig,
    )
    from starVLA.mapanything_llava3d.model.modeling_llava3d_v2 import (
        LLaVA3DForCausalLMV2,
    )
    from starVLA.mapanything_llava3d.model.modeling_mapanything import MapAnythingWrapper
    from starVLA.mapanything_llava3d.model.modeling_mapanything_llava3d_vlm import (
        MapAnythingLlava3DForConditionalGeneration,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    vision_tower = AutoModel.from_pretrained(
        args.siglip_path, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None
    )

    text_cfg = AutoConfig.from_pretrained(args.llava3d_path, trust_remote_code=True)
    setattr(text_cfg, "llava3d_model_type", args.llava3d_model_type)
    setattr(text_cfg, "llava3d_pretrained_path", args.llava3d_path)
    language_model = LLaVA3DForCausalLMV2(text_cfg)

    class _Cfg:
        pass

    map_cfg = _Cfg()
    setattr(map_cfg, "mapanything_model_name_or_path", args.mapanything_path)
    geometric_model = MapAnythingWrapper(map_cfg)

    config = MapAnythingLlava3DConfig(
        text_config=text_cfg,
        mapanything_config={"model_name_or_path": args.mapanything_path},
        vision_model_name_or_path=args.siglip_path,
        language_model_name_or_path=args.llava3d_path,
        mapanything_model_name_or_path=args.mapanything_path,
        use_spatial_token=False,
        use_geometric_branch=bool(args.use_geometric_branch),
        image_token_index=-200,
    )

    model = MapAnythingLlava3DForConditionalGeneration(
        config=config,
        vision_tower=vision_tower,
        language_model=language_model,
        mapanything_model=geometric_model,
    )

    state_dict = model.state_dict()
    required_prefixes = [
        "language_model",
        "vision_tower",
        "vision_projector",
        "geometric_model",
        "geometric_projector",
        "fusion_projector",
    ]
    missing = [p for p in required_prefixes if not _has_prefix(state_dict, p)]
    if missing:
        print(f"WARNING: missing parameter groups in assembled model: {missing}")
    else:
        print("OK: assembled model contains all required parameter groups.")

    model.save_pretrained(output_dir, safe_serialization=bool(args.safe_serialization))
    config.save_pretrained(output_dir)
    print(f"Saved base VLM checkpoint to: {output_dir}")


def main():
    args = parse_args()
    build_base_checkpoint(args)


if __name__ == "__main__":
    main()
