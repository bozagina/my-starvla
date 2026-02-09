#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Tuple


def _ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _resolve_index_file(pretrained_path: str) -> Optional[Path]:
    candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    base = Path(pretrained_path)
    for name in candidates:
        p = base / name
        if p.is_file():
            return p
    return None


def _resolve_weight_files(pretrained_path: str) -> Tuple[Path, ...]:
    base = Path(pretrained_path)
    index_file = _resolve_index_file(pretrained_path)
    if index_file is not None:
        data = json.loads(index_file.read_text())
        shard_files = sorted(set(data.get("weight_map", {}).values()))
        return tuple(base / f for f in shard_files)
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for name in candidates:
        p = base / name
        if p.is_file():
            return (p,)
    raise FileNotFoundError(
        f"No model weights found under {pretrained_path} (missing index or weight files)."
    )


def _pick_probe_key(index_file: Path) -> str:
    data = json.loads(index_file.read_text())
    keys = list(data.get("weight_map", {}).keys())
    for k in keys:
        if "model.layers.0.self_attn.q_proj.weight" in k:
            return k
    for k in keys:
        if "model.layers.0" in k:
            return k
    return keys[0]


def _find_weight_key_in_state_dict(state_dict, target_key: str) -> str:
    if target_key in state_dict:
        return target_key
    # fallback for wrappers
    if target_key.startswith("model.") and ("model." + target_key) in state_dict:
        return "model." + target_key
    if target_key.startswith("model."):
        alt = target_key.replace("model.", "", 1)
        if alt in state_dict:
            return alt
    return target_key


def _run_checks(llava_path: str, base_vlm_path: str) -> Tuple[float, bool]:
    _ensure_repo_root_on_path()

    import torch  # noqa: F401
    from transformers import AutoConfig
    from transformers.modeling_utils import load_state_dict

    from starVLA.mapanything_llava3d.model.modeling_llava3d_v2 import (
        LLaVA3DForCausalLMV2,
    )
    from starVLA.mapanything_llava3d.model.modeling_mapanything_llava3d_vlm import (
        MapAnythingLlava3DForConditionalGeneration,
    )

    # 1) verify wrapper loads LLM weights and no warning output
    cfg = AutoConfig.from_pretrained(llava_path, trust_remote_code=True)
    setattr(cfg, "llava3d_model_type", "llama")
    setattr(cfg, "llava3d_pretrained_path", llava_path)
    m = LLaVA3DForCausalLMV2(cfg)

    index_file = _resolve_index_file(llava_path)
    if index_file is None:
        raise FileNotFoundError("Missing index file for LLaVA checkpoint.")
    probe_key = _pick_probe_key(index_file)
    data = json.loads(index_file.read_text())
    shard_name = data["weight_map"][probe_key]
    shard = load_state_dict(str(Path(llava_path) / shard_name))
    probe_key = _find_weight_key_in_state_dict(shard, probe_key)
    w_loaded = shard[probe_key]
    w_model = m.model.state_dict()[probe_key]
    max_abs_diff = (w_loaded - w_model).abs().max().item()

    # 2) inspect base checkpoint index for known legacy keys
    base_index = _resolve_index_file(base_vlm_path)
    if base_index is not None:
        base_data = json.loads(base_index.read_text())
        base_keys = list(base_data.get("weight_map", {}).keys())
        has_ls_weight = any("ls1.weight" in k or "ls2.weight" in k for k in base_keys)
        has_mm_projector = any("mm_projector" in k for k in base_keys)
        print(f"base_index_file: {base_index}")
        print(f"base_has_ls_weight: {has_ls_weight}")
        print(f"base_has_mm_projector: {has_mm_projector}")
        # Verify shard contents to detect stale/extra keys in weight files.
        shard_name = next(iter(set(base_data.get("weight_map", {}).values())), None)
        if shard_name is not None:
            shard_path = Path(base_vlm_path) / shard_name
            shard_sd = load_state_dict(str(shard_path))
            shard_has_ls_weight = any(
                "ls1.weight" in k or "ls2.weight" in k for k in shard_sd.keys()
            )
            shard_has_mm_projector = any("mm_projector" in k for k in shard_sd.keys())
            print(f"base_shard_file: {shard_path}")
            print(f"base_shard_has_ls_weight: {shard_has_ls_weight}")
            print(f"base_shard_has_mm_projector: {shard_has_mm_projector}")
    else:
        print("base_index_file: NOT FOUND")

    # 3) verify VLM assembly loads without LLaVA3D warning
    _ = MapAnythingLlava3DForConditionalGeneration.from_pretrained(
        base_vlm_path, trust_remote_code=True
    )

    return max_abs_diff, True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava-path", required=True)
    parser.add_argument("--base-vlm-path", required=True)
    parser.add_argument("--log-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    warnings.simplefilter("always")

    with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        print("== Verify LLaVA wrapper load ==")
        print(f"llava_path: {args.llava_path}")
        print(f"base_vlm_path: {args.base_vlm_path}")
        try:
            max_abs_diff, _ = _run_checks(args.llava_path, args.base_vlm_path)
            print(f"max_abs_diff (probe weight): {max_abs_diff}")
            print("RESULT: PASS")
        except Exception as e:
            print("RESULT: FAIL")
            print(repr(e))
            raise

    # After run, scan log for known warning patterns
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    warn_patterns = [
        "Some weights of LLaVA3DForCausalLMV2 were not initialized",
        "Some weights of the model checkpoint",
        "were not used when initializing LlavaLlamaForCausalLM",
    ]
    found = [p for p in warn_patterns if p in log_text]
    if found:
        print("WARNING patterns detected in log:")
        for p in found:
            print(f"- {p}")
    else:
        print("No known warning patterns found in log.")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
