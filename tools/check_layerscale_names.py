#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _scan_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_gamma = bool(re.search(r"ls[12]\.gamma", text))
    has_weight = bool(re.search(r"ls[12]\.weight", text))
    has_layerscale_cls = "LayerScale" in text
    return {
        "path": str(path),
        "has_layerscale_cls": has_layerscale_cls,
        "has_ls_gamma": has_gamma,
        "has_ls_weight": has_weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = [
        root / "starVLA/mapanything_llava3d/model/map-anything/mapanything/models/external/pi3/layers/block.py",
        root / "starVLA/mapanything_llava3d/model/map-anything/mapanything/models/external/vggt/layers/block.py",
    ]

    print("LayerScale name check")
    for p in targets:
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        info = _scan_file(p)
        print(f"FILE: {info['path']}")
        print(f"  has LayerScale class: {info['has_layerscale_cls']}")
        print(f"  has ls*.gamma refs:   {info['has_ls_gamma']}")
        print(f"  has ls*.weight refs:  {info['has_ls_weight']}")


if __name__ == "__main__":
    main()
