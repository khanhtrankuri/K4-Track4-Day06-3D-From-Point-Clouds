"""Convert sparse-conv weights of an MMDetection3D checkpoint from spconv2 layout
(out, kD, kH, kW, in) to the mmcv-ops layout (kD, kH, kW, in, out).

Needed when spconv is not installed: mmcv's SparseConv expects the second layout,
and loading the original checkpoint only prints a "size mismatch" warning, leaving
the middle encoder randomly initialised (the detector then outputs zero boxes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="original checkpoint (.pth)")
    ap.add_argument("--dst", required=True, type=Path, help="converted checkpoint (.pth)")
    ap.add_argument("--prefix", default="middle_encoder.", help="only convert 5-D weights under this prefix")
    args = ap.parse_args()
    ckpt = torch.load(args.src, map_location="cpu")
    state = ckpt["state_dict"]
    converted = 0
    for key, value in state.items():
        if key.startswith(args.prefix) and value.dim() == 5:
            state[key] = value.permute(1, 2, 3, 4, 0).contiguous()
            converted += 1
    torch.save(ckpt, args.dst)
    print(f"converted {converted} sparse-conv weights -> {args.dst}")


if __name__ == "__main__":
    main()
