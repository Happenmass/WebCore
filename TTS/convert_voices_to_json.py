#!/usr/bin/env python
# Convert Kokoro voice packs from .pt to .json for web usage.

import argparse
import json
from pathlib import Path

import torch


def convert_one(pt_path: Path, out_dir: Path):
    pack = torch.load(pt_path, weights_only=True).cpu().numpy()
    out_path = out_dir / f"{pt_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pack.tolist(), f)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro voice .pt to JSON.")
    parser.add_argument(
        "--voices-dir",
        default="TTS/hexgrad/Kokoro-82M-v1.1-zh/voices",
        help="Directory containing .pt voice packs.",
    )
    parser.add_argument(
        "--out-dir",
        default="TTS/web/voices",
        help="Output directory for JSON files.",
    )
    args = parser.parse_args()

    voices_dir = Path(args.voices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(voices_dir.glob("*.pt"))
    if not pt_files:
        raise SystemExit(f"No .pt files found in {voices_dir}")

    for pt in pt_files:
        out_path = convert_one(pt, out_dir)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
