"""
Batch feature extraction for EuroSAT using DINOv2.

Usage:
    python scripts/extract_features.py                        # CLS only
    python scripts/extract_features.py --patches              # CLS + patches
    python scripts/extract_features.py --dataset eurosat --batch-size 256
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from dino_sense.model import DINOv2Extractor
from dino_sense.data import load_eurosat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eurosat", choices=["eurosat"])
    p.add_argument("--data-root", default="./data/eurosat")
    p.add_argument("--output-dir", default="./features")
    p.add_argument("--model", default="dinov2_vits14_reg")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patches", action="store_true", help="Also extract patch tokens (~5 GB float16)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.model}...")
    extractor = DINOv2Extractor(model_name=args.model, device=args.device)
    print(f"Device: {extractor.device}")

    print(f"Loading EuroSAT from {args.data_root}...")
    loader, classes = load_eurosat(root=args.data_root, batch_size=args.batch_size)
    print(f"Classes: {classes}")
    print(f"Batches: {len(loader)} × {args.batch_size}")

    if args.patches:
        cls, patches, labels = extractor.extract_both(loader)
        extractor.save_features(cls, labels, args.output_dir, patches=patches)
    else:
        cls, labels = extractor.extract_cls(loader)
        extractor.save_features(cls, labels, args.output_dir)

    # Quick sanity check
    print(f"\nSanity check:")
    print(f"  CLS shape:    {cls.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label range:  [{labels.min()}, {labels.max()}]")
    counts = labels.bincount()
    print(f"  Class counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
