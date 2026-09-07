"""Train one model.

Three protocols are supported, selected with ``--split``:

``cross_center``  train on one center, evaluate on the other. Measures transfer to an unseen center.
``pooled``        train on both centers, evaluate on a held-out slice of the same mixture. This is the
                  condition the deployed model is trained under and the one used to confirm whether a change is adopted.
``final``         train on every labeled image and export a deployable checkpoint. No evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import DEFAULT
from training.data import load_split
from validation.metrics import bootstrap_evaluation, compute_metrics
from model import NeoplasiaClassifier, add_lora, build_backbone, export_for_inference
from training.trainer import predict, train_model


def build(config, args, device):
    backbone = build_backbone(args.dinov2_repo, args.backbone_weights, config.img_size)
    model = NeoplasiaClassifier(backbone, pooling=config.pooling)
    return add_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["cross_center", "pooled", "final"], default="pooled")
    parser.add_argument("--split_csv", type=Path, default=Path("data/splits/pooled_holdout.csv"))
    parser.add_argument("--data_root", type=Path, default=Path("data/train"))
    parser.add_argument("--dinov2_repo", type=Path, default=Path("third_party/dinov2"))
    parser.add_argument("--backbone_weights", type=Path, default=None)
    parser.add_argument("--pooling", choices=["attention", "mean"], default=DEFAULT.pooling)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT.epochs)
    parser.add_argument("--export", type=Path, default=None,
                        help="write a deployable checkpoint here (required for --split final)")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = replace(DEFAULT, pooling=args.pooling, epochs=args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.split == "final":
        table = pd.read_csv(args.split_csv)
        train_paths = np.array([str(args.data_root / p) for p in table.image_path])
        train_labels = table.target.to_numpy(dtype=int)
        test_paths = test_labels = None
    else:
        train_paths, train_labels = load_split(args.split_csv, args.data_root, "train")
        test_paths, test_labels = load_split(args.split_csv, args.data_root, "test")

    print(f"split={args.split} pooling={config.pooling} seed={args.seed} | "
          f"train {len(train_labels)} ({int(train_labels.sum())} positive)")

    model = build(config, args, device)
    train_model(model, train_paths, train_labels, config, device)

    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        export_for_inference(model, args.export, config.img_size)

    if test_paths is None:
        return

    scores = predict(model, test_paths, test_labels, config, device)
    point = compute_metrics(test_labels, scores)
    interval = bootstrap_evaluation(test_labels, scores, random_state=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{config.pooling}_seed{args.seed}"
    pd.DataFrame({"image_path": test_paths, "target": test_labels, "score": scores}).to_csv(
        args.out_dir / f"predictions_{stem}.csv", index=False)
    (args.out_dir / f"metrics_{stem}.json").write_text(json.dumps(point, indent=2))
    interval.to_csv(args.out_dir / f"bootstrap_{stem}.csv")

    print("\npoint estimates:")
    for name, value in point.items():
        print(f"  {name:18s} {value:.4f}")
    print(f"\nbootstrap median and 95% interval:\n{interval.to_string()}")


if __name__ == "__main__":
    main()
