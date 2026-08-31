"""Score a saved prediction file, or an ensemble of them, with the challenge metrics.

Ensemble members are combined by averaging their per-frame probabilities, which is how the submitted
container combines its members.

Usage:
    python evaluate.py results/predictions_pooled_attention_seed42.csv
    python evaluate.py results/predictions_pooled_attention_seed4*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rare26.metrics import bootstrap_evaluation, compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=42, help="seed for the bootstrap resampling")
    args = parser.parse_args()

    frames = [pd.read_csv(p).sort_values("image_path").reset_index(drop=True)
              for p in args.predictions]
    reference = frames[0]
    for frame in frames[1:]:
        if not reference.image_path.equals(frame.image_path):
            sys.exit("prediction files cover different images and cannot be combined")

    scores = np.mean([f.score.to_numpy() for f in frames], axis=0)
    targets = reference.target.to_numpy(dtype=int)
    print(f"{len(frames)} member(s), {len(targets)} frames, {int(targets.sum())} positive\n")

    for name, value in compute_metrics(targets, scores).items():
        print(f"  {name:18s} {value:.4f}")
    print(f"\nbootstrap median and 95% interval:\n"
          f"{bootstrap_evaluation(targets, scores, random_state=args.seed).to_string()}")


if __name__ == "__main__":
    main()
