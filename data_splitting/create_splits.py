"""Generate the data splits used for training and validation.

The RARE26 training set contains images from two centres. Four split definitions are produced, each as a
CSV in ``data/splits``:

``center1_train_center2_test.csv``  train on centre 1, evaluate on centre 2
``center2_train_center1_test.csv``  train on centre 2, evaluate on centre 1
``pooled_holdout.csv``              both centres pooled for training, with a held-out slice stratified
                                    jointly by centre and label
``5fold_cv.csv``                    stratified 5-fold cross-validation over all images

The two cross-centre splits measure transfer to an unseen centre. The pooled split matches the condition
the deployed model is trained under, since the final model is fitted on both centres; it is the split used
to decide whether a change is adopted.

Each CSV has the columns:
    image_path   path to the image, relative to the data root
    sample_id    file name, unique per image
    center       ``center_1`` or ``center_2``
    class_name   ``ndbe`` (non-dysplastic Barrett's oesophagus) or ``neo`` (neoplasia)
    target       0 for ndbe, 1 for neo
    split        ``train`` or ``test`` (``fold_0`` .. ``fold_4`` for cross-validation)

Usage:
    python data_splitting/create_splits.py --data_root data/train --out_dir data/splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASS_TO_TARGET = {"ndbe": 0, "neo": 1}
POOLED_TEST_FRACTION = 0.2
N_FOLDS = 5
SEED = 42


def enumerate_images(data_root: Path) -> pd.DataFrame:
    """Enumerate ``<data_root>/<center>/<class>/*.png`` into a sorted table."""
    rows = []
    for center_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for class_name, target in CLASS_TO_TARGET.items():
            for image in sorted((center_dir / class_name).glob("*.png")):
                rows.append({
                    "image_path": str(image.relative_to(data_root)),
                    "sample_id": image.name,
                    "center": center_dir.name,
                    "class_name": class_name,
                    "target": target,
                })
    if not rows:
        raise FileNotFoundError(
            f"no images found under {data_root}; expected <center>/<ndbe|neo>/*.png")
    return pd.DataFrame(rows).sort_values("image_path").reset_index(drop=True)


def cross_center_split(df: pd.DataFrame, train_center: str, test_center: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out.center.map({train_center: "train", test_center: "test"})
    return out.dropna(subset=["split"])


def pooled_holdout_split(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    """Hold out a slice of the pooled data, stratified jointly by centre and label.

    Stratifying on the pair rather than on the label alone keeps both centres proportionally represented
    in the held-out slice, so the slice reflects the mixture the model is trained on.
    """
    out = df.copy()
    stratify_key = out.center + "_" + out.target.astype(str)
    train_idx, test_idx = train_test_split(
        out.index, test_size=test_fraction, stratify=stratify_key, random_state=seed)
    out.loc[train_idx, "split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def kfold_split(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(out, out.target)):
        out.loc[out.index[val_idx], "split"] = f"fold_{fold}"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data/train"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = enumerate_images(args.data_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = df.groupby(["center", "class_name"]).size().unstack(fill_value=0)
    print(f"{len(df)} images\n{counts}\n")

    outputs = {
        "center1_train_center2_test.csv": cross_center_split(df, "center_1", "center_2"),
        "center2_train_center1_test.csv": cross_center_split(df, "center_2", "center_1"),
        "pooled_holdout.csv": pooled_holdout_split(df, POOLED_TEST_FRACTION, args.seed),
        "5fold_cv.csv": kfold_split(df, N_FOLDS, args.seed),
    }
    for name, table in outputs.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        summary = table.groupby("split").agg(n=("target", "size"), n_positive=("target", "sum"))
        print(f"{path}\n{summary}\n")


if __name__ == "__main__":
    main()
