"""Evaluation metrics.

The challenge scores the median, over bootstrap resamples at roughly 1% prevalence, of the positive
predictive value at 90% recall (PPV@90Recall). 

``bootstrap_evaluation`` reproduces the challenge's own resampling scheme so local numbers are comparable
to the leaderboard. ``partial_auc_high_sensitivity`` integrates specificity over a band of the ROC curve
around that operating point, which is more stable than reading a single point when positives are scarce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

_trapezoid = getattr(np, "trapezoid", np.trapz)


def partial_auc_high_sensitivity(y_true, y_score, min_tpr: float = 0.90,
                                 max_tpr: float = 1.00) -> float:
    """Normalized partial AUC over a true-positive-rate band: mean specificity while sensitivity is in
    ``[min_tpr, max_tpr]``. Rank-based, so it needs no calibration.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr_lo = float(np.interp(min_tpr, tpr, fpr))
    fpr_hi = float(np.interp(max_tpr, tpr, fpr))
    inside = (tpr >= min_tpr) & (tpr <= max_tpr)
    tpr_band = np.concatenate([[min_tpr], tpr[inside], [max_tpr]])
    fpr_band = np.concatenate([[fpr_lo], fpr[inside], [fpr_hi]])
    return float(1.0 - _trapezoid(fpr_band, tpr_band) / (max_tpr - min_tpr))


def fpr_at_recall(y_true, y_score, recall: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(recall, tpr, fpr))


def ppv_at_recall(y_true, y_score, recall: float = 0.90) -> float:
    """PPV at a given recall, interpolated on the empirical precision-recall curve."""
    precision, rec, _ = precision_recall_curve(y_true, y_score)
    return float(np.interp(recall, rec[::-1], precision[::-1]))


def compute_metrics(y_true, y_score) -> dict:
    return {
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "PPV@90Recall": ppv_at_recall(y_true, y_score),
        "FPR@90Recall": fpr_at_recall(y_true, y_score),
        "pAUC@90Recall": partial_auc_high_sensitivity(y_true, y_score),
        "pAUC[0.86,0.90]": partial_auc_high_sensitivity(y_true, y_score, 0.86, 0.90), 
    }


def bootstrap_evaluation(y_true, y_score, n_bootstrap: int = 1000, imbalance_ratio: int = 100,
                         random_state: int | None = None) -> pd.DataFrame:
    """Median and 95% interval of each metric under the challenge's resampling scheme.

    Every iteration keeps all negatives fixed and resamples the positives, with replacement, down to
    ``n_negatives / imbalance_ratio``, holding the prevalence at roughly 1%.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    negative_true, negative_score = y_true[y_true == 0], y_score[y_true == 0]
    positive_true, positive_score = y_true[y_true == 1], y_score[y_true == 1]
    n_positive = max(1, len(negative_true) // imbalance_ratio)

    rng = np.random.default_rng(random_state)
    records = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(positive_true), size=n_positive)
        records.append(compute_metrics(np.concatenate([negative_true, positive_true[idx]]),
                                       np.concatenate([negative_score, positive_score[idx]])))
    return pd.DataFrame(records).describe(percentiles=[0.025, 0.5, 0.975]).loc[["2.5%", "50%", "97.5%"]]
