"""
Threshold selection and degeneracy checks for pooled validation / final-test reporting.

Used by dann_adapt_v3_1.py and weak_label_adapt_v3_1.py (V3.1 literature-aligned protocol).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def best_f1_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Grid search thresholds from unique scores; return (threshold, best_f1)."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if scores.size == 0:
        return 0.5, 0.0
    uniq = np.unique(scores)
    if uniq.size > 512:
        qs = np.linspace(0.0, 1.0, num=512)
        thr_grid = np.quantile(scores, qs)
        uniq = np.unique(thr_grid)
    best_thr = float(np.median(scores))
    best_f1 = -1.0
    tot_pos = float(targets.sum())
    tot_neg = float(len(targets) - tot_pos)
    if tot_pos <= 0 or tot_neg <= 0:
        return best_thr, 0.0
    order = np.argsort(scores)
    s_sorted = scores[order]
    t_sorted = targets[order]
    tp = float(t_sorted.sum())
    fp = 0.0
    fn = 0.0
    tn = tot_neg
    prev = None
    for thr, lab in zip(s_sorted.tolist(), t_sorted.tolist()):
        if prev is not None and thr != prev:
            prec = tp / max(tp + fp, 1e-12)
            rec = tp / max(tp + fn, 1e-12)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(prev + (thr - prev) / 2.0)
        prev = thr
        if lab == 1:
            tp -= 1.0
            fn += 1.0
        else:
            tn -= 1.0
            fp += 1.0
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1e-12)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(s_sorted[-1])
    return best_thr, float(best_f1)


def predictions_degenerate(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Detect all-positive / all-negative thresholding on a finite sample."""
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return {"all_positive": False, "all_negative": True, "n": 0}
    pred = scores >= float(threshold)
    pos_rate = float(pred.mean())
    return {
        "all_positive": bool(pos_rate >= 1.0),
        "all_negative": bool(pos_rate <= 0.0),
        "positive_rate": pos_rate,
        "n": int(scores.size),
    }


def specificity_at_threshold(targets: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= float(threshold)
    cm = confusion_matrix(targets, pred, labels=[0, 1])
    if cm.size != 4:
        return float("nan")
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / max(tn + fp, 1))
    return spec


def resolve_threshold_for_test(
    policy: str,
    *,
    fold_summaries: list[dict[str, Any]],
    all_val_pred: pd.DataFrame,
    scores_col: str,
    calibrated: bool,
    fixed_threshold: float,
) -> tuple[float, dict[str, Any]]:
    """
    Pick a threshold for final-test metrics.

    Policies:
      - mean_fold_best_f1: mean of per-fold val ``best_threshold`` (legacy V2 behaviour).
      - median_fold_best_f1: median of those thresholds.
      - pooled_val_f1_opt: single threshold maximizing F1 on pooled validation rows.
      - fixed: use ``fixed_threshold``.
    """
    meta: dict[str, Any] = {"policy": policy, "calibrated": bool(calibrated)}
    key_metrics = "val_calibrated_metrics" if calibrated else "val_metrics"
    thr_key = "best_threshold"

    fold_vals = [
        float(s[key_metrics][thr_key])
        for s in fold_summaries
        if isinstance(s.get(key_metrics), dict) and s[key_metrics].get(thr_key) is not None
    ]

    if policy == "fixed":
        meta["source"] = "fixed_cli"
        return float(fixed_threshold), meta

    if policy == "mean_fold_best_f1":
        thr = float(np.mean(fold_vals)) if fold_vals else 0.5
        meta["source"] = "mean_fold_best_f1"
        meta["n_folds_with_threshold"] = len(fold_vals)
        return thr, meta

    if policy == "median_fold_best_f1":
        thr = float(np.median(fold_vals)) if fold_vals else 0.5
        meta["source"] = "median_fold_best_f1"
        meta["n_folds_with_threshold"] = len(fold_vals)
        return thr, meta

    if policy == "pooled_val_f1_opt":
        if all_val_pred.empty or scores_col not in all_val_pred.columns:
            thr = float(np.mean(fold_vals)) if fold_vals else 0.5
            meta["source"] = "fallback_mean_fold_best_f1_missing_pool"
            meta["warning"] = "empty_or_missing_scores_column"
            return thr, meta
        y = all_val_pred["target"].values.astype(np.int64)
        s = all_val_pred[scores_col].values.astype(np.float64)
        thr, f1 = best_f1_threshold(s, y)
        meta["source"] = "pooled_val_f1_opt"
        meta["pooled_val_f1_opt"] = float(f1)
        return float(thr), meta

    raise ValueError(f"Unknown threshold policy: {policy!r}")
