#!/usr/bin/env python3
"""
Evaluate weak labels (healthy vs unhealthy proxy) against zero-shot Task1 scores.

Treats disease-context / cow-context labels as noisy proxies — reports separation metrics only,
not validated pain recognition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None  # type: ignore[misc, assignment]


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if roc_auc_score is None:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def main() -> int:
    p = argparse.ArgumentParser(description="Weak-label proxy metrics from zero-shot predictions CSV.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv)
    v_sub = df[df["video_health_status"].isin(["Healthy", "Unhealthy"])].copy()
    v_y = (v_sub["video_health_status"] == "Unhealthy").astype(np.int64).values
    v_scores = v_sub["pain_prob"].values.astype(np.float64)

    c_sub = df[df["cow_health_status"].isin(["Healthy", "Unhealthy"])].copy()
    c_y = (c_sub["cow_health_status"] == "Unhealthy").astype(np.int64).values
    c_scores = c_sub["pain_prob"].values.astype(np.float64)

    lines = []
    lines.append(f"n_sequences={len(df)}")
    auc_v = _binary_auc(v_y, v_scores)
    auc_c = _binary_auc(c_y, c_scores)
    lines.append(f"AUC_video_health_proxy={auc_v}")
    lines.append(f"AUC_cow_health_proxy={auc_c}")

    by_cond = df.groupby("health_condition")["pain_prob"].agg(["count", "mean", "std"]).reset_index()
    lines.append("\nMean pain_prob by health_condition:\n" + by_cond.to_string(index=False))

    text = "\n".join(lines)
    print(text)

    if args.out_json:
        import json

        payload = {
            "n": int(len(df)),
            "auc_video_health_proxy": auc_v,
            "auc_cow_health_proxy": auc_c,
            "mean_pain_prob_by_condition": by_cond.to_dict(orient="records"),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
