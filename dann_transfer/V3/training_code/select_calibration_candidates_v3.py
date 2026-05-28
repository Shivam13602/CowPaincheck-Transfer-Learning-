#!/usr/bin/env python3
"""
Select Holstein/Jersey clips for blinded veterinary scoring.

V3 ranks clips by uncertainty, model disagreement, and diversity across cows,
health conditions, and source videos. The output is an annotation worklist; it
does not create pain labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _binary_entropy(p: pd.Series) -> pd.Series:
    p2 = p.astype(float).clip(1e-12, 1.0 - 1e-12)
    return -(p2 * np.log(p2) + (1.0 - p2) * np.log(1.0 - p2))


def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo = float(s.min()) if len(s) else 0.0
    hi = float(s.max()) if len(s) else 0.0
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(s), dtype=np.float64), index=s.index)
    return (s - lo) / (hi - lo)


def _read_prediction_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path).copy()
        if "sequence_index" not in df.columns:
            raise ValueError(f"{path} is missing sequence_index.")
        df["model_source"] = path.stem
        frames.append(df)
    if not frames:
        raise ValueError("At least one --predictions-csv is required.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _aggregate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if "pain_prob" in df.columns:
        df["pain_uncertainty_entropy"] = _binary_entropy(df["pain_prob"])
    elif "task2_entropy" in df.columns:
        df["pain_uncertainty_entropy"] = pd.to_numeric(df["task2_entropy"], errors="coerce")
    else:
        raise ValueError("Predictions must include pain_prob or task2_entropy.")

    key_cols = ["sequence_index"]
    first_cols = [
        c
        for c in [
            "cow_id",
            "target",
            "target_label",
            "video_health_status",
            "cow_health_status",
            "health_condition",
            "dataset_root",
            "relative_path",
            "sequence_folder",
            "detection_rate",
            "filled_rate",
            "mean_detection_confidence",
            "min_detection_confidence",
        ]
        if c in df.columns
    ]
    agg_spec: dict[str, Any] = {c: (c, "first") for c in first_cols}
    agg_spec.update(
        {
            "n_model_predictions": ("model_source", "nunique"),
            "model_sources": ("model_source", lambda x: ",".join(sorted(set(map(str, x))))),
            "uncertainty_mean": ("pain_uncertainty_entropy", "mean"),
            "uncertainty_max": ("pain_uncertainty_entropy", "max"),
        }
    )
    if "pain_prob" in df.columns:
        agg_spec.update(
            {
                "pain_prob_mean": ("pain_prob", "mean"),
                "pain_prob_std": ("pain_prob", "std"),
                "pain_prob_min": ("pain_prob", "min"),
                "pain_prob_max": ("pain_prob", "max"),
            }
        )
    if "pain_logit_mc_std" in df.columns:
        agg_spec["mc_std_mean"] = ("pain_logit_mc_std", "mean")
    out = df.groupby(key_cols, as_index=False).agg(**agg_spec)
    if "pain_prob_std" not in out.columns:
        out["pain_prob_std"] = 0.0
    out["pain_prob_std"] = out["pain_prob_std"].fillna(0.0)
    if "mc_std_mean" not in out.columns:
        out["mc_std_mean"] = 0.0
    out["vet_priority_score"] = (
        0.45 * _norm(out["uncertainty_mean"])
        + 0.35 * _norm(out["pain_prob_std"])
        + 0.20 * _norm(out["mc_std_mean"])
    )
    return out.sort_values(["vet_priority_score", "uncertainty_mean"], ascending=False).reset_index(drop=True)


def _diverse_select(df: pd.DataFrame, *, n: int, max_per_cow: int, max_per_condition: int, max_per_source: int) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    seen_seq: set[int] = set()
    cow_counts: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    def can_take(row: pd.Series, strict: bool) -> bool:
        if int(row["sequence_index"]) in seen_seq:
            return False
        if not strict:
            return True
        cow = str(row.get("cow_id", ""))
        cond = str(row.get("health_condition", row.get("target_label", "")))
        src = str(row.get("dataset_root", row.get("relative_path", "")))
        if cow_counts.get(cow, 0) >= max_per_cow:
            return False
        if cond and condition_counts.get(cond, 0) >= max_per_condition:
            return False
        if src and source_counts.get(src, 0) >= max_per_source:
            return False
        return True

    for strict in (True, False):
        for _, row in df.iterrows():
            if len(selected_rows) >= n:
                break
            if not can_take(row, strict=strict):
                continue
            selected_rows.append(row)
            seq = int(row["sequence_index"])
            seen_seq.add(seq)
            cow = str(row.get("cow_id", ""))
            cond = str(row.get("health_condition", row.get("target_label", "")))
            src = str(row.get("dataset_root", row.get("relative_path", "")))
            cow_counts[cow] = cow_counts.get(cow, 0) + 1
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1
        if len(selected_rows) >= n:
            break

    if not selected_rows:
        return pd.DataFrame(columns=list(df.columns))
    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1))
    out["selection_reason"] = "uncertainty_disagreement_diversity"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Select V3 veterinary-scoring candidates from model predictions.")
    p.add_argument("--predictions-csv", type=Path, action="append", required=True, help="Prediction CSV. Repeat for multiple models.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-per-cow", type=int, default=3)
    p.add_argument("--max-per-condition", type=int, default=10)
    p.add_argument("--max-per-source", type=int, default=15)
    p.add_argument("--out-csv", type=Path, default=Path("v3_vet_scoring_candidates.csv"))
    p.add_argument("--summary-json", type=Path, default=None)
    args = p.parse_args()

    raw = _read_prediction_csvs([p.resolve() for p in args.predictions_csv])
    ranked = _aggregate_predictions(raw)
    selected = _diverse_select(
        ranked,
        n=int(args.n),
        max_per_cow=int(args.max_per_cow),
        max_per_condition=int(args.max_per_condition),
        max_per_source=int(args.max_per_source),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out_csv, index=False)

    summary = {
        "n_input_rows": int(len(raw)),
        "n_unique_sequences": int(ranked["sequence_index"].nunique()),
        "n_selected": int(len(selected)),
        "prediction_csvs": [str(p.resolve()) for p in args.predictions_csv],
        "out_csv": str(args.out_csv.resolve()),
    }
    summary_path = args.summary_json or args.out_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} candidate rows to {args.out_csv.resolve()}")
    print(f"Wrote summary to {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
