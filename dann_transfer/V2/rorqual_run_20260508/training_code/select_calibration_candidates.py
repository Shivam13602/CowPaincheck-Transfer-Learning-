#!/usr/bin/env python3
"""
Pick sequences for expert veterinary scoring using uncertainty (Task2 entropy) plus cow diversity.

Reads holstein_zero_shot_predictions_*.csv produced by evaluate_holstein_zero_shot_v2.9.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Select calibration / annotation candidates.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--n", type=int, default=45, help="Target number of clips (default 45).")
    p.add_argument("--out-csv", type=Path, default=Path("calibration_candidates.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv).copy()
    if "task2_entropy" not in df.columns:
        raise ValueError("predictions CSV must include task2_entropy (re-run zero-shot script).")

    df = df.sort_values("task2_entropy", ascending=False)

    sel_idx: list[int] = []
    seen_cow: set[str] = set()

    for _, row in df.iterrows():
        if len(sel_idx) >= args.n:
            break
        cow = str(row["cow_id"])
        if cow in seen_cow:
            continue
        sel_idx.append(int(row["sequence_index"]))
        seen_cow.add(cow)

    if len(sel_idx) < args.n:
        for _, row in df.iterrows():
            if len(sel_idx) >= args.n:
                break
            i = int(row["sequence_index"])
            if i in sel_idx:
                continue
            sel_idx.append(i)

    sel = df[df["sequence_index"].isin(sel_idx)].sort_values("task2_entropy", ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(sel)} rows to {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
