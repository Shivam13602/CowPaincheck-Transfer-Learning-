#!/usr/bin/env python3
"""Summarize V6 trial folders into a ranked table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> dict | None:
    s4 = run_dir / "dann_summary.json"
    s3 = run_dir / "weak_label_cv_summary.json"
    if s4.is_file():
        data = json.loads(s4.read_text(encoding="utf-8"))
        ft = data.get("final_test", {})
        kind = "S4"
    elif s3.is_file():
        data = json.loads(s3.read_text(encoding="utf-8"))
        ft = data.get("final_test", {})
        kind = "S3"
    else:
        return None

    seq = ft.get("sequence_metrics", {})
    vid = ft.get("video_metrics", {})
    cow = ft.get("cow_metrics", {})
    return {
        "run": run_dir.name,
        "kind": kind,
        "seq_auc": seq.get("auc"),
        "seq_f1": seq.get("f1"),
        "seq_recall": seq.get("recall"),
        "tp": seq.get("tp"),
        "fn": seq.get("fn"),
        "fp": seq.get("fp"),
        "tn": seq.get("tn"),
        "video_auc": vid.get("auc"),
        "cow_auc": cow.get("auc"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=15)
    args = p.parse_args()

    rows = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir():
            continue
        m = load_metrics(d)
        if m:
            rows.append(m)

    rows.sort(key=lambda r: (r["seq_auc"] or 0.0, r["seq_f1"] or 0.0), reverse=True)
    if not rows:
        print("No trial summaries found.")
        return 1

    print(
        f"{'run':28} {'kind':4} {'seq_auc':>8} {'seq_f1':>7} {'rec':>6} "
        f"{'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'vid_auc':>8} {'cow_auc':>8}"
    )
    for r in rows[: args.top_k]:
        print(
            f"{r['run'][:28]:28} {r['kind']:4} {r['seq_auc']:8.4f} {r['seq_f1']:7.3f} {r['seq_recall']:6.3f} "
            f"{r['tp']:4} {r['fn']:4} {r['fp']:4} {r['tn']:4} {r['video_auc']:8.4f} {r['cow_auc']:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
