#!/usr/bin/env python3
"""Print ranked V6 leaderboard from v6_results_analysis.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "v6_results_analysis.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=DEFAULT)
    args = p.parse_args()
    data = json.loads(args.json.read_text(encoding="utf-8"))
    rows = data["conditions"]
    print(
        f"{'condition':28} {'st':2} {'seq_auc':>8} {'seq_f1':>7} {'rec':>6} "
        f"{'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'f1_opt':>7} {'vid_auc':>8} {'cow_auc':>8} {'thr':>6}"
    )
    for r in rows:
        s = r["sequence"]
        v = r.get("video") or {}
        c = r.get("cow") or {}
        print(
            f"{r['condition'][:28]:28} {r['stage']:2} {s['auc']:8.4f} {s['f1']:7.3f} {s['recall']:6.3f} "
            f"{s['tp']:4} {s['fn']:4} {s['fp']:4} {s['tn']:4} {s['f1_opt']:7.3f} "
            f"{v.get('auc', 0) or 0:8.4f} {c.get('auc', 0) or 0:8.4f} {r.get('threshold', 0):6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
