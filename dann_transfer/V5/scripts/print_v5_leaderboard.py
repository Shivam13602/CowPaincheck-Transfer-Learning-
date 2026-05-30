#!/usr/bin/env python3
"""Print leaderboard from v5_results_analysis.json."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "results" / "v5_results_analysis.json"
d = json.loads(p.read_text(encoding="utf-8"))
rows = []
for c in d["conditions"]:
    seq = c["sequence"]
    vid = c.get("video") or {}
    cow = c["cow"]
    cal = c.get("calibrated_sequence") or {}
    label = c["condition"]
    rows.append(
        (
            label,
            c.get("stage"),
            seq.get("auc"),
            seq.get("f1"),
            seq.get("recall"),
            seq.get("tp"),
            seq.get("fn"),
            seq.get("fp"),
            seq.get("tn"),
            vid.get("auc"),
            cow.get("auc"),
            cal.get("f1"),
            cal.get("recall"),
        )
    )
rows.sort(key=lambda r: (r[2] or 0), reverse=True)
print(f"{'condition':28} {'stage':4} {'seq_auc':>8} {'seq_f1':>7} {'rec':>6} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'vid_auc':>8} {'cow_auc':>8} {'cal_f1':>7} {'cal_rec':>7}")
for r in rows:
    print(f"{r[0]:28} {r[1] or '':4} {r[2]:8.4f} {r[3]:7.3f} {r[4]:6.3f} {r[5]:4} {r[6]:4} {r[7]:4} {r[8]:4} {r[9]:8.4f} {r[10]:8.4f} {r[11]:7.3f} {r[12]:7.3f}")
