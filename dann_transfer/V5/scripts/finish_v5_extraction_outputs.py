#!/usr/bin/env python3
"""Finalize v5 extraction after a crash on rejected_windows.csv write.

Reads completed_manifest.csv + candidate_windows.csv, writes processing_statistics.json
and rejected_windows.csv (candidates not in manifest). Does not re-run YOLO.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "CowPaincheck-Transfer-Learning" / "datasets" / "thesis_stride8_qa"))
from create_thesis_stride8_sequences import write_csv  # noqa: E402


def main() -> int:
    out = ROOT / "Transferlearning" / "cow_face_sequences_thesis_stride8_v5" / "output"
    manifest = out / "completed_manifest.csv"
    candidates = out / "candidate_windows.csv"
    if not manifest.is_file():
        print(f"Missing {manifest}", file=sys.stderr)
        return 2

    import csv

    with manifest.open(newline="", encoding="utf-8") as f:
        completed = list(csv.DictReader(f))
    done_candidates = {str(r.get("candidate_index", "")).strip() for r in completed if r.get("candidate_index")}

    rejected: list[dict] = []
    if candidates.is_file():
        with candidates.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row.get("candidate_index", "")).strip()
                if cid not in done_candidates:
                    row.setdefault("reject_reason", "not_in_completed_manifest")
                    rejected.append(dict(row))

    stats_path = out / "processing_statistics.json"
    stats = {}
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["completed_sequences"] = len(completed)
    stats["rejected_windows"] = len(rejected)
    stats["candidate_windows"] = stats.get("candidate_windows") or (len(completed) + len(rejected))
    stats["completed_summary"] = {
        "cow_health_counts": dict(Counter(r.get("cow_health_status", "") for r in completed)),
        "video_health_counts": dict(Counter(r.get("video_health_status", "") for r in completed)),
        "session_counts": dict(Counter(r.get("session_category", "") for r in completed)),
        "unique_cows": len({r.get("cow_id") for r in completed}),
        "dataset_counts": dict(Counter(r.get("dataset_root", "") for r in completed)),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_csv(out / "rejected_windows.csv", rejected)
    print(f"completed={len(completed)} rejected={len(rejected)} stats={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
