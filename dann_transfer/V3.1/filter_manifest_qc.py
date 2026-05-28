#!/usr/bin/env python3
"""
Offline manifest QC for Holstein face sequences (V3.1).

Reads ``completed_manifest.csv``, applies detection-quality gates, writes:
  - filtered manifest CSV (subset of rows passing QC)
  - ``manifest_qc_report.csv`` with per-row audit reasons

Does not modify original frames on disk.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Holstein completed_manifest.csv by QC gates.")
    p.add_argument("--input", type=Path, required=True, help="Input completed_manifest.csv")
    p.add_argument("--output", type=Path, required=True, help="Output filtered manifest CSV path")
    p.add_argument("--qc-report", type=Path, default=None, help="Optional audit CSV path (default: alongside output)")
    p.add_argument("--min-mean-detection-confidence", type=float, default=None)
    p.add_argument("--max-filled-frames", type=int, default=None)
    p.add_argument("--min-detected-frames", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})

    audit_path = args.qc_report or args.output.with_name(args.output.stem + "_qc_audit.csv")
    kept_rows: list[dict[str, str]] = []
    audit_lines: list[dict[str, object]] = []

    for row in rows:
        reasons: list[str] = []
        mean_conf_s = row.get("mean_detection_confidence") or ""
        try:
            mean_conf = float(mean_conf_s) if mean_conf_s.strip() else None
        except ValueError:
            mean_conf = None

        if args.min_mean_detection_confidence is not None:
            if mean_conf is None:
                reasons.append("missing_mean_detection_confidence")
            elif mean_conf < float(args.min_mean_detection_confidence):
                reasons.append("mean_conf_below_threshold")

        try:
            filled = int(float(row.get("filled_frames") or 0))
        except ValueError:
            filled = 0
        if args.max_filled_frames is not None and filled > int(args.max_filled_frames):
            reasons.append("filled_frames_above_threshold")

        try:
            detected = int(float(row.get("detected_frames") or 0))
        except ValueError:
            detected = 0
        if args.min_detected_frames is not None and detected < int(args.min_detected_frames):
            reasons.append("detected_frames_below_threshold")

        audit_lines.append(
            {
                "sequence_index": row.get("sequence_index", ""),
                "cow_id": row.get("cow_id", ""),
                "kept": len(reasons) == 0,
                "skip_reasons": ";".join(reasons),
                "mean_detection_confidence": mean_conf,
                "filled_frames": filled,
                "detected_frames": detected,
            }
        )
        if not reasons:
            kept_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_lines[0].keys()) if audit_lines else ["sequence_index"])
        w.writeheader()
        w.writerows(audit_lines)

    print(f"Kept {len(kept_rows)} / {len(rows)} rows → {args.output}")
    print(f"QC audit → {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
