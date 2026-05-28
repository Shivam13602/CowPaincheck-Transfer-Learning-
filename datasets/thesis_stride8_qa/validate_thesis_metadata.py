#!/usr/bin/env python3
"""Validate thesis_stride8_qa manifests and per-sequence metadata consistency."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ALLOWED_UNHEALTHY_SESSIONS = {"during_exercise", "after_exercise", "sudden_fall"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "Transferlearning/cow_face_sequences_thesis_stride8/output"
    )
    manifest_path = output / "completed_manifest.csv"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}")
        return 2

    rows = load_csv(manifest_path)
    errors: list[str] = []

    index_by_seq: dict[str, Path] = {}
    for base in (output / "sequences" / "healthy", output / "sequences" / "unhealthy"):
        if not base.is_dir():
            continue
        for meta_file in base.rglob("metadata.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            index_by_seq[str(meta.get("sequence_index"))] = meta_file

    for row in rows:
        seq_idx = row.get("sequence_index", "?")
        meta_path = index_by_seq.get(str(seq_idx))

        if meta_path is None:
            errors.append(f"sequence_index {seq_idx}: metadata.json not found on disk")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checks = [
            ("cow_id", row.get("cow_id"), meta.get("cow_id")),
            ("cow_health_status", row.get("cow_health_status"), meta.get("cow_health_status")),
            ("video_health_status", row.get("video_health_status"), meta.get("video_health_status")),
            ("session_category", row.get("session_category"), meta.get("session_category")),
            ("start_second", row.get("start_second"), meta.get("start_second")),
            ("end_second", row.get("end_second"), meta.get("end_second")),
            ("dataset_version", "thesis_stride8_qa", meta.get("dataset_version")),
        ]
        for field, expected, actual in checks:
            if str(expected) != str(actual):
                errors.append(
                    f"sequence_index {seq_idx} field {field}: manifest={expected!r} metadata={actual!r}"
                )

        if meta.get("cow_health_status") == "Unhealthy" and meta.get("session_category") not in ALLOWED_UNHEALTHY_SESSIONS:
            errors.append(
                f"sequence_index {seq_idx}: unhealthy cow has disallowed session {meta.get('session_category')!r}"
            )

        frames = list(meta_path.parent.glob("frame_*.jpg"))
        expected_frames = int(meta.get("frames_per_sequence", 240))
        if len(frames) != expected_frames:
            errors.append(
                f"sequence_index {seq_idx}: expected {expected_frames} frames, found {len(frames)} in {meta_path.parent}"
            )

        stride = float(meta.get("window_stride_seconds", 0))
        overlap = float(meta.get("window_overlap_seconds", 0))
        if abs(stride - 8.0) > 1e-3 or abs(overlap - 2.0) > 1e-3:
            errors.append(
                f"sequence_index {seq_idx}: stride/overlap mismatch stride={stride} overlap={overlap}"
            )

    selected = load_csv(output / "selected_videos.csv") if (output / "selected_videos.csv").is_file() else []
    for row in selected:
        if row.get("cow_health_status") == "Unhealthy" and row.get("session_category") not in ALLOWED_UNHEALTHY_SESSIONS:
            errors.append(
                f"selected video {row.get('relative_path')}: unhealthy session {row.get('session_category')!r}"
            )

    print(f"Validated {len(rows)} completed sequences.")
    print(f"Selected source videos: {len(selected)}")
    if errors:
        print(f"FAILED with {len(errors)} issue(s):")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    print("PASS: manifests, metadata, frame counts, session rules, and stride/overlap are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
