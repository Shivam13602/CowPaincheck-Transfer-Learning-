"""Rebuild completed_manifest.csv from on-disk metadata.json files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"
STATS_PATH = OUT / "processing_statistics.json"
MANIFEST_PATH = OUT / "completed_manifest.csv"


def load_metadata_rows() -> list[dict]:
    rows: list[dict] = []
    for base in (OUT / "sequences" / "healthy", OUT / "sequences" / "unhealthy"):
        if not base.is_dir():
            continue
        for meta_path in base.rglob("metadata.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["sequence_dir"] = str(meta_path.parent)
            rows.append(meta)
    rows.sort(key=lambda r: int(r["sequence_index"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = load_metadata_rows()
    write_csv(MANIFEST_PATH, rows)

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    stats["completed_sequences"] = len(rows)
    stats["completed_summary"] = {
        "cow_health_counts": {},
        "video_health_counts": {},
        "session_counts": {},
        "unique_cows": len({r["cow_id"] for r in rows}),
        "dataset_counts": {},
    }
    from collections import Counter

    stats["completed_summary"]["cow_health_counts"] = dict(Counter(r["cow_health_status"] for r in rows))
    stats["completed_summary"]["video_health_counts"] = dict(Counter(r["video_health_status"] for r in rows))
    stats["completed_summary"]["session_counts"] = dict(Counter(r["session_category"] for r in rows))
    stats["completed_summary"]["dataset_counts"] = dict(Counter(r["dataset_root"] for r in rows))
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Rebuilt manifest with {len(rows)} sequences from on-disk metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
