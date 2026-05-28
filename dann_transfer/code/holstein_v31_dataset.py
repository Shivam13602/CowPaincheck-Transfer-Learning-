"""
Holstein manifest loading for V3.1 — extends v2.9 bundle with optional QC filters.

QC uses columns already present in ``completed_manifest.csv``:
``mean_detection_confidence``, ``filled_frames``, ``detected_frames``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from holstein_v29_dataset import HolsteinManifestBundle, iter_manifest_rows, row_to_ucaps_entry


def load_holstein_bundle_v31(
    manifest_csv: Path,
    sequence_dataset_root: Path,
    *,
    min_mean_detection_confidence: float | None = None,
    max_filled_frames: int | None = None,
    min_detected_frames: int | None = None,
) -> tuple[HolsteinManifestBundle, list[dict[str, Any]]]:
    """
    Like ``load_holstein_bundle`` but drops sequences failing QC gates.

    Returns ``(bundle, qc_audit_rows)`` where each audit row describes keep/skip.
    """
    sequences: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    root = Path(sequence_dataset_root)

    for row in iter_manifest_rows(manifest_csv):
        seq, meta = row_to_ucaps_entry(row, root)
        idx = int(meta["sequence_index"])
        reasons: list[str] = []

        mean_conf = meta.get("mean_detection_confidence")
        if min_mean_detection_confidence is not None:
            if mean_conf is None:
                reasons.append("missing_mean_detection_confidence")
            elif float(mean_conf) < float(min_mean_detection_confidence):
                reasons.append(f"mean_conf_below_{min_mean_detection_confidence}")

        filled = int(meta.get("filled_frames", 0) or 0)
        if max_filled_frames is not None and filled > int(max_filled_frames):
            reasons.append(f"filled_frames_above_{max_filled_frames}")

        detected = int(meta.get("detected_frames", 0) or 0)
        if min_detected_frames is not None and detected < int(min_detected_frames):
            reasons.append(f"detected_frames_below_{min_detected_frames}")

        folder = Path(meta["sequence_folder"])
        if not folder.is_dir():
            reasons.append("missing_sequence_folder")

        audit.append(
            {
                "sequence_index": idx,
                "cow_id": meta.get("cow_id"),
                "kept": len(reasons) == 0,
                "skip_reasons": ";".join(reasons) if reasons else "",
                "mean_detection_confidence": mean_conf,
                "filled_frames": filled,
                "detected_frames": detected,
            }
        )

        if reasons:
            continue

        frames = sorted(folder.glob("frame_*.jpg")) + sorted(folder.glob("frame_*.png"))
        if not frames:
            audit[-1]["kept"] = False
            audit[-1]["skip_reasons"] = (audit[-1]["skip_reasons"] + ";no_frames").strip(";")
            continue

        sequences.append(seq)
        meta_rows.append(meta)

    bundle = HolsteinManifestBundle(sequences=sequences, metadata=meta_rows, sequence_dataset_root=root)
    return bundle, audit

