"""
Build UCAPS v2.9-compatible sequence entries from cow_face_sequences_10s_250 completed_manifest.csv.

Each row resolves to the same output folder path used by yolo_cow_face/create_10s_face_sequences.py
(sequence_{idx:04d}_{dataset_slug}_{hash}/).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def slugify(text: str, fallback: str = "item", max_len: int = 70) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (slug or fallback)[:max_len]


def stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def resolve_sequence_dir(sequence_dataset_root: Path, row: dict[str, str]) -> Path:
    """Return the directory containing frame_*.jpg for one completed_manifest row."""
    dataset_version = str(row.get("dataset_version", "")).strip()
    if dataset_version == "thesis_stride8_qa":
        return _resolve_thesis_stride8_dir(sequence_dataset_root, row)

    sequence_index = int(float(row["sequence_index"]))
    dataset_root = str(row["dataset_root"]).strip()
    relative_path = str(row["relative_path"]).strip()
    cow_id = str(row["cow_id"]).strip()
    cow_health = str(row["cow_health_status"]).strip()
    path_hash = stable_hash(f"{dataset_root}/{relative_path}", length=8)
    name = f"sequence_{sequence_index:04d}_{slugify(dataset_root, max_len=22)}_{path_hash}"
    cow_folder = f"cow_{cow_id}_{cow_health.lower()}"
    return (
        Path(sequence_dataset_root)
        / "sequences"
        / cow_health.lower()
        / cow_folder
        / name
    )


def _resolve_thesis_stride8_dir(sequence_dataset_root: Path, row: dict[str, str]) -> Path:
    """Match folders created by create_thesis_stride8_sequences.py."""
    sequence_index = int(float(row["sequence_index"]))
    dataset_root = str(row["dataset_root"]).strip()
    relative_path = str(row["relative_path"]).strip()
    cow_id = str(row["cow_id"]).strip()
    cow_health = str(row["cow_health_status"]).strip().lower()
    start_second = float(row.get("start_second", 0) or 0)
    path_hash = stable_hash(f"{dataset_root}/{relative_path}", length=8)
    start_tag = f"t{int(round(start_second)):03d}s"
    slug = slugify(dataset_root, max_len=18)
    cow_folder = sequence_dataset_root / "sequences" / cow_health / f"cow_{cow_id}_{cow_health}"
    candidates = [
        cow_folder / f"sequence_{slug}_{path_hash}_{start_tag}_seq{sequence_index:05d}",
        cow_folder / f"sequence_{slug}_{path_hash}_{start_tag}",
    ]
    for folder in candidates:
        if folder.is_dir():
            return folder
    if cow_folder.is_dir():
        for meta_path in cow_folder.glob("*/metadata.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if int(meta.get("sequence_index", -1)) == sequence_index:
                return meta_path.parent
    raise FileNotFoundError(
        f"Missing thesis_stride8_qa folder for sequence_index={sequence_index} under {cow_folder}"
    )


def iter_manifest_rows(manifest_csv: Path) -> Iterator[dict[str, str]]:
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: (v if v is not None else "") for k, v in row.items()}


def row_to_ucaps_entry(row: dict[str, str], sequence_dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    UCAPS FacialPainDataset expects sequence_path relative to sequence_dir (dataset root).
    moment=M0 => Task1 placeholder non-pain; logits are the inference targets for Holstein.
    animal=int(sequence_index) gives stable grouping without colliding on cow_id repeats across clips.
    """
    folder = resolve_sequence_dir(sequence_dataset_root, row)
    rel = folder.relative_to(Path(sequence_dataset_root))
    idx = int(float(row["sequence_index"]))
    seq_id = f"holstein_seq_{idx:04d}"
    cow_id_s = str(row["cow_id"]).strip()
    try:
        animal_id = int(float(cow_id_s))
    except ValueError:
        animal_id = idx
    seq = {
        "sequence_id": seq_id,
        "sequence_path": rel.as_posix(),
        "animal": animal_id,
        "moment": "M0",
    }
    meta = {
        "sequence_index": idx,
        "cow_id": str(row["cow_id"]).strip(),
        "cow_health_status": str(row["cow_health_status"]).strip(),
        "video_health_status": str(row["video_health_status"]).strip(),
        "health_condition": str(row.get("health_condition", "")).strip(),
        "dataset_root": str(row["dataset_root"]).strip(),
        "relative_path": str(row["relative_path"]).strip(),
        "detected_frames": int(float(row.get("detected_frames", 0) or 0)),
        "filled_frames": int(float(row.get("filled_frames", 0) or 0)),
        "mean_detection_confidence": float(row["mean_detection_confidence"])
        if row.get("mean_detection_confidence")
        else None,
        "sequence_folder": str(folder),
    }
    return seq, meta


@dataclass(frozen=True)
class HolsteinManifestBundle:
    sequences: list[dict[str, Any]]
    metadata: list[dict[str, Any]]
    sequence_dataset_root: Path


def load_holstein_bundle(manifest_csv: Path, sequence_dataset_root: Path) -> HolsteinManifestBundle:
    sequences: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    root = Path(sequence_dataset_root)
    for row in iter_manifest_rows(manifest_csv):
        seq, meta = row_to_ucaps_entry(row, root)
        folder = Path(meta["sequence_folder"])
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing sequence folder for index {meta['sequence_index']}: {folder}")
        frames = sorted(folder.glob("frame_*.jpg")) + sorted(folder.glob("frame_*.png"))
        if not frames:
            raise FileNotFoundError(f"No frames under {folder}")
        sequences.append(seq)
        meta_rows.append(meta)
    return HolsteinManifestBundle(sequences=sequences, metadata=meta_rows, sequence_dataset_root=root)
