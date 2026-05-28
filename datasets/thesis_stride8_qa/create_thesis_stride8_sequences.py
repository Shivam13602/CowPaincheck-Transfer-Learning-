#!/usr/bin/env python3
"""
Create thesis Holstein/Jersey cow-face sequences with balanced video selection.

Rules:
- Unhealthy cows: only During exercise, After exercise, and Cow 349 sudden-fall videos.
- Healthy cows: any session (before / during / after exercise).
- Same target video count per cow (--videos-per-cow); cows with fewer eligible videos use all
  available and are flagged in the selection manifest.
- Sliding windows: 10 s length, 8 s stride (2 s overlap), 240 frames at 24 FPS.
- QA-filtered output compatible with holstein_v29_dataset / V3 trainers.

Does not modify baseline_10s_250.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_HEALTH_LABELS = {"Healthy", "Unhealthy"}
UNHEALTHY_ALLOWED_SESSIONS = {"during_exercise", "after_exercise", "sudden_fall"}
DATASET_VERSION = "thesis_stride8_qa"


@dataclass(frozen=True)
class VideoRecord:
    row_id: int
    dataset_root: str
    relative_path: str
    video_path: Path
    filename: str
    cow_id: str
    video_health_status: str
    cow_health_status: str
    health_condition: str
    top_level_folder: str
    second_level_folder: str
    session_category: str
    frame_count: int
    fps: float
    duration_sec: float


@dataclass(frozen=True)
class WindowRecord:
    candidate_index: int
    record: VideoRecord
    start_second: float
    end_second: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thesis cow-face 10s sliding-window sequence extraction.")
    p.add_argument(
        "--inventory",
        type=Path,
        default=Path("cow_video_dataset_analysis.csv"),
        help="Video inventory CSV from analyze_cow_video_datasets.py.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root containing dataset folders when CSV absolute paths are unavailable.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Trained YOLO cow-face weights (best.pt). Required unless --dry-run.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Transferlearning/cow_face_sequences_thesis_stride8/output"),
        help="Output directory for sequences and manifests.",
    )
    p.add_argument("--videos-per-cow", type=int, default=3, help="Target number of source videos per cow.")
    p.add_argument("--seed", type=int, default=42, help="Reproducible video selection seed.")
    p.add_argument("--sequence-seconds", type=float, default=10.0)
    p.add_argument("--stride-seconds", type=float, default=8.0, help="8 s stride => 2 s overlap for 10 s windows.")
    p.add_argument("--target-fps", type=float, default=24.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--conf", type=float, default=0.60)
    p.add_argument("--crop-pad", type=float, default=0.08)
    p.add_argument("--min-detection-rate", type=float, default=0.90)
    p.add_argument("--max-filled-rate", type=float, default=0.10)
    p.add_argument("--min-mean-confidence", type=float, default=0.80)
    p.add_argument("--min-min-confidence", type=float, default=0.60)
    p.add_argument("--ignore-cow-ids", default="409")
    p.add_argument(
        "--strict-videos-per-cow",
        action="store_true",
        help="Skip cows with fewer eligible videos than --videos-per-cow.",
    )
    p.add_argument(
        "--yolo-batch-size",
        type=int,
        default=32,
        help="Frames per YOLO predict() call (240 frames => 8 batches at 32).",
    )
    p.add_argument(
        "--device",
        default="0",
        help="Ultralytics device string, e.g. 0 or cpu.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def classify_session(relative_path: str, dataset_root: str) -> str:
    path_lower = relative_path.lower()
    if dataset_root == "Cow 349 - Unhealthy (sudden fall)":
        return "sudden_fall"
    if "during exercise" in path_lower:
        return "during_exercise"
    if "after exercise" in path_lower:
        return "after_exercise"
    if "before going out" in path_lower or "before exercise" in path_lower:
        return "before_exercise"
    return "other"


def _bool_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _int_csv(value: str) -> int:
    return int(float(value)) if value else 0


def _float_csv(value: str) -> float:
    return float(value) if value else 0.0


def slugify(text: str, fallback: str = "item", max_len: int = 70) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (slug or fallback)[:max_len]


def stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_video_path(row: dict[str, str], inventory: Path, dataset_root: Path | None) -> Path:
    absolute = Path(row.get("absolute_path", ""))
    if absolute.exists():
        return absolute
    base = dataset_root if dataset_root is not None else inventory.resolve().parent
    relative_parts = [part for part in row["relative_path"].split("/") if part]
    return base / row["dataset_root"] / Path(*relative_parts)


def resolve_model_path(explicit: Path | None, inventory: Path) -> Path | None:
    if explicit is not None:
        return explicit
    root = inventory.resolve().parent
    candidates = [
        root / "yolo_cow_face" / "yolo26s.pt",
        root / "yolo_cow_face" / "yolo11s.pt",
        root / "yolo_cow_face" / "yolov8n.pt",
        root / "yolo_cow_face" / "best.pt",
        root / "Transferlearning" / "Dann transfer" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def cow_level_health(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cow_id = row.get("cow_id", "").strip()
        if cow_id:
            grouped[cow_id].add(row.get("health_status", "").strip())
    out: dict[str, str] = {}
    for cow_id, labels in grouped.items():
        if "Unhealthy" in labels:
            out[cow_id] = "Unhealthy"
        elif "Healthy" in labels:
            out[cow_id] = "Healthy"
        else:
            out[cow_id] = "Unknown"
    return out


def load_inventory(inventory: Path, dataset_root: Path | None, ignore_cow_ids: set[str]) -> list[VideoRecord]:
    if not inventory.is_file():
        raise FileNotFoundError(f"Inventory CSV not found: {inventory}")
    with inventory.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    cow_health = cow_level_health(raw_rows)
    records: list[VideoRecord] = []
    for idx, row in enumerate(raw_rows, start=1):
        cow_id = row.get("cow_id", "").strip()
        health_status = row.get("health_status", "").strip()
        if not _bool_csv(row.get("readable", "")):
            continue
        if not cow_id or cow_id in ignore_cow_ids:
            continue
        if health_status not in VIDEO_HEALTH_LABELS:
            continue
        path = resolve_video_path(row, inventory, dataset_root)
        dataset_root_name = row["dataset_root"]
        relative_path = row["relative_path"]
        records.append(
            VideoRecord(
                row_id=idx,
                dataset_root=dataset_root_name,
                relative_path=relative_path,
                video_path=path,
                filename=row.get("filename", Path(relative_path).name),
                cow_id=cow_id,
                video_health_status=health_status,
                cow_health_status=cow_health.get(cow_id, health_status),
                health_condition=row.get("health_condition", "").strip(),
                top_level_folder=row.get("top_level_folder", "").strip(),
                second_level_folder=row.get("second_level_folder", "").strip(),
                session_category=classify_session(relative_path, dataset_root_name),
                frame_count=_int_csv(row.get("frame_count", "")),
                fps=_float_csv(row.get("fps", "")),
                duration_sec=_float_csv(row.get("duration_sec", "")),
            )
        )
    return records


def eligible_for_cow_pool(record: VideoRecord) -> bool:
    if record.duration_sec < 10.0 or record.frame_count <= 0 or record.fps <= 0:
        return False
    if not record.video_path.exists():
        return False
    if record.cow_health_status == "Unhealthy":
        return record.session_category in UNHEALTHY_ALLOWED_SESSIONS
    return True


def group_eligible_by_cow(records: list[VideoRecord]) -> dict[str, list[VideoRecord]]:
    grouped: dict[str, list[VideoRecord]] = defaultdict(list)
    for record in records:
        if eligible_for_cow_pool(record):
            grouped[record.cow_id].append(record)
    for cow_id in grouped:
        grouped[cow_id].sort(
            key=lambda r: (
                r.session_category,
                r.dataset_root,
                r.relative_path,
            )
        )
    return grouped


def _record_sort_key(record: VideoRecord) -> tuple[Any, ...]:
    cow_num = int(record.cow_id) if record.cow_id.isdigit() else record.cow_id
    return (record.cow_health_status, cow_num, record.relative_path)


def select_videos_for_cow(
    cow_id: str,
    pool: list[VideoRecord],
    target_n: int,
    rng: random.Random,
) -> tuple[list[VideoRecord], dict[str, Any]]:
    if not pool:
        return [], {
            "cow_id": cow_id,
            "target_videos": target_n,
            "selected_videos": 0,
            "status": "no_eligible_videos",
            "selected_sessions": "",
        }

    if len(pool) <= target_n:
        selected = list(pool)
    elif pool[0].cow_health_status == "Unhealthy":
        selected = _select_unhealthy_videos(pool, target_n, rng)
    else:
        selected = _select_diverse_videos(pool, target_n, rng)

    sessions = Counter(r.session_category for r in selected)
    return selected, {
        "cow_id": cow_id,
        "cow_health_status": pool[0].cow_health_status,
        "target_videos": target_n,
        "selected_videos": len(selected),
        "eligible_videos": len(pool),
        "status": "partial" if len(selected) < target_n else "ok",
        "selected_sessions": ";".join(f"{k}:{v}" for k, v in sorted(sessions.items())),
        "datasets": ";".join(sorted({r.dataset_root for r in selected})),
    }


def _select_diverse_videos(pool: list[VideoRecord], target_n: int, rng: random.Random) -> list[VideoRecord]:
    by_session: dict[str, list[VideoRecord]] = defaultdict(list)
    for record in pool:
        by_session[record.session_category].append(record)
    for session_records in by_session.values():
        rng.shuffle(session_records)

    session_order = sorted(by_session)
    rng.shuffle(session_order)
    selected: list[VideoRecord] = []
    offsets = {session: 0 for session in session_order}
    while len(selected) < target_n:
        progressed = False
        for session in session_order:
            offset = offsets[session]
            if offset < len(by_session[session]):
                selected.append(by_session[session][offset])
                offsets[session] += 1
                progressed = True
                if len(selected) >= target_n:
                    break
        if not progressed:
            break
    return selected[:target_n]


def _select_unhealthy_videos(pool: list[VideoRecord], target_n: int, rng: random.Random) -> list[VideoRecord]:
    during = [r for r in pool if r.session_category == "during_exercise"]
    after = [r for r in pool if r.session_category == "after_exercise"]
    sudden = [r for r in pool if r.session_category == "sudden_fall"]
    rng.shuffle(during)
    rng.shuffle(after)
    rng.shuffle(sudden)

    selected: list[VideoRecord] = []
    if sudden and target_n >= 1:
        selected.append(sudden[0])

    if target_n >= 2:
        if during:
            selected.append(during[0])
        elif after:
            selected.append(after[0])

    if target_n >= 3:
        if after:
            selected.append(after[0])
        elif during and len(during) > 1:
            selected.append(during[1])

    remaining = [r for r in pool if r not in selected]
    rng.shuffle(remaining)
    for record in remaining:
        if len(selected) >= target_n:
            break
        selected.append(record)
    return selected[:target_n]


def build_selected_video_manifest(
    by_cow: dict[str, list[VideoRecord]],
    target_n: int,
    seed: int,
    strict: bool,
) -> tuple[list[VideoRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    selected_all: list[VideoRecord] = []
    cow_summary: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for cow_id in sorted(by_cow, key=lambda value: int(value) if value.isdigit() else value):
        pool = by_cow[cow_id]
        if strict and len(pool) < target_n:
            cow_summary.append(
                {
                    "cow_id": cow_id,
                    "cow_health_status": pool[0].cow_health_status if pool else "",
                    "target_videos": target_n,
                    "selected_videos": 0,
                    "eligible_videos": len(pool),
                    "status": "skipped_strict",
                    "selected_sessions": "",
                    "datasets": "",
                }
            )
            continue

        selected, summary = select_videos_for_cow(cow_id, pool, target_n, rng)
        cow_summary.append(summary)
        for rank, record in enumerate(selected, start=1):
            selected_all.append(record)
            selected_rows.append(
                {
                    "selection_rank": len(selected_rows) + 1,
                    "cow_id": record.cow_id,
                    "cow_health_status": record.cow_health_status,
                    "video_health_status": record.video_health_status,
                    "health_condition": record.health_condition,
                    "session_category": record.session_category,
                    "dataset_root": record.dataset_root,
                    "relative_path": record.relative_path,
                    "video_path": str(record.video_path),
                    "duration_sec": f"{record.duration_sec:.6f}",
                    "fps": f"{record.fps:.6f}",
                    "frame_count": record.frame_count,
                    "target_videos_per_cow": target_n,
                    "selection_status": summary["status"],
                }
            )

    selected_all.sort(key=_record_sort_key)
    return selected_all, selected_rows, cow_summary


def make_windows(records: list[VideoRecord], args: argparse.Namespace) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    idx = 0
    for record in sorted(records, key=_record_sort_key):
        latest = max(0.0, record.duration_sec - float(args.sequence_seconds))
        start = 0.0
        while start <= latest + 1e-6:
            idx += 1
            windows.append(
                WindowRecord(
                    idx,
                    record,
                    round(start, 6),
                    round(start + float(args.sequence_seconds), 6),
                )
            )
            start += float(args.stride_seconds)
    return windows


def candidate_row(window: WindowRecord) -> dict[str, Any]:
    record = window.record
    return {
        "candidate_index": window.candidate_index,
        "cow_id": record.cow_id,
        "cow_health_status": record.cow_health_status,
        "video_health_status": record.video_health_status,
        "health_condition": record.health_condition,
        "session_category": record.session_category,
        "dataset_root": record.dataset_root,
        "relative_path": record.relative_path,
        "video_path": str(record.video_path),
        "window_start_sec": f"{window.start_second:.6f}",
        "window_end_sec": f"{window.end_second:.6f}",
        "duration_sec": f"{record.duration_sec:.6f}",
        "fps": f"{record.fps:.6f}",
        "frame_count": record.frame_count,
    }


def make_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite or choose a new output path.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "sequences").mkdir(exist_ok=True)


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int, pad: float) -> tuple[int, int, int, int] | None:
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 1 or box_h <= 1:
        return None
    left = max(0, int(round(x1 - box_w * pad)))
    top = max(0, int(round(y1 - box_h * pad)))
    right = min(width, int(round(x2 + box_w * pad)))
    bottom = min(height, int(round(y2 + box_h * pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def largest_detection(result: Any, frame_width: int, frame_height: int, crop_pad: float) -> tuple[dict[str, Any] | None, int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, 0
    best: dict[str, Any] | None = None
    valid_count = 0
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
        clamped = clamp_box(x1, y1, x2, y2, frame_width, frame_height, crop_pad)
        if clamped is None:
            continue
        left, top, right, bottom = clamped
        confidence = float(box.conf[0].detach().cpu())
        area = int((right - left) * (bottom - top))
        valid_count += 1
        candidate = {"bbox_xyxy": [left, top, right, bottom], "confidence": confidence, "area": area}
        if best is None or area > best["area"] or (area == best["area"] and confidence > best["confidence"]):
            best = candidate
    return best, valid_count


def sequence_dir_for(output: Path, window: WindowRecord, sequence_index: int) -> Path:
    record = window.record
    path_hash = stable_hash(f"{record.dataset_root}/{record.relative_path}", length=8)
    start_tag = f"t{int(round(window.start_second)):03d}s"
    name = (
        f"sequence_{slugify(record.dataset_root, max_len=18)}_{path_hash}_{start_tag}_seq{sequence_index:05d}"
    )
    cow_folder = f"cow_{record.cow_id}_{record.cow_health_status.lower()}"
    return output / "sequences" / record.cow_health_status.lower() / cow_folder / name


def _qa_pass(metadata: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(metadata["detection_rate"]) < float(args.min_detection_rate):
        reasons.append("low_detection_rate")
    if float(metadata["filled_rate"]) > float(args.max_filled_rate):
        reasons.append("high_filled_rate")
    mean_conf = metadata.get("mean_detection_confidence")
    min_conf = metadata.get("min_detection_confidence")
    if mean_conf is None or float(mean_conf) < float(args.min_mean_confidence):
        reasons.append("low_mean_confidence")
    if min_conf is None or float(min_conf) < float(args.min_min_confidence):
        reasons.append("low_min_confidence")
    return len(reasons) == 0, reasons


def _apply_detection_to_frame(
    cv2: Any,
    frame: Any,
    detection: dict[str, Any] | None,
    n_faces: int,
    output_idx: int,
    source_frame: int,
    source_second: float,
    args: argparse.Namespace,
) -> tuple[Any | None, dict[str, Any], float | None, int | None, float | None]:
    if detection is None:
        return None, {
            "output_frame": output_idx,
            "source_frame": source_frame,
            "source_second": f"{source_second:.6f}",
            "status": "filled_no_detection",
            "confidence": "",
            "bbox_xyxy": "",
            "n_faces": n_faces,
            "crop_area": "",
            "blur_laplacian_var": "",
        }, None, None, None

    left, top, right, bottom = detection["bbox_xyxy"]
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None, {
            "output_frame": output_idx,
            "source_frame": source_frame,
            "source_second": f"{source_second:.6f}",
            "status": "filled_empty_crop",
            "confidence": "",
            "bbox_xyxy": "",
            "n_faces": n_faces,
            "crop_area": "",
            "blur_laplacian_var": "",
        }, None, None, None

    resized = cv2.resize(crop, (int(args.crop_size), int(args.crop_size)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return resized, {
        "output_frame": output_idx,
        "source_frame": source_frame,
        "source_second": f"{source_second:.6f}",
        "status": "detected",
        "confidence": f"{float(detection['confidence']):.6f}",
        "bbox_xyxy": " ".join(str(v) for v in detection["bbox_xyxy"]),
        "n_faces": n_faces,
        "crop_area": int(detection["area"]),
        "blur_laplacian_var": f"{blur_score:.6f}",
    }, float(detection["confidence"]), int(detection["area"]), blur_score


def process_window(
    cv2: Any,
    model: Any,
    window: WindowRecord,
    sequence_dir: Path,
    sequence_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    record = window.record
    frames_per_sequence = int(round(float(args.sequence_seconds) * float(args.target_fps)))
    batch_size = max(1, int(args.yolo_batch_size))
    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        return None, {**candidate_row(window), "reject_reason": "video_open_failed"}

    crops: list[Any | None] = [None] * frames_per_sequence
    frame_rows: list[dict[str, Any] | None] = [None] * frames_per_sequence
    confidences: list[float] = []
    crop_areas: list[int] = []
    blur_scores: list[float] = []
    multi_face_frames = 0
    readable: list[dict[str, Any]] = []
    try:
        for output_idx in range(frames_per_sequence):
            source_second = window.start_second + (output_idx / float(args.target_fps))
            source_frame = min(record.frame_count - 1, max(0, int(round(source_second * record.fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame_rows[output_idx] = {
                    "output_frame": output_idx,
                    "source_frame": source_frame,
                    "source_second": f"{source_second:.6f}",
                    "status": "read_failed",
                    "confidence": "",
                    "bbox_xyxy": "",
                    "n_faces": 0,
                    "crop_area": "",
                    "blur_laplacian_var": "",
                }
                continue
            readable.append(
                {
                    "output_idx": output_idx,
                    "source_frame": source_frame,
                    "source_second": source_second,
                    "frame": frame,
                }
            )

        for batch_start in range(0, len(readable), batch_size):
            batch = readable[batch_start : batch_start + batch_size]
            batch_frames = [item["frame"] for item in batch]
            results = model.predict(
                batch_frames,
                conf=float(args.conf),
                verbose=False,
                device=str(args.device),
            )
            for item, result in zip(batch, results, strict=True):
                frame = item["frame"]
                height, width = frame.shape[:2]
                detection, n_faces = largest_detection(result, width, height, float(args.crop_pad))
                if n_faces > 1:
                    multi_face_frames += 1
                output_idx = int(item["output_idx"])
                crop, row, conf, area, blur = _apply_detection_to_frame(
                    cv2,
                    frame,
                    detection,
                    n_faces,
                    output_idx,
                    int(item["source_frame"]),
                    float(item["source_second"]),
                    args,
                )
                crops[output_idx] = crop
                frame_rows[output_idx] = row
                if conf is not None:
                    confidences.append(conf)
                if area is not None:
                    crop_areas.append(area)
                if blur is not None:
                    blur_scores.append(blur)
    finally:
        cap.release()

    frame_rows_out: list[dict[str, Any]] = []
    for idx in range(frames_per_sequence):
        if frame_rows[idx] is None:
            frame_rows_out.append(
                {
                    "output_frame": idx,
                    "source_frame": "",
                    "source_second": "",
                    "status": "filled_no_detection",
                    "confidence": "",
                    "bbox_xyxy": "",
                    "n_faces": 0,
                    "crop_area": "",
                    "blur_laplacian_var": "",
                }
            )
        else:
            frame_rows_out.append(frame_rows[idx])
    frame_rows = frame_rows_out

    valid_indices = [idx for idx, crop in enumerate(crops) if crop is not None]
    if not valid_indices:
        return None, {**candidate_row(window), "reject_reason": "no_detected_frames"}

    first_valid = valid_indices[0]
    last_crop = crops[first_valid]
    for idx, crop in enumerate(crops):
        if crop is None:
            crops[idx] = last_crop
            if str(frame_rows[idx]["status"]).startswith("filled"):
                frame_rows[idx]["status"] = "filled_from_nearest_previous_detection"
        else:
            last_crop = crop
    for idx in range(first_valid):
        crops[idx] = crops[first_valid]
        frame_rows[idx]["status"] = "filled_from_first_detection"

    detected_frames = sum(1 for row in frame_rows if row["status"] == "detected")
    filled_frames = frames_per_sequence - detected_frames
    overlap_seconds = float(args.sequence_seconds) - float(args.stride_seconds)
    metadata = {
        "sequence_index": sequence_index,
        "dataset_version": DATASET_VERSION,
        "candidate_index": window.candidate_index,
        "cow_id": record.cow_id,
        "cow_health_status": record.cow_health_status,
        "video_health_status": record.video_health_status,
        "health_condition": record.health_condition,
        "session_category": record.session_category,
        "dataset_root": record.dataset_root,
        "relative_path": record.relative_path,
        "video_path": str(record.video_path),
        "sequence_seconds": float(args.sequence_seconds),
        "target_fps": float(args.target_fps),
        "frames_per_sequence": frames_per_sequence,
        "crop_size": int(args.crop_size),
        "start_second": float(window.start_second),
        "end_second": float(window.end_second),
        "window_stride_seconds": float(args.stride_seconds),
        "window_overlap_seconds": overlap_seconds,
        "source_fps": float(record.fps),
        "source_frame_count": int(record.frame_count),
        "source_duration_sec": float(record.duration_sec),
        "detected_frames": int(detected_frames),
        "filled_frames": int(filled_frames),
        "detection_rate": float(detected_frames / frames_per_sequence),
        "filled_rate": float(filled_frames / frames_per_sequence),
        "confidence_threshold": float(args.conf),
        "yolo_batch_size": int(batch_size),
        "yolo_device": str(args.device),
        "mean_detection_confidence": statistics.mean(confidences) if confidences else None,
        "min_detection_confidence": min(confidences) if confidences else None,
        "max_detection_confidence": max(confidences) if confidences else None,
        "multi_face_frames": int(multi_face_frames),
        "multi_face_rate": float(multi_face_frames / frames_per_sequence),
        "crop_area_mean": statistics.mean(crop_areas) if crop_areas else None,
        "crop_area_min": min(crop_areas) if crop_areas else None,
        "crop_area_max": max(crop_areas) if crop_areas else None,
        "blur_laplacian_var_mean": statistics.mean(blur_scores) if blur_scores else None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    ok, reasons = _qa_pass(metadata, args)
    if not ok:
        return None, {**metadata, "reject_reason": ";".join(reasons)}

    sequence_dir.mkdir(parents=True, exist_ok=True)
    for idx, crop in enumerate(crops):
        cv2.imwrite(str(sequence_dir / f"frame_{idx:04d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    write_csv(sequence_dir / "frames.csv", frame_rows)
    (sequence_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, {**metadata, "reject_reason": ""}


def write_readme(output: Path, stats: dict[str, Any], args: argparse.Namespace) -> None:
    readme = f"""# Thesis Cow-Face Sequence Dataset (`{DATASET_VERSION}`)

Generated by `create_thesis_stride8_sequences.py`.

## Video selection rules

- **Unhealthy cows (17):** only `During exercise`, `After exercise`, and **Cow 349 sudden-fall** videos.
- **Healthy cows (15):** any session (before / during / after exercise).
- **Target videos per cow:** `{args.videos_per_cow}` (seed `{args.seed}`).
- Cows with fewer eligible videos use all available clips and are marked `partial` in `selected_videos_by_cow.csv`.
- Cow `409` excluded (unknown health label).

## Windowing rules

- Window length: `{args.sequence_seconds:g}` s
- Stride: `{args.stride_seconds:g}` s (overlap `{args.sequence_seconds - args.stride_seconds:g}` s)
- Stored frames: `{int(round(args.sequence_seconds * args.target_fps))}` at `{args.target_fps:g}` FPS
- Crop: `{args.crop_size}` px, YOLO conf `{args.conf:g}`, pad `{args.crop_pad:g}`
- QA pass: detection rate >= `{args.min_detection_rate:g}`, filled rate <= `{args.max_filled_rate:g}`

## Summary

```json
{json.dumps(stats, indent=2)}
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ignore_cow_ids = {item.strip() for item in str(args.ignore_cow_ids).split(",") if item.strip()}
    model_path = resolve_model_path(args.model, args.inventory)
    if not args.dry_run and model_path is None:
        print("ERROR: --model is required unless --dry-run is set.", file=sys.stderr)
        return 2

    make_output_dir(args.output, bool(args.overwrite))
    records = load_inventory(args.inventory, args.dataset_root, ignore_cow_ids)
    by_cow = group_eligible_by_cow(records)
    selected_videos, selected_rows, cow_summary = build_selected_video_manifest(
        by_cow,
        int(args.videos_per_cow),
        int(args.seed),
        bool(args.strict_videos_per_cow),
    )
    windows = make_windows(selected_videos, args)

    write_csv(args.output / "selected_videos.csv", selected_rows)
    write_csv(args.output / "selected_videos_by_cow.csv", cow_summary)
    write_csv(args.output / "candidate_windows.csv", [candidate_row(w) for w in windows])

    stats: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "dry_run": bool(args.dry_run),
        "videos_per_cow_target": int(args.videos_per_cow),
        "selection_seed": int(args.seed),
        "strict_videos_per_cow": bool(args.strict_videos_per_cow),
        "sequence_seconds": float(args.sequence_seconds),
        "stride_seconds": float(args.stride_seconds),
        "window_overlap_seconds": float(args.sequence_seconds) - float(args.stride_seconds),
        "target_fps": float(args.target_fps),
        "frames_per_sequence": int(round(float(args.sequence_seconds) * float(args.target_fps))),
        "selected_source_videos": int(len(selected_videos)),
        "selected_unique_cows": len({v.cow_id for v in selected_videos}),
        "candidate_windows": int(len(windows)),
        "completed_sequences": 0,
        "rejected_windows": 0,
        "selection_summary": {
            "healthy_cows": sum(1 for row in cow_summary if row.get("cow_health_status") == "Healthy" and row.get("selected_videos", 0) > 0),
            "unhealthy_cows": sum(1 for row in cow_summary if row.get("cow_health_status") == "Unhealthy" and row.get("selected_videos", 0) > 0),
            "partial_cows": [row["cow_id"] for row in cow_summary if row.get("status") == "partial"],
            "skipped_cows": [row["cow_id"] for row in cow_summary if row.get("status") == "skipped_strict"],
        },
        "errors": [],
    }

    if args.dry_run:
        (args.output / "processing_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        write_readme(args.output, stats, args)
        print(f"Dry run complete. Selected videos: {len(selected_videos)}")
        print(f"Candidate windows: {len(windows)}")
        print(f"Partial cows: {stats['selection_summary']['partial_cows']}")
        print(f"Manifests written to: {args.output}")
        return 0

    import cv2  # type: ignore[import-not-found]
    from ultralytics import YOLO  # type: ignore[import-not-found]

    if model_path is None or not model_path.is_file():
        print(f"ERROR: model weights not found: {model_path}", file=sys.stderr)
        return 2

    model = YOLO(str(model_path))
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    start_time = time.time()

    for window in windows:
        sequence_index = len(completed) + 1
        sequence_dir = sequence_dir_for(args.output, window, sequence_index)
        print(
            f"[candidate {window.candidate_index:06d}] seq={sequence_index:05d} "
            f"cow={window.record.cow_id} start={window.start_second:.1f}s "
            f"video={window.record.relative_path}",
            flush=True,
        )
        try:
            metadata, audit = process_window(cv2, model, window, sequence_dir, sequence_index, args)
        except Exception as exc:  # noqa: BLE001
            metadata = None
            audit = {**candidate_row(window), "reject_reason": f"exception:{exc}"}
            stats["errors"].append(audit)
        if metadata is None:
            rejected.append(audit)
            continue
        completed.append(metadata)

    stats["completed_sequences"] = int(len(completed))
    stats["rejected_windows"] = int(len(rejected))
    stats["processing_time_sec"] = float(time.time() - start_time)
    stats["completed_summary"] = {
        "cow_health_counts": dict(Counter(item["cow_health_status"] for item in completed)),
        "video_health_counts": dict(Counter(item["video_health_status"] for item in completed)),
        "session_counts": dict(Counter(item["session_category"] for item in completed)),
        "unique_cows": len({item["cow_id"] for item in completed}),
        "dataset_counts": dict(Counter(item["dataset_root"] for item in completed)),
    }
    write_csv(args.output / "completed_manifest.csv", completed)
    write_csv(args.output / "rejected_windows.csv", rejected)
    (args.output / "processing_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_readme(args.output, stats, args)
    print(
        f"Completed {len(completed)} QA-passing sequences from {len(selected_videos)} source videos "
        f"in {stats['processing_time_sec'] / 60.0:.1f} minutes."
    )
    print(f"Rejected windows: {len(rejected)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
