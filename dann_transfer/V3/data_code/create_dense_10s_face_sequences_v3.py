#!/usr/bin/env python3
"""
Create a dense QA-filtered Holstein/Jersey cow-face sequence dataset.

V3 keeps the original 250-sequence dataset fixed for comparability and writes a
separate dense dataset, normally named cow_face_sequences_10s_v3_dense. Windows
are 10 seconds long with 5-second stride; only windows passing detection/fill
and confidence QA are written to completed_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_HEALTH_LABELS = {"Healthy", "Unhealthy"}


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
    p = argparse.ArgumentParser(description="Create V3 dense 10s cow-face sequence windows with QA filtering.")
    p.add_argument("--inventory", type=Path, default=Path("cow_video_dataset_analysis.csv"))
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None, help="YOLO face detector best.pt. Required unless --dry-run.")
    p.add_argument("--output", type=Path, default=Path("cow_face_sequences_10s_v3_dense"))
    p.add_argument("--sequence-seconds", type=float, default=10.0)
    p.add_argument("--stride-seconds", type=float, default=5.0)
    p.add_argument("--target-fps", type=float, default=24.0)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--conf", type=float, default=0.60)
    p.add_argument("--crop-pad", type=float, default=0.08)
    p.add_argument("--min-detection-rate", type=float, default=0.90)
    p.add_argument("--max-filled-rate", type=float, default=0.10)
    p.add_argument("--min-mean-confidence", type=float, default=0.80)
    p.add_argument("--min-min-confidence", type=float, default=0.60)
    p.add_argument("--max-windows-per-video", type=int, default=None)
    p.add_argument("--max-windows-per-cow", type=int, default=None)
    p.add_argument("--ignore-cow-ids", default="409")
    p.add_argument("--dry-run", action="store_true", help="Write candidate/audit manifests without loading YOLO.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
        records.append(
            VideoRecord(
                row_id=idx,
                dataset_root=row["dataset_root"],
                relative_path=row["relative_path"],
                video_path=path,
                filename=row.get("filename", Path(row["relative_path"]).name),
                cow_id=cow_id,
                video_health_status=health_status,
                cow_health_status=cow_health.get(cow_id, health_status),
                health_condition=row.get("health_condition", "").strip(),
                top_level_folder=row.get("top_level_folder", "").strip(),
                second_level_folder=row.get("second_level_folder", "").strip(),
                frame_count=_int_csv(row.get("frame_count", "")),
                fps=_float_csv(row.get("fps", "")),
                duration_sec=_float_csv(row.get("duration_sec", "")),
            )
        )
    return records


def eligible_records(records: list[VideoRecord], sequence_seconds: float) -> tuple[list[VideoRecord], list[VideoRecord]]:
    eligible: list[VideoRecord] = []
    rejected: list[VideoRecord] = []
    for record in records:
        if record.duration_sec >= sequence_seconds and record.frame_count > 0 and record.fps > 0 and record.video_path.exists():
            eligible.append(record)
        else:
            rejected.append(record)
    return eligible, rejected


def make_windows(records: list[VideoRecord], args: argparse.Namespace) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    cow_counts: dict[str, int] = defaultdict(int)
    idx = 0
    for record in sorted(records, key=lambda r: (r.cow_health_status, int(r.cow_id) if r.cow_id.isdigit() else r.cow_id, r.relative_path)):
        latest = max(0.0, record.duration_sec - float(args.sequence_seconds))
        starts: list[float] = []
        s = 0.0
        while s <= latest + 1e-6:
            starts.append(round(s, 6))
            s += float(args.stride_seconds)
        if args.max_windows_per_video is not None:
            starts = starts[: int(args.max_windows_per_video)]
        for start in starts:
            if args.max_windows_per_cow is not None and cow_counts[record.cow_id] >= int(args.max_windows_per_cow):
                break
            idx += 1
            cow_counts[record.cow_id] += 1
            windows.append(WindowRecord(idx, record, start, start + float(args.sequence_seconds)))
    return windows


def candidate_row(window: WindowRecord) -> dict[str, Any]:
    r = window.record
    return {
        "candidate_index": window.candidate_index,
        "cow_id": r.cow_id,
        "cow_health_status": r.cow_health_status,
        "video_health_status": r.video_health_status,
        "health_condition": r.health_condition,
        "dataset_root": r.dataset_root,
        "relative_path": r.relative_path,
        "video_path": str(r.video_path),
        "window_start_sec": f"{window.start_second:.6f}",
        "window_end_sec": f"{window.end_second:.6f}",
        "duration_sec": f"{r.duration_sec:.6f}",
        "fps": f"{r.fps:.6f}",
        "frame_count": r.frame_count,
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


def sequence_dir_for(output: Path, sequence_index: int, window: WindowRecord) -> Path:
    record = window.record
    path_hash = stable_hash(f"{record.dataset_root}/{record.relative_path}", length=8)
    name = f"sequence_{sequence_index:04d}_{slugify(record.dataset_root, max_len=22)}_{path_hash}"
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


def process_window(cv2: Any, model: Any, window: WindowRecord, sequence_dir: Path, sequence_index: int, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    record = window.record
    frames_per_sequence = int(round(float(args.sequence_seconds) * float(args.target_fps)))
    cap = cv2.VideoCapture(str(record.video_path))
    if not cap.isOpened():
        return None, {**candidate_row(window), "reject_reason": "video_open_failed"}

    crops: list[Any | None] = []
    frame_rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    crop_areas: list[int] = []
    blur_scores: list[float] = []
    multi_face_frames = 0
    try:
        for output_idx in range(frames_per_sequence):
            source_second = window.start_second + (output_idx / float(args.target_fps))
            source_frame = min(record.frame_count - 1, max(0, int(round(source_second * record.fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                crops.append(None)
                frame_rows.append(
                    {
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
                )
                continue
            height, width = frame.shape[:2]
            detection = None
            n_faces = 0
            for result in model(frame, conf=float(args.conf), verbose=False):
                result_detection, result_faces = largest_detection(result, width, height, float(args.crop_pad))
                n_faces += result_faces
                if result_detection is None:
                    continue
                if detection is None or result_detection["area"] > detection["area"]:
                    detection = result_detection
            if n_faces > 1:
                multi_face_frames += 1
            if detection is None:
                crops.append(None)
                frame_rows.append(
                    {
                        "output_frame": output_idx,
                        "source_frame": source_frame,
                        "source_second": f"{source_second:.6f}",
                        "status": "filled_no_detection",
                        "confidence": "",
                        "bbox_xyxy": "",
                        "n_faces": n_faces,
                        "crop_area": "",
                        "blur_laplacian_var": "",
                    }
                )
                continue
            left, top, right, bottom = detection["bbox_xyxy"]
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                crops.append(None)
                frame_rows.append(
                    {
                        "output_frame": output_idx,
                        "source_frame": source_frame,
                        "source_second": f"{source_second:.6f}",
                        "status": "filled_empty_crop",
                        "confidence": "",
                        "bbox_xyxy": "",
                        "n_faces": n_faces,
                        "crop_area": "",
                        "blur_laplacian_var": "",
                    }
                )
                continue
            resized = cv2.resize(crop, (int(args.crop_size), int(args.crop_size)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            crops.append(resized)
            confidences.append(float(detection["confidence"]))
            crop_areas.append(int(detection["area"]))
            blur_scores.append(blur_score)
            frame_rows.append(
                {
                    "output_frame": output_idx,
                    "source_frame": source_frame,
                    "source_second": f"{source_second:.6f}",
                    "status": "detected",
                    "confidence": f"{float(detection['confidence']):.6f}",
                    "bbox_xyxy": " ".join(str(v) for v in detection["bbox_xyxy"]),
                    "n_faces": n_faces,
                    "crop_area": int(detection["area"]),
                    "blur_laplacian_var": f"{blur_score:.6f}",
                }
            )
    finally:
        cap.release()

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
    metadata = {
        "sequence_index": sequence_index,
        "candidate_index": window.candidate_index,
        "cow_id": record.cow_id,
        "cow_health_status": record.cow_health_status,
        "video_health_status": record.video_health_status,
        "health_condition": record.health_condition,
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
        "source_fps": float(record.fps),
        "source_frame_count": int(record.frame_count),
        "source_duration_sec": float(record.duration_sec),
        "detected_frames": int(detected_frames),
        "filled_frames": int(filled_frames),
        "detection_rate": float(detected_frames / frames_per_sequence),
        "filled_rate": float(filled_frames / frames_per_sequence),
        "confidence_threshold": float(args.conf),
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
    readme = f"""# V3 Dense Cow-Face 10-Second Sequence Dataset

Generated by `V3/data_code/create_dense_10s_face_sequences_v3.py`.

Rules:

- 10-second windows, 5-second stride by default.
- Target frame rate `{args.target_fps:g}` FPS, crop size `{args.crop_size}`.
- YOLO confidence threshold `{args.conf:g}`, crop padding `{args.crop_pad:g}`.
- QA pass requires detection rate >= `{args.min_detection_rate:g}`, filled rate <= `{args.max_filled_rate:g}`, mean confidence >= `{args.min_mean_confidence:g}`, and min confidence >= `{args.min_min_confidence:g}`.
- Dataset label remains weak `video_health_status`; it is not veterinary pain ground truth.
- This dense dataset must be reported separately from baseline_10s_250.

Summary:

```json
{json.dumps(stats, indent=2)}
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ignore_cow_ids = {item.strip() for item in str(args.ignore_cow_ids).split(",") if item.strip()}
    if not args.dry_run and args.model is None:
        print("ERROR: --model is required unless --dry-run is set.", file=sys.stderr)
        return 2

    make_output_dir(args.output, bool(args.overwrite))
    records = load_inventory(args.inventory, args.dataset_root, ignore_cow_ids)
    eligible, rejected_videos = eligible_records(records, float(args.sequence_seconds))
    windows = make_windows(eligible, args)
    write_csv(args.output / "candidate_windows.csv", [candidate_row(w) for w in windows])
    write_csv(args.output / "rejected_inventory_rows.csv", [candidate_row(WindowRecord(i + 1, r, 0.0, 0.0)) for i, r in enumerate(rejected_videos)])

    stats: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "sequence_seconds": float(args.sequence_seconds),
        "stride_seconds": float(args.stride_seconds),
        "target_fps": float(args.target_fps),
        "frames_per_sequence": int(round(float(args.sequence_seconds) * float(args.target_fps))),
        "qa": {
            "min_detection_rate": float(args.min_detection_rate),
            "max_filled_rate": float(args.max_filled_rate),
            "min_mean_confidence": float(args.min_mean_confidence),
            "min_min_confidence": float(args.min_min_confidence),
        },
        "eligible_videos": int(len(eligible)),
        "candidate_windows": int(len(windows)),
        "completed_sequences": 0,
        "rejected_windows": 0,
        "errors": [],
    }
    if args.dry_run:
        (args.output / "processing_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        write_readme(args.output, stats, args)
        print(f"Dry run complete. Candidate windows: {len(windows)}")
        print(f"Candidate manifest: {args.output / 'candidate_windows.csv'}")
        return 0

    import cv2  # type: ignore[import-not-found]
    from ultralytics import YOLO  # type: ignore[import-not-found]

    if not args.model.is_file():
        print(f"ERROR: model weights not found: {args.model}", file=sys.stderr)
        return 2
    model = YOLO(str(args.model))
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    start_time = time.time()

    for window in windows:
        sequence_index = len(completed) + 1
        sequence_dir = sequence_dir_for(args.output, sequence_index, window)
        print(
            f"[candidate {window.candidate_index:06d}] seq={sequence_index:05d} cow={window.record.cow_id} "
            f"start={window.start_second:.1f}s video={window.record.relative_path}",
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
        "unique_cows": len({item["cow_id"] for item in completed}),
        "dataset_counts": dict(Counter(item["dataset_root"] for item in completed)),
    }
    write_csv(args.output / "completed_manifest.csv", completed)
    write_csv(args.output / "rejected_windows.csv", rejected)
    (args.output / "processing_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_readme(args.output, stats, args)
    print(f"Completed {len(completed)} QA-passing dense sequences in {stats['processing_time_sec'] / 60.0:.1f} minutes.")
    print(f"Rejected windows: {len(rejected)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
