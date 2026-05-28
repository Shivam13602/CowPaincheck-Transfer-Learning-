"""Headline comparison: thesis_stride8_qa (V4) vs baseline_10s_250 (V3 matrix).

Reads summary.json from V4 results and V3 baseline run.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4_ROOT = HERE.parent
THESIS_ROOT = V4_ROOT / "results"
BASELINE_ROOT = V4_ROOT.parent / "V3" / "rorqual_run_20260515_12326664" / "v3_baseline_10s_250"


def load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def metrics_row(label: str, summary: dict | None) -> dict:
    if summary is None:
        return {"run": label}
    ft = summary.get("final_test", {}) or {}
    seq = ft.get("sequence_metrics", {}) or {}
    vid = ft.get("video_metrics", {}) or {}
    cow = ft.get("cow_metrics", {}) or {}
    seq_cal = ft.get("calibrated_sequence_metrics", {}) or {}
    vid_cal = ft.get("calibrated_video_metrics", {}) or {}
    cow_cal = ft.get("calibrated_cow_metrics", {}) or {}
    return {
        "run": label,
        "seq_auc": seq.get("auc"),
        "seq_bacc": seq.get("balanced_accuracy"),
        "vid_auc": vid.get("auc"),
        "vid_bacc": vid.get("balanced_accuracy"),
        "cow_auc": cow.get("auc"),
        "cow_bacc": cow.get("balanced_accuracy"),
        "seq_auc_cal": seq_cal.get("auc"),
        "vid_auc_cal": vid_cal.get("auc"),
        "cow_auc_cal": cow_cal.get("auc"),
        "threshold": ft.get("threshold_from_pooled_validation_specificity_constrained"),
        "n_seq": seq.get("n"),
        "n_vid": vid.get("n"),
        "n_cow": cow.get("n"),
    }


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    candidates = [
        ("thesis_stride8_qa · weak_focal", THESIS_ROOT / "weak_focal" / "weak_label_cv_summary.json"),
        ("thesis_stride8_qa · coral_w_0.10", THESIS_ROOT / "coral_w_0.10" / "dann_summary.json"),
        ("baseline_10s_250 · weak_focal", BASELINE_ROOT / "weak_focal" / "weak_label_cv_summary.json"),
        ("baseline_10s_250 · coral_w_0.10", BASELINE_ROOT / "coral_w_0.10" / "dann_summary.json"),
    ]
    rows = [metrics_row(label, load_summary(path)) for label, path in candidates]

    columns = [
        ("run", 36),
        ("n_seq", 6), ("seq_auc", 8), ("seq_bacc", 9),
        ("n_vid", 6), ("vid_auc", 8), ("vid_bacc", 9),
        ("n_cow", 6), ("cow_auc", 8), ("cow_bacc", 9),
        ("threshold", 10),
    ]
    header = " | ".join(name.rjust(w) if i else name.ljust(w) for i, (name, w) in enumerate(columns))
    sep = "-+-".join("-" * w for _, w in columns)
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for i, (name, w) in enumerate(columns):
            val = fmt(row.get(name))
            cells.append(val.rjust(w) if i else val.ljust(w))
        print(" | ".join(cells))

    print()
    print("Calibrated AUC (temperature-scaled):")
    for row in rows:
        if all(row.get(k) is None for k in ("seq_auc_cal", "vid_auc_cal", "cow_auc_cal")):
            continue
        print(f"  {row['run']:<36}  seq={fmt(row.get('seq_auc_cal'))}  vid={fmt(row.get('vid_auc_cal'))}  cow={fmt(row.get('cow_auc_cal'))}")

    print()
    print("Reports:")
    for label, path in candidates:
        report = path.with_name(path.stem.replace("_summary", "_report") + ".md")
        if report.is_file():
            print(f"  {label}: {report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
