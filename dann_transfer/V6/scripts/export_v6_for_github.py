#!/usr/bin/env python3
"""Copy V6 trial outputs into results/vast_auto/ with only GitHub-safe report files."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "vast_auto"
DST = ROOT / "results" / "vast_auto"

S3_KEEP = {
    "weak_label_cv_summary.json",
    "weak_label_cv_report.md",
    "weak_label_cv_diagnostics.json",
    "weak_label_cv_fold_summary.csv",
    "weak_label_cv_splits.json",
    "weak_label_cv_run.json",
    "weak_label_cv_test_predictions.csv",
    "weak_label_cv_test_cow_aggregates.csv",
    "weak_label_cv_test_video_aggregates.csv",
    "weak_label_cv_test_calibrated_cow_aggregates.csv",
    "weak_label_cv_test_calibrated_video_aggregates.csv",
}
S4_KEEP = {
    "dann_summary.json",
    "dann_report.md",
    "dann_diagnostics.json",
    "dann_fold_summary.csv",
    "dann_splits.json",
    "dann_run.json",
    "dann_test_predictions.csv",
    "dann_test_cow_aggregates.csv",
    "dann_test_video_aggregates.csv",
    "dann_test_calibrated_cow_aggregates.csv",
    "dann_test_calibrated_video_aggregates.csv",
}

TRIALS = list(S3_KEEP)  # placeholder
STAGE_A = [
    "A_s3_focal_g1p5_cb",
    "A_s3_focal_g2p5_cb",
    "A_s3_gce_q0p6_cb",
    "A_s3_gce_q0p8_cb",
]
STAGE_B = [
    "B_s4_dann_dw0p15",
    "B_s4_dann_dw0p20",
    "B_s4_coral_w0p02",
]


def prune_trial(trial_dir: Path, keep: set[str]) -> None:
    if not trial_dir.is_dir():
        return
    for child in list(trial_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name not in keep:
            child.unlink()


def main() -> None:
    for name in STAGE_A:
        prune_trial(DST / name, S3_KEEP)
    for name in STAGE_B:
        prune_trial(DST / name, S4_KEEP)
    for log in ("stageA.log", "stageB.log", "queue.log", "queue_watcher.log"):
        p = DST / log
        if p.is_file():
            p.unlink()
    print(f"Pruned {DST} for GitHub (reports/CSVs/JSON only).")


if __name__ == "__main__":
    main()
