# Rorqual V2 run snapshot — 2026-05-08

Frozen copies for **Alliance Rorqual** job completing **`holstein_task1_*_v2_run_sbatch`** (full training + GPU eval-only refresh same day).

## Contents

| Path | Description |
|------|-------------|
| **`dann_sbatch/`** | **Full DANN output tree**: 14× `fold_*` (checkpoints `best_dann.pt`, `best_dann_proxy_fallback.pt` where written, `history.csv`, `val_predictions.csv`), pooled `dann_predictions.csv`, `dann_test_predictions.csv`, cow aggregates, `dann_splits.json`, `dann_fold_summary.csv`, `dann_run.json`, `dann_report.md`, `dann_summary.json`, `dann_diagnostics.json`. |
| **`weak_gce_sbatch/`** | **Full weak-label GCE tree**: same fold layout with `best_weak_task1.pt`, pooled CSVs, `weak_label_cv_splits.json`, fold summary, reports, `weak_label_cv_summary.json`, `weak_label_cv_diagnostics.json`, run metadata. |
| **`results/`** | Convenience copies of **reports**, **ensemble summaries**, and **`*_diagnostics.json`** (same files as the roots of `dann_sbatch/` and `weak_gce_sbatch/`). |
| **`training_code/*.py`** | **Snapshots** of scripts used for this protocol; **canonical** versions remain in `Dann_transfer/` repo root — edit those, then refresh snapshots if needed. |

Total artifact count (typical): **66** files under `dann_sbatch/`, **52** under `weak_gce_sbatch/` (includes ~2.5 MB checkpoints per fold).

## Cluster paths (reference)

- Data: `/scratch/shiv136/project_data/` (`cow_face_sequences_10s_250`, `ucaps_source`, `v2.9_20260222_144752`)
- Training outputs (live): `.../runs/holstein_task1_dann_v2_run_sbatch`, `.../runs/holstein_task1_weak_gce_v2_run_sbatch`
- Pipeline: `~/Dann_transfer/V2/run_task1_vast.sh` via `sbatch_task1_rorqual.sh`

## Eval-only rerun

GPU `--eval-only` refresh timestamps embedded in reports: **20260508T154623Z** (DANN), **20260508T154642Z** (weak).

## Refreshing this folder from another machine

After `scp` or rsync from Rorqual scratch into local dirs such as `rorqual_results_dann_sbatch/`, mirror into here:

```powershell
robocopy path\to\rorqual_results_dann_sbatch "V2\rorqual_run_20260508\dann_sbatch" /E
robocopy path\to\rorqual_results_weak_sbatch "V2\rorqual_run_20260508\weak_gce_sbatch" /E
```

Then recopy the six files under `results/` from those roots if you want `results/` to stay in sync.
