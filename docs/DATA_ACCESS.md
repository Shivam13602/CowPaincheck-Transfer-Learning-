# Data and Checkpoint Access

This repository contains **code, manifests, metrics, and reports only**. Image sequences, UCAPS source bundles, and model checkpoints are excluded via `.gitignore` because of size (~7+ GB locally).

## Target-domain Holstein/Jersey sequences

| Dataset | Sequences | Local path (author workspace) | Rorqual scratch |
|---------|----------:|-------------------------------|-------------------|
| `baseline_10s_250` | 250 | `cow_face_sequences_10s_250/sequences/` | `/scratch/shiv136/project_data/cow_face_sequences_10s_250/output/` |
| `thesis_stride8_qa` | 549 | `cow_face_sequences_thesis_stride8/output/sequences/` | `/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output/` |

**Manifests included in repo:** `datasets/baseline_10s_250/completed_manifest.csv`, `datasets/thesis_stride8_qa/output/completed_manifest.csv`, and related inventory CSVs.

## Source-domain UCAPS data

| Asset | Approx. size | Location |
|-------|-------------:|----------|
| UCAPS source project + sequences | ~1.9 GB (tar) | Local: `ucaps_source_for_dann.tar` (extract to project root) |
| Rorqual scratch | — | `/scratch/shiv136/project_data/ucaps_source/` |

## Pretrained checkpoints

| Checkpoint set | Purpose | Location |
|----------------|---------|----------|
| UCAPS v2.9 Task1/Task2 folds | Initialization for weak-label and DANN | `/scratch/shiv136/project_data/v2.9_20260222_144752/` |
| Per-fold best models | Reproduce ensemble test predictions | Under each run output, e.g. `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/<condition>/fold_*/best_*.pt` |

See also [`dann_transfer/CHECKPOINTS_README.md`](../dann_transfer/CHECKPOINTS_README.md).

## YOLO face detector

Face cropping for sequence extraction uses a trained YOLO `.pt` file (`best.pt`, ~18 MB). Place locally at repo-adjacent workspace or set path in extraction scripts. Not committed to GitHub.

## Re-downloading results from Rorqual

Training outputs on Alliance Rorqual (Compute Canada):

| Experiment | Scratch output path |
|------------|---------------------|
| V3 baseline matrix (job 12326664) | `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/` |
| V4 thesis_stride8_qa (job 13253646) | `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/` |

Metrics and reports for V3/V4 are mirrored in `dann_transfer/V3/rorqual_run_*` and `dann_transfer/V4/results/`.

## Contact

For access to raw video or veterinary-scored subsets, contact the repository author(s). This public repo documents methods and reported metrics only.
