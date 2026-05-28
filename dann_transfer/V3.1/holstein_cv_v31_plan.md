# Holstein CV / V3.1 experiment matrix

Use this checklist when scheduling GPU runs. **Freeze V2 numbers** before interpreting V3.1 deltas.

## Phase A — Data & QC (highest ROI)

| ID | Change | Script / knob | Primary metric |
|----|--------|---------------|----------------|
| A1 | Offline QC manifest | [`filter_manifest_qc.py`](filter_manifest_qc.py) → point `TARGET_MANIFEST` at filtered CSV | PR-AUC, Brier/ECE |
| A2 | Multi-window sequences | [`yolo_cow_face/create_10s_face_sequences_v3_1.py`](../../yolo_cow_face/create_10s_face_sequences_v3_1.py) `--multi-window-k` | Val AUC stability |
| A3 | Inline QC during train | `--manifest-min-mean-detection-confidence`, `--manifest-max-filled-frames`, `--manifest-write-qc-audit` | Rows kept vs skipped (audit CSV) |

## Phase B — Thresholding & calibration

| ID | Change | Knob | Notes |
|----|--------|------|------|
| B1 | Legacy behaviour | `--test-threshold-policy mean_fold_best_f1` | Matches V2 ensemble aggregation |
| B2 | Pooled val F1-opt | `--test-threshold-policy pooled_val_f1_opt` | Uses pooled validation predictions |
| B3 | Degenerate predictions | JSON `degenerate_sequence_predictions_*` | Flags all-positive / all-negative |

## Phase C — Domain adaptation variants

| ID | Mode | CLI | Reference |
|----|------|-----|-----------|
| C1 | Vanilla DANN | `--da-mode dann` | Ganin et al., JMLR 2016 |
| C2 | CDAN-style conditioning | `--da-mode cdan` | Long et al., NeurIPS 2018 |
| C3 | MDD-inspired aux head | `--da-mode mdd` (+ `--mdd-*`) | Saito et al., ICCV 2019 |
| C4 | Domain loss schedule | `--domain-warmup-epochs`, `--domain-ramp-epochs` | Stabilizes early training |

## Phase D — Evaluation variance

| ID | Protocol | Script |
|----|----------|--------|
| D1 | Multi-seed repeats | [`run_eval_repeat_splits.sh`](run_eval_repeat_splits.sh) |
| D2 | Repeated cow held-out sets | Extend split JSON seeds (future work; needs enough cows) | |

## Suggested first smoke

1. `bash V3.1/run_task1_v3_1.sh` with `DANN_EPOCHS=2` `WEAK_EPOCHS=2` locally or on cluster.
2. Re-run with `EXTRA_WEAK_ARGS="--test-threshold-policy pooled_val_f1_opt"` and compare degeneracy flags in `weak_label_cv_summary.json`.
