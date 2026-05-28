# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260508T154642Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `14` folds x `2` validation cows
- Initialization: `/scratch/shiv136/project_data/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `gce` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.0 | 374,402 | 21 | nan | 0.0 | 0.0 | 0.0 | nan | 0.0 | 0.0 | 0 | 21 | 0 | 0 | nan | 0.0 | 0.0 | 1.0 | nan | 0.0 | nan |
| 1 | 0 | 0.8462 | 310,417 | 21 | 0.8462 | 0.7647 | 0.8462 | 0.619 | 0.5 | 0.619 | 1.0 | 0 | 8 | 0 | 13 | 1.0 | 0.6667 | 1.0 | 0.3322 | 0.8462 | 0.8462 | 1.0 |
| 2 | 2 | 0.9 | 415,436 | 16 | 0.9 | 0.6667 | 0.8333 | 0.6875 | 0.7167 | 0.5556 | 0.8333 | 6 | 4 | 1 | 5 | 1.0 | 1.0 | 1.0 | 0.3829 | 0.9 | 0.9091 | 1.0 |
| 3 | 0 | 1.0 | 349,439 | 10 | 1.0 | 0.6667 | 1.0 | 0.6 | 0.6667 | 0.5 | 1.0 | 2 | 4 | 0 | 4 | 1.0 | 0.6667 | 1.0 | 1.1131 | 1.0 | 0.8889 | 1.0 |
| 4 | 5 | 1.0 | 421,428 | 13 | 1.0 | 0.9167 | 1.0 | 0.8462 | 0.5 | 0.8462 | 1.0 | 0 | 2 | 0 | 11 | 1.0 | 0.6667 | 1.0 | 0.3534 | 1.0 | 1.0 | 1.0 |
| 5 | 32 | 0.95 | 406,438 | 18 | 0.95 | 0.0 | 0.7143 | 0.4444 | 0.5 | 0.0 | 0.0 | 8 | 0 | 10 | 0 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.95 | 0.7143 | 1.0 |
| 6 | 2 | 0.9333 | 255,387 | 19 | 0.9333 | 0.6897 | 0.8421 | 0.5263 | 0.5 | 0.5263 | 1.0 | 0 | 9 | 0 | 10 | 1.0 | 0.6667 | 1.0 | 1.2562 | 0.9333 | 0.9 | 1.0 |
| 7 | 2 | 0.7667 | 432,433 | 16 | 0.7667 | 0.6667 | 0.7692 | 0.6875 | 0.75 | 1.0 | 0.5 | 6 | 0 | 5 | 5 | 1.0 | 1.0 | 1.0 | 0.363 | 0.7667 | 0.8333 | 1.0 |
| 8 | 21 | 0.4333 | 378,427 | 16 | 0.4333 | 0.4 | 0.5455 | 0.625 | 0.5667 | 0.5 | 0.3333 | 8 | 2 | 4 | 2 | 0.0 | 0.0 | 0.6667 | 10.0 | 0.4333 | 0.5455 | 0.0 |
| 9 | 17 | 0.1727 | 370,394 | 21 | 0.1727 | 0.0 | 0.6875 | 0.2857 | 0.3 | 0.0 | 0.0 | 6 | 4 | 11 | 0 | 0.0 | 0.0 | 0.6667 | 10.0 | 0.1727 | 0.6875 | 0.0 |
| 10 | 10 | 0.5 | 323,352 | 10 | 0.5 | 0.2 | 0.8889 | 0.2 | 0.3125 | 0.5 | 0.125 | 1 | 1 | 7 | 1 | 0.0 | 0.0 | 0.6667 | 10.0 | 0.5 | 0.8889 | 0.0 |
| 11 | 8 | 0.96 | 354,405 | 15 | 0.96 | 0.8696 | 0.9091 | 0.8 | 0.7 | 0.7692 | 1.0 | 2 | 3 | 0 | 10 | 1.0 | 0.6667 | 1.0 | 0.4247 | 0.96 | 0.9 | 1.0 |
| 12 | 7 | 0.575 | 425,426 | 14 | 0.575 | 0.7059 | 0.8333 | 0.6429 | 0.675 | 0.8571 | 0.6 | 3 | 1 | 4 | 6 | 1.0 | 1.0 | 1.0 | 1.2218 | 0.575 | 0.8333 | 1.0 |
| 13 | 0 | 1.0 | 355,446 | 11 | 1.0 | 0.625 | 1.0 | 0.4545 | 0.5 | 0.4545 | 1.0 | 0 | 6 | 0 | 5 | 1.0 | 0.6667 | 1.0 | 0.7454 | 1.0 | 1.0 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.3393 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.4 | 0.6341 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.3393 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.3964 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.3964 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.5868 | 2 | 1 |
| 403 | 11 | 1.0 | 0.5096 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5971 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4683 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2925 | 0.8018 | 0.2275 |
| validation_calibrated_prob | 0.2449 | 0.67 | 0.1611 |
| final_test_raw_prob | 0.2809 | 0.7613 | 0.2113 |
| final_test_calibrated_prob | 0.257 | 0.7072 | 0.0635 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/holstein_task1_weak_gce_v2_run_sbatch/weak_label_cv_diagnostics.json`
