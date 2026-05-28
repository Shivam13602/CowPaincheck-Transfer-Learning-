# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260514T203004Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `baseline_10s_250`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/scratch/shiv136/project_data/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `bce` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.8576 | 374,402,432,433 | 37 | 0.7889 | 0.0 | 0.4615 | 0.7297 | 0.5 | 0.0 | 0.0 | 27 | 0 | 10 | 0 | 1.0 | 0.0 | 0.4 | 0.2329 | 0.7889 | 0.5882 | 1.0 |
| 1 | 12 | 0.5113 | 310,378,417,427 | 37 | 0.4678 | 0.5 | 0.6786 | 0.5135 | 0.5146 | 0.5294 | 0.4737 | 10 | 8 | 10 | 9 | 0.5 | 0.5 | 0.6667 | 10.0 | 0.4678 | 0.6786 | 0.5 |
| 2 | 0 | 0.7129 | 370,394,415,436 | 37 | 0.6676 | 0.1905 | 0.6296 | 0.5405 | 0.5088 | 0.5 | 0.1176 | 18 | 2 | 15 | 2 | 0.75 | 0.0 | 0.6667 | 1.5765 | 0.6676 | 0.6296 | 0.75 |
| 3 | 41 | 0.6255 | 323,349,352,439 | 20 | 0.5104 | 0.5263 | 0.75 | 0.55 | 0.5833 | 0.7143 | 0.4167 | 6 | 2 | 7 | 5 | 0.75 | 0.6667 | 0.6667 | 4.32 | 0.5104 | 0.75 | 0.5 |
| 4 | 4 | 0.9259 | 354,405,421,428 | 28 | 0.8776 | 0.8511 | 0.8571 | 0.75 | 0.5476 | 0.7692 | 0.9524 | 1 | 6 | 1 | 20 | 1.0 | 0.6667 | 1.0 | 0.227 | 0.8776 | 0.875 | 1.0 |
| 5 | 0 | 0.7698 | 406,425,426,438 | 32 | 0.6292 | 0.4444 | 0.7692 | 0.5312 | 0.6083 | 0.8571 | 0.3 | 11 | 1 | 14 | 6 | 1.0 | 0.0 | 0.6667 | 1.6671 | 0.6292 | 0.7692 | 1.0 |
| 6 | 6 | 0.9253 | 255,355,387,446 | 30 | 0.8933 | 0.7568 | 0.8125 | 0.7 | 0.7 | 0.6364 | 0.9333 | 7 | 8 | 1 | 14 | 1.0 | 0.8 | 1.0 | 0.4187 | 0.8933 | 0.8387 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.5788 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5291 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5291 | 0.5673 | 0.75 | 0.3846 | True | 13 | 3 | 12 | 1 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.5788 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.5 | 0.45 | 0.8 | 0.45 | 0.45 | 0.75 | 0.5 | 1.0 | 0.75 | 0.45 | 0.75 | 0.5 | 1.0 | True | 2 | 0 | 2 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4483 | 0.4423 | 0.3846 | 0.3846 | 0.3846 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5291 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5291 | 0.5673 | 0.75 | 0.3846 | True | 8 | 8 | 8 | 5 |
| f1 | 29 | 13 | 16 | 0.0 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5291 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5291 | 0.5673 | 0.75 | 0.3846 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.5788 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5291 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5291 | 0.5673 | 0.75 | 0.3846 | True | 13 | 3 | 12 | 1 |
| specificity_constrained | 29 | 13 | 16 | 0.5788 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5291 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5291 | 0.5673 | 0.75 | 0.3846 | True | 13 | 3 | 12 | 1 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.5079 | 0.5172 | 0.5048 | 0.4167 | 0.4545 | 0.3846 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5111 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5111 | 0.5673 | 0.75 | 0.3846 | True | 10 | 6 | 8 | 5 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.5079 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.4836 | 0.75 | 0.5 | 1.0 | 0.75 | 0.4836 | 0.75 | 0.5 | 1.0 | True | 1 | 1 | 1 | 1 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4483 | 0.4423 | 0.3846 | 0.3846 | 0.3846 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5111 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5111 | 0.5673 | 0.75 | 0.3846 | True | 8 | 8 | 8 | 5 |
| f1 | 29 | 13 | 16 | 0.2581 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5111 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5111 | 0.5673 | 0.75 | 0.3846 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.5079 | 0.5172 | 0.5048 | 0.4167 | 0.4545 | 0.3846 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5111 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5111 | 0.5673 | 0.75 | 0.3846 | True | 10 | 6 | 8 | 5 |
| specificity_constrained | 29 | 13 | 16 | 0.5079 | 0.5172 | 0.5048 | 0.4167 | 0.4545 | 0.3846 | 0.4327 | 0.05 | 0.619 | 0.05 | 0.5111 | 0.5673 | 0.75 | 0.3846 | 0.5673 | 0.5111 | 0.5673 | 0.75 | 0.3846 | True | 10 | 6 | 8 | 5 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.526 | 2 | 1 |
| 403 | 11 | 1.0 | 0.4667 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5449 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4477 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2348 | 0.6628 | 0.0996 |
| validation_calibrated_prob | 0.2155 | 0.6119 | 0.0724 |
| final_test_raw_prob | 0.2741 | 0.7438 | 0.2164 |
| final_test_calibrated_prob | 0.2574 | 0.7082 | 0.1078 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_bce/weak_label_cv_diagnostics.json`
