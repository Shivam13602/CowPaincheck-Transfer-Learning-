# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260514T224939Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `baseline_10s_250`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/scratch/shiv136/project_data/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `focal` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.9059 | 374,402,432,433 | 37 | 0.8741 | 0.0 | 0.5714 | 0.7297 | 0.5 | 0.0 | 0.0 | 27 | 0 | 10 | 0 | 1.0 | 0.0 | 1.0 | 0.1986 | 0.8741 | 0.6667 | 1.0 |
| 1 | 6 | 0.4994 | 310,378,417,427 | 37 | 0.4532 | 0.4444 | 0.6786 | 0.5946 | 0.6023 | 0.75 | 0.3158 | 16 | 2 | 13 | 6 | 0.5 | 0.0 | 0.6667 | 10.0 | 0.4532 | 0.6786 | 0.5 |
| 2 | 0 | 0.6303 | 370,394,415,436 | 37 | 0.55 | 0.1905 | 0.6296 | 0.5405 | 0.5088 | 0.5 | 0.1176 | 18 | 2 | 15 | 2 | 0.75 | 0.0 | 0.6667 | 1.6492 | 0.55 | 0.6296 | 0.75 |
| 3 | 2 | 0.5943 | 323,349,352,439 | 20 | 0.6354 | 0.2857 | 0.75 | 0.5 | 0.5833 | 1.0 | 0.1667 | 8 | 0 | 10 | 2 | 0.5 | 0.6667 | 0.6667 | 2.6695 | 0.6354 | 0.75 | 0.5 |
| 4 | 2 | 0.9259 | 354,405,421,428 | 28 | 0.8776 | 0.8261 | 0.8571 | 0.7143 | 0.5238 | 0.76 | 0.9048 | 1 | 6 | 2 | 19 | 1.0 | 0.6667 | 0.6667 | 0.05 | 0.8776 | 0.878 | 1.0 |
| 5 | 22 | 0.7623 | 406,425,426,438 | 32 | 0.6125 | 0.1739 | 0.7692 | 0.4062 | 0.5083 | 0.6667 | 0.1 | 11 | 1 | 18 | 2 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.6125 | 0.7692 | 1.0 |
| 6 | 0 | 0.9227 | 255,355,387,446 | 30 | 0.9022 | 0.3333 | 0.72 | 0.6 | 0.6 | 1.0 | 0.2 | 15 | 0 | 12 | 3 | 1.0 | 0.6667 | 1.0 | 0.3139 | 0.9022 | 0.8235 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.4507 | 0.4138 | 0.4255 | 0.4516 | 0.3889 | 0.5385 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4726 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4726 | 0.637 | 0.8125 | 0.4615 | True | 5 | 11 | 6 | 7 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4507 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.4585 | 0.75 | 0.5 | 1.0 | 0.75 | 0.4585 | 0.75 | 0.5 | 1.0 | True | 0 | 2 | 0 | 2 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4726 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4726 | 0.637 | 0.8125 | 0.4615 | True | 13 | 3 | 12 | 1 |
| f1 | 29 | 13 | 16 | 0.432 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4726 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4726 | 0.637 | 0.8125 | 0.4615 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.4507 | 0.4138 | 0.4255 | 0.4516 | 0.3889 | 0.5385 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4726 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4726 | 0.637 | 0.8125 | 0.4615 | True | 5 | 11 | 6 | 7 |
| specificity_constrained | 29 | 13 | 16 | 0.4507 | 0.4138 | 0.4255 | 0.4516 | 0.3889 | 0.5385 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4726 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4726 | 0.637 | 0.8125 | 0.4615 | True | 5 | 11 | 6 | 7 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.4631 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4923 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4923 | 0.637 | 0.8125 | 0.4615 | True | 0 | 16 | 0 | 13 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4631 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.4883 | 0.75 | 0.5 | 1.0 | 0.75 | 0.4883 | 0.75 | 0.5 | 1.0 | True | 0 | 2 | 0 | 2 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4923 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4923 | 0.637 | 0.8125 | 0.4615 | True | 13 | 3 | 12 | 1 |
| f1 | 29 | 13 | 16 | 0.2942 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4923 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4923 | 0.637 | 0.8125 | 0.4615 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.3061 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4923 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4923 | 0.637 | 0.8125 | 0.4615 | True | 0 | 16 | 0 | 13 |
| specificity_constrained | 29 | 13 | 16 | 0.4631 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.476 | 0.05 | 0.619 | 0.05 | 0.4923 | 0.637 | 0.8125 | 0.4615 | 0.637 | 0.4923 | 0.637 | 0.8125 | 0.4615 | True | 0 | 16 | 0 | 13 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.4865 | 2 | 1 |
| 403 | 11 | 1.0 | 0.4631 | 11 | 1 |
| 404 | 10 | 0.0 | 0.4961 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4539 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2406 | 0.6742 | 0.07 |
| validation_calibrated_prob | 0.2185 | 0.6156 | 0.0984 |
| final_test_raw_prob | 0.2565 | 0.7064 | 0.0782 |
| final_test_calibrated_prob | 0.2514 | 0.6959 | 0.0445 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_focal/weak_label_cv_diagnostics.json`
