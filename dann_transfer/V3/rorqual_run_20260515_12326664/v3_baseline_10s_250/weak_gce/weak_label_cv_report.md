# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260514T213939Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `baseline_10s_250`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/scratch/shiv136/project_data/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `gce` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 26 | 0.8731 | 374,402,432,433 | 37 | 0.8037 | 0.625 | 0.6667 | 0.8378 | 0.7315 | 0.8333 | 0.5 | 26 | 1 | 5 | 5 | 1.0 | 0.0 | 1.0 | 0.7096 | 0.8037 | 0.6667 | 1.0 |
| 1 | 17 | 0.6119 | 310,378,417,427 | 37 | 0.4854 | 0.5 | 0.6786 | 0.4595 | 0.4576 | 0.4762 | 0.5263 | 7 | 11 | 9 | 10 | 0.75 | 0.6667 | 0.6667 | 10.0 | 0.4854 | 0.6786 | 0.5 |
| 2 | 0 | 0.7121 | 370,394,415,436 | 37 | 0.6676 | 0.1905 | 0.6296 | 0.5405 | 0.5088 | 0.5 | 0.1176 | 18 | 2 | 15 | 2 | 0.75 | 0.0 | 0.6667 | 1.7639 | 0.6676 | 0.6667 | 0.75 |
| 3 | 35 | 0.6432 | 323,349,352,439 | 20 | 0.5312 | 0.5 | 0.75 | 0.5 | 0.5208 | 0.625 | 0.4167 | 5 | 3 | 7 | 5 | 0.75 | 0.6667 | 0.6667 | 10.0 | 0.5312 | 0.75 | 0.5 |
| 4 | 51 | 0.9116 | 354,405,421,428 | 28 | 0.8776 | 0.7647 | 0.913 | 0.7143 | 0.8095 | 1.0 | 0.619 | 7 | 0 | 8 | 13 | 1.0 | 1.0 | 1.0 | 1.2755 | 0.8776 | 0.913 | 1.0 |
| 5 | 23 | 0.7915 | 406,425,426,438 | 32 | 0.6625 | 0.3846 | 0.7692 | 0.5 | 0.5833 | 0.8333 | 0.25 | 11 | 1 | 15 | 5 | 1.0 | 0.0 | 1.0 | 10.0 | 0.6625 | 0.7692 | 1.0 |
| 6 | 19 | 0.9353 | 255,355,387,446 | 30 | 0.9156 | 0.7778 | 0.8276 | 0.7333 | 0.7333 | 0.6667 | 0.9333 | 8 | 7 | 1 | 14 | 1.0 | 0.8 | 1.0 | 1.1841 | 0.9156 | 0.8571 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.7832 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5774 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5774 | 0.5048 | 0.625 | 0.3846 | True | 13 | 3 | 12 | 1 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.7832 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.3797 | 0.75 | 0.5 | 1.0 | 0.75 | 0.3797 | 0.75 | 0.5 | 1.0 | True | 2 | 0 | 2 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4138 | 0.4111 | 0.3704 | 0.3571 | 0.3846 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5774 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5774 | 0.5048 | 0.625 | 0.3846 | True | 7 | 9 | 8 | 5 |
| f1 | 29 | 13 | 16 | 0.0 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5774 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5774 | 0.5048 | 0.625 | 0.3846 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.7832 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5774 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5774 | 0.5048 | 0.625 | 0.3846 | True | 13 | 3 | 12 | 1 |
| specificity_constrained | 29 | 13 | 16 | 0.7832 | 0.4828 | 0.4447 | 0.1176 | 0.25 | 0.0769 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5774 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5774 | 0.5048 | 0.625 | 0.3846 | True | 13 | 3 | 12 | 1 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.4619 | 0.4138 | 0.4183 | 0.4138 | 0.375 | 0.4615 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5156 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5156 | 0.5048 | 0.625 | 0.3846 | True | 6 | 10 | 7 | 6 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4619 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.473 | 0.75 | 0.5 | 1.0 | 0.75 | 0.473 | 0.75 | 0.5 | 1.0 | True | 0 | 2 | 0 | 2 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4138 | 0.4111 | 0.3704 | 0.3571 | 0.3846 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5156 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5156 | 0.5048 | 0.625 | 0.3846 | True | 7 | 9 | 8 | 5 |
| f1 | 29 | 13 | 16 | 0.4521 | 0.4483 | 0.4567 | 0.4667 | 0.4118 | 0.5385 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5156 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5156 | 0.5048 | 0.625 | 0.3846 | True | 6 | 10 | 6 | 7 |
| youden | 29 | 13 | 16 | 0.4521 | 0.4483 | 0.4567 | 0.4667 | 0.4118 | 0.5385 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5156 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5156 | 0.5048 | 0.625 | 0.3846 | True | 6 | 10 | 6 | 7 |
| specificity_constrained | 29 | 13 | 16 | 0.4619 | 0.4138 | 0.4183 | 0.4138 | 0.375 | 0.4615 | 0.399 | 0.05 | 0.619 | 0.05 | 0.5156 | 0.5048 | 0.625 | 0.3846 | 0.5048 | 0.5156 | 0.5048 | 0.625 | 0.3846 | True | 6 | 10 | 7 | 6 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.566 | 2 | 1 |
| 403 | 11 | 1.0 | 0.3911 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5685 | 0 | 0 |
| 408 | 6 | 0.0 | 0.3684 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2469 | 0.695 | 0.1542 |
| validation_calibrated_prob | 0.212 | 0.6068 | 0.091 |
| final_test_raw_prob | 0.342 | 0.9209 | 0.3082 |
| final_test_calibrated_prob | 0.2612 | 0.7157 | 0.1011 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.5 | 0.0 | 1.0 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/weak_gce/weak_label_cv_diagnostics.json`
