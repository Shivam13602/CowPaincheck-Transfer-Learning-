# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260529T014003Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `thesis_stride8_qa_549_interim`
- Final test cows: `["363", "370", "378", "403", "404", "408", "433", "436"]`
- Inner folds: `5` folds x `4` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/scratch/shiv136/project_data/v2.9_20260502_181533`
- Freeze CNN: `True`

- Task1 proxy loss: `bce` | class-balanced effective-number weighting: `True`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6 | 0.9359 | 354,402,415,428 | 76 | 0.9054 | 0.0 | 0.8276 | 0.3816 | 0.5 | 0.0 | 0.0 | 29 | 0 | 47 | 0 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.9054 | 0.7642 | 1.0 |
| 1 | 0 | 0.423 | 323,387,425,432 | 42 | 0.3149 | 0.0 | 0.5517 | 0.619 | 0.5 | 0.0 | 0.0 | 26 | 0 | 16 | 0 | 0.5 | 0.0 | 0.6667 | 1.0 | 0.3149 | 0.5517 | 0.5 |
| 2 | 11 | 0.8499 | 394,406,426,438 | 43 | 0.7717 | 0.0 | 0.6349 | 0.5349 | 0.5 | 0.0 | 0.0 | 23 | 0 | 20 | 0 | 1.0 | 0.0 | 0.6667 | 6.2243 | 0.7717 | 0.6349 | 1.0 |
| 3 | 4 | 0.8161 | 310,405,421,439 | 70 | 0.7164 | 0.1053 | 0.871 | 0.2714 | 0.5278 | 1.0 | 0.0556 | 16 | 0 | 51 | 3 | 1.0 | 0.0 | 1.0 | 10.0 | 0.7164 | 0.871 | 1.0 |
| 4 | 0 | 0.8224 | 255,352,355,446 | 49 | 0.7098 | 0.381 | 0.8193 | 0.4694 | 0.6176 | 1.0 | 0.2353 | 15 | 0 | 26 | 8 | 1.0 | 0.0 | 0.8 | 10.0 | 0.7098 | 0.8193 | 1.0 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4262 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4262 | 0.6842 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4231 | 0.05 | 0.48 | 0.05 | 0.3283 | 0.5577 | 0.6154 | 0.5 | 0.5577 | 0.3283 | 0.5577 | 0.6154 | 0.5 | True | 13 | 0 | 6 | 0 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4262 | 0.625 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.05 | 0.5455 | 0.05 | 0.325 | 0.6333 | 0.6 | 0.6667 | 0.6333 | 0.325 | 0.6333 | 0.6 | 0.6667 | True | 5 | 0 | 3 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.296 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |
| youden | 143 | 45 | 98 | 0.296 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |
| specificity_constrained | 143 | 45 | 98 | 0.4262 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3311 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4784 | 0.6434 | 0.4874 | 0.1053 | 0.25 | 0.0667 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | True | 89 | 9 | 42 | 3 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4784 | 0.5789 | 0.4679 | 0.2 | 0.25 | 0.1667 | 0.4231 | 0.05 | 0.48 | 0.05 | 0.476 | 0.5577 | 0.6154 | 0.5 | 0.5577 | 0.476 | 0.5577 | 0.6154 | 0.5 | True | 10 | 3 | 5 | 1 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4784 | 0.625 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.05 | 0.5455 | 0.05 | 0.4754 | 0.6333 | 0.6 | 0.6667 | 0.6333 | 0.4754 | 0.6333 | 0.6 | 0.6667 | True | 5 | 0 | 3 | 0 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.4784 | 0.6434 | 0.4874 | 0.1053 | 0.25 | 0.0667 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | True | 89 | 9 | 42 | 3 |
| youden | 143 | 45 | 98 | 0.4784 | 0.6434 | 0.4874 | 0.1053 | 0.25 | 0.0667 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | True | 89 | 9 | 42 | 3 |
| specificity_constrained | 143 | 45 | 98 | 0.4784 | 0.6434 | 0.4874 | 0.1053 | 0.25 | 0.0667 | 0.4798 | 0.05 | 0.4787 | 0.05 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4764 | 0.5586 | 0.8061 | 0.3111 | True | 89 | 9 | 42 | 3 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.3391 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.3201 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.3501 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.3256 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.3139 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3176 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.3456 | 3 | 1 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 7 | 378 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 0.0 | 0.3171 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 1.0 | 0.321 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 11 | 433 | Yashan Dhaliwal RAC Data 2025 | 12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 1.0 | 0.33 | 11 | 1 |
| Yashan Dhaliwal RAC Data 2025::19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 17 | 378 | Yashan Dhaliwal RAC Data 2025 | 19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 0.0 | 0.3189 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 10 | 436 | Yashan Dhaliwal RAC Data 2025 | 20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 0.0 | 0.3206 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 16 | 370 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 0.0 | 0.3266 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 9 | 436 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 0.0 | 0.3234 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 5 | 370 | Yashan Dhaliwal RAC Data 2025 | 24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 0.0 | 0.3901 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 9 | 378 | Yashan Dhaliwal RAC Data 2025 | 28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 0.0 | 0.3336 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 1.0 | 0.3332 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 3 | 370 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 0.0 | 0.3495 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 11 | 436 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 0.0 | 0.3174 | 0 | 0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.3139 | 6 | 1 |
| 370 | 24 | 0.0 | 0.3427 | 0 | 0 |
| 378 | 33 | 0.0 | 0.3225 | 0 | 0 |
| 403 | 10 | 1.0 | 0.326 | 10 | 1 |
| 404 | 5 | 0.0 | 0.3239 | 0 | 0 |
| 408 | 6 | 0.0 | 0.3297 | 0 | 0 |
| 433 | 29 | 1.0 | 0.3282 | 29 | 1 |
| 436 | 30 | 0.0 | 0.3203 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2982 | 0.7969 | 0.2661 |
| validation_calibrated_prob | 0.2506 | 0.6943 | 0.1464 |
| final_test_raw_prob | 0.2167 | 0.6252 | 0.0135 |
| final_test_calibrated_prob | 0.2417 | 0.6765 | 0.161 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1960 | 0.3875 | 0.0 | 0.8571 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1952 | 0.375 | 0.0 | 1.0 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_test_predictions.csv`
- `test_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S3_weak_bce/weak_label_cv_diagnostics.json`
