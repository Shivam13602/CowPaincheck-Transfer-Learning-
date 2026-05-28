# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260526T191821Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `thesis_stride8_qa`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `9` folds x `3` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/scratch/shiv136/project_data/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `focal` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 41 | 1.0 | 352,438,439 | 21 | 1.0 | 0.8696 | 1.0 | 0.8571 | 0.8636 | 0.7692 | 1.0 | 8 | 3 | 0 | 10 | 1.0 | 1.0 | 1.0 | 0.1579 | 1.0 | 1.0 | 1.0 |
| 1 | 11 | 0.9262 | 255,406,417 | 61 | 0.8989 | 0.6316 | 0.8 | 0.7705 | 0.7261 | 0.9231 | 0.48 | 35 | 1 | 13 | 12 | 1.0 | 0.0 | 1.0 | 0.2922 | 0.8989 | 0.8333 | 1.0 |
| 2 | 2 | 0.9512 | 428,433,436 | 69 | 0.9388 | 0.0 | 0.8621 | 0.5797 | 0.5 | 0.0 | 0.0 | 40 | 0 | 29 | 0 | 1.0 | 0.0 | 1.0 | 0.5911 | 0.9388 | 0.8235 | 1.0 |
| 3 | 1 | 0.6988 | 355,405,427 | 32 | 0.5083 | 0.0 | 0.5854 | 0.625 | 0.5 | 0.0 | 0.0 | 20 | 0 | 12 | 0 | 1.0 | 0.0 | 0.5 | 0.5411 | 0.5083 | 0.5455 | 1.0 |
| 4 | 0 | 0.845 | 378,394,432 | 57 | 0.7596 | 0.1538 | 0.3478 | 0.807 | 0.5287 | 0.3333 | 0.1 | 45 | 2 | 9 | 1 | 1.0 | 0.0 | 1.0 | 0.1437 | 0.7596 | 0.4211 | 1.0 |
| 5 | 45 | 0.8271 | 310,323,426 | 48 | 0.7353 | 0.0571 | 0.8395 | 0.3125 | 0.5147 | 1.0 | 0.0294 | 14 | 0 | 33 | 1 | 1.0 | 0.0 | 1.0 | 10.0 | 0.7353 | 0.8293 | 1.0 |
| 6 | 7 | 0.9886 | 354,402,415 | 66 | 0.9888 | 0.907 | 0.9574 | 0.8788 | 0.9149 | 1.0 | 0.8298 | 19 | 0 | 8 | 39 | 1.0 | 1.0 | 1.0 | 0.1091 | 0.9888 | 0.9677 | 1.0 |
| 7 | 59 | 0.9981 | 349,387,425 | 102 | 0.9981 | 0.9888 | 0.9944 | 0.9804 | 0.9889 | 1.0 | 0.9778 | 12 | 0 | 2 | 88 | 1.0 | 0.6667 | 1.0 | 0.2474 | 0.9981 | 0.9944 | 1.0 |
| 8 | 2 | 0.7137 | 370,421,446 | 66 | 0.5317 | 0.3774 | 0.7778 | 0.5 | 0.5982 | 0.9091 | 0.2381 | 23 | 1 | 32 | 10 | 1.0 | 0.0 | 0.8 | 10.0 | 0.5317 | 0.7778 | 1.0 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 16 | 11 | 0.4653 | 0.5185 | 0.5511 | 0.48 | 0.6667 | 0.375 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4025 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4025 | 0.6136 | 0.7273 | 0.5 | True | 8 | 3 | 10 | 6 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 3 | 4 | 0.4653 | 0.5714 | 0.5417 | 0.4 | 0.5 | 0.3333 | 0.4167 | 0.05 | 0.6 | 0.05 | 0.5 | 0.6667 | 1.0 | 0.3333 | 0.6667 | 0.5 | 0.6667 | 1.0 | 0.3333 | True | 3 | 1 | 2 | 1 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4653 | 0.75 | 0.75 | 0.6667 | 1.0 | 0.5 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.4861 | 0.75 | 1.0 | 0.5 | 0.75 | 0.4861 | 0.75 | 1.0 | 0.5 | True | 2 | 0 | 1 | 1 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 27 | 16 | 11 | 0.5 | 0.5185 | 0.5653 | 0.4348 | 0.7143 | 0.3125 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4025 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4025 | 0.6136 | 0.7273 | 0.5 | True | 9 | 2 | 11 | 5 |
| f1 | 27 | 16 | 11 | 0.4339 | 0.5556 | 0.5824 | 0.5385 | 0.7 | 0.4375 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4025 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4025 | 0.6136 | 0.7273 | 0.5 | True | 8 | 3 | 9 | 7 |
| youden | 27 | 16 | 11 | 0.4653 | 0.5185 | 0.5511 | 0.48 | 0.6667 | 0.375 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4025 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4025 | 0.6136 | 0.7273 | 0.5 | True | 8 | 3 | 10 | 6 |
| specificity_constrained | 27 | 16 | 11 | 0.4653 | 0.5185 | 0.5511 | 0.48 | 0.6667 | 0.375 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4025 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4025 | 0.6136 | 0.7273 | 0.5 | True | 8 | 3 | 10 | 6 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 16 | 11 | 0.3599 | 0.5926 | 0.5 | 0.7442 | 0.5926 | 1.0 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4598 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4598 | 0.6136 | 0.7273 | 0.5 | True | 0 | 11 | 0 | 16 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 3 | 4 | 0.3599 | 0.4286 | 0.5 | 0.6 | 0.4286 | 1.0 | 0.4167 | 0.05 | 0.6 | 0.05 | 0.5 | 0.6667 | 1.0 | 0.3333 | 0.6667 | 0.5 | 0.6667 | 1.0 | 0.3333 | True | 0 | 4 | 0 | 3 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.3599 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.05 | 0.6667 | 0.05 | 0.4941 | 0.75 | 1.0 | 0.5 | 0.75 | 0.4941 | 0.75 | 1.0 | 0.5 | True | 0 | 2 | 0 | 2 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 27 | 16 | 11 | 0.5 | 0.5185 | 0.5653 | 0.4348 | 0.7143 | 0.3125 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4598 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4598 | 0.6136 | 0.7273 | 0.5 | True | 9 | 2 | 11 | 5 |
| f1 | 27 | 16 | 11 | 0.3371 | 0.5926 | 0.5 | 0.7442 | 0.5926 | 1.0 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4598 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4598 | 0.6136 | 0.7273 | 0.5 | True | 0 | 11 | 0 | 16 |
| youden | 27 | 16 | 11 | 0.3599 | 0.5926 | 0.5 | 0.7442 | 0.5926 | 1.0 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4598 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4598 | 0.6136 | 0.7273 | 0.5 | True | 0 | 11 | 0 | 16 |
| specificity_constrained | 27 | 16 | 11 | 0.3599 | 0.5926 | 0.5 | 0.7442 | 0.5926 | 1.0 | 0.4205 | 0.05 | 0.7442 | 0.05 | 0.4598 | 0.6136 | 0.7273 | 0.5 | 0.6136 | 0.4598 | 0.6136 | 0.7273 | 0.5 | True | 0 | 11 | 0 | 16 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.4979 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.4421 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.401 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.3632 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.519 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3335 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.3762 | 3 | 1 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.519 | 6 | 1 |
| 403 | 10 | 1.0 | 0.3463 | 10 | 1 |
| 404 | 5 | 0.0 | 0.4533 | 0 | 0 |
| 408 | 6 | 0.0 | 0.3695 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.1958 | 0.5763 | 0.155 |
| validation_calibrated_prob | 0.1478 | 0.4293 | 0.1227 |
| final_test_raw_prob | 0.2801 | 0.7577 | 0.2723 |
| final_test_calibrated_prob | 0.2592 | 0.7119 | 0.1305 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_test_predictions.csv`
- `test_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/weak_focal/weak_label_cv_diagnostics.json`
