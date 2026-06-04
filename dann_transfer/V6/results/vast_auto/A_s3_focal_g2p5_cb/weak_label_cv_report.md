# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260531T055441Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Dataset version: `thesis_stride8_qa_549_interim`
- Final test cows: `["363", "370", "378", "403", "404", "408", "433", "436"]`
- Inner folds: `5` folds x `4` validation cows
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Initialization: `/root/data/v2.9_20260502_181533`
- Freeze CNN: `True`

- Task1 proxy loss: `focal` | class-balanced effective-number weighting: `True`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.9415 | 354,402,415,428 | 76 | 0.9149 | 0.0417 | 0.7642 | 0.3947 | 0.5106 | 1.0 | 0.0213 | 29 | 0 | 46 | 1 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.9149 | 0.7642 | 1.0 |
| 1 | 19 | 0.6965 | 323,387,425,432 | 42 | 0.6514 | 0.0 | 0.5517 | 0.619 | 0.5 | 0.0 | 0.0 | 26 | 0 | 16 | 0 | 0.75 | 0.0 | 0.6667 | 1.1473 | 0.6514 | 0.5517 | 0.75 |
| 2 | 0 | 0.4604 | 394,406,426,438 | 43 | 0.3565 | 0.0952 | 0.6349 | 0.5581 | 0.525 | 1.0 | 0.05 | 23 | 0 | 19 | 1 | 0.5 | 0.0 | 0.6667 | 2.3599 | 0.3565 | 0.6349 | 0.5 |
| 3 | 0 | 0.8131 | 310,405,421,439 | 70 | 0.7199 | 0.1695 | 0.871 | 0.3 | 0.5463 | 1.0 | 0.0926 | 16 | 0 | 49 | 5 | 1.0 | 0.0 | 0.8 | 10.0 | 0.7199 | 0.871 | 1.0 |
| 4 | 35 | 0.9578 | 255,352,355,446 | 49 | 0.949 | 0.0571 | 0.8947 | 0.3265 | 0.5147 | 1.0 | 0.0294 | 15 | 0 | 33 | 1 | 1.0 | 0.0 | 1.0 | 10.0 | 0.949 | 0.8193 | 1.0 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.3677 | 0.6154 | 0.5932 | 0.466 | 0.4138 | 0.5333 | 0.6107 | 0.35 | 0.4813 | 0.35 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | True | 64 | 34 | 21 | 24 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.3677 | 0.5263 | 0.609 | 0.5263 | 0.3846 | 0.8333 | 0.5385 | 0.05 | 0.48 | 0.05 | 0.3768 | 0.6795 | 0.6923 | 0.6667 | 0.6795 | 0.3768 | 0.6795 | 0.6923 | 0.6667 | True | 5 | 8 | 1 | 5 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.3677 | 0.75 | 0.8 | 0.75 | 0.6 | 1.0 | 0.6667 | 0.05 | 0.5455 | 0.05 | 0.3685 | 0.8 | 0.6 | 1.0 | 0.8 | 0.3685 | 0.8 | 0.6 | 1.0 | True | 3 | 2 | 0 | 3 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6107 | 0.35 | 0.4813 | 0.35 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.3677 | 0.6154 | 0.5932 | 0.466 | 0.4138 | 0.5333 | 0.6107 | 0.35 | 0.4813 | 0.35 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | True | 64 | 34 | 21 | 24 |
| youden | 143 | 45 | 98 | 0.3677 | 0.6154 | 0.5932 | 0.466 | 0.4138 | 0.5333 | 0.6107 | 0.35 | 0.4813 | 0.35 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | True | 64 | 34 | 21 | 24 |
| specificity_constrained | 143 | 45 | 98 | 0.3677 | 0.6154 | 0.5932 | 0.466 | 0.4138 | 0.5333 | 0.6107 | 0.35 | 0.4813 | 0.35 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.3711 | 0.6442 | 0.7551 | 0.5333 | True | 64 | 34 | 21 | 24 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4865 | 0.6434 | 0.4754 | 0.0377 | 0.125 | 0.0222 | 0.6107 | 0.05 | 0.4787 | 0.05 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | True | 91 | 7 | 44 | 1 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4865 | 0.6316 | 0.4615 | 0.0 | 0.0 | 0.0 | 0.5385 | 0.05 | 0.48 | 0.05 | 0.4812 | 0.6795 | 0.6923 | 0.6667 | 0.6795 | 0.4812 | 0.6795 | 0.6923 | 0.6667 | True | 12 | 1 | 6 | 0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4865 | 0.625 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6667 | 0.05 | 0.5455 | 0.05 | 0.4799 | 0.8 | 0.6 | 1.0 | 0.8 | 0.4799 | 0.8 | 0.6 | 1.0 | True | 5 | 0 | 3 | 0 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.6107 | 0.05 | 0.4787 | 0.05 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.4865 | 0.6434 | 0.4754 | 0.0377 | 0.125 | 0.0222 | 0.6107 | 0.05 | 0.4787 | 0.05 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | True | 91 | 7 | 44 | 1 |
| youden | 143 | 45 | 98 | 0.4865 | 0.6434 | 0.4754 | 0.0377 | 0.125 | 0.0222 | 0.6107 | 0.05 | 0.4787 | 0.05 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | True | 91 | 7 | 44 | 1 |
| specificity_constrained | 143 | 45 | 98 | 0.4865 | 0.6434 | 0.4754 | 0.0377 | 0.125 | 0.0222 | 0.6107 | 0.05 | 0.4787 | 0.05 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | 0.6442 | 0.4803 | 0.6442 | 0.7551 | 0.5333 | True | 91 | 7 | 44 | 1 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.3959 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.3729 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.3866 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.3598 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.3781 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3588 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.3934 | 3 | 1 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 7 | 378 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 0.0 | 0.3728 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 1.0 | 0.3699 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 11 | 433 | Yashan Dhaliwal RAC Data 2025 | 12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 1.0 | 0.3774 | 11 | 1 |
| Yashan Dhaliwal RAC Data 2025::19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 17 | 378 | Yashan Dhaliwal RAC Data 2025 | 19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 0.0 | 0.3612 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 10 | 436 | Yashan Dhaliwal RAC Data 2025 | 20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 0.0 | 0.361 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 16 | 370 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 0.0 | 0.371 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 9 | 436 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 0.0 | 0.3594 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 5 | 370 | Yashan Dhaliwal RAC Data 2025 | 24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 0.0 | 0.4309 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 9 | 378 | Yashan Dhaliwal RAC Data 2025 | 28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 0.0 | 0.3762 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 1.0 | 0.3781 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 3 | 370 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 0.0 | 0.4094 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 11 | 436 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 0.0 | 0.3588 | 0 | 0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.3781 | 6 | 1 |
| 370 | 24 | 0.0 | 0.3882 | 0 | 0 |
| 378 | 33 | 0.0 | 0.3677 | 0 | 0 |
| 403 | 10 | 1.0 | 0.3692 | 10 | 1 |
| 404 | 5 | 0.0 | 0.3775 | 0 | 0 |
| 408 | 6 | 0.0 | 0.3643 | 0 | 0 |
| 433 | 29 | 1.0 | 0.3753 | 29 | 1 |
| 436 | 30 | 0.0 | 0.3597 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2697 | 0.733 | 0.2239 |
| validation_calibrated_prob | 0.2484 | 0.69 | 0.1461 |
| final_test_raw_prob | 0.2177 | 0.6272 | 0.0571 |
| final_test_calibrated_prob | 0.2429 | 0.6789 | 0.1657 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1960 | 0.6667 | 0.2 | 1.0 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1952 | 0.6667 | 0.25 | 1.0 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `weak_label_cv_diagnostics.json`.


## Artifacts

- `split_json`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_splits.json`
- `summary_json`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_summary.json`
- `fold_summary_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_test_predictions.csv`
- `test_video_aggregates_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/root/runs/v6_auto/A_s3_focal_g2p5_cb/weak_label_cv_diagnostics.json`
