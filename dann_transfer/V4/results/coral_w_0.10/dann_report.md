# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260526T232104Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Dataset version: `thesis_stride8_qa`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `9` folds x `3` validation cows
- Source project: `/scratch/shiv136/project_data/ucaps_source`
- Source fold: `0` | source train n: `309`
- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.
- Source Task1 retention gate: AUC >= max(`0.55`, initial_source_auc - `0.03`); absolute `0.7` is diagnostic.
- Source Task1 loss: `bce` | source SupCon weight: `0.0` | class-balanced: `False`
- Alignment loss: `coral` | domain weight: `0.0` | coral weight: `0.1` | domain lambda max: `1.0`
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Target weak BCE weight: `0.0` starting epoch `5`
- SSL checkpoint dir: `None`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a domain-adaptation diagnostic, not as validated pain detection.

## Validation Folds — UCAPS source Task1 vs Holstein proxy

_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; `source_task1_retention_pass` = primary source-retention gate; `checkpoint_selected_from_proxy_fallback` = no epoch passed retention so the best proxy epoch was used._

| fold | best_epoch | best_score | proxy_selection_score | source_task1_auc_init | source_task1_retention_floor | source_task1_retention_pass | source_task1_sanity_floor | source_task1_sanity_pass | checkpoint_selected_from_proxy_fallback | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc | source_task1_auc | source_task1_f1 | source_task1_f1_opt | source_task1_balanced_accuracy | source_task1_precision | source_task1_recall | source_task1_best_threshold | source_task2_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 35 | 0.8314 | 0.8314 | 0.5718 | 0.55 | True | 0.7 | False | False | 352,438,439 | 21 | 0.7364 | 0.6667 | 0.6667 | 0.7619 | 0.75 | 1.0 | 0.5 | 1.0 | 0.0 | 1.0 | 0.7786 | 0.7364 | 0.6667 | 1.0 | 0.6282 | 0.7308 | 0.7857 | 0.6447 | 0.7037 | 0.76 | 0.35 | 0.4048 |
| 1 | 1 | 0.861 | 0.861 | 0.5718 | 0.55 | True | 0.7 | False | False | 255,406,417 | 61 | 0.7533 | 0.8372 | 0.8372 | 0.8852 | 0.86 | 1.0 | 0.72 | 1.0 | 1.0 | 1.0 | 0.1546 | 0.7533 | 0.8372 | 1.0 | 0.5906 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 2 | 3 | 0.9649 | 0.9649 | 0.5718 | 0.55 | True | 0.7 | False | False | 428,433,436 | 69 | 0.956 | 0.8302 | 0.8302 | 0.8696 | 0.8543 | 0.9167 | 0.7586 | 1.0 | 1.0 | 1.0 | 0.05 | 0.956 | 0.9123 | 1.0 | 0.5906 | 0.7407 | 0.7719 | 0.6353 | 0.6897 | 0.8 | 0.45 | 0.4048 |
| 3 | 11 | 0.4708 | 0.4708 | 0.5718 | 0.55 | True | 0.7 | False | False | 355,405,427 | 32 | 0.4167 | 0.0 | 0.5455 | 0.625 | 0.5 | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 | 0.7857 | 0.4167 | 0.5455 | 0.5 | 0.6094 | 0.7407 | 0.7857 | 0.6353 | 0.6897 | 0.8 | 0.45 | 0.4048 |
| 4 | 14 | 0.455 | 0.455 | 0.5718 | 0.55 | True | 0.7 | False | False | 378,394,432 | 57 | 0.4 | 0.0 | 0.2985 | 0.8246 | 0.5 | 0.0 | 0.0 | 0.5 | 0.0 | 0.5 | 0.3195 | 0.4 | 0.303 | 0.5 | 0.6071 | 0.7636 | 0.7857 | 0.6553 | 0.7 | 0.84 | 0.45 | 0.4048 |
| 5 | 47 | 0.5957 | 0.5957 | 0.5718 | 0.55 | True | 0.7 | False | False | 310,323,426 | 48 | 0.6071 | 0.0571 | 0.8293 | 0.3125 | 0.5147 | 1.0 | 0.0294 | 0.5 | 0.0 | 0.8 | 10.0 | 0.6071 | 0.8293 | 0.5 | 0.6188 | 0.68 | 0.7719 | 0.6047 | 0.68 | 0.68 | 0.35 | 0.4048 |
| 6 | 0 | 0.9792 | 0.9792 | 0.5718 | 0.55 | True | 0.7 | False | False | 354,402,415 | 66 | 0.9843 | 0.94 | 0.9691 | 0.9091 | 0.8421 | 0.8868 | 1.0 | 1.0 | 1.0 | 1.0 | 0.1542 | 0.9843 | 0.9691 | 1.0 | 0.5882 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 7 | 1 | 0.9832 | 0.9832 | 0.5718 | 0.55 | True | 0.7 | False | False | 349,387,425 | 102 | 0.975 | 0.9595 | 0.9595 | 0.9314 | 0.9611 | 1.0 | 0.9222 | 1.0 | 0.6667 | 0.8 | 0.078 | 0.975 | 0.9714 | 1.0 | 0.5953 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 8 | 29 | 0.7714 | 0.7714 | 0.5718 | 0.55 | True | 0.7 | False | False | 370,421,446 | 66 | 0.631 | 0.3137 | 0.8 | 0.4697 | 0.5744 | 0.8889 | 0.1905 | 1.0 | 0.0 | 0.8 | 10.0 | 0.631 | 0.7778 | 1.0 | 0.6165 | 0.7059 | 0.7857 | 0.6247 | 0.6923 | 0.72 | 0.4 | 0.4048 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 16 | 11 | 0.5 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5309 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5309 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 3 | 4 | 0.5 | 0.4286 | 0.375 | 0.0 | 0.0 | 0.0 | 0.1667 | 0.05 | 0.6 | 0.05 | 0.55 | 0.5 | 1.0 | 0.0 | 0.5 | 0.55 | 0.5 | 1.0 | 0.0 | True | 3 | 1 | 3 | 0 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.5 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.6667 | 0.05 | 0.5 | 0.5 | 1.0 | 0.0 | 0.5 | 0.5 | 0.5 | 1.0 | 0.0 | True | 2 | 0 | 2 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 27 | 16 | 11 | 0.5 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5309 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5309 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |
| f1 | 27 | 16 | 11 | 0.4962 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5309 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5309 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |
| youden | 27 | 16 | 11 | 0.5 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5309 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5309 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |
| specificity_constrained | 27 | 16 | 11 | 0.5 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5309 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5309 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 27 | 16 | 11 | 0.4866 | 0.2963 | 0.3494 | 0.0952 | 0.2 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5125 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5125 | 0.5312 | 1.0 | 0.0625 | True | 7 | 4 | 15 | 1 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 3 | 4 | 0.4866 | 0.2857 | 0.25 | 0.0 | 0.0 | 0.0 | 0.1667 | 0.05 | 0.6 | 0.05 | 0.55 | 0.5 | 1.0 | 0.0 | 0.5 | 0.55 | 0.5 | 1.0 | 0.0 | True | 2 | 2 | 3 | 0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4866 | 0.25 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.05 | 0.6667 | 0.05 | 0.5 | 0.5 | 1.0 | 0.0 | 0.5 | 0.5 | 0.5 | 1.0 | 0.0 | True | 1 | 1 | 2 | 0 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 27 | 16 | 11 | 0.5 | 0.4074 | 0.4858 | 0.1111 | 0.5 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5125 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5125 | 0.5312 | 1.0 | 0.0625 | True | 10 | 1 | 15 | 1 |
| f1 | 27 | 16 | 11 | 0.4866 | 0.2963 | 0.3494 | 0.0952 | 0.2 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5125 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5125 | 0.5312 | 1.0 | 0.0625 | True | 7 | 4 | 15 | 1 |
| youden | 27 | 16 | 11 | 0.4866 | 0.2963 | 0.3494 | 0.0952 | 0.2 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5125 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5125 | 0.5312 | 1.0 | 0.0625 | True | 7 | 4 | 15 | 1 |
| specificity_constrained | 27 | 16 | 11 | 0.4866 | 0.2963 | 0.3494 | 0.0952 | 0.2 | 0.0625 | 0.1989 | 0.05 | 0.7442 | 0.05 | 0.5125 | 0.5312 | 1.0 | 0.0625 | 0.5312 | 0.5125 | 0.5312 | 1.0 | 0.0625 | True | 7 | 4 | 15 | 1 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.526 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.4583 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.4776 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.4216 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.4184 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.406 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.4638 | 3 | 1 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.4184 | 6 | 1 |
| 403 | 10 | 1.0 | 0.4233 | 10 | 1 |
| 404 | 5 | 0.0 | 0.4718 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4309 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2068 | 0.6035 | 0.1397 |
| validation_calibrated_prob | 0.1458 | 0.4401 | 0.0957 |
| final_test_raw_prob | 0.2818 | 0.7576 | 0.3058 |
| final_test_calibrated_prob | 0.2616 | 0.7163 | 0.1752 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1714 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_test_predictions.csv`
- `test_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_thesis_stride8_qa/coral_w_0.10/dann_diagnostics.json`
