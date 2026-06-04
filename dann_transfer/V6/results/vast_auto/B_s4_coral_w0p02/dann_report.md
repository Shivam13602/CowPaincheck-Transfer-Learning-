# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260601T053713Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Dataset version: `thesis_stride8_qa_549_interim`
- Final test cows: `["363", "370", "378", "403", "404", "408", "433", "436"]`
- Inner folds: `5` folds x `4` validation cows
- Source project: `/root/data/ucaps_source`
- Source fold: `0` | source train n: `309`
- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.
- Source Task1 retention gate: AUC >= max(`0.55`, initial_source_auc - `0.03`); absolute `0.7` is diagnostic.
- Source Task1 loss: `bce` | source SupCon weight: `0.0` | class-balanced: `False`
- Alignment loss: `coral` | domain weight: `0.0` | coral weight: `0.02` | domain lambda max: `1.0`
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Target weak BCE weight: `0.0` starting epoch `5`
- SSL checkpoint dir: `None`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a domain-adaptation diagnostic, not as validated pain detection.

## Validation Folds — UCAPS source Task1 vs Holstein proxy

_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; `source_task1_retention_pass` = primary source-retention gate; `checkpoint_selected_from_proxy_fallback` = no epoch passed retention so the best proxy epoch was used._

| fold | best_epoch | best_score | proxy_selection_score | source_task1_auc_init | source_task1_retention_floor | source_task1_retention_pass | source_task1_sanity_floor | source_task1_sanity_pass | checkpoint_selected_from_proxy_fallback | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc | source_task1_auc | source_task1_f1 | source_task1_f1_opt | source_task1_balanced_accuracy | source_task1_precision | source_task1_recall | source_task1_best_threshold | source_task2_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 0.9227 | 0.9227 | 0.5882 | 0.5582 | True | 0.7 | False | False | 354,402,415,428 | 76 | 0.8855 | 0.0417 | 0.7642 | 0.3947 | 0.5106 | 1.0 | 0.0213 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.8855 | 0.7642 | 1.0 | 0.6141 | 0.5 | 0.7857 | 0.4847 | 0.5789 | 0.44 | 0.35 | 0.4762 |
| 1 | 0 | 0.4373 | 0.4373 | 0.5882 | 0.5582 | True | 0.7 | False | False | 323,387,425,432 | 42 | 0.3413 | 0.0 | 0.5517 | 0.619 | 0.5 | 0.0 | 0.0 | 0.5 | 0.0 | 0.6667 | 0.933 | 0.3413 | 0.5517 | 0.5 | 0.5953 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |
| 2 | 0 | 0.4575 | 0.4575 | 0.5882 | 0.5582 | True | 0.7 | False | False | 394,406,426,438 | 43 | 0.35 | 0.4 | 0.6349 | 0.6512 | 0.625 | 1.0 | 0.25 | 0.5 | 0.0 | 0.6667 | 1.146 | 0.35 | 0.6349 | 0.5 | 0.5929 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |
| 3 | 46 | 0.872 | 0.872 | 0.5882 | 0.5582 | True | 0.7 | False | False | 310,405,421,439 | 70 | 0.8056 | 0.0 | 0.871 | 0.2286 | 0.5 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 10.0 | 0.8056 | 0.871 | 1.0 | 0.6141 | 0.4286 | 0.7857 | 0.4447 | 0.5294 | 0.36 | 0.25 | 0.4524 |
| 4 | 0 | 0.8232 | 0.8232 | 0.5882 | 0.5582 | True | 0.7 | False | False | 255,352,355,446 | 49 | 0.7118 | 0.4545 | 0.8193 | 0.5102 | 0.6471 | 1.0 | 0.2941 | 1.0 | 0.0 | 1.0 | 10.0 | 0.7118 | 0.8193 | 1.0 | 0.5953 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4636 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4636 | 0.6842 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4231 | 0.05 | 0.48 | 0.05 | 0.365 | 0.5577 | 0.6154 | 0.5 | 0.5577 | 0.365 | 0.5577 | 0.6154 | 0.5 | True | 13 | 0 | 6 | 0 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4636 | 0.625 | 0.5 | 0.0 | 0.0 | 0.0 | 0.3333 | 0.05 | 0.5455 | 0.05 | 0.3584 | 0.5333 | 0.4 | 0.6667 | 0.5333 | 0.4 | 0.5 | 1.0 | 0.0 | True | 5 | 0 | 3 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.2385 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |
| youden | 143 | 45 | 98 | 0.4636 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |
| specificity_constrained | 143 | 45 | 98 | 0.4636 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.3678 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4712 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4712 | 0.3158 | 0.5 | 0.48 | 0.3158 | 1.0 | 0.4103 | 0.05 | 0.48 | 0.05 | 0.4784 | 0.5577 | 0.6154 | 0.5 | 0.5577 | 0.4784 | 0.5577 | 0.6154 | 0.5 | True | 0 | 13 | 0 | 6 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4712 | 0.375 | 0.5 | 0.5455 | 0.375 | 1.0 | 0.3333 | 0.05 | 0.5455 | 0.05 | 0.4773 | 0.5333 | 0.4 | 0.6667 | 0.5333 | 0.5 | 0.5 | 1.0 | 0.0 | True | 0 | 5 | 0 | 3 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6853 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | True | 98 | 0 | 45 | 0 |
| f1 | 143 | 45 | 98 | 0.471 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |
| youden | 143 | 45 | 98 | 0.4712 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |
| specificity_constrained | 143 | 45 | 98 | 0.4712 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.4703 | 0.05 | 0.4787 | 0.05 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | 0.5586 | 0.4789 | 0.5586 | 0.8061 | 0.3111 | True | 0 | 98 | 0 | 45 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.3797 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.3554 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.3969 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.3627 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.3453 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3507 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.3803 | 3 | 1 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 7 | 378 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 0.0 | 0.3474 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 1.0 | 0.356 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 11 | 433 | Yashan Dhaliwal RAC Data 2025 | 12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 1.0 | 0.3672 | 11 | 1 |
| Yashan Dhaliwal RAC Data 2025::19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 17 | 378 | Yashan Dhaliwal RAC Data 2025 | 19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 0.0 | 0.3529 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 10 | 436 | Yashan Dhaliwal RAC Data 2025 | 20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 0.0 | 0.3554 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 16 | 370 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 0.0 | 0.3614 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 9 | 436 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 0.0 | 0.3571 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 5 | 370 | Yashan Dhaliwal RAC Data 2025 | 24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 0.0 | 0.4461 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 9 | 378 | Yashan Dhaliwal RAC Data 2025 | 28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 0.0 | 0.3733 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 1.0 | 0.3694 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 3 | 370 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 0.0 | 0.3929 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 11 | 436 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 0.0 | 0.35 | 0 | 0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.3453 | 6 | 1 |
| 370 | 24 | 0.0 | 0.383 | 0 | 0 |
| 378 | 33 | 0.0 | 0.3573 | 0 | 0 |
| 403 | 10 | 1.0 | 0.3596 | 10 | 1 |
| 404 | 5 | 0.0 | 0.3603 | 0 | 0 |
| 408 | 6 | 0.0 | 0.3684 | 0 | 0 |
| 433 | 29 | 1.0 | 0.3644 | 29 | 1 |
| 436 | 30 | 0.0 | 0.3539 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.3062 | 0.8161 | 0.2747 |
| validation_calibrated_prob | 0.2514 | 0.696 | 0.1566 |
| final_test_raw_prob | 0.2196 | 0.6312 | 0.0479 |
| final_test_calibrated_prob | 0.2425 | 0.6782 | 0.1633 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1960 | 0.3125 | 0.0 | 0.8333 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1952 | 0.3125 | 0.0 | 0.8333 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_splits.json`
- `summary_json`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_summary.json`
- `fold_summary_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_fold_summary.csv`
- `val_predictions_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_predictions.csv`
- `test_predictions_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_test_predictions.csv`
- `test_video_aggregates_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/root/runs/v6_auto/B_s4_coral_w0p02/dann_diagnostics.json`
