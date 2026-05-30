# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260530T140700Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Dataset version: `thesis_stride8_qa_549_interim`
- Final test cows: `["363", "370", "378", "403", "404", "408", "433", "436"]`
- Inner folds: `5` folds x `4` validation cows
- Source project: `/scratch/shiv136/project_data/ucaps_source`
- Source fold: `0` | source train n: `309`
- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.
- Source Task1 retention gate: AUC >= max(`0.55`, initial_source_auc - `0.03`); absolute `0.7` is diagnostic.
- Source Task1 loss: `bce` | source SupCon weight: `0.0` | class-balanced: `False`
- Alignment loss: `coral` | domain weight: `0.0` | coral weight: `0.01` | domain lambda max: `1.0`
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Target weak BCE weight: `0.0` starting epoch `5`
- SSL checkpoint dir: `None`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a domain-adaptation diagnostic, not as validated pain detection.

## Validation Folds — UCAPS source Task1 vs Holstein proxy

_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; `source_task1_retention_pass` = primary source-retention gate; `checkpoint_selected_from_proxy_fallback` = no epoch passed retention so the best proxy epoch was used._

| fold | best_epoch | best_score | proxy_selection_score | source_task1_auc_init | source_task1_retention_floor | source_task1_retention_pass | source_task1_sanity_floor | source_task1_sanity_pass | checkpoint_selected_from_proxy_fallback | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc | source_task1_auc | source_task1_f1 | source_task1_f1_opt | source_task1_balanced_accuracy | source_task1_precision | source_task1_recall | source_task1_best_threshold | source_task2_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 29 | 0.9496 | 0.9496 | 0.5882 | 0.5582 | True | 0.7 | False | False | 354,402,415,428 | 76 | 0.931 | 0.12 | 0.8545 | 0.4211 | 0.5319 | 1.0 | 0.0638 | 1.0 | 0.0 | 0.8 | 10.0 | 0.931 | 0.7642 | 1.0 | 0.5929 | 0.7636 | 0.7857 | 0.6553 | 0.7 | 0.84 | 0.4 | 0.4524 |
| 1 | 2 | 0.4918 | 0.4918 | 0.5882 | 0.5582 | True | 0.7 | False | False | 323,387,425,432 | 42 | 0.4519 | 0.0 | 0.5517 | 0.619 | 0.5 | 0.0 | 0.0 | 0.5 | 0.0 | 0.6667 | 0.8185 | 0.4519 | 0.5517 | 0.5 | 0.5812 | 0.7636 | 0.7857 | 0.6553 | 0.7 | 0.84 | 0.45 | 0.4524 |
| 2 | 0 | 0.434 | 0.434 | 0.5882 | 0.5582 | True | 0.7 | False | False | 394,406,426,438 | 43 | 0.2978 | 0.3846 | 0.6349 | 0.6279 | 0.6033 | 0.8333 | 0.25 | 0.5 | 0.0 | 0.6667 | 1.4957 | 0.2978 | 0.6349 | 0.5 | 0.5812 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |
| 3 | 50 | 0.7881 | 0.7881 | 0.5882 | 0.5582 | True | 0.7 | False | False | 310,405,421,439 | 70 | 0.6829 | 0.4932 | 0.876 | 0.4714 | 0.6354 | 0.9474 | 0.3333 | 1.0 | 0.6667 | 0.8 | 10.0 | 0.6829 | 0.871 | 1.0 | 0.6118 | 0.7636 | 0.7857 | 0.6553 | 0.7 | 0.84 | 0.4 | 0.4524 |
| 4 | 0 | 0.8341 | 0.8341 | 0.5882 | 0.5582 | True | 0.7 | False | False | 255,352,355,446 | 49 | 0.7294 | 0.6667 | 0.8193 | 0.6531 | 0.75 | 1.0 | 0.5 | 1.0 | 0.6667 | 1.0 | 0.6045 | 0.7294 | 0.8193 | 1.0 | 0.5812 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4622 | 0.6434 | 0.4934 | 0.1356 | 0.2857 | 0.0889 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.3983 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.3983 | 0.5908 | 0.5816 | 0.6 | True | 88 | 10 | 41 | 4 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4622 | 0.5263 | 0.3846 | 0.0 | 0.0 | 0.0 | 0.4487 | 0.05 | 0.48 | 0.05 | 0.4136 | 0.641 | 0.6154 | 0.6667 | 0.641 | 0.4136 | 0.641 | 0.6154 | 0.6667 | True | 10 | 3 | 6 | 0 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4622 | 0.625 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.05 | 0.5455 | 0.05 | 0.4215 | 0.5667 | 0.8 | 0.3333 | 0.5667 | 0.4215 | 0.5667 | 0.8 | 0.3333 | True | 5 | 0 | 3 | 0 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6573 | 0.4916 | 0.0755 | 0.25 | 0.0444 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.3983 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.3983 | 0.5908 | 0.5816 | 0.6 | True | 92 | 6 | 43 | 2 |
| f1 | 143 | 45 | 98 | 0.3567 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.3983 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.3983 | 0.5908 | 0.5816 | 0.6 | True | 0 | 98 | 0 | 45 |
| youden | 143 | 45 | 98 | 0.4622 | 0.6434 | 0.4934 | 0.1356 | 0.2857 | 0.0889 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.3983 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.3983 | 0.5908 | 0.5816 | 0.6 | True | 88 | 10 | 41 | 4 |
| specificity_constrained | 143 | 45 | 98 | 0.4622 | 0.6434 | 0.4934 | 0.1356 | 0.2857 | 0.0889 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.3983 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.3983 | 0.5908 | 0.5816 | 0.6 | True | 88 | 10 | 41 | 4 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4854 | 0.6154 | 0.491 | 0.2029 | 0.2917 | 0.1556 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.4775 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.4775 | 0.5908 | 0.5816 | 0.6 | True | 81 | 17 | 38 | 7 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4854 | 0.4737 | 0.3462 | 0.0 | 0.0 | 0.0 | 0.4487 | 0.05 | 0.48 | 0.05 | 0.4809 | 0.641 | 0.6154 | 0.6667 | 0.641 | 0.4809 | 0.641 | 0.6154 | 0.6667 | True | 9 | 4 | 6 | 0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4854 | 0.5 | 0.4 | 0.0 | 0.0 | 0.0 | 0.4 | 0.05 | 0.5455 | 0.05 | 0.4827 | 0.5667 | 0.8 | 0.3333 | 0.5667 | 0.4827 | 0.5667 | 0.8 | 0.3333 | True | 4 | 1 | 3 | 0 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6573 | 0.4916 | 0.0755 | 0.25 | 0.0444 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.4775 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.4775 | 0.5908 | 0.5816 | 0.6 | True | 92 | 6 | 43 | 2 |
| f1 | 143 | 45 | 98 | 0.4853 | 0.6154 | 0.491 | 0.2029 | 0.2917 | 0.1556 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.4775 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.4775 | 0.5908 | 0.5816 | 0.6 | True | 81 | 17 | 38 | 7 |
| youden | 143 | 45 | 98 | 0.4854 | 0.6154 | 0.491 | 0.2029 | 0.2917 | 0.1556 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.4775 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.4775 | 0.5908 | 0.5816 | 0.6 | True | 81 | 17 | 38 | 7 |
| specificity_constrained | 143 | 45 | 98 | 0.4854 | 0.6154 | 0.491 | 0.2029 | 0.2917 | 0.1556 | 0.593 | 0.05 | 0.4787 | 0.05 | 0.4775 | 0.5908 | 0.5816 | 0.6 | 0.5908 | 0.4775 | 0.5908 | 0.5816 | 0.6 | True | 81 | 17 | 38 | 7 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.454 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.4089 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.4725 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.4085 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.3829 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3848 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.4337 | 3 | 1 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 7 | 378 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 0.0 | 0.3851 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 1.0 | 0.4261 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 11 | 433 | Yashan Dhaliwal RAC Data 2025 | 12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 1.0 | 0.4266 | 11 | 1 |
| Yashan Dhaliwal RAC Data 2025::19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 17 | 378 | Yashan Dhaliwal RAC Data 2025 | 19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 0.0 | 0.3891 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 10 | 436 | Yashan Dhaliwal RAC Data 2025 | 20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 0.0 | 0.3917 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 16 | 370 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 0.0 | 0.4008 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 9 | 436 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 0.0 | 0.3882 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 5 | 370 | Yashan Dhaliwal RAC Data 2025 | 24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 0.0 | 0.5631 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 9 | 378 | Yashan Dhaliwal RAC Data 2025 | 28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 0.0 | 0.427 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 1.0 | 0.4184 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 3 | 370 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 0.0 | 0.5025 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 11 | 436 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 0.0 | 0.3794 | 0 | 0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.3829 | 6 | 1 |
| 370 | 24 | 0.0 | 0.4473 | 0 | 0 |
| 378 | 33 | 0.0 | 0.3986 | 0 | 0 |
| 403 | 10 | 1.0 | 0.3994 | 10 | 1 |
| 404 | 5 | 0.0 | 0.4179 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4191 | 0 | 0 |
| 433 | 29 | 1.0 | 0.4239 | 29 | 1 |
| 436 | 30 | 0.0 | 0.3861 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2547 | 0.7026 | 0.1798 |
| validation_calibrated_prob | 0.2449 | 0.6825 | 0.1526 |
| final_test_raw_prob | 0.225 | 0.6423 | 0.0955 |
| final_test_calibrated_prob | 0.2427 | 0.6786 | 0.1654 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1960 | 0.4 | 0.0 | 0.9333 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1952 | 0.375 | 0.0 | 1.0 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_test_predictions.csv`
- `test_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_coral_w_0.01/dann_diagnostics.json`
