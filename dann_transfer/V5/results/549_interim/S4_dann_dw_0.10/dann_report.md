# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260530T060041Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Dataset version: `thesis_stride8_qa_549_interim`
- Final test cows: `["363", "370", "378", "403", "404", "408", "433", "436"]`
- Inner folds: `5` folds x `4` validation cows
- Source project: `/scratch/shiv136/project_data/ucaps_source`
- Source fold: `0` | source train n: `309`
- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.
- Source Task1 retention gate: AUC >= max(`0.55`, initial_source_auc - `0.03`); absolute `0.7` is diagnostic.
- Source Task1 loss: `bce` | source SupCon weight: `0.0` | class-balanced: `False`
- Alignment loss: `domain` | domain weight: `0.1` | coral weight: `0.1` | domain lambda max: `1.0`
- Primary V3 threshold: pooled validation Youden/balanced-accuracy threshold with specificity >= `0.5` when feasible.
- Target weak BCE weight: `0.0` starting epoch `5`
- SSL checkpoint dir: `None`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a domain-adaptation diagnostic, not as validated pain detection.

## Validation Folds — UCAPS source Task1 vs Holstein proxy

_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; `source_task1_retention_pass` = primary source-retention gate; `checkpoint_selected_from_proxy_fallback` = no epoch passed retention so the best proxy epoch was used._

| fold | best_epoch | best_score | proxy_selection_score | source_task1_auc_init | source_task1_retention_floor | source_task1_retention_pass | source_task1_sanity_floor | source_task1_sanity_pass | checkpoint_selected_from_proxy_fallback | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc | source_task1_auc | source_task1_f1 | source_task1_f1_opt | source_task1_balanced_accuracy | source_task1_precision | source_task1_recall | source_task1_best_threshold | source_task2_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 0.9485 | 0.9485 | 0.5882 | 0.5582 | True | 0.7 | False | False | 354,402,415,428 | 76 | 0.934 | 0.1923 | 0.8791 | 0.4474 | 0.5532 | 1.0 | 0.1064 | 1.0 | 0.0 | 1.0 | 1.4395 | 0.934 | 0.7642 | 1.0 | 0.5882 | 0.6923 | 0.7719 | 0.5953 | 0.6667 | 0.72 | 0.4 | 0.4524 |
| 1 | 2 | 0.4951 | 0.4951 | 0.5882 | 0.5582 | True | 0.7 | False | False | 323,387,425,432 | 42 | 0.4591 | 0.0 | 0.5517 | 0.619 | 0.5 | 0.0 | 0.0 | 0.5 | 0.0 | 0.6667 | 0.8226 | 0.4591 | 0.5517 | 0.5 | 0.5812 | 0.7636 | 0.7857 | 0.6553 | 0.7 | 0.84 | 0.45 | 0.4524 |
| 2 | 0 | 0.434 | 0.434 | 0.5882 | 0.5582 | True | 0.7 | False | False | 394,406,426,438 | 43 | 0.2978 | 0.3704 | 0.6349 | 0.6047 | 0.5815 | 0.7143 | 0.25 | 0.5 | 0.0 | 0.6667 | 1.5293 | 0.2978 | 0.6349 | 0.5 | 0.5812 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |
| 3 | 50 | 0.818 | 0.818 | 0.5882 | 0.5582 | True | 0.7 | False | False | 310,405,421,439 | 70 | 0.7303 | 0.5333 | 0.871 | 0.5 | 0.6539 | 0.9524 | 0.3704 | 1.0 | 0.6667 | 0.8 | 10.0 | 0.7303 | 0.871 | 1.0 | 0.6188 | 0.717 | 0.7857 | 0.6153 | 0.6786 | 0.76 | 0.35 | 0.4524 |
| 4 | 1 | 0.8321 | 0.8321 | 0.5882 | 0.5582 | True | 0.7 | False | False | 255,352,355,446 | 49 | 0.7314 | 0.6667 | 0.8193 | 0.6531 | 0.75 | 1.0 | 0.5 | 1.0 | 0.6667 | 1.0 | 0.646 | 0.7314 | 0.8193 | 1.0 | 0.5788 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.45 | 0.4524 |

## Final 4-Cow Test Set — Sequence-Level (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4201 | 0.6294 | 0.5493 | 0.3614 | 0.3947 | 0.3333 | 0.5927 | 0.4 | 0.4912 | 0.4 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | True | 75 | 23 | 30 | 15 |

## Video-Level Final Test Metrics (mean prob per source clip)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4201 | 0.6316 | 0.641 | 0.5333 | 0.4444 | 0.6667 | 0.4487 | 0.05 | 0.48 | 0.05 | 0.4199 | 0.641 | 0.6154 | 0.6667 | 0.641 | 0.4199 | 0.641 | 0.6154 | 0.6667 | True | 8 | 5 | 2 | 4 |

## Cow-Level Final Test Metrics (mean prob per animal)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4201 | 0.375 | 0.3667 | 0.2857 | 0.25 | 0.3333 | 0.4 | 0.05 | 0.5455 | 0.05 | 0.4272 | 0.5667 | 0.8 | 0.3333 | 0.5667 | 0.4272 | 0.5667 | 0.8 | 0.3333 | True | 2 | 3 | 2 | 1 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6643 | 0.5027 | 0.1111 | 0.3333 | 0.0667 | 0.5927 | 0.4 | 0.4912 | 0.4 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | True | 92 | 6 | 42 | 3 |
| f1 | 143 | 45 | 98 | 0.386 | 0.4545 | 0.542 | 0.473 | 0.3398 | 0.7778 | 0.5927 | 0.4 | 0.4912 | 0.4 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | True | 30 | 68 | 10 | 35 |
| youden | 143 | 45 | 98 | 0.4201 | 0.6294 | 0.5493 | 0.3614 | 0.3947 | 0.3333 | 0.5927 | 0.4 | 0.4912 | 0.4 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | True | 75 | 23 | 30 | 15 |
| specificity_constrained | 143 | 45 | 98 | 0.4201 | 0.6294 | 0.5493 | 0.3614 | 0.3947 | 0.3333 | 0.5927 | 0.4 | 0.4912 | 0.4 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.3996 | 0.6019 | 0.5816 | 0.6222 | True | 75 | 23 | 30 | 15 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 45 | 98 | 0.4278 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.5927 | 0.05 | 0.4787 | 0.05 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | True | 0 | 98 | 0 | 45 |

## Calibrated Video-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 19 | 6 | 13 | 0.4278 | 0.3158 | 0.5 | 0.48 | 0.3158 | 1.0 | 0.4487 | 0.05 | 0.48 | 0.05 | 0.472 | 0.641 | 0.6154 | 0.6667 | 0.641 | 0.472 | 0.641 | 0.6154 | 0.6667 | True | 0 | 13 | 0 | 6 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 3 | 5 | 0.4278 | 0.375 | 0.5 | 0.5455 | 0.375 | 1.0 | 0.3333 | 0.05 | 0.5455 | 0.05 | 0.4746 | 0.5667 | 0.8 | 0.3333 | 0.5667 | 0.4746 | 0.5667 | 0.8 | 0.3333 | True | 0 | 5 | 0 | 3 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 143 | 45 | 98 | 0.5 | 0.6643 | 0.5027 | 0.1111 | 0.3333 | 0.0667 | 0.5927 | 0.05 | 0.4787 | 0.05 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | True | 92 | 6 | 42 | 3 |
| f1 | 143 | 45 | 98 | 0.4234 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.5927 | 0.05 | 0.4787 | 0.05 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | True | 0 | 98 | 0 | 45 |
| youden | 143 | 45 | 98 | 0.4278 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.5927 | 0.05 | 0.4787 | 0.05 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | True | 0 | 98 | 0 | 45 |
| specificity_constrained | 143 | 45 | 98 | 0.4278 | 0.3147 | 0.5 | 0.4787 | 0.3147 | 1.0 | 0.5927 | 0.05 | 0.4787 | 0.05 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | 0.6019 | 0.4648 | 0.6019 | 0.5816 | 0.6222 | True | 0 | 98 | 0 | 45 |

## Final Test Video Aggregates

| video_key | n_sequences | cow_id | dataset_root | relative_path | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Truro Cow Video Data::healthy cows after exercise/Sep 29/404.mov | 1 | 404 | Truro Cow Video Data | healthy cows after exercise/Sep 29/404.mov | 0.0 | 0.4558 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Oct 6/404.mov | 4 | 404 | Truro Cow Video Data | healthy cows before going out/Oct 6/404.mov | 0.0 | 0.4137 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 4/408.mov | 1 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 4/408.mov | 0.0 | 0.4743 | 0 | 0 |
| Truro Cow Video Data::healthy cows before going out/Sep 8/408.mov | 5 | 408 | Truro Cow Video Data | healthy cows before going out/Sep 8/408.mov | 0.0 | 0.4155 | 0 | 0 |
| Truro Cow Video Data::unhealthy cows after exercise/Nov 28/363-lameness.mov | 6 | 363 | Truro Cow Video Data | unhealthy cows after exercise/Nov 28/363-lameness.mov | 1.0 | 0.3842 | 6 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Oct 6/403-lameness.mov | 7 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Oct 6/403-lameness.mov | 1.0 | 0.3865 | 7 | 1 |
| Truro Cow Video Data::unhealthy cows after exercise/Sep 5/403- lameness.mov | 3 | 403 | Truro Cow Video Data | unhealthy cows after exercise/Sep 5/403- lameness.mov | 1.0 | 0.4381 | 3 | 1 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 7 | 378 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/378/Video 2025-03-10, 7 01 58 AM.mov | 0.0 | 0.3868 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 10 March 2025/During exercise/433/Video 2025-03-10, 6 56 17 AM.mov | 1.0 | 0.4322 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 11 | 433 | Yashan Dhaliwal RAC Data 2025 | 12 March 2025/After exercise/433/Video 2025-03-12, 8 15 27 AM.mov | 1.0 | 0.4304 | 11 | 1 |
| Yashan Dhaliwal RAC Data 2025::19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 17 | 378 | Yashan Dhaliwal RAC Data 2025 | 19 Feb 2025/After Exercise/378/Video 2025-02-19, 9 46 39 AM.mov | 0.0 | 0.392 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 10 | 436 | Yashan Dhaliwal RAC Data 2025 | 20 Feb 2025/Before Exercise/436/Video 2025-02-20, 8 32 57 AM.mov | 0.0 | 0.3942 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 16 | 370 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/370/Video 2025-02-21, 9 55 13 AM.mov | 0.0 | 0.4036 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 9 | 436 | Yashan Dhaliwal RAC Data 2025 | 21 Feb 2025/After Exercise/436/Video 2025-02-21, 9 58 30 AM.mov | 0.0 | 0.3895 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 5 | 370 | Yashan Dhaliwal RAC Data 2025 | 24 Feb 2025/During Exercise/370/Video 2025-02-24, 9 44 01 AM.mov | 0.0 | 0.5698 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 9 | 378 | Yashan Dhaliwal RAC Data 2025 | 28 Feb 2025/After Exercise/378/Video 2025-02-28, 9 54 22 AM.mov | 0.0 | 0.4325 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 9 | 433 | Yashan Dhaliwal RAC Data 2025 | 7 March 2025/During Exercise/433/Video 2025-03-07, 7 11 43 AM.mov | 1.0 | 0.4243 | 9 | 1 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 3 | 370 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/Before Exercise/370/Video 2025-03-19, 7 00 34 AM.mov | 0.0 | 0.5119 | 0 | 0 |
| Yashan Dhaliwal RAC Data 2025::March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 11 | 436 | Yashan Dhaliwal RAC Data 2025 | March 19 2025/During exercise/436/Video 2025-03-19, 7 39 56 AM.mov | 0.0 | 0.3806 | 0 | 0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 6 | 1.0 | 0.3842 | 6 | 1 |
| 370 | 24 | 0.0 | 0.4518 | 0 | 0 |
| 378 | 33 | 0.0 | 0.4019 | 0 | 0 |
| 403 | 10 | 1.0 | 0.402 | 10 | 1 |
| 404 | 5 | 0.0 | 0.4221 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4253 | 0 | 0 |
| 433 | 29 | 1.0 | 0.4291 | 29 | 1 |
| 436 | 30 | 0.0 | 0.3878 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2473 | 0.6874 | 0.1737 |
| validation_calibrated_prob | 0.2436 | 0.6798 | 0.1541 |
| final_test_raw_prob | 0.2255 | 0.6434 | 0.1099 |
| final_test_calibrated_prob | 0.2392 | 0.6715 | 0.155 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1960 | 0.4 | 0.0 | 0.9333 | 0.5 | 0.5 | 0.5 |
| final_test_calibrated | 1952 | 0.3125 | 0.0 | 0.875 | 0.5 | 0.5 | 0.5 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_test_predictions.csv`
- `test_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_test_video_aggregates.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_test_cow_aggregates.csv`
- `test_calibrated_video_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_test_calibrated_video_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549/S4_dann_dw_0.10/dann_diagnostics.json`
