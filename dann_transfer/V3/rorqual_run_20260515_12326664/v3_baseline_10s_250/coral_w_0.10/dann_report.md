# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260515T105143Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Dataset version: `baseline_10s_250`
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
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
| 0 | 2 | 0.9009 | 0.9009 | 0.5718 | 0.55 | True | 0.7 | False | False | 374,402,432,433 | 37 | 0.8704 | 0.6 | 0.6 | 0.7838 | 0.7259 | 0.6 | 0.6 | 1.0 | 1.0 | 1.0 | 0.1223 | 0.8704 | 0.6452 | 1.0 | 0.5906 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 1 | 78 | 0.7591 | 0.7591 | 0.5718 | 0.55 | True | 0.7 | False | False | 310,378,417,427 | 37 | 0.6199 | 0.2609 | 0.7059 | 0.5405 | 0.5512 | 0.75 | 0.1579 | 1.0 | 0.0 | 0.6667 | 2.5293 | 0.6199 | 0.6786 | 1.0 | 0.6235 | 0.68 | 0.7857 | 0.6047 | 0.68 | 0.68 | 0.4 | 0.4048 |
| 2 | 15 | 0.5004 | 0.5004 | 0.5718 | 0.55 | True | 0.7 | False | False | 370,394,415,436 | 37 | 0.4559 | 0.1905 | 0.6296 | 0.5405 | 0.5088 | 0.5 | 0.1176 | 0.5 | 0.0 | 0.6667 | 5.7623 | 0.4559 | 0.6296 | 0.5 | 0.6141 | 0.717 | 0.7719 | 0.6153 | 0.6786 | 0.76 | 0.4 | 0.4048 |
| 3 | 61 | 0.5026 | 0.5026 | 0.5718 | 0.55 | True | 0.7 | False | False | 323,349,352,439 | 20 | 0.4687 | 0.2667 | 0.75 | 0.45 | 0.5208 | 0.6667 | 0.1667 | 0.5 | 0.6667 | 0.6667 | 10.0 | 0.4687 | 0.75 | 0.5 | 0.6141 | 0.717 | 0.7857 | 0.6153 | 0.6786 | 0.76 | 0.4 | 0.4048 |
| 4 | 0 | 0.8762 | 0.8762 | 0.5718 | 0.55 | True | 0.7 | False | False | 354,405,421,428 | 28 | 0.8095 | 0.8372 | 0.8571 | 0.75 | 0.6429 | 0.8182 | 0.8571 | 1.0 | 0.8 | 1.0 | 0.231 | 0.8095 | 0.8889 | 1.0 | 0.5976 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.6 | 0.4048 |
| 5 | 17 | 0.5469 | 0.5469 | 0.5718 | 0.55 | True | 0.7 | False | False | 406,425,426,438 | 32 | 0.3875 | 0.1818 | 0.7692 | 0.4375 | 0.55 | 1.0 | 0.1 | 0.75 | 0.0 | 0.6667 | 10.0 | 0.3875 | 0.7692 | 0.75 | 0.6094 | 0.717 | 0.7719 | 0.6153 | 0.6786 | 0.76 | 0.4 | 0.4048 |
| 6 | 1 | 0.896 | 0.896 | 0.5718 | 0.55 | True | 0.7 | False | False | 255,355,387,446 | 30 | 0.8578 | 0.7143 | 0.8125 | 0.6 | 0.6 | 0.5556 | 1.0 | 1.0 | 0.6667 | 0.8 | 0.6819 | 0.8578 | 0.7742 | 1.0 | 0.5859 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.516 | 0.5172 | 0.512 | 0.4615 | 0.4615 | 0.4615 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4258 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5192 | 0.5745 | 0.6875 | 0.4615 | True | 9 | 7 | 7 | 6 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.516 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.45 | 0.8 | 0.45 | 0.45 | 0.75 | 0.5 | 1.0 | 0.75 | 0.45 | 0.75 | 0.5 | 1.0 | True | 1 | 1 | 1 | 1 |

## Raw Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4828 | 0.4808 | 0.4444 | 0.4286 | 0.4615 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4258 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5192 | 0.5745 | 0.6875 | 0.4615 | True | 8 | 8 | 7 | 6 |
| f1 | 29 | 13 | 16 | 0.3455 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4258 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5192 | 0.5745 | 0.6875 | 0.4615 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.516 | 0.5172 | 0.512 | 0.4615 | 0.4615 | 0.4615 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4258 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5192 | 0.5745 | 0.6875 | 0.4615 | True | 9 | 7 | 7 | 6 |
| specificity_constrained | 29 | 13 | 16 | 0.516 | 0.5172 | 0.512 | 0.4615 | 0.4615 | 0.4615 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4258 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5192 | 0.5745 | 0.6875 | 0.4615 | True | 9 | 7 | 7 | 6 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 13 | 16 | 0.4855 | 0.5172 | 0.5337 | 0.5625 | 0.4737 | 0.6923 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4822 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5046 | 0.5745 | 0.6875 | 0.4615 | True | 6 | 10 | 4 | 9 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 0.4855 | 0.75 | 0.75 | 0.8 | 0.6667 | 1.0 | 0.5 | 0.5 | 0.8 | 0.5 | 0.4928 | 0.75 | 0.5 | 1.0 | 0.75 | 0.4928 | 0.75 | 0.5 | 1.0 | True | 1 | 1 | 0 | 2 |

## Calibrated Final Test Threshold Policies

| threshold_policy | n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | f1_threshold | youden_threshold | youden_balanced_accuracy | youden_specificity | youden_sensitivity | balanced_accuracy_opt | specificity_constrained_threshold | specificity_constrained_balanced_accuracy | specificity_constrained_specificity | specificity_constrained_sensitivity | specificity_constraint_met | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 29 | 13 | 16 | 0.5 | 0.4828 | 0.4808 | 0.4444 | 0.4286 | 0.4615 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4822 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5046 | 0.5745 | 0.6875 | 0.4615 | True | 8 | 8 | 7 | 6 |
| f1 | 29 | 13 | 16 | 0.4372 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4822 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5046 | 0.5745 | 0.6875 | 0.4615 | True | 0 | 16 | 0 | 13 |
| youden | 29 | 13 | 16 | 0.4855 | 0.5172 | 0.5337 | 0.5625 | 0.4737 | 0.6923 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4822 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5046 | 0.5745 | 0.6875 | 0.4615 | True | 6 | 10 | 4 | 9 |
| specificity_constrained | 29 | 13 | 16 | 0.4855 | 0.5172 | 0.5337 | 0.5625 | 0.4737 | 0.6923 | 0.5769 | 0.05 | 0.619 | 0.05 | 0.4822 | 0.6875 | 0.375 | 1.0 | 0.6875 | 0.5046 | 0.5745 | 0.6875 | 0.4615 | True | 6 | 10 | 4 | 9 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.5382 | 2 | 1 |
| 403 | 11 | 1.0 | 0.5039 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5592 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4351 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.239 | 0.67 | 0.1297 |
| validation_calibrated_prob | 0.2145 | 0.6053 | 0.0815 |
| final_test_raw_prob | 0.2636 | 0.7257 | 0.1798 |
| final_test_calibrated_prob | 0.2517 | 0.6967 | 0.055 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/v3_baseline_10s_250/coral_w_0.10/dann_diagnostics.json`
