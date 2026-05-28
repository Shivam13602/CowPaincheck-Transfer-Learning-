# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260504T221531Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Source project: `/workspace/necessary_filesV1/ucaps_transfer_dann/ucaps_source`
- Source fold: `0` | source train n: `309`
- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.
- Source Task1 sanity floor: `0.7` AUC
- Source Task1 loss: `bce` | source SupCon weight: `0.0` | class-balanced: `False`
- Domain weight: `0.5` | domain lambda max: `1.0`
- Target weak BCE weight: `0.0` starting epoch `5`
- SSL checkpoint dir: `None`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a domain-adaptation diagnostic, not as validated pain detection.

## Validation Folds — UCAPS source Task1 vs Holstein proxy

_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; `checkpoint_selected_from_proxy_fallback` = no epoch passed the source AUC floor so the best proxy epoch was used._

| fold | best_epoch | best_score | proxy_selection_score | source_task1_sanity_floor | source_task1_sanity_pass | checkpoint_selected_from_proxy_fallback | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc | source_task1_auc | source_task1_f1 | source_task1_f1_opt | source_task1_balanced_accuracy | source_task1_precision | source_task1_recall | source_task1_best_threshold | source_task2_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | -inf | 0.9 | 0.7 | False | True | 374,402,432,433 | 37 | 0.9 | 0.6364 | 0.6667 | 0.7838 | 0.7574 | 0.5833 | 0.7 | 1.0 | 0.6667 | 1.0 | 0.1366 | 0.9 | 0.6923 | 1.0 | 0.5859 | 0.7636 | 0.7636 | 0.6553 | 0.7 | 0.84 | 0.5 | 0.4048 |
| 1 | 17 | -inf | 0.6023 | 0.7 | False | True | 310,378,417,427 | 37 | 0.6023 | 0.1 | 0.6786 | 0.5135 | 0.5263 | 1.0 | 0.0526 | 0.75 | 0.0 | 0.6667 | 4.0854 | 0.6023 | 0.6786 | 0.75 | 0.6259 | 0.7547 | 0.7857 | 0.6647 | 0.7143 | 0.8 | 0.4 | 0.4048 |
| 2 | 13 | -inf | 0.4588 | 0.7 | False | True | 370,394,415,436 | 37 | 0.4588 | 0.1818 | 0.6296 | 0.5135 | 0.4838 | 0.4 | 0.1176 | 0.5 | 0.6667 | 0.6667 | 7.0501 | 0.4588 | 0.6296 | 0.5 | 0.6235 | 0.7778 | 0.7857 | 0.6847 | 0.7241 | 0.84 | 0.45 | 0.4048 |
| 3 | 0 | -inf | 0.4583 | 0.7 | False | True | 323,349,352,439 | 20 | 0.4583 | 0.56 | 0.75 | 0.45 | 0.4167 | 0.5385 | 0.5833 | 0.5 | 0.6667 | 0.6667 | 4.4185 | 0.4583 | 0.75 | 0.5 | 0.5976 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 4 | 0 | -inf | 0.7755 | 0.7 | False | True | 354,405,421,428 | 28 | 0.7755 | 0.8372 | 0.875 | 0.75 | 0.6429 | 0.8182 | 0.8571 | 1.0 | 0.8 | 1.0 | 0.2697 | 0.7755 | 0.8889 | 1.0 | 0.5953 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 5 | 1 | -inf | 0.3833 | 0.7 | False | True | 406,425,426,438 | 32 | 0.3833 | 0.3871 | 0.7692 | 0.4062 | 0.4417 | 0.5455 | 0.3 | 0.5 | 0.5 | 0.6667 | 10.0 | 0.3833 | 0.7692 | 0.5 | 0.5882 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 6 | 1 | -inf | 0.8622 | 0.7 | False | True | 255,355,387,446 | 30 | 0.8622 | 0.7317 | 0.7692 | 0.6333 | 0.6333 | 0.5769 | 1.0 | 1.0 | 0.6667 | 1.0 | 0.8625 | 0.8622 | 0.8387 | 1.0 | 0.5835 | 0.7719 | 0.7719 | 0.6459 | 0.6875 | 0.88 | 0.5 | 0.4048 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.25 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5577 | 0.45 | 0.6842 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.25 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.2143 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5577 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.2143 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.6057 | 2 | 1 |
| 403 | 11 | 1.0 | 0.5531 | 11 | 1 |
| 404 | 10 | 0.0 | 0.6132 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4727 | 0 | 0 |

## Artifacts

- `split_json`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_splits.json`
- `summary_json`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_summary.json`
- `fold_summary_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_fold_summary.csv`
- `val_predictions_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_predictions.csv`
- `test_predictions_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_test_predictions.csv`
- `test_cow_aggregates_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_dann_vast_run/dann_test_calibrated_cow_aggregates.csv`
