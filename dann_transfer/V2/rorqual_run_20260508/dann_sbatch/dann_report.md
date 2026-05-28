# Holstein/Jersey DANN Cow-Held-Out Adaptation

## Metric roles

- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. These are the only *pain-ground-truth* metrics in this report.
- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.
- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); thresholds are chosen on inner validation and applied to the final test without test tuning.

- Generated (UTC): `20260508T154623Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1` weak proxy)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `14` folds x `2` validation cows
- Source project: `/scratch/shiv136/project_data/ucaps_source`
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
| 0 | 0 | -inf | 0.0 | 0.7 | False | True | 374,402 | 21 | nan | 0.0 | 0.0 | 0.9048 | nan | 0.0 | 0.0 | nan | 0.0 | 0.0 | 1.0 | nan | 0.0 | nan | 0.5882 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 1 | 2 | -inf | 0.75 | 0.7 | False | True | 310,417 | 21 | 0.75 | 0.6364 | 0.7647 | 0.619 | 0.6442 | 0.7778 | 0.5385 | 1.0 | 1.0 | 1.0 | 0.2476 | 0.75 | 0.7647 | 1.0 | 0.5859 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 2 | 2 | -inf | 0.9333 | 0.7 | False | True | 415,436 | 16 | 0.9333 | 0.8333 | 0.9091 | 0.875 | 0.8667 | 0.8333 | 0.8333 | 1.0 | 1.0 | 1.0 | 0.05 | 0.9333 | 0.9091 | 1.0 | 0.5788 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 3 | 0 | -inf | 0.875 | 0.7 | False | True | 349,439 | 10 | 0.875 | 0.6667 | 0.75 | 0.6 | 0.6667 | 0.5 | 1.0 | 1.0 | 0.6667 | 1.0 | 1.3808 | 0.875 | 0.75 | 1.0 | 0.5882 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 4 | 0 | -inf | 1.0 | 0.7 | False | True | 421,428 | 13 | 1.0 | 0.7778 | 0.9565 | 0.6923 | 0.8182 | 1.0 | 0.6364 | 1.0 | 1.0 | 1.0 | 0.2932 | 1.0 | 0.9524 | 1.0 | 0.5859 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.6 | 0.4048 |
| 5 | 34 | -inf | 0.5 | 0.7 | False | True | 406,438 | 18 | 0.5 | 0.0 | 0.7143 | 0.4444 | 0.5 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 10.0 | 0.5 | 0.7143 | 1.0 | 0.6376 | 0.7778 | 0.8 | 0.6847 | 0.7241 | 0.84 | 0.45 | 0.4048 |
| 6 | 1 | -inf | 0.9 | 0.7 | False | True | 255,387 | 19 | 0.9 | 0.8 | 0.8182 | 0.7368 | 0.7222 | 0.6667 | 1.0 | 1.0 | 0.6667 | 1.0 | 0.3221 | 0.9 | 0.8571 | 1.0 | 0.5835 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |
| 7 | 3 | -inf | 0.7333 | 0.7 | False | True | 432,433 | 16 | 0.7333 | 0.625 | 0.7692 | 0.625 | 0.6667 | 0.8333 | 0.5 | 1.0 | 1.0 | 1.0 | 0.2818 | 0.7333 | 0.7692 | 1.0 | 0.6047 | 0.7857 | 0.7857 | 0.6753 | 0.7097 | 0.88 | 0.5 | 0.4048 |
| 8 | 59 | -inf | 0.6667 | 0.7 | False | True | 378,427 | 16 | 0.6667 | 0.0 | 0.5714 | 0.625 | 0.5 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.6667 | 0.5714 | 1.0 | 0.6306 | 0.7059 | 0.7857 | 0.6247 | 0.6923 | 0.72 | 0.35 | 0.4048 |
| 9 | 68 | -inf | 0.4545 | 0.7 | False | True | 370,394 | 21 | 0.4545 | 0.0 | 0.6875 | 0.381 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6667 | 10.0 | 0.4545 | 0.6875 | 0.0 | 0.6518 | 0.5957 | 0.7778 | 0.5447 | 0.6364 | 0.56 | 0.35 | 0.4048 |
| 10 | 66 | -inf | 0.625 | 0.7 | False | True | 323,352 | 10 | 0.625 | 0.0 | 0.8889 | 0.2 | 0.5 | 0.0 | 0.0 | 1.0 | 0.0 | 0.6667 | 10.0 | 0.625 | 0.8889 | 1.0 | 0.6353 | 0.5652 | 0.7857 | 0.5247 | 0.619 | 0.52 | 0.3 | 0.2381 |
| 11 | 0 | -inf | 0.96 | 0.7 | False | True | 354,405 | 15 | 0.96 | 0.8333 | 0.9474 | 0.7333 | 0.6 | 0.7143 | 1.0 | 1.0 | 0.6667 | 1.0 | 0.3379 | 0.96 | 0.9091 | 1.0 | 0.5882 | 0.7586 | 0.7857 | 0.6165 | 0.6667 | 0.88 | 0.6 | 0.4048 |
| 12 | 1 | -inf | 0.675 | 0.7 | False | True | 425,426 | 14 | 0.675 | 0.7059 | 0.8333 | 0.6429 | 0.675 | 0.8571 | 0.6 | 1.0 | 1.0 | 1.0 | 0.3463 | 0.675 | 0.8333 | 1.0 | 0.5859 | 0.7719 | 0.7719 | 0.6459 | 0.6875 | 0.88 | 0.5 | 0.4048 |
| 13 | 1 | -inf | 0.9667 | 0.7 | False | True | 355,446 | 11 | 0.9667 | 0.625 | 0.8333 | 0.4545 | 0.5 | 0.4545 | 1.0 | 1.0 | 0.6667 | 1.0 | 1.3236 | 0.9667 | 0.8333 | 1.0 | 0.5906 | 0.7719 | 0.7857 | 0.6459 | 0.6875 | 0.88 | 0.55 | 0.4048 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.2964 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5577 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.2964 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.3071 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5577 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.3071 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.5 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.5559 | 2 | 1 |
| 403 | 11 | 1.0 | 0.5289 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5782 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4553 | 0 | 0 |

## Diagnostics (pooled validation / final test)

| subset | brier | nll | ece |
| --- | --- | --- | --- |
| validation_raw_prob | 0.2351 | 0.6625 | 0.1393 |
| validation_calibrated_prob | 0.2066 | 0.588 | 0.0839 |
| final_test_raw_prob | 0.2647 | 0.728 | 0.1436 |
| final_test_calibrated_prob | 0.2535 | 0.7005 | 0.1155 |
### Cow-level bootstrap 95% CI (resample cows)

| subset | n_boot_ok | auc_median | auc_ci95_low | auc_ci95_high | bacc_median | bacc_ci95_low | bacc_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final_test_raw | 1743 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
| final_test_calibrated | 1714 | 0.5 | 0.0 | 1.0 | 0.75 | 0.5 | 1.0 |
Full reliability bins and PR-curve samples: see `dann_diagnostics.json`.


## Artifacts

- `split_json`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_splits.json`
- `summary_json`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_summary.json`
- `fold_summary_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_fold_summary.csv`
- `val_predictions_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_predictions.csv`
- `test_predictions_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_test_predictions.csv`
- `test_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_test_calibrated_cow_aggregates.csv`
- `diagnostics_json`: `/scratch/shiv136/project_data/runs/holstein_task1_dann_v2_run_sbatch/dann_diagnostics.json`
