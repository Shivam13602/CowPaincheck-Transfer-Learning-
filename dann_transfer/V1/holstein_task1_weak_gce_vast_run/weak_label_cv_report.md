# Holstein/Jersey Weak-Label Cow-Held-Out CV

## Metric roles

- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. There is no UCAPS pain ground truth on the target domain.
- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.

- Generated (UTC): `20260504T233131Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Initialization: `/workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752`
- Freeze CNN: `False`

- Task1 proxy loss: `gce` | class-balanced effective-number weighting: `False`
- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt | temperature | val_calibrated_auc | val_calibrated_f1_opt | val_calibrated_cow_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.6593 | 374,402,432,433 | 37 | 0.6593 | 0.0 | 0.4255 | 0.7297 | 0.5 | 0.0 | 0.0 | 27 | 0 | 10 | 0 | 1.0 | 0.0 | 0.4 | 0.2475 | 0.6593 | 0.4615 | 1.0 |
| 1 | 0 | 0.462 | 310,378,417,427 | 37 | 0.462 | 0.2963 | 0.6786 | 0.4865 | 0.4942 | 0.5 | 0.2105 | 14 | 4 | 15 | 4 | 0.25 | 0.0 | 0.6667 | 10.0 | 0.462 | 0.6786 | 0.25 |
| 2 | 8 | 0.4 | 370,394,415,436 | 37 | 0.4 | 0.3226 | 0.6296 | 0.4324 | 0.4221 | 0.3571 | 0.2941 | 11 | 9 | 12 | 5 | 0.5 | 0.5 | 0.6667 | 10.0 | 0.4 | 0.6296 | 0.5 |
| 3 | 8 | 0.4271 | 323,349,352,439 | 20 | 0.4271 | 0.5 | 0.75 | 0.5 | 0.5208 | 0.625 | 0.4167 | 5 | 3 | 7 | 5 | 0.5 | 0.5 | 0.6667 | 5.8023 | 0.4271 | 0.75 | 0.5 |
| 4 | 6 | 0.8571 | 354,405,421,428 | 28 | 0.8571 | 0.8947 | 0.8947 | 0.8571 | 0.9048 | 1.0 | 0.8095 | 7 | 0 | 4 | 17 | 1.0 | 1.0 | 1.0 | 0.1924 | 0.8571 | 0.9 | 1.0 |
| 5 | 5 | 0.575 | 406,425,426,438 | 32 | 0.575 | 0.1739 | 0.7692 | 0.4062 | 0.5083 | 0.6667 | 0.1 | 11 | 1 | 18 | 2 | 0.5 | 0.0 | 0.6667 | 10.0 | 0.575 | 0.7692 | 0.5 |
| 6 | 0 | 0.92 | 255,355,387,446 | 30 | 0.92 | 0.7692 | 0.8824 | 0.7 | 0.7 | 0.625 | 1.0 | 6 | 9 | 0 | 15 | 1.0 | 0.8 | 1.0 | 0.4354 | 0.92 | 0.8824 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.1857 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.4712 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.1857 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.75 | 0.45 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Calibrated Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.2071 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.4712 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Calibrated Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.2071 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.75 | 0.05 | 0.6667 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.5711 | 2 | 1 |
| 403 | 11 | 1.0 | 0.4711 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5557 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4441 | 0 | 0 |

## Artifacts

- `split_json`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_splits.json`
- `summary_json`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_summary.json`
- `fold_summary_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_test_cow_aggregates.csv`
- `test_calibrated_cow_aggregates_csv`: `/workspace/necessary_filesV1/Dann_transfer/holstein_task1_weak_gce_vast_run/weak_label_cv_test_calibrated_cow_aggregates.csv`
