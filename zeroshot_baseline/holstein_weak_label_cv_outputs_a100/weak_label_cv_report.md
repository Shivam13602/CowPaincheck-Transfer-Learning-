# Holstein/Jersey Weak-Label Cow-Held-Out CV

- Generated (UTC): `20260502T172453Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Initialization: `/workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752`
- Freeze CNN: `True`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.7296 | 374,402,432,433 | 37 | 0.7296 | 0.0 | 0.4255 | 0.7297 | 0.5 | 0.0 | 0.0 | 27 | 0 | 10 | 0 | 1.0 | 0.0 | 0.4 |
| 1 | 0 | 0.5029 | 310,378,417,427 | 37 | 0.5029 | 0.1818 | 0.6786 | 0.5135 | 0.5249 | 0.6667 | 0.1053 | 17 | 1 | 17 | 2 | 0.5 | 0.0 | 0.6667 |
| 2 | 0 | 0.4647 | 370,394,415,436 | 37 | 0.4647 | 0.4 | 0.6296 | 0.5946 | 0.5721 | 0.625 | 0.2941 | 17 | 3 | 12 | 5 | 0.5 | 0.5 | 0.6667 |
| 3 | 0 | 0.5208 | 323,349,352,439 | 20 | 0.5208 | 0.4211 | 0.75 | 0.45 | 0.4792 | 0.5714 | 0.3333 | 5 | 3 | 8 | 4 | 0.5 | 0.6667 | 0.6667 |
| 4 | 4 | 0.8163 | 354,405,421,428 | 28 | 0.8163 | 0.6 | 0.8636 | 0.5714 | 0.7143 | 1.0 | 0.4286 | 7 | 0 | 12 | 9 | 1.0 | 1.0 | 1.0 |
| 5 | 8 | 0.4542 | 406,425,426,438 | 32 | 0.4542 | 0.1739 | 0.7692 | 0.4062 | 0.5083 | 0.6667 | 0.1 | 11 | 1 | 18 | 2 | 0.5 | 0.0 | 0.6667 |
| 6 | 4 | 0.7333 | 255,355,387,446 | 30 | 0.7333 | 0.4 | 0.6977 | 0.6 | 0.6 | 0.8 | 0.2667 | 14 | 1 | 11 | 4 | 1.0 | 0.6667 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.1571 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.5481 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.1571 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.45 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.4915 | 2 | 1 |
| 403 | 11 | 1.0 | 0.4741 | 11 | 1 |
| 404 | 10 | 0.0 | 0.5117 | 0 | 0 |
| 408 | 6 | 0.0 | 0.4361 | 0 | 0 |

## Artifacts

- `split_json`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_splits.json`
- `summary_json`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_summary.json`
- `fold_summary_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100/weak_label_cv_test_cow_aggregates.csv`
