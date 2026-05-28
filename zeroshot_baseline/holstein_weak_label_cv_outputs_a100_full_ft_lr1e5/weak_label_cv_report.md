# Holstein/Jersey Weak-Label Cow-Held-Out CV

- Generated (UTC): `20260502T184911Z`
- Label column: `video_health_status` (`Healthy=0`, `Unhealthy=1`)
- Final test cows: `["363", "403", "404", "408"]`
- Inner folds: `7` folds x `4` validation cows
- Initialization: `/workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752`
- Freeze CNN: `False`

These labels are weak disease-context proxies, not veterinary pain scores. Use this as a transfer-learning diagnostic, not as validated pain detection.

## Validation Folds

| fold | best_epoch | best_score | val_cows | val_n | val_auc | val_f1 | val_f1_opt | val_accuracy | val_balanced_accuracy | val_precision | val_recall | val_tn | val_fp | val_fn | val_tp | val_cow_auc | val_cow_f1 | val_cow_f1_opt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.8778 | 374,402,432,433 | 37 | 0.8778 | 0.5882 | 0.7273 | 0.6216 | 0.7407 | 0.4167 | 1.0 | 13 | 14 | 0 | 10 | 1.0 | 0.5 | 1.0 |
| 1 | 0 | 0.5409 | 310,378,417,427 | 37 | 0.5409 | 0.6522 | 0.6909 | 0.5676 | 0.5614 | 0.5556 | 0.7895 | 6 | 12 | 4 | 15 | 0.5 | 0.6667 | 0.6667 |
| 2 | 0 | 0.3824 | 370,394,415,436 | 37 | 0.3824 | 0.5714 | 0.6296 | 0.4324 | 0.4618 | 0.4375 | 0.8235 | 2 | 18 | 3 | 14 | 0.5 | 0.6667 | 0.6667 |
| 3 | 11 | 0.4167 | 323,349,352,439 | 20 | 0.4167 | 0.4348 | 0.75 | 0.35 | 0.3333 | 0.4545 | 0.4167 | 2 | 6 | 7 | 5 | 0.5 | 0.4 | 0.6667 |
| 4 | 14 | 0.8912 | 354,405,421,428 | 28 | 0.8912 | 0.8261 | 0.9231 | 0.7143 | 0.5238 | 0.76 | 0.9048 | 1 | 6 | 2 | 19 | 1.0 | 0.6667 | 1.0 |
| 5 | 11 | 0.4167 | 406,425,426,438 | 32 | 0.4167 | 0.3871 | 0.7692 | 0.4062 | 0.4417 | 0.5455 | 0.3 | 7 | 5 | 14 | 6 | 0.5 | 0.5 | 0.6667 |
| 6 | 1 | 0.9111 | 255,355,387,446 | 30 | 0.9111 | 0.6977 | 0.7857 | 0.5667 | 0.5667 | 0.5357 | 1.0 | 2 | 13 | 0 | 15 | 1.0 | 0.6667 | 1.0 |

## Final 4-Cow Test Set (Ensemble of Inner Fold Models)

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29.0 | 13.0 | 16.0 | 0.35 | 0.4483 | 0.5 | 0.619 | 0.4483 | 1.0 | 0.4567 | 0.05 | 0.619 | 0.0 | 16.0 | 0.0 | 13.0 |

## Cow-Level Final Test Metrics

| n | positives | negatives | threshold | accuracy | balanced_accuracy | f1 | precision | recall | auc | best_threshold | f1_opt | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 2.0 | 2.0 | 0.35 | 0.5 | 0.5 | 0.6667 | 0.5 | 1.0 | 0.5 | 0.6 | 0.8 | 0.0 | 2.0 | 0.0 | 2.0 |

## Final Test Cow Aggregates

| cow_id | n_sequences | target_mean | pain_prob | positives | target |
| --- | --- | --- | --- | --- | --- |
| 363 | 2 | 1.0 | 0.6677 | 2 | 1 |
| 403 | 11 | 1.0 | 0.6131 | 11 | 1 |
| 404 | 10 | 0.0 | 0.6798 | 0 | 0 |
| 408 | 6 | 0.0 | 0.555 | 0 | 0 |

## Artifacts

- `split_json`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_splits.json`
- `summary_json`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_summary.json`
- `fold_summary_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_fold_summary.csv`
- `val_predictions_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_predictions.csv`
- `test_predictions_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_test_predictions.csv`
- `test_cow_aggregates_csv`: `/workspace/ucaps_transfer/v2.9/holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/weak_label_cv_test_cow_aggregates.csv`
