# Task 1 pain / no-pain improvement — results snapshot

This note summarizes **synced** runs after refocusing DANN on **UCAPS Task 1 only** (no source Task2 loss, no target weak BCE by default). Holstein columns remain **weak `video_health_status` proxies**, not veterinary pain scores.

## Where the artifacts live

| Run | Location |
| --- | --- |
| Earlier SSL + DANN + Task2 auxiliary (`--source-task2-weight` > 0) | `vast_ai_results_20260502/holstein_dann_outputs/` |
| Task1-only DANN + SSL init (`--ckpt-kind task1`, `--source-task2-weight 0`) | `holstein_task1_dann_baseline/` |

## Holstein proxy final test (ensemble of inner folds)

| Experiment | Seq AUC | Seq balanced acc | Cow AUC | Cow balanced acc | Notes |
| --- | --- | --- | --- | --- | --- |
| SSL + DANN (May 2026, prior matrix) | 0.481 | 0.500 | 0.750 | 0.500 | Source Task1 AUC weak (~0.54); Task2 aux on. |
| Task1-only DANN baseline (synced) | 0.466 | 0.500 | 0.750 | 0.500 | Same degenerate threshold behavior (all positives at seq level in confusion table). |

Calibrated test metrics for the Task1-only run are in `holstein_task1_dann_baseline/dann_report.md` and `dann_summary.json` (`final_test.calibrated_*`).

## Source (UCAPS) sanity

On the Task1-only synced run, **per-fold source Task1 validation AUC stayed below the default sanity floor (0.7)** on all inner folds, so **checkpoints were effectively stuck at epoch 0** under the old selection rule.

**Code fix (current `dann_adapt_v2.9.py`):** if no epoch passes the floor, the fold now keeps the **best Holstein proxy-validation** checkpoint instead, sets `checkpoint_selected_from_proxy_fallback=true` in `dann_fold_summary.csv`, and prints a warning. Re-run on GPU to measure impact.

## Weak-label fine-tuning ablations

`weak_label_adapt_v2.9.py` supports `--task1-loss bce|focal|gce`, `--class-balanced`, and validation temperature scaling. Use **gce** / **focal** only as **proxy-label** diagnostics; do not mix them into the main UCAPS pain claim without strong source sanity.

## Remote execution

See `run_task1_vast.sh` for ordering: DANN dry run → Task1-only DANN → optional weak-proxy GCE ablation.
