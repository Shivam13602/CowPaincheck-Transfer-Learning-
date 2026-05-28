# UCAPS v2.9 checkpoints (source of truth and local mirror)

## Authoritative location (Google Drive)

Sync this folder to your machine before running zero-shot inference or baseline reproduction:

- `MyDrive/facial_pain_project_v2/checkpoints_v2.9/v2.9_20260221_014705`

The bundle contains **nine fold checkpoints** for the v2.9 9-fold CV (typically one `best.pt` per fold folder, or filenames equivalent to the training script’s `best_model_v2.9_task2_fold_*.pt`).

## Recommended local mirror (this repo)

Mirror the Drive folder to:

- `checkpoints_v2.9/v2.9_20260221_014705/`

Keep the same internal layout as Drive so paths stay reproducible across machines.

## Filename mapping

The evaluator ([`evaluate_test_set_v2.9_cli.py`](evaluate_test_set_v2.9_cli.py)) expects v2.9 names such as:

- `best_model_v2.9_task2_fold_0.pt` … `best_model_v2.9_task2_fold_8.pt` (for `--ckpt_kind task2`)

If Drive stores `best.pt` inside per-fold subfolders, run:

```bash
python prepare_ucaps_checkpoints.py --source "<path-to-synced-drive-folder>" --dest "<repo>/Transferlearning/v2.9/checkpoints_v2.9/v2.9_20260221_014705"
```

That script copies or symlinks into the standard names the evaluator and zero-shot script use.

## Verification

```bash
python verify_ucaps_checkpoint_bundle.py --checkpoint-dir "<repo>/Transferlearning/v2.9/checkpoints_v2.9/v2.9_20260221_014705" --ckpt-kind task2
```

## Full UCAPS baseline reproduction (animals 14 and 17)

Held-out evaluation also requires the original project assets (`train_val_test_splits_v2.json`, `sequence_label_mapping_v2.json`, and UCAPS frame sequences). See [`BASELINE_REPRODUCTION.md`](BASELINE_REPRODUCTION.md).

## Related automation in this folder

| Script | Purpose |
| --- | --- |
| [`prepare_ucaps_checkpoints.py`](prepare_ucaps_checkpoints.py) | Normalize Drive `best.pt` layout to `best_model_v2.9_task2_fold_*.pt` names. |
| [`verify_ucaps_checkpoint_bundle.py`](verify_ucaps_checkpoint_bundle.py) | Confirm nine folds exist and probe one checkpoint’s `cfg`. |
| [`holstein_v29_dataset.py`](holstein_v29_dataset.py) | Map `completed_manifest.csv` rows to sequence folders + UCAPS sequence dicts. |
| [`evaluate_holstein_zero_shot_v2.9.py`](evaluate_holstein_zero_shot_v2.9.py) | Nine-fold mean-logit ensemble on the 250 Holstein sequences; writes CSV + report. |
| [`evaluate_weak_label_proxies.py`](evaluate_weak_label_proxies.py) | AUC / summaries for weak healthy-unhealthy proxies vs Task1 probability. |
| [`select_calibration_candidates.py`](select_calibration_candidates.py) | Pick high-entropy clips with cow diversity for vet scoring. |
| [`weak_label_adapt_v2.9.py`](weak_label_adapt_v2.9.py) | Cow-held-out split skeleton (`--dry-run`) for future fine-tuning. |
| [`ucaps_v29_eval_loader.py`](ucaps_v29_eval_loader.py) | Loads `evaluate_test_set_v2.9_cli.py` (invalid dotted module name) safely. |
