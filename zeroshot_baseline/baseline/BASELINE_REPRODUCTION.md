# Reproduce UCAPS v2.9 held-out metrics (animals 14 and 17)

The checkpoint folder for this Colab run is `v2.9_20260221_014705`. The saved benchmark JSONs in this upload bundle are still useful for sanity checks, but point your `--checkpoint_dir` at the `014705` checkpoint folder.

To **re-run** [`evaluate_test_set_v2.9_cli.py`](evaluate_test_set_v2.9_cli.py) locally you need:

1. **Checkpoints** — sync from Drive (see [`CHECKPOINTS_README.md`](CHECKPOINTS_README.md)) and run [`prepare_ucaps_checkpoints.py`](prepare_ucaps_checkpoints.py) if filenames differ.

2. **Project JSON and frames** — the evaluator expects a `facial_pain_project_v2`-style directory containing at least:
   - `train_val_test_splits_v2.json`
   - `sequence_label_mapping_v2.json`
   - UCAPS extracted sequences under the `sequence/` folder referenced by the mapping.

This repository snapshot may omit those large assets; copy them from your Drive project next to the checkpoints or pass `--project_dir` / `--sequence_dir` explicitly.

3. **Command (example)**

```bash
python evaluate_test_set_v2.9_cli.py ^
  --project_dir "path/to/facial_pain_project_v2" ^
  --sequence_dir "path/to/facial_pain_project_v2/sequence" ^
  --checkpoint_dir "path/to/checkpoints_v2.9/v2.9_20260221_014705" ^
  --ckpt_kind task2 ^
  --ensemble mean_logits
```

Compare printed metrics to [`test_eval_v2.9_task1_animals_14-17_ensemble.json`](v2.9_20260221_223056/test_eval_v2.9_task1_animals_14-17_ensemble.json) / Task2 ensemble JSON in the same folder.
