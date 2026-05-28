# Drive upload — minimal (run tests only)

This folder holds **only** what you need on **Google Drive / Colab** to:

1. **Load** UCAPS v2.9 checkpoints and run **tests** (no new training from scratch in this bundle).
2. Run the **Holstein zero-shot** evaluation on your **`cow_face_sequences_10s_250`** manifest + frames.

**Do not upload from here:** image sequences (stay as `cow_face_sequences_10s_250` on Drive). **Weights** (`.pt`) upload separately into `checkpoints_v2.9/...` per `CHECKPOINTS_README.md`.

For **what we are doing**, **label checks**, **zero-shot results**, and **fine-tuning ideas**, read [`Holstein_UCAPS_Transfer_First_Experiment_README.md`](Holstein_UCAPS_Transfer_First_Experiment_README.md).

**Latest Holstein zero-shot outputs (local copy):**  
[`holstein_zero_shot_outputs_144752-20260502T051858Z-3-001/holstein_zero_shot_outputs_144752/`](holstein_zero_shot_outputs_144752-20260502T051858Z-3-001/holstein_zero_shot_outputs_144752) — checkpoints used: **`v2.9_20260222_144752`** (9-fold Task2 Stage2).

---

## Files to upload (this folder)

| File | Why |
|------|-----|
| `v2.9_training_classification.py` | Defines CNN+LSTM+attention **architecture** and dataset classes; required when **loading** checkpoints (even for inference only). |
| `evaluate_test_set_v2.9_cli.py` | **Test** source model on UCAPS held-out animals 14 & 17 (needs UCAPS JSON + frames on Drive — not the Holstein folder). |
| `evaluate_holstein_zero_shot_v2.9.py` | **Test** nine-fold ensemble on Holstein **completed_manifest.csv** + sequence folders. |
| `holstein_v29_dataset.py` | Maps manifest rows → sequence paths. |
| `ucaps_v29_eval_loader.py` | Loads evaluator module (filename contains `v2.9`). |
| `prepare_ucaps_checkpoints.py` | Renames Drive `best.pt` → `best_model_v2.9_task2_fold_*.pt` if needed. |
| `verify_ucaps_checkpoint_bundle.py` | Confirms nine folds present. |
| `CHECKPOINTS_README.md` | Drive checkpoint path + naming. |
| `BASELINE_REPRODUCTION.md` | How to rerun UCAPS baseline test when project JSON exists. |
| `checkpoints_v2.9/v2.9_20260221_014705/` | Example placeholder path on Drive; **zero-shot run used** `checkpoints_v2.9/v2.9_20260222_144752/` (see experiment README). |
| `v2.9_20260221_223056_reference/` | Only `config_v2.9.json` + two **ensemble JSON** targets to compare your baseline rerun. |

**Not included (still under [`../v2.9/`](../v2.9/)):** validation-only scripts, bootstrap/calibration extras, weak-label helpers — copy from `v2.9/` only if you need them later.

---

## Separate uploads (large)

| Asset | Location on Drive (example) |
|-------|-------------------------------|
| Nine fold weights | `MyDrive/.../checkpoints_v2.9/v2.9_20260222_144752/` (run used for Holstein zero-shot; adjust if you standardise on another run tag) |
| Holstein sequences | `.../cow_face_sequences_10s_250/` with `completed_manifest.csv` + `sequences/` |

---

## Quick commands (after Drive paths are set)

- **Holstein zero-shot:** see `CHECKPOINTS_README.md` and `evaluate_holstein_zero_shot_v2.9.py --help`.
- **UCAPS baseline test:** `BASELINE_REPRODUCTION.md` (requires original UCAPS splits + sequence tree).
