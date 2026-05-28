# What we are doing (plain English)

## The idea

We already have a **deep learning model (UCAPS v2.9)** trained on **beef cattle** during **castration pain**. It predicts **pain vs no pain** and a **3-class pain moment** from **face video**.

Your **new data** are **Holstein/Jersey dairy cows** filmed in normal farm settings. Labels come from **health folders and disease context** (lameness, mastitis, etc.), **not** from a pain scale at the exact moment of filming. So we call those labels **weak proxies**: they say “this cow was in a painful *condition*,” not “this face proves pain right now.”

**Experiment 1** asks a modest question:

> If we run the **already-trained** UCAPS model on our dairy face clips **without retraining** (“zero-shot”), do scores **line up at all** with healthy vs unhealthy proxies and with different conditions?

That tells us whether **transfer** is worth more work (vet scoring, fine-tuning). It does **not** by itself prove the model “detects pain” on dairy cows.

---

## What we already built (sequences)

- From raw videos we built **250** clips, each **10 seconds**, **240 frames**, face crops from a **YOLO** detector (largest face), saved under `cow_face_sequences_10s_250/sequences/...`.
- Every clip has a row in **`completed_manifest.csv`**: cow ID, **cow-level** health, **video-level** health (folder context), **condition** text, dataset source (Truro / Yashan / Cow 349).

---

## Label check (`completed_manifest.csv`)

Automated checks on the current manifest:

| Check | Result |
|-------|--------|
| Rows | **250** (matches target set size) |
| Duplicate `sequence_index` | **None** |
| Missing `cow_id` | **None** |
| `cow_health_status` | **Healthy** 105 · **Unhealthy** 145 |
| `video_health_status` | **Healthy** 123 · **Unhealthy** 127 |

**Why cow_health and video_health differ:** Some cows are labeled **unhealthy at cow level** (they had or later had disease) but a **specific clip** can sit in a **healthy** folder (e.g. **378**, **417**) — that is **intentional** and documented in the farm analysis; do not “fix” those rows without protocol change.

**`health_condition`** mixes folder text (“healthy folder”) and clinical hints (“lameness”, “possible mastitis”). Use it for **group summaries**, not as gold-standard diagnosis.

**Multiple clips per cow:** The selection strategy **revisited cows** on purpose (round-robin coverage). Cow IDs appear many times in the manifest; that is expected.

---

## What “training” means in this Drive bundle

- **`v2.9_training_classification.py`** is included because PyTorch needs the **same class definitions** that were used when the `.pt` files were saved. Loading checkpoints for **testing** is not the same as starting a **new** training run from zero.
- **Retraining / fine-tuning** Holstein data would be a **later** step (cow-held-out splits, noisy labels). This minimal **`baseline/`** folder is aimed at **running evaluations** on Colab once weights + manifest + frames are on Drive.

---

## Two kinds of “test”

1. **UCAPS baseline (optional):** Same model on the **original UCAPS test animals** (14, 17) to confirm your Colab run matches saved benchmark JSON — needs original UCAPS JSON + frames on Drive (`evaluate_test_set_v2.9_cli.py`).
2. **Holstein zero-shot (main):** Same nine-fold ensemble on the **250 dairy sequences** (`evaluate_holstein_zero_shot_v2.9.py`).

---

## Zero-shot run completed (Colab, `v2.9_20260222_144752`)

**Why this checkpoint run:** The nine-fold **Task2-best Stage2** weights under run tag **`v2.9_20260222_144752`** were the first complete flat bundle available on Drive (all folds `0..8`). They are **not** the same artifact as joint run `014705`; document that clearly in any write-up.

**Command (Colab):**

```text
python evaluate_holstein_zero_shot_v2.9.py
  --manifest-csv .../cow_face_sequences_10s_250/completed_manifest.csv
  --sequence-root .../cow_face_sequences_10s_250
  --checkpoint-dir .../checkpoints_v2.9/v2.9_20260222_144752
  --ckpt-kind task2
  --train-py .../baseline/v2.9_training_classification.py
  --out-dir .../baseline/holstein_zero_shot_outputs_144752
  --num-workers 8
  --device cuda
```

**Local synced outputs (this repo):**  
[`holstein_zero_shot_outputs_144752-20260502T051858Z-3-001/holstein_zero_shot_outputs_144752/`](holstein_zero_shot_outputs_144752-20260502T051858Z-3-001/holstein_zero_shot_outputs_144752)  
(outer folder name is from Google Drive for desktop when resolving duplicates.)

**Artifacts:** `holstein_zero_shot_predictions_*.csv`, `holstein_zero_shot_report.md`, `group_summary_*.csv`, `holstein_zero_shot_run_*.json`.

### Results summary (mean Task1 `pain_prob`)

From `holstein_zero_shot_report.md` (UTC stamp `20260502T051222Z`), **nine folds** `[0..8]`, **N = 250** sequences.

| Grouping | Level | Mean pain_prob (approx.) |
|----------|--------|----------------------------|
| Video context | Healthy (n=123) | **0.431** |
| Video context | Unhealthy (n=127) | **0.437** |
| Cow label | Healthy (n=105) | **0.429** |
| Cow label | Unhealthy (n=145) | **0.437** |

**Interpretation:** Mean scores for healthy- vs unhealthy-context clips sit **very close together** (~0.43 vs ~0.44). So **aggregate zero-shot transfer to these weak labels is weak**: the beef-trained pain detector does **not**, on its own, produce a strong separation of your proxy healthy vs unhealthy buckets.

Per **`health_condition`**, means spread a bit more (e.g. “possible metritis” higher on average than some “healthy folder” rows), but subgroups are **small** — use only as **hypothesis-generating**, not proof of pain.

---

## How to optimise and fine-tune toward better results

Treat everything below as a **roadmap**. Each step should use **cow-held-out** validation (or grouped CV); never random clip splits.

### A. Labels and targets (highest leverage)

1. **Veterinary pain scores** on 30–60 diverse clips (uncertainty + condition coverage). This is the only path to a **true** target-domain pain label and proper calibration.
2. **Reframe the training target** while weak labels remain: predict **`video_health_status`** or **`painful_condition_proxy`** explicitly, and phrase results as **disease-context classification**, not validated pain.

### B. Model adaptation (after zero-shot baseline is frozen)

3. **Temperature scaling / Platt scaling** on a small validation set (even weak labels) to fix **domain miscalibration** mid-range probabilities.
4. **Frozen CNN, train head + LSTM + attention** on weak labels; then **selective unfreeze** of last CNN block at low LR.
5. **Noisy-label robustness:** label smoothing, generalized cross-entropy, or small-loss filtering; **down-weight** ambiguous cows (378, 417) and **hold out** cow 349 unless analysing severe cases separately.

### C. Representation and domain shift

6. **Self-supervised pretraining** on Holstein face crops (contrastive or temporal consistency), then fine-tune heads or full model with small LR.
7. **Optional domain alignment** (CORAL / MMD / adversarial) on pooled embeddings **after** B shows you still have a large domain gap.

### D. Data and inputs

8. **Quality gates:** exclude or down-weight sequences with many **filled** frames or low mean YOLO confidence; rerun ablations with/without them.
9. **Temporal / MIL:** aggregate multiple clips per cow (mean, max, or attention pooling) so one “quiet” clip does not define the cow.

### E. Evaluation hygiene

10. **Bootstrap CIs** on cow-level metrics; report **per-cow** and **per-condition** tables, not only global means.
11. **Task2 as exploratory** only (acute vs residual confusion on source); prioritise **Task1** until vet labels exist.

---

## Reference numbers (sanity check)

In `v2.9_20260221_223056_reference/`:

- `test_eval_v2.9_task1_animals_14-17_ensemble.json`
- `test_eval_v2.9_task2_animals_14-17_ensemble.json`
- `config_v2.9.json`

Use these only to compare **after** you successfully run the UCAPS baseline with your full UCAPS project on Drive.
