# V5 Split Strategy — 8-Cow Test + Balanced Cow-Held-Out K-Fold

This document specifies the V5 evaluation design and how to reproduce it with [`make_v5_splits.py`](make_v5_splits.py). It replaces the fragile 4-cow test used in V0–V4 with an **8-cow held-out test** plus a **class-balanced cow-held-out K-fold** in which every CV cow is validated **exactly once**.

> Labels remain the weak `video_health_status` proxy. Class balancing of cows uses the per-cow `cow_health_status` (a single label per animal); the trainer still optimizes the per-sequence `video_health_status` proxy.

---

## 1. Design rules

1. **8 fixed test cows** (4 Healthy + 4 Unhealthy), never in train or val. The legacy four (363, 403, 404, 408) stay inside the eight so V1–V4 numbers remain comparable; four more (2 Healthy + 2 Unhealthy) are added.
2. **Balanced CV rotation.** The remaining cows are split by class; we take an equal number of Healthy and Unhealthy cows into the CV rotation. Folds are **equal sized**, every fold has both classes, and **each CV cow is validated exactly once** (the standard K-fold property).
3. **Train-only pool.** Surplus cows (because the inventory has more Unhealthy than Healthy animals) and explicitly force-routed cows go to a **train-only pool**: present in every fold's training set, never validated. Cow **349** (87 sudden-fall sequences) is force-routed here so it can no longer inflate inner validation as it did in V4.
4. **Per-cow sequence cap.** A cap (default: median per-cow sequence count of the train pool) is recorded for documentation; runtime imbalance control is handled by the trainer's `--cow-balanced-sampler` (inverse cow-frequency weighting).
5. **Determinism.** Seed 42; the generated split JSON is committed and frozen for the V5 run.

---

## 2. Concrete instantiation on `thesis_stride8_qa` (current manifest)

Running the generator on `datasets/thesis_stride8_qa/output/completed_manifest.csv` (31 usable cows) yields:

- **Test (8 cows, 143 seqs):** `363, 370, 378, 403, 404, 408, 433, 436` — 4 Healthy (370, 404, 408, 436) + 4 Unhealthy (363, 378, 403, 433).
- **CV rotation (20 cows, 280 seqs):** 10 Healthy + 10 Unhealthy → **5 folds × 4 val cows (2H + 2U each)**, each cow validated once.

| Fold | Val Healthy | Val Unhealthy |
|------|-------------|---------------|
| 0 | 428, 402 | 415, 354 |
| 1 | 387, 432 | 323, 425 |
| 2 | 406, 426 | 394, 438 |
| 3 | 439, 405 | 421, 310 |
| 4 | 352, 355 | 446, 255 |

- **Train-only (3 cows, 126 seqs):** `349` (forced), `417`, `427` (surplus Unhealthy). Always in training, never validated.
- **Per-cow cap (documentation):** 12 sequences (median); 10 cows exceed it and are down-weighted by the cow-balanced sampler at train time.

Frozen artifacts: [`splits/v5_split.json`](splits/v5_split.json) and [`splits/v5_split.audit.csv`](splits/v5_split.audit.csv).

> **After dataset enlargement (S0):** re-run the generator on the enlarged manifest. With more Healthy-cow sequences/videos the CV rotation can grow (e.g., 6 folds), and the test cows can be re-confirmed for condition coverage. The 8 test cows and cow 349's train-only status should be held fixed across the enlargement for comparability.

---

## 3. Why this is academically defensible

- **Each animal validated once** → a complete, non-overlapping cow-held-out cross-validation over the rotation set, the standard for subject-independent evaluation in animal-pain DL (e.g., Feighelstein et al., 2026; Zhang et al., 2025).
- **8-cow balanced test** → cow-level bootstrap CIs become informative (the 4-cow V0–V4 test gave CIs spanning [0, 1]).
- **Class-balanced folds + both-classes-per-fold** → no degenerate folds; pooled thresholds are meaningful.
- **349 train-only** → removes the single biggest source of inner-validation inflation (V4 fold 7) while still using its data for representation learning.
- **No leakage** → test cows never appear in any fold's train or val; SSL (S5) also restricts to fold-train cows only.

---

## 4. Generator usage

```bash
cd dann_transfer/V5
python make_v5_splits.py \
  --manifest ../../datasets/thesis_stride8_qa/output/completed_manifest.csv \
  --out splits/v5_split.json
```

Useful flags:

- `--test-cow-ids 363,403,404,408,370,436,433,378` — choose the 8 test cows (default shown).
- `--train-only-cows 349` — force cows into the train-only pool (default 349).
- `--val-cows-per-fold 4` — must be even (half Healthy, half Unhealthy); controls fold size and therefore fold count.
- `--health-for-balance cow_health_status` — per-cow class used for balancing.
- `--max-seqs-per-cow 0` — 0 uses the median train-pool count as the documented cap.

The script prints the exact V3-trainer flags and writes a per-cow audit CSV (role + cap status).

---

## 5. Trainer integration

The V3 trainers (`weak_label_adapt_v3.py`, `dann_adapt_v3.py`) build folds internally from `--test-cow-ids` + `--val-cows-per-fold` with the same seeded class-balanced assignment, so passing the printed flags reproduces this design closely:

```
--test-cow-ids 363,370,378,403,404,408,433,436 \
--val-cows-per-fold 4 --require-val-both-classes --cow-balanced-sampler
```

To make the trainer consume **exactly** the frozen `v5_split.json` partition, the V3 trainers now accept **`--split-json`** (implemented in `weak_label_adapt_v3.py._build_split_plan()` and threaded through `dann_adapt_v3.py._make_split_args()`). When set:

- test cows and validation folds are read directly from the JSON (`test_cows`, `folds[*].val_cows`);
- any manifest cow that is neither a test cow nor in a validation fold becomes **train-only** (present in every fold's training set, never validated) — this is how cow **349**, 417, 427 are handled;
- `--test-cow-ids` / `--val-cows-per-fold` / `--test-cows` are ignored for partitioning (kept only for logging);
- the loader validates that every referenced cow exists, that test and val do not overlap, and that no cow is validated in more than one fold.

This is the path the V5 run scripts use (`--split-json` is in `COMMON_TARGET_ARGS`). The change is backward compatible: without `--split-json` the trainers behave exactly as in V3/V4. The frozen JSON is the **source of truth** for reporting and audit.

---

## 6. Reporting requirements (per stage)

For every stage S2–S9, report on the **8-cow test**:

- Sequence-, video-, and cow-level AUC, balanced accuracy, F1, precision/recall, confusion table.
- Calibration: Brier, ECE (raw and temperature-scaled).
- Bootstrap 95% CIs for cow-level AUC and balanced accuracy.
- Per-cow mean `pain_prob` table (8 rows) with correct/incorrect at the chosen threshold.
- Threshold policy: pooled validation, specificity ≥ 0.5 (primary), with `fixed_0.5` / `youden` reported alongside.
- Inner-fold val AUC per fold (to expose any residual fold inflation now that 349 is train-only).
- **Condition-stratified breakdown** of the test cows by `health_condition` (see §7 and experiment_matrix S10).

---

## 7. Track B — lameness-vs-healthy focused proxy (condition labels only)

We have no veterinary pain scores, so the closest-to-pain label we can build is **lameness (orthopedic pain) vs healthy**. This is a second label track layered on the *same cow split philosophy*, using `health_condition` instead of `video_health_status`.

### 7.1 Per-cow conditions (current manifest)

Unhealthy-cow conditions (sequence counts):

| Condition | Cows |
|-----------|------|
| **Lameness family** (positive for Track B) | 255 (lame/lameness 25), 323 (13), 354 (41), 363 (6), 394 (10), 403 (10), 415 (6), 421 (33), 425 (3), 427 (12) — **10 cows** |
| possible mastitis | 310 (21), 433 (29) |
| possible metritis | 438 (10) |
| fresh cows | 446 (9) |
| sudden fall | 349 (87) |
| **label mismatch** (cow=Unhealthy but all videos `healthy`) | **378 (33), 417 (27)** |

Healthy cows (negative for Track B): the 14 `cow_health_status=Healthy` cows.

### 7.2 Two data-quality actions (do in S0 before Track B)

1. **Resolve 378 and 417.** They are flagged Unhealthy at cow level but every QA-pass video is condition `healthy`. For Track B treat them by **video condition** (negative/healthy) or **exclude** them; document the decision. They should *not* be silent "unhealthy" positives.
2. **Note the binary-track test contamination.** In Track A these two cows inflate the "unhealthy" side with healthy-looking clips; report Track A with and without them as a sensitivity check.

### 7.3 Track B split (cow-consistent, reuse the generator)

Track B keeps the same cow-held-out discipline but its positive class is **lameness**, so the test/CV cows should be drawn from the 10 lameness cows + 14 healthy cows:

- Build a **filtered manifest** `completed_manifest_lameness.csv`: keep healthy cows (label 0) and lameness-family cows (label 1); drop mastitis/metritis/fresh/fall/mismatch sequences.
- Re-run the generator on it: `python make_v5_splits.py --manifest .../completed_manifest_lameness.csv --health-for-balance cow_health_status --out splits/v5_split_lameness.json` (lameness cows are `Unhealthy`, so the existing class logic works once the filter is applied).
- **Caveat on the shared 8-cow test:** under Track B the current test set yields only **2 clean lameness positives** (363, 403); 433 is mastitis (dropped) and 378 is a healthy-condition mismatch. So either (a) accept a healthy-heavy Track B test, or (b) generate a **Track-B-specific** test set with ~3 lameness + 3 healthy cows (e.g., positives from {354, 421, 363, 403, 427}, negatives from the healthy pool). Keep cow 349 train-only in both tracks.

Report Track A (binary) and Track B (lameness) side by side; gate G6 in `experiment_matrix.md` asks whether Track B AUC exceeds Track A — evidence the transferred pain features key on the pain-relevant orthopedic condition rather than generic illness.
