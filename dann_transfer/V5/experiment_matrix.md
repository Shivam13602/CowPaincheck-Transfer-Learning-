# V5 Experiment Matrix (S0–S10)

Concrete, runnable configuration for every V5 stage. All runs use the **new complete checkpoint set** `v2.9_20260502_181533` (9 folds × {combined, task1, task2}), the **enlarged thesis dataset**, and the **8-cow balanced split** from [`split_strategy.md`](split_strategy.md). Reuse the V3 trainers under `../V3/training_code/` (`weak_label_adapt_v3.py`, `dann_adapt_v3.py`); CDAN/MDD use the literature fork `../code/dann_adapt_v3_1.py`.

**Shared target args** (every weak/DANN run):

```
--manifest-csv   <enlarged completed_manifest.csv>
--sequence-root  <enlarged output root>
--dataset-version thesis_stride8_qa_v5
--label-column   video_health_status
--test-cow-ids   363,370,378,403,404,408,433,436
--val-cows-per-fold 4 --require-val-both-classes --cow-balanced-sampler
--threshold-min-specificity 0.50
--diag-bootstrap-samples 2000
--batch-size 32 --num-workers 20
```

**Checkpoint flags:** `--checkpoint-dir <.../v2.9_20260502_181533> --ckpt-kind task1 --init-fold <0..8>`. Where "per-fold init" is noted, set `--init-fold` to the fold index so each adaptation fold starts from its matched source fold.

**Reporting (every stage):** sequence / video / cow AUC, balanced accuracy, F1; Brier/ECE raw + calibrated; bootstrap CIs; per-cow table; primary spec-constrained threshold + `fixed_0.5`/`youden` alongside; per-fold inner val AUC. **Plus a condition-stratified breakdown** (per-`health_condition` mean `pain_prob` and AUC vs healthy) — see S10.

### Label tracks (no veterinary pain scores)

We have disease-condition labels only. Every stage is evaluated under **two label definitions**:

- **Track A — binary proxy (primary, comparable to V1–V4):** `--label-column video_health_status` (Healthy=0 / Unhealthy=1).
- **Track B — focused lameness-vs-healthy proxy (closest to pain):** positive = lameness family (`lameness`, `lame`, `lameness/stiffness`), negative = healthy (`healthy`, `healthy folder`); **drop** systemic/ambiguous conditions (`possible mastitis`, `possible metritis`, `fresh cows`, `sudden fall`) from the labeled set. Implemented as a **filtered manifest** (`completed_manifest_lameness.csv`); the 8-cow split and folds are unchanged (cow membership is identical; only sequence labels/inclusion change). See [`split_strategy.md`](split_strategy.md) §7.

Run the core matrix once per track. Track A is the headline transfer result; Track B is the disease-specificity result.

---

## Stage gates (stop/branch rules)

| Gate | Condition | Action |
|------|-----------|--------|
| G1 | S1 source Task1 AUC acceptable on UCAPS held-out | proceed; else fix checkpoint staging |
| G2 | S2 zero-shot floor recorded | every later stage must beat this cow-level metric to claim transfer |
| G3 | S4 source Task1 retention pass (AUC ≥ max(0.55, init−0.03)) | interpret alignment; else reduce alignment weight |
| G4 | threshold not collapsed (test tn > 0, recall < 1.0) | report as classifier; else ranking-only |
| G5 | cow-level CI excludes 0.5 on 8-cow test | claim cow-level **disease-context** separation; else ranking-only |
| G6 | Track B (lameness-vs-healthy) AUC > Track A AUC | evidence the model keys on pain-relevant orthopedic condition, not generic illness |

---

## S0 — Data + split + QA

- **Enlarge** the dataset toward the "Rich" 4-videos/cow plan using `datasets/thesis_stride8_qa/create_thesis_stride8_sequences.py --videos-per-cow 4` (or a denser stride), keeping the same QA gates (detection ≥ 90%, fill ≤ 10%, mean conf ≥ 0.80). Preserve the frozen `baseline_10s_250`; report V5 separately.
- **Generate split:** `python make_v5_splits.py --manifest <enlarged manifest> --out splits/v5_split.json`.
- **QA tables:** rows per cow / class / condition / source; detection-rate, fill-rate, confidence histograms; multi-face diagnostics; duplicate-window and cow/session leakage checks; confirm test cows isolated.
- **Output:** `results/S0_data_qa/` (QA CSVs + figures), frozen `splits/v5_split.json`.

## S1 — Source sanity (gate G1)

- Evaluate the new v2.9 `task1` checkpoints on the UCAPS held-out animals (14, 17) to confirm pain discrimination survived the new training run.
- Tool: `../code/evaluate_test_set_v2.9_cli.py` (or `evaluate_test_set_v2.9_cli.py` from the v2.9 bundle) against `v2.9_20260502_181533`.
- **Report:** Task1 AUC/F1/balanced accuracy per fold + ensemble; calibration if logits available.

## S2 — Zero-shot re-baseline (gate G2)

- 9-fold **source ensemble** inference on the 8-cow test, no target training. Establishes the honest transfer floor on the V5 split.
- Tool: `evaluate_holstein_zero_shot_v2.9.py` over all 9 `task1` fold checkpoints; mean logit → `pain_prob`.
- **Report:** video/cow proxy AUC on the 8 test cows; per-cow probability table. Expect ~0.5 (history); this is the bar to beat.

## S3 — Weak-label heads, frozen CNN

Three losses, frozen CNN, class- + cow-balanced. Per-fold init from matched source fold.

| ID | `--task1-loss` | extra flags |
|----|----------------|-------------|
| S3-bce | `bce` | `--freeze-cnn --class-balanced` |
| S3-focal | `focal` | `--freeze-cnn --focal-gamma 2.0 --class-balanced` |
| S3-gce | `gce` | `--freeze-cnn --gce-q 0.7 --class-balanced` |
| S3-focal-ft (ablation) | `focal` | (no `--freeze-cnn`; documents overfit risk) |

```
python ../V3/training_code/weak_label_adapt_v3.py <shared target args> \
  --checkpoint-dir <.../v2.9_20260502_181533> --ckpt-kind task1 --init-fold 0 \
  --task1-loss focal --focal-gamma 2.0 --freeze-cnn --class-balanced \
  --num-epochs 80 --learning-rate 1e-4 --select-metric v3_composite \
  --out-dir results/S3_weak_focal
```

Expectation from history: focal best for cow balanced accuracy; BCE/GCE harsh on recall.

## S4 — Domain alignment, source-retention-gated (gate G3)

Init `task1`, source Task1 weight 1.0, target weak 0.0, retention floor 0.55 / margin 0.03.

| ID | alignment | key flags |
|----|-----------|-----------|
| S4-dann-0.10 | domain | `--alignment-loss domain --domain-weight 0.10` |
| S4-dann-0.25 | domain | `--alignment-loss domain --domain-weight 0.25` |
| S4-dann-0.50 | domain | `--alignment-loss domain --domain-weight 0.50` |
| S4-coral-0.01 | coral | `--alignment-loss coral --domain-weight 0.0 --coral-weight 0.01` |
| S4-coral-0.05 | coral | `--alignment-loss coral --domain-weight 0.0 --coral-weight 0.05` |
| S4-coral-0.10 | coral | `--alignment-loss coral --domain-weight 0.0 --coral-weight 0.10` |
| S4-domain-coral | domain_coral | `--alignment-loss domain_coral --domain-weight 0.10 --coral-weight 0.05` |
| S4-cdan (fork) | CDAN/MDD | run via `../code/dann_adapt_v3_1.py` (verify `--da-mode cdan`/`mdd` flags) |

```
python ../V3/training_code/dann_adapt_v3.py <shared target args> \
  --source-project-dir <ucaps_source> --source-sequence-dir <ucaps_source/sequence> \
  --checkpoint-dir <.../v2.9_20260502_181533> --ckpt-kind task1 --init-fold 0 \
  --alignment-loss coral --domain-weight 0.0 --coral-weight 0.05 \
  --source-task1-weight 1.0 --source-task2-weight 0.0 --target-weak-weight 0.0 \
  --source-task1-retention-floor 0.55 --source-task1-retention-margin 0.03 \
  --num-epochs 80 --learning-rate 1e-5 --select-metric v3_composite \
  --out-dir results/S4_coral_0.05
```

**CORAL warmup note (V4 fix):** V4 CORAL collapsed (seq AUC 0.199) on dense overlapping data. Run CORAL with a low weight (0.01–0.05) and verify the alignment term ramps after a warmup; if `dann_adapt_v3.py` applies CORAL from epoch 0, gate it behind a few warmup epochs (small code note) or prefer `domain_coral` with low coral weight. Always confirm retention pass before interpreting.

## S5 — SSL target adaptation

- Leakage-safe SimSiam on **fold-train cows only**: `../code/ssl_pretrain_holstein_v2.9.py` → one SSL checkpoint per fold under `results/S5_ssl/fold_{fold}/best_ssl_simsiam.pt`.
- Re-run S3 and S4 best configs with `--ssl-checkpoint-dir results/S5_ssl --ssl-checkpoint-pattern "fold_{fold}/best_ssl_simsiam.pt"`.

| ID | init | downstream |
|----|------|-----------|
| S5-W-ucaps | UCAPS only | S3-focal |
| S5-W-ssl | UCAPS + SimSiam | S3-focal |
| S5-D-ucaps | UCAPS only | S4 best |
| S5-D-ssl | UCAPS + SimSiam | S4 best |

Transductive SSL (all target cows) only as a clearly-labelled ablation.

## S6 — Few-shot / low-variance probes

- Export frozen embeddings (UCAPS, best S4/S5 encoder) for all sequences.
- Train low-variance heads on the same folds: **logistic regression** and **prototypical** classifier (cow-prototype distance). No backbone fine-tune.
- **Report** on the 8-cow test; compare against S3/S4 fine-tunes (probes often win on tiny target data).
- Tool: small sklearn script under `scripts/` (to add); reuses the embedding export path.

## S7 — Temporal density + pooling

- `--max-frames` sweep {32, 64, 96} on the best S3/S4 config (watch VRAM; lower batch for 96).
- Sliding-window inference aggregation (mean / trimmed-mean / max), chosen on **validation only**.
- Attention-based temporal pooling vs LSTM-final (literature-motivated, sec. 3 of the review) — config/flag depending on model support; document if a code change is needed.

```
python ../V3/training_code/weak_label_adapt_v3.py <shared target args> \
  --checkpoint-dir <ckpt> --ckpt-kind task1 --task1-loss focal --freeze-cnn \
  --max-frames 64 --out-dir results/S7_focal_mf64
```

## S8 — Ensembling + calibration + decision policy

- Ensemble fold checkpoints (mean logit); temperature scaling per fold → mean temperature on test (already in V3 reports).
- Choose the final non-collapsing operating point on validation; verify gate G4 on the 8-cow test.
- **Report** raw vs calibrated tables and the final decision threshold with confusion matrix.

## S9 — Stretch: region-based micro-expression

Targets eyebrow (orbital) tightening + muzzle tension explicitly (review sec. 4).

1. **Keypoints:** add a cow facial-keypoint step (e.g., YOLOv8-Pose, as in Zhang et al. 2025) to locate eye, muzzle, ear regions on the existing 224×224 face crops.
2. **Region crops:** export per-frame eye + muzzle (+ ear) patches.
3. **Model:** region encoder (shared CNN) + **temporal attention** pooling; **fuse** regions (do not rely on the eyebrow cue alone — reliability caveat, review sec. 2.3).
4. **Eval:** same 8-cow split, same metrics; compare against the best whole-face S3/S4/S7 model.

Scoped as stretch because it needs new landmark infrastructure not in the current pipeline. Low-cost precursor: region-biased spatial attention on the whole-face model if landmarks are available.

## S10 — Condition-stratified analysis + lameness-vs-healthy proxy (thesis headline)

We have no veterinary pain scores, so this stage replaces the old vet-label track. It extracts the maximum scientific value from the **disease-condition labels we do have** and is the headline deliverable.

**Condition distribution (current `thesis_stride8_qa` manifest):** healthy 234 · lameness family 159 (`lameness` 67 + `lame` 59 + `lameness/stiffness` 33) · possible mastitis 50 · sudden fall 87 (cow 349) · possible metritis 10 · fresh cows 9.

**S10a — Condition-stratified evaluation (applied to the best model from S3/S4/S7, both tracks):**
- For the held-out test cows, report **per-`health_condition` mean `pain_prob`** and **per-condition AUC vs healthy**.
- Hypothesis (literature-grounded): **lameness (orthopedic pain) separates best**; systemic illness (mastitis, metritis) weaker; `fresh cows` / `sudden fall` weakest.
- Output: a condition × metric table + per-condition probability plot. This turns "AUC 0.55 on Healthy/Unhealthy" into a specific, defensible statement about *which* conditions the transferred pain features detect.

**S10b — Focused lameness-vs-healthy proxy (Track B):**
- Build the filtered manifest (lameness family = 1, healthy = 0; drop mastitis/metritis/fresh/fall):
  ```bash
  # one-off filter (pseudocode): keep rows where health_condition in {healthy, healthy folder,
  # lameness, lame, lameness/stiffness}; map lameness-family -> Unhealthy, healthy -> Healthy.
  ```
- Re-run S2 (zero-shot), S3 (focal), S4 (best alignment), and S6 (probes) on the filtered manifest with the **same 8-cow split**.
- Compare Track B vs Track A (gate G6): if lameness-vs-healthy AUC exceeds generic Healthy/Unhealthy AUC, that is evidence the model keys on the **pain-relevant** condition rather than generic illness context.

**Claims this supports:** disease-context discrimination, and condition-specificity (lameness detectability). **Still not** a pain claim — lameness-vs-healthy is a condition proxy, not a pain score. A pain claim would require a future validated pain-scored set (out of scope; no vet/grimace labels available).

---

## Suggested run order and compute

1. S0 (CPU/GPU extraction) → S1 → S2 (fast, eval-only).
2. S3 (3 losses) + S4 (alignment sweep) as the core matrix — see [`scripts/run_v5_matrix_rorqual.sh`](scripts/run_v5_matrix_rorqual.sh).
3. S5 (SSL) on the best S3/S4 only.
4. S6/S7/S8 refinements on the best line.
5. S10 condition-stratified analysis + Track B (lameness-vs-healthy) on the best line; S9 stretch as a separate effort.

Mirror the V4 `results/<condition>/` layout (report.md + summary.json + fold CSV + predictions + video/cow aggregates + diagnostics) so `summarize` tooling and figures carry over.
