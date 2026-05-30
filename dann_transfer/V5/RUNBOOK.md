# V5 Runbook — Enlarge, Upload to Rorqual, Run

Operational steps to run the V5 matrix. Adjust `shiv136` / paths to your account.

> Evaluation protocol (confirmed): 8 test cows are **never** in train or val; the remaining cows go through **cow-held-out K-fold** (5 folds × 4 val cows, each CV cow validated exactly once); the final test score is the **ensemble** (mean logit across the 5 fold models). Cow 349 + surplus are train-only. The trainers consume the frozen `v5_split.json` via `--split-json`.

---

## Phase A — Run now on Rorqual (549-seq, interim)

Use the **frozen V4 549-sequence** dataset already on scratch. Local **v5 extraction keeps running**; this does not touch it.

1. **Upload** code + split + new checkpoints (skip dataset):

```powershell
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning/dann_transfer/V5/scripts/upload_to_rorqual_interim.ps1
```

2. **Submit** on Rorqual:

```bash
ssh shiv136@rorqual.alliancecan.ca
cd ~/Dann_transfer
sbatch dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh
```

- Data: `/scratch/shiv136/project_data/cow_face_sequences_thesis_stride8/output` (549 seq)
- Split: `v5_split.json` (already valid on 549 — dry-run verified)
- Checkpoints: `v2.9_20260502_181533` (27 files)
- Results: `/scratch/shiv136/project_data/runs/v5_thesis_8cow_549` (separate from enlarged run)

Quick test (weak-label only):

```bash
sbatch --export=ALL,RUN_S4_DANN=0,RUN_S4_CORAL=0 dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh
```

## Phase B — After local v5 extraction finishes (enlarged)

Order: **enlarge dataset → regenerate split → upload → submit** (full RUNBOOK below).

---

## Step 1 — Enlarge the dataset (4 videos/cow)

Run where the raw Truro/Yashan videos and the YOLO weights live (local GPU box or Rorqual). This does **not** overwrite the frozen 549-seq set; it writes a new output dir.

**GPU-optimized launcher (recommended):**

```powershell
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning/dann_transfer/V5/scripts/run_extraction_gpu_fast.ps1
```

This uses **sequential video decode** (one seek per window, not 240), **FP16 YOLO** (`--yolo-half`), **batch 48** (raise on 8GB+ with `$env:YOLO_BATCH_SIZE=96`), and `model.fuse()`.

**H100 on Rorqual/Narval (optional):** `sbatch dann_transfer/V5/scripts/sbatch_extraction_h100.sh` runs `run_extraction_gpu_h100.sh` with `--yolo-batch-size 256` (override `YOLO_BATCH_SIZE` if OOM).

Manual equivalent:

```powershell
cd "C:\Users\shivp\Downloads\Research\DATASET"
python CowPaincheck-Transfer-Learning/datasets/thesis_stride8_qa/create_thesis_stride8_sequences.py `
  --inventory cow_video_dataset_analysis.csv `
  --dataset-root . `
  --model yolo_cow_face/yolo26s.pt `
  --videos-per-cow 4 `
  --stride-seconds 8 `
  --seed 42 `
  --output Transferlearning/cow_face_sequences_thesis_stride8_v5/output `
  --overwrite `
  --device 0 `
  --yolo-batch-size 48 `
  --yolo-imgsz 640 `
  --yolo-half `
  --jpeg-quality 90
```

**Why GPU util looked like 0%:** the old code spent ~95% of time on **240 random `cap.set()` seeks per window** (CPU/disk). YOLO only ran in short bursts, so `nvidia-smi` often sampled between bursts. After the fix, throughput is much higher; on a **4GB GPU** you will still see bursty utilization (decode + writing 240 JPEGs per accepted window are CPU/disk bound). For sustained high GPU util, run on **Rorqual** with a larger GPU and `--yolo-batch-size 96` or `128` (if VRAM allows).

Then validate:

```powershell
python CowPaincheck-Transfer-Learning/datasets/thesis_stride8_qa/validate_thesis_metadata.py `
  Transferlearning/cow_face_sequences_thesis_stride8_v5/output
```

Expect ~1,100 candidate windows → ~600–750 QA-pass sequences (4 videos/cow). 6 cows will be partial (352, 363, 415, 427, 428, 446) — documented inventory limits.

**Local helpers (after extraction finishes):**

```powershell
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning/dann_transfer/V5/scripts/run_after_extraction.ps1
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning/dann_transfer/V5/scripts/upload_to_rorqual.ps1
```

## Step 2 — Regenerate the frozen split on the enlarged manifest

```powershell
cd CowPaincheck-Transfer-Learning/dann_transfer/V5
python make_v5_splits.py `
  --manifest ../../../Transferlearning/cow_face_sequences_thesis_stride8_v5/output/completed_manifest.csv `
  --out splits/v5_split.json
```

Review the printed summary: confirm 8 test cows (4H+4U), equal-size balanced folds, and cow 349 in train-only. Commit `splits/v5_split.json` + `splits/v5_split.audit.csv`.

### Optional — Track B (lameness-vs-healthy) split

Build a lameness-filtered manifest (healthy + lameness-family only; see `split_strategy.md` §7), then:

```powershell
python make_v5_splits.py --manifest <.../completed_manifest_lameness.csv> --out splits/v5_split_lameness.json
```

## Step 3 — Upload to Rorqual

Stage the enlarged sequences, manifest, split JSON, the complete checkpoint set, and the code. From a shell with `rsync`/`scp` access:

```bash
ACCT=shiv136
SCR=/scratch/${ACCT}/project_data

# 3a. Enlarged dataset (sequences + manifest)
rsync -av --progress \
  Transferlearning/cow_face_sequences_thesis_stride8_v5/output/ \
  ${ACCT}@rorqual.alliancecan.ca:${SCR}/cow_face_sequences_thesis_stride8_v5/output/

# 3b. Frozen split JSON
rsync -av CowPaincheck-Transfer-Learning/dann_transfer/V5/splits/ \
  ${ACCT}@rorqual.alliancecan.ca:${SCR}/v5/splits/

# 3c. Complete v2.9 checkpoint set (27 files)
rsync -av "remote_upload_ucaps/Transferlearning/v2.9/v2.9_20260502_181533-20260528T182044Z-3-001/v2.9_20260502_181533/" \
  ${ACCT}@rorqual.alliancecan.ca:${SCR}/v2.9_20260502_181533/

# 3d. Code (V3 trainers with the new --split-json loader + V5 scripts)
rsync -av --exclude '__pycache__' \
  CowPaincheck-Transfer-Learning/dann_transfer/ \
  ${ACCT}@rorqual.alliancecan.ca:~/Dann_transfer/dann_transfer/
```

UCAPS source (`ucaps_source/` with `sequence/`) for the DANN/CORAL arms should already be on scratch from V3/V4; if not, upload it to `${SCR}/ucaps_source`.

## Step 4 — Submit the job

```bash
ssh ${ACCT}@rorqual.alliancecan.ca
cd ~/Dann_transfer

sbatch --export=ALL,\
TARGET_MANIFEST=${SCR}/cow_face_sequences_thesis_stride8_v5/output/completed_manifest.csv,\
TARGET_ROOT=${SCR}/cow_face_sequences_thesis_stride8_v5/output,\
CKPT_DIR=${SCR}/v2.9_20260502_181533,\
SPLIT_JSON=${SCR}/v5/splits/v5_split.json,\
OUT_ROOT=${SCR}/runs/v5_thesis_8cow \
  dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual.sh
```

The script runs a dry-run split audit first (writes the plan without training), then S3 (bce/focal/gce, frozen CNN) + S4 (DANN dw {0.10,0.25,0.50} and CORAL {0.01,0.05,0.10}). Toggle stages with `RUN_S3_WEAK`, `RUN_S4_DANN`, `RUN_S4_CORAL`.

**H100 throughput (defaults in `scripts/h100_env.sh`):** `NUM_WORKERS=24`, `WEAK_BATCH_SIZE=128`, `DANN_BATCH_SIZE=48`, `DATALOADER_PREFETCH=4`, persistent workers, TF32 + AMP in the V3 trainers. On CUDA OOM:

```bash
sbatch --export=ALL,WEAK_BATCH_SIZE=64,DANN_BATCH_SIZE=24,NUM_WORKERS=16 ...
```

**Narval:** same matrix — `sbatch dann_transfer/V5/scripts/sbatch_v5_matrix_narval.sh` (account `def-sureshra`; adjust if your allocation differs).

## Step 5 — Verify and pull results

```bash
# Confirm the split loaded as intended (look for the [split-json] line and train_only cows)
grep -m1 "\[split-json\]" ${SCR}/runs/v5_thesis_8cow/logs/*.out

# After completion, sync reports back
rsync -av ${ACCT}@rorqual.alliancecan.ca:${SCR}/runs/v5_thesis_8cow/ \
  CowPaincheck-Transfer-Learning/dann_transfer/V5/results/
```

Each condition writes the V4-style bundle (report.md + summary.json + fold CSV + predictions + video/cow aggregates + diagnostics) under `results/<condition>/`.

## Sanity checklist before `sbatch`

**Phase A (549 interim):** `v5_split.json` on scratch; 549 seq under `thesis_stride8`; new ckpts uploaded; `ucaps_source` for S4.

**Phase B (enlarged):**

- [ ] `v5_split.json` regenerated on the **enlarged** manifest (not the 549-seq one).
- [ ] `[split-json]` preflight shows 8 test cows, 5×4 folds, 349 train-only.
- [ ] 27 checkpoint files present under `CKPT_DIR` (9 × task1/task2/combined).
- [ ] `ucaps_source` present for S4 (DANN/CORAL) — S3 weak-label does not need it.
- [ ] Zero-shot (S2) run separately via `evaluate_holstein_zero_shot_v2.9.py` over the 9 task1 folds to record the floor.

> I cannot upload to or submit on Rorqual for you (no cluster access). Run Steps 1–5 from your machine/account; everything above is prepared and the trainers now honor `--split-json`.
