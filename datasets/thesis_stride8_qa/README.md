# thesis_stride8_qa — Dense Thesis Sequence Dataset

**549 QA-pass sequences** · 10 s windows · **8 s stride** (2 s overlap) · Used in [**V4**](../../dann_transfer/V4/README.md)

![Dataset distribution](../../docs/figures/thesis_dataset_distribution.png)

**Image crops are not in this repo.** Manifests, statistics, and extraction scripts are included. See [`docs/DATA_ACCESS.md`](../../docs/DATA_ACCESS.md).

The frozen baseline [`baseline_10s_250`](../baseline_10s_250/) is **not modified**.

---

# Thesis Cow-Face Sequence Extraction (`thesis_stride8_qa`)

This folder contains the **masters-thesis sequence pipeline** for Holstein/Jersey transfer learning. It builds a new dataset version with:

- **Balanced source-video selection** (same target count per cow)
- **Session filters** for unhealthy cows
- **Sliding 10 s windows** with **2 s overlap** (8 s stride)
- **QA-filtered face crops** compatible with UCAPS v2.9 / V3 trainers

The frozen baseline dataset `Transferlearning/cow_face_sequences_10s_250/` is **not modified**.

---

## Folder layout

```text
cow_face_sequences_thesis_stride8/
├── README.md                              ← this file
├── create_thesis_stride8_sequences.py     ← extraction script
└── output/                                ← created when you run the script
    ├── README.md                          ← auto-generated run summary
    ├── selected_videos.csv
    ├── selected_videos_by_cow.csv
    ├── candidate_windows.csv
    ├── completed_manifest.csv             ← after full run
    ├── rejected_windows.csv
    ├── processing_statistics.json
    └── sequences/
        ├── healthy/
        │   └── cow_<id>_healthy/
        │       └── sequence_<dataset>_<hash>_t000s/
        │           ├── frame_0000.jpg … frame_0239.jpg
        │           ├── frames.csv
        │           └── metadata.json
        └── unhealthy/
            └── cow_<id>_unhealthy/
                └── ...
```

---

## Video selection rules

| Cow group | Count | Allowed source videos |
|-----------|------:|------------------------|
| **Healthy** | 15 | Any session: before / during / after exercise |
| **Unhealthy** | 17 | **During exercise** and **After exercise** only, plus **Cow 349 sudden-fall** clips |

Additional rules:

- Cow **409** excluded (health label unknown).
- Only videos with **duration ≥ 10 s** and readable metadata.
- **Same target video count per cow** (`--videos-per-cow`, default **3**).
- Cows with fewer eligible clips use all available and are flagged `partial` in `selected_videos_by_cow.csv`.

### Cow 349 (sudden fall)

The four videos under `Cow 349 - Unhealthy (sudden fall)/` are included as **`sudden_fall`** session clips for unhealthy cow **349**. They count toward the unhealthy pool and are prioritized when selecting videos for that cow.

---

## Windowing rules (UCAPS-aligned)

| Parameter | Value |
|-----------|------:|
| Window length | 10 s |
| Stride | 8 s |
| Overlap | 2 s |
| Stored FPS | 24 |
| Frames per sequence | 240 |
| Crop size | 224×224 |
| YOLO confidence | 0.60 |
| Crop padding | 0.08 |
| Face policy | Largest detected face; forward-fill missing frames |

### QA pass thresholds

| Gate | Threshold |
|------|-----------|
| Detection rate | ≥ 90% |
| Filled-frame rate | ≤ 10% |
| Mean detection confidence | ≥ 0.80 |
| Min detection confidence | ≥ 0.60 |

Labels remain **`video_health_status`** (weak health proxy), not veterinary pain scores.

---

## How many videos for a masters thesis?

Based on `cow_video_dataset_analysis.csv` (565 classified videos, 32 cows, cow 409 excluded):

| Plan | Videos / cow | Source videos | Est. sliding windows | Best for |
|------|-------------:|--------------:|---------------------:|----------|
| **Conservative** | **2** | **63** | **~580** | All 32 cows, strict balance; only cow **363** has 1 clip (partial) |
| **Recommended** | **3** | **92** | **~850** | Best trade-off: all cows, manageable compute, enough temporal diversity |
| **Rich** | **4** | **118** | **~1,100** | More data; 6 cows partially filled (352, 363, 415, 427, 428, 446) |

After QA filtering, expect roughly **55–70%** of candidate windows to pass → **~470–600 completed sequences** at the recommended **3 videos/cow** setting.

### Why 3 videos per cow is the thesis default

1. **All 32 cows stay in the study** (critical for cow-held-out CV and defensible n in a thesis).
2. **~92 source videos** is comparable to published cattle pain transfer work (Feighelstein et al. 2026 used 17 bulls with 3-min clips; your design adds more cows with shorter windows).
3. **~850 sliding windows** gives enough sequences for weak-label + DANN experiments without the redundancy of a full dense pass (~6,600 windows).
4. Unhealthy cows use **exercise-context clips only**, matching the hypothesis that movement/stress context is more informative for disease-related facial change.
5. Healthy cows get **session-balanced sampling** (before / during / after) for fair coverage.

### Partial-cow exceptions at 3 videos/cow

| Cow | Selected | Reason |
|-----|----------|--------|
| 363 | 1 | Only 1 during/after exercise clip in inventory |
| 352 | 2 | Only 2 eligible healthy clips total |
| 428 | 2 | Only 2 eligible healthy clips total |

Document these in your thesis methods section as inventory limits, not protocol failures.

---

## Usage

Run from the **DATASET workspace root** (`c:\Users\shivp\Downloads\Research\DATASET`).

### 1. Dry run (no YOLO, writes selection manifests)

```powershell
python Transferlearning/cow_face_sequences_thesis_stride8/create_thesis_stride8_sequences.py `
  --inventory cow_video_dataset_analysis.csv `
  --dataset-root . `
  --videos-per-cow 3 `
  --dry-run
```

### 2. Full extraction (GPU recommended)

Use `--overwrite` if `output/` already contains a dry-run.

```powershell
cd "C:\Users\shivp\Downloads\Research\DATASET"

python Transferlearning/cow_face_sequences_thesis_stride8/create_thesis_stride8_sequences.py `
  --inventory cow_video_dataset_analysis.csv `
  --dataset-root . `
  --model yolo_cow_face/yolo26s.pt `
  --videos-per-cow 3 `
  --output Transferlearning/cow_face_sequences_thesis_stride8/output `
  --overwrite
```

Replace `PATH\TO\best.pt` with your trained cow-face YOLO weights. **Default for this thesis run:** `yolo_cow_face/yolo26s.pt`.

### 3. Validate metadata after the run

```powershell
python Transferlearning/cow_face_sequences_thesis_stride8/validate_thesis_metadata.py `
  Transferlearning/cow_face_sequences_thesis_stride8/output
```

### 4. Conservative 2-video plan

```powershell
python Transferlearning/cow_face_sequences_thesis_stride8/create_thesis_stride8_sequences.py `
  --videos-per-cow 2 `
  --dry-run
```

### 5. Strict mode (skip cows below target count)

```powershell
python Transferlearning/cow_face_sequences_thesis_stride8/create_thesis_stride8_sequences.py `
  --videos-per-cow 3 `
  --strict-videos-per-cow `
  --dry-run
```

---

## Training integration

Point V3 trainers at:

```text
--manifest-csv Transferlearning/cow_face_sequences_thesis_stride8/output/completed_manifest.csv
--sequence-root Transferlearning/cow_face_sequences_thesis_stride8/output
--dataset-version thesis_stride8_qa
```

Keep final test cows fixed: **`363`, `403`, `404`, `408`**.

Report this dataset separately from `baseline_10s_250` and any future dense stride-5/stride-8 full-inventory runs.

---

## Related files

| File | Purpose |
|------|---------|
| `cow_video_dataset_analysis.csv` | Source video inventory |
| `Truro_Cow_Video_Data_Analysis.md` | Dataset statistics |
| `yolo_cow_face/create_10s_face_sequences.py` | Original 250-sequence baseline builder |
| `Transferlearning/Dann transfer/V3/data_code/create_dense_10s_face_sequences_v3.py` | Full-inventory dense builder (all videos) |

---

## Methods text (copy-ready)

> We extracted 10-second face-crop sequences using a trained YOLO cow-face detector (confidence ≥ 0.60, 224×224 crops). Windows were generated with an 8-second stride (2-second overlap) and resampled to 24 FPS (240 frames per sequence). For unhealthy cows, only during- and after-exercise videos were used, including sudden-fall recordings for cow 349. Healthy cows contributed up to three videos from any session. Sequences failing detection-quality gates (detection rate < 90% or filled-frame rate > 10%) were excluded. Labels were weak `video_health_status` proxies, not veterinary pain scores.
