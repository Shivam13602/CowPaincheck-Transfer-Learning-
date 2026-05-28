# UCAPS → Holstein/Jersey Transfer Learning: Progress Report

**Date:** May 2026  
**Scope:** Adapt UCAPS v2.9 cattle facial-pain model to Canadian Holstein/Jersey dairy video using **existing labels only** (`video_health_status`: Healthy / Unhealthy).  
**Important limitation:** All reported metrics measure **weak health-proxy separation**, not validated veterinary pain detection.

---

## 1. Problem and data

We transfer a source model trained on UCAPS beef-cattle castration pain to a target domain of farm-recorded Holstein/Jersey clips. Source and target differ in breed, camera setup, clinical context, and label semantics. The target label is a folder-level health proxy (`Healthy` vs `Unhealthy`), not a pain score.

**Fixed evaluation protocol (all main runs):**
- 32 cows in the classified inventory (cow 409 excluded)
- **Held-out test cows:** 363, 403, 404, 408 (29 sequences in baseline set)
- Cow-level splits: no cow appears in train, validation, and test for the same experiment

**Baseline sequence dataset (frozen):** `cow_face_sequences_10s_250` — 250 sequences, one 10 s center window per selected video, 240 frames @ 24 FPS, 224×224 YOLO face crops.

---

## 2. Completed transfer-learning experiments

**20 distinct model conditions** have been trained and evaluated on the shared 4-cow test set. Summary:

| Phase | Protocol | Main result (final test seq AUC) | Key issue |
|-------|----------|----------------------------------|-----------|
| V1 / V2 / V3.1 | 14 folds × 2 val cows | ~0.47–0.56 | **All-positive threshold** (recall = 1.0, bacc = 0.5) |
| **V3 baseline matrix** (Rorqual job 12326664) | **7 folds × 4 val cows**, spec-constrained threshold | **Best seq AUC 0.577 (CORAL)** | Cow AUC still ~0.50; proxy labels only |

**V3 baseline matrix (8 conditions, synced locally):** weak BCE/GCE/focal, DANN domain weights 0.0–0.25, CORAL w=0.10. Outcomes:
- V3 **fixed threshold degeneracy** (no more all-positive predictions at primary threshold)
- **CORAL** gave best sequence-level ranking (0.577 vs 0.558 prior DANN best)
- **DANN family** reached calibrated cow bacc 0.75 but cow AUC remained 0.50 (n = 4 cows)
- Inner CV AUC (~0.64–0.71) **did not transfer** to the tiny held-out test set

Documentation updated: `V3/README.md`, main `Dann transfer/README.md` (20-run inventory), `V3.1/V3.1.md`.

---

## 3. New thesis sequence dataset (in progress)

To increase temporal coverage **without new labels**, we built a dedicated extraction pipeline:

**Location:** `Transferlearning/cow_face_sequences_thesis_stride8/`

| Design choice | Value |
|---------------|-------|
| Source videos | **3 per cow** (92 total), seed 42 |
| Unhealthy cows | **During + after exercise** only; cow **349 sudden-fall** included |
| Healthy cows | Any session (before / during / after) |
| Windows | 10 s length, **8 s stride (2 s overlap)**, UCAPS-aligned |
| Detector | **YOLO26s** (`yolo_cow_face/yolo26s.pt`) — chosen after side-by-side comparison with v8n and yolo11s |
| QA gates | Detection ≥ 90%, fill ≤ 10%, mean conf ≥ 0.80 |
| `dataset_version` | `thesis_stride8_qa` |

**Expected output:** ~900 candidate windows → **~500–600 QA-pass sequences** (estimated 55–65% pass rate).

**Status:** Local extraction running on GTX 1650 (fresh run after clearing partial output). Scripts: `create_thesis_stride8_sequences.py`, `validate_thesis_metadata.py`.

**Frozen for comparability:** `baseline_10s_250` and all prior V1–V3 run artifacts are unchanged.

---

## 4. Next plan (current labels only)

All steps below use **`video_health_status`** only. No veterinary scoring or new label collection in this phase.

1. **Finish sequence extraction** — complete `thesis_stride8_qa` dataset; run `validate_thesis_metadata.py`; document final counts in README (sequences, cows, session mix, QA rejection reasons).

2. **Train V3 best settings on new data** — on Rorqual or local GPU, run only:
   - `weak_focal` (best spec-constrained bacc on baseline)
   - `coral_w_0.10` (best seq AUC on baseline)
   - Same 7×4 CV, same test cows 363/403/404/408, `dataset_version=thesis_stride8_qa`.

3. **Compare datasets** — table: `baseline_10s_250` (250 seq) vs `thesis_stride8_qa` (~500–600 seq) on identical metrics (seq AUC, cow AUC, bacc, recall, calibration). Test whether sliding windows + session filtering improve proxy transfer under fixed labels.

4. **Optional ablations (same labels)** — SSL → CORAL on thesis data; cow-balanced sampling cap per epoch; per-cow / per-session error analysis on held-out test.

5. **Thesis write-up** — methods: video selection rules, windowing, YOLO26s QA, V3 protocol; results: 20-run lineage + new thesis-dataset runs; discussion: proxy-label ceiling, cow-held-out limits, label semantics vs pain claims.

**Explicitly out of scope for this plan:** new pain labels, vet-scoring shortlists, or deployment claims.

---

## 5. Bottom line

We have a **reproducible transfer pipeline** and **documented failure modes** (threshold collapse in V2/V3.1, modest gains in V3). The current work shifts from broad model sweeps to a **better target dataset under the same proxy labels** — balanced videos, exercise-filtered unhealthy clips, UCAPS-aligned sliding windows, and YOLO26s crops. Success is measured by **improved proxy metrics on the fixed 4-cow test**, not by claiming validated dairy pain detection.
