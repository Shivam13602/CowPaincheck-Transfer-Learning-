# DANN transfer experiment — history and results

This document records **Holstein/Jersey cow-held-out DANN** runs and related baselines, with **metric roles** spelled out so results are not misinterpreted as veterinary pain detection on the target herd.

---

## Metric roles (read before comparing numbers)

| Metric family | Meaning |
|---------------|---------|
| **UCAPS source `source_task1_*`** | True **binary pain vs no-pain** on UCAPS validation clips (scientific pain ground truth on source only). |
| **Holstein `val_*` / final test** | **`video_health_status`** (or chosen column): **weak health proxy** (Healthy vs Unhealthy). AUC/F1 here measure **proxy-label separation**, not validated target-domain pain. |
| **Calibrated** | Post-hoc **temperature scaling** on inner-validation logits (Guo et al., ICML 2017); thresholds from validation means applied to test without test tuning. |
| **`source_task2_*` (legacy runs)** | Auxiliary UCAPS moment/intensity head accuracy when Task2 loss was enabled; **not** Holstein pain. |

---

## Protocol (unchanged across runs)

- **Target:** 250 ten-second sequences, 32 cows; label column typically `video_health_status`.
- **Final test cows:** `363`, `403`, `404`, `408` (29 clips: 16 healthy proxy, 13 unhealthy proxy).
- **Inner CV:** 7 folds, 4 validation cows per fold; **ensemble** of per-fold best checkpoints on final test.
- **Source:** UCAPS project clips + `sequence` frames; initialization from UCAPS v2.9 checkpoints when noted.
- **Optional SSL:** SimSiam on **fold-train** Holstein cows only (non-transductive main claim).

Artifacts per DANN run usually include: `dann_splits.json`, `dann_run.json`, `dann_fold_summary.csv`, `dann_predictions.csv`, `dann_test_predictions.csv`, `dann_test_cow_aggregates.csv`, `dann_summary.json`, `dann_report.md`, `fold_<i>/best_dann.pt`, `fold_<i>/history.csv`.

---

## Run E0 — Zero-shot and weak-label baselines (reference)

Summarized in `vast_ai_results_20260502/RESULTS_SUMMARY.md` and archived zero-shot work under `../zeroshot learning/`.

| Stage | What was measured | Notable numbers (Holstein proxy) |
|-------|-------------------|----------------------------------|
| Zero-shot UCAPS → all 250 clips | Proxy separation | Video-health AUC **0.529**; healthy vs unhealthy mean pain prob ~**0.431** vs **0.437** (weak separation). |
| Freeze-CNN weak fine-tune | Proxy | Mean val AUC **0.603**; final seq AUC **0.548**; cow AUC **0.500**. |
| Full-backbone weak fine-tune | Proxy | Mean val AUC **0.634**; final seq AUC **0.457**; cow AUC **0.500**. |
| SSL → weak fine-tune | Proxy | Mean val AUC **0.610**; final seq AUC **0.447**; cow AUC **0.500**. |

---

## Run E2/E3 — SSL + DANN (May 2026 Vast sync, Task2 auxiliary on source)

**Local folder:** `vast_ai_results_20260502/holstein_dann_outputs/`  
**Remote logs (original):** `vast_ai_results_20260502/dann_train.log`

**Training design (typical for this run):**

- SSL-initialized encoder (`holstein_ssl_outputs` per fold).
- DANN: source **Task1 + Task2** auxiliary (e.g. `--source-task2-weight` > 0 in the older recipe), domain loss, **no** or low target weak BCE for the main claim.
- Inner validation selection on **Holstein proxy** AUC with **UCAPS source sanity** logged.

**Holstein proxy — final test (from `dann_summary.json`):**

| Metric | Value |
|--------|------:|
| Sequence AUC | **0.481** |
| Sequence balanced accuracy | **0.500** |
| Cow-level AUC | **0.750** |
| Cow-level balanced accuracy | **0.500** |
| Mean val threshold (applied to test) | **0.314** |

**Interpretation:** Higher **cow-level AUC** with **balanced accuracy 0.50** and confusion tables predicting all unhealthy at sequence level indicates **ranking / separation signal**, not a calibrated clinical classifier on the proxy labels.

**UCAPS source sanity (from `RESULTS_SUMMARY.md`):**

- Mean source Task1 AUC ~**0.536**; mean source Task1 F1 ~**0.602**.
- Mean source Task2 accuracy ~**0.262** (near random for 3-class).
- **Optional late weak-target-BCE DANN** was **skipped** because source sanity was already weak.

**Artifacts:** `holstein_dann_outputs/*` inside `vast_ai_results_20260502/`; SSL and weak-label sibling folders in the same archive directory.

---

## Run Task1-only baseline — DANN with SSL, no source Task2 loss (synced)

**Local folder:** `holstein_task1_dann_baseline/`  
**Typical remote path (old instance):** `/workspace/ucaps_transfer_dann/Dann_transfer/holstein_task1_dann_baseline/`

**Training design:**

- `--ckpt-kind task1` (UCAPS Task1 init).
- `--source-task2-weight 0.0`, `--target-weak-weight 0.0`.
- `--source-task1-sanity-floor 0.70`.
- SSL init: `holstein_ssl_outputs/fold_{fold}/best_ssl_simsiam.pt`.

**Holstein proxy — final test (`dann_summary.json` → `final_test`):**

| Metric | Raw | Calibrated (where present) |
|--------|-----|------------------------------|
| Sequence AUC | **0.466** | **0.466** (same in synced summary) |
| Sequence balanced accuracy | **0.500** | **0.500** |
| Cow AUC | **0.750** | **0.500** (calibrated cow AUC dropped) |
| Cow balanced accuracy | **0.500** | **0.500** |
| Mean validation temperature | — | **4.908** |
| Mean calibrated threshold | — | **0.264** |

**Per-fold note (`dann_fold_summary.csv`):** All folds showed `source_task1_sanity_pass=False` and `best_epoch=0` with `best_score=-inf` under the **old** selection rule (sanity gate forced selection score to −inf when AUC < floor), so checkpoints did not track later proxy-improving epochs. **`dann_adapt_v2.9.py` was later updated** so that if no epoch passes the floor, the fold keeps the **best proxy-validation** checkpoint and sets `checkpoint_selected_from_proxy_fallback` in the fold summary.

**Source Task1 AUC per fold (validation sanity, UCAPS truth):** roughly **0.38–0.53** across folds in that CSV — **below 0.7** everywhere.

---

## Run Task1-only optimized (SSL + DANN, updated script) — remote only

**Planned / executed on prior Vast instance (deleted):**

- Output dir: `holstein_task1_dann_optimized`
- Log: `task1_dann_optimized.log`
- Same CLI pattern as baseline but with **post-fix** `dann_adapt_v2.9.py` (proxy fallback + reporting).

**Status:** Instance was **deleted** before results were synced into this repo. **Re-run** on a new GPU instance and `scp -r` the output folder when complete.

---

## Code and ops pointers

| Item | Location |
|------|----------|
| DANN script | `dann_adapt_v2.9.py` |
| Weak-label / calibration helpers | `weak_label_adapt_v2.9.py` |
| Vast helper (paths as env defaults) | `run_task1_vast.sh` |
| Short Task1 vs prior DANN comparison | `TASK1_IMPROVEMENT_RESULTS.md` |
| May 2026 bundle narrative | `vast_ai_results_20260502/RESULTS_SUMMARY.md` |

**Peer-reviewed methods cited in project docs:** DANN (Ganin et al., JMLR 2016), SimSiam (Chen & He, CVPR 2021), SupCon (Khosla et al., NeurIPS 2020), focal / class-balanced / GCE and temperature scaling as implemented in argparse and reports.

---

## Vast SSH (ports change per boot)

**Defaults in `watch_remote_and_start_dann.ps1` / `upload_and_train.ps1`:** host **`199.126.203.145`**, port **`18387`** (update when Vast shows new Connect).

**Direct:**

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_ai_a100" -p 18387 root@199.126.203.145 -L 8080:localhost:8080
```

**Proxy jump** (when Vast lists a gateway, example `ssh2.vast.ai` + gateway port — use the exact **Connect** string from the UI):

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_ai_a100" -J root@ssh2.vast.ai:<GATEWAY_PORT> -p 18387 root@199.126.203.145 -L 8080:localhost:8080
```

Gateway-only lines from Vast (e.g. `ssh -p <port> root@ssh2.vast.ai`) are not a substitute for the instance hop; use **`-J`** with the instance `ssh -p 18387 root@199.126.203.145` line above.

**What was found on first connect:**

| Check | Result |
|-------|--------|
| GPU | **NVIDIA A100-PCIE-40GB** |
| CUDA (toolkit) | **12.4** |
| `/workspace` | **Missing** on first boot — created as `/workspace/ucaps_transfer_dann/Dann_transfer/` for code layout. |
| System `python3` | **No PyTorch** — use venv `/workspace/ucaps_venv`. **Pip:** CUDA wheels from `https://download.pytorch.org/whl/cu124`, then PyPI: `pandas numpy tqdm scikit-learn Pillow`. Prefer **`vast_bootstrap_pip.sh`** via `nohup` (see `upload_and_train.ps1`) so a long install survives SSH disconnect; log `/workspace/ucaps_venv_bootstrap.log`, done marker `/workspace/ucaps_venv_bootstrap.done`. |
| Datasets / UCAPS ckpts | **Not present** until you upload — **re-attach the Vast volume** or **rsync/scp**: Holstein `cow_face_sequences_10s_250`, UCAPS `v2.9` checkpoints, `ucaps_source` (+ `sequence/`), optional `holstein_ssl_outputs` for SSL-init. |

**Auto-start when ready:** from this repo folder run `.\watch_remote_and_start_dann.ps1` (edit host/port if Vast changes them). It polls until torch + manifest + sequence + `best_model_v2.9_task1_fold_0.pt` exist, then launches `nohup` DANN (adds `--ssl-checkpoint-dir` only if `holstein_ssl_outputs/fold_0/best_ssl_simsiam.pt` exists).

**One-shot upload + venv + wait for UCAPS + train:** `.\upload_and_train.ps1` (same folder) uploads local `../cow_face_sequences_10s_250` and core scripts, runs two-step pip on the instance, then waits for you to place UCAPS checkpoints + `ucaps_source` on the server before starting DANN.

**Code uploaded to the instance (this repo):**

- `dann_adapt_v2.9.py`, `weak_label_adapt_v2.9.py`, `v2.9_training_classification.py`, `holstein_v29_dataset.py`, `ssl_pretrain_holstein_v2.9.py`, `run_task1_vast.sh` → `/workspace/ucaps_transfer_dann/Dann_transfer/`

**Full DANN command (paths must exist first):** same as `run_task1_vast.sh` / `README.md` Task1-only block; use `/workspace/ucaps_venv/bin/python` once the venv is ready. Example:

```bash
cd /workspace/ucaps_transfer_dann/Dann_transfer
nohup /workspace/ucaps_venv/bin/python dann_adapt_v2.9.py \
  --manifest-csv /workspace/ucaps_transfer/cow_face_sequences_10s_250/completed_manifest.csv \
  --sequence-root /workspace/ucaps_transfer/cow_face_sequences_10s_250 \
  --source-project-dir /workspace/ucaps_transfer_dann/ucaps_source \
  --source-sequence-dir /workspace/ucaps_transfer_dann/ucaps_source/sequence \
  --checkpoint-dir /workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752 \
  --ckpt-kind task1 --init-fold 0 \
  --ssl-checkpoint-dir holstein_ssl_outputs \
  --out-dir holstein_task1_dann_optimized \
  --test-cow-ids 363,403,404,408 \
  --num-epochs 20 --batch-size 8 --num-workers 2 --learning-rate 1e-5 \
  --source-task2-weight 0.0 --target-weak-weight 0.0 --source-task1-sanity-floor 0.70 \
  > task1_dann_optimized.log 2>&1 &
```
