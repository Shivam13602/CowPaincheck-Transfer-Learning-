# CowPaincheck Transfer Learning

Transfer learning from **UCAPS** (beef cattle, castration-related facial pain) to **Holstein/Jersey dairy** farm video using weak `video_health_status` labels (Healthy vs Unhealthy-proxy).

> **Important:** Holstein metrics measure **disease-context / health-proxy separation**, not validated veterinary pain scores. UCAPS source training uses true pain labels; target domain does not.

Repository: [github.com/Shivam13602/CowPaincheck-Transfer-Learning-](https://github.com/Shivam13602/CowPaincheck-Transfer-Learning-)

Extended narrative: [`CowPain Transfer.md`](CowPain%20Transfer.md)

---

## Latest result (V6 — June 2026)

**VastAI autoresearch** on the same **549-sequence / 8-cow held-out test** protocol as V5 ([`dann_transfer/V5/splits/v5_split.json`](dann_transfer/V5/splits/v5_split.json)).

| What we ran | Outcome |
|-------------|---------|
| **Stage A** — 4 weak-label heads (focal γ=1.5/2.5, GCE q=0.6/0.8) + **class-balanced** loss, frozen CNN, 80 epochs | **Best: `A_s3_focal_g2p5_cb`** — seq AUC **0.611**, F1 **0.466**, recall **0.533** |
| **Stage B** — 3 alignment runs (DANN dw 0.15/0.20, CORAL 0.02), no target weak BCE | **Failed on test** — seq AUC ~0.47, **0 TP** at primary threshold |

Full analysis: [`dann_transfer/V6/v6.md`](dann_transfer/V6/v6.md) · Versioned outputs: [`dann_transfer/V6/results/`](dann_transfer/V6/results/)

---

## What we did in V6

1. **Same data and split as V5** — 143 test sequences from 8 cows (4 Healthy + 4 Unhealthy by `cow_health_status`); 5-fold cow CV on 23 train cows; checkpoint `v2.9_20260502_181533`.
2. **Stage A (loss geometry)** — tested whether **focal loss**, **GCE**, and **effective-number class balancing** fix V5’s weak-label failure (S3 AUC ~0.48).
3. **Stage B (alignment)** — retested DANN/CORAL with new weights on VastAI; did **not** repeat V5’s best weights (DANN 0.25, CORAL 0.10).
4. **Evaluation** — built into trainers: sequence / video / cow metrics, threshold sweep, temperature scaling, bootstrap CIs. No separate eval job.

Autoresearch tooling: [`dann_transfer/V6/auto research/`](dann_transfer/V6/auto%20research/)

---

## Results at a glance

### Sequence level (143 test clips, primary threshold τ)

| Trial | Seq AUC | F1 | Recall | TP | FN | FP | TN |
|-------|--------:|---:|-------:|---:|---:|---:|---:|
| **A_s3_focal_g2p5_cb** (best) | **0.611** | **0.466** | **0.533** | 24 | 21 | 34 | 64 |
| A_s3_gce_q0p6_cb | 0.547 | 0.484 | 1.000 | 45 | 0 | 96 | 2 |
| A_s3_focal_g1p5_cb | 0.543 | 0.385 | 0.444 | 20 | 25 | 39 | 59 |
| B_s4_dann_dw0p15 (S4) | 0.467 | 0.000 | 0.000 | 0 | 45 | 1 | 97 |
| V5 best S4 (reference) | 0.593 | 0.366 | 0.333 | 15 | 30 | 22 | 76 |

- **45** test sequences are Unhealthy-proxy; **98** are Healthy-proxy.
- Best focal run catches **24/45** unhealthy clips but flags **34** healthy clips (high FP).
- GCE q=0.6 is the opposite failure: **all-positive** (96 FP, 2 TN) at τ≈0.035.

### Cow level (8 held-out cows, sequence-majority / aggregate metrics)

| Cow ID | Health proxy | n seq | Best focal (γ=2.5) — seq positives | Cow-level note |
|--------|--------------|------:|----------------------------------|----------------|
| 363 | Unhealthy | 6 | 6/6 flagged | Unhealthy cow fully detected at clip level |
| 370 | Healthy | 24 | 0/24 | Clean on this cow |
| 378 | Healthy | 33 | 0/33 | Clean (cow healthy; many clips labeled healthy) |
| 403 | Unhealthy | 10 | 10/10 flagged | Full recall on clips |
| 404 | Healthy | 5 | 0/5 | Clean |
| 408 | Healthy | 6 | 0/6 | Clean |
| 433 | Unhealthy | 29 | 29/29 flagged | Full recall on clips |
| 436 | Healthy | 30 | 0/30 | Clean |

Cow AUC on best focal: **0.667** (vs ~0.40 for V5 S4). Bootstrap CIs remain wide (only 8 test cows).

### Disease context (best focal, clip level)

| Condition | % clips flagged positive | Comment |
|-----------|-------------------------:|---------|
| possible mastitis | **59%** | Easier proxy pattern |
| lameness | **44%** | Still under-detected vs mastitis |
| healthy / healthy folder | **34–36%** | Main FP source |

---

## Main failure modes

| Mode | Where | What happens |
|------|-------|----------------|
| **Score compression** | All stages | Sigmoid outputs cluster in a narrow band (~0.33–0.39). Small shifts move many clips across τ → unstable TP/FP. |
| **Threshold collapse (low)** | GCE q=0.6, V5 BCE | τ too low → predict almost everything Unhealthy (96 FP). |
| **Threshold collapse (high)** | V6 Stage B, V5 BCE/GCE | τ too high → predict everything Healthy (**0 TP**). |
| **Val ≠ test** | Stage B, historical S3 | Inner 4-cow val AUC up to **0.89**; 8-cow test AUC **~0.47–0.48**. |
| **Calibration trap** | S4 / focal | Temperature scaling at primary τ can destroy usable F1; report **raw** metrics. |
| **Proxy ≠ pain** | Whole project | Model learns herd/disease **context** (mastitis, lameness folders), not graded pain. |
| **Label noise** | Cow 378 etc. | Cow-level label disagrees with clip-level `video_health_status` on some animals. |

---

## Why transferring “pain signals” is hard here

1. **Domain shift** — UCAPS: controlled beef, castration protocol, expert pain labels. Target: field dairy, different breeds, lighting, and behaviour; labels are **video-level health proxy**, not pain scores.
2. **Weak labels** — `video_health_status` mixes mastitis, lameness, and healthy folders; it is not equivalent to UCAPS moment/pain taxonomy.
3. **Subtle facial signal** — Micro-expressions (orbital tightening, ear posture) are low-amplitude and easily drowned by pose, motion blur, and crop quality.
4. **Small test** — Only **8 cows** and **143** overlapping sequences; metrics have high variance (bootstrap cow AUC CI spans most of [0, 1]).
5. **Class geometry** — Healthy clips dominate; without **class-balanced** loss (V6 Stage A), models collapse to predict majority class.
6. **Alignment is not automatic** — V5 showed DANN/CORAL can fix ranking when weights match; V6 Stage B used different weights and **regressed**, showing alignment must be tuned jointly with loss heads.

**Progress:** V6 Stage A proves **loss + class balancing** can exceed V5 alignment on seq AUC; combining that head with V5-quality DANN/CORAL is the next step.

---

## Experiment timeline

| Stage | Folder | Best seq AUC | Report |
|-------|--------|-------------:|--------|
| Phase 0 | [`zeroshot_baseline/`](zeroshot_baseline/) | 0.548 | zeroshot README |
| V3 | [`dann_transfer/V3/`](dann_transfer/V3/) | 0.577 | V3 README |
| V4 | [`dann_transfer/V4/`](dann_transfer/V4/) | 0.421 | V4 README |
| V5 | [`dann_transfer/V5/`](dann_transfer/V5/) | 0.593 (S4) | [`v5.md`](dann_transfer/V5/v5.md) |
| **V6** | [`dann_transfer/V6/`](dann_transfer/V6/) | **0.611 (Stage A)** | [`v6.md`](dann_transfer/V6/v6.md) |

---

## Repository map

```
CowPaincheck-Transfer-Learning/
├── README.md                 ← this file (GitHub landing)
├── CowPain Transfer.md       ← long-form project overview
├── docs/                     ← DATA_ACCESS, thesis report, figures
├── datasets/                 ← manifests (not raw video)
├── zeroshot_baseline/
└── dann_transfer/
    ├── V1/ … V5/             ← frozen experiment lines
    └── V6/                   ← autoresearch + Vast results (June 2026)
        ├── v6.md
        ├── results/vast_auto/
        └── auto research/
```

**Data and checkpoints are not in git.** See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

---

## Reproduce V6 analysis locally

```powershell
# After downloading results from Vast (see dann_transfer/V6/scripts/download_results_vast.ps1)
python dann_transfer/V6/scripts/build_v6_results_analysis.py
python dann_transfer/V6/scripts/print_v6_leaderboard.py
```

---

## Citation

Academic/research use. Citation to be added upon thesis publication.
