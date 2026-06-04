# V6 results (versioned on GitHub)

| Path | Contents |
|------|----------|
| [`vast_auto/`](vast_auto/) | VastAI run outputs for **7 trials** — JSON summaries, MD reports, test CSVs only |
| [`v6_results_analysis.json`](v6_results_analysis.json) | Consolidated metrics (regenerate with `../scripts/build_v6_results_analysis.py`) |

## Trials in `vast_auto/`

**Stage A (weak-label, `weak_label_adapt_v3.py`):**

- `A_s3_focal_g1p5_cb` — focal γ=1.5, class-balanced  
- `A_s3_focal_g2p5_cb` — focal γ=2.5, class-balanced (**best overall**)  
- `A_s3_gce_q0p6_cb` — GCE q=0.6, class-balanced  
- `A_s3_gce_q0p8_cb` — GCE q=0.8, class-balanced  

**Stage B (alignment, `dann_adapt_v3.py`):**

- `B_s4_dann_dw0p15` — DANN domain weight 0.15  
- `B_s4_dann_dw0p20` — DANN domain weight 0.20  
- `B_s4_coral_w0p02` — CORAL weight 0.02  

## Not in git

- Training checkpoints (`.pt`), `fold_*` dirs, full training logs  
- Raw sequences / UCAPS source (see [`../../../docs/DATA_ACCESS.md`](../../../docs/DATA_ACCESS.md))

## Prune local copy before push

```bash
python dann_transfer/V6/scripts/export_v6_for_github.py
```

Keeps only report-grade files (~1.5 MB total).
