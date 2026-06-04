# V6 — Autoresearch (VastAI, June 2026)

Automated Stage A/B sweep on the **V5 549-seq / 8-cow** protocol. Karpathy-style loop: [`auto research/`](auto%20research/).

## Read first

| Document | Contents |
|----------|----------|
| **[`v6.md`](v6.md)** | Full results: leaderboard, per-cow, disease-wise, failure modes, vs V5 |
| **[`results/README.md`](results/README.md)** | What is committed to GitHub |
| **[`../../README.md`](../../README.md)** | Repo landing page (summary + why transfer is hard) |

## Headline

- **Best:** `A_s3_focal_g2p5_cb` — focal γ=2.5, class-balanced, freeze CNN → seq AUC **0.611**, F1 **0.466**, 24 TP / 21 FN / 34 FP on 143 test sequences.
- **Stage B (DANN/CORAL):** did not beat V5; test AUC ~0.47, threshold collapse (0 TP).

## Folder layout

```
V6/
  v6.md
  README.md
  results/
    vast_auto/           ← 7 trials, reports only (no .pt, no fold checkpoints)
    v6_results_analysis.json
  scripts/
    build_v6_results_analysis.py
    print_v6_leaderboard.py
    download_results_vast.ps1
    export_v6_for_github.py
  auto research/
    run_autoresearch.py, search_space*.json, program.md
```

## Regenerate metrics JSON

```bash
python dann_transfer/V6/scripts/build_v6_results_analysis.py
python dann_transfer/V6/scripts/print_v6_leaderboard.py
```
