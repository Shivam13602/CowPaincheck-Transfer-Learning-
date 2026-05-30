# V5 results (versioned)

| Path | Contents |
|------|----------|
| [`549_interim/`](549_interim/) | Rorqual run `v5_thesis_8cow_549` — 549 QA sequences, 8-cow test. Reports/JSON/CSVs only (no `.pt` checkpoints). |
| [`v5_results_analysis.json`](v5_results_analysis.json) | Machine-readable metrics for all 9 conditions (regenerate with `../scripts/build_v5_results_analysis.py`). |

**Stages in `549_interim`:** `S3_weak_{bce,focal,gce}`, `S4_dann_dw_{0.10,0.25,0.50}`, `S4_coral_w_{0.01,0.05,0.10}`.

Full interpretation: [`../v5.md`](../v5.md).
