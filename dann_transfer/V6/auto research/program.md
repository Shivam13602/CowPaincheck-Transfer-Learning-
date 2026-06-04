# V6 Auto Research Program

This file is the operating policy for autonomous trial generation and selection.

## Mission
Improve V5 transfer performance while preserving thesis-grade rigor:
- raise sequence AUC and operational F1,
- reduce threshold collapse behavior,
- improve lameness capture without exploding healthy false positives.

## Rules of operation
1. Keep data split, checkpoint source, and evaluation protocol fixed unless explicitly changing one factor as an ablation.
2. Change one major axis at a time:
   - loss family,
   - weighting/sampler policy,
   - alignment mode/weight,
   - temporal pooling mode.
3. Every trial must emit:
   - summary JSON,
   - report markdown,
   - test predictions CSV.
4. Promote a trial only if:
   - seq AUC improves over baseline,
   - primary-threshold confusion is non-degenerate,
   - condition-wise behavior is interpretable.

## Candidate families
- A: Loss and weighting geometry (weak/focal/gce/class-balanced).
- B: DANN and CORAL alignment schedule refinement.
- C: Temporal pooling and inference aggregation (mean/trimmed/max windows).
- D: Micro-expression-focused region and motion-aware variants (when pipeline hooks are added).

## Baseline anchors
- S3 weak focal (V5): seq AUC ~0.487
- S4 dann_dw_0.25 (V5): seq AUC ~0.592, seq F1 ~0.366

## Trial hygiene
- Use deterministic seeds.
- Log runtime environment (GPU, driver, CUDA, package versions).
- Keep run IDs and paths stable.
- Do not overwrite previous run folders.
