#!/usr/bin/env bash
# Example: multi-seed evaluation variance for V3.1 (weak-label or DANN eval-only).
#
# Requires completed runs under distinct OUT dirs OR use this only as a template for Slurm arrays.
#
# Usage (repo root):
#   bash V3.1/run_eval_repeat_splits.sh
#
# Override:
#   SEEDS="42 43 44" WEAK_OUT_ROOT=./weak_runs bash V3.1/run_eval_repeat_splits.sh
set -euo pipefail

SEEDS="${SEEDS:-42 43}"
MANIFEST="${TARGET_MANIFEST:-./cow_face_sequences_10s_250/completed_manifest.csv}"
SEQ_ROOT="${TARGET_ROOT:-./cow_face_sequences_10s_250}"
CKPT_DIR="${CKPT_DIR:-./v2.9_20260222_144752}"

for s in $SEEDS; do
  echo "=== seed=$s eval-only weak (expects existing checkpoints under OUT dir) ==="
  OUT="${WEAK_OUT_ROOT:-./holstein_task1_weak_gce_v31_run_seed_${s}}"
  if [[ ! -d "$OUT" ]]; then
    echo "Skip: missing $OUT (train first or set WEAK_OUT_ROOT)."
    continue
  fi
  python weak_label_adapt_v3_1.py \
    --manifest-csv "$MANIFEST" \
    --sequence-root "$SEQ_ROOT" \
    --checkpoint-dir "$CKPT_DIR" \
    --ckpt-kind task2 \
    --out-dir "$OUT" \
    --seed "$s" \
    --eval-only
done

echo "Done. Compare weak_label_cv_summary.json across OUT dirs."
