#!/usr/bin/env bash
# Re-run ensemble test + metrics + reports from completed fold checkpoints (no training).
# Requires the same manifest/sequence paths as training and an --out-dir that already contains:
#   DANN: dann_splits.json, fold_*/best_dann.pt, and preferably dann_summary.json
#   Weak: weak_label_cv_splits.json, fold_*/best_weak_task1.pt, and preferably weak_label_cv_summary.json
#
# Usage on Vast (from Dann_transfer root):
#   export PATH="/workspace/ucaps_venv/bin:${PATH}"
#   bash V2/run_eval_only_vast.sh
set -euo pipefail

TARGET_MANIFEST="${TARGET_MANIFEST:-/workspace/ucaps_transfer/cow_face_sequences_10s_250/completed_manifest.csv}"
TARGET_ROOT="${TARGET_ROOT:-/workspace/ucaps_transfer/cow_face_sequences_10s_250}"
SRC_PROJECT="${SRC_PROJECT:-/workspace/ucaps_transfer_dann/ucaps_source}"
SRC_SEQ="${SRC_SEQ:-/workspace/ucaps_transfer_dann/ucaps_source/sequence}"
CKPT_DIR="${CKPT_DIR:-/workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752}"

NUM_WORKERS="${NUM_WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT_DANN_BASE="${OUT_DANN_BASE:-./holstein_task1_dann_v2_run}"
OUT_WEAK_GCE="${OUT_WEAK_GCE:-./holstein_task1_weak_gce_v2_run}"

MAX_FRAMES_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  MAX_FRAMES_ARGS=(--max-frames "${MAX_FRAMES}")
fi

SLIDING_ARGS=()
if [[ -n "${INFER_SLIDING_RAW_SPAN:-}" ]]; then
  SLIDING_ARGS=(--infer-sliding-raw-span "${INFER_SLIDING_RAW_SPAN}")
  [[ -n "${INFER_SLIDING_STRIDE:-}" ]] && SLIDING_ARGS+=(--infer-sliding-stride "${INFER_SLIDING_STRIDE}")
  [[ -n "${INFER_SLIDING_AGGREGATE:-}" ]] && SLIDING_ARGS+=(--infer-sliding-aggregate "${INFER_SLIDING_AGGREGATE}")
fi

MC_ARGS=()
if [[ -n "${EVAL_MC_SAMPLES:-}" ]] && [[ "${EVAL_MC_SAMPLES}" != "0" ]]; then
  MC_ARGS=(--eval-mc-samples "${EVAL_MC_SAMPLES}")
fi

BOOT_ARGS=()
if [[ -n "${DIAG_BOOTSTRAP_SAMPLES+x}" ]]; then
  BOOT_ARGS=(--diag-bootstrap-samples "${DIAG_BOOTSTRAP_SAMPLES}")
fi

echo "== DANN eval-only -> ${OUT_DANN_BASE} =="
python dann_adapt_v2.9.py \
  --eval-only \
  --manifest-csv "$TARGET_MANIFEST" \
  --sequence-root "$TARGET_ROOT" \
  --source-project-dir "$SRC_PROJECT" \
  --source-sequence-dir "$SRC_SEQ" \
  --checkpoint-dir "$CKPT_DIR" \
  --ckpt-kind task1 \
  --init-fold 0 \
  "${MAX_FRAMES_ARGS[@]}" \
  "${SLIDING_ARGS[@]}" \
  "${MC_ARGS[@]}" \
  "${BOOT_ARGS[@]}" \
  --out-dir "$OUT_DANN_BASE" \
  --label-column video_health_status \
  --test-cows 4 \
  --test-cow-ids 363,403,404,408 \
  --val-cows-per-fold 2 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --domain-weight 0.5 \
  --source-task2-weight 0.0 \
  --target-weak-weight 0.0 \
  --source-task1-sanity-floor 0.7

echo "== Weak-label GCE eval-only -> ${OUT_WEAK_GCE} =="
python weak_label_adapt_v2.9.py \
  --eval-only \
  --manifest-csv "$TARGET_MANIFEST" \
  --sequence-root "$TARGET_ROOT" \
  --checkpoint-dir "$CKPT_DIR" \
  --ckpt-kind task2 \
  --init-fold 0 \
  "${MAX_FRAMES_ARGS[@]}" \
  "${SLIDING_ARGS[@]}" \
  "${MC_ARGS[@]}" \
  "${BOOT_ARGS[@]}" \
  --out-dir "$OUT_WEAK_GCE" \
  --label-column video_health_status \
  --test-cows 4 \
  --test-cow-ids 363,403,404,408 \
  --val-cows-per-fold 2 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --task1-loss gce \
  --gce-q 0.7

echo "Eval-only pass complete."
