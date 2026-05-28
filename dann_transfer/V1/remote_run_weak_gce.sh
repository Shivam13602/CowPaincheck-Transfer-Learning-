#!/usr/bin/env bash
# Run on Vast after evaluate_test_set_v2.9_cli.py is present (weak_label imports it).
set -euo pipefail
export PATH="/workspace/ucaps_venv/bin:${PATH}"
cd /workspace/ucaps_transfer_dann/Dann_transfer
LOG="${WEAK_LOG:-/workspace/weak_gce_vast2.log}"
exec >>"$LOG" 2>&1
echo "=== weak_label start $(date -u) ==="
python weak_label_adapt_v2.9.py \
  --manifest-csv /workspace/ucaps_transfer/cow_face_sequences_10s_250/completed_manifest.csv \
  --sequence-root /workspace/ucaps_transfer/cow_face_sequences_10s_250 \
  --checkpoint-dir /workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752 \
  --ckpt-kind task2 \
  --init-fold 0 \
  --out-dir ./holstein_task1_weak_gce_vast_run \
  --label-column video_health_status \
  --test-cows 4 \
  --val-cows-per-fold 4 \
  --num-epochs 10 \
  --batch-size 8 \
  --task1-loss gce \
  --gce-q 0.7
echo "=== weak_label done $(date -u) ==="
