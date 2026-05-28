#!/usr/bin/env bash
set -euo pipefail
export PATH="/workspace/ucaps_venv/bin:${PATH}"
cd /workspace/ucaps_transfer_dann/Dann_transfer
LOG="${WEAK_ONLY_LOG:-/workspace/weak_only_retry.log}"
exec >>"$LOG" 2>&1
echo "=== weak_only_retry start $(date -u) ==="
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
echo "=== weak_only_retry done $(date -u) ==="
touch /workspace/weak_only_retry.done
