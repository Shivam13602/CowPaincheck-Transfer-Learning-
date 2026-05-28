#!/usr/bin/env bash
# Vast: weak-label GCE (Holstein) + optional UCAPS pretrained task1 eval; log everything.
set -euo pipefail
export PATH="/workspace/ucaps_venv/bin:${PATH}"
cd /workspace/ucaps_transfer_dann/Dann_transfer
LOG="${V1_FINISH_LOG:-/workspace/v1_finish.log}"
exec >>"$LOG" 2>&1
echo "=== v1_finish start $(date -u) ==="

echo "== Holstein weak-label GCE (7-fold CV + held-out test ensemble) =="
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

echo "== UCAPS pretrained checkpoints: task1 held-out test (from splits JSON) =="
mkdir -p ./V1_artifacts_staging/ucaps_pretrained_task1_eval
set +e
python evaluate_test_set_v2.9_cli.py \
  --project_dir /workspace/ucaps_transfer_dann/ucaps_source \
  --sequence_dir /workspace/ucaps_transfer_dann/ucaps_source/sequence \
  --checkpoint_dir /workspace/ucaps_transfer/v2.9/checkpoints_v2.9-20260502T160826Z-3-001/checkpoints_v2.9/v2.9_20260222_144752 \
  --ckpt_kind task1 \
  --save_dir ./V1_artifacts_staging/ucaps_pretrained_task1_eval \
  --ensemble mean_logits \
  --optimize_task1_threshold
EV=$?
set -e
echo "UCAPS eval exit code: $EV (non-zero is OK if paths mismatch)"

echo "=== v1_finish done $(date -u) ==="
touch /workspace/v1_finish.done
