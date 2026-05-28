#!/usr/bin/env bash
# Vast.ai one-shot launcher for V2 protocol (paths match necessary_filesV1-style layout).
set -eu
BUNDLE="${BUNDLE:-/workspace/necessary_filesV1}"
LOG="${TRAIN_LOG:-/workspace/task1_vast_train_v2.log}"

exec >>"$LOG" 2>&1
echo "=== vast_launch_task1_v2 $(date -u) ==="

mkdir -p /workspace/ucaps_transfer_dann
ln -sfn "$BUNDLE/Dann_transfer" /workspace/ucaps_transfer_dann/Dann_transfer
ln -sfn "$BUNDLE/ucaps_transfer_dann/ucaps_source" /workspace/ucaps_transfer_dann/ucaps_source
ln -sfn "$BUNDLE/ucaps_transfer" /workspace/ucaps_transfer

test -f /workspace/ucaps_transfer/cow_face_sequences_10s_250/completed_manifest.csv
test -d /workspace/ucaps_transfer_dann/Dann_transfer

if [[ ! -f /workspace/ucaps_venv_bootstrap.done ]] || ! /workspace/ucaps_venv/bin/python -c "import torch" &>/dev/null; then
  echo "Running vast_bootstrap_pip.sh (first time or broken venv)..."
  bash /workspace/ucaps_transfer_dann/Dann_transfer/vast_bootstrap_pip.sh
fi

export PATH="/workspace/ucaps_venv/bin:$PATH"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /workspace/ucaps_transfer_dann/Dann_transfer
export OUT_DANN_BASE="${OUT_DANN_BASE:-./holstein_task1_dann_v2_run}"
export OUT_WEAK_GCE="${OUT_WEAK_GCE:-./holstein_task1_weak_gce_v2_run}"
echo "OUT_DANN_BASE=$OUT_DANN_BASE OUT_WEAK_GCE=$OUT_WEAK_GCE"
exec bash V2/run_task1_vast.sh
