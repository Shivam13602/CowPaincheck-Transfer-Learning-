#!/usr/bin/env bash
# Run on the Vast instance (survives SSH drop if started with nohup).
# Installs /workspace/ucaps_venv with PyTorch cu124 + deps for DANN scripts.
set -euo pipefail
LOG="${LOG:-/workspace/ucaps_venv_bootstrap.log}"
exec >>"$LOG" 2>&1
echo "=== vast_bootstrap_pip start $(date -u) ==="
rm -f /workspace/ucaps_venv_bootstrap.done
if [[ -d /workspace/ucaps_venv ]]; then
  if ! /workspace/ucaps_venv/bin/python -c "import torch" &>/dev/null; then
    echo "Removing incomplete venv (torch import failed)"
    rm -rf /workspace/ucaps_venv
  fi
fi
if [[ ! -x /workspace/ucaps_venv/bin/python ]]; then
  python3 -m venv /workspace/ucaps_venv
fi
/workspace/ucaps_venv/bin/pip install -q --upgrade pip
/workspace/ucaps_venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
/workspace/ucaps_venv/bin/pip install pandas numpy tqdm scikit-learn Pillow
/workspace/ucaps_venv/bin/python -c "import torch; print('OK', torch.__version__, torch.cuda.is_available())"
date -u > /workspace/ucaps_venv_bootstrap.done
echo "=== vast_bootstrap_pip done $(date -u) ==="
