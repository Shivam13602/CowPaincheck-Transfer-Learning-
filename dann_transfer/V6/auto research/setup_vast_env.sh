#!/usr/bin/env bash
# One-time environment setup for V6 autoresearch on a VastAI A100 instance.
set -euo pipefail

echo "== Python / pip =="
python3 --version
python3 -m pip install --upgrade pip

echo "== PyTorch (CUDA 12.1 wheels) =="
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "== Core deps =="
python3 -m pip install numpy pandas scikit-learn pillow tqdm

echo "== Sanity =="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== Expected staged data layout =="
cat <<'LAYOUT'
/root/data/thesis_stride8/output/completed_manifest.csv
/root/data/thesis_stride8/output/sequences/...
/root/data/v2.9_20260502_181533/ (task1 fold checkpoints)   <-- from Rorqual
/root/data/ucaps_source/ + /root/data/ucaps_source/sequence/ <-- from Rorqual (S4 only)
LAYOUT

echo "Done. Next: dry-run the V6 runner with the vast search space."
