#!/usr/bin/env bash
# Shared Alliance H100 (Rorqual / Narval) runtime knobs for V5 training.
# Source from run_v5_matrix_*.sh or sbatch wrappers. Override via --export=ALL,WEAK_BATCH_SIZE=64,...
#
# OOM fallback (single-GPU): WEAK_BATCH_SIZE=64 DANN_BATCH_SIZE=24 NUM_WORKERS=16

export NUM_WORKERS="${NUM_WORKERS:-24}"
export DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-4}"
export DATALOADER_PERSISTENT="${DATALOADER_PERSISTENT:-1}"

# Weak-label (frozen CNN): large batches are safe on 80GB H100.
export WEAK_BATCH_SIZE="${WEAK_BATCH_SIZE:-128}"
# DANN/CORAL: dual source+target streams; keep moderate to avoid OOM with 32 frames.
export DANN_BATCH_SIZE="${DANN_BATCH_SIZE:-48}"
export BATCH_SIZE="${BATCH_SIZE:-${WEAK_BATCH_SIZE}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Set in Python main() too; exporting helps child processes / libraries.
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
