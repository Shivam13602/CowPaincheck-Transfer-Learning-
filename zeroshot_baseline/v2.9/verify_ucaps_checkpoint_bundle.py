#!/usr/bin/env python3
"""Verify nine fold checkpoints exist and load cfg from one checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ucaps_v29_eval_loader import load_evaluate_test_cli_module

_eval = load_evaluate_test_cli_module()
_ckpt_path = _eval._ckpt_path
_infer_available_folds = _eval._infer_available_folds


def main() -> int:
    p = argparse.ArgumentParser(description="Verify UCAPS v2.9 checkpoint bundle.")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--ckpt-kind", choices=("combined", "task1", "task2"), default="task2")
    args = p.parse_args()

    d = args.checkpoint_dir.resolve()
    folds = _infer_available_folds(d, args.ckpt_kind)
    print(f"checkpoint_dir: {d}")
    print(f"ckpt_kind: {args.ckpt_kind}")
    print(f"folds found: {folds} (count={len(folds)})")

    if len(folds) != 9:
        print("WARNING: expected 9 folds for full manuscript ensemble.")

    if not folds:
        return 2

    probe = _ckpt_path(d, int(folds[0]), args.ckpt_kind)
    ckpt = torch.load(probe, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    print(f"probe_file: {probe.name}")
    print(f"checkpoint_label: {ckpt.get('checkpoint_label', '')}")
    if isinstance(cfg, dict):
        print(f"cfg.run_tag: {cfg.get('run_tag', '')} max_frames={cfg.get('max_frames')} resolution={cfg.get('resolution')}")

    missing = [i for i in range(9) if not _ckpt_path(d, i, args.ckpt_kind).exists()]
    if missing:
        print(f"Missing flat evaluator filenames for indices: {missing}")
        print("(Scripts expect best_model_v2.9_*_fold_K.pt in this folder root, not only fold_K/best.pt.)")
        alt: list[str] = []
        for i in missing:
            for sub in (f"fold_{i}", f"Fold_{i}", str(i)):
                bp = d / sub / "best.pt"
                if bp.is_file():
                    alt.append(f"  fold {i}: found {bp.relative_to(d)}")
                    break
        if alt:
            print("Weights may still exist in subfolders:")
            print("\n".join(alt))
            print(
                'Flatten with:\n  python prepare_ucaps_checkpoints.py '
                f'--source "{d}" --dest "{d}" --ckpt-kind {args.ckpt_kind}'
            )
        return 1

    print("OK: folds 0-8 all present (flat names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
