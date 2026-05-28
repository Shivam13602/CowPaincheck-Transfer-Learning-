#!/usr/bin/env python3
"""
Normalize UCAPS v2.9 fold weights from a Google Drive export into filenames expected by
evaluate_test_set_v2.9_cli.py and evaluate_holstein_zero_shot_v2.9.py.

Typical Drive layout: each fold folder contains best.pt (names may vary).
Target layout: best_model_v2.9_task2_fold_{k}.pt for k in 0..8.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional


def _infer_fold_index(path: Path) -> Optional[int]:
    """Try to read fold index from a parent directory name (fold_3, 3, fold-3, ...)."""
    for part in path.parts[::-1]:
        m = re.search(r"(?:^|[_\-])(?:fold|f)[_\-]?(\d+)$", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.fullmatch(r"(\d+)", part)
        if m2 and path.parent.name.lower() in {"folds", "checkpoint", "checkpoints"}:
            return int(m2.group(1))
    for part in path.parts[::-1]:
        m3 = re.search(r"(\d+)", part)
        if m3:
            return int(m3.group(1))
    return None


def discover_normalized(source: Path, ckpt_kind: str) -> Optional[list[tuple[int, Path]]]:
    pairs: list[tuple[int, Path]] = []
    if ckpt_kind == "combined":
        for fp in sorted(source.glob("best_model_v2.9_fold_*.pt")):
            m = re.search(r"v2\.9_fold_(\d+)\.pt$", fp.name)
            if m:
                pairs.append((int(m.group(1)), fp))
    elif ckpt_kind in {"task1", "task2"}:
        pat = f"best_model_v2.9_{ckpt_kind}_fold_*.pt"
        for fp in sorted(source.glob(pat)):
            m = re.search(r"_fold_(\d+)\.pt$", fp.name)
            if m:
                pairs.append((int(m.group(1)), fp))
    else:
        return None
    if len(pairs) == 9 and len({k for k, _ in pairs}) == 9:
        return sorted(pairs, key=lambda t: t[0])
    return None


def _flat_prefix(ckpt_kind: str) -> str:
    return {
        "task2": "best_model_v2.9_task2_fold_",
        "task1": "best_model_v2.9_task1_fold_",
        "combined": "best_model_v2.9_fold_",
    }[ckpt_kind]


def collect_fold_pairs(source: Path, ckpt_kind: str) -> list[tuple[int, Path]]:
    """
    Resolve one weight file per fold 0..8 without duplicate-key conflicts.

    Prefer flat evaluator names in `source/` root; for any missing fold, use
    `fold_k/best.pt` (or `Fold_k`, numeric folder).
    """
    by_fold: dict[int, Path] = {}

    if ckpt_kind == "combined":
        glob_pat = "best_model_v2.9_fold_*.pt"
        fold_re = re.compile(r"v2\.9_fold_(\d+)\.pt$")
    else:
        glob_pat = f"best_model_v2.9_{ckpt_kind}_fold_*.pt"
        fold_re = re.compile(r"_fold_(\d+)\.pt$")

    for fp in sorted(source.glob(glob_pat)):
        m = fold_re.search(fp.name)
        if m:
            by_fold[int(m.group(1))] = fp

    for i in range(9):
        if i in by_fold:
            continue
        for sub in (f"fold_{i}", f"Fold_{i}", str(i)):
            candidate = source / sub / "best.pt"
            if candidate.is_file():
                by_fold[i] = candidate
                break

    if len(by_fold) != 9 or set(by_fold.keys()) != set(range(9)):
        missing = [i for i in range(9) if i not in by_fold]
        raise RuntimeError(
            f"Could not find weights for folds {missing} under {source}. "
            "Expected flat best_model_v2.9_*_fold_k.pt files and/or fold_k/best.pt for each k in 0..8."
        )

    return sorted(by_fold.items(), key=lambda t: t[0])


def discover_best_files(source: Path) -> list[tuple[int, Path]]:
    """Legacy path discovery via rglob + fold inference (may duplicate folds if both flat and nested exist)."""
    candidates: list[tuple[int, Path]] = []
    for fp in sorted(source.rglob("*.pt")):
        name_lower = fp.name.lower()
        if name_lower not in {"best.pt", "best_model.pt"} and not name_lower.startswith("best"):
            continue
        inferred = _infer_fold_index(fp)
        if inferred is None:
            continue
        candidates.append((inferred, fp))

    if len({k for k, _ in candidates}) == len(candidates) and len(candidates) == 9:
        return sorted(candidates, key=lambda t: t[0])

    raise RuntimeError(
        f"Could not reliably map 9 folds under {source}. Found {len(candidates)} inferable best*.pt paths "
        f"({len({k for k, _ in candidates})} unique folds). "
        "If you have both fold_k/best.pt and flat best_model_*.pt files, use the updated prepare script "
        "that merges them, or remove duplicates."
    )


def link_or_copy(src: Path, dst: Path, *, use_symlink: bool) -> None:
    """Copy or symlink src → dst. If they are the same file, no-op (safe when --source == --dest)."""
    src_r = src.resolve()
    dst_r = dst.resolve()
    if src_r == dst_r:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink and os.name != "nt":
        os.symlink(src_r, dst_r)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize UCAPS v2.9 fold checkpoints for local evaluation.")
    p.add_argument("--source", type=Path, required=True, help="Synced Drive folder (contains fold best.pt files).")
    p.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory (e.g. checkpoints_v2.9/v2.9_20260221_014705/).",
    )
    p.add_argument(
        "--ckpt-kind",
        choices=("task2", "task1", "combined"),
        default="task2",
        help="Which evaluator checkpoint variant to write (default: task2, manuscript ensemble).",
    )
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy on POSIX systems.")
    args = p.parse_args()

    source_dir = args.source.resolve()
    dest_dir = args.dest.resolve()
    pairs = discover_normalized(source_dir, str(args.ckpt_kind))
    if pairs is None:
        try:
            pairs = collect_fold_pairs(source_dir, str(args.ckpt_kind))
        except RuntimeError:
            pairs = discover_best_files(source_dir)
    prefix = _flat_prefix(str(args.ckpt_kind))

    written = []
    skipped_same = 0
    for fold_idx, fp in pairs:
        dst = args.dest / f"{prefix}{fold_idx}.pt"
        before = fp.resolve() == dst.resolve()
        link_or_copy(fp, dst, use_symlink=bool(args.symlink))
        if before:
            skipped_same += 1
        else:
            written.append(dst)

    if skipped_same == len(pairs) and not written:
        print(f"OK: all {len(pairs)} checkpoints already use standard names under {dest_dir}. Nothing to copy.")
        return 0

    print(f"Wrote {len(written)} checkpoints to {dest_dir}:")
    for w in written:
        print(f"  - {w.name}")
    if skipped_same:
        print(f"(Skipped {skipped_same} same-path files already in target layout.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
