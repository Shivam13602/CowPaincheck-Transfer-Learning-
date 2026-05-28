#!/usr/bin/env python3
"""
UCAPS v2.9 nine-fold zero-shot inference on Holstein/Jersey face sequences (10s / 240 frames -> 32 @ 112).

Requires normalized checkpoints (see prepare_ucaps_checkpoints.py) and completed_manifest.csv from
cow_face_sequences_10s_250.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch

from holstein_v29_dataset import load_holstein_bundle
from ucaps_v29_eval_loader import load_evaluate_test_cli_module

_eval_cli = load_evaluate_test_cli_module()
_ckpt_path = _eval_cli._ckpt_path
_infer_available_folds = _eval_cli._infer_available_folds
_load_v2_9_module = _eval_cli._load_v2_9_module
_predict_one_fold = _eval_cli._predict_one_fold


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _mean_ensemble(
    mod: Any,
    *,
    sequences: List[dict],
    sequence_dir: Path,
    checkpoint_dir: Path,
    ckpt_kind: str,
    device: torch.device,
    batch_size: int | None,
    num_workers: int,
    max_batches: int | None,
) -> dict[str, Any]:
    fold_indices = sorted(_infer_available_folds(checkpoint_dir, ckpt_kind))
    if not fold_indices:
        raise RuntimeError(
            f"No checkpoints matching kind={ckpt_kind!r} in {checkpoint_dir}. "
            "Sync Drive weights and run prepare_ucaps_checkpoints.py."
        )

    per_fold = []
    for fold_idx in fold_indices:
        ckpt_path = _ckpt_path(checkpoint_dir, fold_idx, ckpt_kind)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold_idx}: {ckpt_path}")
        pred = _predict_one_fold(
            mod=mod,
            ckpt_path=ckpt_path,
            sequences=sequences,
            sequence_dir=sequence_dir,
            device=device,
            batch_size_override=batch_size,
            num_workers_override=num_workers,
            max_batches=max_batches,
        )
        pred["fold"] = int(fold_idx)
        per_fold.append(pred)

    pain_stack = np.stack([r["pain_logits"] for r in per_fold], axis=0)
    t2_stack = np.stack([r["task2_logits"] for r in per_fold], axis=0)
    pain_mean = pain_stack.mean(axis=0)
    t2_mean = t2_stack.mean(axis=0)

    pain_prob = 1.0 / (1.0 + np.exp(-pain_mean))
    t2_prob = _softmax(t2_mean.astype(np.float64)).astype(np.float32)

    clip_ids = per_fold[0]["clip_ids"]
    animals = per_fold[0]["animals"]

    return {
        "fold_indices": fold_indices,
        "pain_logits": pain_mean,
        "pain_prob": pain_prob,
        "task2_logits": t2_mean,
        "task2_prob": t2_prob,
        "clip_ids": clip_ids,
        "animals": animals,
    }


def _summarize(df: pd.DataFrame, col: str, prob_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col):
        rows.append(
            {
                col: key,
                "n": len(g),
                "pain_prob_mean": float(g[prob_col].mean()),
                "pain_prob_std": float(g[prob_col].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(col)


def main() -> int:
    p = argparse.ArgumentParser(description="Holstein/Jersey zero-shot UCAPS v2.9 ensemble inference.")
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("../cow_face_sequences_10s_250/completed_manifest.csv"),
        help="completed_manifest.csv from sequence extraction.",
    )
    p.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("../cow_face_sequences_10s_250"),
        help="Root folder containing sequences/ subfolders.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints_v2.9/v2.9_20260221_014705"),
        help="Directory with nine fold checkpoints (after prepare_ucaps_checkpoints.py).",
    )
    p.add_argument("--ckpt-kind", choices=("task2", "task1", "combined"), default="task2")
    p.add_argument("--train-py", type=Path, default=None, help="Optional path to v2.9_training_classification.py.")
    p.add_argument("--out-dir", type=Path, default=Path("holstein_zero_shot_outputs"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    args = p.parse_args()

    manifest = args.manifest_csv.resolve()
    seq_root = args.sequence_root.resolve()
    ckpt_dir = args.checkpoint_dir.resolve()

    bundle = load_holstein_bundle(manifest, seq_root)
    extra_dirs = [Path(__file__).resolve().parent, ckpt_dir.parent, Path.cwd()]
    train_py = str(args.train_py) if args.train_py else None
    mod = _load_v2_9_module(train_py, search_dirs=extra_dirs)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ens = _mean_ensemble(
        mod,
        sequences=bundle.sequences,
        sequence_dir=seq_root,
        checkpoint_dir=ckpt_dir,
        ckpt_kind=str(args.ckpt_kind),
        device=device,
        batch_size=args.batch_size,
        num_workers=int(args.num_workers),
        max_batches=args.max_batches,
    )

    meta = bundle.metadata
    rows = []
    for i, m in enumerate(meta):
        rows.append(
            {
                **m,
                "pain_logit": float(ens["pain_logits"][i]),
                "pain_prob": float(ens["pain_prob"][i]),
                "task2_logit_0": float(ens["task2_logits"][i, 0]),
                "task2_logit_1": float(ens["task2_logits"][i, 1]),
                "task2_logit_2": float(ens["task2_logits"][i, 2]),
                "task2_prob_no_pain": float(ens["task2_prob"][i, 0]),
                "task2_prob_acute": float(ens["task2_prob"][i, 1]),
                "task2_prob_residual": float(ens["task2_prob"][i, 2]),
                "task2_entropy": float(
                    -np.sum(ens["task2_prob"][i] * np.log(np.clip(ens["task2_prob"][i], 1e-12, None)))
                ),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"holstein_zero_shot_predictions_{ts}.csv"
    df.to_csv(csv_path, index=False)

    summ_video = _summarize(df, "video_health_status", "pain_prob")
    summ_cow = _summarize(df, "cow_health_status", "pain_prob")
    summ_cond = _summarize(df, "health_condition", "pain_prob")
    summ_ds = _summarize(df, "dataset_root", "pain_prob")

    summ_video.to_csv(out_dir / f"group_summary_video_health_{ts}.csv", index=False)
    summ_cow.to_csv(out_dir / f"group_summary_cow_health_{ts}.csv", index=False)
    summ_cond.to_csv(out_dir / f"group_summary_condition_{ts}.csv", index=False)
    summ_ds.to_csv(out_dir / f"group_summary_dataset_{ts}.csv", index=False)

    report_path = out_dir / "holstein_zero_shot_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Holstein/Jersey zero-shot (UCAPS v2.9)\n\n")
        f.write(f"- Generated (UTC): `{ts}`\n")
        f.write(f"- Checkpoints: `{ckpt_dir}` (`ckpt_kind={args.ckpt_kind}`)\n")
        f.write(f"- Folds used: `{json.dumps(ens['fold_indices'])}`\n")
        f.write(f"- Sequences: {len(df)}\n")
        f.write(f"- Predictions CSV: `{csv_path.name}`\n\n")
        f.write("## Pain probability by video-level health context\n\n")
        f.write(_markdown_table(summ_video))
        f.write("\n\n## Pain probability by cow-level health label\n\n")
        f.write(_markdown_table(summ_cow))
        f.write("\n\n## Pain probability by disease/proxy condition label\n\n")
        f.write(_markdown_table(summ_cond))
        f.write("\n\n## Pain probability by dataset root\n\n")
        f.write(_markdown_table(summ_ds))
        f.write("\n")

    meta_json = {
        "utc": ts,
        "checkpoint_dir": str(ckpt_dir),
        "ckpt_kind": args.ckpt_kind,
        "folds": ens["fold_indices"],
        "manifest_csv": str(manifest),
        "sequence_root": str(seq_root),
        "predictions_csv": str(csv_path),
    }
    (out_dir / f"holstein_zero_shot_run_{ts}.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")

    print(f"Wrote predictions: {csv_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
