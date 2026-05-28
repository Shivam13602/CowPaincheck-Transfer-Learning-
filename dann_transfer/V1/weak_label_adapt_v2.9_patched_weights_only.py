#!/usr/bin/env python3
"""
Cow-held-out weak-label adaptation for UCAPS v2.9 on Holstein/Jersey sequences.

This is a Task1-only experiment. It treats video-level health context as a noisy
binary proxy:

    Healthy   -> no-pain proxy (0, UCAPS moment M0)
    Unhealthy -> pain proxy    (1, UCAPS moment M2)

The split is cow-held-out: 4 cows are reserved as a final test set, then the
remaining cows are partitioned into inner folds with 4 validation cows per fold.
Do not describe the output as validated pain detection until veterinary pain
scores exist.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from holstein_v29_dataset import load_holstein_bundle
from ucaps_v29_eval_loader import load_evaluate_test_cli_module


VALID_LABELS = {"Healthy": 0, "Unhealthy": 1}
LABEL_TO_MOMENT = {"Healthy": "M0", "Unhealthy": "M2"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune/evaluate UCAPS v2.9 Task1 on Holstein/Jersey weak labels "
            "with cow-held-out splits. Healthy=0, Unhealthy=1."
        )
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("../cow_face_sequences_10s_250/completed_manifest.csv"),
        help="completed_manifest.csv from cow_face_sequences_10s_250.",
    )
    p.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("../cow_face_sequences_10s_250"),
        help="Root containing the sequences/ folder.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional UCAPS v2.9 checkpoint folder used for pretrained initialization.",
    )
    p.add_argument("--ckpt-kind", choices=("task2", "task1", "combined"), default="task2")
    p.add_argument("--init-fold", type=int, default=0, help="Source UCAPS fold checkpoint used to initialize every weak-label fold.")
    p.add_argument("--ssl-checkpoint-dir", type=Path, default=None, help="Optional output dir from ssl_pretrain_holstein_v2.9.py for fold-specific initialization.")
    p.add_argument("--ssl-checkpoint-pattern", type=str, default="fold_{fold}/best_ssl_simsiam.pt")
    p.add_argument("--from-scratch", action="store_true", help="Train without UCAPS checkpoint initialization.")
    p.add_argument("--train-py", type=Path, default=None, help="Optional path to v2.9_training_classification.py.")
    p.add_argument("--out-dir", type=Path, default=Path("holstein_weak_label_cv_outputs"))
    p.add_argument(
        "--label-column",
        choices=("video_health_status", "cow_health_status"),
        default="video_health_status",
        help="Weak binary target column. Recommended: video_health_status.",
    )
    p.add_argument("--test-cows", type=int, default=4, help="Number of cows held out for final test.")
    p.add_argument(
        "--test-cow-ids",
        type=str,
        default=None,
        help="Optional comma-separated cow IDs for the final test set. Overrides --test-cows.",
    )
    p.add_argument("--val-cows-per-fold", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--task2-mode", choices=("3class", "4class"), default=None)
    p.add_argument("--freeze-cnn", action="store_true", help="Freeze CNN feature extractor; train LSTM/attention/heads.")
    p.add_argument("--no-aug", action="store_true", help="Disable training-time augmentations.")
    p.add_argument("--no-stratified-sampler", action="store_true", help="Disable moment/label-balanced sampling.")
    p.add_argument("--use-moment-weighting", action="store_true", help="Use UCAPS moment weights; off by default for weak labels.")
    p.add_argument("--task1-loss", choices=("bce", "focal", "gce"), default="bce", help="Task1 weak-proxy loss: BCE, focal loss, or generalized cross entropy.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss (Lin et al., ICCV 2017).")
    p.add_argument("--gce-q", type=float, default=0.7, help="q for generalized cross entropy (Zhang and Sabuncu, NeurIPS 2018).")
    p.add_argument("--class-balanced", action="store_true", help="Use effective-number class weights (Cui et al., CVPR 2019).")
    p.add_argument("--class-balanced-beta", type=float, default=0.999, help="Beta for effective-number class weighting.")
    p.add_argument("--select-metric", choices=("auc", "f1_opt", "f1"), default="auc")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Only build and write split audit; do not train.")
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _load_train_module(train_py: Path | None, checkpoint_dir: Path | None) -> Any:
    eval_cli = load_evaluate_test_cli_module()
    extra_dirs = [Path(__file__).resolve().parent, Path.cwd()]
    if checkpoint_dir is not None:
        extra_dirs.extend([checkpoint_dir, checkpoint_dir.parent])
    train_py_s = str(train_py.resolve()) if train_py else None
    return eval_cli._load_v2_9_module(train_py_s, search_dirs=extra_dirs)


def _checkpoint_path(checkpoint_dir: Path, fold_idx: int, ckpt_kind: str) -> Path:
    if ckpt_kind == "combined":
        return checkpoint_dir / f"best_model_v2.9_fold_{fold_idx}.pt"
    if ckpt_kind == "task1":
        return checkpoint_dir / f"best_model_v2.9_task1_fold_{fold_idx}.pt"
    if ckpt_kind == "task2":
        return checkpoint_dir / f"best_model_v2.9_task2_fold_{fold_idx}.pt"
    raise ValueError(f"Unknown ckpt_kind={ckpt_kind!r}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        alt = path.parent / f"fold_{path.stem.split('_')[-1]}" / "best.pt"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def _cfg_from_checkpoint_or_default(mod: Any, ckpt: dict[str, Any] | None) -> Any:
    cfg = mod.Config()
    cfg_dict = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(cfg_dict, dict):
        for key, value in cfg_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def _apply_cfg_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    cfg.task1_weight = 1.0
    cfg.task2_weight = 0.0
    cfg.num_epochs = int(args.num_epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.num_workers = int(args.num_workers)
    cfg.learning_rate = float(args.learning_rate)
    if args.weight_decay is not None and hasattr(cfg, "weight_decay"):
        cfg.weight_decay = float(args.weight_decay)
    if args.max_frames is not None:
        cfg.max_frames = int(args.max_frames)
    if args.resolution is not None:
        cfg.resolution = tuple(int(x) for x in args.resolution)
    if args.task2_mode is not None:
        cfg.task2_mode = str(args.task2_mode)
    if hasattr(cfg, "freeze_cnn"):
        cfg.freeze_cnn = bool(args.freeze_cnn)
    if hasattr(cfg, "use_augmentations") and bool(args.no_aug):
        cfg.use_augmentations = False
    if hasattr(cfg, "use_stratified_sampler"):
        cfg.use_stratified_sampler = not bool(args.no_stratified_sampler)
    if hasattr(cfg, "use_moment_weighting"):
        cfg.use_moment_weighting = bool(args.use_moment_weighting)
    return cfg


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _seed_everything(mod: Any, seed: int) -> None:
    if hasattr(mod, "seed_everything"):
        mod.seed_everything(int(seed))
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _prepare_records(manifest_csv: Path, sequence_root: Path, label_column: str) -> list[dict[str, Any]]:
    bundle = load_holstein_bundle(manifest_csv, sequence_root)
    records: list[dict[str, Any]] = []
    skipped: list[tuple[int, str]] = []
    for seq, meta in zip(bundle.sequences, bundle.metadata):
        label_text = str(meta.get(label_column, "")).strip()
        if label_text not in VALID_LABELS:
            skipped.append((int(meta.get("sequence_index", -1)), label_text))
            continue
        seq2 = deepcopy(seq)
        seq2["moment"] = LABEL_TO_MOMENT[label_text]
        target = int(VALID_LABELS[label_text])
        meta2 = deepcopy(meta)
        meta2["weak_label_column"] = label_column
        meta2["weak_label_text"] = label_text
        meta2["weak_label_target"] = target
        records.append(
            {
                "seq": seq2,
                "meta": meta2,
                "target": target,
                "cow_id": str(meta2["cow_id"]),
                "sequence_index": int(meta2["sequence_index"]),
            }
        )
    if skipped:
        print(f"WARNING: skipped {len(skipped)} rows with labels outside {sorted(VALID_LABELS)} in {label_column}.")
    if not records:
        raise RuntimeError(f"No usable rows found in {manifest_csv} for label column {label_column!r}.")
    return records


def _cow_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [{"cow_id": r["cow_id"], "target": int(r["target"])} for r in records]
    df = pd.DataFrame(rows)
    out = (
        df.groupby("cow_id")
        .agg(n_sequences=("target", "size"), positives=("target", "sum"), target_mean=("target", "mean"))
        .reset_index()
    )
    out["negatives"] = out["n_sequences"] - out["positives"]
    out["majority_target"] = (out["target_mean"] >= 0.5).astype(int)
    return out.sort_values(["majority_target", "cow_id"]).reset_index(drop=True)


def _parse_cow_ids(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _balanced_test_cows(cows: pd.DataFrame, n: int, seed: int, explicit: list[str] | None) -> list[str]:
    all_cows = set(cows["cow_id"].astype(str))
    if explicit is not None:
        missing = sorted(set(explicit) - all_cows)
        if missing:
            raise ValueError(f"--test-cow-ids contains cows not in manifest: {missing}")
        if len(explicit) != n:
            print(f"INFO: --test-cow-ids has {len(explicit)} cows; overriding --test-cows={n}.")
        return list(dict.fromkeys(explicit))

    if n <= 0 or n >= len(cows):
        raise ValueError(f"--test-cows must be in [1, {len(cows) - 1}], got {n}.")

    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for label in (0, 1):
        items = cows.loc[cows["majority_target"] == label, "cow_id"].astype(str).tolist()
        items = sorted(items)
        rng.shuffle(items)
        by_label[label] = items

    n_pos = n // 2
    n_neg = n - n_pos
    wanted = {0: n_neg, 1: n_pos}
    selected: list[str] = []
    for label in (0, 1):
        take = min(wanted[label], len(by_label[label]))
        selected.extend(by_label[label][:take])
        by_label[label] = by_label[label][take:]

    if len(selected) < n:
        rest = by_label[0] + by_label[1]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    return selected


def _make_validation_folds(cows: pd.DataFrame, train_pool_cows: list[str], val_cows_per_fold: int, seed: int) -> list[list[str]]:
    if val_cows_per_fold <= 0:
        raise ValueError("--val-cows-per-fold must be > 0.")
    if len(train_pool_cows) % val_cows_per_fold != 0:
        raise ValueError(
            f"{len(train_pool_cows)} train-pool cows is not divisible by --val-cows-per-fold={val_cows_per_fold}. "
            "For this dataset, 32 total cows - 4 test cows = 28, which gives 7 folds of 4 cows."
        )

    n_folds = len(train_pool_cows) // val_cows_per_fold
    label_by_cow = {
        str(row.cow_id): int(row.majority_target)
        for row in cows[cows["cow_id"].astype(str).isin(set(train_pool_cows))].itertuples(index=False)
    }
    rng = random.Random(seed + 17)
    folds: list[list[str]] = [[] for _ in range(n_folds)]

    for label in (0, 1):
        label_cows = [c for c in train_pool_cows if label_by_cow[c] == label]
        label_cows = sorted(label_cows)
        rng.shuffle(label_cows)
        for cow in label_cows:
            candidates = [i for i in range(n_folds) if len(folds[i]) < val_cows_per_fold]
            if not candidates:
                raise RuntimeError("Internal split error: all validation folds are already full.")
            best_i = min(
                candidates,
                key=lambda i: (
                    sum(1 for c in folds[i] if label_by_cow[c] == label),
                    len(folds[i]),
                    i,
                ),
            )
            folds[best_i].append(cow)

    for i, fold in enumerate(folds):
        if len(fold) != val_cows_per_fold:
            raise RuntimeError(f"Fold {i} has {len(fold)} validation cows, expected {val_cows_per_fold}.")
    return [sorted(fold) for fold in folds]


def _records_for_cows(records: list[dict[str, Any]], cows: Iterable[str]) -> list[dict[str, Any]]:
    cow_set = {str(c) for c in cows}
    return [r for r in records if r["cow_id"] in cow_set]


def _split_summary(records: list[dict[str, Any]], cows: Iterable[str]) -> dict[str, Any]:
    cow_list = sorted(str(c) for c in cows)
    subset = _records_for_cows(records, cow_list)
    targets = np.array([r["target"] for r in subset], dtype=np.int64)
    return {
        "cows": cow_list,
        "n_cows": len(cow_list),
        "n_sequences": int(len(subset)),
        "healthy_sequences": int((targets == 0).sum()) if len(targets) else 0,
        "unhealthy_sequences": int((targets == 1).sum()) if len(targets) else 0,
    }


def _build_split_plan(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cows = _cow_table(records)
    explicit_test = _parse_cow_ids(args.test_cow_ids)
    test_cows = _balanced_test_cows(cows, int(args.test_cows), int(args.seed), explicit_test)
    all_cows = sorted(cows["cow_id"].astype(str).tolist())
    train_pool_cows = sorted(c for c in all_cows if c not in set(test_cows))
    val_folds = _make_validation_folds(cows, train_pool_cows, int(args.val_cows_per_fold), int(args.seed))

    folds = []
    for fold_idx, val_cows in enumerate(val_folds):
        train_cows = sorted(c for c in train_pool_cows if c not in set(val_cows))
        folds.append(
            {
                "fold": fold_idx,
                "train": _split_summary(records, train_cows),
                "val": _split_summary(records, val_cows),
            }
        )

    split = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "seed": int(args.seed),
        "label_column": str(args.label_column),
        "label_mapping": {"Healthy": 0, "Unhealthy": 1},
        "n_sequences": len(records),
        "n_cows": len(all_cows),
        "test": _split_summary(records, test_cows),
        "train_pool": _split_summary(records, train_pool_cows),
        "folds": folds,
        "cow_table": cows.to_dict(orient="records"),
    }
    return split


def _print_split_plan(split: dict[str, Any]) -> None:
    print("=" * 80)
    print("HOLSTEIN WEAK-LABEL COW SPLIT")
    print(f"label_column={split['label_column']} | seed={split['seed']}")
    print(
        "test: "
        f"cows={split['test']['cows']} | "
        f"seq={split['test']['n_sequences']} | "
        f"H={split['test']['healthy_sequences']} U={split['test']['unhealthy_sequences']}"
    )
    print(
        "train_pool: "
        f"n_cows={split['train_pool']['n_cows']} | "
        f"seq={split['train_pool']['n_sequences']} | "
        f"H={split['train_pool']['healthy_sequences']} U={split['train_pool']['unhealthy_sequences']}"
    )
    for fold in split["folds"]:
        tr = fold["train"]
        va = fold["val"]
        print(
            f"fold {fold['fold']}: "
            f"train_cows={tr['n_cows']} train_seq={tr['n_sequences']} H={tr['healthy_sequences']} U={tr['unhealthy_sequences']} | "
            f"val_cows={va['cows']} val_seq={va['n_sequences']} H={va['healthy_sequences']} U={va['unhealthy_sequences']}"
        )
    print("=" * 80)


def _best_f1_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (scores >= thr).astype(np.int64)
        f1 = f1_score(targets, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def _safe_auc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return None
    return float(roc_auc_score(targets, scores))


def _binary_metrics(targets: np.ndarray, scores: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(targets) == 0:
        return {
            "n": 0,
            "positives": 0,
            "negatives": 0,
            "threshold": float(threshold),
            "accuracy": None,
            "balanced_accuracy": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "auc": None,
            "best_threshold": None,
            "f1_opt": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    pred = (scores >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(targets, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    best_thr, best_f1 = _best_f1_threshold(scores, targets)
    return {
        "n": int(len(targets)),
        "positives": int(targets.sum()),
        "negatives": int(len(targets) - targets.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, pred)) if len(np.unique(targets)) > 1 else None,
        "f1": float(f1_score(targets, pred, zero_division=0)),
        "precision": float(precision_score(targets, pred, zero_division=0)),
        "recall": float(recall_score(targets, pred, zero_division=0)),
        "auc": _safe_auc(targets, scores),
        "best_threshold": float(best_thr),
        "f1_opt": float(best_f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _fit_temperature_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_t = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    targets_t = torch.as_tensor(np.asarray(targets, dtype=np.float32))
    if logits_t.numel() == 0 or len(torch.unique(targets_t)) < 2:
        return 1.0
    log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return 1.0
    return float(torch.exp(log_temperature).detach().clamp(0.05, 10.0).item())


def _probs_from_logits(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-logits))


def _add_calibrated_probabilities(pred_df: pd.DataFrame, *, temperature: float) -> pd.DataFrame:
    out = pred_df.copy()
    if "pain_logit" not in out.columns:
        return out
    out["temperature"] = float(temperature)
    out["pain_prob_calibrated"] = _probs_from_logits(out["pain_logit"].values, temperature=float(temperature))
    return out


def _calibrated_metrics(pred_df: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, Any]:
    if pred_df.empty or "pain_prob_calibrated" not in pred_df.columns:
        return _binary_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.float64), threshold=threshold)
    return _binary_metrics(pred_df["target"].values, pred_df["pain_prob_calibrated"].values, threshold=threshold)


def _calibrated_cow_metrics(pred_df: pd.DataFrame, *, threshold: float = 0.5) -> tuple[dict[str, Any], pd.DataFrame]:
    if pred_df.empty or "pain_prob_calibrated" not in pred_df.columns:
        return _cow_level_metrics(pred_df, threshold=threshold)
    tmp = pred_df.copy()
    tmp["pain_prob"] = tmp["pain_prob_calibrated"]
    return _cow_level_metrics(tmp, threshold=threshold)


def _cow_level_metrics(pred_df: pd.DataFrame, *, threshold: float = 0.5) -> tuple[dict[str, Any], pd.DataFrame]:
    if pred_df.empty:
        return _binary_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.float64), threshold=threshold), pd.DataFrame()
    cow_df = (
        pred_df.groupby("cow_id")
        .agg(
            n_sequences=("target", "size"),
            target_mean=("target", "mean"),
            pain_prob=("pain_prob", "mean"),
            positives=("target", "sum"),
        )
        .reset_index()
    )
    cow_df["target"] = (cow_df["target_mean"] >= 0.5).astype(np.int64)
    metrics = _binary_metrics(cow_df["target"].values, cow_df["pain_prob"].values, threshold=threshold)
    return metrics, cow_df


def _metric_for_selection(metrics: dict[str, Any], select_metric: str) -> float:
    order = [select_metric, "f1_opt", "f1", "accuracy"]
    for key in order:
        value = metrics.get(key)
        if value is None:
            continue
        value_f = float(value)
        if not math.isnan(value_f):
            return value_f
    return float("-inf")


def _make_loader(mod: Any, records: list[dict[str, Any]], sequence_root: Path, cfg: Any, *, augment: bool, shuffle: bool) -> DataLoader:
    DatasetCls = getattr(mod, "FacialPainDataset_v2_9", None) or getattr(mod, "FacialPainDataset_v2_8")
    ds = DatasetCls([r["seq"] for r in records], sequence_root, cfg, augment=augment, global_cache={})
    loader_extra: dict[str, Any] = {}
    if int(cfg.num_workers) > 0:
        loader_extra["prefetch_factor"] = 2
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        **loader_extra,
    )


def _make_train_loader(mod: Any, records: list[dict[str, Any]], sequence_root: Path, cfg: Any) -> DataLoader:
    DatasetCls = getattr(mod, "FacialPainDataset_v2_9", None) or getattr(mod, "FacialPainDataset_v2_8")
    seqs = [r["seq"] for r in records]
    ds = DatasetCls(seqs, sequence_root, cfg, augment=True, global_cache={})
    sampler = None
    if bool(getattr(cfg, "use_stratified_sampler", False)) and hasattr(mod, "create_stratified_sampler"):
        sampler = mod.create_stratified_sampler(seqs)
    loader_extra: dict[str, Any] = {}
    if int(cfg.num_workers) > 0:
        loader_extra["prefetch_factor"] = 2
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        **loader_extra,
    )


@torch.no_grad()
def _predict_records(
    model: nn.Module,
    loader: DataLoader,
    records: list[dict[str, Any]],
    *,
    device: torch.device,
    split: str,
    fold: int | str,
    max_batches: int | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    offset = 0
    for batch_idx, (x, y, meta) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        out, _ = model(x)
        logits = out["pain_logits"].detach().cpu().numpy().astype(np.float64)
        probs = 1.0 / (1.0 + np.exp(-logits))
        targets = y["pain_binary"].detach().cpu().numpy().astype(np.int64)
        for i in range(len(logits)):
            rec = records[offset + i]
            m = rec["meta"]
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "sequence_index": int(m["sequence_index"]),
                    "cow_id": str(m["cow_id"]),
                    "target": int(targets[i]),
                    "target_label": str(m["weak_label_text"]),
                    "health_condition": str(m.get("health_condition", "")),
                    "dataset_root": str(m.get("dataset_root", "")),
                    "relative_path": str(m.get("relative_path", "")),
                    "detected_frames": int(m.get("detected_frames", 0)),
                    "filled_frames": int(m.get("filled_frames", 0)),
                    "mean_detection_confidence": m.get("mean_detection_confidence", None),
                    "pain_logit": float(logits[i]),
                    "pain_prob": float(probs[i]),
                }
            )
        offset += int(x.size(0))
    return pd.DataFrame(rows)


def _ssl_checkpoint_path(args: argparse.Namespace, fold_idx: int) -> Path | None:
    ssl_dir = getattr(args, "ssl_checkpoint_dir", None)
    if ssl_dir is None:
        return None
    pattern = str(getattr(args, "ssl_checkpoint_pattern", "fold_{fold}/best_ssl_simsiam.pt"))
    return (Path(ssl_dir) / pattern.format(fold=int(fold_idx))).resolve()


def _make_model(
    mod: Any,
    cfg: Any,
    device: torch.device,
    init_ckpt: dict[str, Any] | None,
    *,
    freeze_cnn: bool,
    ssl_checkpoint_path: Path | None = None,
) -> nn.Module:
    ModelCls = getattr(mod, "TemporalPainModel_v2_9", None) or getattr(mod, "TemporalPainModel_v2_8")
    model = ModelCls(cfg).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
    if ssl_checkpoint_path is not None:
        if not ssl_checkpoint_path.exists():
            raise FileNotFoundError(f"SSL checkpoint not found for weak-label initialization: {ssl_checkpoint_path}")
        ssl_ckpt = torch.load(ssl_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ssl_ckpt["model_state_dict"], strict=True)
    if freeze_cnn and hasattr(model, "cnn"):
        _set_requires_grad(model.cnn, False)
    return model


def _task2_counts(mod: Any, records: list[dict[str, Any]], cfg: Any) -> list[int]:
    k = int(mod.task2_num_classes(cfg.task2_mode))
    labels = [int(mod.moment_to_task2(r["seq"].get("moment", "M0"), task2_mode=cfg.task2_mode)) for r in records]
    return [int(sum(1 for y in labels if y == i)) for i in range(k)]


def _pos_weight(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor | None:
    labels = np.array([r["target"] for r in records], dtype=np.int64)
    pos = float(labels.sum())
    neg = float(len(labels) - labels.sum())
    if pos > 0 and neg > 0:
        return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    return None


def _effective_num_weights(labels: np.ndarray, beta: float, device: torch.device) -> torch.Tensor | None:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.array([(labels == 0).sum(), (labels == 1).sum()], dtype=np.float64)
    if np.any(counts <= 0):
        return None
    beta = float(np.clip(beta, 0.0, 0.999999))
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


class Task1ProxyLoss(nn.Module):
    def __init__(
        self,
        *,
        loss_type: str,
        pos_weight: torch.Tensor | None,
        class_weights: torch.Tensor | None,
        focal_gamma: float,
        gce_q: float,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type)
        self.focal_gamma = float(focal_gamma)
        self.gce_q = float(gce_q)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone())
        else:
            self.pos_weight = None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().clone())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.loss_type == "gce":
            probs = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
            loss = (1.0 - pt.pow(self.gce_q)) / max(self.gce_q, 1e-6)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
            if self.loss_type == "focal":
                probs = torch.sigmoid(logits)
                pt = torch.where(targets > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
                loss = bce * torch.pow(1.0 - pt, self.focal_gamma)
            else:
                loss = bce
        if self.class_weights is not None:
            weights = torch.where(targets > 0.5, self.class_weights[1], self.class_weights[0])
            loss = loss * weights
        return loss


def _train_one_fold(
    *,
    mod: Any,
    base_cfg: Any,
    args: argparse.Namespace,
    fold: dict[str, Any],
    records: list[dict[str, Any]],
    sequence_root: Path,
    device: torch.device,
    init_ckpt: dict[str, Any] | None,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    fold_idx = int(fold["fold"])
    _seed_everything(mod, int(args.seed) + fold_idx)
    cfg = deepcopy(base_cfg)
    train_records = _records_for_cows(records, fold["train"]["cows"])
    val_records = _records_for_cows(records, fold["val"]["cows"])

    ssl_path = _ssl_checkpoint_path(args, fold_idx)
    model = _make_model(mod, cfg, device, init_ckpt, freeze_cnn=bool(args.freeze_cnn), ssl_checkpoint_path=ssl_path)
    train_loader = _make_train_loader(mod, train_records, sequence_root, cfg)
    val_loader = _make_loader(mod, val_records, sequence_root, cfg, augment=False, shuffle=False)

    pos_weight = None if bool(args.class_balanced) else _pos_weight(train_records, device)
    class_weights = (
        _effective_num_weights(np.array([r["target"] for r in train_records], dtype=np.int64), float(args.class_balanced_beta), device)
        if bool(args.class_balanced)
        else None
    )
    loss_task1 = Task1ProxyLoss(
        loss_type=str(args.task1_loss),
        pos_weight=pos_weight,
        class_weights=class_weights,
        focal_gamma=float(args.focal_gamma),
        gce_q=float(args.gce_q),
    )
    t2_counts = _task2_counts(mod, train_records, cfg)
    loss_task2, task2_ce_weights, _drw_weights = mod._make_task2_loss(cfg, t2_counts=t2_counts, device=device)
    if loss_task2 is not None:
        loss_task2 = loss_task2.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=float(getattr(cfg, "min_lr", 1e-6)))
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not bool(args.no_amp)) else None

    fold_dir = run_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_weak_task1.pt"
    history_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_summary: dict[str, Any] | None = None
    best_pred_df = pd.DataFrame()

    for epoch in range(int(cfg.num_epochs)):
        train_loss = mod.train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            scaler=scaler,
            ema=None,
            loss_task1=loss_task1,
            loss_task2=loss_task2,
            task2_ce_weights=task2_ce_weights,
            max_batches=args.max_train_batches,
        )
        val_pred = _predict_records(
            model,
            val_loader,
            val_records,
            device=device,
            split="val",
            fold=fold_idx,
            max_batches=args.max_val_batches,
        )
        val_metrics = _binary_metrics(val_pred["target"].values, val_pred["pain_prob"].values)
        cow_metrics, _cow_df = _cow_level_metrics(val_pred)
        temperature = _fit_temperature_from_logits(val_pred["pain_logit"].values, val_pred["target"].values)
        val_pred_calibrated = _add_calibrated_probabilities(val_pred, temperature=temperature)
        val_calibrated_metrics = _calibrated_metrics(val_pred_calibrated)
        val_calibrated_cow_metrics, _val_calibrated_cow_df = _calibrated_cow_metrics(val_pred_calibrated)
        score = _metric_for_selection(val_metrics, str(args.select_metric))
        scheduler.step(score)
        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "task1_loss": str(args.task1_loss),
            "class_balanced": bool(args.class_balanced),
            "temperature": float(temperature),
            "selection_metric": str(args.select_metric),
            "selection_score": float(score),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_cow_{k}": v for k, v in cow_metrics.items()},
            **{f"val_calibrated_{k}": v for k, v in val_calibrated_metrics.items()},
            **{f"val_calibrated_cow_{k}": v for k, v in val_calibrated_cow_metrics.items()},
        }
        history_rows.append(row)
        print(
            f"fold {fold_idx} epoch {epoch + 1}/{cfg.num_epochs}: "
            f"loss={train_loss:.4f} auc={val_metrics.get('auc')} "
            f"f1={val_metrics.get('f1'):.3f} f1_opt={val_metrics.get('f1_opt'):.3f} score={score:.3f}"
        )
        if score > best_score:
            best_score = float(score)
            best_summary = {
                "fold": fold_idx,
                "best_epoch": epoch,
                "best_score": float(score),
                "selection_metric": str(args.select_metric),
                "val_metrics": val_metrics,
                "val_cow_metrics": cow_metrics,
                "temperature": float(temperature),
                "val_calibrated_metrics": val_calibrated_metrics,
                "val_calibrated_cow_metrics": val_calibrated_cow_metrics,
                "train": fold["train"],
                "val": fold["val"],
            }
            best_pred_df = val_pred_calibrated.copy()
            torch.save(
                {
                    "version": "v2.9_holstein_weak_label_task1",
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": _cpu_state_dict(model),
                    "cfg": _cfg_to_dict(cfg),
                    "split": fold,
                    "val_metrics": val_metrics,
                    "val_cow_metrics": cow_metrics,
                    "temperature": float(temperature),
                    "val_calibrated_metrics": val_calibrated_metrics,
                    "val_calibrated_cow_metrics": val_calibrated_cow_metrics,
                    "label_column": str(args.label_column),
                    "label_mapping": {"Healthy": 0, "Unhealthy": 1},
                    "ssl_checkpoint_path": str(ssl_path) if ssl_path else None,
                    "args": vars(args),
                },
                best_path,
            )

    pd.DataFrame(history_rows).to_csv(fold_dir / "history.csv", index=False)
    best_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False)
    if best_summary is None:
        raise RuntimeError(f"Fold {fold_idx} did not produce a valid checkpoint.")
    return best_summary, best_pred_df, best_path


def _load_weak_model(mod: Any, ckpt_path: Path, device: torch.device) -> tuple[nn.Module, Any, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_checkpoint_or_default(mod, ckpt)
    ModelCls = getattr(mod, "TemporalPainModel_v2_9", None) or getattr(mod, "TemporalPainModel_v2_8")
    model = ModelCls(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg, ckpt


def _ensemble_test_predictions(
    *,
    mod: Any,
    best_paths: list[Path],
    records: list[dict[str, Any]],
    sequence_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    logit_stack: list[np.ndarray] = []
    first_df: pd.DataFrame | None = None
    for path in best_paths:
        model, cfg, ckpt = _load_weak_model(mod, path, device)
        cfg.batch_size = int(args.batch_size)
        cfg.num_workers = int(args.num_workers)
        loader = _make_loader(mod, records, sequence_root, cfg, augment=False, shuffle=False)
        pred_df = _predict_records(
            model,
            loader,
            records,
            device=device,
            split="test",
            fold=f"ensemble_member_{ckpt['fold']}",
            max_batches=args.max_val_batches,
        )
        logit_stack.append(pred_df["pain_logit"].values.astype(np.float64))
        if first_df is None:
            first_df = pred_df.copy()
    if first_df is None:
        raise RuntimeError("No fold checkpoints available for test ensemble.")
    mean_logits = np.stack(logit_stack, axis=0).mean(axis=0)
    first_df["fold"] = "ensemble"
    first_df["pain_logit"] = mean_logits
    first_df["pain_prob"] = 1.0 / (1.0 + np.exp(-mean_logits))
    return first_df


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    val = summary["val_metrics"]
    cow = summary["val_cow_metrics"]
    return {
        "fold": summary["fold"],
        "best_epoch": summary["best_epoch"],
        "best_score": summary["best_score"],
        "val_cows": ",".join(summary["val"]["cows"]),
        "val_n": val["n"],
        "val_auc": val["auc"],
        "val_f1": val["f1"],
        "val_f1_opt": val["f1_opt"],
        "val_accuracy": val["accuracy"],
        "val_balanced_accuracy": val["balanced_accuracy"],
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_tn": val["tn"],
        "val_fp": val["fp"],
        "val_fn": val["fn"],
        "val_tp": val["tp"],
        "val_cow_auc": cow["auc"],
        "val_cow_f1": cow["f1"],
        "val_cow_f1_opt": cow["f1_opt"],
        "temperature": summary.get("temperature"),
        "val_calibrated_auc": summary.get("val_calibrated_metrics", {}).get("auc"),
        "val_calibrated_f1_opt": summary.get("val_calibrated_metrics", {}).get("f1_opt"),
        "val_calibrated_cow_auc": summary.get("val_calibrated_cow_metrics", {}).get("auc"),
    }


def _write_report(
    *,
    report_path: Path,
    args: argparse.Namespace,
    split: dict[str, Any],
    fold_summary_df: pd.DataFrame,
    test_metrics: dict[str, Any],
    test_cow_metrics: dict[str, Any],
    test_cow_df: pd.DataFrame,
    outputs: dict[str, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Holstein/Jersey Weak-Label Cow-Held-Out CV\n\n")
        f.write(
            "## Metric roles\n\n"
            "- This script **only** evaluates on Holstein `video_health_status` (or selected column) as a **weak proxy**. "
            "There is no UCAPS pain ground truth on the target domain.\n"
            "- **Calibrated** tables use validation-fitted temperature scaling (Guo et al., ICML 2017), separate from raw AUC.\n\n"
        )
        f.write(f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}`\n")
        f.write(f"- Label column: `{args.label_column}` (`Healthy=0`, `Unhealthy=1`)\n")
        f.write(f"- Final test cows: `{json.dumps(split['test']['cows'])}`\n")
        f.write(f"- Inner folds: `{len(split['folds'])}` folds x `{args.val_cows_per_fold}` validation cows\n")
        f.write(f"- Initialization: `{'from scratch' if args.from_scratch else str(args.checkpoint_dir)}`\n")
        f.write(f"- Freeze CNN: `{bool(args.freeze_cnn)}`\n\n")
        f.write(f"- Task1 proxy loss: `{args.task1_loss}` | class-balanced effective-number weighting: `{bool(args.class_balanced)}`\n")
        f.write("- Calibration: validation-fitted temperature scaling is reported separately from raw AUC.\n\n")
        f.write(
            "These labels are weak disease-context proxies, not veterinary pain scores. "
            "Use this as a transfer-learning diagnostic, not as validated pain detection.\n\n"
        )
        f.write("## Validation Folds\n\n")
        f.write(_markdown_table(fold_summary_df.round(4)))
        f.write("\n\n## Final 4-Cow Test Set (Ensemble of Inner Fold Models)\n\n")
        f.write(_markdown_table(pd.DataFrame([test_metrics]).round(4)))
        f.write("\n\n## Cow-Level Final Test Metrics\n\n")
        f.write(_markdown_table(pd.DataFrame([test_cow_metrics]).round(4)))
        if "calibrated_sequence_metrics" in outputs:
            f.write("\n\n## Calibrated Final Test Metrics\n\n")
            calibrated_metrics = json.loads(outputs["calibrated_sequence_metrics"])
            calibrated_cow_metrics = json.loads(outputs["calibrated_cow_metrics"])
            f.write(_markdown_table(pd.DataFrame([calibrated_metrics]).round(4)))
            f.write("\n\n## Calibrated Cow-Level Final Test Metrics\n\n")
            f.write(_markdown_table(pd.DataFrame([calibrated_cow_metrics]).round(4)))
        f.write("\n\n## Final Test Cow Aggregates\n\n")
        f.write(_markdown_table(test_cow_df.round(4)))
        f.write("\n\n## Artifacts\n\n")
        for key, value in outputs.items():
            if key.startswith("calibrated_"):
                continue
            f.write(f"- `{key}`: `{value}`\n")


def main() -> int:
    args = _parse_args()
    manifest_csv = args.manifest_csv.resolve()
    sequence_root = args.sequence_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _prepare_records(manifest_csv, sequence_root, str(args.label_column))
    split = _build_split_plan(records, args)
    _print_split_plan(split)
    split_path = out_dir / "weak_label_cv_splits.json"
    _write_json(split_path, split)
    print(f"Wrote split audit: {split_path}")

    if bool(args.dry_run):
        print("Dry run complete: no training was started.")
        return 0

    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else None
    if not bool(args.from_scratch) and checkpoint_dir is None:
        raise ValueError("Pass --checkpoint-dir for UCAPS initialization, or use --from-scratch.")

    mod = _load_train_module(args.train_py, checkpoint_dir)
    _seed_everything(mod, int(args.seed))

    init_ckpt = None
    init_ckpt_path = None
    if not bool(args.from_scratch):
        assert checkpoint_dir is not None
        init_ckpt_path = _checkpoint_path(checkpoint_dir, int(args.init_fold), str(args.ckpt_kind))
        init_ckpt = _load_checkpoint(init_ckpt_path)
        print(f"Initializing weak-label folds from: {init_ckpt_path}")

    base_cfg = _cfg_from_checkpoint_or_default(mod, init_ckpt)
    base_cfg = _apply_cfg_overrides(base_cfg, args)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "manifest_csv": str(manifest_csv),
        "sequence_root": str(sequence_root),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "init_ckpt_path": str(init_ckpt_path) if init_ckpt_path else None,
        "args": vars(args),
        "cfg": _cfg_to_dict(base_cfg),
    }
    _write_json(out_dir / "weak_label_cv_run.json", run_meta)

    fold_summaries: list[dict[str, Any]] = []
    val_predictions: list[pd.DataFrame] = []
    best_paths: list[Path] = []
    for fold in split["folds"]:
        print("\n" + "=" * 80)
        print(f"Training weak-label fold {fold['fold']}")
        print("=" * 80)
        summary, pred_df, best_path = _train_one_fold(
            mod=mod,
            base_cfg=base_cfg,
            args=args,
            fold=fold,
            records=records,
            sequence_root=sequence_root,
            device=device,
            init_ckpt=init_ckpt,
            run_dir=out_dir,
        )
        fold_summaries.append(summary)
        val_predictions.append(pred_df)
        best_paths.append(best_path)

    fold_summary_df = pd.DataFrame([_summary_row(s) for s in fold_summaries])
    fold_summary_path = out_dir / "weak_label_cv_fold_summary.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)

    all_val_pred = pd.concat(val_predictions, ignore_index=True) if val_predictions else pd.DataFrame()
    val_predictions_path = out_dir / "weak_label_cv_predictions.csv"
    all_val_pred.to_csv(val_predictions_path, index=False)

    test_records = _records_for_cows(records, split["test"]["cows"])
    test_pred = _ensemble_test_predictions(
        mod=mod,
        best_paths=best_paths,
        records=test_records,
        sequence_root=sequence_root,
        device=device,
        args=args,
    )
    threshold_values = [s["val_metrics"].get("best_threshold") for s in fold_summaries if s["val_metrics"].get("best_threshold") is not None]
    test_threshold = float(np.mean(threshold_values)) if threshold_values else 0.5
    test_metrics = _binary_metrics(test_pred["target"].values, test_pred["pain_prob"].values, threshold=test_threshold)
    test_cow_metrics, test_cow_df = _cow_level_metrics(test_pred, threshold=test_threshold)
    temperatures = [float(s.get("temperature", 1.0)) for s in fold_summaries if s.get("temperature") is not None]
    test_temperature = float(np.mean(temperatures)) if temperatures else 1.0
    test_pred = _add_calibrated_probabilities(test_pred, temperature=test_temperature)
    calibrated_threshold_values = [
        s.get("val_calibrated_metrics", {}).get("best_threshold")
        for s in fold_summaries
        if s.get("val_calibrated_metrics", {}).get("best_threshold") is not None
    ]
    calibrated_test_threshold = float(np.mean(calibrated_threshold_values)) if calibrated_threshold_values else 0.5
    test_calibrated_metrics = _calibrated_metrics(test_pred, threshold=calibrated_test_threshold)
    test_calibrated_cow_metrics, test_calibrated_cow_df = _calibrated_cow_metrics(test_pred, threshold=calibrated_test_threshold)

    test_predictions_path = out_dir / "weak_label_cv_test_predictions.csv"
    test_pred.to_csv(test_predictions_path, index=False)
    test_cow_path = out_dir / "weak_label_cv_test_cow_aggregates.csv"
    test_cow_df.to_csv(test_cow_path, index=False)
    test_calibrated_cow_path = out_dir / "weak_label_cv_test_calibrated_cow_aggregates.csv"
    test_calibrated_cow_df.to_csv(test_calibrated_cow_path, index=False)

    summary_payload = {
        "run": run_meta,
        "split": split,
        "fold_summaries": fold_summaries,
        "final_test": {
            "threshold_from_validation_mean": test_threshold,
            "temperature_from_validation_mean": test_temperature,
            "sequence_metrics": test_metrics,
            "cow_metrics": test_cow_metrics,
            "calibrated_threshold_from_validation_mean": calibrated_test_threshold,
            "calibrated_sequence_metrics": test_calibrated_metrics,
            "calibrated_cow_metrics": test_calibrated_cow_metrics,
        },
        "artifacts": {
            "split_json": str(split_path),
            "fold_summary_csv": str(fold_summary_path),
            "val_predictions_csv": str(val_predictions_path),
            "test_predictions_csv": str(test_predictions_path),
            "test_cow_aggregates_csv": str(test_cow_path),
            "test_calibrated_cow_aggregates_csv": str(test_calibrated_cow_path),
        },
    }
    summary_path = out_dir / "weak_label_cv_summary.json"
    _write_json(summary_path, summary_payload)

    report_path = out_dir / "weak_label_cv_report.md"
    _write_report(
        report_path=report_path,
        args=args,
        split=split,
        fold_summary_df=fold_summary_df,
        test_metrics=test_metrics,
        test_cow_metrics=test_cow_metrics,
        test_cow_df=test_cow_df,
        outputs={
            "split_json": str(split_path),
            "summary_json": str(summary_path),
            "fold_summary_csv": str(fold_summary_path),
            "val_predictions_csv": str(val_predictions_path),
            "test_predictions_csv": str(test_predictions_path),
            "test_cow_aggregates_csv": str(test_cow_path),
            "test_calibrated_cow_aggregates_csv": str(test_calibrated_cow_path),
            "calibrated_sequence_metrics": json.dumps(test_calibrated_metrics, default=_json_default),
            "calibrated_cow_metrics": json.dumps(test_calibrated_cow_metrics, default=_json_default),
        },
    )

    print(f"Wrote fold summary: {fold_summary_path}")
    print(f"Wrote validation predictions: {val_predictions_path}")
    print(f"Wrote test predictions: {test_predictions_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
