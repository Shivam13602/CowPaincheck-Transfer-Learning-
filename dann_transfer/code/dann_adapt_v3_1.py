#!/usr/bin/env python3
"""
Domain-adversarial UCAPS -> Holstein/Jersey adaptation (**V3.1** fork).

This script keeps the Holstein/Jersey cow-held-out protocol from ``weak_label_adapt_v3_1.py``,
but trains each fold with labeled UCAPS source clips plus target-domain Holstein clips.
V3.1 adds optional **CDAN-style** domain conditioning (Long et al., NeurIPS 2018),
**MDD-style** dual-head discrepancy (Saito et al., ICCV 2019), and **domain-loss warmup/ramp**.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
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
from torch.autograd import Function
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from holstein_eval_thresholds import predictions_degenerate, resolve_threshold_for_test


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_weak_module() -> Any:
    return _load_module(HERE / "weak_label_adapt_v3_1.py", "weak_label_adapt_v3_1")


def _load_train_module(train_py: Path | None) -> Any:
    path = train_py.resolve() if train_py else (HERE / "v2.9_training_classification.py")
    return _load_module(path, "ucaps_v2_9_training")


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


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _cfg_from_checkpoint_or_default(mod: Any, ckpt: dict[str, Any] | None) -> Any:
    cfg = mod.Config()
    cfg_dict = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(cfg_dict, dict):
        for key, value in cfg_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def _apply_cfg_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    if bool(getattr(args, "task1_only", False)):
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
    if hasattr(cfg, "use_augmentations") and bool(args.no_aug):
        cfg.use_augmentations = False
    if hasattr(cfg, "use_stratified_sampler"):
        cfg.use_stratified_sampler = not bool(args.no_stratified_sampler)
    if hasattr(cfg, "use_moment_weighting"):
        cfg.use_moment_weighting = bool(args.use_moment_weighting)
    if hasattr(cfg, "freeze_cnn"):
        cfg.freeze_cnn = bool(args.freeze_cnn)
    return cfg


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        return GradientReversalFn.apply(x, float(lambda_))


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _feature_dim(cfg: Any) -> int:
    return int(cfg.lstm_hidden_size) * (2 if bool(getattr(cfg, "use_bidirectional_lstm", False)) else 1)


def _dann_lambda(progress: float) -> float:
    p = float(np.clip(progress, 0.0, 1.0))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def _domain_in_dim(cfg: Any, args: argparse.Namespace) -> int:
    base = _feature_dim(cfg)
    if str(args.da_mode) == "cdan":
        return base + 2
    return base


def _effective_domain_weight(epoch: int, args: argparse.Namespace) -> float:
    """Warmup (zero domain loss) then optional linear ramp to ``domain_weight``."""
    w = float(args.domain_weight)
    warmup = int(getattr(args, "domain_warmup_epochs", 0))
    ramp = int(getattr(args, "domain_ramp_epochs", 0))
    if warmup > 0 and epoch < warmup:
        return 0.0
    if ramp <= 0:
        return w
    rel = epoch - warmup
    t = min(1.0, max(0, rel) / float(max(ramp, 1)))
    return w * t


def _detach_binary_task_probs(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).unsqueeze(1)
    return torch.cat([1.0 - p, p], dim=1).detach()


def _make_split_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        label_column=args.label_column,
        test_cows=args.test_cows,
        test_cow_ids=args.test_cow_ids,
        val_cows_per_fold=args.val_cows_per_fold,
        seed=args.seed,
    )


def _make_loader_from_sequences(mod: Any, sequences: list[dict[str, Any]], sequence_root: Path, cfg: Any, *, augment: bool, shuffle: bool) -> DataLoader:
    DatasetCls = getattr(mod, "FacialPainDataset_v2_9", None) or getattr(mod, "FacialPainDataset_v2_8")
    ds = DatasetCls(sequences, sequence_root, cfg, augment=augment, global_cache={})
    sampler = None
    if augment and shuffle and bool(getattr(cfg, "use_stratified_sampler", False)) and hasattr(mod, "create_stratified_sampler"):
        sampler = mod.create_stratified_sampler(sequences)
    loader_extra: dict[str, Any] = {}
    if int(cfg.num_workers) > 0:
        loader_extra["prefetch_factor"] = 2
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=augment,
        **loader_extra,
    )


def _make_loader_from_records(mod: Any, records: list[dict[str, Any]], sequence_root: Path, cfg: Any, *, augment: bool, shuffle: bool) -> DataLoader:
    return _make_loader_from_sequences(mod, [r["seq"] for r in records], sequence_root, cfg, augment=augment, shuffle=shuffle)


def _load_source_bundle(mod: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path, dict[str, Any]]:
    if args.source_project_dir is None:
        raise ValueError("--source-project-dir is required for DANN training.")
    project_dir = args.source_project_dir.resolve()
    sequence_root = args.source_sequence_dir.resolve() if args.source_sequence_dir else (project_dir.parent / "sequence")
    splits_file = args.source_splits_json.resolve() if args.source_splits_json else (project_dir / "train_val_test_splits_v2.json")
    mapping_file = args.source_mapping_json.resolve() if args.source_mapping_json else (project_dir / "sequence_label_mapping_v2.json")
    if not splits_file.exists():
        raise FileNotFoundError(f"Missing source splits JSON: {splits_file}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Missing source mapping JSON: {mapping_file}")
    if not sequence_root.exists():
        raise FileNotFoundError(f"Missing source sequence root: {sequence_root}")

    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    if isinstance(mapping, dict):
        if "sequences" in mapping:
            all_sequences = mapping["sequences"]
        else:
            all_sequences = [{"sequence_id": k, **v} for k, v in mapping.items()]
    else:
        all_sequences = mapping
    folds = mod.get_folds_from_splits(splits)
    source_fold_idx = int(args.source_fold if args.source_fold is not None else args.init_fold)
    if source_fold_idx < 0 or source_fold_idx >= len(folds):
        raise ValueError(f"--source-fold {source_fold_idx} is outside available source folds 0..{len(folds) - 1}")
    fold = folds[source_fold_idx]
    if bool(args.source_use_train_val):
        train_animals = set(int(a) for a in splits.get("train_val_animals", []))
        val_animals = set(int(a) for a in fold.get("val_animals", []))
    else:
        train_animals = set(int(a) for a in fold["train_animals"])
        val_animals = set(int(a) for a in fold["val_animals"])
    train = [s for s in all_sequences if mod._seq_animal_id(s) in train_animals]
    val = [s for s in all_sequences if mod._seq_animal_id(s) in val_animals]
    if not train:
        raise RuntimeError("No source training sequences selected.")
    meta = {
        "source_project_dir": str(project_dir),
        "source_sequence_root": str(sequence_root),
        "source_splits_json": str(splits_file),
        "source_mapping_json": str(mapping_file),
        "source_fold": source_fold_idx,
        "source_use_train_val": bool(args.source_use_train_val),
        "n_source_train": len(train),
        "n_source_val": len(val),
    }
    return train, val, sequence_root, meta


def _pos_weight_source(mod: Any, sequences: list[dict[str, Any]], device: torch.device) -> torch.Tensor | None:
    labels = np.array([int(mod.moment_to_task1_binary(s.get("moment", "unknown"))) for s in sequences], dtype=np.int64)
    pos = float(labels.sum())
    neg = float(len(labels) - labels.sum())
    if pos > 0 and neg > 0:
        return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    return None


def _task2_weights(mod: Any, sequences: list[dict[str, Any]], cfg: Any, device: torch.device) -> torch.Tensor | None:
    k = int(mod.task2_num_classes(cfg.task2_mode))
    labels = np.array([int(mod.moment_to_task2(s.get("moment", "unknown"), task2_mode=cfg.task2_mode)) for s in sequences], dtype=np.int64)
    counts = np.array([(labels == i).sum() for i in range(k)], dtype=np.float64)
    if np.any(counts <= 0):
        return None
    weights = counts.sum() / (k * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def _source_sanity_metrics(
    *,
    mod: Any,
    weak: Any,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    if len(loader.dataset) == 0:
        return {}
    model.eval()
    pain_scores: list[np.ndarray] = []
    pain_targets: list[np.ndarray] = []
    task2_pred: list[np.ndarray] = []
    task2_true: list[np.ndarray] = []
    for b_idx, (x, y, _meta) in enumerate(loader):
        if max_batches is not None and b_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        out, _ = model(x)
        logits = out["pain_logits"].detach().cpu().numpy().astype(np.float64)
        pain_scores.append(1.0 / (1.0 + np.exp(-logits)))
        pain_targets.append(y["pain_binary"].detach().cpu().numpy().astype(np.int64))
        task2_pred.append(out["task2_logits"].argmax(dim=1).detach().cpu().numpy().astype(np.int64))
        task2_true.append(y["task2"].detach().cpu().numpy().astype(np.int64))
    if not pain_scores:
        return {}
    scores = np.concatenate(pain_scores)
    targets = np.concatenate(pain_targets)
    t2p = np.concatenate(task2_pred)
    t2t = np.concatenate(task2_true)
    out = {f"source_task1_{k}": v for k, v in weak._binary_metrics(targets, scores).items()}
    out["source_task2_accuracy"] = float((t2p == t2t).mean()) if len(t2t) else None
    return out


def _load_model_state_from_ssl(model: nn.Module, ssl_ckpt_dir: Path | None, fold_idx: int, pattern: str) -> Path | None:
    if ssl_ckpt_dir is None:
        return None
    rel = pattern.format(fold=fold_idx)
    path = (ssl_ckpt_dir / rel).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SSL checkpoint not found for fold {fold_idx}: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return path


def _make_model(mod: Any, cfg: Any, device: torch.device, init_ckpt: dict[str, Any] | None, args: argparse.Namespace, fold_idx: int) -> tuple[nn.Module, Path | None]:
    ModelCls = getattr(mod, "TemporalPainModel_v2_9", None) or getattr(mod, "TemporalPainModel_v2_8")
    model = ModelCls(cfg).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
    ssl_path = _load_model_state_from_ssl(model, args.ssl_checkpoint_dir.resolve() if args.ssl_checkpoint_dir else None, fold_idx, str(args.ssl_checkpoint_pattern))
    if not hasattr(model, "extract_features"):
        raise AttributeError("TemporalPainModel_v2_9 must expose extract_features() for DANN.")
    if bool(args.freeze_cnn) and hasattr(model, "cnn"):
        _set_requires_grad(model.cnn, False)
    return model, ssl_path


def _domain_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().detach().item())


def _target_weak_weight(args: argparse.Namespace, epoch: int, progress: float) -> float:
    if float(args.target_weak_weight) <= 0.0:
        return 0.0
    if epoch < int(args.target_weak_start_epoch):
        return 0.0
    if bool(args.ramp_target_weak):
        return float(args.target_weak_weight) * float(np.clip(progress, 0.0, 1.0))
    return float(args.target_weak_weight)


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


def _task1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    pos_weight: torch.Tensor | None,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    gce_q: float,
) -> torch.Tensor:
    targets = targets.float()
    if loss_type == "gce":
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
        per_sample = (1.0 - pt.pow(float(gce_q))) / max(float(gce_q), 1e-6)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        if loss_type == "focal":
            probs = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
            per_sample = bce * torch.pow(1.0 - pt, float(focal_gamma))
        else:
            per_sample = bce
    if class_weights is not None:
        weights = torch.where(targets > 0.5, class_weights[1], class_weights[0])
        per_sample = per_sample * weights
    return per_sample.mean()


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.size(0) < 2:
        return features.new_zeros(())
    labels = labels.view(-1).long()
    features = F.normalize(features.float(), dim=1)
    logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid = positive_mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return features.new_zeros(())
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _train_one_fold(
    *,
    mod: Any,
    weak: Any,
    base_cfg: Any,
    args: argparse.Namespace,
    fold: dict[str, Any],
    target_records: list[dict[str, Any]],
    target_split: dict[str, Any],
    source_train: list[dict[str, Any]],
    source_val: list[dict[str, Any]],
    source_root: Path,
    target_root: Path,
    device: torch.device,
    init_ckpt: dict[str, Any] | None,
    init_ckpt_path: Path | None,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    fold_idx = int(fold["fold"])
    _seed_everything(int(args.seed) + fold_idx)
    cfg = deepcopy(base_cfg)
    cfg = _apply_cfg_overrides(cfg, args)

    target_train = weak._records_for_cows(target_records, fold["train"]["cows"])
    target_val = weak._records_for_cows(target_records, fold["val"]["cows"])
    source_loader = _make_loader_from_sequences(mod, source_train, source_root, cfg, augment=True, shuffle=True)
    target_loader = _make_loader_from_records(mod, target_train, target_root, cfg, augment=True, shuffle=True)
    target_val_loader = weak._make_loader(mod, target_val, target_root, cfg, augment=False, shuffle=False)
    source_val_loader = _make_loader_from_sequences(mod, source_val, source_root, cfg, augment=False, shuffle=False) if source_val else None

    model, ssl_path = _make_model(mod, cfg, device, init_ckpt, args, fold_idx)
    domain_in = _domain_in_dim(cfg, args)
    domain_head = DomainClassifier(domain_in, hidden_dim=int(args.domain_hidden_dim), dropout=float(args.domain_dropout)).to(device)
    grl = GradientReversalLayer()

    head_mdd: nn.Module | None = None
    if str(args.da_mode) == "mdd":
        head_mdd = nn.Linear(_feature_dim(cfg), 1).to(device)

    params = [p for p in list(model.parameters()) + list(domain_head.parameters()) if p.requires_grad]
    if head_mdd is not None:
        params.extend([p for p in head_mdd.parameters() if p.requires_grad])
    optimizer = AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(getattr(cfg, "weight_decay", 0.0)))
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(args.lr_scheduler_patience),
        min_lr=float(getattr(cfg, "min_lr", 1e-7)),
    )
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not bool(args.no_amp)) else None

    source_labels = np.array([int(mod.moment_to_task1_binary(s.get("moment", "unknown"))) for s in source_train], dtype=np.int64)
    source_pos_weight = None if bool(args.source_class_balanced) else _pos_weight_source(mod, source_train, device)
    source_class_weights = (
        _effective_num_weights(source_labels, float(args.class_balanced_beta), device)
        if bool(args.source_class_balanced)
        else None
    )
    target_pos_weight = weak._pos_weight(target_train, device)
    source_task2_weights = _task2_weights(mod, source_train, cfg, device)
    target_weak_loss = nn.BCEWithLogitsLoss(pos_weight=target_pos_weight)

    fold_dir = run_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_dann.pt"
    best_proxy_fallback_path = fold_dir / "best_dann_proxy_fallback.pt"
    best_sanity_score = float("-inf")
    best_proxy_score = float("-inf")
    best_sanity_summary: dict[str, Any] | None = None
    best_sanity_pred_df = pd.DataFrame()
    best_proxy_summary: dict[str, Any] | None = None
    best_proxy_pred_df = pd.DataFrame()
    history: list[dict[str, Any]] = []
    steps_per_epoch = max(1, max(len(source_loader), len(target_loader)))

    for epoch in range(int(cfg.num_epochs)):
        model.train()
        domain_head.train()
        if head_mdd is not None:
            head_mdd.train()
        if bool(args.freeze_cnn) and hasattr(model, "cnn"):
            model.cnn.eval()
        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)
        totals = {
            "loss": 0.0,
            "source_task1_loss": 0.0,
            "source_task2_loss": 0.0,
            "source_supcon_loss": 0.0,
            "domain_loss": 0.0,
            "target_weak_loss": 0.0,
            "domain_acc": 0.0,
        }
        n_steps = 0
        eff_dw_last = 0.0
        for step in tqdm(range(steps_per_epoch), desc=f"DANN fold {fold_idx}", leave=False, ascii=True):
            if args.max_train_batches is not None and step >= int(args.max_train_batches):
                break
            (sx, sy, _smeta), src_iter = _next_batch(src_iter, source_loader)
            (tx, ty, _tmeta), tgt_iter = _next_batch(tgt_iter, target_loader)
            sx = sx.to(device, non_blocking=True)
            tx = tx.to(device, non_blocking=True)
            sy1 = sy["pain_binary"].to(device, non_blocking=True)
            sy2 = sy["task2"].to(device, non_blocking=True)
            ty1 = ty["pain_binary"].to(device, non_blocking=True)
            global_step = epoch * steps_per_epoch + step
            progress = global_step / max(1, int(cfg.num_epochs) * steps_per_epoch - 1)
            lambda_grl = float(args.domain_lambda_max) * _dann_lambda(progress)
            target_weight = _target_weak_weight(args, epoch, progress)

            optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                sf, _ = model.extract_features(sx, apply_dropout=True)
                tf, _ = model.extract_features(tx, apply_dropout=True)
                src_pain_logits = model.head_task1(sf).squeeze(-1)
                src_task2_logits = model.head_task2(sf)
                tgt_pain_logits = model.head_task1(tf).squeeze(-1)

                l_source_task1 = _task1_loss(
                    src_pain_logits,
                    sy1,
                    loss_type=str(args.source_task1_loss),
                    pos_weight=source_pos_weight,
                    class_weights=source_class_weights,
                    focal_gamma=float(args.focal_gamma),
                    gce_q=float(args.gce_q),
                )
                if float(args.source_task2_weight) > 0.0:
                    l_source_task2 = F.cross_entropy(
                        src_task2_logits,
                        sy2.long(),
                        weight=source_task2_weights,
                        label_smoothing=float(getattr(cfg, "label_smoothing", 0.0)),
                    )
                else:
                    l_source_task2 = src_task2_logits.new_zeros(())
                if float(args.source_supcon_weight) > 0.0:
                    l_source_supcon = _supervised_contrastive_loss(sf.float(), sy1, float(args.supcon_temperature))
                else:
                    l_source_supcon = sf.new_zeros(())

                eff_dw = _effective_domain_weight(epoch, args)
                l_mdd_total = sf.new_zeros(())
                if head_mdd is not None:
                    z2_src = head_mdd(sf).squeeze(-1)
                    z2_tgt = head_mdd(tf).squeeze(-1)
                    l_mdd_cls = _task1_loss(
                        z2_src,
                        sy1,
                        loss_type=str(args.source_task1_loss),
                        pos_weight=source_pos_weight,
                        class_weights=source_class_weights,
                        focal_gamma=float(args.focal_gamma),
                        gce_q=float(args.gce_q),
                    )
                    p1s = torch.sigmoid(src_pain_logits)
                    p2s = torch.sigmoid(z2_src)
                    p1t = torch.sigmoid(tgt_pain_logits)
                    p2t = torch.sigmoid(z2_tgt)
                    l_mdd_disc = (p1s - p2s).abs().mean() - float(args.mdd_disc_weight) * (p1t - p2t).abs().mean()
                    l_mdd_total = float(args.mdd_aux_weight) * l_mdd_cls + float(args.mdd_weight) * l_mdd_disc

                if str(args.da_mode) == "cdan":
                    sf_cd = torch.cat([sf, _detach_binary_task_probs(src_pain_logits)], dim=1)
                    tf_cd = torch.cat([tf, _detach_binary_task_probs(tgt_pain_logits)], dim=1)
                    features = torch.cat([sf_cd, tf_cd], dim=0)
                else:
                    features = torch.cat([sf, tf], dim=0)
                domain_targets = torch.cat(
                    [
                        torch.zeros(sf.size(0), dtype=torch.long, device=device),
                        torch.ones(tf.size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                domain_logits = domain_head(grl(features, lambda_grl))
                l_domain = F.cross_entropy(domain_logits, domain_targets)
                l_target_weak = target_weak_loss(tgt_pain_logits, ty1)
                loss = (
                    float(args.source_task1_weight) * l_source_task1
                    + float(args.source_task2_weight) * l_source_task2
                    + float(args.source_supcon_weight) * l_source_supcon
                    + eff_dw * l_domain
                    + target_weight * l_target_weak
                    + l_mdd_total
                )

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
                optimizer.step()

            totals["loss"] += float(loss.detach().item())
            totals["source_task1_loss"] += float(l_source_task1.detach().item())
            totals["source_task2_loss"] += float(l_source_task2.detach().item())
            totals["source_supcon_loss"] += float(l_source_supcon.detach().item())
            totals["domain_loss"] += float(l_domain.detach().item())
            totals["target_weak_loss"] += float(l_target_weak.detach().item())
            totals["domain_acc"] += _domain_accuracy(domain_logits.detach(), domain_targets)
            n_steps += 1
            eff_dw_last = float(eff_dw)

        val_pred = weak._predict_records(
            model,
            target_val_loader,
            target_val,
            device=device,
            split="val",
            fold=fold_idx,
            max_batches=args.max_val_batches,
            mod=mod,
            sequence_root=target_root,
            args=args,
        )
        val_metrics = weak._binary_metrics(val_pred["target"].values, val_pred["pain_prob"].values)
        val_cow_metrics, _val_cow_df = weak._cow_level_metrics(val_pred)
        temperature = weak._fit_temperature_from_logits(val_pred["pain_logit"].values, val_pred["target"].values)
        val_pred_calibrated = weak._add_calibrated_probabilities(val_pred, temperature=temperature)
        val_calibrated_metrics = weak._calibrated_metrics(val_pred_calibrated)
        val_calibrated_cow_metrics, _val_calibrated_cow_df = weak._calibrated_cow_metrics(val_pred_calibrated)
        source_metrics = (
            _source_sanity_metrics(mod=mod, weak=weak, model=model, loader=source_val_loader, device=device, max_batches=args.max_val_batches)
            if source_val_loader is not None
            else {}
        )
        proxy_score = weak._metric_for_selection(val_metrics, str(args.select_metric))
        source_auc = source_metrics.get("source_task1_auc")
        sanity_floor = float(args.source_task1_sanity_floor)
        source_sanity_pass = (
            sanity_floor <= 0.0
            or (source_auc is not None and not math.isnan(float(source_auc)) and float(source_auc) >= sanity_floor)
        )
        score = float(proxy_score) if source_sanity_pass else float("-inf")
        scheduler.step(float(proxy_score) if math.isfinite(float(proxy_score)) else 0.0)
        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "selection_metric": str(args.select_metric),
            "selection_score": float(score),
            "proxy_selection_score": float(proxy_score),
            "source_task1_sanity_floor": float(sanity_floor),
            "source_task1_sanity_pass": bool(source_sanity_pass),
            "temperature": float(temperature),
            "lambda_grl_last": float(lambda_grl),
            "effective_domain_weight_last": float(eff_dw_last),
            "target_weak_weight_last": float(target_weight),
            **{k: float(v / max(1, n_steps)) for k, v in totals.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_cow_{k}": v for k, v in val_cow_metrics.items()},
            **{f"val_calibrated_{k}": v for k, v in val_calibrated_metrics.items()},
            **{f"val_calibrated_cow_{k}": v for k, v in val_calibrated_cow_metrics.items()},
            **source_metrics,
        }
        history.append(row)
        print(
            f"fold {fold_idx} epoch {epoch + 1}/{cfg.num_epochs}: "
            f"loss={row['loss']:.4f} dom_acc={row['domain_acc']:.3f} "
            f"val_auc={val_metrics.get('auc')} val_f1_opt={val_metrics.get('f1_opt')} "
            f"source_auc={source_auc} sanity_pass={source_sanity_pass}"
        )
        ckpt_payload = {
            "version": "v3.1_holstein_dann",
            "fold": fold_idx,
            "epoch": epoch,
            "model_state_dict": _cpu_state_dict(model),
            "domain_head_state_dict": _cpu_state_dict(domain_head),
            "head_mdd_state_dict": _cpu_state_dict(head_mdd) if head_mdd is not None else None,
            "cfg": _cfg_to_dict(cfg),
            "split": fold,
            "val_metrics": val_metrics,
            "val_cow_metrics": val_cow_metrics,
            "temperature": float(temperature),
            "val_calibrated_metrics": val_calibrated_metrics,
            "val_calibrated_cow_metrics": val_calibrated_cow_metrics,
            "source_metrics": source_metrics,
            "source_task1_sanity_floor": float(sanity_floor),
            "source_task1_sanity_pass": bool(source_sanity_pass),
            "label_column": str(args.label_column),
            "label_mapping": {"Healthy": 0, "Unhealthy": 1},
            "init_ckpt_path": str(init_ckpt_path) if init_ckpt_path else None,
            "ssl_checkpoint_path": str(ssl_path) if ssl_path else None,
            "args": vars(args),
        }
        if source_sanity_pass and float(proxy_score) > best_sanity_score:
            best_sanity_score = float(proxy_score)
            best_sanity_summary = {
                "fold": fold_idx,
                "best_epoch": epoch,
                "best_score": float(score),
                "selection_metric": str(args.select_metric),
                "proxy_selection_score": float(proxy_score),
                "source_task1_sanity_floor": float(sanity_floor),
                "source_task1_sanity_pass": bool(source_sanity_pass),
                "checkpoint_selected_from_proxy_fallback": False,
                "val_metrics": val_metrics,
                "val_cow_metrics": val_cow_metrics,
                "temperature": float(temperature),
                "val_calibrated_metrics": val_calibrated_metrics,
                "val_calibrated_cow_metrics": val_calibrated_cow_metrics,
                "source_metrics": source_metrics,
                "train": fold["train"],
                "val": fold["val"],
                "ssl_checkpoint_path": str(ssl_path) if ssl_path else None,
            }
            best_sanity_pred_df = val_pred_calibrated.copy()
            torch.save({**ckpt_payload, "checkpoint_role": "sanity_gated_best"}, best_path)

        if float(proxy_score) > best_proxy_score:
            best_proxy_score = float(proxy_score)
            best_proxy_summary = {
                "fold": fold_idx,
                "best_epoch": epoch,
                "best_score": float(score),
                "selection_metric": str(args.select_metric),
                "proxy_selection_score": float(proxy_score),
                "source_task1_sanity_floor": float(sanity_floor),
                "source_task1_sanity_pass": bool(source_sanity_pass),
                "val_metrics": val_metrics,
                "val_cow_metrics": val_cow_metrics,
                "temperature": float(temperature),
                "val_calibrated_metrics": val_calibrated_metrics,
                "val_calibrated_cow_metrics": val_calibrated_cow_metrics,
                "source_metrics": source_metrics,
                "train": fold["train"],
                "val": fold["val"],
                "ssl_checkpoint_path": str(ssl_path) if ssl_path else None,
            }
            best_proxy_pred_df = val_pred_calibrated.copy()
            torch.save({**ckpt_payload, "checkpoint_role": "best_proxy_metric"}, best_proxy_fallback_path)

    if best_sanity_summary is not None:
        best_summary = dict(best_sanity_summary)
        best_summary.setdefault("checkpoint_selected_from_proxy_fallback", False)
        best_pred_df = best_sanity_pred_df
    elif best_proxy_summary is not None:
        best_summary = dict(best_proxy_summary)
        best_summary["checkpoint_selected_from_proxy_fallback"] = True
        best_pred_df = best_proxy_pred_df
        if best_proxy_fallback_path.exists():
            best_path.write_bytes(best_proxy_fallback_path.read_bytes())
        print(
            f"WARNING fold {fold_idx}: no epoch passed source Task1 sanity floor {sanity_floor}; "
            f"using best Holstein proxy-{args.select_metric} checkpoint from epoch {best_summary['best_epoch']} instead."
        )
    else:
        raise RuntimeError(f"Fold {fold_idx} did not produce any checkpoint (no validation scores).")

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    best_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False)
    return best_summary, best_pred_df, best_path


def _load_dann_model(mod: Any, ckpt_path: Path, device: torch.device) -> tuple[nn.Module, Any, dict[str, Any]]:
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
    weak: Any,
    best_paths: list[Path],
    records: list[dict[str, Any]],
    sequence_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    logit_stack: list[np.ndarray] = []
    first_df: pd.DataFrame | None = None
    for path in best_paths:
        model, cfg, ckpt = _load_dann_model(mod, path, device)
        cfg.batch_size = int(args.batch_size)
        cfg.num_workers = int(args.num_workers)
        loader = weak._make_loader(mod, records, sequence_root, cfg, augment=False, shuffle=False)
        pred_df = weak._predict_records(
            model,
            loader,
            records,
            device=device,
            split="test",
            fold=f"ensemble_member_{ckpt['fold']}",
            max_batches=args.max_val_batches,
            mod=mod,
            sequence_root=sequence_root,
            args=args,
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
    src = summary.get("source_metrics", {})
    return {
        "fold": summary["fold"],
        "best_epoch": summary["best_epoch"],
        "best_score": summary["best_score"],
        "proxy_selection_score": summary.get("proxy_selection_score"),
        "source_task1_sanity_floor": summary.get("source_task1_sanity_floor"),
        "source_task1_sanity_pass": summary.get("source_task1_sanity_pass"),
        "checkpoint_selected_from_proxy_fallback": bool(summary.get("checkpoint_selected_from_proxy_fallback", False)),
        "val_cows": ",".join(summary["val"]["cows"]),
        "val_n": val["n"],
        "val_auc": val["auc"],
        "val_f1": val["f1"],
        "val_f1_opt": val["f1_opt"],
        "val_accuracy": val["accuracy"],
        "val_balanced_accuracy": val["balanced_accuracy"],
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_cow_auc": cow["auc"],
        "val_cow_f1": cow["f1"],
        "val_cow_f1_opt": cow["f1_opt"],
        "temperature": summary.get("temperature"),
        "val_calibrated_auc": summary.get("val_calibrated_metrics", {}).get("auc"),
        "val_calibrated_f1_opt": summary.get("val_calibrated_metrics", {}).get("f1_opt"),
        "val_calibrated_cow_auc": summary.get("val_calibrated_cow_metrics", {}).get("auc"),
        "source_task1_auc": src.get("source_task1_auc"),
        "source_task1_f1": src.get("source_task1_f1"),
        "source_task1_f1_opt": src.get("source_task1_f1_opt"),
        "source_task1_balanced_accuracy": src.get("source_task1_balanced_accuracy"),
        "source_task1_precision": src.get("source_task1_precision"),
        "source_task1_recall": src.get("source_task1_recall"),
        "source_task1_best_threshold": src.get("source_task1_best_threshold"),
        "source_task2_accuracy": src.get("source_task2_accuracy"),
    }


def _sorted_fold_ckpts(out_dir: Path, basename: str) -> list[Path]:
    pairs: list[tuple[int, Path]] = []
    for fd in sorted(out_dir.glob("fold_*")):
        if not fd.is_dir():
            continue
        parts = fd.name.split("_", 1)
        if len(parts) != 2 or parts[0] != "fold":
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        ck = fd / basename
        if ck.is_file():
            pairs.append((idx, ck))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def _dann_finalize_outputs(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    split_path: Path,
    run_meta: dict[str, Any],
    weak: Any,
    mod: Any,
    target_split: dict[str, Any],
    target_records: list[dict[str, Any]],
    target_root: Path,
    device: torch.device,
    fold_summaries: list[dict[str, Any]],
    best_paths: list[Path],
    all_val_pred: pd.DataFrame,
    source_meta: dict[str, Any],
    write_val_predictions_csv: bool,
) -> None:
    fold_summary_df = pd.DataFrame([_summary_row(s) for s in fold_summaries])
    fold_summary_path = out_dir / "dann_fold_summary.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)
    val_predictions_path = out_dir / "dann_predictions.csv"
    if write_val_predictions_csv:
        all_val_pred.to_csv(val_predictions_path, index=False)

    test_records = weak._records_for_cows(target_records, target_split["test"]["cows"])
    test_pred = _ensemble_test_predictions(
        mod=mod,
        weak=weak,
        best_paths=best_paths,
        records=test_records,
        sequence_root=target_root,
        device=device,
        args=args,
    )
    test_threshold, thr_meta_raw = resolve_threshold_for_test(
        str(args.test_threshold_policy),
        fold_summaries=fold_summaries,
        all_val_pred=all_val_pred,
        scores_col="pain_prob",
        calibrated=False,
        fixed_threshold=float(args.test_threshold_fixed),
    )
    test_metrics = weak._binary_metrics(test_pred["target"].values, test_pred["pain_prob"].values, threshold=test_threshold)
    test_cow_metrics, test_cow_df = weak._cow_level_metrics(test_pred, threshold=test_threshold)
    deg_raw = predictions_degenerate(test_pred["target"].values, test_pred["pain_prob"].values, test_threshold)
    temperatures = [float(s.get("temperature", 1.0)) for s in fold_summaries if s.get("temperature") is not None]
    test_temperature = float(np.mean(temperatures)) if temperatures else 1.0
    test_pred = weak._add_calibrated_probabilities(test_pred, temperature=test_temperature)
    scores_cal_available = not all_val_pred.empty and "pain_prob_calibrated" in all_val_pred.columns
    calibrated_test_threshold, thr_meta_cal = resolve_threshold_for_test(
        str(args.test_threshold_policy),
        fold_summaries=fold_summaries,
        all_val_pred=all_val_pred if scores_cal_available else pd.DataFrame(),
        scores_col="pain_prob_calibrated",
        calibrated=True,
        fixed_threshold=float(args.test_threshold_fixed),
    )
    test_calibrated_metrics = weak._calibrated_metrics(test_pred, threshold=calibrated_test_threshold)
    test_calibrated_cow_metrics, test_calibrated_cow_df = weak._calibrated_cow_metrics(test_pred, threshold=calibrated_test_threshold)
    deg_cal = predictions_degenerate(
        test_pred["target"].values,
        test_pred["pain_prob_calibrated"].values,
        calibrated_test_threshold,
    )

    test_predictions_path = out_dir / "dann_test_predictions.csv"
    test_pred.to_csv(test_predictions_path, index=False)
    test_cow_path = out_dir / "dann_test_cow_aggregates.csv"
    test_cow_df.to_csv(test_cow_path, index=False)
    test_calibrated_cow_path = out_dir / "dann_test_calibrated_cow_aggregates.csv"
    test_calibrated_cow_df.to_csv(test_calibrated_cow_path, index=False)

    n_bins = int(args.diag_ece_bins)
    val_diag_raw = weak._prediction_diagnostics_block(all_val_pred, scores_col="pain_prob", targets_col="target", n_bins=n_bins)
    val_diag_cal: dict[str, Any] = {}
    if not all_val_pred.empty and "pain_prob_calibrated" in all_val_pred.columns:
        val_diag_cal = weak._prediction_diagnostics_block(
            all_val_pred, scores_col="pain_prob_calibrated", targets_col="target", n_bins=n_bins
        )
    test_diag_raw = weak._prediction_diagnostics_block(test_pred, scores_col="pain_prob", targets_col="target", n_bins=n_bins)
    test_diag_cal: dict[str, Any] = {}
    if "pain_prob_calibrated" in test_pred.columns:
        test_diag_cal = weak._prediction_diagnostics_block(
            test_pred, scores_col="pain_prob_calibrated", targets_col="target", n_bins=n_bins
        )
    diagnostics_path = out_dir / "dann_diagnostics.json"
    n_boot = int(getattr(args, "diag_bootstrap_samples", 0) or 0)
    cow_boot_raw = weak._bootstrap_cow_metrics_ci(test_cow_df, prob_col="pain_prob", seed=int(args.seed), n_boot=n_boot)
    cow_boot_cal = weak._bootstrap_cow_metrics_ci(
        test_calibrated_cow_df, prob_col="pain_prob", seed=int(args.seed) + 1, n_boot=n_boot
    )
    diagnostics_payload = {
        "validation_pool_raw_prob": val_diag_raw,
        "validation_pool_calibrated_prob": val_diag_cal,
        "final_test_raw_prob": test_diag_raw,
        "final_test_calibrated_prob": test_diag_cal,
        "cow_level_bootstrap_final_test_raw_prob": cow_boot_raw,
        "cow_level_bootstrap_final_test_calibrated_prob": cow_boot_cal,
    }
    _write_json(diagnostics_path, diagnostics_payload)

    summary_payload = {
        "run": run_meta,
        "split": target_split,
        "fold_summaries": fold_summaries,
        "final_test": {
            "threshold_policy": str(args.test_threshold_policy),
            "threshold_selection_meta_raw": thr_meta_raw,
            "threshold_selection_meta_calibrated": thr_meta_cal,
            "degenerate_sequence_predictions_raw": deg_raw,
            "degenerate_sequence_predictions_calibrated": deg_cal,
            "threshold_from_validation_mean": test_threshold,
            "temperature_from_validation_mean": test_temperature,
            "sequence_metrics": test_metrics,
            "cow_metrics": test_cow_metrics,
            "calibrated_threshold_from_validation_mean": calibrated_test_threshold,
            "calibrated_sequence_metrics": test_calibrated_metrics,
            "calibrated_cow_metrics": test_calibrated_cow_metrics,
            "diagnostics": diagnostics_payload,
        },
        "artifacts": {
            "split_json": str(split_path),
            "fold_summary_csv": str(fold_summary_path),
            "val_predictions_csv": str(val_predictions_path),
            "test_predictions_csv": str(test_predictions_path),
            "test_cow_aggregates_csv": str(test_cow_path),
            "test_calibrated_cow_aggregates_csv": str(test_calibrated_cow_path),
            "diagnostics_json": str(diagnostics_path),
        },
    }
    summary_path = out_dir / "dann_summary.json"
    _write_json(summary_path, summary_payload)

    report_path = out_dir / "dann_report.md"
    _write_report(
        report_path=report_path,
        args=args,
        split=target_split,
        source_meta=source_meta,
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
            "diagnostics_json": str(diagnostics_path),
            "calibrated_sequence_metrics": json.dumps(test_calibrated_metrics, default=_json_default),
            "calibrated_cow_metrics": json.dumps(test_calibrated_cow_metrics, default=_json_default),
            "diagnostics_raw_json": json.dumps(diagnostics_payload, default=_json_default),
        },
    )
    print(f"Wrote DANN summary: {summary_path}")
    print(f"Wrote DANN diagnostics: {diagnostics_path}")
    print(f"Wrote DANN report: {report_path}")


def _run_dann_eval_only(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    split_path = out_dir / "dann_splits.json"
    summary_path = out_dir / "dann_summary.json"
    if not split_path.is_file():
        raise SystemExit(f"--eval-only requires prior split file: {split_path}")
    target_split = json.loads(split_path.read_text(encoding="utf-8"))
    n_folds = len(target_split.get("folds") or [])
    best_paths = _sorted_fold_ckpts(out_dir, "best_dann.pt")
    if len(best_paths) != n_folds:
        raise SystemExit(
            f"--eval-only: found {len(best_paths)} fold_*/best_dann.pt under {out_dir}, expected {n_folds}."
        )
    fold_summaries: list[dict[str, Any]]
    run_meta: dict[str, Any]
    source_meta: dict[str, Any]
    if summary_path.is_file():
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        fold_summaries = list(prev.get("fold_summaries") or [])
        run_meta = dict(prev.get("run") or {})
        source_meta = dict(run_meta.get("source") or {})
        run_meta["eval_only_rerun_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    else:
        print(f"WARNING: missing {summary_path}; using default thresholds (0.5) and temperature 1.0.")
        fold_summaries = []
        run_meta = {
            "eval_only_rerun_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "manifest_csv": str(args.manifest_csv.resolve()),
            "sequence_root": str(args.sequence_root.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
            "args": vars(args),
        }
        source_meta = {}
    weak = _load_weak_module()
    manifest_csv = args.manifest_csv.resolve()
    target_root = args.sequence_root.resolve()
    target_records = weak._prepare_records(manifest_csv, target_root, str(args.label_column), args)
    mod = _load_train_module(args.train_py)
    _seed_everything(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_predictions_path = out_dir / "dann_predictions.csv"
    all_val_pred = pd.read_csv(val_predictions_path) if val_predictions_path.is_file() else pd.DataFrame()
    print(f"DANN --eval-only: {len(best_paths)} folds -> ensemble test on device {device}")
    _dann_finalize_outputs(
        args=args,
        out_dir=out_dir,
        split_path=split_path,
        run_meta=run_meta,
        weak=weak,
        mod=mod,
        target_split=target_split,
        target_records=target_records,
        target_root=target_root,
        device=device,
        fold_summaries=fold_summaries,
        best_paths=best_paths,
        all_val_pred=all_val_pred,
        source_meta=source_meta,
        write_val_predictions_csv=False,
    )
    return 0


def _write_report(
    *,
    report_path: Path,
    args: argparse.Namespace,
    split: dict[str, Any],
    source_meta: dict[str, Any],
    fold_summary_df: pd.DataFrame,
    test_metrics: dict[str, Any],
    test_cow_metrics: dict[str, Any],
    test_cow_df: pd.DataFrame,
    outputs: dict[str, str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Holstein/Jersey DANN Cow-Held-Out Adaptation\n\n")
        f.write(
            "## Metric roles\n\n"
            "- **UCAPS source validation** columns (`source_task1_*`): true Task1 pain vs no-pain labels from the source project. "
            "These are the only *pain-ground-truth* metrics in this report.\n"
            "- **Holstein validation / test** columns (`val_*`, final test tables): `video_health_status` or chosen label column — "
            "a **weak health proxy**, not veterinary pain scores. Treat AUC/F1 here as proxy-label separation only.\n"
            "- **Calibrated** tables: probabilities after validation-fitted temperature scaling (Guo et al., ICML 2017); "
            "thresholds are chosen on inner validation and applied to the final test without test tuning.\n\n"
        )
        f.write(f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}`\n")
        f.write(f"- Label column: `{args.label_column}` (`Healthy=0`, `Unhealthy=1` weak proxy)\n")
        f.write(f"- Final test cows: `{json.dumps(split['test']['cows'])}`\n")
        f.write(f"- Inner folds: `{len(split['folds'])}` folds x `{args.val_cows_per_fold}` validation cows\n")
        f.write(f"- Source project: `{source_meta.get('source_project_dir')}`\n")
        f.write(f"- Source fold: `{source_meta.get('source_fold')}` | source train n: `{source_meta.get('n_source_train')}`\n")
        f.write("- Task focus: `Task1 pain/no-pain only`; Task2 loss is disabled unless explicitly overridden.\n")
        f.write(f"- Source Task1 sanity floor: `{args.source_task1_sanity_floor}` AUC\n")
        f.write(f"- Source Task1 loss: `{args.source_task1_loss}` | source SupCon weight: `{args.source_supcon_weight}` | class-balanced: `{bool(args.source_class_balanced)}`\n")
        f.write(f"- Domain weight: `{args.domain_weight}` | domain lambda max: `{args.domain_lambda_max}`\n")
        f.write(
            f"- V3.1 DA mode: `{args.da_mode}` | domain warmup epochs: `{args.domain_warmup_epochs}` | "
            f"domain ramp epochs: `{args.domain_ramp_epochs}`\n"
        )
        f.write(f"- Target weak BCE weight: `{args.target_weak_weight}` starting epoch `{args.target_weak_start_epoch}`\n")
        f.write(f"- SSL checkpoint dir: `{args.ssl_checkpoint_dir}`\n\n")
        f.write(
            "These labels are weak disease-context proxies, not veterinary pain scores. "
            "Use this as a domain-adaptation diagnostic, not as validated pain detection.\n\n"
        )
        f.write("## Validation Folds — UCAPS source Task1 vs Holstein proxy\n\n")
        f.write(
            "_Fold table: `source_task1_*` = UCAPS true Task1 sanity track; `val_*` = Holstein proxy; "
            "`checkpoint_selected_from_proxy_fallback` = no epoch passed the source AUC floor so the best proxy epoch was used._\n\n"
        )
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
        if "diagnostics_raw_json" in outputs:
            d = json.loads(outputs["diagnostics_raw_json"])
            f.write("\n\n## Diagnostics (pooled validation / final test)\n\n")
            diag_rows: list[dict[str, Any]] = []
            for name, block in (
                ("validation_raw_prob", d.get("validation_pool_raw_prob")),
                ("validation_calibrated_prob", d.get("validation_pool_calibrated_prob")),
                ("final_test_raw_prob", d.get("final_test_raw_prob")),
                ("final_test_calibrated_prob", d.get("final_test_calibrated_prob")),
            ):
                if not block:
                    continue
                ece_block = block.get("ece") or {}
                diag_rows.append(
                    {
                        "subset": name,
                        "brier": block.get("brier"),
                        "nll": block.get("nll"),
                        "ece": ece_block.get("ece"),
                    }
                )
            if diag_rows:
                f.write(_markdown_table(pd.DataFrame(diag_rows).round(4)))
            boot_rows: list[dict[str, Any]] = []
            for label, block in (
                ("final_test_raw", d.get("cow_level_bootstrap_final_test_raw_prob")),
                ("final_test_calibrated", d.get("cow_level_bootstrap_final_test_calibrated_prob")),
            ):
                if not block or not isinstance(block, dict):
                    continue
                auc_b = block.get("auc")
                if not isinstance(auc_b, dict):
                    continue
                bal_b = block.get("balanced_accuracy_at_0p5") or {}
                boot_rows.append(
                    {
                        "subset": label,
                        "n_boot_ok": block.get("n_boot_successful"),
                        "auc_median": auc_b.get("median"),
                        "auc_ci95_low": auc_b.get("ci95_low"),
                        "auc_ci95_high": auc_b.get("ci95_high"),
                        "bacc_median": bal_b.get("median"),
                        "bacc_ci95_low": bal_b.get("ci95_low"),
                        "bacc_ci95_high": bal_b.get("ci95_high"),
                    }
                )
            if boot_rows:
                f.write("\n### Cow-level bootstrap 95% CI (resample cows)\n\n")
                f.write(_markdown_table(pd.DataFrame(boot_rows).round(4)))
            f.write("\nFull reliability bins and PR-curve samples: see `dann_diagnostics.json`.\n")
        f.write("\n\n## Artifacts\n\n")
        for key, value in outputs.items():
            if key.startswith("calibrated_") or key == "diagnostics_raw_json":
                continue
            f.write(f"- `{key}`: `{value}`\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DANN adaptation from labeled UCAPS source to Holstein/Jersey target.")
    p.add_argument("--manifest-csv", type=Path, default=Path("../cow_face_sequences_10s_250/completed_manifest.csv"))
    p.add_argument("--sequence-root", type=Path, default=Path("../cow_face_sequences_10s_250"))
    p.add_argument("--source-project-dir", type=Path, default=None, help="UCAPS project dir containing train_val_test_splits_v2.json and sequence_label_mapping_v2.json.")
    p.add_argument("--source-sequence-dir", type=Path, default=None, help="UCAPS sequence root. Defaults to parent(source_project_dir)/sequence.")
    p.add_argument("--source-splits-json", type=Path, default=None)
    p.add_argument("--source-mapping-json", type=Path, default=None)
    p.add_argument("--source-fold", type=int, default=None, help="UCAPS fold used for source train/val sanity metrics. Defaults to --init-fold.")
    p.add_argument("--source-use-train-val", action="store_true", help="Use all UCAPS train_val_animals for source training; source val still uses --source-fold val animals.")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional UCAPS v2.9 checkpoint folder for initialization.")
    p.add_argument("--ckpt-kind", choices=("task2", "task1", "combined"), default="task2")
    p.add_argument("--init-fold", type=int, default=0)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--ssl-checkpoint-dir", type=Path, default=None, help="Optional output dir from ssl_pretrain_holstein_v2.9.py.")
    p.add_argument("--ssl-checkpoint-pattern", type=str, default="fold_{fold}/best_ssl_simsiam.pt")
    p.add_argument("--train-py", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("holstein_dann_outputs"))
    p.add_argument("--label-column", choices=("video_health_status", "cow_health_status"), default="video_health_status")
    p.add_argument("--test-cows", type=int, default=4)
    p.add_argument("--test-cow-ids", type=str, default=None)
    p.add_argument("--val-cows-per-fold", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=6,
        help="ReduceLROnPlateau: epochs without validation improvement before LR is reduced (higher = less aggressive decay).",
    )
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--task2-mode", choices=("3class", "4class"), default=None)
    p.add_argument("--freeze-cnn", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--no-stratified-sampler", action="store_true")
    p.add_argument("--use-moment-weighting", action="store_true")
    p.add_argument(
        "--task1-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Task1-only transfer: cfg task2 weight 0; keep Task2 head for checkpoint compatibility. Use --no-task1-only for legacy Task1+Task2 training.",
    )
    p.add_argument("--source-task1-weight", type=float, default=1.0)
    p.add_argument("--source-task2-weight", type=float, default=0.0)
    p.add_argument("--source-task1-loss", choices=("bce", "focal", "gce"), default="bce")
    p.add_argument("--source-class-balanced", action="store_true", help="Use effective-number source Task1 class weights (Cui et al., CVPR 2019).")
    p.add_argument("--class-balanced-beta", type=float, default=0.999)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--gce-q", type=float, default=0.7)
    p.add_argument("--source-supcon-weight", type=float, default=0.0, help="Weight for source supervised contrastive loss on pooled embeddings (Khosla et al., NeurIPS 2020).")
    p.add_argument("--supcon-temperature", type=float, default=0.07)
    p.add_argument("--source-task1-sanity-floor", type=float, default=0.70, help="Reject fold checkpoints unless source Task1 AUC reaches this floor. Set <=0 to disable.")
    p.add_argument("--domain-weight", type=float, default=0.5)
    p.add_argument("--domain-lambda-max", type=float, default=1.0)
    p.add_argument("--domain-hidden-dim", type=int, default=128)
    p.add_argument("--domain-dropout", type=float, default=0.2)
    p.add_argument(
        "--da-mode",
        choices=("dann", "cdan", "mdd"),
        default="dann",
        help="Domain adaptation variant: vanilla DANN, CDAN-style conditional inputs (Long et al. 2018), or MDD-style dual-head discrepancy (Saito et al. 2019).",
    )
    p.add_argument("--domain-warmup-epochs", type=int, default=0, help="Epochs with zero domain-adversarial loss before ramping in.")
    p.add_argument(
        "--domain-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp domain loss weight from 0 to --domain-weight over this many epochs after warmup (0 disables ramp).",
    )
    p.add_argument("--mdd-weight", type=float, default=0.1, help="Weight for MDD discrepancy term when --da-mode mdd.")
    p.add_argument("--mdd-aux-weight", type=float, default=0.5, help="Auxiliary Task1 loss weight on the MDD linear head.")
    p.add_argument("--mdd-disc-weight", type=float, default=1.0, help="Multiplier on target-domain discrepancy (maximized).")
    p.add_argument(
        "--test-threshold-policy",
        choices=("mean_fold_best_f1", "median_fold_best_f1", "pooled_val_f1_opt", "fixed"),
        default="mean_fold_best_f1",
        help="How to set final-test thresholds from validation (V3.1). Default matches legacy mean-of-folds.",
    )
    p.add_argument("--test-threshold-fixed", type=float, default=0.5, help="Used when --test-threshold-policy fixed.")
    p.add_argument(
        "--manifest-min-mean-detection-confidence",
        type=float,
        default=None,
        help="Optional QC: drop Holstein manifest rows below this mean YOLO confidence.",
    )
    p.add_argument("--manifest-max-filled-frames", type=int, default=None, help="Optional QC: drop rows with more filled frames.")
    p.add_argument("--manifest-min-detected-frames", type=int, default=None, help="Optional QC: require at least this many detected frames.")
    p.add_argument(
        "--manifest-write-qc-audit",
        action="store_true",
        help="When QC filters are set, write holstein_manifest_qc_audit.csv under --out-dir.",
    )
    p.add_argument("--target-weak-weight", type=float, default=0.0)
    p.add_argument("--target-weak-start-epoch", type=int, default=5)
    p.add_argument("--ramp-target-weak", action="store_true")
    p.add_argument("--select-metric", choices=("auc", "f1_opt", "f1"), default="auc")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--infer-sliding-raw-span", type=int, default=None)
    p.add_argument("--infer-sliding-stride", type=int, default=None)
    p.add_argument("--infer-sliding-aggregate", choices=("mean", "trimmed_mean", "max"), default="mean")
    p.add_argument("--eval-mc-samples", type=int, default=0)
    p.add_argument("--diag-ece-bins", type=int, default=15)
    p.add_argument(
        "--diag-bootstrap-samples",
        type=int,
        default=2000,
        help="Cow bootstrap draws for 95%% CI on cow-level metrics (0 disables).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: load fold_*/best_dann.pt, re-run ensemble test and rewrite reports. Requires out-dir with dann_splits.json; dann_summary.json supplies val thresholds if present.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(getattr(args, "eval_only", False)):
        if bool(args.dry_run):
            raise SystemExit("Choose either --eval-only or --dry-run, not both.")
        return _run_dann_eval_only(args)
    manifest_csv = args.manifest_csv.resolve()
    target_root = args.sequence_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weak = _load_weak_module()
    target_records = weak._prepare_records(manifest_csv, target_root, str(args.label_column), args)
    target_split = weak._build_split_plan(target_records, _make_split_args(args))
    weak._print_split_plan(target_split)
    split_path = out_dir / "dann_splits.json"
    _write_json(split_path, target_split)
    if bool(args.dry_run):
        _write_json(out_dir / "dann_dry_run.json", {"split_json": str(split_path), "source_project_dir": str(args.source_project_dir)})
        print("Dry run complete: no DANN training was started.")
        return 0

    if args.checkpoint_dir is None and not bool(args.from_scratch):
        raise ValueError("Pass --checkpoint-dir for UCAPS initialization, or use --from-scratch.")

    mod = _load_train_module(args.train_py)
    _seed_everything(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    source_train, source_val, source_root, source_meta = _load_source_bundle(mod, args)

    init_ckpt = None
    init_ckpt_path = None
    if not bool(args.from_scratch):
        assert args.checkpoint_dir is not None
        init_ckpt_path = _checkpoint_path(args.checkpoint_dir.resolve(), int(args.init_fold), str(args.ckpt_kind))
        init_ckpt = _load_checkpoint(init_ckpt_path)
        print(f"Initializing DANN folds from: {init_ckpt_path}")

    base_cfg = _cfg_from_checkpoint_or_default(mod, init_ckpt)
    base_cfg = _apply_cfg_overrides(base_cfg, args)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "manifest_csv": str(manifest_csv),
        "sequence_root": str(target_root),
        "source": source_meta,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "init_ckpt_path": str(init_ckpt_path) if init_ckpt_path else None,
        "ssl_checkpoint_dir": str(args.ssl_checkpoint_dir.resolve()) if args.ssl_checkpoint_dir else None,
        "args": vars(args),
        "cfg": _cfg_to_dict(base_cfg),
    }
    _write_json(out_dir / "dann_run.json", run_meta)

    fold_summaries: list[dict[str, Any]] = []
    val_predictions: list[pd.DataFrame] = []
    best_paths: list[Path] = []
    for fold in target_split["folds"]:
        print("\n" + "=" * 80)
        print(f"Training DANN fold {fold['fold']}")
        print("=" * 80)
        summary, pred_df, best_path = _train_one_fold(
            mod=mod,
            weak=weak,
            base_cfg=base_cfg,
            args=args,
            fold=fold,
            target_records=target_records,
            target_split=target_split,
            source_train=source_train,
            source_val=source_val,
            source_root=source_root,
            target_root=target_root,
            device=device,
            init_ckpt=init_ckpt,
            init_ckpt_path=init_ckpt_path,
            run_dir=out_dir,
        )
        fold_summaries.append(summary)
        val_predictions.append(pred_df)
        best_paths.append(best_path)

    all_val_pred = pd.concat(val_predictions, ignore_index=True) if val_predictions else pd.DataFrame()
    _dann_finalize_outputs(
        args=args,
        out_dir=out_dir,
        split_path=split_path,
        run_meta=run_meta,
        weak=weak,
        mod=mod,
        target_split=target_split,
        target_records=target_records,
        target_root=target_root,
        device=device,
        fold_summaries=fold_summaries,
        best_paths=best_paths,
        all_val_pred=all_val_pred,
        source_meta=source_meta,
        write_val_predictions_csv=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
