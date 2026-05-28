#!/usr/bin/env python3
"""
Split-aware self-supervised pretraining for UCAPS v2.9 on Holstein/Jersey face clips.

Default objective: SimSiam over two sequence-consistent augmented views of each 10s clip.
The script writes one SSL checkpoint per cow-held-out fold, using only that fold's training
cows by default to avoid validation/test leakage.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
    return _load_module(HERE / "weak_label_adapt_v3.py", "weak_label_adapt_v3")


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
    if hasattr(cfg, "use_augmentations"):
        cfg.use_augmentations = not bool(args.no_aug)
    if hasattr(cfg, "temporal_sampling"):
        cfg.temporal_sampling = str(args.temporal_sampling)
    if hasattr(cfg, "time_reverse_p"):
        cfg.time_reverse_p = float(args.time_reverse_p)
    return cfg


def _freeze_for_ssl(model: nn.Module, freeze_cnn: bool) -> None:
    if freeze_cnn and hasattr(model, "cnn"):
        for p in model.cnn.parameters():
            p.requires_grad = False


def _feature_dim(model: nn.Module, cfg: Any) -> int:
    return int(cfg.lstm_hidden_size) * (2 if bool(getattr(cfg, "use_bidirectional_lstm", False)) else 1)


class TwoViewDataset(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        x1, _y1, meta = self.base_ds[idx]
        x2, _y2, _meta2 = self.base_ds[idx]
        return x1, x2, meta


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, predictor: bool = False):
        super().__init__()
        if predictor:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim, affine=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _simsiam_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def _make_loader(mod: Any, records: list[dict[str, Any]], sequence_root: Path, cfg: Any) -> DataLoader:
    DatasetCls = getattr(mod, "FacialPainDataset_v2_9", None) or getattr(mod, "FacialPainDataset_v2_8")
    base_ds = DatasetCls([r["seq"] for r in records], sequence_root, cfg, augment=True, global_cache={})
    ds = TwoViewDataset(base_ds)
    loader_extra: dict[str, Any] = {}
    if int(cfg.num_workers) > 0:
        loader_extra["prefetch_factor"] = 2
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **loader_extra,
    )


def _select_records_for_fold(weak: Any, records: list[dict[str, Any]], split: dict[str, Any], fold: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    if bool(args.transductive_target_ssl):
        return "transductive_all_target_cows", records
    scope = str(args.ssl_cow_scope)
    if scope == "fold_train":
        return "fold_train_cows", weak._records_for_cows(records, fold["train"]["cows"])
    if scope == "train_pool":
        return "train_pool_cows", weak._records_for_cows(records, split["train_pool"]["cows"])
    raise ValueError(f"Unknown ssl cow scope: {scope}")


def _make_split_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        label_column=args.label_column,
        dataset_version=getattr(args, "dataset_version", ""),
        test_cows=args.test_cows,
        test_cow_ids=args.test_cow_ids,
        val_cows_per_fold=args.val_cows_per_fold,
        require_val_both_classes=getattr(args, "require_val_both_classes", True),
        seed=args.seed,
    )


def _fold_indices(split: dict[str, Any], only_fold: int | None) -> Iterable[dict[str, Any]]:
    for fold in split["folds"]:
        if only_fold is not None and int(fold["fold"]) != int(only_fold):
            continue
        yield fold


def _save_ssl_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    cfg: Any,
    fold: dict[str, Any],
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    scope_name: str,
    init_ckpt_path: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": "v3_holstein_ssl_simsiam",
            "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "fold": int(fold["fold"]),
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "projector_state_dict": {k: v.detach().cpu() for k, v in projector.state_dict().items()},
            "predictor_state_dict": {k: v.detach().cpu() for k, v in predictor.state_dict().items()},
            "cfg": _cfg_to_dict(cfg),
            "split": fold,
            "history": history,
            "ssl_objective": "simsiam",
            "ssl_cow_scope": scope_name,
            "init_ckpt_path": str(init_ckpt_path) if init_ckpt_path else None,
            "args": vars(args),
        },
        path,
    )


def _train_one_fold(
    *,
    mod: Any,
    weak: Any,
    base_cfg: Any,
    init_ckpt: dict[str, Any] | None,
    init_ckpt_path: Path | None,
    records: list[dict[str, Any]],
    split: dict[str, Any],
    fold: dict[str, Any],
    sequence_root: Path,
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fold_idx = int(fold["fold"])
    _seed_everything(int(args.seed) + fold_idx)

    cfg = deepcopy(base_cfg)
    cfg = _apply_cfg_overrides(cfg, args)

    scope_name, ssl_records = _select_records_for_fold(weak, records, split, fold, args)
    if len(ssl_records) < int(args.batch_size):
        raise RuntimeError(f"Fold {fold_idx} has only {len(ssl_records)} SSL records; reduce --batch-size.")

    ModelCls = getattr(mod, "TemporalPainModel_v2_9", None) or getattr(mod, "TemporalPainModel_v2_8")
    model = ModelCls(cfg).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model_state_dict"], strict=True)
    if not hasattr(model, "extract_features"):
        raise AttributeError("TemporalPainModel_v2_9 must expose extract_features() for SSL pretraining.")
    _freeze_for_ssl(model, bool(args.freeze_cnn))

    feat_dim = _feature_dim(model, cfg)
    projector = MLP(feat_dim, int(args.projector_hidden_dim), int(args.projection_dim)).to(device)
    predictor = MLP(int(args.projection_dim), int(args.predictor_hidden_dim), int(args.projection_dim), predictor=True).to(device)

    params = [p for p in list(model.parameters()) + list(projector.parameters()) + list(predictor.parameters()) if p.requires_grad]
    optimizer = AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay or 0.0))
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not bool(args.no_amp)) else None
    loader = _make_loader(mod, ssl_records, sequence_root, cfg)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    fold_dir = out_dir / f"fold_{fold_idx}"
    best_path = fold_dir / "best_ssl_simsiam.pt"

    for epoch in range(int(args.num_epochs)):
        model.train()
        projector.train()
        predictor.train()
        if bool(args.freeze_cnn) and hasattr(model, "cnn"):
            model.cnn.eval()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, (x1, x2, _meta) in enumerate(tqdm(loader, desc=f"SSL fold {fold_idx}", leave=False, ascii=True)):
            if args.max_train_batches is not None and batch_idx >= int(args.max_train_batches):
                break
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                h1, _ = model.extract_features(x1, apply_dropout=False)
                h2, _ = model.extract_features(x2, apply_dropout=False)
                z1 = projector(h1)
                z2 = projector(h2)
                p1 = predictor(z1)
                p2 = predictor(z2)
                loss = 0.5 * (_simsiam_loss(p1, z2) + _simsiam_loss(p2, z1))
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, float(args.grad_clip))
                optimizer.step()
            total_loss += float(loss.detach().item())
            n_batches += 1

        mean_loss = total_loss / max(1, n_batches)
        row = {
            "fold": fold_idx,
            "epoch": epoch,
            "ssl_loss": float(mean_loss),
            "n_batches": int(n_batches),
            "n_sequences": int(len(ssl_records)),
            "ssl_cow_scope": scope_name,
        }
        history.append(row)
        print(f"fold {fold_idx} epoch {epoch + 1}/{args.num_epochs}: ssl_loss={mean_loss:.4f} records={len(ssl_records)} scope={scope_name}")
        if mean_loss < best_loss:
            best_loss = float(mean_loss)
            _save_ssl_checkpoint(
                path=best_path,
                model=model,
                projector=projector,
                predictor=predictor,
                cfg=cfg,
                fold=fold,
                history=history,
                args=args,
                scope_name=scope_name,
                init_ckpt_path=init_ckpt_path,
            )

    fold_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(history).to_csv(fold_dir / "ssl_history.csv", index=False)
    except Exception:
        _write_json(fold_dir / "ssl_history.json", {"history": history})

    return {
        "fold": fold_idx,
        "best_loss": best_loss,
        "best_checkpoint": str(best_path),
        "ssl_cow_scope": scope_name,
        "n_ssl_sequences": int(len(ssl_records)),
        "train_cows": fold["train"]["cows"],
        "val_cows": fold["val"]["cows"],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-supervised Holstein/Jersey pretraining for UCAPS v2.9.")
    p.add_argument("--manifest-csv", type=Path, default=Path("../cow_face_sequences_10s_250/completed_manifest.csv"))
    p.add_argument("--sequence-root", type=Path, default=Path("../cow_face_sequences_10s_250"))
    p.add_argument("--dataset-version", type=str, default="baseline_10s_250")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional UCAPS v2.9 checkpoint folder for initialization.")
    p.add_argument("--ckpt-kind", choices=("task2", "task1", "combined"), default="task2")
    p.add_argument("--init-fold", type=int, default=0)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--train-py", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("holstein_ssl_outputs"))
    p.add_argument("--label-column", choices=("video_health_status", "cow_health_status"), default="video_health_status")
    p.add_argument("--test-cows", type=int, default=4)
    p.add_argument("--test-cow-ids", type=str, default=None)
    p.add_argument("--val-cows-per-fold", type=int, default=4)
    p.add_argument("--require-val-both-classes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ssl-cow-scope", choices=("fold_train", "train_pool"), default="fold_train")
    p.add_argument("--transductive-target-ssl", action="store_true", help="Research-only: include all target cows, including validation/test.")
    p.add_argument("--fold", type=int, default=None, help="Optional single fold to pretrain.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--projector-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--predictor-hidden-dim", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--task2-mode", choices=("3class", "4class"), default=None)
    p.add_argument("--temporal-sampling", choices=("linspace", "uniform_offset", "random_clip"), default="uniform_offset")
    p.add_argument("--time-reverse-p", type=float, default=0.1)
    p.add_argument("--freeze-cnn", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--threshold-min-specificity", type=float, default=0.50, help=argparse.SUPPRESS)
    p.add_argument("--diag-bootstrap-samples", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--expected-frames", type=int, default=240)
    p.add_argument("--qa-min-detection-rate", type=float, default=0.0)
    p.add_argument("--qa-max-filled-rate", type=float, default=1.0)
    p.add_argument("--qa-min-mean-confidence", type=float, default=0.0)
    p.add_argument("--qa-min-min-confidence", type=float, default=0.0)
    p.add_argument("--qa-audit-csv", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_csv = args.manifest_csv.resolve()
    sequence_root = args.sequence_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weak = _load_weak_module()
    records = weak._prepare_records(manifest_csv, sequence_root, str(args.label_column))
    records, qa_audit_df, qa_summary = weak._qa_filter_records(records, args)
    qa_audit_path = args.qa_audit_csv.resolve() if args.qa_audit_csv else out_dir / "sequence_qa_audit.csv"
    qa_audit_path.parent.mkdir(parents=True, exist_ok=True)
    qa_audit_df.to_csv(qa_audit_path, index=False)
    split = weak._build_split_plan(records, _make_split_args(args))
    weak._print_split_plan(split)
    split_path = out_dir / "ssl_splits.json"
    _write_json(split_path, split)
    print(f"Wrote QA audit: {qa_audit_path}")

    selected_folds = list(_fold_indices(split, args.fold))
    if not selected_folds:
        raise ValueError(f"No fold matched --fold={args.fold}")
    if bool(args.dry_run):
        payload = {
            "split_json": str(split_path),
            "selected_folds": [int(f["fold"]) for f in selected_folds],
            "ssl_cow_scope": "transductive_all_target_cows" if args.transductive_target_ssl else args.ssl_cow_scope,
            "qa_summary": qa_summary,
            "qa_audit_csv": str(qa_audit_path),
        }
        _write_json(out_dir / "ssl_dry_run.json", payload)
        print("Dry run complete: no SSL training was started.")
        return 0

    if args.checkpoint_dir is None and not bool(args.from_scratch):
        raise ValueError("Pass --checkpoint-dir for UCAPS initialization, or use --from-scratch.")

    mod = _load_train_module(args.train_py)
    _seed_everything(int(args.seed))

    init_ckpt = None
    init_ckpt_path = None
    if not bool(args.from_scratch):
        assert args.checkpoint_dir is not None
        init_ckpt_path = _checkpoint_path(args.checkpoint_dir.resolve(), int(args.init_fold), str(args.ckpt_kind))
        init_ckpt = _load_checkpoint(init_ckpt_path)
        print(f"Initializing SSL model from: {init_ckpt_path}")

    base_cfg = _cfg_from_checkpoint_or_default(mod, init_ckpt)
    base_cfg = _apply_cfg_overrides(base_cfg, args)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "manifest_csv": str(manifest_csv),
        "sequence_root": str(sequence_root),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "init_ckpt_path": str(init_ckpt_path) if init_ckpt_path else None,
        "args": vars(args),
        "cfg": _cfg_to_dict(base_cfg),
        "qa_summary": qa_summary,
        "qa_audit_csv": str(qa_audit_path),
    }
    _write_json(out_dir / "ssl_run.json", run_meta)

    summaries: list[dict[str, Any]] = []
    for fold in selected_folds:
        print("\n" + "=" * 80)
        print(f"SSL pretraining fold {fold['fold']}")
        print("=" * 80)
        summaries.append(
            _train_one_fold(
                mod=mod,
                weak=weak,
                base_cfg=base_cfg,
                init_ckpt=init_ckpt,
                init_ckpt_path=init_ckpt_path,
                records=records,
                split=split,
                fold=fold,
                sequence_root=sequence_root,
                out_dir=out_dir,
                device=device,
                args=args,
            )
        )

    summary_path = out_dir / "ssl_summary.json"
    _write_json(
        summary_path,
        {
            "run": run_meta,
            "split": split,
            "fold_summaries": summaries,
            "artifacts": {
                "split_json": str(split_path),
                "summary_json": str(summary_path),
            },
        },
    )
    print(f"Wrote SSL summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
