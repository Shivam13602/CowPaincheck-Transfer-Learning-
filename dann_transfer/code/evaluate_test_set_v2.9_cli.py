# ============================================================================
# TEST SET EVALUATION - v2.9 (Dual Classification)
#
# Evaluates v2.9 checkpoints (best_model_v2.9*_fold_*.pt) on held-out test animals
# from train_val_test_splits_v2.json (default: [14, 17]).
#
# - Ensembling across folds: mean_logits (recommended) or majority vote
# - Colab/Jupyter friendly: ignores unknown CLI args (e.g. -f kernel.json)
# - Reuses the exact v2.9 dataset/model definitions by importing:
#   v2.9_training_classification.py (loaded via importlib)
# ============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms.functional as TF


# ----------------------------------------------------------------------------
# Standalone fallback (no training .py file available)
#
# If `v2.9_training_classification.py` is not present on disk, these minimal
# definitions allow evaluation purely from checkpoints + mapping JSONs.
# They match the v2.9 CNN+LSTM+attention architecture and eval-time dataset
# behavior (deterministic linspace sampling when augment=False).
# ----------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def task2_num_classes(task2_mode: str) -> int:
    if task2_mode == "3class":
        return 3
    if task2_mode == "4class":
        return 4
    raise ValueError(f"Unknown task2_mode={task2_mode!r}")


def moment_to_task1_binary(moment: str) -> int:
    return 1 if str(moment).strip().upper() in {"M2", "M3", "M4"} else 0


def moment_to_task2(moment: str, *, task2_mode: str) -> int:
    m = str(moment).strip().upper()
    if task2_mode == "3class":
        if m in {"M0", "M1"}:
            return 0
        if m == "M2":
            return 1
        if m in {"M3", "M4"}:
            return 2
        return 0
    if task2_mode == "4class":
        if m in {"M0", "M1"}:
            return 0
        if m == "M2":
            return 1
        if m == "M3":
            return 2
        if m == "M4":
            return 3
        return 0
    raise ValueError(f"Unknown task2_mode={task2_mode!r}")


def normalize_clip_id(s: str) -> str:
    s = str(s).replace("\\", "/").strip()
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"[^a-zA-Z0-9/_\\-\\.]+", "_", s)
    return s.strip("/").strip()


@dataclass
class Config:
    # Data
    task2_mode: str = "3class"
    input_mode: str = "rgb"  # rgb or grayst
    max_frames: int = 32
    resolution: Tuple[int, int] = (112, 112)
    temporal_sampling: str = "linspace"  # linspace/uniform_offset/random_clip
    time_reverse_p: float = 0.0

    # Loader
    batch_size: int = 16
    num_workers: int = 2

    # Model
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    use_bidirectional_lstm: bool = False
    dropout: float = 0.3


class FacialPainDataset_v2_9(Dataset):
    def __init__(
        self,
        sequences: List[dict],
        sequence_dir: Path,
        cfg: Config,
        *,
        augment: bool,
        global_cache: Optional[dict] = None,
    ):
        self.sequences = sequences
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.augment = bool(augment)
        self.global_cache = global_cache if global_cache is not None else {}

    def __len__(self) -> int:
        return len(self.sequences)

    def _cache_key(self, seq_info: dict, idx: int) -> str:
        if seq_info.get("sequence_path"):
            return str(seq_info["sequence_path"])
        seq_id = seq_info.get("sequence_id", "seq")
        animal = seq_info.get("animal", seq_info.get("animal_id", "unknown"))
        moment = seq_info.get("moment", "unknown")
        return f"{seq_id}_{animal}_{moment}_{idx}"

    def _find_frames_path(self, seq_info: dict) -> Optional[Path]:
        if seq_info.get("sequence_path"):
            seq_path = self.sequence_dir / str(seq_info["sequence_path"]).replace("\\", "/")
        elif seq_info.get("sequence_id"):
            seq_path = self.sequence_dir / str(seq_info["sequence_id"]).replace("\\", "/")
        else:
            return None

        if seq_path.exists():
            frames = sorted(list(seq_path.glob("*.jpg")) + list(seq_path.glob("*.png")))
            if frames:
                return seq_path

        for subdir_name in ["sequence_001", "frames", "images"]:
            subdir = seq_path / subdir_name
            if subdir.exists():
                frames = sorted(list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")))
                if frames:
                    return subdir
        return None

    def _select_indices(self, n: int, need: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((need,), dtype=np.int64)
        if n == 1:
            return np.zeros((need,), dtype=np.int64)

        # Deterministic baseline for val/test
        if (not self.augment) or (str(self.cfg.temporal_sampling) == "linspace"):
            return np.linspace(0, n - 1, need, dtype=np.int64)

        if str(self.cfg.temporal_sampling) == "random_clip":
            if n >= need:
                start = random.randint(0, n - need)
                return (start + np.arange(need, dtype=np.int64)).astype(np.int64)
            idxs = np.arange(n, dtype=np.int64)
            pad = np.full((need - n,), n - 1, dtype=np.int64)
            return np.concatenate([idxs, pad], axis=0)

        # uniform_offset sampling
        step = max(1, (n - 1) // max(1, (need - 1)))
        max_offset = max(0, (n - 1) - (step * (need - 1)))
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        idxs = offset + (step * np.arange(need, dtype=np.int64))
        return np.clip(idxs, 0, n - 1).astype(np.int64)

    def __getitem__(self, idx: int):
        seq_info = self.sequences[idx]
        cache_key = self._cache_key(seq_info, idx)

        if cache_key in self.global_cache:
            frame_dir = self.global_cache[cache_key].get("path")
            frame_files = self.global_cache[cache_key].get("files")
        else:
            frame_dir = self._find_frames_path(seq_info)
            if frame_dir and frame_dir.exists():
                frame_files = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
                frame_files = frame_files if frame_files else None
            else:
                frame_files = None
            self.global_cache[cache_key] = {"path": frame_dir, "files": frame_files}

        moment = str(seq_info.get("moment", "unknown"))
        animal = int(seq_info.get("animal", seq_info.get("animal_id", -1)))
        seq_id = str(seq_info.get("sequence_id", f"seq_{idx}"))
        seq_path = str(seq_info.get("sequence_path", "")) if "sequence_path" in seq_info else ""
        clip_id = normalize_clip_id(seq_path or seq_id)

        y1 = moment_to_task1_binary(moment)
        y2 = moment_to_task2(moment, task2_mode=str(self.cfg.task2_mode))

        decode_errors = 0
        n_available_frames = int(len(frame_files)) if isinstance(frame_files, list) else 0

        need_raw = int(self.cfg.max_frames) + (2 if str(self.cfg.input_mode) == "grayst" else 0)
        if frame_files is None or len(frame_files) == 0:
            dummy = Image.new("RGB", tuple(self.cfg.resolution), color="black")
            frames = [dummy] * need_raw
            is_dummy = True
        else:
            is_dummy = False
            idxs = self._select_indices(len(frame_files), need_raw)
            if self.augment and (random.random() < float(self.cfg.time_reverse_p)):
                idxs = idxs[::-1].copy()

            selected = [frame_files[int(i)] for i in idxs.tolist()]
            frames = []
            last_ok: Optional[Image.Image] = None
            for fp in selected:
                try:
                    img = Image.open(fp).convert("RGB")
                    frames.append(img)
                    last_ok = img
                except Exception:
                    decode_errors += 1
                    frames.append(last_ok if last_ok is not None else Image.new("RGB", tuple(self.cfg.resolution), color="black"))

        if str(self.cfg.input_mode) == "grayst":
            g = [im.convert("L") for im in frames]
            stacked: List[Image.Image] = []
            for i in range(int(self.cfg.max_frames)):
                stacked.append(Image.merge("RGB", (g[i], g[i + 1], g[i + 2])))
            frames = stacked

        tensors: List[torch.Tensor] = []
        for im in frames:
            im = TF.resize(im, tuple(self.cfg.resolution))
            t = TF.to_tensor(im)
            t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tensors.append(t)
        x = torch.stack(tensors, dim=0)  # (T,C,H,W)

        labels = {
            "pain_binary": torch.tensor(float(y1), dtype=torch.float32),
            "task2": torch.tensor(int(y2), dtype=torch.long),
        }
        meta = {
            "animal": animal,
            "moment": moment,
            "sequence_id": seq_id,
            "sequence_path": seq_path,
            "clip_id": clip_id,
            "is_dummy": bool(is_dummy),
            "n_available_frames": int(n_available_frames),
            "n_decode_errors": int(decode_errors),
        }
        return x, labels, meta


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(int(hidden_size), 1)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.attn(lstm_out), dim=1)
        ctx = torch.sum(w * lstm_out, dim=1)
        return ctx, w


class TemporalPainModel_v2_9(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.task2_num_classes = task2_num_classes(str(cfg.task2_mode))

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cnn_output_size = 256

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=int(cfg.lstm_hidden_size),
            num_layers=int(cfg.lstm_num_layers),
            batch_first=True,
            bidirectional=bool(cfg.use_bidirectional_lstm),
            dropout=float(cfg.dropout) if int(cfg.lstm_num_layers) > 1 else 0.0,
        )
        lstm_out_size = int(cfg.lstm_hidden_size) * (2 if bool(cfg.use_bidirectional_lstm) else 1)
        self.attn = AttentionLayer(lstm_out_size)
        self.dropout = nn.Dropout(float(cfg.dropout))

        self.head_task1 = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(64, 1),
        )
        self.head_task2 = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(64, int(self.task2_num_classes)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        B, T_, C, H, W = x.shape
        x = x.view(B * T_, C, H, W)
        f = self.cnn(x).view(B * T_, -1)
        f = f.view(B, T_, self.cnn_output_size)

        lstm_out, _ = self.lstm(f)
        ctx, attn_w = self.attn(lstm_out)
        ctx = self.dropout(ctx)
        out = {
            "pain_logits": self.head_task1(ctx).squeeze(-1),
            "task2_logits": self.head_task2(ctx),
        }
        return out, attn_w


def _running_in_notebook() -> bool:
    try:
        import sys

        return ("ipykernel" in sys.modules) or ("google.colab" in sys.modules)
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate UCAPS v2.9 checkpoints on held-out test animals.")

    p.add_argument(
        "--train_py",
        type=str,
        default=None,
        help="Path to v2.9_training_classification.py (useful in notebooks where __file__ is undefined).",
    )

    p.add_argument("--base_path", type=str, default=None, help="Base path (Colab: /content/drive/MyDrive).")
    p.add_argument("--project_dir", type=str, default=None, help="Directory containing splits/mapping JSONs.")
    p.add_argument("--sequence_dir", type=str, default=None, help="Directory containing extracted sequences/frames.")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing v2.9 checkpoints.")
    p.add_argument("--run_tag", type=str, default=None, help="Optional subfolder under checkpoints/results.")
    p.add_argument(
        "--ckpt_kind",
        choices=["combined", "task1", "task2", "all"],
        default="combined",
        help="Which best checkpoint to evaluate: combined (default), task1, task2, or all (runs all three).",
    )

    p.add_argument("--test_animals", type=int, nargs="+", default=None, help="Override test animals list.")
    p.add_argument("--folds", type=int, nargs="*", default=None, help="Specific fold indices to evaluate.")
    p.add_argument(
        "--ensemble",
        choices=["mean_logits", "majority"],
        default="mean_logits",
        help="Ensemble method across folds.",
    )

    p.add_argument("--batch_size", type=int, default=None, help="Override batch size (default: from checkpoint cfg).")
    p.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers (default: from cfg).")
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or leave unset for auto.")
    p.add_argument("--max_batches", type=int, default=None, help="Limit number of eval batches (for quick smoke tests).")
    p.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="List available v2.9 checkpoints under --checkpoint_dir and exit.",
    )

    p.add_argument(
        "--task1_threshold_logit",
        type=float,
        default=0.0,
        help="Task1 threshold on logits (default 0.0 == prob 0.5).",
    )
    p.add_argument(
        "--optimize_task1_threshold",
        action="store_true",
        help="Also compute best Task1 F1 over a grid of probability thresholds (diagnostic; not a fair test metric).",
    )

    p.add_argument("--save_dir", type=str, default=None, help="Directory to write CSV/JSON outputs (default: results_v2.9).")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"WARNING: Ignoring unknown CLI args: {unknown}")
    return args


def _maybe_apply_env_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """
    Colab-friendly: if you paste this file into a notebook cell (no `!python ...`),
    you can't pass CLI args. This lets you control common options via env vars.

    Supported env vars:
    - UCAPS_V2_9_RUN_TAG
    - UCAPS_V2_9_CKPT_KIND: combined|task1|task2|all
    - UCAPS_V2_9_ENSEMBLE: mean_logits|majority
    - UCAPS_V2_9_TEST_ANIMALS: e.g. "14,17" or "14 17"
    - UCAPS_V2_9_TASK1_THRESHOLD_LOGIT: e.g. "0.0"
    """

    def _get(name: str) -> Optional[str]:
        v = os.environ.get(name)
        v = v.strip() if isinstance(v, str) else None
        return v or None

    run_tag = _get("UCAPS_V2_9_RUN_TAG")
    if run_tag and not args.run_tag:
        args.run_tag = run_tag

    ckpt_kind = _get("UCAPS_V2_9_CKPT_KIND")
    if ckpt_kind and (not getattr(args, "ckpt_kind", None) or str(args.ckpt_kind) == "combined"):
        args.ckpt_kind = str(ckpt_kind).strip()

    ensemble = _get("UCAPS_V2_9_ENSEMBLE")
    if ensemble and (not getattr(args, "ensemble", None) or str(args.ensemble) == "mean_logits"):
        args.ensemble = str(ensemble).strip()

    test_animals = _get("UCAPS_V2_9_TEST_ANIMALS")
    if test_animals and args.test_animals is None:
        parts = re.split(r"[,\s]+", test_animals.strip())
        parts = [p for p in parts if p]
        try:
            args.test_animals = [int(p) for p in parts]
        except Exception:
            print(f"WARNING: Could not parse UCAPS_V2_9_TEST_ANIMALS={test_animals!r} (expected ints).")

    thr = _get("UCAPS_V2_9_TASK1_THRESHOLD_LOGIT")
    if thr and float(getattr(args, "task1_threshold_logit", 0.0)) == 0.0:
        try:
            args.task1_threshold_logit = float(thr)
        except Exception:
            print(f"WARNING: Could not parse UCAPS_V2_9_TASK1_THRESHOLD_LOGIT={thr!r} (expected float).")

    return args


def _load_v2_9_module(train_py: Optional[str], *, search_dirs: Optional[List[Path]] = None) -> Any:
    this_file = globals().get("__file__")
    base_dir = Path(this_file).resolve().parent if this_file else Path.cwd()

    def _norm(p: Path) -> Path:
        try:
            return p.expanduser()
        except Exception:
            return p

    candidates: List[Path] = []
    if train_py:
        p = _norm(Path(train_py))
        candidates.append(p if p.is_absolute() else (base_dir / p))
    else:
        candidates.extend(
            [
                base_dir / "v2.9_training_classification.py",
                base_dir / "Ucaps_raw_videos" / "v2.9_training_classification.py",
            ]
        )

    if search_dirs:
        for d in search_dirs:
            if not d:
                continue
            d = _norm(Path(d))
            candidates.append(d / "v2.9_training_classification.py")
            candidates.append(d / "Ucaps_raw_videos" / "v2.9_training_classification.py")
            candidates.append(d / "ucaps_raw_videos" / "v2.9_training_classification.py")

    # Colab-friendly common locations (only checked if they exist)
    candidates.extend(
        [
            Path("/content/drive/MyDrive/facial_pain_project_v2/v2.9_training_classification.py"),
            Path("/content/drive/MyDrive/facial_pain_project_v2/Ucaps_raw_videos/v2.9_training_classification.py"),
            Path("/content/drive/MyDrive/VIDEOS FACIAL BOVINE/Ucaps_raw_videos/v2.9_training_classification.py"),
        ]
    )

    train_path = None
    for c in candidates:
        try:
            if c.exists():
                train_path = c
                break
        except Exception:
            continue

    # Fallback: shallow recursive search under provided search_dirs (avoid scanning entire Drive).
    if train_path is None and search_dirs:
        for root in search_dirs:
            try:
                root = _norm(Path(root))
                if not root.exists() or not root.is_dir():
                    continue
            except Exception:
                continue

            try:
                # Limit depth to keep this fast/safe.
                root_parts = len(root.resolve().parts)
                for p in root.rglob("v2.9_training_classification.py"):
                    try:
                        if len(p.resolve().parts) - root_parts <= 4:
                            train_path = p
                            break
                    except Exception:
                        continue
                if train_path is not None:
                    break
            except Exception:
                continue

    if train_path is None:
        # Notebook fallback: if the training code was executed in the same kernel (pasted into a Colab cell),
        # the needed classes may already exist in __main__.
        try:
            import sys

            main_mod = sys.modules.get("__main__")
            if main_mod is not None:
                    required = ("Config", "FacialPainDataset_v2_9", "TemporalPainModel_v2_9")
                    if all(hasattr(main_mod, name) for name in required):
                        print(
                            "INFO: Using v2.9 classes from the active notebook kernel (__main__) "
                            "because v2.9_training_classification.py was not found on disk."
                        )
                        return main_mod
        except Exception:
            pass

        unique = []
        seen = set()
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                unique.append(s)
        raise FileNotFoundError(
            "Could not locate `v2.9_training_classification.py`.\n"
            "Fix: pass `--train_py /full/path/to/v2.9_training_classification.py`.\n"
            f"Tried:\n- " + "\n- ".join(unique[:20])
        )

    spec = importlib.util.spec_from_file_location("ucaps_v2_9_training", str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for: {train_path}")

    # IMPORTANT: register in sys.modules before exec_module so dataclasses can
    # resolve cls.__module__ during @dataclass processing.
    import sys

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _detect_base_path() -> Path:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive")
    except Exception:
        this_file = globals().get("__file__")
        if this_file:
            return Path(this_file).resolve().parents[1]
        return Path.cwd()


def _resolve_project_dir(base_path: Path) -> Path:
    candidates = [base_path / "facial_pain_project_v2", base_path]
    for c in candidates:
        if (c / "train_val_test_splits_v2.json").exists() and (c / "sequence_label_mapping_v2.json").exists():
            return c
    raise FileNotFoundError(
        "Could not find project dir containing train_val_test_splits_v2.json + sequence_label_mapping_v2.json. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _resolve_sequence_dir(base_path: Path, project_dir: Path) -> Path:
    candidates = [base_path / "sequence", project_dir / "sequence"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find sequence dir. Tried: {[str(c) for c in candidates]}")


def _resolve_checkpoint_dir(project_dir: Path, run_tag: str) -> Path:
    root = project_dir / "checkpoints_v2.9"
    c = (root / run_tag) if run_tag else root
    if not c.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {c}")
    return c


def _resolve_results_dir(project_dir: Path, run_tag: str) -> Path:
    root = project_dir / "results_v2.9"
    c = (root / run_tag) if run_tag else root
    c.mkdir(parents=True, exist_ok=True)
    return c


def _ckpt_path(checkpoint_dir: Path, fold_idx: int, ckpt_kind: str) -> Path:
    if ckpt_kind == "combined":
        return checkpoint_dir / f"best_model_v2.9_fold_{fold_idx}.pt"
    if ckpt_kind == "task1":
        return checkpoint_dir / f"best_model_v2.9_task1_fold_{fold_idx}.pt"
    if ckpt_kind == "task2":
        return checkpoint_dir / f"best_model_v2.9_task2_fold_{fold_idx}.pt"
    raise ValueError(f"Unknown ckpt_kind={ckpt_kind!r}")


def _compute_task1_metrics(pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(targets, pred)) if len(targets) else 0.0,
        "F1": float(f1_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "Precision": float(precision_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "Recall": float(recall_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "N": int(len(targets)),
    }


def _compute_task2_metrics(pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(targets, pred)) if len(targets) else 0.0,
        "F1_weighted": float(f1_score(targets, pred, average="weighted", zero_division=0)) if len(targets) else 0.0,
        "F1_macro": float(f1_score(targets, pred, average="macro", zero_division=0)) if len(targets) else 0.0,
        "Precision_weighted": float(precision_score(targets, pred, average="weighted", zero_division=0))
        if len(targets)
        else 0.0,
        "Recall_weighted": float(recall_score(targets, pred, average="weighted", zero_division=0)) if len(targets) else 0.0,
        "N": int(len(targets)),
    }


def _best_f1_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(np.int64)
        f1 = f1_score(targets, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, float(best_f1)


def _majority_vote_int(preds: np.ndarray, num_classes: int) -> np.ndarray:
    # preds: (F,N) int
    out = np.zeros((preds.shape[1],), dtype=np.int64)
    for i in range(preds.shape[1]):
        counts = np.bincount(preds[:, i], minlength=int(num_classes))
        out[i] = int(counts.argmax())
    return out


def _infer_available_folds(checkpoint_dir: Path, ckpt_kind: str) -> List[int]:
    if ckpt_kind == "combined":
        pat = "best_model_v2.9_fold_*.pt"
    else:
        pat = f"best_model_v2.9_{ckpt_kind}_fold_*.pt"
    fold_indices: List[int] = []
    for fp in checkpoint_dir.glob(pat):
        name = fp.name
        try:
            fold_str = name.split("_fold_")[-1].replace(".pt", "")
            fold_indices.append(int(fold_str))
        except Exception:
            continue
    return sorted(set(fold_indices))


def _run_dir_score(run_dir: Path, kinds: List[str]) -> Tuple[int, float]:
    total = 0
    for kind in kinds:
        total += len(_infer_available_folds(run_dir, kind))
    try:
        mtime = float(run_dir.stat().st_mtime)
    except Exception:
        mtime = 0.0
    return total, mtime


def _autodetect_run_tag(checkpoints_root: Path, kinds: List[str]) -> Tuple[Optional[str], Optional[Path]]:
    """
    If `checkpoints_root` has no fold checkpoints directly under it, try to find a
    subfolder that contains v2.9 checkpoints. This is useful in Colab when the user
    forgets to pass --run_tag and checkpoints live under checkpoints_v2.9/<run_tag>/.

    Returns: (run_tag, run_dir) or (None, None) if not found.
    """
    if not checkpoints_root.exists() or not checkpoints_root.is_dir():
        return None, None

    # If there are already checkpoints in the root, don't guess.
    if any(_infer_available_folds(checkpoints_root, k) for k in kinds):
        return None, None

    candidates: List[Tuple[str, Path, int, float]] = []
    try:
        for d in checkpoints_root.iterdir():
            if not d.is_dir():
                continue
            score, mtime = _run_dir_score(d, kinds)
            if score > 0:
                candidates.append((d.name, d, score, mtime))
    except Exception:
        return None, None

    if not candidates:
        return None, None

    # Prefer highest checkpoint coverage; tie-break by mtime then lexicographic (timestamp tags).
    candidates.sort(key=lambda x: (x[2], x[3], x[0]), reverse=True)
    run_tag, run_dir, _, _ = candidates[0]
    return run_tag, run_dir


def _print_checkpoint_inventory(checkpoint_dir: Path) -> None:
    print("=" * 80)
    print(f"Checkpoint inventory: {checkpoint_dir}")
    print("=" * 80)
    for kind in ("combined", "task1", "task2"):
        folds = _infer_available_folds(checkpoint_dir, kind)
        print(f"{kind:8s}: {len(folds)} folds -> {folds if folds else '[]'}")


@torch.no_grad()
def _predict_one_fold(
    *,
    mod: Any,
    ckpt_path: Path,
    sequences: List[dict],
    sequence_dir: Path,
    device: torch.device,
    batch_size_override: Optional[int],
    num_workers_override: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError(f"Checkpoint missing cfg dict: {ckpt_path}")

    cfg = mod.Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if batch_size_override is not None:
        cfg.batch_size = int(batch_size_override)
    if num_workers_override is not None:
        cfg.num_workers = int(num_workers_override)

    DatasetCls = getattr(mod, "FacialPainDataset_v2_9", None) or getattr(mod, "FacialPainDataset_v2_8")
    ds = DatasetCls(sequences, sequence_dir, cfg, augment=False, global_cache={})
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    ModelCls = getattr(mod, "TemporalPainModel_v2_9", None) or getattr(mod, "TemporalPainModel_v2_8")
    model = ModelCls(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    clip_ids: List[str] = []
    animals: List[int] = []
    moments: List[str] = []
    sequence_ids: List[str] = []
    sequence_paths: List[str] = []

    pain_logits_all: List[np.ndarray] = []
    pain_true_all: List[np.ndarray] = []
    task2_logits_all: List[np.ndarray] = []
    task2_true_all: List[np.ndarray] = []

    for b_idx, (x, y, meta) in enumerate(loader):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x.to(device, non_blocking=True)
        out, _ = model(x)

        pain_logits_all.append(out["pain_logits"].detach().cpu().numpy())
        task2_logits_all.append(out["task2_logits"].detach().cpu().numpy())

        pain_true_all.append(y["pain_binary"].detach().cpu().numpy().astype(np.int64))
        task2_true_all.append(y["task2"].detach().cpu().numpy().astype(np.int64))

        clip_ids.extend([str(s) for s in meta["clip_id"]])
        animals.extend([int(a) for a in meta["animal"]])
        moments.extend([str(m) for m in meta["moment"]])
        sequence_ids.extend([str(s) for s in meta["sequence_id"]])
        sequence_paths.extend([str(s) for s in meta["sequence_path"]])

    pain_logits = np.concatenate(pain_logits_all, axis=0) if pain_logits_all else np.zeros((0,), dtype=np.float32)
    task2_logits = np.concatenate(task2_logits_all, axis=0) if task2_logits_all else np.zeros((0, 3), dtype=np.float32)
    pain_targets = np.concatenate(pain_true_all, axis=0) if pain_true_all else np.zeros((0,), dtype=np.int64)
    task2_targets = np.concatenate(task2_true_all, axis=0) if task2_true_all else np.zeros((0,), dtype=np.int64)

    return {
        "ckpt_path": str(ckpt_path),
        "cfg": cfg,
        "clip_ids": np.array(clip_ids),
        "animals": np.array(animals),
        "moments": np.array(moments),
        "sequence_ids": np.array(sequence_ids),
        "sequence_paths": np.array(sequence_paths),
        "pain_logits": pain_logits,
        "task2_logits": task2_logits,
        "pain_targets": pain_targets,
        "task2_targets": task2_targets,
    }


def _evaluate_one_kind(
    *,
    mod: Any,
    ckpt_kind: str,
    test_animals: List[int],
    test_seqs: List[dict],
    checkpoint_dir: Path,
    sequence_dir: Path,
    device: torch.device,
    folds: Optional[List[int]],
    ensemble: str,
    task1_threshold_logit: float,
    optimize_task1_threshold: bool,
    batch_size: Optional[int],
    num_workers: Optional[int],
    max_batches: Optional[int],
    save_dir: Path,
) -> None:
    if folds is not None and len(folds) > 0:
        fold_indices = [int(x) for x in folds]
    else:
        fold_indices = _infer_available_folds(checkpoint_dir, ckpt_kind)

    if not fold_indices:
        raise RuntimeError(f"No folds found to evaluate in {checkpoint_dir} for ckpt_kind={ckpt_kind}.")

    per_fold: List[Dict[str, Any]] = []
    for fold_idx in fold_indices:
        ckpt_path = _ckpt_path(checkpoint_dir, fold_idx, ckpt_kind)
        if not ckpt_path.exists():
            print(f"Skipping fold {fold_idx}: missing checkpoint {ckpt_path.name}")
            continue

        pred = _predict_one_fold(
            mod=mod,
            ckpt_path=ckpt_path,
            sequences=test_seqs,
            sequence_dir=sequence_dir,
            device=device,
            batch_size_override=batch_size,
            num_workers_override=num_workers,
            max_batches=max_batches,
        )
        pred["fold"] = int(fold_idx)
        per_fold.append(pred)
        cfg = pred["cfg"]
        print(
            f"Fold {fold_idx}: loaded {ckpt_path.name} | task2_mode={cfg.task2_mode} input_mode={cfg.input_mode} "
            f"max_frames={cfg.max_frames} bs={cfg.batch_size} workers={cfg.num_workers}"
        )

    if not per_fold:
        raise RuntimeError(f"No folds evaluated (missing checkpoints?) in {checkpoint_dir} for ckpt_kind={ckpt_kind}.")

    # Ensure alignment across folds before ensembling.
    base_clip_ids = per_fold[0]["clip_ids"]
    for r in per_fold[1:]:
        if not np.array_equal(base_clip_ids, r["clip_ids"]):
            raise RuntimeError(
                "Per-fold prediction ordering mismatch by clip_id. "
                "Cannot ensemble safely; ensure deterministic dataloader ordering."
            )

    pain_targets = per_fold[0]["pain_targets"]
    task2_targets = per_fold[0]["task2_targets"]

    pain_logits_stack = np.stack([r["pain_logits"] for r in per_fold], axis=0)  # (F,N)
    task2_logits_stack = np.stack([r["task2_logits"] for r in per_fold], axis=0)  # (F,N,K)
    K = int(task2_logits_stack.shape[-1])

    thr_logit = float(task1_threshold_logit)
    if ensemble == "mean_logits":
        pain_logits_ens = pain_logits_stack.mean(axis=0)
        task2_logits_ens = task2_logits_stack.mean(axis=0)
        pain_pred_ens = (pain_logits_ens >= thr_logit).astype(np.int64)
        task2_pred_ens = np.argmax(task2_logits_ens, axis=1).astype(np.int64)
    else:
        pain_pred_stack = (pain_logits_stack >= thr_logit).astype(np.int64)
        task2_pred_stack = np.argmax(task2_logits_stack, axis=2).astype(np.int64)
        pain_pred_ens = _majority_vote_int(pain_pred_stack, num_classes=2)
        task2_pred_ens = _majority_vote_int(task2_pred_stack, num_classes=K)
        # Use mean logits for CSV (logits/probs columns); metrics above are from majority vote.
        pain_logits_ens = pain_logits_stack.mean(axis=0)
        task2_logits_ens = task2_logits_stack.mean(axis=0)

    t1 = _compute_task1_metrics(pain_pred_ens, pain_targets)
    t2 = _compute_task2_metrics(task2_pred_ens, task2_targets)

    print("\n" + "=" * 80)
    print(f"ENSEMBLE RESULTS ({ckpt_kind})")
    print("=" * 80)
    print(
        f"Task1: acc={t1['Accuracy']:.4f} f1={t1['F1']:.4f} p={t1['Precision']:.4f} r={t1['Recall']:.4f} N={t1['N']}"
    )
    print(
        f"Task2: acc={t2['Accuracy']:.4f} f1w={t2['F1_weighted']:.4f} f1m={t2['F1_macro']:.4f} "
        f"pw={t2['Precision_weighted']:.4f} rw={t2['Recall_weighted']:.4f} N={t2['N']}"
    )

    t1_opt = None
    if bool(optimize_task1_threshold):
        probs = 1.0 / (1.0 + np.exp(-pain_logits_stack.mean(axis=0)))
        best_thr, best_f1 = _best_f1_threshold(probs, pain_targets)
        t1_opt = {"best_thr_prob": float(best_thr), "best_f1": float(best_f1)}
        print(f"Task1_opt (diagnostic): f1={best_f1:.4f} @ prob_thr={best_thr:.2f}")

    # Test set label distribution (helps debug class collapse)
    uniq, cnt = np.unique(task2_targets, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    print(f"Task2 target distribution: {dist}")

    task2_mode = str(per_fold[0]["cfg"].task2_mode)
    if task2_mode == "3class":
        class_names = ["No Pain (M0/M1)", "Acute Pain (M2)", "Residual (M3/M4)"]
    else:
        class_names = ["No Pain (M0/M1)", "Acute (M2)", "Declining (M3)", "Recovery (M4)"]

    print("\nTask2 classification report:")
    labels = list(range(len(class_names)))
    print(
        classification_report(
            task2_targets,
            task2_pred_ens,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Task2 confusion matrix:")
    print(confusion_matrix(task2_targets, task2_pred_ens, labels=labels))

    # Save outputs
    prefix = f"test_eval_v2.9_{ckpt_kind}_animals_" + "-".join([str(a) for a in test_animals])

    per_fold_rows = []
    for r in per_fold:
        pain_pred = (r["pain_logits"] >= thr_logit).astype(np.int64)
        task2_pred = np.argmax(r["task2_logits"], axis=1).astype(np.int64)
        t1f = _compute_task1_metrics(pain_pred, r["pain_targets"])
        t2f = _compute_task2_metrics(task2_pred, r["task2_targets"])
        cfg = r["cfg"]
        per_fold_rows.append(
            {
                "fold": int(r["fold"]),
                "task2_mode": str(cfg.task2_mode),
                "input_mode": str(cfg.input_mode),
                "max_frames": int(cfg.max_frames),
                "resolution": str(tuple(cfg.resolution)),
                "task1_acc": t1f["Accuracy"],
                "task1_f1": t1f["F1"],
                "task2_acc": t2f["Accuracy"],
                "task2_f1_weighted": t2f["F1_weighted"],
                "task2_f1_macro": t2f["F1_macro"],
                "task1_threshold_logit": thr_logit,
                "N": int(t1f["N"]),
                "ckpt_path": str(r["ckpt_path"]),
            }
        )

    pd.DataFrame(per_fold_rows).to_csv(save_dir / f"{prefix}_per_fold.csv", index=False)

    # Save per-sample predictions (logits/probs required for calibration_v2.9.py and error analysis).
    task1_prob = 1.0 / (1.0 + np.exp(-pain_logits_ens))
    t2_logits = task2_logits_ens
    t2_shift = t2_logits - np.max(t2_logits, axis=1, keepdims=True)
    t2_exp = np.exp(t2_shift)
    task2_prob = t2_exp / np.clip(np.sum(t2_exp, axis=1, keepdims=True), 1e-12, None)

    pred_cols: Dict[str, Any] = {
        "clip_id": per_fold[0]["clip_ids"],
        "sequence_id": per_fold[0]["sequence_ids"],
        "sequence_path": per_fold[0]["sequence_paths"],
        "animal": per_fold[0]["animals"],
        "moment": per_fold[0]["moments"],
        "task1_true": pain_targets,
        "task1_pred": pain_pred_ens,
        "task1_logit": pain_logits_ens,
        "task1_prob": task1_prob,
        "task2_true": task2_targets,
        "task2_pred": task2_pred_ens,
        "task2_conf": task2_prob.max(axis=1) if len(task2_prob) else np.zeros((0,), dtype=np.float32),
    }
    K = int(t2_logits.shape[1]) if isinstance(t2_logits, np.ndarray) and t2_logits.ndim == 2 else 0
    for k in range(K):
        pred_cols[f"task2_logit_{k}"] = t2_logits[:, k]
        pred_cols[f"task2_prob_{k}"] = task2_prob[:, k]

    pd.DataFrame(pred_cols).to_csv(save_dir / f"{prefix}_predictions.csv", index=False)

    with open(save_dir / f"{prefix}_ensemble.json", "w") as f:
        json.dump(
            {
                "task1": t1,
                "task2": t2,
                "task2_mode": task2_mode,
                "ckpt_kind": str(ckpt_kind),
                "folds": sorted([int(r["fold"]) for r in per_fold]),
                "ensemble": str(ensemble),
                "task1_threshold_logit": float(thr_logit),
                "optimize_task1_threshold": bool(optimize_task1_threshold),
                "task1_opt": t1_opt,
                "test_animals": test_animals,
                "N": int(len(pain_targets)),
            },
            f,
            indent=2,
        )

    print(f"\nSaved outputs to: {save_dir}")


def main() -> None:
    args = _parse_args()
    args = _maybe_apply_env_overrides(args)
    extra_dirs: List[Path] = []
    if args.project_dir:
        extra_dirs.append(Path(args.project_dir))
    if args.base_path:
        extra_dirs.append(Path(args.base_path))
    if args.checkpoint_dir:
        ckpt = Path(args.checkpoint_dir)
        extra_dirs.append(ckpt)
        # Common layout: <project_dir>/checkpoints_v2.9
        try:
            extra_dirs.append(ckpt.parent)
        except Exception:
            pass
    extra_dirs.append(Path.cwd())
    mod = _load_v2_9_module(args.train_py, search_dirs=extra_dirs)

    base_path = Path(args.base_path) if args.base_path else _detect_base_path()
    project_dir = Path(args.project_dir) if args.project_dir else _resolve_project_dir(base_path)
    sequence_dir = Path(args.sequence_dir) if args.sequence_dir else _resolve_sequence_dir(base_path, project_dir)

    run_tag = str(args.run_tag) if args.run_tag else ""
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else _resolve_checkpoint_dir(project_dir=project_dir, run_tag=run_tag)
    )
    results_dir = _resolve_results_dir(project_dir=project_dir, run_tag=run_tag)
    save_dir = Path(args.save_dir) if args.save_dir else results_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((project_dir / "train_val_test_splits_v2.json").read_text())
    mapping = json.loads((project_dir / "sequence_label_mapping_v2.json").read_text())
    if isinstance(mapping, dict) and "sequences" in mapping:
        all_sequences = mapping["sequences"]
    elif isinstance(mapping, dict):
        all_sequences = [{"sequence_id": k, **v} for k, v in mapping.items()]
    else:
        all_sequences = mapping

    splits_test_animals = splits.get("test_animals", [14, 17])
    test_animals = args.test_animals if args.test_animals is not None else splits_test_animals
    test_animals = [int(a) for a in test_animals]
    if args.test_animals is not None and list(map(int, splits_test_animals)) != test_animals:
        print(f"WARNING: Using overridden --test_animals={test_animals} (splits.json default is {splits_test_animals}).")
    test_seqs = [s for s in all_sequences if int(s.get("animal", s.get("animal_id", -1))) in set(test_animals)]

    if not test_seqs:
        raise RuntimeError(f"No test sequences found for test_animals={test_animals} in mapping.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If user forgot --run_tag, try to auto-detect it from subfolders under checkpoints_v2.9.
    kinds_to_check = ["combined", "task1", "task2"] if str(args.ckpt_kind) == "all" else [str(args.ckpt_kind)]
    if not any(_infer_available_folds(checkpoint_dir, k) for k in kinds_to_check):
        inferred_tag, inferred_dir = _autodetect_run_tag(checkpoint_dir, kinds=kinds_to_check)
        if inferred_tag and inferred_dir:
            print(
                f"WARNING: No v2.9 checkpoints found directly under {checkpoint_dir}. "
                f"Auto-detected run_tag={inferred_tag!r} in subfolder {inferred_dir}."
            )
            run_tag = inferred_tag
            checkpoint_dir = inferred_dir
            results_dir = _resolve_results_dir(project_dir=project_dir, run_tag=run_tag)
            if args.save_dir is None:
                save_dir = results_dir
                save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVAL v2.9 (held-out test animals)")
    print(f"project_dir: {project_dir}")
    print(f"sequence_dir: {sequence_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"ckpt_kind: {args.ckpt_kind} | ensemble: {args.ensemble}")
    print(f"test_animals: {test_animals} | test_sequences: {len(test_seqs)}")
    print(f"device: {device}")
    print("=" * 80)

    if _running_in_notebook() and (not args.run_tag) and (str(args.ckpt_kind) == "combined"):
        print(
            "TIP (notebook): if you pasted this evaluator into a cell, set env vars to control options, e.g.\n"
            "  os.environ['UCAPS_V2_9_RUN_TAG']='v2.9_YYYYMMDD_HHMMSS'\n"
            "  os.environ['UCAPS_V2_9_CKPT_KIND']='task2'\n"
            "then rerun this cell."
        )

    if bool(args.list_checkpoints):
        _print_checkpoint_inventory(checkpoint_dir)
        return

    kinds = ["combined", "task1", "task2"] if str(args.ckpt_kind) == "all" else [str(args.ckpt_kind)]
    for kind in kinds:
        _evaluate_one_kind(
            mod=mod,
            ckpt_kind=kind,
            test_animals=test_animals,
            test_seqs=test_seqs,
            checkpoint_dir=checkpoint_dir,
            sequence_dir=sequence_dir,
            device=device,
            folds=args.folds,
            ensemble=str(args.ensemble),
            task1_threshold_logit=float(args.task1_threshold_logit),
            optimize_task1_threshold=bool(args.optimize_task1_threshold),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            save_dir=save_dir,
        )

    print("\nEvaluation complete.")


def run_eval(
    *,
    run_tag: Optional[str] = None,
    ckpt_kind: str = "combined",
    ensemble: str = "mean_logits",
    test_animals: Optional[List[int]] = None,
    project_dir: Optional[str] = None,
    sequence_dir: Optional[str] = None,
    base_path: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Notebook-friendly entrypoint (useful when you paste this file into a cell).

    Examples:
        run_eval(run_tag="v2.9_20260221_014705", ckpt_kind="task2")
        run_eval(run_tag="v2.9_20260221_014705", ckpt_kind="all")
    """
    argv = ["eval_v2.9"]
    if base_path:
        argv += ["--base_path", str(base_path)]
    if project_dir:
        argv += ["--project_dir", str(project_dir)]
    if sequence_dir:
        argv += ["--sequence_dir", str(sequence_dir)]
    if run_tag:
        argv += ["--run_tag", str(run_tag)]
    if ckpt_kind:
        argv += ["--ckpt_kind", str(ckpt_kind)]
    if ensemble:
        argv += ["--ensemble", str(ensemble)]
    if test_animals:
        argv += ["--test_animals"] + [str(int(a)) for a in test_animals]
    if device:
        argv += ["--device", str(device)]

    import sys

    old = sys.argv[:]
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
