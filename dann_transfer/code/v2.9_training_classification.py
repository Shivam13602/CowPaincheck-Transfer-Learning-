# ============================================================================
# PHASE 2: MODEL TRAINING (v2.9 - CLASSIFICATION, v2.5-stable defaults + researcher ablations)
#
# Baseline:
# - v2.5 "small-data" recipe: custom 2D CNN + LSTM + attention, dual heads
#   (Task1: pain binary, Task2: 3-class intensity moment).
#
# v2.9 philosophy (fix v2.8 regressions on small UCAPS data):
# - Default back to v2.5-like stability: CE/BCE losses, shared moment weighting, fixed linspace sampling,
#   no EMA, no time reversal, no label smoothing.
# - Keep the good engineering from v2.8: run isolation (run_tag), split leakage checks, val distribution debug.
# - Provide explicit knobs for ablations (enable EMA / alternate losses / temporal jitter only when requested).
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------
# Resume controls (Colab-safe)
# ----------------------------
AUTO_RESUME = True
START_FOLD = int(os.environ.get("START_FOLD", "0"))
SAVE_RESUME_CHECKPOINT = False


# ----------------------------
# Reproducibility utilities
# ----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_fast_mode() -> None:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def running_in_notebook() -> bool:
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def _looks_like_drive_path(p: Path) -> bool:
    s = str(p).replace("\\", "/").lower()
    return ("/content/drive" in s) or ("mydrive" in s)


# ----------------------------
# Checkpoint helpers (best-only + optional resume)
# ----------------------------
def best_checkpoint_path(checkpoint_dir: Path, fold_idx: int) -> Path:
    return checkpoint_dir / f"best_model_v2.9_fold_{fold_idx}.pt"


def task1_checkpoint_path(checkpoint_dir: Path, fold_idx: int) -> Path:
    return checkpoint_dir / f"best_model_v2.9_task1_fold_{fold_idx}.pt"


def task2_checkpoint_path(checkpoint_dir: Path, fold_idx: int) -> Path:
    return checkpoint_dir / f"best_model_v2.9_task2_fold_{fold_idx}.pt"


def resume_checkpoint_path(checkpoint_dir: Path, fold_idx: int) -> Path:
    return checkpoint_dir / f"resume_checkpoint_v2.9_fold_{fold_idx}.pt"


def resume_checkpoint_path_stage(checkpoint_dir: Path, fold_idx: int, *, stage: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", str(stage).strip().lower()).strip("_")
    safe = safe or "stage"
    return checkpoint_dir / f"resume_checkpoint_v2.9_{safe}_fold_{fold_idx}.pt"


def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}


def save_best_checkpoint(
    *,
    checkpoint_dir: Path,
    fold_idx: int,
    epoch: int,
    model_state_dict: Dict[str, torch.Tensor],
    cfg,
    out_path: Path,
    label: str,
    best_metric_name: str,
    best_metric: float,
    best_components: Dict[str, float],
) -> None:
    ckpt = {
        "version": "v2.9",
        "arch": "cnn2d_lstm_attention",
        "fold": int(fold_idx),
        "epoch": int(epoch),
        "model_state_dict": _cpu_state_dict(model_state_dict),
        "best_metric": float(best_metric),
        "best_metric_name": str(best_metric_name),
        "checkpoint_label": str(label),
        "best_components": {
            k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in best_components.items()
        },
        "cfg": cfg.__dict__,
    }
    torch.save(ckpt, out_path)
    print(f"{label} saved: {out_path.name} ({best_metric_name}={best_metric:.4f})")


def save_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    fold_idx: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_combined_metric: float,
    cfg,
    ema_shadow: Optional[Dict[str, torch.Tensor]],
    out_path: Optional[Path] = None,
) -> None:
    ckpt = {
        "version": "v2.9",
        "arch": "cnn2d_lstm_attention",
        "fold": int(fold_idx),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_combined_metric": float(best_combined_metric),
        "cfg": cfg.__dict__,
        "ema_shadow": _cpu_state_dict(ema_shadow) if isinstance(ema_shadow, dict) else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, out_path if out_path is not None else resume_checkpoint_path(checkpoint_dir, fold_idx))


def try_resume(
    *,
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional["EMA"],
) -> Tuple[int, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"WARNING: Could not restore optimizer state ({type(e).__name__}: {e}).")
    try:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception as e:
        print(f"WARNING: Could not restore scheduler state ({type(e).__name__}: {e}).")
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if ema is not None:
        shadow = ckpt.get("ema_shadow")
        if isinstance(shadow, dict) and shadow:
            ema.shadow = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in shadow.items()}
        else:
            # Backward compatibility: if the resume checkpoint has no EMA shadow, start EMA from the resumed weights.
            ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    rng = ckpt.get("rng_state")
    if rng is not None:
        try:
            random.setstate(rng.get("python"))
            np.random.set_state(rng.get("numpy"))
            torch.set_rng_state(rng.get("torch"))
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng.get("torch_cuda"))
        except Exception:
            pass

    start_epoch = int(ckpt["epoch"]) + 1
    best_combined_metric = float(ckpt.get("best_combined_metric", float("-inf")))
    return start_epoch, best_combined_metric


# ----------------------------
# Split schema helpers
# ----------------------------
def get_folds_from_splits(splits_dict: dict) -> List[dict]:
    cv = splits_dict.get("cv_folds")
    if isinstance(cv, list) and len(cv) > 0:
        return cv

    folds = splits_dict.get("folds")
    if isinstance(folds, list) and len(folds) > 0:
        return folds

    if isinstance(cv, dict):
        items = []
        for k, v in cv.items():
            if not isinstance(v, dict):
                continue
            if "train_animals" in v and "val_animals" in v:
                m = re.match(r"^fold[_\\-]?(\\d+)$", str(k).strip().lower())
                if m:
                    items.append((int(m.group(1)), v))
                elif str(k).isdigit():
                    items.append((int(k), v))
        if items:
            items.sort(key=lambda x: x[0])
            return [v for _, v in items]

    raise KeyError("Could not find folds in splits JSON (expected cv_folds or folds).")


# ----------------------------
# Config (v2.9 defaults; stable v2.5-first recipe)
# ----------------------------
@dataclass
class Config:
    # Data
    max_frames: int = 32
    resolution: Tuple[int, int] = (112, 112)
    num_workers: int = 8
    input_mode: str = "rgb"  # rgb | grayst

    # CV
    num_folds: int = 9

    # Task 2 definition (keep v2.5 comparable default)
    task2_mode: str = "3class"

    # Training scheme
    training_scheme: str = "joint"  # joint | two_stage
    stage1_epochs: Optional[int] = None  # if None and two_stage, uses num_epochs//2
    stage2_epochs: Optional[int] = None  # if None and two_stage, uses num_epochs-stage1_epochs
    stage2_lr: Optional[float] = None  # if None, uses learning_rate

    # Training (v2.5-style defaults)
    num_epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    warmup_epochs: int = 2
    lr_patience: int = 5
    min_lr: float = 1e-7

    # Model
    lstm_hidden_size: int = 128
    dropout: float = 0.3
    use_bidirectional_lstm: bool = False
    lstm_num_layers: int = 1
    freeze_cnn: bool = False  # if True, keep CNN in eval and train only temporal/head

    # EMA (off by default for small UCAPS; enable explicitly via --ema)
    use_ema: bool = False
    ema_decay: float = 0.999

    # Loss weighting (dual task)
    task1_weight: float = 1.0
    task2_weight: float = 1.0

    # Task2 loss (stable default: plain CE with inverse-frequency weights)
    task2_loss_type: str = "ce"  # ce | sce | balanced_softmax | cb_focal | ldam_drw | gce
    label_smoothing: float = 0.0

    # CB-Focal
    task2_cb_beta: float = 0.9999
    task2_focal_gamma: float = 2.0

    # LDAM-DRW
    task2_ldam_max_m: float = 0.5
    task2_ldam_s: float = 30.0
    task2_drw_start_epoch: int = 10

    # SCE
    task2_sce_alpha: float = 1.0
    task2_sce_beta: float = 0.1

    # GCE
    task2_gce_q: float = 0.7

    # Imbalance + sampling
    use_stratified_sampler: bool = True
    use_moment_weighting: bool = True
    # Shared moment weighting (v2.5 behavior): applied to both tasks
    moment_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"M0": 1.0, "M1": 1.0, "M2": 4.0, "M3": 2.0, "M4": 1.2}
    )

    # Temporal sampling/augment
    temporal_sampling: str = "linspace"  # linspace | uniform_offset | random_clip
    time_reverse_p: float = 0.0

    # Augmentations (v2.5-style default; can be disabled via CLI)
    use_augmentations: bool = True
    use_consistent_aug: bool = True
    aug_use_hflip: bool = True
    aug_use_affine: bool = True
    aug_use_color_jitter: bool = True
    aug_use_blur: bool = True

    # Best selection (v2.5-style combined metric)
    best_metric_task1_weight: float = 0.5
    best_metric_task2_weight: float = 0.5
    # Which Task2 metric feeds the combined selection metric (v2.5 used weighted-F1).
    combined_task2_metric: str = "weighted"  # macro | weighted
    # Which Task2 metric defines the "Best Task2 model" checkpoint.
    best_task2_metric: str = "weighted"  # macro | weighted
    best_metric_use_task1_opt_f1: bool = False
    # Print val-time prediction distributions (collapse detector)
    print_val_distributions: bool = True
    early_stop_patience: int = 20
    early_stop_min_epochs: int = 5
    early_stop_min_delta: float = 1e-3

    # Paths
    project_dirname: str = "facial_pain_project_v2"
    checkpoint_subdir: str = "checkpoints_v2.9"
    results_subdir: str = "results_v2.9"
    run_tag: str = ""
    auto_run_tag: bool = True


# ----------------------------
# Label mapping (v2.5-compatible defaults)
# ----------------------------
def moment_to_task1_binary(moment: str) -> int:
    return 1 if moment in ["M2", "M3", "M4"] else 0


def task2_num_classes(task2_mode: str) -> int:
    if task2_mode == "3class":
        return 3
    if task2_mode == "4class":
        return 4
    raise ValueError(f"Unknown task2_mode={task2_mode!r} (expected '3class' or '4class').")


def moment_to_task2(moment: str, *, task2_mode: str) -> int:
    if task2_mode == "3class":
        if moment in ["M0", "M1"]:
            return 0
        if moment == "M2":
            return 1
        if moment in ["M3", "M4"]:
            return 2
        return 0

    if task2_mode == "4class":
        if moment in ["M0", "M1"]:
            return 0
        if moment == "M2":
            return 1
        if moment == "M3":
            return 2
        if moment == "M4":
            return 3
        return 0

    raise ValueError(f"Unknown task2_mode={task2_mode!r} (expected '3class' or '4class').")


def normalize_clip_id(s: str) -> str:
    s = str(s).replace("\\", "/")
    s = re.sub(r"[\\s/]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.strip("_")


def _seq_animal_id(seq_info: dict) -> int:
    a = seq_info.get("animal", seq_info.get("animal_id", -1))
    try:
        return int(a)
    except Exception:
        return -1


def _seq_clip_key(seq_info: dict) -> str:
    seq_path = str(seq_info.get("sequence_path", "") or "")
    seq_id = str(seq_info.get("sequence_id", "") or "")
    return normalize_clip_id(seq_path or seq_id)


def _dir_has_files(d: Path) -> bool:
    try:
        if not d.exists():
            return False
        for p in d.iterdir():
            if p.is_file():
                return True
        return False
    except Exception:
        return False


def _ensure_unique_run_tag(project_dir: Path, cfg: Config) -> str:
    base = f"v2.9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tag = base
    i = 1
    while (project_dir / cfg.checkpoint_subdir / tag).exists() or (project_dir / cfg.results_subdir / tag).exists():
        tag = f"{base}_{i}"
        i += 1
    return tag


def check_splits_for_leakage(*, splits: dict, sequences: List[dict]) -> None:
    """
    Hard-fail on obvious leakage:
      - test animals appearing in any train/val animals
      - train/val animal overlap within a fold
      - clip overlap (sequence_path/sequence_id) across train and val within a fold
      - any test clip appearing in any fold train/val split
    """
    errors: List[str] = []
    warns: List[str] = []

    test_animals = {int(a) for a in splits.get("test_animals", [])}
    train_val_animals = {int(a) for a in splits.get("train_val_animals", [])}

    if not test_animals:
        warns.append("splits.json has empty/missing test_animals. Leakage checks are weaker without a held-out set.")

    if test_animals & train_val_animals:
        errors.append(f"test_animals overlaps train_val_animals: {sorted(test_animals & train_val_animals)}")

    folds = get_folds_from_splits(splits)

    # Build test clip set
    test_seqs = [s for s in sequences if _seq_animal_id(s) in test_animals]
    test_clips = {_seq_clip_key(s) for s in test_seqs}

    # Duplicate clip ids in mapping (not necessarily leakage, but often indicates duplicated samples)
    clip_counts = Counter([_seq_clip_key(s) for s in sequences if _seq_clip_key(s)])
    dup_clips = [(k, v) for k, v in clip_counts.items() if v > 1]
    if dup_clips:
        dup_clips.sort(key=lambda x: (-x[1], x[0]))
        warns.append(f"Found {len(dup_clips)} duplicate clip_ids in mapping (top={dup_clips[:5]}).")

    # sequence_id is known non-unique in UCAPS; warn if duplicates exist without sequence_path usage.
    seq_id_counts = Counter([str(s.get('sequence_id', '')) for s in sequences if str(s.get('sequence_id', '')).strip()])
    dup_seq_ids = sum(1 for _, v in seq_id_counts.items() if v > 1)
    if dup_seq_ids:
        warns.append(
            f"sequence_id is not unique in mapping ({dup_seq_ids} duplicated ids). "
            f"Always use sequence_path/clip_id for caching and leakage checks."
        )

    # Clip -> (animal, moment) consistency
    clip_to_label: Dict[str, Tuple[int, str]] = {}
    for s in sequences:
        ck = _seq_clip_key(s)
        if not ck:
            continue
        lab = (_seq_animal_id(s), str(s.get("moment", "unknown")))
        prev = clip_to_label.get(ck)
        if prev is None:
            clip_to_label[ck] = lab
        elif prev != lab:
            errors.append(f"clip_id label conflict: {ck} has {prev} and {lab} (mapping corruption -> leakage risk)")
            break

    # Animals present in mapping
    mapping_animals = {_seq_animal_id(s) for s in sequences if _seq_animal_id(s) >= 0}
    declared_animals = set(train_val_animals) | set(test_animals)
    extra = sorted(mapping_animals - declared_animals)
    missing = sorted(declared_animals - mapping_animals)
    if extra:
        warns.append(f"Animals present in mapping but not in splits: {extra}")
    if missing:
        warns.append(f"Animals declared in splits but missing in mapping: {missing}")

    # Fold checks
    all_fold_train_val_clips: set[str] = set()
    for i, fold in enumerate(folds):
        tr = {int(a) for a in fold.get("train_animals", [])}
        va = {int(a) for a in fold.get("val_animals", [])}

        inter = tr & va
        if inter:
            errors.append(f"Fold {i}: train_animals ∩ val_animals is not empty: {sorted(inter)}")

        inter_test = (tr | va) & test_animals
        if inter_test:
            errors.append(f"Fold {i}: train/val contains test animals: {sorted(inter_test)}")

        tr_seqs = [s for s in sequences if _seq_animal_id(s) in tr]
        va_seqs = [s for s in sequences if _seq_animal_id(s) in va]
        tr_clips = {_seq_clip_key(s) for s in tr_seqs}
        va_clips = {_seq_clip_key(s) for s in va_seqs}

        clip_overlap = (tr_clips & va_clips) - {""}
        if clip_overlap:
            sample = sorted(list(clip_overlap))[:5]
            errors.append(f"Fold {i}: train/val clip overlap detected (sample={sample}, n={len(clip_overlap)})")

        # Test clip overlap
        leak_to_train = (test_clips & tr_clips) - {""}
        leak_to_val = (test_clips & va_clips) - {""}
        if leak_to_train:
            sample = sorted(list(leak_to_train))[:5]
            errors.append(f"Fold {i}: test clips appear in train split (sample={sample}, n={len(leak_to_train)})")
        if leak_to_val:
            sample = sorted(list(leak_to_val))[:5]
            errors.append(f"Fold {i}: test clips appear in val split (sample={sample}, n={len(leak_to_val)})")

        all_fold_train_val_clips |= tr_clips
        all_fold_train_val_clips |= va_clips

    if errors:
        msg = "\n".join(["Split leakage check FAILED:"] + [f"- {e}" for e in errors])
        if warns:
            msg += "\n\nWarnings:\n" + "\n".join([f"- {w}" for w in warns])
        raise RuntimeError(msg)

    print("Split leakage check: OK")
    if warns:
        print("Split leakage warnings:")
        for w in warns:
            print(f"- {w}")


class VideoAugment:
    """
    Sequence-consistent augmentation: the same sampled params are applied to all frames.
    This avoids introducing artificial temporal flicker that can confuse temporal models.
    """

    def __init__(
        self,
        *,
        use_hflip: bool,
        use_affine: bool,
        use_color_jitter: bool,
        use_blur: bool,
        input_mode: str,
    ):
        self.use_hflip = bool(use_hflip)
        self.use_affine = bool(use_affine)
        self.use_color_jitter = bool(use_color_jitter)
        self.use_blur = bool(use_blur)
        self.input_mode = str(input_mode)

        # v2.5-like ranges
        self.flip_p = 0.5
        self.degrees = 10
        self.translate = (0.1, 0.1)
        self.scale = (0.9, 1.1)
        self.color_jitter = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.blur_kernel = (3, 7)
        self.blur_sigma = (0.1, 2.0)

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        if not frames:
            return frames

        do_hflip = self.use_hflip and (random.random() < self.flip_p)

        # Shared affine params
        if self.use_affine:
            angle = random.uniform(-self.degrees, self.degrees)
            trans_x = random.uniform(-self.translate[0], self.translate[0])
            trans_y = random.uniform(-self.translate[1], self.translate[1])
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            angle, trans_x, trans_y, scale = 0.0, 0.0, 0.0, 1.0

        # Shared color params
        if self.use_color_jitter:
            b = 1.0 + random.uniform(-self.color_jitter["brightness"], self.color_jitter["brightness"])
            c = 1.0 + random.uniform(-self.color_jitter["contrast"], self.color_jitter["contrast"])
            s = 1.0 + random.uniform(-self.color_jitter["saturation"], self.color_jitter["saturation"])
            h = random.uniform(-self.color_jitter["hue"], self.color_jitter["hue"])
        else:
            b, c, s, h = 1.0, 1.0, 1.0, 0.0

        # GrayST uses 3 grayscale channels; hue/saturation are not meaningful.
        if self.input_mode == "grayst":
            s, h = 1.0, 0.0

        blur = T.GaussianBlur(kernel_size=self.blur_kernel, sigma=self.blur_sigma) if self.use_blur else None

        out: List[Image.Image] = []
        for img in frames:
            if do_hflip:
                img = TF.hflip(img)
            if self.use_affine:
                img = TF.affine(
                    img,
                    angle=angle,
                    translate=[int(trans_x * img.size[0]), int(trans_y * img.size[1])],
                    scale=scale,
                    shear=[0.0, 0.0],
                )
            if self.use_color_jitter:
                img = TF.adjust_brightness(img, b)
                img = TF.adjust_contrast(img, c)
                if self.input_mode != "grayst":
                    img = TF.adjust_saturation(img, s)
                    img = TF.adjust_hue(img, h)
            if blur is not None:
                img = blur(img)
            out.append(img)
        return out


# ----------------------------
# Dataset (frames on disk)
# ----------------------------
class FacialPainDataset_v2_8(Dataset):
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

        self.video_aug: Optional[VideoAugment] = None
        self.frame_aug: Optional[T.Compose] = None
        if self.augment and bool(self.cfg.use_augmentations):
            if bool(self.cfg.use_consistent_aug):
                self.video_aug = VideoAugment(
                    use_hflip=self.cfg.aug_use_hflip,
                    use_affine=self.cfg.aug_use_affine,
                    use_color_jitter=self.cfg.aug_use_color_jitter,
                    use_blur=self.cfg.aug_use_blur,
                    input_mode=self.cfg.input_mode,
                )
            else:
                # Per-frame augmentation (faster to implement, but can introduce temporal flicker)
                aug: List[object] = []
                if bool(self.cfg.aug_use_hflip):
                    aug.append(T.RandomHorizontalFlip(p=0.5))
                if bool(self.cfg.aug_use_affine):
                    aug.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
                # For GrayST, hue/saturation jitter is not meaningful; keep it off.
                if bool(self.cfg.aug_use_color_jitter) and self.cfg.input_mode != "grayst":
                    aug.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
                if bool(self.cfg.aug_use_blur):
                    aug.append(T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)))
                if aug:
                    self.frame_aug = T.Compose(aug)

    def __len__(self) -> int:
        return len(self.sequences)

    def _cache_key(self, seq_info: dict, idx: int) -> str:
        # IMPORTANT: sequence_id is not unique in this dataset. Prefer sequence_path (clip-unique).
        if "sequence_path" in seq_info and seq_info.get("sequence_path"):
            return str(seq_info["sequence_path"])
        # Fallback: include idx to avoid collisions.
        seq_id = seq_info.get("sequence_id", "seq")
        animal = seq_info.get("animal", seq_info.get("animal_id", "unknown"))
        moment = seq_info.get("moment", "unknown")
        return f"{seq_id}_{animal}_{moment}_{idx}"

    def _find_frames_path(self, seq_info: dict) -> Optional[Path]:
        if "sequence_path" in seq_info:
            seq_path = self.sequence_dir / str(seq_info["sequence_path"]).replace("\\", "/")
        elif "sequence_id" in seq_info:
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
        if (not self.augment) or (self.cfg.temporal_sampling == "linspace"):
            return np.linspace(0, n - 1, need, dtype=np.int64)

        if self.cfg.temporal_sampling == "random_clip":
            if n >= need:
                start = random.randint(0, n - need)
                return (start + np.arange(need, dtype=np.int64)).astype(np.int64)
            idxs = np.arange(n, dtype=np.int64)
            pad = np.full((need - n,), n - 1, dtype=np.int64)
            return np.concatenate([idxs, pad], axis=0)

        # Default: uniform_offset sampling (adds temporal diversity but keeps global coverage)
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

        # Labels
        y1 = moment_to_task1_binary(moment)
        y2 = moment_to_task2(moment, task2_mode=self.cfg.task2_mode)

        decode_errors = 0
        n_available_frames = int(len(frame_files)) if isinstance(frame_files, list) else 0

        # Load frames (train: jittered sampling; val/test: deterministic linspace)
        need_raw = int(self.cfg.max_frames) + (2 if self.cfg.input_mode == "grayst" else 0)
        if frame_files is None or len(frame_files) == 0:
            dummy = Image.new("RGB", self.cfg.resolution, color="black")
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
                    frames.append(last_ok if last_ok is not None else Image.new("RGB", self.cfg.resolution, color="black"))

        # GrayST mode: stack 3 consecutive grayscale frames as channels
        if self.cfg.input_mode == "grayst":
            g = [im.convert("L") for im in frames]
            stacked: List[Image.Image] = []
            for i in range(int(self.cfg.max_frames)):
                stacked.append(Image.merge("RGB", (g[i], g[i + 1], g[i + 2])))
            frames = stacked

        if self.video_aug is not None:
            frames = self.video_aug(frames)
        elif self.frame_aug is not None:
            frames = [self.frame_aug(im) for im in frames]

        # To tensor + normalize
        tensors: List[torch.Tensor] = []
        for im in frames:
            im = TF.resize(im, self.cfg.resolution)
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
            # Debug / data health
            "is_dummy": bool(is_dummy),
            "n_available_frames": int(n_available_frames),
            "n_decode_errors": int(decode_errors),
        }
        return x, labels, meta


# Backward-compat: some notebooks/scripts still reference the v2.7 dataset name.
FacialPainDataset_v2_7 = FacialPainDataset_v2_8
FacialPainDataset_v2_9 = FacialPainDataset_v2_8


def create_stratified_sampler(sequences: List[dict]) -> WeightedRandomSampler:
    counts: Counter = Counter()
    for s in sequences:
        counts[str(s.get("moment", "unknown"))] += 1
    total = float(sum(counts.values()))
    moment_weights = {m: (total / max(float(c), 1.0)) for m, c in counts.items()}
    sample_weights = [float(moment_weights.get(str(s.get("moment", "unknown")), 1.0)) for s in sequences]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# ----------------------------
# Model (v2.5-style CNN + LSTM + attention)
# ----------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(int(hidden_size), 1)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.attn(lstm_out), dim=1)  # (B,T,1)
        ctx = torch.sum(w * lstm_out, dim=1)  # (B,H)
        return ctx, w


class TemporalPainModel_v2_8(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.task2_num_classes = task2_num_classes(cfg.task2_mode)

        # 2D CNN feature extractor (same as v2.5)
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

    def extract_features(self, x: torch.Tensor, *, apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,C,H,W)
        B, T_, C, H, W = x.shape
        x = x.view(B * T_, C, H, W)
        f = self.cnn(x).view(B * T_, -1)  # (B*T,256)
        f = f.view(B, T_, self.cnn_output_size)  # (B,T,256)

        lstm_out, _ = self.lstm(f)  # (B,T,H)
        ctx, attn_w = self.attn(lstm_out)
        if apply_dropout:
            ctx = self.dropout(ctx)
        return ctx, attn_w

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ctx, attn_w = self.extract_features(x, apply_dropout=True)

        out = {
            "pain_logits": self.head_task1(ctx).squeeze(-1),
            "task2_logits": self.head_task2(ctx),
        }
        return out, attn_w


# Backward-compat: some notebooks/scripts still reference the v2.7 model name.
TemporalPainModel_v2_7 = TemporalPainModel_v2_8
TemporalPainModel_v2_9 = TemporalPainModel_v2_8


# ----------------------------
# EMA helper (safe for BatchNorm buffers)
# ----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            v = v.detach()
            if k not in self.shadow:
                self.shadow[k] = v.clone()
                continue
            if not v.is_floating_point():
                self.shadow[k] = v.clone()
                continue
            self.shadow[k].mul_(self.decay).add_(v, alpha=(1.0 - self.decay))

    def apply_to(self, model: nn.Module) -> dict:
        current = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return current

    def restore(self, model: nn.Module, state: dict) -> None:
        model.load_state_dict(state, strict=False)


# ----------------------------
# Task2 imbalance / robustness losses
# ----------------------------
def effective_num(n: int, beta: float) -> float:
    if n <= 0:
        return 0.0
    return (1.0 - beta**n) / (1.0 - beta)


def class_balanced_weights(class_counts: List[int], beta: float) -> torch.Tensor:
    eff = np.array([effective_num(n, beta) for n in class_counts], dtype=np.float64)
    eff = np.maximum(eff, 1e-8)
    w = 1.0 / eff
    w = w / w.sum() * len(class_counts)
    return torch.tensor(w, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: List[int],
        *,
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.cb_w = class_balanced_weights(class_counts, beta=float(beta))  # (K,)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        K = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / (K - 1) if K > 1 else 0.0)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        p_t = (probs * true_dist).sum(dim=1).clamp(min=1e-8)
        focal = (1.0 - p_t) ** self.gamma
        ce = -(true_dist * log_probs).sum(dim=1)

        w = self.cb_w.to(logits.device)[target]
        loss = w * focal * ce
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax (logit-adjusted CE) for long-tailed classification.
    """

    def __init__(self, class_counts: List[int], label_smoothing: float = 0.0):
        super().__init__()
        counts = np.array(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        self.register_buffer("log_counts", torch.tensor(np.log(counts), dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        adj = logits + self.log_counts.to(logits.device)
        return F.cross_entropy(adj, target, label_smoothing=self.label_smoothing, reduction=reduction)


class LDAMLoss(nn.Module):
    """
    LDAM (Cao et al., NeurIPS 2019) with optional deferred re-weighting (DRW).
    """

    def __init__(
        self,
        class_counts: List[int],
        *,
        max_m: float = 0.5,
        s: float = 30.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        counts = np.array(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        m_list = 1.0 / np.sqrt(np.sqrt(counts))
        m_list = m_list * (float(max_m) / float(np.max(m_list)))
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = float(s)
        self.class_weight = class_weight

    def set_class_weight(self, w: Optional[torch.Tensor]) -> None:
        self.class_weight = w

    def forward(self, logits: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        K = logits.size(1)
        if self.m_list.numel() != K:
            raise ValueError(f"LDAMLoss mismatch: m_list has {self.m_list.numel()} classes but logits has {K}.")
        margins = self.m_list.to(logits.device)[target]
        one_hot = F.one_hot(target, num_classes=K).to(logits.dtype)
        logits_m = logits - (one_hot * margins.unsqueeze(1))
        scaled = self.s * logits_m
        return F.cross_entropy(scaled, target, weight=self.class_weight, reduction=reduction)


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, *, alpha: float = 1.0, beta: float = 0.1, eps: float = 1e-4, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        K = logits.size(1)
        ce = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing, reduction="none")
        pred = F.softmax(logits, dim=1).clamp(min=self.eps, max=1.0)
        one_hot = F.one_hot(target, num_classes=K).to(logits.dtype)
        one_hot = one_hot.clamp(min=self.eps, max=1.0)
        rce = -(pred * torch.log(one_hot)).sum(dim=1)
        loss = (self.alpha * ce) + (self.beta * rce)
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, *, q: float = 0.7, eps: float = 1e-8):
        super().__init__()
        if not (0.0 < q <= 1.0):
            raise ValueError("GCE q must be in (0, 1].")
        self.q = float(q)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        idx = torch.arange(target.numel(), device=target.device)
        p_t = probs[idx, target].clamp(min=self.eps, max=1.0)
        loss = (1.0 - (p_t**self.q)) / self.q
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()


# ----------------------------
# Metrics + train/val helpers
# ----------------------------
def best_f1_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (probs >= thr).astype(np.int64)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


@torch.no_grad()
def compute_task1_metrics(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    return {
        "acc": float(accuracy_score(targets, pred)) if len(targets) else 0.0,
        "f1": float(f1_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "precision": float(precision_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "recall": float(recall_score(targets, pred, zero_division=0)) if len(targets) else 0.0,
        "pred_pos_frac": float(pred.mean()) if len(pred) else 0.0,
    }


@torch.no_grad()
def compute_task2_metrics(logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    pred = np.argmax(logits, axis=1).astype(np.int64) if len(logits) else np.zeros((0,), dtype=np.int64)
    out = {
        "acc": float(accuracy_score(targets, pred)) if len(targets) else 0.0,
        "f1_weighted": float(f1_score(targets, pred, average="weighted", zero_division=0)) if len(targets) else 0.0,
        "f1_macro": float(f1_score(targets, pred, average="macro", zero_division=0)) if len(targets) else 0.0,
        "precision_weighted": float(precision_score(targets, pred, average="weighted", zero_division=0))
        if len(targets)
        else 0.0,
        "recall_weighted": float(recall_score(targets, pred, average="weighted", zero_division=0)) if len(targets) else 0.0,
    }
    # Per-class recall (helpful for diagnosing residual class)
    if isinstance(logits, np.ndarray) and logits.ndim == 2 and logits.shape[0] > 0:
        K = int(logits.shape[1])
    else:
        K = int(np.max(targets)) + 1 if len(targets) else 0

    # Pred/target distributions (collapse detector)
    if len(targets) and K > 0:
        pred_counts = np.bincount(pred, minlength=K).astype(np.int64)
        true_counts = np.bincount(targets.astype(np.int64), minlength=K).astype(np.int64)
        for k in range(int(K)):
            out[f"pred_count_c{k}"] = float(pred_counts[k])
            out[f"pred_frac_c{k}"] = float(pred_counts[k] / max(1, len(targets)))
            out[f"true_count_c{k}"] = float(true_counts[k])
            out[f"true_frac_c{k}"] = float(true_counts[k] / max(1, len(targets)))
    else:
        for k in range(int(K)):
            out[f"pred_count_c{k}"] = 0.0
            out[f"pred_frac_c{k}"] = 0.0
            out[f"true_count_c{k}"] = 0.0
            out[f"true_frac_c{k}"] = 0.0

    for k in range(int(K)):
        mask = (targets == k)
        out[f"recall_c{k}"] = float(((pred[mask]) == k).mean()) if mask.sum() else 0.0
    return out


def _moment_weights(moments: List[str], *, weights: Dict[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(weights.get(str(m), 1.0)) for m in moments], device=device, dtype=torch.float32)


def _make_task2_loss(
    cfg: Config,
    *,
    t2_counts: List[int],
    device: torch.device,
) -> Tuple[Optional[nn.Module], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Returns (loss_module_or_none, ce_weights_or_none, drw_weights_or_none).
    """
    if cfg.task2_loss_type == "ce":
        K = len(t2_counts)
        total = float(sum(t2_counts))
        ce_w = torch.tensor(
            [(total / (float(K) * max(float(c), 1.0))) for c in t2_counts],
            dtype=torch.float32,
            device=device,
        )
        return None, ce_w, None

    if cfg.task2_loss_type == "balanced_softmax":
        return BalancedSoftmaxLoss(t2_counts, label_smoothing=cfg.label_smoothing), None, None

    if cfg.task2_loss_type == "cb_focal":
        return (
            ClassBalancedFocalLoss(
                t2_counts,
                beta=cfg.task2_cb_beta,
                gamma=cfg.task2_focal_gamma,
                label_smoothing=cfg.label_smoothing,
            ),
            None,
            None,
        )

    if cfg.task2_loss_type == "ldam_drw":
        drw_w = class_balanced_weights(t2_counts, beta=cfg.task2_cb_beta).to(device)
        return (
            LDAMLoss(t2_counts, max_m=cfg.task2_ldam_max_m, s=cfg.task2_ldam_s, class_weight=None),
            None,
            drw_w,
        )

    if cfg.task2_loss_type == "sce":
        return (
            SymmetricCrossEntropyLoss(
                alpha=cfg.task2_sce_alpha,
                beta=cfg.task2_sce_beta,
                label_smoothing=cfg.label_smoothing,
            ),
            None,
            None,
        )

    if cfg.task2_loss_type == "gce":
        return (GeneralizedCrossEntropyLoss(q=cfg.task2_gce_q), None, None)

    raise ValueError(f"Unknown task2_loss_type={cfg.task2_loss_type!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[EMA],
    loss_task1: nn.Module,
    loss_task2: Optional[nn.Module],
    task2_ce_weights: Optional[torch.Tensor],
    max_batches: Optional[int],
) -> float:
    model.train()
    if bool(getattr(cfg, "freeze_cnn", False)):
        model.cnn.eval()
    total_loss = 0.0
    n_batches = 0

    for b_idx, (x, y, meta) in enumerate(tqdm(loader, desc="Train", leave=False, ascii=True)):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x.to(device, non_blocking=True)
        y1 = y["pain_binary"].to(device, non_blocking=True)
        y2 = y["task2"].to(device, non_blocking=True)
        moments = list(meta["moment"])

        if cfg.use_moment_weighting:
            mw = _moment_weights(moments, weights=cfg.moment_loss_weights, device=device)
        else:
            mw = torch.ones(x.size(0), device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            out, _ = model(x)

            l1_per = loss_task1(out["pain_logits"], y1)
            l1 = (l1_per * mw).mean()

            logits2 = out["task2_logits"]
            if cfg.task2_loss_type == "ce":
                l2_per = F.cross_entropy(
                    logits2,
                    y2.long(),
                    weight=task2_ce_weights,
                    label_smoothing=float(cfg.label_smoothing),
                    reduction="none",
                )
            else:
                if loss_task2 is None:
                    raise RuntimeError("loss_task2 is None but task2_loss_type != 'ce'")
                l2_per = loss_task2(logits2, y2.long(), reduction="none")  # type: ignore[arg-type]
            l2 = (l2_per * mw).mean()

            loss = (float(cfg.task1_weight) * l1) + (float(cfg.task2_weight) * l2)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)


def freeze_for_task2_head_only(model: nn.Module) -> None:
    _set_requires_grad(model, False)
    if not hasattr(model, "head_task2"):
        raise AttributeError("Model has no attribute 'head_task2' required for two-stage training.")
    _set_requires_grad(getattr(model, "head_task2"), True)


def train_one_epoch_task2_head_only(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    scaler: Optional[torch.cuda.amp.GradScaler],
    ema: Optional[EMA],
    loss_task2: Optional[nn.Module],
    task2_ce_weights: Optional[torch.Tensor],
    max_batches: Optional[int],
) -> float:
    # Linear-probe style: frozen backbone in eval, Task2 head in train.
    model.eval()
    if hasattr(model, "head_task2"):
        getattr(model, "head_task2").train()

    total_loss = 0.0
    n_batches = 0

    for b_idx, (x, y, meta) in enumerate(tqdm(loader, desc="TrainS2", leave=False, ascii=True)):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x.to(device, non_blocking=True)
        y2 = y["task2"].to(device, non_blocking=True)
        moments = list(meta["moment"])

        if cfg.use_moment_weighting:
            mw = _moment_weights(moments, weights=cfg.moment_loss_weights, device=device)
        else:
            mw = torch.ones(x.size(0), device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            out, _ = model(x)
            logits2 = out["task2_logits"]
            if cfg.task2_loss_type == "ce":
                l2_per = F.cross_entropy(
                    logits2,
                    y2.long(),
                    weight=task2_ce_weights,
                    label_smoothing=float(cfg.label_smoothing),
                    reduction="none",
                )
            else:
                if loss_task2 is None:
                    raise RuntimeError("loss_task2 is None but task2_loss_type != 'ce'")
                l2_per = loss_task2(logits2, y2.long(), reduction="none")  # type: ignore[arg-type]

            l2 = (l2_per * mw).mean()
            loss = float(cfg.task2_weight) * l2

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    cfg: Config,
    loss_task1: nn.Module,
    loss_task2: Optional[nn.Module],
    task2_ce_weights: Optional[torch.Tensor],
    max_batches: Optional[int],
) -> Tuple[float, Dict[str, float], Dict[str, float], float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    dummy_samples = 0
    decode_errors = 0
    available_frames_total = 0

    pain_logits_all: List[np.ndarray] = []
    pain_true_all: List[np.ndarray] = []
    task2_logits_all: List[np.ndarray] = []
    task2_true_all: List[np.ndarray] = []

    for b_idx, (x, y, meta) in enumerate(tqdm(loader, desc="Val", leave=False, ascii=True)):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x.to(device, non_blocking=True)
        y1 = y["pain_binary"].to(device, non_blocking=True)
        y2 = y["task2"].to(device, non_blocking=True)
        moments = list(meta["moment"])
        n_samples += int(x.size(0))

        # Data health stats (optional; present in v2.8+ dataset meta)
        try:
            is_dummy = meta.get("is_dummy", None)
            if is_dummy is not None:
                if torch.is_tensor(is_dummy):
                    dummy_samples += int(is_dummy.sum().item())
                else:
                    dummy_samples += int(sum(bool(v) for v in is_dummy))
        except Exception:
            pass
        try:
            n_err = meta.get("n_decode_errors", None)
            if n_err is not None:
                if torch.is_tensor(n_err):
                    decode_errors += int(n_err.sum().item())
                else:
                    decode_errors += int(sum(int(v) for v in n_err))
        except Exception:
            pass
        try:
            n_av = meta.get("n_available_frames", None)
            if n_av is not None:
                if torch.is_tensor(n_av):
                    available_frames_total += int(n_av.sum().item())
                else:
                    available_frames_total += int(sum(int(v) for v in n_av))
        except Exception:
            pass

        if cfg.use_moment_weighting:
            mw = _moment_weights(moments, weights=cfg.moment_loss_weights, device=device)
        else:
            mw = torch.ones(x.size(0), device=device, dtype=torch.float32)

        out, _ = model(x)

        l1_per = loss_task1(out["pain_logits"], y1)
        l1 = (l1_per * mw).mean()

        logits2 = out["task2_logits"]
        if cfg.task2_loss_type == "ce":
            l2_per = F.cross_entropy(
                logits2,
                y2.long(),
                weight=task2_ce_weights,
                label_smoothing=float(cfg.label_smoothing),
                reduction="none",
            )
        else:
            if loss_task2 is None:
                raise RuntimeError("loss_task2 is None but task2_loss_type != 'ce'")
            l2_per = loss_task2(logits2, y2.long(), reduction="none")  # type: ignore[arg-type]
        l2 = (l2_per * mw).mean()

        loss = (float(cfg.task1_weight) * l1) + (float(cfg.task2_weight) * l2)
        total_loss += float(loss.detach().item())
        n_batches += 1

        pain_logits_all.append(out["pain_logits"].detach().cpu().numpy())
        pain_true_all.append(y1.detach().cpu().numpy())
        task2_logits_all.append(out["task2_logits"].detach().cpu().numpy())
        task2_true_all.append(y2.detach().cpu().numpy())

    val_loss = total_loss / max(1, n_batches)

    pain_logits = np.concatenate(pain_logits_all, axis=0) if pain_logits_all else np.zeros((0,), dtype=np.float32)
    pain_targets = np.concatenate(pain_true_all, axis=0) if pain_true_all else np.zeros((0,), dtype=np.float32)
    t2_logits = np.concatenate(task2_logits_all, axis=0) if task2_logits_all else np.zeros((0, 3), dtype=np.float32)
    t2_targets = np.concatenate(task2_true_all, axis=0) if task2_true_all else np.zeros((0,), dtype=np.int64)

    t1 = compute_task1_metrics(pain_logits, pain_targets.astype(np.int64))
    probs = 1.0 / (1.0 + np.exp(-pain_logits))
    thr, t1_f1_opt = best_f1_threshold(probs, pain_targets.astype(np.int64))
    t2 = compute_task2_metrics(t2_logits, t2_targets.astype(np.int64))

    # Attach data-health stats (helps debug "all black" / missing-frame collapse)
    t2["n_val"] = float(len(t2_targets))
    t2["dummy_count"] = float(dummy_samples)
    t2["dummy_frac"] = float(dummy_samples / max(1, n_samples))
    t2["decode_errors"] = float(decode_errors)
    t2["avg_available_frames"] = float(available_frames_total / max(1, n_samples))

    return val_loss, t1, t2, float(thr), float(t1_f1_opt)


# ----------------------------
# CLI + path resolution
# ----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UCAPS v2.9 (v2.5-stable defaults + optional researcher ablations).")

    # paths
    p.add_argument("--base_path", type=str, default=None)
    p.add_argument("--project_dir", type=str, default=None)
    p.add_argument("--sequence_dir", type=str, default=None)
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument(
        "--no_auto_run_tag",
        action="store_true",
        help="Disable auto-generated run_tag. If unset and --run_tag is not provided, a unique run_tag is created to prevent overwriting.",
    )
    p.add_argument(
        "--overwrite_run",
        action="store_true",
        help="Allow writing into an existing (non-empty) run folder. Use with care; otherwise choose a new --run_tag.",
    )
    p.add_argument("--check_splits", action="store_true", help="Only run split/leakage checks and exit.")
    p.add_argument("--skip_split_check", action="store_true", help="Skip split/leakage checks (not recommended).")

    # core knobs
    p.add_argument("--task2_mode", choices=["3class", "4class"], default=None)
    p.add_argument("--input_mode", choices=["rgb", "grayst"], default=None, help="rgb (default) or grayst")
    p.add_argument(
        "--task2_loss_type",
        choices=["balanced_softmax", "ce", "cb_focal", "ldam_drw", "sce", "gce"],
        default=None,
    )
    p.add_argument(
        "--training_scheme",
        choices=["joint", "two_stage"],
        default=None,
        help="Training scheme (default: joint). two_stage trains Task1 then freezes backbone and trains Task2 head.",
    )
    p.add_argument("--two_stage", action="store_true", help="Alias for --training_scheme two_stage.")
    p.add_argument("--stage1_epochs", type=int, default=None, help="Two-stage: Stage1 epochs (default: num_epochs//2).")
    p.add_argument("--stage2_epochs", type=int, default=None, help="Two-stage: Stage2 epochs (default: num_epochs-stage1).")
    p.add_argument("--stage2_lr", type=float, default=None, help="Two-stage: Stage2 learning rate (default: learning_rate).")
    p.add_argument("--start_fold", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true")

    # speed/augmentation toggles (CPU/I/O bound)
    p.add_argument("--no_aug", action="store_true", help="Disable all training augmentations (fastest).")
    p.add_argument("--no_consistent_aug", action="store_true", help="Disable sequence-consistent augmentation.")
    p.add_argument("--no_affine", action="store_true", help="Disable RandomAffine (saves CPU).")
    p.add_argument("--no_blur", action="store_true", help="Disable GaussianBlur (saves CPU).")
    p.add_argument("--no_color_jitter", action="store_true", help="Disable ColorJitter (saves CPU).")
    p.add_argument(
        "--drive_workers",
        type=int,
        default=None,
        help="If reading from /content/drive and --num_workers is unset, use this worker count in notebook mode (default: 4).",
    )

    # data/training
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--lstm_hidden_size", type=int, default=None)
    p.add_argument("--lstm_num_layers", type=int, default=None)
    p.add_argument("--bidirectional_lstm", action="store_true", help="Use bidirectional LSTM (more capacity).")
    p.add_argument("--freeze_cnn", action="store_true", help="Freeze CNN backbone (train only LSTM/heads).")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--task1_weight", type=float, default=None)
    p.add_argument("--task2_weight", type=float, default=None)

    # scheduler + early stop
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--lr_patience", type=int, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--early_stop_min_epochs", type=int, default=None)
    p.add_argument("--early_stop_min_delta", type=float, default=None)

    # EMA
    p.add_argument("--ema", action="store_true", help="Enable EMA (default: disabled).")
    p.add_argument("--no_ema", action="store_true", help="Disable EMA (default: disabled).")
    p.add_argument("--ema_decay", type=float, default=None)

    # temporal sampling
    p.add_argument("--temporal_sampling", choices=["linspace", "uniform_offset", "random_clip"], default=None)
    p.add_argument("--time_reverse_p", type=float, default=None, help="Train-time probability of reversing time.")

    # loss params
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--task2_cb_beta", type=float, default=None)
    p.add_argument("--task2_focal_gamma", type=float, default=None)
    p.add_argument("--task2_ldam_max_m", type=float, default=None)
    p.add_argument("--task2_ldam_s", type=float, default=None)
    p.add_argument("--task2_drw_start_epoch", type=int, default=None)
    p.add_argument("--task2_sce_alpha", type=float, default=None)
    p.add_argument("--task2_sce_beta", type=float, default=None)
    p.add_argument("--task2_gce_q", type=float, default=None)

    # moment weights (comma-separated: M0=1,M2=4,...)
    p.add_argument("--moment_weights", type=str, default=None, help="Shared moment weights like M2=4.0,M3=2.0")
    # Backward compat: v2.8 used per-task moment weights. v2.9 applies weights shared across tasks.
    p.add_argument("--task1_moment_weights", type=str, default=None)
    p.add_argument("--task2_moment_weights", type=str, default=None)

    # best selection weights
    p.add_argument("--best_metric_task1_weight", type=float, default=None)
    p.add_argument("--best_metric_task2_weight", type=float, default=None)
    p.add_argument(
        "--combined_task2_metric",
        choices=["macro", "weighted"],
        default=None,
        help="Which Task2 F1 to use in the combined selection metric (default: weighted).",
    )
    p.add_argument(
        "--best_task2_metric",
        choices=["macro", "weighted"],
        default=None,
        help="Which Task2 F1 defines the Best-Task2 checkpoint (default: weighted).",
    )
    p.add_argument("--no_task1_opt_f1", action="store_true", help="Use Task1 fixed-threshold F1 for selection.")
    p.add_argument("--no_val_dists", action="store_true", help="Disable val prediction distribution printing.")

    # toggles
    p.add_argument("--no_stratified_sampler", action="store_true")
    p.add_argument("--no_moment_weighting", action="store_true")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--no_auto_resume", action="store_true")
    p.add_argument("--save_resume_checkpoint", action="store_true")

    # sanity
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"WARNING: Ignoring unknown CLI args: {unknown}")
    return args


def _parse_moment_weights_arg(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid moment weight item {part!r}. Expected like M2=4.0.")
        k, v = part.split("=", 1)
        key = str(k).strip().upper()
        if key not in {"M0", "M1", "M2", "M3", "M4"}:
            raise ValueError(f"Invalid moment key {key!r}. Expected one of M0..M4.")
        out[key] = float(v)
    return out


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.task2_mode is not None:
        cfg.task2_mode = str(args.task2_mode)
    if getattr(args, "input_mode", None) is not None:
        cfg.input_mode = str(args.input_mode)
    if getattr(args, "task2_loss_type", None) is not None:
        cfg.task2_loss_type = str(args.task2_loss_type)
    if getattr(args, "training_scheme", None) is not None:
        cfg.training_scheme = str(args.training_scheme)
    if bool(getattr(args, "two_stage", False)):
        cfg.training_scheme = "two_stage"
    if args.run_tag is not None:
        cfg.run_tag = str(args.run_tag)

    mapping = {
        "num_epochs": int,
        "stage1_epochs": int,
        "stage2_epochs": int,
        "batch_size": int,
        "num_workers": int,
        "max_frames": int,
        "learning_rate": float,
        "stage2_lr": float,
        "weight_decay": float,
        "grad_clip": float,
        "lstm_hidden_size": int,
        "lstm_num_layers": int,
        "dropout": float,
        "task1_weight": float,
        "task2_weight": float,
        "warmup_epochs": int,
        "lr_patience": int,
        "min_lr": float,
        "early_stop_patience": int,
        "early_stop_min_epochs": int,
        "early_stop_min_delta": float,
        "best_metric_task1_weight": float,
        "best_metric_task2_weight": float,
        "ema_decay": float,
        "time_reverse_p": float,
        "label_smoothing": float,
        "task2_cb_beta": float,
        "task2_focal_gamma": float,
        "task2_ldam_max_m": float,
        "task2_ldam_s": float,
        "task2_drw_start_epoch": int,
        "task2_sce_alpha": float,
        "task2_sce_beta": float,
        "task2_gce_q": float,
    }
    for k, cast in mapping.items():
        v = getattr(args, k, None)
        if v is None:
            continue
        setattr(cfg, k, cast(v))

    if args.resolution is not None:
        cfg.resolution = (int(args.resolution[0]), int(args.resolution[1]))

    if getattr(args, "temporal_sampling", None) is not None:
        cfg.temporal_sampling = str(args.temporal_sampling)

    if bool(args.no_stratified_sampler):
        cfg.use_stratified_sampler = False
    if bool(args.no_moment_weighting):
        cfg.use_moment_weighting = False
    if bool(getattr(args, "no_aug", False)):
        cfg.use_augmentations = False
    if bool(getattr(args, "no_consistent_aug", False)):
        cfg.use_consistent_aug = False
    if bool(getattr(args, "no_affine", False)):
        cfg.aug_use_affine = False
    if bool(getattr(args, "no_blur", False)):
        cfg.aug_use_blur = False
    if bool(getattr(args, "no_color_jitter", False)):
        cfg.aug_use_color_jitter = False
    if bool(getattr(args, "ema", False)):
        cfg.use_ema = True
    if bool(getattr(args, "no_ema", False)):
        cfg.use_ema = False
    if bool(getattr(args, "bidirectional_lstm", False)):
        cfg.use_bidirectional_lstm = True
    if bool(getattr(args, "freeze_cnn", False)):
        cfg.freeze_cnn = True
    if bool(getattr(args, "no_task1_opt_f1", False)):
        cfg.best_metric_use_task1_opt_f1 = False
    if getattr(args, "combined_task2_metric", None) is not None:
        cfg.combined_task2_metric = str(args.combined_task2_metric)
    if getattr(args, "best_task2_metric", None) is not None:
        cfg.best_task2_metric = str(args.best_task2_metric)
    if bool(getattr(args, "no_val_dists", False)):
        cfg.print_val_distributions = False

    if getattr(args, "moment_weights", None):
        cfg.moment_loss_weights.update(_parse_moment_weights_arg(args.moment_weights))
    if getattr(args, "task1_moment_weights", None):
        print("WARNING: v2.9 uses shared moment weights; applying --task1_moment_weights to both tasks.")
        cfg.moment_loss_weights.update(_parse_moment_weights_arg(args.task1_moment_weights))
    if getattr(args, "task2_moment_weights", None):
        print("WARNING: v2.9 uses shared moment weights; applying --task2_moment_weights to both tasks.")
        cfg.moment_loss_weights.update(_parse_moment_weights_arg(args.task2_moment_weights))



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


def _resolve_project_dir(base_path: Path, cfg: Config) -> Path:
    candidates = [base_path / cfg.project_dirname, base_path]
    for c in candidates:
        if (c / "train_val_test_splits_v2.json").exists() and (c / "sequence_label_mapping_v2.json").exists():
            return c
    raise FileNotFoundError(
        "Could not find project dir containing train_val_test_splits_v2.json + sequence_label_mapping_v2.json."
    )


def _resolve_sequence_dir(base_path: Path, project_dir: Path) -> Path:
    candidates = [base_path / "sequence", project_dir / "sequence"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find sequence dir (expected base_path/sequence or project_dir/sequence).")


def _resolve_run_dirs(project_dir: Path, cfg: Config) -> Tuple[Path, Path]:
    ckpt_root = project_dir / cfg.checkpoint_subdir
    res_root = project_dir / cfg.results_subdir
    ckpt_dir = (ckpt_root / cfg.run_tag) if cfg.run_tag else ckpt_root
    res_dir = (res_root / cfg.run_tag) if cfg.run_tag else res_root
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, res_dir


def main() -> None:
    args = _parse_args()
    cfg = Config()
    _apply_cli_overrides(cfg, args)
    if bool(getattr(args, "no_auto_run_tag", False)):
        cfg.auto_run_tag = False

    if args.dry_run:
        cfg.num_epochs = 1
        args.max_train_batches = 1
        args.max_val_batches = 1
        cfg.use_stratified_sampler = False
        print("dry_run enabled: 1 fold, 1 epoch, 1 train/val batch")

    seed_everything(int(args.seed))
    if args.fast:
        enable_fast_mode()

    auto_resume = bool(AUTO_RESUME)
    if args.auto_resume:
        auto_resume = True
    if args.no_auto_resume:
        auto_resume = False
    do_save_resume = bool(SAVE_RESUME_CHECKPOINT or bool(args.save_resume_checkpoint))

    base_path = Path(args.base_path) if args.base_path else _detect_base_path()
    project_dir = Path(args.project_dir) if args.project_dir else _resolve_project_dir(base_path, cfg)
    sequence_dir = Path(args.sequence_dir) if args.sequence_dir else _resolve_sequence_dir(base_path, project_dir)

    splits_file = project_dir / "train_val_test_splits_v2.json"
    mapping_file = project_dir / "sequence_label_mapping_v2.json"
    if not splits_file.exists() or not mapping_file.exists():
        raise FileNotFoundError(f"Missing splits/mapping in {project_dir}.")

    with open(splits_file, "r") as f:
        splits = json.load(f)
    with open(mapping_file, "r") as f:
        sequence_mapping = json.load(f)

    if isinstance(sequence_mapping, dict):
        if "sequences" in sequence_mapping:
            all_sequences = sequence_mapping["sequences"]
        else:
            all_sequences = [{"sequence_id": k, **v} for k, v in sequence_mapping.items()]
    else:
        all_sequences = sequence_mapping

    folds = get_folds_from_splits(splits)

    if bool(getattr(args, "check_splits", False)):
        check_splits_for_leakage(splits=splits, sequences=all_sequences)
        return
    if not bool(getattr(args, "skip_split_check", False)):
        check_splits_for_leakage(splits=splits, sequences=all_sequences)

    # Auto-generate run_tag to prevent overwriting, unless explicitly disabled.
    if bool(cfg.auto_run_tag) and (not str(cfg.run_tag)):
        cfg.run_tag = _ensure_unique_run_tag(project_dir, cfg)
        print(f"Auto-generated run_tag (to avoid overwriting): {cfg.run_tag}")

    checkpoint_dir, results_dir = _resolve_run_dirs(project_dir, cfg)

    # Guard against accidental overwrites if the run folder already has artifacts.
    if (not bool(getattr(args, "overwrite_run", False))) and (_dir_has_files(checkpoint_dir) or _dir_has_files(results_dir)):
        has_resume = any(checkpoint_dir.glob("resume_checkpoint_v2.9*_fold_*.pt"))
        if not (bool(auto_resume) and bool(has_resume)):
            raise RuntimeError(
                f"Refusing to overwrite existing run outputs in:\n"
                f"- {checkpoint_dir}\n- {results_dir}\n\n"
                f"Choose a new --run_tag (recommended), or pass --overwrite_run.\n"
                f"If you intended to resume, re-run with --save_resume_checkpoint enabled in the original run."
            )

    # Worker defaults (Colab): prefer workers>0 (Drive I/O is otherwise extremely slow).
    if running_in_notebook() and args.num_workers is None:
        if _looks_like_drive_path(sequence_dir):
            cfg.num_workers = int(args.drive_workers) if args.drive_workers is not None else 4
        # else: keep cfg.num_workers default

    print("=" * 80)
    print("TRAINING v2.9 (v2.5-stable CNN+LSTM dual classification)")
    print(f"project_dir: {project_dir}")
    print(f"sequence_dir: {sequence_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"results_dir: {results_dir}")
    print(f"run_tag: {cfg.run_tag!r} (auto_run_tag={bool(cfg.auto_run_tag)})")
    print(
        f"scheme={cfg.training_scheme} | freeze_cnn={bool(cfg.freeze_cnn)} | "
        f"stage1_epochs={cfg.stage1_epochs} | stage2_epochs={cfg.stage2_epochs} | stage2_lr={cfg.stage2_lr}"
    )
    print(
        f"task2_mode={cfg.task2_mode} | input_mode={cfg.input_mode} | loss2={cfg.task2_loss_type} | "
        f"epochs={cfg.num_epochs} | bs={cfg.batch_size} | lr={cfg.learning_rate:g} | workers={cfg.num_workers}"
    )
    print(
        f"sampler: stratified={bool(cfg.use_stratified_sampler)} | moment_weighting={bool(cfg.use_moment_weighting)} | "
        f"ema={bool(cfg.use_ema)} | auto_resume={bool(auto_resume)} | save_resume={bool(do_save_resume)}"
    )
    print(
        f"temporal_sampling={cfg.temporal_sampling} | time_reverse_p={cfg.time_reverse_p:g} | "
        f"bidirectional_lstm={bool(cfg.use_bidirectional_lstm)} | lstm_layers={cfg.lstm_num_layers}"
    )
    if running_in_notebook() and int(cfg.num_workers) == 0 and _looks_like_drive_path(sequence_dir):
        print("WARNING: num_workers=0 while reading from Google Drive -> very slow (CPU/I/O bound).")
        print("TIP: add --fast --num_workers 2 (or 4) and consider --max_frames 16 / --no_blur.")
    if not bool(cfg.use_augmentations):
        print("Augmentations: disabled (--no_aug).")
    else:
        aug_flags = []
        if cfg.aug_use_hflip:
            aug_flags.append("hflip")
        if cfg.aug_use_affine:
            aug_flags.append("affine")
        if cfg.aug_use_color_jitter:
            aug_flags.append("jitter")
        if cfg.aug_use_blur:
            aug_flags.append("blur")
        aug_kind = "consistent" if bool(cfg.use_consistent_aug) else "per-frame"
        print(f"Augmentations ({aug_kind}): {', '.join(aug_flags) if aug_flags else 'none'}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA not available. In Colab: Runtime -> Change runtime type -> GPU.")

    # Persist config for reproducibility
    cfg_path = results_dir / "config_v2.9.json"
    with open(cfg_path, "w") as f:
        json.dump({"cfg": cfg.__dict__, "args": vars(args)}, f, indent=2)
    print(f"Saved config: {cfg_path}")

    start_fold = int(args.start_fold) if args.start_fold is not None else int(START_FOLD)
    print(f"Start fold: {start_fold}")

    fold_summaries: List[dict] = []

    for fold_idx in range(start_fold, int(cfg.num_folds)):
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{cfg.num_folds - 1}")
        print("=" * 80)

        fold = folds[fold_idx]
        train_animals = fold["train_animals"]
        val_animals = fold["val_animals"]

        train_animals_set = {int(a) for a in train_animals}
        val_animals_set = {int(a) for a in val_animals}
        train_seqs = [s for s in all_sequences if _seq_animal_id(s) in train_animals_set]
        val_seqs = [s for s in all_sequences if _seq_animal_id(s) in val_animals_set]
        print(f"Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")

        # Task1 pos_weight (neg/pos)
        t1_labels = np.array([moment_to_task1_binary(s.get("moment", "unknown")) for s in train_seqs], dtype=np.int64)
        pos = float(t1_labels.sum())
        neg = float(len(t1_labels) - t1_labels.sum())
        pos_weight = None
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

        # Task2 counts (for imbalance-aware loss)
        K = task2_num_classes(cfg.task2_mode)
        t2_labels = np.array(
            [moment_to_task2(s.get("moment", "unknown"), task2_mode=cfg.task2_mode) for s in train_seqs],
            dtype=np.int64,
        )
        t2_counts = [int((t2_labels == k).sum()) for k in range(K)]
        print(f"Task2 train counts: {t2_counts}")

        # Data
        global_cache: dict = {}
        train_ds = FacialPainDataset_v2_8(train_seqs, sequence_dir, cfg, augment=True, global_cache=global_cache)
        val_ds = FacialPainDataset_v2_8(val_seqs, sequence_dir, cfg, augment=False, global_cache=global_cache)

        sampler = None
        if cfg.use_stratified_sampler:
            sampler = create_stratified_sampler(train_seqs)

        notebook = running_in_notebook()
        use_persistent_workers = bool(cfg.num_workers > 0 and (not notebook))
        loader_extra = {}
        if cfg.num_workers > 0:
            loader_extra["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            **loader_extra,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            **loader_extra,
        )

        scheme = str(getattr(cfg, "training_scheme", "joint")).strip().lower()
        if scheme not in {"joint", "two_stage"}:
            raise ValueError(f"Unknown training_scheme={cfg.training_scheme!r} (expected 'joint' or 'two_stage').")

        if scheme == "two_stage":
            # ---------------------------------
            # Two-stage training (research ablation):
            #   Stage1: Task1 only
            #   Stage2: Freeze backbone + Task1 head, train Task2 head
            # ---------------------------------
            if cfg.stage1_epochs is not None and cfg.stage2_epochs is not None:
                stage1_epochs = int(cfg.stage1_epochs)
                stage2_epochs = int(cfg.stage2_epochs)
            else:
                total = int(cfg.num_epochs)
                stage1_epochs = int(cfg.stage1_epochs) if cfg.stage1_epochs is not None else max(1, total // 2)
                stage2_epochs = int(cfg.stage2_epochs) if cfg.stage2_epochs is not None else max(1, total - stage1_epochs)

            if stage1_epochs <= 0 or stage2_epochs <= 0:
                raise ValueError(f"Invalid stage epochs: stage1={stage1_epochs}, stage2={stage2_epochs}")

            stage2_lr = float(cfg.stage2_lr) if cfg.stage2_lr is not None else float(cfg.learning_rate)
            print(f"Two-stage enabled: stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs} (total={stage1_epochs + stage2_epochs})")
            print(f"Two-stage stage2_lr={stage2_lr:g} | freeze_cnn={bool(cfg.freeze_cnn)} | ema={bool(cfg.use_ema)}")

            # -----------------
            # Stage 1 (Task1 only)
            # -----------------
            cfg_s1 = replace(cfg, task2_weight=0.0)
            model = TemporalPainModel_v2_8(cfg_s1).to(device)
            if bool(cfg_s1.freeze_cnn):
                _set_requires_grad(model.cnn, False)

            ema1: Optional[EMA] = EMA(model, decay=float(cfg_s1.ema_decay)) if bool(cfg_s1.use_ema) else None
            loss_task1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            loss_task2, task2_ce_weights, drw_weights = _make_task2_loss(cfg_s1, t2_counts=t2_counts, device=device)
            if loss_task2 is not None:
                loss_task2 = loss_task2.to(device)

            optimizer1 = AdamW(model.parameters(), lr=float(cfg_s1.learning_rate), weight_decay=float(cfg_s1.weight_decay))
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                mode="min",
                factor=0.5,
                patience=int(cfg_s1.lr_patience),
                min_lr=float(cfg_s1.min_lr),
            )

            scaler1: Optional[torch.cuda.amp.GradScaler] = None
            if device.type == "cuda":
                scaler1 = torch.cuda.amp.GradScaler()

            best_task1 = float("-inf")
            best_task1_epoch = -1
            patience = 0

            # Warm-start from existing Task1 checkpoint, if present.
            t1_ckpt_path = task1_checkpoint_path(checkpoint_dir, fold_idx)
            if t1_ckpt_path.exists():
                try:
                    prev = torch.load(t1_ckpt_path, map_location="cpu", weights_only=False)
                    best_task1 = float(prev.get("best_metric", float("-inf")))
                    best_task1_epoch = int(prev.get("epoch", -1)) + 1
                except Exception:
                    pass

            start_epoch1 = 0
            resume_s1 = resume_checkpoint_path_stage(checkpoint_dir, fold_idx, stage="stage1")
            if auto_resume and resume_s1.exists():
                print(f"Found Stage1 resume checkpoint: {resume_s1.name}")
                start_epoch1, resumed_best = try_resume(
                    checkpoint_path=resume_s1,
                    device=device,
                    model=model,
                    optimizer=optimizer1,
                    scheduler=scheduler1,
                    scaler=scaler1,
                    ema=ema1,
                )
                best_task1 = max(best_task1, float(resumed_best))
                print(f"Resumed Stage1 from epoch {start_epoch1} (best_task1={best_task1:.4f})")

            history_s1: List[dict] = []
            print("\n" + "-" * 80)
            print(f"STAGE 1/2 (Fold {fold_idx}): Task1-only training")
            print(f"Stage1 epochs: {stage1_epochs} | start_epoch: {start_epoch1} | lr: {cfg_s1.learning_rate:g}")
            print("-" * 80)
            for epoch in range(start_epoch1, int(stage1_epochs)):
                # Warmup LR (v2.5-style)
                if int(cfg_s1.warmup_epochs) > 0 and epoch < int(cfg_s1.warmup_epochs):
                    warm_lr = float(cfg_s1.learning_rate) * float(epoch + 1) / float(cfg_s1.warmup_epochs)
                    for pg in optimizer1.param_groups:
                        pg["lr"] = warm_lr

                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer=optimizer1,
                    device=device,
                    cfg=cfg_s1,
                    scaler=scaler1,
                    ema=ema1,
                    loss_task1=loss_task1,
                    loss_task2=loss_task2,
                    task2_ce_weights=task2_ce_weights,
                    max_batches=args.max_train_batches,
                )

                ema_backup: Optional[dict] = None
                if ema1 is not None:
                    ema_backup = ema1.apply_to(model)

                val_loss, t1, t2, t1_thr, t1_f1_opt = validate(
                    model,
                    val_loader,
                    device=device,
                    cfg=cfg_s1,
                    loss_task1=loss_task1,
                    loss_task2=loss_task2,
                    task2_ce_weights=task2_ce_weights,
                    max_batches=args.max_val_batches,
                )

                t1_f1_fixed = float(t1.get("f1", 0.0))
                t1_sel = float(t1_f1_opt) if bool(cfg_s1.best_metric_use_task1_opt_f1) else t1_f1_fixed
                combined = t1_sel

                if epoch >= int(cfg_s1.warmup_epochs):
                    scheduler1.step(float(val_loss))

                lr = float(optimizer1.param_groups[0]["lr"])
                print(
                    f"S1 Epoch {epoch+1:03d}/{stage1_epochs:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                    f"T1 f1 {t1_f1_fixed:.3f} (opt {t1_f1_opt:.3f} @thr {t1_thr:.2f}) | "
                    f"sel {combined:.3f} | lr {lr:.2e}"
                )
                if bool(cfg_s1.print_val_distributions):
                    K = int(task2_num_classes(cfg_s1.task2_mode))
                    pred_counts = [int(t2.get(f"pred_count_c{k}", 0.0)) for k in range(K)]
                    true_counts = [int(t2.get(f"true_count_c{k}", 0.0)) for k in range(K)]
                    print(
                        f"  ValDist | T1 pred_pos={t1.get('pred_pos_frac', 0.0):.2f} | "
                        f"T2 pred={pred_counts} true={true_counts} | "
                        f"dummy={int(t2.get('dummy_count', 0.0))}/{int(max(1, t2.get('n_val', 0.0)))} "
                        f"(avg_avail_frames={t2.get('avg_available_frames', 0.0):.1f}, decode_err={int(t2.get('decode_errors', 0.0))})"
                    )

                row = {
                    "stage": "stage1",
                    "fold": int(fold_idx),
                    "epoch": int(epoch + 1),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "task1_acc": float(t1["acc"]),
                    "task1_f1_fixed": float(t1_f1_fixed),
                    "task1_thr_opt": float(t1_thr),
                    "task1_f1_opt": float(t1_f1_opt),
                    "task1_f1_sel": float(t1_sel),
                    "task2_acc": float(t2["acc"]),
                    "task2_f1_weighted": float(t2["f1_weighted"]),
                    "task2_f1_macro": float(t2["f1_macro"]),
                    "combined_metric": float(combined),
                    "lr": float(lr),
                    "val_used_ema": float(1.0 if (ema1 is not None) else 0.0),
                }
                history_s1.append(row)

                min_delta = float(cfg_s1.early_stop_min_delta)
                if (t1_sel - best_task1) > min_delta:
                    best_task1 = float(t1_sel)
                    best_task1_epoch = int(epoch + 1)
                    patience = 0
                    save_best_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        cfg=cfg_s1,
                        out_path=task1_checkpoint_path(checkpoint_dir, fold_idx),
                        label="Best Task1 model (Stage1)",
                        best_metric_name="task1_f1_sel",
                        best_metric=best_task1,
                        best_components=row,
                    )
                else:
                    patience += 1

                if ema1 is not None and ema_backup is not None:
                    ema1.restore(model, ema_backup)

                if do_save_resume:
                    save_resume_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer1,
                        scheduler=scheduler1,
                        scaler=scaler1,
                        best_combined_metric=best_task1,
                        cfg=cfg_s1,
                        ema_shadow=(ema1.shadow if ema1 is not None else None),
                        out_path=resume_s1,
                    )

                if (epoch + 1) >= int(cfg_s1.early_stop_min_epochs) and patience >= int(cfg_s1.early_stop_patience):
                    print(f"Stage1 early stopping: patience {patience}/{cfg_s1.early_stop_patience}")
                    break

                if args.dry_run:
                    break

            hist_s1_csv = results_dir / f"training_history_v2.9_fold_{fold_idx}_stage1.csv"
            pd.DataFrame(history_s1).to_csv(hist_s1_csv, index=False)
            print(f"Saved Stage1 history: {hist_s1_csv}")
            print(f"Stage1 best_task1_sel_f1: {best_task1:.4f} @ epoch {best_task1_epoch}")

            # Load best Stage1 weights for Stage2.
            if t1_ckpt_path.exists():
                ckpt = torch.load(t1_ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=True)

            # -----------------
            # Stage 2 (Task2 head only)
            # -----------------
            cfg_s2 = replace(cfg, task1_weight=0.0)
            freeze_for_task2_head_only(model)

            print("\n" + "-" * 80)
            print(f"STAGE 2/2 (Fold {fold_idx}): Task2-head-only fine-tuning (backbone + Task1 head frozen)")
            print(f"Stage2 epochs: {stage2_epochs} | stage2_lr: {stage2_lr:g}")
            print("-" * 80)

            ema2: Optional[EMA] = EMA(model, decay=float(cfg_s2.ema_decay)) if bool(cfg_s2.use_ema) else None
            stage2_params = [p for p in model.parameters() if p.requires_grad]
            optimizer2 = AdamW(stage2_params, lr=float(stage2_lr), weight_decay=float(cfg_s2.weight_decay))
            scheduler2 = ReduceLROnPlateau(
                optimizer2,
                mode="min",
                factor=0.5,
                patience=int(cfg_s2.lr_patience),
                min_lr=float(cfg_s2.min_lr),
            )

            scaler2: Optional[torch.cuda.amp.GradScaler] = None
            if device.type == "cuda":
                scaler2 = torch.cuda.amp.GradScaler()

            best_combined = float("-inf")
            best_combined_epoch = -1
            best_task2_macro = float("-inf")
            best_task2_macro_epoch = -1
            best_task2_weighted = float("-inf")
            best_task2_weighted_epoch = -1
            patience2 = 0

            # Initialize from existing checkpoints (if present)
            for pth, kind in [
                (best_checkpoint_path(checkpoint_dir, fold_idx), "combined"),
                (task2_checkpoint_path(checkpoint_dir, fold_idx), "task2"),
            ]:
                if not pth.exists():
                    continue
                try:
                    prev = torch.load(pth, map_location="cpu", weights_only=False)
                    metric = float(prev.get("best_metric", float("-inf")))
                    ep = int(prev.get("epoch", -1)) + 1
                    if kind == "combined":
                        best_combined, best_combined_epoch = metric, ep
                    else:
                        metric_name = str(prev.get("best_metric_name", "")).lower()
                        if "weighted" in metric_name:
                            best_task2_weighted, best_task2_weighted_epoch = metric, ep
                        elif "macro" in metric_name:
                            best_task2_macro, best_task2_macro_epoch = metric, ep
                except Exception:
                    pass

            start_epoch2 = 0
            resume_s2 = resume_checkpoint_path_stage(checkpoint_dir, fold_idx, stage="stage2")
            if auto_resume and resume_s2.exists():
                print(f"Found Stage2 resume checkpoint: {resume_s2.name}")
                start_epoch2, resumed_best = try_resume(
                    checkpoint_path=resume_s2,
                    device=device,
                    model=model,
                    optimizer=optimizer2,
                    scheduler=scheduler2,
                    scaler=scaler2,
                    ema=ema2,
                )
                best_combined = max(best_combined, float(resumed_best))
                print(f"Resumed Stage2 from epoch {start_epoch2} (best_combined={best_combined:.4f})")

            history_s2: List[dict] = []
            for epoch in range(start_epoch2, int(stage2_epochs)):
                # LDAM-DRW: start re-weighting after a few epochs
                if cfg_s2.task2_loss_type == "ldam_drw" and isinstance(loss_task2, LDAMLoss):
                    if drw_weights is not None and epoch >= int(cfg_s2.task2_drw_start_epoch):
                        loss_task2.set_class_weight(drw_weights)
                    else:
                        loss_task2.set_class_weight(None)

                # Warmup LR
                if int(cfg_s2.warmup_epochs) > 0 and epoch < int(cfg_s2.warmup_epochs):
                    warm_lr = float(stage2_lr) * float(epoch + 1) / float(cfg_s2.warmup_epochs)
                    for pg in optimizer2.param_groups:
                        pg["lr"] = warm_lr

                train_loss = train_one_epoch_task2_head_only(
                    model,
                    train_loader,
                    optimizer=optimizer2,
                    device=device,
                    cfg=cfg_s2,
                    scaler=scaler2,
                    ema=ema2,
                    loss_task2=loss_task2,
                    task2_ce_weights=task2_ce_weights,
                    max_batches=args.max_train_batches,
                )

                ema_backup: Optional[dict] = None
                if ema2 is not None:
                    ema_backup = ema2.apply_to(model)

                val_loss, t1, t2, t1_thr, t1_f1_opt = validate(
                    model,
                    val_loader,
                    device=device,
                    cfg=cfg_s2,
                    loss_task1=loss_task1,
                    loss_task2=loss_task2,
                    task2_ce_weights=task2_ce_weights,
                    max_batches=args.max_val_batches,
                )

                w1 = float(cfg.best_metric_task1_weight)
                w2 = float(cfg.best_metric_task2_weight)
                denom = (w1 + w2) if (w1 + w2) > 0 else 1.0
                t1_f1_fixed = float(t1.get("f1", 0.0))
                t1_sel = float(t1_f1_opt) if bool(cfg.best_metric_use_task1_opt_f1) else t1_f1_fixed
                if str(cfg.combined_task2_metric).lower() == "weighted":
                    t2_sel = float(t2.get("f1_weighted", 0.0))
                else:
                    t2_sel = float(t2.get("f1_macro", 0.0))
                combined = ((w1 * t1_sel) + (w2 * t2_sel)) / denom

                if epoch >= int(cfg_s2.warmup_epochs):
                    scheduler2.step(float(val_loss))

                lr = float(optimizer2.param_groups[0]["lr"])
                print(
                    f"S2 Epoch {epoch+1:03d}/{stage2_epochs:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                    f"T1 f1 {t1_f1_fixed:.3f} (opt {t1_f1_opt:.3f} @thr {t1_thr:.2f}) | "
                    f"T2 f1m {t2['f1_macro']:.3f} (f1w {t2['f1_weighted']:.3f}) | "
                    f"combined {combined:.3f} | lr {lr:.2e}"
                )
                if bool(cfg_s2.print_val_distributions):
                    K = int(task2_num_classes(cfg_s2.task2_mode))
                    pred_counts = [int(t2.get(f"pred_count_c{k}", 0.0)) for k in range(K)]
                    true_counts = [int(t2.get(f"true_count_c{k}", 0.0)) for k in range(K)]
                    print(
                        f"  ValDist | T1 pred_pos={t1.get('pred_pos_frac', 0.0):.2f} | "
                        f"T2 pred={pred_counts} true={true_counts} | "
                        f"dummy={int(t2.get('dummy_count', 0.0))}/{int(max(1, t2.get('n_val', 0.0)))} "
                        f"(avg_avail_frames={t2.get('avg_available_frames', 0.0):.1f}, decode_err={int(t2.get('decode_errors', 0.0))})"
                    )

                row = {
                    "stage": "stage2",
                    "fold": int(fold_idx),
                    "epoch": int(epoch + 1),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "task1_acc": float(t1["acc"]),
                    "task1_f1_fixed": float(t1_f1_fixed),
                    "task1_thr_opt": float(t1_thr),
                    "task1_f1_opt": float(t1_f1_opt),
                    "task1_f1_sel": float(t1_sel),
                    "task2_acc": float(t2["acc"]),
                    "task2_f1_weighted": float(t2["f1_weighted"]),
                    "task2_f1_macro": float(t2["f1_macro"]),
                    "combined_metric": float(combined),
                    "lr": float(lr),
                    "val_used_ema": float(1.0 if (ema2 is not None) else 0.0),
                }
                history_s2.append(row)

                min_delta = float(cfg_s2.early_stop_min_delta)
                if (combined - best_combined) > min_delta:
                    best_combined = float(combined)
                    best_combined_epoch = int(epoch + 1)
                    patience2 = 0
                    save_best_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        cfg=cfg_s2,
                        out_path=best_checkpoint_path(checkpoint_dir, fold_idx),
                        label="Best combined model (Stage2)",
                        best_metric_name="combined_metric",
                        best_metric=best_combined,
                        best_components={
                            **row,
                            "selection_task1": float(t1_sel),
                            "selection_task2": float(t2_sel),
                        },
                    )
                else:
                    patience2 += 1

                t2_macro = float(t2.get("f1_macro", 0.0))
                t2_weighted = float(t2.get("f1_weighted", 0.0))

                if (t2_macro - best_task2_macro) > min_delta:
                    best_task2_macro = t2_macro
                    best_task2_macro_epoch = int(epoch + 1)
                    if str(cfg.best_task2_metric).lower() == "macro":
                        save_best_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            fold_idx=fold_idx,
                            epoch=epoch,
                            model_state_dict=model.state_dict(),
                            cfg=cfg_s2,
                            out_path=task2_checkpoint_path(checkpoint_dir, fold_idx),
                            label="Best Task2 model (Stage2)",
                            best_metric_name="task2_f1_macro",
                            best_metric=best_task2_macro,
                            best_components=row,
                        )

                if (t2_weighted - best_task2_weighted) > min_delta:
                    best_task2_weighted = t2_weighted
                    best_task2_weighted_epoch = int(epoch + 1)
                    if str(cfg.best_task2_metric).lower() == "weighted":
                        save_best_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            fold_idx=fold_idx,
                            epoch=epoch,
                            model_state_dict=model.state_dict(),
                            cfg=cfg_s2,
                            out_path=task2_checkpoint_path(checkpoint_dir, fold_idx),
                            label="Best Task2 model (Stage2)",
                            best_metric_name="task2_f1_weighted",
                            best_metric=best_task2_weighted,
                            best_components=row,
                        )

                if ema2 is not None and ema_backup is not None:
                    ema2.restore(model, ema_backup)

                if do_save_resume:
                    save_resume_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer2,
                        scheduler=scheduler2,
                        scaler=scaler2,
                        best_combined_metric=best_combined,
                        cfg=cfg_s2,
                        ema_shadow=(ema2.shadow if ema2 is not None else None),
                        out_path=resume_s2,
                    )

                if (epoch + 1) >= int(cfg_s2.early_stop_min_epochs) and patience2 >= int(cfg_s2.early_stop_patience):
                    print(f"Stage2 early stopping: patience {patience2}/{cfg_s2.early_stop_patience}")
                    break

                if args.dry_run:
                    break

            hist_s2_csv = results_dir / f"training_history_v2.9_fold_{fold_idx}_stage2.csv"
            pd.DataFrame(history_s2).to_csv(hist_s2_csv, index=False)
            print(f"Saved Stage2 history: {hist_s2_csv}")
            print(f"Best combined (Stage2): {best_combined:.4f} @ epoch {best_combined_epoch}")
            print(f"Best task2_macro_f1 (Stage2): {best_task2_macro:.4f} @ epoch {best_task2_macro_epoch}")
            print(f"Best task2_weighted_f1 (Stage2): {best_task2_weighted:.4f} @ epoch {best_task2_weighted_epoch}")
            print(f"Best Task2 checkpoint metric: {str(cfg.best_task2_metric).lower()}")

            fold_summaries.append(
                {
                    "fold": int(fold_idx),
                    "task2_mode": str(cfg.task2_mode),
                    "training_scheme": "two_stage",
                    "stage1_epochs": int(stage1_epochs),
                    "stage2_epochs": int(stage2_epochs),
                    "freeze_cnn": float(1.0 if bool(cfg.freeze_cnn) else 0.0),
                    "best_combined_metric": float(best_combined),
                    "best_combined_epoch": int(best_combined_epoch),
                    "best_task1_sel_f1": float(best_task1),
                    "best_task1_epoch": int(best_task1_epoch),
                    "best_task2_f1": float(
                        best_task2_weighted if str(cfg.best_task2_metric).lower() == "weighted" else best_task2_macro
                    ),
                    "best_task2_epoch": int(
                        best_task2_weighted_epoch if str(cfg.best_task2_metric).lower() == "weighted" else best_task2_macro_epoch
                    ),
                    "best_task2_f1_macro": float(best_task2_macro),
                    "best_task2_macro_epoch": int(best_task2_macro_epoch),
                    "best_task2_f1_weighted": float(best_task2_weighted),
                    "best_task2_weighted_epoch": int(best_task2_weighted_epoch),
                    "combined_task2_metric": str(cfg.combined_task2_metric).lower(),
                    "best_task2_metric": str(cfg.best_task2_metric).lower(),
                    "best_model_combined": best_checkpoint_path(checkpoint_dir, fold_idx).name,
                    "best_model_task1": task1_checkpoint_path(checkpoint_dir, fold_idx).name,
                    "best_model_task2": task2_checkpoint_path(checkpoint_dir, fold_idx).name,
                }
            )

            if args.dry_run:
                break

            continue

        # -----------------
        # Joint training (default)
        # -----------------
        model = TemporalPainModel_v2_8(cfg).to(device)
        if bool(cfg.freeze_cnn):
            _set_requires_grad(model.cnn, False)
        ema: Optional[EMA] = EMA(model, decay=float(cfg.ema_decay)) if bool(cfg.use_ema) else None

        loss_task1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        loss_task2, task2_ce_weights, drw_weights = _make_task2_loss(cfg, t2_counts=t2_counts, device=device)
        if loss_task2 is not None:
            loss_task2 = loss_task2.to(device)

        optimizer = AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(cfg.lr_patience),
            min_lr=float(cfg.min_lr),
        )

        scaler: Optional[torch.cuda.amp.GradScaler] = None
        if device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # Best tracking (combined metric drives early stopping)
        best_combined = float("-inf")
        best_task1 = float("-inf")
        best_task2_macro = float("-inf")
        best_task2_weighted = float("-inf")
        best_combined_epoch = -1
        best_task1_epoch = -1
        best_task2_macro_epoch = -1
        best_task2_weighted_epoch = -1
        patience = 0

        # Initialize from existing checkpoints (if present)
        for pth, kind in [
            (best_checkpoint_path(checkpoint_dir, fold_idx), "combined"),
            (task1_checkpoint_path(checkpoint_dir, fold_idx), "task1"),
            (task2_checkpoint_path(checkpoint_dir, fold_idx), "task2"),
        ]:
            if not pth.exists():
                continue
            try:
                prev = torch.load(pth, map_location="cpu", weights_only=False)
                metric = float(prev.get("best_metric", float("-inf")))
                ep = int(prev.get("epoch", -1)) + 1
                if kind == "combined":
                    best_combined, best_combined_epoch = metric, ep
                elif kind == "task1":
                    best_task1, best_task1_epoch = metric, ep
                else:
                    metric_name = str(prev.get("best_metric_name", "")).lower()
                    if "weighted" in metric_name:
                        best_task2_weighted, best_task2_weighted_epoch = metric, ep
                    elif "macro" in metric_name:
                        best_task2_macro, best_task2_macro_epoch = metric, ep
                    else:
                        # Fallback: interpret based on current preference
                        if str(cfg.best_task2_metric).lower() == "weighted":
                            best_task2_weighted, best_task2_weighted_epoch = metric, ep
                        else:
                            best_task2_macro, best_task2_macro_epoch = metric, ep
            except Exception:
                pass

        start_epoch = 0
        resume_path = resume_checkpoint_path(checkpoint_dir, fold_idx)
        if auto_resume and resume_path.exists():
            print(f"Found resume checkpoint: {resume_path.name}")
            start_epoch, resumed_best = try_resume(
                checkpoint_path=resume_path,
                device=device,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
            )
            best_combined = max(best_combined, float(resumed_best))
            print(f"Resumed from epoch {start_epoch} (best_combined={best_combined:.4f})")

        history: List[dict] = []

        for epoch in range(start_epoch, int(cfg.num_epochs)):
            # LDAM-DRW: start re-weighting after a few epochs
            if cfg.task2_loss_type == "ldam_drw" and isinstance(loss_task2, LDAMLoss):
                if drw_weights is not None and epoch >= int(cfg.task2_drw_start_epoch):
                    loss_task2.set_class_weight(drw_weights)
                else:
                    loss_task2.set_class_weight(None)

            # Warmup LR (v2.5-style)
            if int(cfg.warmup_epochs) > 0 and epoch < int(cfg.warmup_epochs):
                warm_lr = float(cfg.learning_rate) * float(epoch + 1) / float(cfg.warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                scaler=scaler,
                ema=ema,
                loss_task1=loss_task1,
                loss_task2=loss_task2,
                task2_ce_weights=task2_ce_weights,
                max_batches=args.max_train_batches,
            )

            # Validate with EMA weights (if enabled), and save best checkpoints using EMA weights.
            ema_backup: Optional[dict] = None
            if ema is not None:
                ema_backup = ema.apply_to(model)

            val_loss, t1, t2, t1_thr, t1_f1_opt = validate(
                model,
                val_loader,
                device=device,
                cfg=cfg,
                loss_task1=loss_task1,
                loss_task2=loss_task2,
                task2_ce_weights=task2_ce_weights,
                max_batches=args.max_val_batches,
            )

            w1 = float(cfg.best_metric_task1_weight)
            w2 = float(cfg.best_metric_task2_weight)
            denom = (w1 + w2) if (w1 + w2) > 0 else 1.0
            t1_f1_fixed = float(t1.get("f1", 0.0))
            t1_sel = float(t1_f1_opt) if bool(cfg.best_metric_use_task1_opt_f1) else t1_f1_fixed
            if str(cfg.combined_task2_metric).lower() == "weighted":
                t2_sel = float(t2.get("f1_weighted", 0.0))
            else:
                t2_sel = float(t2.get("f1_macro", 0.0))
            combined = ((w1 * t1_sel) + (w2 * t2_sel)) / denom

            if epoch >= int(cfg.warmup_epochs):
                scheduler.step(float(val_loss))

            lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch+1:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"T1 f1 {t1_f1_fixed:.3f} (opt {t1_f1_opt:.3f} @thr {t1_thr:.2f}) | "
                f"T2 f1m {t2['f1_macro']:.3f} (f1w {t2['f1_weighted']:.3f}) | "
                f"combined {combined:.3f} | lr {lr:.2e}"
            )
            if bool(cfg.print_val_distributions):
                K = int(task2_num_classes(cfg.task2_mode))
                pred_counts = [int(t2.get(f"pred_count_c{k}", 0.0)) for k in range(K)]
                true_counts = [int(t2.get(f"true_count_c{k}", 0.0)) for k in range(K)]
                print(
                    f"  ValDist | T1 pred_pos={t1.get('pred_pos_frac', 0.0):.2f} | "
                    f"T2 pred={pred_counts} true={true_counts} | "
                    f"dummy={int(t2.get('dummy_count', 0.0))}/{int(max(1, t2.get('n_val', 0.0)))} "
                    f"(avg_avail_frames={t2.get('avg_available_frames', 0.0):.1f}, decode_err={int(t2.get('decode_errors', 0.0))})"
                )

            row = {
                "fold": int(fold_idx),
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "task1_acc": float(t1["acc"]),
                "task1_f1_fixed": float(t1_f1_fixed),
                "task1_thr_opt": float(t1_thr),
                "task1_f1_opt": float(t1_f1_opt),
                "task1_f1_sel": float(t1_sel),
                "task2_acc": float(t2["acc"]),
                "task2_f1_weighted": float(t2["f1_weighted"]),
                "task2_f1_macro": float(t2["f1_macro"]),
                **{f"task2_{k}": float(v) for k, v in t2.items() if k not in {"acc", "f1_weighted", "f1_macro"}},
                "combined_metric": float(combined),
                "lr": float(lr),
                "val_used_ema": float(1.0 if (ema is not None) else 0.0),
            }
            history.append(row)

            min_delta = float(cfg.early_stop_min_delta)

            # Best combined (primary)
            if (combined - best_combined) > min_delta:
                best_combined = float(combined)
                best_combined_epoch = int(epoch + 1)
                patience = 0
                save_best_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    fold_idx=fold_idx,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    cfg=cfg,
                    out_path=best_checkpoint_path(checkpoint_dir, fold_idx),
                    label="Best combined model",
                    best_metric_name="combined_metric",
                    best_metric=best_combined,
                    best_components={
                        **row,
                        "selection_task1": float(t1_sel),
                        "selection_task2": float(t2_sel),
                    },
                )
            else:
                patience += 1

            # Best Task1
            if (float(t1_sel) - best_task1) > min_delta:
                best_task1 = float(t1_sel)
                best_task1_epoch = int(epoch + 1)
                save_best_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    fold_idx=fold_idx,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    cfg=cfg,
                    out_path=task1_checkpoint_path(checkpoint_dir, fold_idx),
                    label="Best Task1 model",
                    best_metric_name="task1_f1_sel",
                    best_metric=best_task1,
                    best_components=row,
                )

            # Best Task2 (track both macro and weighted; save whichever cfg.best_task2_metric requests)
            t2_macro = float(t2.get("f1_macro", 0.0))
            t2_weighted = float(t2.get("f1_weighted", 0.0))

            if (t2_macro - best_task2_macro) > min_delta:
                best_task2_macro = t2_macro
                best_task2_macro_epoch = int(epoch + 1)
                if str(cfg.best_task2_metric).lower() == "macro":
                    save_best_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        cfg=cfg,
                        out_path=task2_checkpoint_path(checkpoint_dir, fold_idx),
                        label="Best Task2 model",
                        best_metric_name="task2_f1_macro",
                        best_metric=best_task2_macro,
                        best_components=row,
                    )

            if (t2_weighted - best_task2_weighted) > min_delta:
                best_task2_weighted = t2_weighted
                best_task2_weighted_epoch = int(epoch + 1)
                if str(cfg.best_task2_metric).lower() == "weighted":
                    save_best_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        cfg=cfg,
                        out_path=task2_checkpoint_path(checkpoint_dir, fold_idx),
                        label="Best Task2 model",
                        best_metric_name="task2_f1_weighted",
                        best_metric=best_task2_weighted,
                        best_components=row,
                    )

            # Restore raw weights for the next training epoch / resume checkpoint.
            if ema is not None and ema_backup is not None:
                ema.restore(model, ema_backup)

            if do_save_resume:
                save_resume_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    fold_idx=fold_idx,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_combined_metric=best_combined,
                    cfg=cfg,
                    ema_shadow=(ema.shadow if ema is not None else None),
                )

            if (epoch + 1) >= int(cfg.early_stop_min_epochs) and patience >= int(cfg.early_stop_patience):
                print(f"Early stopping: patience {patience}/{cfg.early_stop_patience}")
                break

            if args.dry_run:
                break

        # Save fold history
        hist_csv = results_dir / f"training_history_v2.9_fold_{fold_idx}.csv"
        pd.DataFrame(history).to_csv(hist_csv, index=False)
        print(f"Saved fold history: {hist_csv}")
        print(f"Best combined: {best_combined:.4f} @ epoch {best_combined_epoch}")
        print(f"Best task1_sel_f1: {best_task1:.4f} @ epoch {best_task1_epoch}")
        print(f"Best task2_macro_f1: {best_task2_macro:.4f} @ epoch {best_task2_macro_epoch}")
        print(f"Best task2_weighted_f1: {best_task2_weighted:.4f} @ epoch {best_task2_weighted_epoch}")
        print(f"Best Task2 checkpoint metric: {str(cfg.best_task2_metric).lower()}")

        fold_summaries.append(
            {
                "fold": int(fold_idx),
                "task2_mode": str(cfg.task2_mode),
                "training_scheme": "joint",
                "freeze_cnn": float(1.0 if bool(cfg.freeze_cnn) else 0.0),
                "best_combined_metric": float(best_combined),
                "best_combined_epoch": int(best_combined_epoch),
                "best_task1_sel_f1": float(best_task1),
                "best_task1_epoch": int(best_task1_epoch),
                # Backward-compat convenience: these represent the metric/epoch used by the saved Best-Task2 checkpoint.
                "best_task2_f1": float(best_task2_weighted if str(cfg.best_task2_metric).lower() == "weighted" else best_task2_macro),
                "best_task2_epoch": int(
                    best_task2_weighted_epoch if str(cfg.best_task2_metric).lower() == "weighted" else best_task2_macro_epoch
                ),
                "best_task2_f1_macro": float(best_task2_macro),
                "best_task2_macro_epoch": int(best_task2_macro_epoch),
                "best_task2_f1_weighted": float(best_task2_weighted),
                "best_task2_weighted_epoch": int(best_task2_weighted_epoch),
                "combined_task2_metric": str(cfg.combined_task2_metric).lower(),
                "best_task2_metric": str(cfg.best_task2_metric).lower(),
                "best_model_combined": best_checkpoint_path(checkpoint_dir, fold_idx).name,
                "best_model_task1": task1_checkpoint_path(checkpoint_dir, fold_idx).name,
                "best_model_task2": task2_checkpoint_path(checkpoint_dir, fold_idx).name,
            }
        )

        if args.dry_run:
            break

    summary_path = results_dir / "fold_summary_v2.9.csv"
    pd.DataFrame(fold_summaries).to_csv(summary_path, index=False)
    print("\nTraining complete.")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
