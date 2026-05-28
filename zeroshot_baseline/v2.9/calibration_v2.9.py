# ============================================================================
# CALIBRATION METRICS - v2.9 (held-out test)
#
# Task 2 of 2: Compute Brier score, ECE (10-bin), and reliability diagram data
# from the same predictions CSV produced by evaluate_test_set_v2.9_cli.py.
#
# Usage (one of):
#   python calibration_v2.9.py --predictions_csv path/to/predictions.csv
#   python calibration_v2.9.py --run_tag v2.9_20260221_014705 --ckpt_kind task2
#
# Output: JSON with Brier, ECE, and per-bin reliability data; CSV for plotting.
# If --out_json/--out_reliability_csv are not provided, files are auto-written
# next to the predictions CSV (Colab-friendly default behavior).
# ============================================================================

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration metrics (Brier, ECE, reliability) for v2.9 held-out predictions."
    )
    p.add_argument(
        "--predictions_csv",
        type=str,
        default=None,
        help="Path to test_eval_v2.9_*_predictions.csv from evaluate_test_set_v2.9_cli.py",
    )
    p.add_argument("--run_tag", type=str, default=None, help="Run tag (used with --ckpt_kind to find CSV).")
    p.add_argument(
        "--ckpt_kind",
        type=str,
        default="task2",
        choices=["combined", "task1", "task2"],
        help="Checkpoint kind; used with --run_tag to infer predictions path.",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results dir when using --run_tag.",
    )
    p.add_argument("--project_dir", type=str, default=None, help="Project dir when using --run_tag.")
    p.add_argument("--n_bins", type=int, default=10, help="Number of bins for ECE and reliability diagram.")
    p.add_argument("--out_json", type=str, default=None, help="Path to write calibration results JSON.")
    p.add_argument("--out_reliability_csv", type=str, default=None, help="Path for reliability-bin CSV.")
    return p.parse_args()


def _resolve_predictions_path(
    run_tag: str,
    ckpt_kind: str,
    test_animals: List[int],
    results_dir: Optional[Path],
    project_dir: Path,
) -> Path:
    if results_dir is None:
        results_dir = project_dir / "results_v2.9" / run_tag
    animals_str = "-".join(str(a) for a in sorted(test_animals))
    fname = f"test_eval_v2.9_{ckpt_kind}_animals_{animals_str}_predictions.csv"
    path = results_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}. Run evaluate_test_set_v2.9_cli.py first.")
    return path


# ---------- Task 1 (binary) ----------


def brier_score_binary(probs: np.ndarray, targets: np.ndarray) -> float:
    """Brier score for binary: mean((p - y)^2)."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    return float(np.mean((probs - targets) ** 2))


def ece_binary(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    ECE for binary: 10 equal-width bins on predicted probability.
    Returns (ECE, list of {bin_lo, bin_hi, mean_conf, mean_acc, count}).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[-1] = 1.0 + 1e-9
    n = len(probs)
    ece = 0.0
    reliability = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        count = int(np.sum(mask))
        if count == 0:
            reliability.append({"bin_lo": lo, "bin_hi": hi, "mean_conf": np.nan, "mean_acc": np.nan, "count": 0})
            continue
        mean_conf = float(np.mean(probs[mask]))
        mean_acc = float(np.mean(targets[mask]))
        ece += (count / n) * abs(mean_acc - mean_conf)
        reliability.append({"bin_lo": lo, "bin_hi": hi, "mean_conf": mean_conf, "mean_acc": mean_acc, "count": count})
    return float(ece), reliability


# ---------- Task 2 (multi-class) ----------


def brier_score_multiclass(probs: np.ndarray, targets: np.ndarray) -> float:
    """Multi-class Brier: mean over samples of sum over classes (p_c - 1[c==y])^2."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    n, k = probs.shape
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), np.clip(targets, 0, k - 1)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def ece_multiclass(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    ECE for multi-class: bin by confidence (max prob), accuracy in bin = fraction correct.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).ravel()
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    pred_class = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = (pred_class == targets).astype(np.float64)
    n = len(confidence)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[-1] = 1.0 + 1e-9
    ece = 0.0
    reliability = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        count = int(np.sum(mask))
        if count == 0:
            reliability.append({"bin_lo": lo, "bin_hi": hi, "mean_conf": np.nan, "mean_acc": np.nan, "count": 0})
            continue
        mean_conf = float(np.mean(confidence[mask]))
        mean_acc = float(np.mean(correct[mask]))
        ece += (count / n) * abs(mean_acc - mean_conf)
        reliability.append({"bin_lo": lo, "bin_hi": hi, "mean_conf": mean_conf, "mean_acc": mean_acc, "count": count})
    return float(ece), reliability


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM/whitespace from column names so lookup is robust."""
    new_names = {c: c.strip().strip("\ufeff").strip() for c in df.columns}
    if new_names != dict(zip(df.columns, df.columns)):
        df = df.rename(columns=new_names)
    return df


def _get_col(df: pd.DataFrame, *candidates: str):
    """Return first column that exists (exact match)."""
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _ensure_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive task1_prob and task2_prob_* from logits if probability columns are missing."""
    df = df.copy()
    # Prefer exact names, then try case-insensitive
    col_lower = {c.lower(): c for c in df.columns}
    def _has(name: str) -> bool:
        return name in df.columns or name.lower() in col_lower
    def _get(name: str):
        if name in df.columns:
            return df[name].values
        if name.lower() in col_lower:
            return df[col_lower[name.lower()]].values
        return None

    task1_logit = _get("task1_logit")
    task1_prob = _get("task1_prob")
    if task1_prob is None and task1_logit is not None:
        logit = np.asarray(task1_logit, dtype=np.float64)
        df["task1_prob"] = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
    elif task1_prob is not None and "task1_prob" not in df.columns:
        df["task1_prob"] = task1_prob
    elif _get("task1_pred") is not None:
        # Last resort: no logits/probs, use hard predictions as pseudo-probs (calibration will be degenerate)
        warnings.warn(
            "task1_prob/task1_logit missing; using task1_pred as pseudo-probs. Brier/ECE will be degenerate.",
            UserWarning,
            stacklevel=2,
        )
        pred = np.asarray(_get("task1_pred"), dtype=np.float64)
        df["task1_prob"] = np.clip(pred, 1e-6, 1 - 1e-6)

    prob_cols = [c for c in df.columns if c.lower().startswith("task2_prob_") and c[-1].isdigit()]
    logit_cols = [c for c in df.columns if c.lower().startswith("task2_logit_") and c[-1].isdigit()]
    if len(prob_cols) < 3 and len(logit_cols) >= 1:
        logit_cols.sort(key=lambda x: int(x.split("_")[-1]))
        logits = np.column_stack([
            df[logit_cols[k]].values if k < len(logit_cols) else np.zeros(len(df))
            for k in range(3)
        ]).astype(np.float64)
        if logits.shape[1] < 3:
            logits = np.pad(logits, ((0, 0), (0, 3 - logits.shape[1])), constant_values=0)
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(np.clip(shift, -500, 500))
        probs = exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-12, None)
        for k in range(3):
            df[f"task2_prob_{k}"] = probs[:, k]
    # Fallback: only task2_pred (class 0/1/2) -> one-hot pseudo-probs
    if not all(f"task2_prob_{k}" in df.columns for k in range(3)):
        task2_pred = _get("task2_pred")
        if task2_pred is not None:
            warnings.warn(
                "task2_prob_*/task2_logit_* missing; using task2_pred as one-hot pseudo-probs. "
                "Brier/ECE will be degenerate.",
                UserWarning,
                stacklevel=2,
            )
            pred = np.asarray(task2_pred, dtype=np.int64).ravel()
            pred = np.clip(pred, 0, 2)
            for k in range(3):
                p = (pred == k).astype(np.float64)
                df[f"task2_prob_{k}"] = np.clip(p, 1e-6, 1 - 1e-6)
    return df


def run_calibration(df: pd.DataFrame, n_bins: int) -> Dict:
    """Compute Task1 and Task2 Brier, ECE, and reliability data."""
    task1_prob = df["task1_prob"].values
    task1_true = df["task1_true"].values
    task2_true = df["task2_true"].values

    # Task2 probs: columns task2_prob_0, task2_prob_1, task2_prob_2
    prob_cols = [c for c in df.columns if c.startswith("task2_prob_") and c[-1].isdigit()]
    prob_cols.sort(key=lambda x: int(x.split("_")[-1]))
    if not prob_cols:
        task2_prob = np.zeros((len(df), 3), dtype=np.float64)
        for k in range(3):
            if f"task2_prob_{k}" in df.columns:
                task2_prob[:, k] = df[f"task2_prob_{k}"].values
    else:
        task2_prob = df[prob_cols].values.astype(np.float64)

    brier1 = brier_score_binary(task1_prob, task1_true)
    ece1, rel1 = ece_binary(task1_prob, task1_true, n_bins=n_bins)

    brier2 = brier_score_multiclass(task2_prob, task2_true)
    ece2, rel2 = ece_multiclass(task2_prob, task2_true, n_bins=n_bins)

    return {
        "task1": {
            "brier_score": brier1,
            "ece": ece1,
            "n_bins": n_bins,
            "reliability_bins": rel1,
        },
        "task2": {
            "brier_score": brier2,
            "ece": ece2,
            "n_bins": n_bins,
            "reliability_bins": rel2,
        },
        "n_samples": int(len(df)),
    }


def _find_project_dir() -> Optional[Path]:
    """Try common locations for project root (contains train_val_test_splits_v2.json)."""
    candidates = []
    try:
        script_dir = Path(__file__).resolve().parent
        candidates.extend([
            script_dir.parent / "facial_pain_project_v2",
            script_dir / "facial_pain_project_v2",
            script_dir,
            script_dir.parent,
            script_dir.parent.parent,
        ])
    except Exception:
        pass
    try:
        cwd = Path.cwd()
        candidates.extend([
            cwd / "facial_pain_project_v2",
            cwd,
            cwd / "VIDEOS FACIAL BOVINE",
        ])
    except Exception:
        pass
    for base in [Path("/content/drive/MyDrive"), Path.home()]:
        if base.exists():
            candidates.extend([
                base / "facial_pain_project_v2",
                base,
                base / "VIDEOS FACIAL BOVINE",
                base / "Ucaps_raw_videos",
            ])
    seen = set()
    for d in candidates:
        try:
            d = d.resolve()
            if d in seen or not d.exists():
                continue
            seen.add(d)
            if (d / "train_val_test_splits_v2.json").exists():
                return d
        except Exception:
            continue
    return None


def main() -> None:
    args = _parse_args()

    if args.predictions_csv:
        csv_path = Path(args.predictions_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    else:
        if not args.run_tag:
            raise ValueError("Provide either --predictions_csv or --run_tag (and optionally --ckpt_kind).")
        project_dir = Path(args.project_dir) if args.project_dir else _find_project_dir()
        if project_dir is None or not (project_dir / "train_val_test_splits_v2.json").exists():
            raise FileNotFoundError(
                "Cannot find project_dir with train_val_test_splits_v2.json.\n"
                "Either pass the directory that contains it:\n"
                "  --project_dir /content/drive/MyDrive/facial_pain_project_v2\n"
                "or pass the predictions CSV directly:\n"
                "  --predictions_csv /content/drive/MyDrive/.../results_v2.9/v2.9_20260221_223056/test_eval_v2.9_task2_animals_14-17_predictions.csv"
            )
        splits = json.loads((project_dir / "train_val_test_splits_v2.json").read_text())
        test_animals = splits.get("test_animals", [14, 17])
        results_dir = Path(args.results_dir) if args.results_dir else None
        csv_path = _resolve_predictions_path(
            args.run_tag, args.ckpt_kind, test_animals, results_dir, project_dir
        )

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = _ensure_prob_columns(df)
    if "task1_prob" not in df.columns:
        cols = list(df.columns)
        raise ValueError(
            f"Missing task1_prob, task1_logit, and task1_pred in {csv_path}. "
            "CSV must contain task1_prob, task1_logit, or task1_pred. "
            f"Columns in file: {cols}"
        )

    out = run_calibration(df, n_bins=args.n_bins)
    out["predictions_csv"] = str(csv_path)
    out["ckpt_kind"] = args.ckpt_kind

    print("=" * 60)
    print("Calibration (v2.9 held-out test)")
    print(f"Predictions: {csv_path}  |  N = {out['n_samples']}  |  bins = {args.n_bins}")
    print("=" * 60)
    print(f"Task1  Brier: {out['task1']['brier_score']:.4f}   ECE: {out['task1']['ece']:.4f}")
    print(f"Task2  Brier: {out['task2']['brier_score']:.4f}   ECE: {out['task2']['ece']:.4f}")
    print("=" * 60)

    # Auto-write outputs by default (Colab-friendly).
    stem = csv_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]

    out_json_path = Path(args.out_json) if args.out_json else (csv_path.parent / f"{stem}_calibration.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {out_json_path}")

    out_csv_path = (
        Path(args.out_reliability_csv)
        if args.out_reliability_csv
        else (csv_path.parent / f"{stem}_reliability_bins.csv")
    )
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in ["task1", "task2"]:
        for b in out[t]["reliability_bins"]:
            rows.append({"task": t, **b})
    pd.DataFrame(rows).to_csv(out_csv_path, index=False)
    print(f"Wrote: {out_csv_path}")


if __name__ == "__main__":
    main()
