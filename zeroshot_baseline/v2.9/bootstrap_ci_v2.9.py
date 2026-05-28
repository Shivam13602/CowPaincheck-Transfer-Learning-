# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS - v2.9 (held-out test)
#
# Task 1 of 2: Compute 95% bootstrap CIs for held-out metrics using
# sequence-level resampling stratified by animal.
#
# Input: predictions CSV produced by evaluate_test_set_v2.9_cli.py
#   (columns: animal, task1_true, task1_pred, task2_true, task2_pred, ...)
#
# Usage (one of):
#   python bootstrap_ci_v2.9.py --predictions_csv path/to/predictions.csv
#   python bootstrap_ci_v2.9.py --run_tag v2.9_20260221_014705 --ckpt_kind task2
#
# Output: Point estimates and 95% CI (2.5%, 97.5%) for Task1 acc/F1, Task2 acc/F1_weighted.
# If --out_json is not provided, JSON is auto-written next to predictions CSV.
# ============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap 95%% CIs for v2.9 held-out metrics (sequence-level, stratified by animal)."
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
        help="Override results dir, e.g. results_v2.9/<run_tag>. Default: project_dir/results_v2.9/<run_tag>.",
    )
    p.add_argument(
        "--project_dir",
        type=str,
        default=None,
        help="Project dir (for train_val_test_splits_v2.json). Used when resolving --run_tag.",
    )
    p.add_argument("--n_resamples", type=int, default=10_000, help="Number of bootstrap resamples.")
    p.add_argument("--confidence", type=float, default=0.95, help="Confidence level (e.g. 0.95).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap.")
    p.add_argument("--out_json", type=str, default=None, help="Path to write CI results JSON.")
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
    # Standard name from evaluate_test_set_v2.9_cli.py
    animals_str = "-".join(str(a) for a in sorted(test_animals))
    fname = f"test_eval_v2.9_{ckpt_kind}_animals_{animals_str}_predictions.csv"
    path = results_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}. Run evaluate_test_set_v2.9_cli.py first.")
    return path


def _compute_metrics(
    task1_true: np.ndarray,
    task1_pred: np.ndarray,
    task2_true: np.ndarray,
    task2_pred: np.ndarray,
) -> Dict[str, float]:
    n = len(task1_true)
    if n == 0:
        return {
            "task1_accuracy": 0.0,
            "task1_f1": 0.0,
            "task2_accuracy": 0.0,
            "task2_f1_weighted": 0.0,
            "n": 0,
        }
    return {
        "task1_accuracy": float(accuracy_score(task1_true, task1_pred)),
        "task1_f1": float(f1_score(task1_true, task1_pred, zero_division=0)),
        "task2_accuracy": float(accuracy_score(task2_true, task2_pred)),
        "task2_f1_weighted": float(f1_score(task2_true, task2_pred, average="weighted", zero_division=0)),
        "n": n,
    }


def _stratified_resample_by_animal(
    df: pd.DataFrame,
    animal_col: str,
    seed: int,
) -> np.ndarray:
    """Resample row indices so that each resample preserves animal stratification."""
    rng = np.random.default_rng(seed)
    animals = df[animal_col].values
    unique_animals = np.unique(animals)
    indices = np.arange(len(df))
    resampled = []
    for a in unique_animals:
        mask = animals == a
        idx_a = indices[mask]
        n_a = len(idx_a)
        if n_a == 0:
            continue
        chosen = rng.choice(idx_a, size=n_a, replace=True)
        resampled.append(chosen)
    return np.concatenate(resampled, axis=0)


def run_bootstrap(
    df: pd.DataFrame,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Run stratified bootstrap (by animal). Returns point estimates and (low, high) per metric.
    """
    animal_col = "animal"
    if animal_col not in df.columns:
        animal_col = "animal_id" if "animal_id" in df.columns else df.columns[0]
    task1_true = df["task1_true"].values
    task1_pred = df["task1_pred"].values
    task2_true = df["task2_true"].values
    task2_pred = df["task2_pred"].values

    point = _compute_metrics(task1_true, task1_pred, task2_true, task2_pred)

    low_percent = (1 - confidence) / 2 * 100
    high_percent = (1 + confidence) / 2 * 100
    rng = np.random.default_rng(seed)

    boot_task1_acc: List[float] = []
    boot_task1_f1: List[float] = []
    boot_task2_acc: List[float] = []
    boot_task2_f1w: List[float] = []

    for b in range(n_resamples):
        idx = _stratified_resample_by_animal(df, animal_col, seed=rng.integers(0, 2**31))
        t1_t = task1_true[idx]
        t1_p = task1_pred[idx]
        t2_t = task2_true[idx]
        t2_p = task2_pred[idx]
        m = _compute_metrics(t1_t, t1_p, t2_t, t2_p)
        boot_task1_acc.append(m["task1_accuracy"])
        boot_task1_f1.append(m["task1_f1"])
        boot_task2_acc.append(m["task2_accuracy"])
        boot_task2_f1w.append(m["task2_f1_weighted"])

    boot_task1_acc = np.array(boot_task1_acc)
    boot_task1_f1 = np.array(boot_task1_f1)
    boot_task2_acc = np.array(boot_task2_acc)
    boot_task2_f1w = np.array(boot_task2_f1w)

    ci: Dict[str, Tuple[float, float]] = {
        "task1_accuracy": (float(np.percentile(boot_task1_acc, low_percent)), float(np.percentile(boot_task1_acc, high_percent))),
        "task1_f1": (float(np.percentile(boot_task1_f1, low_percent)), float(np.percentile(boot_task1_f1, high_percent))),
        "task2_accuracy": (float(np.percentile(boot_task2_acc, low_percent)), float(np.percentile(boot_task2_acc, high_percent))),
        "task2_f1_weighted": (float(np.percentile(boot_task2_f1w, low_percent)), float(np.percentile(boot_task2_f1w, high_percent))),
    }
    return point, ci


def _find_project_dir() -> Optional[Path]:
    """Try common locations for project root (contains train_val_test_splits_v2.json)."""
    candidates = []
    try:
        script_dir = Path(__file__).resolve().parent
        # Project often lives in facial_pain_project_v2 (Colab Drive)
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
    for col in ["task1_true", "task1_pred", "task2_true", "task2_pred"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {csv_path}.")

    point, ci = run_bootstrap(
        df,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )

    print("=" * 60)
    print("Bootstrap confidence intervals (v2.9 held-out test)")
    print(f"Predictions: {csv_path}")
    print(f"N = {point['n']}  |  resamples = {args.n_resamples}  |  level = {args.confidence:.0%}")
    print("=" * 60)
    print(f"Task1 Accuracy:  {point['task1_accuracy']:.4f}  [{ci['task1_accuracy'][0]:.4f}, {ci['task1_accuracy'][1]:.4f}]")
    print(f"Task1 F1:       {point['task1_f1']:.4f}  [{ci['task1_f1'][0]:.4f}, {ci['task1_f1'][1]:.4f}]")
    print(f"Task2 Accuracy: {point['task2_accuracy']:.4f}  [{ci['task2_accuracy'][0]:.4f}, {ci['task2_accuracy'][1]:.4f}]")
    print(f"Task2 F1 (w):   {point['task2_f1_weighted']:.4f}  [{ci['task2_f1_weighted'][0]:.4f}, {ci['task2_f1_weighted'][1]:.4f}]")
    print("=" * 60)

    out = {
        "point_estimates": point,
        "confidence_intervals": {k: {"low": v[0], "high": v[1]} for k, v in ci.items()},
        "n_resamples": args.n_resamples,
        "confidence": args.confidence,
        "seed": args.seed,
        "predictions_csv": str(csv_path),
    }
    # Auto-write JSON by default (Colab-friendly).
    stem = csv_path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    out_path = Path(args.out_json) if args.out_json else (csv_path.parent / f"{stem}_bootstrap_ci.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
