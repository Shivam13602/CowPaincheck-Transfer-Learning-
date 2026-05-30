#!/usr/bin/env python3
"""Summarize V5 interim weak-label runs (8-cow test, fold ensemble) from downloaded summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pick(d: dict, *keys: str):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def load_condition(path: Path) -> dict | None:
    summary = path / "weak_label_cv_summary.json"
    if not summary.is_file():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    ft = data.get("final_test") or {}
    seq = ft.get("sequence_metrics") or {}
    cow = ft.get("cow_metrics") or {}
    cseq = ft.get("calibrated_sequence_metrics") or {}
    ccow = ft.get("calibrated_cow_metrics") or {}
    boot = (ft.get("diagnostics") or {}).get("cow_level_bootstrap_final_test_raw_prob") or {}
    boot_auc = (boot.get("auc") or {}) if isinstance(boot, dict) else {}
    return {
        "condition": path.name,
        "n_test_seq": seq.get("n"),
        "threshold": ft.get("threshold_from_pooled_validation_specificity_constrained"),
        "seq_auc": seq.get("auc"),
        "seq_bacc": seq.get("balanced_accuracy"),
        "seq_bacc_opt": seq.get("balanced_accuracy_opt"),
        "seq_f1_opt": seq.get("f1_opt"),
        "cow_auc": cow.get("auc"),
        "cow_bacc": cow.get("balanced_accuracy"),
        "cow_bacc_opt": cow.get("balanced_accuracy_opt"),
        "cal_seq_auc": cseq.get("auc"),
        "cal_cow_bacc_opt": ccow.get("balanced_accuracy_opt"),
        "cow_auc_ci_lo": boot_auc.get("ci95_low"),
        "cow_auc_ci_hi": boot_auc.get("ci95_high"),
        "report": str(path / "weak_label_cv_report.md"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "549_interim",
    )
    p.add_argument("--out", type=Path, default=None, help="Write comparison CSV (default: results-dir/comparison.csv)")
    args = p.parse_args()
    root = args.results_dir.resolve()
    if not root.is_dir():
        print(f"Missing results dir: {root}")
        print("Run download_results_549.ps1 first (from Windows, with Duo).")
        return 2

    rows = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            row = load_condition(sub)
            if row:
                rows.append(row)

    if not rows:
        print(f"No weak_label_cv_summary.json under {root}")
        return 2

    cols = [
        "condition",
        "seq_auc",
        "seq_bacc_opt",
        "seq_f1_opt",
        "cow_auc",
        "cow_bacc_opt",
        "cal_seq_auc",
        "cal_cow_bacc_opt",
        "threshold",
        "cow_auc_ci_lo",
        "cow_auc_ci_hi",
        "n_test_seq",
    ]
    print("\n=== V5 interim 8-cow test (549-seq, fold ensemble) ===\n")
    print(" | ".join(f"{c:>14}" for c in cols))
    print("-" * (16 * len(cols)))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):>14}" for c in cols))

    best = max(rows, key=lambda x: float(x["seq_auc"] or -1))
    print(f"\nBest seq AUC (raw): {best['condition']} -> {best['seq_auc']}")
    print(f"Reports: {root}/*/weak_label_cv_report.md")

    if args.out or True:
        out = args.out or (root / "comparison.csv")
        import csv

        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
