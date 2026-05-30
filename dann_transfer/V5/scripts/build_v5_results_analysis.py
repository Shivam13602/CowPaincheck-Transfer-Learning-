#!/usr/bin/env python3
"""Build v5_results_analysis.json from downloaded 549_interim S3 + S4 runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "549_interim"
OUT_JSON = ROOT / "results" / "v5_results_analysis.json"


def metrics_block(m: dict) -> dict:
    return {
        "n": m.get("n"),
        "auc": m.get("auc"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "balanced_accuracy_opt": m.get("balanced_accuracy_opt"),
        "f1": m.get("f1"),
        "f1_opt": m.get("f1_opt"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "threshold": m.get("threshold"),
        "tn": m.get("tn"),
        "fp": m.get("fp"),
        "fn": m.get("fn"),
        "tp": m.get("tp"),
    }


def _disease_breakdown(pred_df: pd.DataFrame, threshold: float) -> list[dict]:
    rows = []
    if pred_df.empty:
        return rows
    grp_col = None
    for c in ("health_condition", "video_health_status", "cow_health_status"):
        if c in pred_df.columns:
            grp_col = c
            break
    if not grp_col:
        return rows
    for key, g in pred_df.groupby(grp_col):
        rows.append(
            {
                "group": str(key),
                "n_seq": len(g),
                "mean_prob": float(g["pain_prob"].mean()),
                "frac_pred_positive": float((g["pain_prob"] >= threshold).mean()),
                "label_positive_rate": float(g["target"].mean()) if "target" in g.columns else None,
            }
        )
    return rows


def _cow_rows(cow_df: pd.DataFrame) -> list[dict]:
    rows = []
    if cow_df.empty or "cow_id" not in cow_df.columns:
        return rows
    for _, r in cow_df.iterrows():
        prob = r.get("pain_prob", r.get("pain_prob_mean", r.get("mean_prob", 0)))
        rows.append(
            {
                "cow_id": str(r.get("cow_id", "")).replace(".0", ""),
                "target": int(r.get("target", r.get("cow_target", 0)) or 0),
                "n_seq": int(r.get("n_sequences", r.get("n_seq", 0)) or 0),
                "pain_prob_mean": float(prob or 0),
                "pred": int(r.get("positives", r.get("pred", 0)) or 0) > 0 if "positives" in r else int(r.get("pred", 0) or 0),
                "correct": bool(r.get("correct", False)),
            }
        )
    return rows


def analyze_s3(name: str, d: Path) -> dict:
    s = json.loads((d / "weak_label_cv_summary.json").read_text(encoding="utf-8"))
    ft = s["final_test"]
    seq = ft["sequence_metrics"]
    cow = ft["cow_metrics"]
    vid = ft.get("video_metrics") or {}
    pred_df = pd.read_csv(d / "weak_label_cv_test_predictions.csv")
    cow_df = pd.read_csv(d / "weak_label_cv_test_cow_aggregates.csv")
    fold_df = pd.read_csv(d / "weak_label_cv_fold_summary.csv")
    thr = ft.get("threshold_from_pooled_validation_specificity_constrained", seq.get("threshold", 0.5))
    boot = (ft.get("diagnostics") or {}).get("cow_level_bootstrap_final_test_raw_prob") or {}
    boot_auc = boot.get("auc") or {}

    return {
        "stage": "S3",
        "condition": name,
        "loss": name.replace("S3_weak_", ""),
        "alignment": None,
        "checkpoint": (s.get("run") or {}).get("checkpoint_dir"),
        "threshold": thr,
        "sequence": metrics_block(seq),
        "video": metrics_block(vid) if vid else None,
        "cow": metrics_block(cow),
        "calibrated_sequence": metrics_block(ft.get("calibrated_sequence_metrics") or {}),
        "calibrated_cow": metrics_block(ft.get("calibrated_cow_metrics") or {}),
        "cow_bootstrap_auc_ci": {
            "median": boot_auc.get("median"),
            "ci95_low": boot_auc.get("ci95_low"),
            "ci95_high": boot_auc.get("ci95_high"),
        },
        "fold_best_val_auc": fold_df["val_auc"].tolist() if "val_auc" in fold_df.columns else [],
        "test_cows": _cow_rows(cow_df),
        "disease_breakdown": _disease_breakdown(pred_df, float(thr or 0.5)),
        "n_test_sequences": int(seq.get("n", 0) or 0),
    }


def analyze_s4(name: str, d: Path) -> dict:
    s = json.loads((d / "dann_summary.json").read_text(encoding="utf-8"))
    ft = s["final_test"]
    seq = ft["sequence_metrics"]
    cow = ft["cow_metrics"]
    vid = ft.get("video_metrics") or {}
    pred_df = pd.read_csv(d / "dann_test_predictions.csv")
    cow_df = pd.read_csv(d / "dann_test_cow_aggregates.csv")
    fold_df = pd.read_csv(d / "dann_fold_summary.csv")
    thr = ft.get("threshold_from_pooled_validation_specificity_constrained", seq.get("threshold", 0.5))
    boot = (ft.get("diagnostics") or {}).get("cow_level_bootstrap_final_test_raw_prob") or {}
    boot_auc = boot.get("auc") or {}
    run = s.get("run") or {}

    align = "dann" if "dann_dw" in name else "coral"
    weight = name.split("_")[-1]

    return {
        "stage": "S4",
        "condition": name,
        "loss": None,
        "alignment": align,
        "alignment_weight": weight,
        "checkpoint": run.get("checkpoint_dir"),
        "domain_weight": run.get("domain_weight"),
        "coral_weight": run.get("coral_weight"),
        "threshold": thr,
        "sequence": metrics_block(seq),
        "video": metrics_block(vid),
        "cow": metrics_block(cow),
        "calibrated_sequence": metrics_block(ft.get("calibrated_sequence_metrics") or {}),
        "calibrated_cow": metrics_block(ft.get("calibrated_cow_metrics") or {}),
        "cow_bootstrap_auc_ci": {
            "median": boot_auc.get("median"),
            "ci95_low": boot_auc.get("ci95_low"),
            "ci95_high": boot_auc.get("ci95_high"),
        },
        "fold_best_val_auc": fold_df["val_auc"].tolist() if "val_auc" in fold_df.columns else [],
        "test_cows": _cow_rows(cow_df),
        "disease_breakdown": _disease_breakdown(pred_df, float(thr or 0.5)),
        "n_test_sequences": int(seq.get("n", 0) or 0),
    }


def main() -> None:
    conditions = []
    s4_local = []
    for sub in sorted(RES.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "weak_label_cv_summary.json").is_file():
            conditions.append(analyze_s3(sub.name, sub))
        elif (sub / "dann_summary.json").is_file():
            conditions.append(analyze_s4(sub.name, sub))
            s4_local.append(sub.name)

    expected_s4 = [
        "S4_dann_dw_0.10",
        "S4_dann_dw_0.25",
        "S4_dann_dw_0.50",
        "S4_coral_w_0.01",
        "S4_coral_w_0.05",
        "S4_coral_w_0.10",
    ]
    missing_s4 = [x for x in expected_s4 if x not in s4_local]

    payload = {
        "experiment": "V5 interim 549-seq",
        "protocol": {
            "dataset": "thesis_stride8 549 sequences",
            "test_cows": ["363", "370", "378", "403", "404", "408", "433", "436"],
            "test_balance": "4 Healthy + 4 Unhealthy (cow_health_status)",
            "cv": "5 folds x 4 val cows, fold ensemble on test",
            "checkpoint": "v2.9_20260502_181533",
            "s4_downloaded": s4_local,
            "s4_missing_local": missing_s4,
        },
        "conditions": conditions,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({len(conditions)} conditions, S4 local: {len(s4_local)}/6)")


if __name__ == "__main__":
    main()
