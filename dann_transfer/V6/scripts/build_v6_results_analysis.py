#!/usr/bin/env python3
"""Build v6_results_analysis.json from downloaded Vast autoresearch runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "vast_auto"
OUT_JSON = ROOT / "results" / "v6_results_analysis.json"

STAGE_A = {
    "A_s3_focal_g1p5_cb": {"loss": "focal", "focal_gamma": 1.5, "class_balanced": True},
    "A_s3_focal_g2p5_cb": {"loss": "focal", "focal_gamma": 2.5, "class_balanced": True},
    "A_s3_gce_q0p6_cb": {"loss": "gce", "gce_q": 0.6, "class_balanced": True},
    "A_s3_gce_q0p8_cb": {"loss": "gce", "gce_q": 0.8, "class_balanced": True},
}
STAGE_B = {
    "B_s4_dann_dw0p15": {"alignment": "dann", "domain_weight": 0.15},
    "B_s4_dann_dw0p20": {"alignment": "dann", "domain_weight": 0.20},
    "B_s4_coral_w0p02": {"alignment": "coral", "coral_weight": 0.02},
}


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
                "pred": int(r.get("pred", 0) or 0),
                "correct": bool(r.get("correct", False)),
            }
        )
    return rows


def analyze_s3(name: str, d: Path, meta: dict) -> dict:
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
        "stage": "A",
        "condition": name,
        **meta,
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


def analyze_s4(name: str, d: Path, meta: dict) -> dict:
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

    return {
        "stage": "B",
        "condition": name,
        **meta,
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
    for name, meta in STAGE_A.items():
        d = RES / name
        if (d / "weak_label_cv_summary.json").is_file():
            conditions.append(analyze_s3(name, d, meta))
    for name, meta in STAGE_B.items():
        d = RES / name
        if (d / "dann_summary.json").is_file():
            conditions.append(analyze_s4(name, d, meta))

    conditions.sort(key=lambda c: (c["sequence"]["auc"] or 0.0, c["sequence"]["f1"] or 0.0), reverse=True)

    payload = {
        "experiment": "V6 autoresearch Vast (549-seq, 8-cow test)",
        "protocol": {
            "dataset": "thesis_stride8 549 sequences",
            "test_cows": ["363", "370", "378", "403", "404", "408", "433", "436"],
            "split_json": "V5/splits/v5_split.json",
            "checkpoint": "v2.9_20260502_181533",
            "platform": "VastAI A100-40GB",
            "stage_a": "weak_label_adapt_v3: focal/GCE + class_balanced, freeze_cnn, 80 ep, lr=1e-4",
            "stage_b": "dann_adapt_v3: DANN/CORAL, no target weak BCE, freeze_cnn, 80 ep, lr=1e-5",
            "v5_baseline_best_seq_auc": 0.593,
            "v5_baseline_best_condition": "S4_coral_w_0.10 / S4_dann_dw_0.25",
        },
        "conditions": conditions,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({len(conditions)} conditions)")


if __name__ == "__main__":
    main()
