"""Generate reproducible figures for repository READMEs from committed metrics files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

V3_BASE = ROOT / "dann_transfer" / "V3" / "rorqual_run_20260515_12326664" / "v3_baseline_10s_250"
V4_ANALYSIS = ROOT / "dann_transfer" / "V4" / "analysis" / "comparison_vs_baseline.csv"
V4_WEAK_COW = ROOT / "dann_transfer" / "V4" / "results" / "weak_focal" / "weak_label_cv_test_cow_aggregates.csv"


def _save(name: str) -> Path:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def load_summary_auc(condition_dir: Path, summary_name: str) -> float | None:
    path = condition_dir / summary_name
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ft = data.get("final_test", {}) or {}
    seq = ft.get("sequence_metrics", {}) or {}
    return seq.get("auc")


def fig_pipeline_overview() -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.2, "Raw farm\nvideo"),
        (2.2, "YOLO face\ncrop"),
        (4.2, "10 s sequences\n(240 frames)"),
        (6.2, "UCAPS v2.9\nCNN-LSTM-Attn"),
        (8.2, "Adaptation\nSSL / weak / DANN"),
        (10.2, "Cow-held-out\neval"),
    ]
    for i, (x, label) in enumerate(steps):
        box = FancyBboxPatch(
            (x, 0.8), 1.6, 1.4, boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.2, edgecolor="#2c5282", facecolor="#ebf8ff",
        )
        ax.add_patch(box)
        ax.text(x + 0.8, 1.5, label, ha="center", va="center", fontsize=9, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(x + 1.75, 1.5), xytext=(x + 1.65, 1.5),
                arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.5),
            )
    ax.text(6, 0.15, "UCAPS beef (source) → Holstein/Jersey dairy (target) · weak proxy labels only",
            ha="center", fontsize=10, color="#4a5568")
    ax.set_title("End-to-end transfer learning pipeline", fontsize=12, fontweight="bold", pad=8)
    return _save("pipeline_overview.png")


def fig_experiment_timeline() -> Path:
    rows = [
        ("Phase 0\nZero-shot", 0.529, "#718096"),
        ("V1\nDANN Vast", 0.48, "#4299e1"),
        ("V2\n14×2 CV", 0.557, "#4299e1"),
        ("V3 weak_focal", 0.476, "#38a169"),
        ("V3 CORAL", 0.577, "#2f855a"),
        ("V4 weak_focal", 0.421, "#d69e2e"),
        ("V4 CORAL", 0.199, "#c05621"),
    ]
    labels, aucs, colors = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, aucs, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="#e53e3e", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_ylabel("Final test sequence AUC (Holstein proxy)")
    ax.set_ylim(0, 0.65)
    ax.set_title("Experiment timeline — sequence-level AUC on fixed 4-cow test set")
    ax.legend(loc="upper right")
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    return _save("experiment_timeline.png")


def fig_v3_condition_matrix() -> Path:
    conditions = [
        ("weak_bce", "weak_label_cv_summary.json"),
        ("weak_gce", "weak_label_cv_summary.json"),
        ("weak_focal", "weak_label_cv_summary.json"),
        ("dann_dw_0.0", "dann_summary.json"),
        ("dann_dw_0.05", "dann_summary.json"),
        ("dann_dw_0.10", "dann_summary.json"),
        ("dann_dw_0.25", "dann_summary.json"),
        ("coral_w_0.10", "dann_summary.json"),
    ]
    names, aucs = [], []
    for cond, fname in conditions:
        auc = load_summary_auc(V3_BASE / cond, fname)
        if auc is not None:
            names.append(cond)
            aucs.append(float(auc))
    order = sorted(range(len(aucs)), key=lambda i: aucs[i], reverse=True)
    names = [names[i] for i in order]
    aucs = [aucs[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2f855a" if n == "coral_w_0.10" else "#4299e1" for n in names]
    bars = ax.barh(names, aucs, color=colors, edgecolor="white")
    ax.axvline(0.5, color="#e53e3e", linestyle="--", linewidth=1)
    ax.set_xlabel("Sequence AUC (final 4-cow test)")
    ax.set_xlim(0.35, 0.62)
    ax.set_title("V3 baseline matrix (baseline_10s_250, job 12326664)")
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    return _save("v3_condition_matrix.png")


def fig_v4_vs_baseline() -> Path:
    seq_rows = []
    with V4_ANALYSIS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["eval_level"] == "sequence":
                seq_rows.append(row)
    labels = [f"{r['condition']}\n({r['dataset'].replace('_', ' ')})" for r in seq_rows]
    auc = [float(r["auc"]) for r in seq_rows]
    bacc = [float(r["balanced_accuracy"]) for r in seq_rows]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w / 2 for i in x], auc, width=w, label="AUC", color="#4299e1")
    ax.bar([i + w / 2 for i in x], bacc, width=w, label="Balanced accuracy", color="#38a169")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color="#e53e3e", linestyle="--", linewidth=1)
    ax.set_ylim(0, 0.85)
    ax.set_title("V4 thesis dataset vs V3 baseline (sequence-level, same test cows)")
    ax.legend()
    return _save("v4_vs_baseline.png")


def fig_v4_cow_probs() -> Path:
    cows, probs, targets = [], [], []
    with V4_WEAK_COW.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cows.append(row["cow_id"])
            probs.append(float(row["pain_prob"]))
            targets.append(int(float(row["target"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e53e3e" if t == 1 else "#38a169" for t in targets]
    bars = ax.bar(cows, probs, color=colors, edgecolor="white")
    ax.axhline(0.4653, color="#2d3748", linestyle="--", linewidth=1, label="Threshold (0.465)")
    ax.set_ylabel("Mean pain probability (weak_focal)")
    ax.set_xlabel("Test cow ID")
    ax.set_ylim(0, 0.65)
    ax.set_title("V4 final test — cow-level predictions (weak_focal)")
    ax.legend(handles=[
        mpatches.Patch(color="#e53e3e", label="Unhealthy proxy"),
        mpatches.Patch(color="#38a169", label="Healthy proxy"),
    ] + [plt.Line2D([0], [0], color="#2d3748", linestyle="--", label="Threshold")])
    for bar, val in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    return _save("v4_cow_probs.png")


def fig_phase0_baselines() -> Path:
    labels = ["Zero-shot", "Freeze-CNN\nweak FT", "Full-backbone\nweak FT", "SSL→weak FT"]
    aucs = [0.529, 0.548, 0.457, 0.447]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, aucs, color="#718096", edgecolor="white")
    ax.axhline(0.5, color="#e53e3e", linestyle="--", linewidth=1)
    ax.set_ylabel("Holstein proxy AUC")
    ax.set_ylim(0.4, 0.6)
    ax.set_title("Phase 0 — pre-DANN baselines (baseline_10s_250)")
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    return _save("phase0_baselines.png")


def fig_thesis_dataset_distribution() -> Path:
    stats_path = ROOT / "datasets" / "thesis_stride8_qa" / "output" / "processing_statistics.json"
    if not stats_path.is_file():
        stats_path = ROOT / "dann_transfer" / "V4" / "dataset" / "processing_statistics.json"
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    completed = data.get("completed_summary", {})
    health = completed.get("cow_health_counts", {"Healthy": 174, "Unhealthy": 375})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].pie(
        [health.get("Healthy", 0), health.get("Unhealthy", 0)],
        labels=["Healthy", "Unhealthy"], autopct="%1.0f%%",
        colors=["#38a169", "#e53e3e"], startangle=90,
    )
    axes[0].set_title("Sequence health labels")
    session = completed.get("session_counts", {})
    if session:
        axes[1].bar(session.keys(), session.values(), color="#4299e1", edgecolor="white")
        axes[1].set_title("Session context")
        axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"thesis_stride8_qa — {data.get('completed_sequences', 549)} QA-pass sequences", fontsize=11, fontweight="bold")
    return _save("thesis_dataset_distribution.png")


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    outputs = [
        fig_pipeline_overview(),
        fig_experiment_timeline(),
        fig_v3_condition_matrix(),
        fig_v4_vs_baseline(),
        fig_v4_cow_probs(),
        fig_phase0_baselines(),
        fig_thesis_dataset_distribution(),
    ]
    for p in outputs:
        print(f"Wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
