#!/usr/bin/env python3
"""Generate the frozen V5 cow-held-out split: an 8-cow balanced test set plus a
class-balanced cow-held-out K-fold where every CV cow is validated exactly once.

Design (see split_strategy.md):
  - 8 fixed test cows (4 Healthy + 4 Unhealthy), never in train/val. The legacy
    four (363, 403, 404, 408) are kept inside the eight for V1-V4 comparability.
  - Remaining cows form the train pool. Class counts are balanced by taking the
    same number of Healthy and Unhealthy cows into the CV rotation; surplus cows
    (and any force-routed cows such as 349) go to a documented train-only pool
    that appears in every fold's training set but is never validated.
  - Folds are equal sized: each has per_class_per_fold Healthy + per_class_per_fold
    Unhealthy cows. Each CV cow is validated exactly once.
  - A per-cow sequence cap is recorded for documentation; the trainer enforces
    imbalance control at runtime via --cow-balanced-sampler.

This script has no torch / numpy dependency. It reads the dataset manifest and
emits a split JSON + an audit CSV, and prints the exact V3-trainer flags to use.

Example:
  python make_v5_splits.py \
      --manifest ../../datasets/thesis_stride8_qa/output/completed_manifest.csv \
      --out splits/v5_split.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

# Weak-proxy label mapping (matches the V3 trainers).
HEALTHY = "Healthy"
UNHEALTHY = "Unhealthy"

DEFAULT_TEST_COWS = ["363", "403", "404", "408", "370", "436", "433", "378"]
# Cows forced into the train-only pool (never validated). Cow 349 is the
# 87-sequence sudden-fall outlier that inflated inner validation in V4.
DEFAULT_TRAIN_ONLY = ["349"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="completed_manifest.csv for the dataset version.")
    p.add_argument("--out", type=Path, default=Path("splits/v5_split.json"), help="Output split JSON path.")
    p.add_argument("--audit-csv", type=Path, default=None, help="Audit CSV path (default: alongside --out).")
    p.add_argument("--label-column", type=str, default="video_health_status",
                   help="Column used as the weak proxy label (default video_health_status).")
    p.add_argument("--cow-column", type=str, default="cow_id")
    p.add_argument("--health-for-balance", type=str, default="cow_health_status",
                   help="Column used to assign each cow a single class for balancing folds.")
    p.add_argument("--test-cow-ids", type=str, default=",".join(DEFAULT_TEST_COWS),
                   help="Comma-separated 8 test cow IDs (4 Healthy + 4 Unhealthy).")
    p.add_argument("--train-only-cows", type=str, default=",".join(DEFAULT_TRAIN_ONLY),
                   help="Comma-separated cow IDs forced into the train-only pool (never validated).")
    p.add_argument("--val-cows-per-fold", type=int, default=4,
                   help="Cows per validation fold; must be even (half Healthy, half Unhealthy).")
    p.add_argument("--max-seqs-per-cow", type=int, default=0,
                   help="Per-cow sequence cap to document; 0 = use median train-pool count.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def majority_class(values: list[str]) -> str:
    """Majority health class for a cow; ties resolve to Unhealthy (conservative)."""
    n_unhealthy = sum(1 for v in values if v.strip().lower().startswith("unhealthy"))
    n_healthy = sum(1 for v in values if v.strip().lower().startswith("healthy"))
    return UNHEALTHY if n_unhealthy >= n_healthy and (n_unhealthy + n_healthy) > 0 else HEALTHY


def load_cows(manifest: Path, cow_col: str, health_col: str) -> dict[str, dict]:
    seqs_by_cow: dict[str, int] = defaultdict(int)
    health_by_cow: dict[str, list[str]] = defaultdict(list)
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if cow_col not in reader.fieldnames:
            raise SystemExit(f"Manifest missing cow column '{cow_col}'. Columns: {reader.fieldnames}")
        if health_col not in reader.fieldnames:
            raise SystemExit(f"Manifest missing health column '{health_col}'. Columns: {reader.fieldnames}")
        for row in reader:
            cow = str(row[cow_col]).strip()
            if not cow:
                continue
            seqs_by_cow[cow] += 1
            health_by_cow[cow].append(str(row[health_col]))
    cows: dict[str, dict] = {}
    for cow, n in seqs_by_cow.items():
        cls = majority_class(health_by_cow[cow])
        cows[cow] = {"health": cls, "n_seqs": n}
    return cows


def build_folds(cv_healthy: list[str], cv_unhealthy: list[str], per_class_per_fold: int) -> list[dict]:
    n_folds = len(cv_healthy) // per_class_per_fold
    folds: list[dict] = []
    for k in range(n_folds):
        h = cv_healthy[k * per_class_per_fold:(k + 1) * per_class_per_fold]
        u = cv_unhealthy[k * per_class_per_fold:(k + 1) * per_class_per_fold]
        folds.append({"fold": k, "val_cows": sorted(h + u, key=lambda c: int(c) if c.isdigit() else c),
                      "val_healthy": h, "val_unhealthy": u})
    return folds


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.val_cows_per_fold % 2 != 0:
        raise SystemExit("--val-cows-per-fold must be even (half Healthy, half Unhealthy).")
    per_class_per_fold = args.val_cows_per_fold // 2

    cows = load_cows(args.manifest, args.cow_column, args.health_for_balance)
    test_cows = [c.strip() for c in args.test_cow_ids.split(",") if c.strip()]
    forced_train_only = [c.strip() for c in args.train_only_cows.split(",") if c.strip()]

    missing = [c for c in test_cows + forced_train_only if c not in cows]
    if missing:
        raise SystemExit(f"Requested cows not present in manifest: {missing}")

    test_h = [c for c in test_cows if cows[c]["health"] == HEALTHY]
    test_u = [c for c in test_cows if cows[c]["health"] == UNHEALTHY]
    if len(test_h) != len(test_u):
        print(f"WARNING: test set not class-balanced: {len(test_h)} Healthy vs {len(test_u)} Unhealthy")

    # Train pool = everything not in test.
    pool = [c for c in cows if c not in test_cows]
    cv_eligible = [c for c in pool if c not in forced_train_only]

    healthy = sorted([c for c in cv_eligible if cows[c]["health"] == HEALTHY])
    unhealthy = sorted([c for c in cv_eligible if cows[c]["health"] == UNHEALTHY])
    rng.shuffle(healthy)
    rng.shuffle(unhealthy)

    balanced = min(len(healthy), len(unhealthy))
    n_folds = balanced // per_class_per_fold
    cv_per_class = n_folds * per_class_per_fold
    if n_folds == 0:
        raise SystemExit("Not enough cows per class to form even one balanced fold; lower --val-cows-per-fold.")

    cv_healthy = healthy[:cv_per_class]
    cv_unhealthy = unhealthy[:cv_per_class]
    surplus = healthy[cv_per_class:] + unhealthy[cv_per_class:]
    train_only = sorted(set(forced_train_only) | set(surplus), key=lambda c: int(c) if c.isdigit() else c)

    folds = build_folds(cv_healthy, cv_unhealthy, per_class_per_fold)

    # Per-cow sequence cap (documentation only; enforced via --cow-balanced-sampler at train time).
    pool_counts = [cows[c]["n_seqs"] for c in pool]
    cap = args.max_seqs_per_cow or int(statistics.median(pool_counts))
    capped = sorted([c for c in pool if cows[c]["n_seqs"] > cap], key=lambda c: int(c) if c.isdigit() else c)

    # Role assignment for audit.
    role: dict[str, str] = {}
    for c in test_cows:
        role[c] = "test"
    for f in folds:
        for c in f["val_cows"]:
            role[c] = f"val_fold_{f['fold']}"
    for c in train_only:
        role[c] = "train_only"

    split = {
        "dataset_version": "thesis_stride8_qa_v5",
        "manifest": str(args.manifest),
        "seed": args.seed,
        "label_column": args.label_column,
        "balance_column": args.health_for_balance,
        "test_cows": sorted(test_cows, key=lambda c: int(c) if c.isdigit() else c),
        "test_class_balance": {"healthy": sorted(test_h), "unhealthy": sorted(test_u)},
        "val_cows_per_fold": args.val_cows_per_fold,
        "per_class_per_fold": per_class_per_fold,
        "n_folds": n_folds,
        "folds": folds,
        "train_only_cows": train_only,
        "per_cow_seq_cap": cap,
        "capped_cows": capped,
        "cow_table": {c: {"health": cows[c]["health"], "n_seqs": cows[c]["n_seqs"],
                          "role": role.get(c, "unused")} for c in sorted(cows, key=lambda x: int(x) if x.isdigit() else x)},
        "trainer_args": {
            "--test-cow-ids": ",".join(sorted(test_cows, key=lambda c: int(c) if c.isdigit() else c)),
            "--val-cows-per-fold": args.val_cows_per_fold,
            "--require-val-both-classes": True,
            "--cow-balanced-sampler": True,
        },
        "notes": (
            "Each CV cow is validated exactly once across the K folds (standard K-fold property). "
            "Folds are equal sized and class-balanced. Surplus + forced cows are train-only (never validated). "
            "The V3 trainers build folds from --test-cow-ids + --val-cows-per-fold (seed 42); to enforce this "
            "exact partition deterministically, add a --split-json loader to weak_label_adapt_v3.py / "
            "dann_adapt_v3.py (see split_strategy.md, section 'Trainer integration')."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2), encoding="utf-8")

    audit_csv = args.audit_csv or args.out.with_suffix(".audit.csv")
    with audit_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cow_id", "health", "n_seqs", "role", "exceeds_cap"])
        for c in split["cow_table"]:
            info = split["cow_table"][c]
            w.writerow([c, info["health"], info["n_seqs"], info["role"], "yes" if c in capped else "no"])

    # Console summary.
    n_test_seqs = sum(cows[c]["n_seqs"] for c in test_cows)
    n_cv_seqs = sum(cows[c]["n_seqs"] for c in cv_healthy + cv_unhealthy)
    n_to_seqs = sum(cows[c]["n_seqs"] for c in train_only)
    print("== V5 split summary ==")
    print(f"Manifest                : {args.manifest}")
    print(f"Test cows (8)           : {split['test_cows']}  ({len(test_h)}H + {len(test_u)}U, {n_test_seqs} seqs)")
    print(f"CV cows                 : {len(cv_healthy)}H + {len(cv_unhealthy)}U  ({n_cv_seqs} seqs)")
    print(f"Folds                   : {n_folds} x {args.val_cows_per_fold} val cows "
          f"({per_class_per_fold}H + {per_class_per_fold}U per fold)")
    for f in folds:
        print(f"  fold {f['fold']}: val H={f['val_healthy']} U={f['val_unhealthy']}")
    print(f"Train-only cows         : {train_only}  ({n_to_seqs} seqs; forced={forced_train_only})")
    print(f"Per-cow seq cap (doc)   : {cap}  | cows exceeding cap: {capped}")
    print(f"Split JSON              : {args.out}")
    print(f"Audit CSV               : {audit_csv}")
    print()
    print("Trainer flags:")
    print(f"  --test-cow-ids {split['trainer_args']['--test-cow-ids']} \\")
    print(f"  --val-cows-per-fold {args.val_cows_per_fold} --require-val-both-classes --cow-balanced-sampler")


if __name__ == "__main__":
    main()
