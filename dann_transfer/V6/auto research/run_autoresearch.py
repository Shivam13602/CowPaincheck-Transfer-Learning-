#!/usr/bin/env python3
"""V6 autoresearch loop runner (command generator + optional execution)."""
from __future__ import annotations

import argparse
import json
import posixpath
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SPACE = Path(__file__).resolve().parent
SEARCH_SPACE = SPACE / "search_space.json"
V3_CODE = REPO_ROOT / "dann_transfer" / "V3" / "training_code"


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _quote(x: object) -> str:
    return shlex.quote(str(x))


def _posix_join(a: str, b: str) -> str:
    return posixpath.join(a.rstrip("/"), b)


def _to_cmd(runner: str, defaults: dict, trial: dict) -> list[str]:
    if runner == "weak_label_adapt_v3":
        py = V3_CODE / "weak_label_adapt_v3.py"
        cmd = [
            "python",
            str(py),
            "--manifest-csv",
            str(defaults["manifest_csv"]),
            "--sequence-root",
            str(defaults["sequence_root"]),
            "--dataset-version",
            str(defaults["dataset_version"]),
            "--split-json",
            str(defaults["split_json"]),
            "--checkpoint-dir",
            str(defaults["checkpoint_dir"]),
            "--ckpt-kind",
            "task1",
            "--init-fold",
            "0",
            "--out-dir",
            _posix_join(defaults["out_root"], trial["id"]),
            "--threshold-min-specificity",
            "0.5",
            "--select-metric",
            "v3_composite",
        ]
    elif runner == "dann_adapt_v3":
        py = V3_CODE / "dann_adapt_v3.py"
        cmd = [
            "python",
            str(py),
            "--manifest-csv",
            str(defaults["manifest_csv"]),
            "--sequence-root",
            str(defaults["sequence_root"]),
            "--dataset-version",
            str(defaults["dataset_version"]),
            "--split-json",
            str(defaults["split_json"]),
            "--source-project-dir",
            str(defaults["source_project_dir"]),
            "--source-sequence-dir",
            str(defaults["source_sequence_dir"]),
            "--checkpoint-dir",
            str(defaults["checkpoint_dir"]),
            "--ckpt-kind",
            "task1",
            "--init-fold",
            "0",
            "--out-dir",
            _posix_join(defaults["out_root"], trial["id"]),
            "--threshold-min-specificity",
            "0.5",
            "--select-metric",
            "v3_composite",
            "--source-task1-retention-floor",
            "0.55",
            "--source-task1-retention-margin",
            "0.03",
            "--source-task1-sanity-floor",
            "0.70",
            "--target-cow-balanced-sampler",
        ]
    else:
        raise ValueError(f"Unknown runner: {runner}")

    for k, v in trial.get("args", {}).items():
        f = _flag(k)
        if isinstance(v, bool):
            if v:
                cmd.append(f)
        else:
            cmd.extend([f, str(v)])
    return cmd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--search-space", type=Path, default=SEARCH_SPACE)
    p.add_argument("--stage", type=str, default=None, help="Filter by stage, e.g. A/B/C")
    p.add_argument("--trial-id", type=str, default=None, help="Run a single trial ID")
    p.add_argument("--execute", action="store_true", help="Actually run commands (default prints only).")
    p.add_argument("--dry-run", action="store_true", help="Explicit no-exec mode (default).")
    args = p.parse_args()

    spec = json.loads(args.search_space.read_text(encoding="utf-8"))
    defaults = spec["defaults"]
    trials = spec["trials"]

    if args.stage:
        trials = [t for t in trials if t.get("stage") == args.stage]
    if args.trial_id:
        trials = [t for t in trials if t["id"] == args.trial_id]

    if not trials:
        print("No trials matched filters.")
        return 1

    run = args.execute and not args.dry_run
    for t in trials:
        cmd = _to_cmd(t["runner"], defaults, t)
        cmd_s = " ".join(_quote(x) for x in cmd)
        print(f"\n[{t['id']}]")
        print(cmd_s)
        if run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(f"\nTrials listed: {len(trials)} | execute={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
