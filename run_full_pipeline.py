"""Launch a clean, non-destructive main experiment run in a separate workspace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ["adult", "bank_marketing", "credit_g", "ccdefault", "electricity"]


def execute(script: str, *args: str, shards: int = 0, cwd: Path) -> None:
    commands = []
    if shards:
        commands = [
            [sys.executable, f"code/{script}", *args, "--shard", f"{i}/{shards}"]
            for i in range(shards)
        ]
    else:
        commands = [[sys.executable, f"code/{script}", *args]]
    processes = []
    for command in commands:
        print("+", " ".join(command), flush=True)
        env = os.environ.copy()
        env["TIME_BUDGET"] = "1000000"
        processes.append(subprocess.Popen(command, cwd=cwd, env=env))
    for process in processes:
        if process.wait() != 0:
            raise SystemExit(f"stage failed: {script}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("full_run"))
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required acknowledgement because a full run is computationally expensive",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("add --execute after reviewing the workspace and compute cost")
    workspace = args.workspace.resolve()
    if workspace == ROOT or ROOT in workspace.parents and workspace.name != "full_run":
        raise SystemExit("choose a dedicated run workspace, not the repository root")
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"workspace must be absent or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "code", workspace / "code", dirs_exist_ok=True)

    for dataset in DATASETS:
        execute("s0_fetch.py", dataset, cwd=workspace)
    execute("s1_train.py", shards=1, cwd=workspace)
    execute("s2_base_explain.py", shards=args.shards, cwd=workspace)
    execute("s3_perturb.py", shards=args.shards, cwd=workspace)
    execute("s4_uncertainty.py", shards=1, cwd=workspace)
    execute("s4b_reliance.py", shards=args.shards, cwd=workspace)
    execute("s5_aggregate.py", "all", cwd=workspace)
    execute("s6_figures.py", cwd=workspace)
    print(f"Main pipeline complete: {workspace}")


if __name__ == "__main__":
    main()

