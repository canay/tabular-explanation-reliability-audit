"""Recompute manuscript-facing summaries and figures from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
HASH_MANIFEST = ROOT / "provenance" / "input_artifact_sha256.json"


def run(*args: str, capture_to: Path | None = None) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr or result.stdout or f"failed: {command}")
    if capture_to is not None:
        capture_to.parent.mkdir(parents=True, exist_ok=True)
        capture_to.write_text(result.stdout, encoding="utf-8")
    elif result.stdout.strip():
        print(result.stdout.rstrip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> None:
    manifest = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            failures.append((relative, expected, actual))
    if failures:
        detail = "\n".join(f"{p}: expected {e}, got {a}" for p, e, a in failures)
        raise SystemExit(f"Frozen-input verification failed:\n{detail}")
    print(f"Verified {len(manifest['files'])} frozen inputs.")


def regenerate_summaries() -> None:
    run("code/s5_aggregate.py", "summary")
    run("analysis/scripts/variance_decomp.py")
    run("analysis/scripts/variance_decomp_families.py")
    run("analysis/scripts/rule_extras.py")
    run("analysis/scripts/decision_rule_robustness.py")
    run("analysis/scripts/round3_extras.py")
    run("analysis/scripts/manuscript_numbers.py")
    run(
        "analysis/scripts/gen_tables.py",
        capture_to=ROOT / "analysis" / "results" / "gen_tables.txt",
    )


def regenerate_figures() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    run("code/s6_figures.py")
    run(
        "code/plot_q1_hardening.py",
        "--result-dir",
        "results/q1_hardening",
        "--fig-dir",
        "figures",
    )
    run("analysis/scripts/round3_figs.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--summaries", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("choose --verify-inputs, --summaries, --figures, or --all")
    if args.verify_inputs or args.all:
        verify_inputs()
    if args.summaries or args.all:
        regenerate_summaries()
    if args.figures or args.all:
        regenerate_figures()


if __name__ == "__main__":
    main()

