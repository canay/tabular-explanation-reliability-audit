"""Generate figures for SCI-f01 Q1 hardening results.

This script reads saved CSV/JSON artifacts only. It must not train models or
rerun explainers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

COL = {
    "real": "#0072B2",
    "random": "#999999",
    "permuted": "#E69F00",
    "constant": "#CC79A7",
    "oracle": "#009E73",
}


def save(fig, fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_control_recovery(result_dir: Path, fig_dir: Path) -> None:
    path = result_dir / "control_recovery_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["variant"] == "spur"].copy()
    if df.empty:
        return
    order = ["random", "permuted", "constant", "real", "oracle"]
    df["source"] = pd.Categorical(df["source"], order, ordered=True)
    df = df.sort_values("source")
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    x = range(len(df))
    ax.bar(x, df["target_top5"], color=[COL.get(s, "#555555") for s in df["source"]])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["source"], rotation=25, ha="right")
    ax.set_ylabel("Known shortcut in top-5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Spurious-feature recovery versus controls")
    save(fig, fig_dir, "fig_q1_control_recovery")


def plot_stability_controls(result_dir: Path, fig_dir: Path) -> None:
    path = result_dir / "cross_seed_control_stability_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["variant"] == "clean"].copy()
    if df.empty:
        return
    order = ["random", "permuted", "constant", "real", "oracle"]
    df["source"] = pd.Categorical(df["source"], order, ordered=True)
    df = df.sort_values("source")
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    x = range(len(df))
    ax.bar(x, df["top5_mean"], color=[COL.get(s, "#555555") for s in df["source"]])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["source"], rotation=25, ha="right")
    ax.set_ylabel("Cross-seed top-5 agreement")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stability metric controls")
    save(fig, fig_dir, "fig_q1_stability_controls")


def plot_lime_budget(result_dir: Path, fig_dir: Path) -> None:
    path = result_dir / "lime_budget_sweep_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["variant"] == "spur"].copy()
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7.0, 3.9))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("budget")
        ax1.plot(sub["budget"], sub["fast_cosine"], marker="o", label=f"{model} cosine")
    ax1.set_xscale("log")
    ax1.set_xlabel("LIME samples")
    ax1.set_ylabel("Mean cosine to matched explainer")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    rt = df.groupby("budget")["seconds_per_instance"].median().reset_index()
    ax2.plot(rt["budget"], rt["seconds_per_instance"], color="#D55E00", marker="s", label="runtime")
    ax2.set_ylabel("Median seconds / instance")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    ax1.set_title("LIME budget quality-runtime trade-off")
    save(fig, fig_dir, "fig_q1_lime_budget_tradeoff")


def plot_synthetic(result_dir: Path, fig_dir: Path) -> None:
    path = result_dir / "synthetic_ground_truth_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["random", "permuted", "constant", "real", "oracle"]
    df["source"] = pd.Categorical(df["source"], order, ordered=True)
    piv = df.pivot_table(index="model", columns="source", values="ap", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    piv[order].plot(kind="bar", ax=ax, color=[COL.get(s, "#555555") for s in order])
    ax.set_ylabel("AUPRC for known feature recovery")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.legend(frameon=False, ncols=3)
    ax.set_title("Controlled ground-truth feature recovery")
    save(fig, fig_dir, "fig_q1_synthetic_ground_truth")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--fig-dir", type=Path, required=True)
    args = p.parse_args()
    plot_control_recovery(args.result_dir, args.fig_dir)
    plot_stability_controls(args.result_dir, args.fig_dir)
    plot_lime_budget(args.result_dir, args.fig_dir)
    plot_synthetic(args.result_dir, args.fig_dir)


if __name__ == "__main__":
    main()
