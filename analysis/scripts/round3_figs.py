"""Figure regeneration from FROZEN artifacts only.
- fig_mcdropout_link: 3 -> 5 dataset panels (2x3 grid, last cell hidden).
- fig_q1_control_recovery: only real/random/oracle (matches caption + tab:hardening), no in-figure title.
- fig_q1_lime_budget_tradeoff: no in-figure title, legend moved clear of the runtime line.
- fig_damage_curves + fig_conf_vs_instability: shared supylabel/supxlabel, constrained layout (no clipping).
Styles copied from code/s6_figures.py and code/plot_q1_hardening.py. Run from project root.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "code")
from common import RESULTS, FIGS, UNC, DATASETS, MODELS_LIST, SIGMAS, METHODS_FOR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300,
})
C = {"linshap": "#0072B2", "treeshap": "#0072B2", "ig": "#0072B2",
     "lime": "#D55E00", "kshap": "#009E73"}
MNAME = {"linshap": "LinearSHAP", "treeshap": "TreeSHAP", "ig": "IG",
         "lime": "LIME", "kshap": "KernelSHAP"}
MODNAME = {"logreg": "Logistic regression", "rf": "Random forest",
           "hgb": "Gradient boosting", "mlp": "MLP"}
DSNAME = {"adult": "Adult", "credit_g": "German Credit", "bank_marketing": "Bank Marketing",
          "ccdefault": "CC default", "electricity": "Electricity"}
DSCOL = {"adult": "#0072B2", "credit_g": "#D55E00", "bank_marketing": "#009E73",
         "ccdefault": "#CC79A7", "electricity": "#56B4E9"}
COL = {"real": "#0072B2", "random": "#999999", "oracle": "#009E73"}

def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def fig_mc5():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharey=True, sharex=False,
                             constrained_layout=True)
    link = pd.read_csv(os.path.join(RESULTS, "mcdropout_link.csv"))
    axf = axes.flat
    for ax, ds in zip(axf, DATASETS):
        z = np.load(os.path.join(UNC, f"mclink_{ds}.npz"))
        ax.scatter(z["pred_std"], z["pert_instab"], s=9, color=DSCOL[ds],
                   alpha=0.6, edgecolor="none")
        r = link[link.ds == ds]
        if not r.empty:
            ax.text(0.05, 0.9, f"$\\rho_s$ = {r.rho_mc_pert_instab.iloc[0]:.2f}",
                    transform=ax.transAxes, fontsize=8.5)
        ax.text(0.05, 0.78, DSNAME[ds], transform=ax.transAxes, fontsize=8.5)
    axes[1, 2].set_visible(False)
    fig.supxlabel("MC-dropout predictive SD", fontsize=10)
    fig.supylabel("IG instability (1 - cosine) at $\\sigma$ = 0.1", fontsize=10)
    save(fig, "fig_mcdropout_link")

def fig_control():
    df = pd.read_csv(os.path.join("results", "q1_hardening", "control_recovery_summary.csv"))
    df = df[(df["variant"] == "spur") & (df["source"].isin(["random", "real", "oracle"]))].copy()
    order = ["random", "real", "oracle"]
    df["source"] = pd.Categorical(df["source"], order, ordered=True)
    df = df.sort_values("source")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = range(len(df))
    ax.bar(x, df["target_top5"], color=[COL[s] for s in df["source"]], width=0.6)
    for xi, v in zip(x, df["target_top5"]):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(order)
    ax.set_ylabel("Known shortcut in top-5")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    save(fig, "fig_q1_control_recovery")

def fig_budget():
    df = pd.read_csv(os.path.join("results", "q1_hardening", "lime_budget_sweep_summary.csv"))
    df = df[df["variant"] == "spur"].copy()
    fig, ax1 = plt.subplots(figsize=(7.0, 3.6))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("budget")
        ax1.plot(sub["budget"], sub["fast_cosine"], marker="o",
                 label=f"{MODNAME.get(model, model)} (cosine)")
    ax1.set_xscale("log")
    ax1.set_xlabel("LIME samples")
    ax1.set_ylabel("Mean cosine to matched explainer")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    rt = df.groupby("budget")["seconds_per_instance"].median().reset_index()
    ax2.plot(rt["budget"], rt["seconds_per_instance"], color="#D55E00", marker="s",
             ls="--", label="Median runtime")
    ax2.set_ylabel("Median seconds / instance")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left",
               bbox_to_anchor=(0.02, 0.98))
    fig.tight_layout()
    save(fig, "fig_q1_lime_budget_tradeoff")

def fig_damage():
    df = pd.read_csv(os.path.join(RESULTS, "damage_curves_pooled.csv"))
    xs = np.arange(len(SIGMAS))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, m in zip(axes.flat, MODELS_LIST):
        for meth in [mm for mm in METHODS_FOR[m] if mm != "kshap"]:
            sub = df[(df.model == m) & (df.method == meth)].sort_values("sigma")
            if sub.empty:
                continue
            xi = [SIGMAS.index(s) for s in sub.sigma]
            ax.errorbar(xi, sub.top5_mean, yerr=sub.top5_std, marker="o", ms=3.5,
                        capsize=2, lw=1.3, color=C[meth], label=MNAME[meth])
        ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in SIGMAS])
        ax.set_ylim(0, 1.02)
        ax.text(0.03, 0.06, MODNAME[m], transform=ax.transAxes, fontsize=9)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12), frameon=False)
    fig.supxlabel(r"Perturbation magnitude $\sigma$ (feature SD)", fontsize=10)
    fig.supylabel("Top-5 overlap with base explanation", fontsize=10)
    save(fig, "fig_damage_curves")

def fig_conf():
    df = pd.read_csv(os.path.join(RESULTS, "conf_instab_inst.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True,
                             constrained_layout=True)
    nb = 5
    for ax, m in zip(axes.flat, MODELS_LIST):
        for meth in [mm for mm in METHODS_FOR[m] if mm != "kshap"]:
            sub = df[(df.model == m) & (df.method == meth)]
            if sub.empty:
                continue
            q = np.quantile(sub.conf, np.linspace(0, 1, nb + 1))
            q[-1] += 1e-9
            xs_, ys, es = [], [], []
            for i in range(nb):
                msk = (sub.conf >= q[i]) & (sub.conf < q[i + 1])
                if msk.sum() < 5:
                    continue
                xs_.append(sub.conf[msk].mean())
                ys.append(sub.instab_seed[msk].mean())
                es.append(sub.instab_seed[msk].std() / np.sqrt(msk.sum()))
            ax.errorbar(xs_, ys, yerr=es, marker="o", ms=3.5, capsize=2, lw=1.3,
                        color=C[meth], label=MNAME[meth])
        ax.text(0.03, 0.92, MODNAME[m], transform=ax.transAxes, fontsize=9)
        ax.legend(frameon=False, loc="upper right")
    fig.supxlabel("Predictive confidence (seed mean)", fontsize=10)
    fig.supylabel("Cross-seed instability (1 - cosine)", fontsize=10)
    save(fig, "fig_conf_vs_instability")

if __name__ == "__main__":
    fig_mc5(); fig_control(); fig_budget(); fig_damage(); fig_conf()
    print("ROUND3 FIGS DONE")
