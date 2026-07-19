"""Stage 6: publication figures (PDF + PNG, 300 dpi, colorblind-safe, serif)."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, FIGS, UNC, DATASETS, MODELS_LIST, SIGMAS, METHODS_FOR

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300,
})
# Okabe-Ito
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

def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

def fig_damage():
    df = pd.read_csv(os.path.join(RESULTS, "damage_curves_pooled.csv"))
    xs = np.arange(len(SIGMAS))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey=True)
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
    for ax in axes[1]: ax.set_xlabel(r"Perturbation magnitude $\sigma$ (feature SD)")
    for ax in axes[:, 0]: ax.set_ylabel("Top-5 overlap with base explanation")
    fig.tight_layout()
    save(fig, "fig_damage_curves")

def fig_crossseed():
    df = pd.read_csv(os.path.join(RESULTS, "crossseed.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    pos, labels = [], []
    x = 0.0
    for m in MODELS_LIST:
        for meth in METHODS_FOR[m]:
            sub = df[(df.model == m) & (df.method == meth)]
            if sub.empty:
                continue
            ax.bar(x, sub.top5_mean.mean(), width=0.8, color=C[meth], alpha=0.85)
            ax.scatter([x] * len(sub), sub.top5_mean, s=12, facecolor="white",
                       edgecolor="black", linewidth=0.5, zorder=3)
            pos.append(x); labels.append(MNAME[meth])
            x += 1.0
        x += 0.7
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Cross-seed top-5 agreement")
    ax.set_ylim(0, 1.02)
    # model group annotations
    x = 0.0
    for m in MODELS_LIST:
        n = len(METHODS_FOR[m])
        ax.text(x + (n - 1) / 2, 1.06, MODNAME[m], ha="center", fontsize=8.5)
        x += n + 0.7
    fig.tight_layout()
    save(fig, "fig_crossseed_agreement")

def fig_conf():
    df = pd.read_csv(os.path.join(RESULTS, "conf_instab_inst.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    nb = 5
    for ax, m in zip(axes.flat, MODELS_LIST):
        for meth in [mm for mm in METHODS_FOR[m] if mm != "kshap"]:
            sub = df[(df.model == m) & (df.method == meth)]
            if sub.empty:
                continue
            q = np.quantile(sub.conf, np.linspace(0, 1, nb + 1))
            q[-1] += 1e-9
            xs, ys, es = [], [], []
            for i in range(nb):
                msk = (sub.conf >= q[i]) & (sub.conf < q[i + 1])
                if msk.sum() < 5:
                    continue
                xs.append(sub.conf[msk].mean())
                ys.append(sub.instab_seed[msk].mean())
                es.append(sub.instab_seed[msk].std() / np.sqrt(msk.sum()))
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=3.5, capsize=2, lw=1.3,
                        color=C[meth], label=MNAME[meth])
        ax.text(0.03, 0.92, MODNAME[m], transform=ax.transAxes, fontsize=9)
        ax.legend(frameon=False, loc="upper right")
    for ax in axes[1]: ax.set_xlabel("Predictive confidence (seed mean)")
    for ax in axes[:, 0]: ax.set_ylabel("Cross-seed instability (1 - cosine)")
    fig.tight_layout()
    save(fig, "fig_conf_vs_instability")

def fig_spurious():
    df = pd.read_csv(os.path.join(RESULTS, "spurious.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    mark = {"linshap": "o", "treeshap": "s", "ig": "^", "lime": "D"}
    for _, r in df.iterrows():
        ax.scatter(r.rel_share, r.sal_share, marker=mark.get(r.method, "o"),
                   s=42, color=DSCOL[r.ds], edgecolor="black", linewidth=0.4,
                   alpha=0.85)
    lim = max(df.rel_share.max(), df.sal_share.max()) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Behavioral reliance share of spurious feature")
    ax.set_ylabel("Attribution salience share of spurious feature")
    from matplotlib.lines import Line2D
    h1 = [Line2D([0], [0], marker=mark[k], ls="", mfc="lightgray", mec="black",
                 label=MNAME[k]) for k in mark]
    h2 = [Line2D([0], [0], marker="o", ls="", color=DSCOL[d], label=DSNAME[d])
          for d in DATASETS]
    ax.legend(handles=h1 + h2, frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_spurious_detection")

def fig_mc():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    link = pd.read_csv(os.path.join(RESULTS, "mcdropout_link.csv"))
    for ax, ds in zip(axes, DATASETS):
        f = os.path.join(UNC, f"mclink_{ds}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ax.scatter(z["pred_std"], z["pert_instab"], s=9, color=DSCOL[ds],
                   alpha=0.6, edgecolor="none")
        r = link[link.ds == ds]
        if not r.empty:
            ax.text(0.05, 0.9, f"$\\rho_s$ = {r.rho_mc_pert_instab.iloc[0]:.2f}",
                    transform=ax.transAxes, fontsize=8.5)
        ax.set_xlabel("MC-dropout predictive SD")
        ax.text(0.05, 0.78, DSNAME[ds], transform=ax.transAxes, fontsize=8.5)
    axes[0].set_ylabel("IG instability (1 - cosine)\nat $\\sigma$ = 0.1")
    fig.tight_layout()
    save(fig, "fig_mcdropout_link")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("damage", "all"): fig_damage()
    if which in ("crossseed", "all"): fig_crossseed()
    if which in ("conf", "all"): fig_conf()
    if which in ("spur", "all"): fig_spurious()
    if which in ("mc", "all"): fig_mc()
    print("FIGSDONE")
