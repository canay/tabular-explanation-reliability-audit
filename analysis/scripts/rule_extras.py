"""Supplementary decision-rule analyses (run from project root):
  - clean descriptive top-5 real-vs-random paired significance test + bootstrap CI;
  - an operating 'report-or-flag' reliability rule validated against cross-seed
         instability, with per-method-group precision / recall / false-positive rate.
Writes analysis/results/rule_extras.txt and decision_rule.csv.
"""
import sys, io, os
sys.path.insert(0, "code")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from common import FAST_METHOD

rng = np.random.default_rng(0)
out = io.StringIO()
def pr(*a): print(*a, file=out)

def boot_ci_mean(x, B=5000):
    x = np.asarray(x, float); n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

# ===== A2-08: clean descriptive top-5, real vs random (and permuted) =====
pr("===== A2-08  Clean descriptive top-5: real vs control =====")
b = pd.read_csv("results/q1_hardening/benchmark_results.csv")
cl = b[(b.variant == "clean") & (b.truth_type == "descriptive_top5")].copy()
key = ["dataset", "model", "method", "seed"]
piv = cl.pivot_table(index=key, columns="source", values=["ap", "precision_at_5"])
for metric in ["ap", "precision_at_5"]:
    real = piv[(metric, "real")].dropna()
    rand = piv[(metric, "random")].reindex(real.index)
    d = (real - rand).dropna()
    real2 = real.reindex(d.index); rand2 = rand.reindex(d.index)
    try:
        W, p = wilcoxon(real2.values, rand2.values)
    except Exception as e:
        W, p = float("nan"), 1.0
    lo, hi = boot_ci_mean(d.values)
    pr(f"  {metric}: n={len(d)} real={real2.mean():.3f} random={rand2.mean():.3f} "
       f"diff(real-random)={d.mean():+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] Wilcoxon p={p:.3f}")
pr("  -> interpretation: CI of (real-random) for AP and P@5 includes/excludes 0?")

# ===== A2-07: operating report-or-flag rule =====
pr("\n===== A2-07  Operating reliability rule (tau=0.80) =====")
raw = pd.read_csv("results/perturb_raw.csv")
ci = pd.read_csv("results/conf_instab_inst.csv")          # has top5_seed (cross-seed)
TAU = 0.80
# Threshold tolerance, 2026-09-21. Every top-5 overlap is an integer count
# over five, so an attainable mean can be mathematically equal to TAU while
# floating point stores it as 0.7999999999999999. Comparing raw therefore
# counted such a unit as below the threshold. TAU_TOL places a value equal
# to TAU on the stable side; it is far below the smallest nonzero distance
# between an attainable value and TAU (0.004 across the thresholds used).
TAU_TOL = 1e-9
# cheap deploy-time diagnostics, per (ds,model,method,inst):
samp = (raw[raw.sigma == 0.0].groupby(["ds","model","method","inst"])["top5"].mean()
        .rename("samp_overlap").reset_index())
pert = (raw[(raw.sigma == 0.25) & (raw.preserved == 1)]
        .groupby(["ds","model","method","inst"])["top5"].mean()
        .rename("pert_overlap").reset_index())
gt = ci[["ds","model","method","inst","top5_seed"]]
df = samp.merge(pert, on=["ds","model","method","inst"]).merge(gt, on=["ds","model","method","inst"])
# cheap flag: low self-consistency under repeated sampling OR small perturbation
df["cheap_overlap"] = df[["samp_overlap","pert_overlap"]].min(axis=1)
df["flag"] = df["cheap_overlap"] < TAU - TAU_TOL
df["unreliable"] = df["top5_seed"] < TAU - TAU_TOL            # ground truth: would change under retraining
df["group"] = np.where(df.method == "lime", "LIME (agnostic)", "Model-matched")
rows = []
for g, sub in df.groupby("group"):
    tp = ((sub.flag) & (sub.unreliable)).sum()
    fp = ((sub.flag) & (~sub.unreliable)).sum()
    fn = ((~sub.flag) & (sub.unreliable)).sum()
    tn = ((~sub.flag) & (~sub.unreliable)).sum()
    n = len(sub)
    prec = tp/(tp+fp) if tp+fp else float("nan")
    rec = tp/(tp+fn) if tp+fn else float("nan")
    fpr = fp/(fp+tn) if fp+tn else float("nan")
    rho = spearmanr(sub.cheap_overlap, sub.top5_seed).statistic
    rows.append(dict(group=g, n=n, base_unreliable=sub.unreliable.mean(), flag_rate=sub.flag.mean(),
                     precision=prec, recall=rec, fpr=fpr, rho_cheap_vs_crossseed=rho))
    pr(f"  {g:16s} n={n:4d} base_unreliable={sub.unreliable.mean():.2f} flag_rate={sub.flag.mean():.2f} "
       f"prec={prec:.2f} recall={rec:.2f} FPR={fpr:.2f} rho(cheap,crossseed)={rho:+.2f}")
pd.DataFrame(rows).to_csv("analysis/results/decision_rule.csv", index=False)
# overall
tp=((df.flag)&(df.unreliable)).sum(); fp=((df.flag)&(~df.unreliable)).sum()
fn=((~df.flag)&(df.unreliable)).sum(); tn=((~df.flag)&(~df.unreliable)).sum()
pr(f"  OVERALL          n={len(df)} prec={tp/(tp+fp):.2f} recall={tp/(tp+fn):.2f} FPR={fp/(fp+tn):.2f}")
pr(f"  (TP={tp} FP={fp} FN={fn} TN={tn})")

open("analysis/results/rule_extras.txt","w").write(out.getvalue())
print(out.getvalue())
