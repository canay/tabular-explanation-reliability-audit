"""Lever #1: a deploy-time per-instance explanation FRAGILITY score that predicts
whether the top-5 set survives retraining, computable from a single fitted model.
Signals (all inference-time, no retraining):
  - sampling self-consistency: top-5 overlap across repeated explainer calls (sigma=0)
  - perturbation self-consistency: top-5 overlap under the prediction-preserving noise
  - NEW top-5 margin: normalized gap |a|_(5) - |a|_(6) from one seed-0 attribution
Ground truth instability: cross-seed top-5 overlap < tau (would change under retraining).
Reports per-group Spearman, ROC-AUC of each signal, and recall at matched FPR for the
old rule vs the margin-augmented rule. Run from project root."""
import sys, os, io; sys.path.insert(0, "code")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from common import EXPL, DATASETS, MODELS_LIST, METHODS_FOR
out = io.StringIO()
def pr(*a): print(*a, file=out)

TAU = 0.80
# Threshold tolerance, 2026-09-21. Every top-5 overlap is an integer count
# over five, so an attainable mean can be mathematically equal to TAU while
# floating point stores it as 0.7999999999999999. Comparing raw therefore
# counted such a unit as below the threshold. TAU_TOL places a value equal
# to TAU on the stable side; it is far below the smallest nonzero distance
# between an attainable value and TAU (0.004 across the thresholds used).
TAU_TOL = 1e-9
def topk_margin(a, k=5):
    m = np.sort(np.abs(a))[::-1]
    if len(m) <= k or m[0] <= 0: return 1.0
    return float((m[k-1] - m[k]) / m[0])     # gap between kth and (k+1)th, normalized by top

# NEW signal: top-5 margin from the seed-0 clean attribution (single fitted model)
rows = []
for ds in DATASETS:
    for mdl in MODELS_LIST:
        for meth in METHODS_FOR[mdl]:
            f = os.path.join(EXPL, f"base_{ds}_{mdl}_{meth}_0_clean.npz")
            if not os.path.exists(f): continue
            A = np.load(f)["attr"]
            for i in range(A.shape[0]):
                rows.append(dict(ds=ds, model=mdl, method=meth, inst=i, margin=topk_margin(A[i])))
mg = pd.DataFrame(rows)

ci = pd.read_csv("results/conf_instab_inst.csv")[["ds","model","method","inst","top5_seed"]]
raw = pd.read_csv("results/perturb_raw.csv")
samp = raw[raw.sigma==0.0].groupby(["ds","model","method","inst"]).top5.mean().rename("samp").reset_index()
pert = raw[(raw.sigma==0.25)&(raw.preserved==1)].groupby(["ds","model","method","inst"]).top5.mean().rename("pert").reset_index()

df = mg.merge(ci, on=["ds","model","method","inst"]) \
       .merge(samp, on=["ds","model","method","inst"], how="left") \
       .merge(pert, on=["ds","model","method","inst"], how="left")
df["samp"] = df["samp"].fillna(1.0); df["pert"] = df["pert"].fillna(1.0)
df["group"] = np.where(df.method=="lime", "LIME", "Model-matched")
df["unstable"] = (df.top5_seed < TAU - TAU_TOL).astype(int)

pr("=== Does top-5 MARGIN predict cross-seed stability? Spearman(margin, crossseed top5) ===")
for g, sub in df.groupby("group"):
    rho = spearmanr(sub.margin, sub.top5_seed).statistic
    pr(f"  {g:14s} n={len(sub):4d} rho={rho:+.3f}  base_unstable={sub.unstable.mean():.2f}")

def recall_at_fpr(score, y, target_fpr=0.20):
    # score high => predict unstable; sweep threshold for FPR<=target, max recall
    order = np.argsort(-score); y = y[order]; s = score[order]
    P = y.sum(); N = len(y)-P
    tp = np.cumsum(y); fp = np.cumsum(1-y)
    fpr = fp/max(N,1); rec = tp/max(P,1)
    ok = fpr <= target_fpr
    return float(rec[ok].max()) if ok.any() else 0.0

pr("\n=== Signal quality (ROC-AUC vs 'unstable'; recall at FPR<=0.20) ===")
for g, sub in df.groupby("group"):
    y = sub.unstable.values
    if y.sum()==0 or y.sum()==len(y):
        pr(f"  {g}: degenerate (base_unstable={y.mean():.2f}), skip"); continue
    s_margin = (1 - sub.margin.values)
    s_old = (1 - np.minimum(sub.samp.values, sub.pert.values))   # old cheap rule score
    s_comb = np.maximum(s_old, s_margin)                          # margin-augmented
    for name, s in [("margin-only", s_margin), ("old(samp,pert)", s_old), ("combined", s_comb)]:
        auc = roc_auc_score(y, s); rec = recall_at_fpr(s, y, 0.20)
        pr(f"  {g:14s} {name:16s} AUC={auc:.3f}  recall@FPR0.20={rec:.2f}")

# Overall + explicit recall-gap-closing comparison at a fixed operating point
pr("\n=== Operating point comparison (model-matched): old rule vs margin-augmented ===")
mm = df[df.group=="Model-matched"]; y = mm.unstable.values
old = (np.minimum(mm.samp.values, mm.pert.values) < TAU - TAU_TOL)
# choose margin threshold delta to match ~0.20 FPR on model-matched
from numpy import quantile
for delta in [0.02, 0.05, 0.08, 0.10]:
    newflag = old | (mm.margin.values < delta)
    tp=((newflag)&(y==1)).sum(); fp=((newflag)&(y==0)).sum()
    fn=((~newflag)&(y==1)).sum(); tn=((~newflag)&(y==0)).sum()
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; fpr=fp/(fp+tn) if fp+tn else 0
    pr(f"  delta={delta:.2f}: prec={prec:.2f} recall={rec:.2f} FPR={fpr:.2f} (old rule recall was 0.16)")

df.to_csv("analysis/results/fragility.csv", index=False)
open("analysis/results/fragility.txt","w").write(out.getvalue())
print(out.getvalue())
