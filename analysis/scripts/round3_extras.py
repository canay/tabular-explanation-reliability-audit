"""Round 3 (A3-01, A3-02, A3-10) supplementary computations from FROZEN artifacts.
No training, no explainer calls. Run from project root.
Outputs: analysis/results/round3_rule_extras.txt (+ tau sweep CSV).
- Matched-sample AUCs: self-consistency score (1 - min(samp,pert)) vs margin, both
  against the same cross-seed instability target, on the SAME n=1000/4000 sample as
  Table tab:rule (rule_extras.py inner-join sample; no fillna imputation).
- Always-flag baseline and tau sensitivity {0.70, 0.80, 0.90} for the rule.
- Wilcoxon effect-size CI verification for the 8 non-Bonferroni cells (A3-02).
"""
import sys, io, os
sys.path.insert(0, "code")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

out = io.StringIO()
def pr(*a): print(*a, file=out)

raw = pd.read_csv("results/perturb_raw.csv")
ci = pd.read_csv("results/conf_instab_inst.csv")
frag = pd.read_csv("analysis/results/fragility.csv")[["ds","model","method","inst","margin"]]

samp = (raw[raw.sigma == 0.0].groupby(["ds","model","method","inst"])["top5"].mean()
        .rename("samp").reset_index())
pert = (raw[(raw.sigma == 0.25) & (raw.preserved == 1)]
        .groupby(["ds","model","method","inst"])["top5"].mean()
        .rename("pert").reset_index())
gt = ci[["ds","model","method","inst","top5_seed"]]
df = samp.merge(pert, on=["ds","model","method","inst"]).merge(gt, on=["ds","model","method","inst"])
df = df.merge(frag, on=["ds","model","method","inst"], how="left")
df["group"] = np.where(df.method == "lime", "LIME", "Model-matched")

pr("===== A3-01: matched-sample AUCs (same sample as tab:rule; no imputation) =====")
for g, sub in df.groupby("group"):
    y = (sub.top5_seed < 0.80).astype(int).values
    s_sc = 1 - np.minimum(sub.samp.values, sub.pert.values)
    auc_sc = roc_auc_score(y, s_sc)
    m = sub.dropna(subset=["margin"])
    ym = (m.top5_seed < 0.80).astype(int).values
    auc_mg = roc_auc_score(ym, 1 - m.margin.values) if len(m) else float("nan")
    pr(f"  {g:14s} n={len(sub):5d} base={y.mean():.3f}  AUC(self-consistency)={auc_sc:.3f}  "
       f"AUC(margin, same sample n={len(m)})={auc_mg:.3f}")
pr("  (fragility.txt extended sample, samp/pert imputed to 1.0 where unavailable: "
   "LIME margin 0.630 vs self-cons 0.600; MM margin 0.575 vs self-cons 0.545)")

pr("\n===== A3-10: always-flag baseline + tau sensitivity (flag and ground truth share tau) =====")
rows = []
for tau in [0.70, 0.80, 0.90]:
    for g, sub in df.groupby("group"):
        flag = np.minimum(sub.samp.values, sub.pert.values) < tau
        unrel = (sub.top5_seed < tau).values
        tp = (flag & unrel).sum(); fp = (flag & ~unrel).sum()
        fn = (~flag & unrel).sum(); tn = (~flag & ~unrel).sum()
        prec = tp/(tp+fp) if tp+fp else float("nan")
        rec = tp/(tp+fn) if tp+fn else float("nan")
        fpr = fp/(fp+tn) if fp+tn else float("nan")
        rows.append(dict(tau=tau, group=g, n=len(sub), base=unrel.mean(),
                         flag_rate=flag.mean(), precision=prec, recall=rec, fpr=fpr))
        pr(f"  tau={tau:.2f} {g:14s} base={unrel.mean():.2f} flag={flag.mean():.2f} "
           f"prec={prec:.2f} rec={rec:.2f} FPR={fpr:.2f}")
    pr("")
for g, sub in df.groupby("group"):
    base = (sub.top5_seed < 0.80).mean()
    pr(f"  always-flag baseline at tau=0.80, {g}: precision={base:.2f} recall=1.00 FPR=1.00")
pd.DataFrame(rows).to_csv("analysis/results/round3_tau_sweep.csv", index=False)

pr("\n===== A3-02: wilcoxon.csv verification =====")
w = pd.read_csv("results/wilcoxon.csv")
pr(f"  nominal p<0.05: {(w.p < 0.05).sum()} of {len(w)}; sig_bonf: {w.sig_bonf.sum()}; "
   f"sig_holm: {w.sig_holm.sum()}; sig_bh: {w.sig_bh.sum()}")
nb = w[~w.sig_bonf]
pr("  8 non-Bonferroni cells (r [CI]):")
for _, r in nb.iterrows():
    pr(f"    {r.ds:12s} {r.model:6s} {r.metric:6s} p={r.p:.3f} r={r.r:+.2f} "
       f"CI=[{r.r_ci_low:+.2f},{r.r_ci_high:+.2f}] holm={r.sig_holm} bh={r.sig_bh}")

open("analysis/results/round3_rule_extras.txt", "w").write(out.getvalue())
print(out.getvalue())
