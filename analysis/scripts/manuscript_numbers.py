"""Extract manuscript-ready numbers from the regenerated authoritative result CSVs.
Run from project root: python analysis/scripts/manuscript_numbers.py
Writes a human-readable digest to analysis/results/manuscript_numbers.txt
"""
import sys, os, io, json
sys.path.insert(0, "code")
import numpy as np, pandas as pd
from common import DATASETS, MODELS_LIST, SEEDS, EXPL

R = "results"
out = io.StringIO()
def pr(*a): print(*a, file=out)

MNAME = {"logreg":"Logistic regression","rf":"Random forest","hgb":"Gradient boosting","mlp":"MLP"}
FASTM = {"linshap":"LinearSHAP","treeshap":"TreeSHAP","ig":"IG","kshap":"KernelSHAP","lime":"LIME"}

# ---- 1. Eval instance counts (base explanation array shapes) ----
pr("=== EVAL INSTANCE COUNTS (base explanation n per cell) ===")
for ds in DATASETS:
    line = [ds]
    for m,meth in [("logreg","linshap"),("logreg","lime"),("mlp","kshap")]:
        f=os.path.join(EXPL,f"base_{ds}_{m}_{meth}_0_clean.npz")
        if os.path.exists(f):
            line.append(f"{m}/{meth}={np.load(f)['attr'].shape[0]}")
    pr("  "+"  ".join(line))

# ---- 2. FLIP table (pooled by model over datasets), percent ----
pr("\n=== TABLE 4 FLIP RATES (pooled by model, %) ===")
fr = pd.read_csv(f"{R}/flip_rates.csv")
sigs=[0.01,0.05,0.1,0.25,0.5]
pr("  model           " + "  ".join(f"s={s}" for s in sigs))
for m in MODELS_LIST:
    vals=[fr[(fr.model==m)&(fr.sigma==s)].flip_rate.mean()*100 for s in sigs]
    pr(f"  {MNAME[m]:16s}" + "  ".join(f"{v:4.2f}" for v in vals))
# pooled preservation at 0.25
pres=(1-fr[fr.sigma==0.25].groupby("model").flip_rate.mean()).mean()*100
pr(f"  pooled preservation @ sigma=0.25 = {pres:.1f}%  (flip {100-pres:.1f}%)")
pmm=fr[fr.sigma==0.25].groupby("model").flip_rate.mean()*100
pr(f"  per-model flip @0.25: " + ", ".join(f"{MNAME[m]} {pmm[m]:.2f}%" for m in MODELS_LIST))

# ---- 3. DAMAGE pooled (model-matched + LIME) top5 across sigma ----
pr("\n=== TABLE 3 DAMAGE pooled top5 (model-matched & LIME) ===")
dp=pd.read_csv(f"{R}/damage_curves_pooled.csv")
sig5=[0.0,0.05,0.1,0.25,0.5]
mm={"logreg":"linshap","rf":"treeshap","hgb":"treeshap","mlp":"ig"}
for m in MODELS_LIST:
    for meth in [mm[m],"lime"]:
        row=[ (dp[(dp.model==m)&(dp.method==meth)&(np.isclose(dp.sigma,s))].top5_mean.values,
               dp[(dp.model==m)&(dp.method==meth)&(np.isclose(dp.sigma,s))].top5_std.values) for s in sig5]
        cells="  ".join(f"{v[0][0]:.2f}+-{v[1][0]:.2f}" if len(v[0]) else "  -  " for v in row)
        pr(f"  {MNAME[m]:16s} {FASTM[meth]:10s} {cells}")
# specific: model-matched top5 range at 0.25 and 0.5; GB spearman & sign at 0.5
mm25=[dp[(dp.model==m)&(dp.method==mm[m])&(np.isclose(dp.sigma,0.25))].top5_mean.values[0] for m in MODELS_LIST]
mm50=[dp[(dp.model==m)&(dp.method==mm[m])&(np.isclose(dp.sigma,0.5))].top5_mean.values[0] for m in MODELS_LIST]
pr(f"  model-matched top5 @0.25 range = {min(mm25):.2f}-{max(mm25):.2f}; @0.5 = {min(mm50):.2f}-{max(mm50):.2f}")
gb=dp[(dp.model=='hgb')&(dp.method=='treeshap')&(np.isclose(dp.sigma,0.5))]
pr(f"  GB TreeSHAP @0.5: spearman={gb.spearman_mean.values[0]:.2f} sign5={gb.sign5_mean.values[0]:.2f}")
lim0=[dp[(dp.model==m)&(dp.method=='lime')&(np.isclose(dp.sigma,0.0))].top5_mean.values[0] for m in MODELS_LIST]
pr(f"  LIME top5 @sigma=0 range = {min(lim0):.2f}-{max(lim0):.2f}")

# ---- 4. CROSSSEED table + method means + retraining range ----
pr("\n=== TABLE 6 CROSSSEED (top5 +- std / cosine) per ds ===")
cs=pd.read_csv(f"{R}/crossseed.csv")
order=[("logreg","linshap"),("logreg","lime"),("rf","treeshap"),("rf","lime"),
       ("hgb","treeshap"),("hgb","lime"),("mlp","ig"),("mlp","kshap"),("mlp","lime")]
for m,meth in order:
    parts=[]
    for ds in DATASETS:
        r=cs[(cs.ds==ds)&(cs.model==m)&(cs.method==meth)]
        if len(r): parts.append(f"{ds[:4]}:{r.top5_mean.values[0]:.2f}+-{r.top5_std.values[0]:.2f}/{r.cosine_mean.values[0]:.2f}")
    pr(f"  {MNAME[m]:16s} {FASTM[meth]:10s} "+"  ".join(parts))
pr("  -- method-level means over all cells --")
for meth in ["treeshap","kshap","ig","lime"]:
    sub=cs[cs.method==meth]
    pr(f"    {FASTM[meth]:10s} top5 mean = {sub.top5_mean.mean():.3f}")
# retraining range: stochastic trainers (exclude logreg-linshap which is deterministic=1.0)
stoch=cs[~((cs.model=='logreg')&(cs.method=='linshap'))]
stoch=cs[cs.method.isin(["treeshap","ig","kshap"])]
pr(f"  retraining replaces top5: {(1-stoch.top5_mean.max())*100:.0f}%-{(1-stoch.top5_mean.min())*100:.0f}% (model-matched/kshap stochastic)")
pr(f"    treeshap rf range {cs[(cs.model=='rf')&(cs.method=='treeshap')].top5_mean.min():.2f}-{cs[(cs.model=='rf')&(cs.method=='treeshap')].top5_mean.max():.2f}")
pr(f"    treeshap hgb range {cs[(cs.model=='hgb')&(cs.method=='treeshap')].top5_mean.min():.2f}-{cs[(cs.model=='hgb')&(cs.method=='treeshap')].top5_mean.max():.2f}")
pr(f"    ig range {cs[cs.method=='ig'].top5_mean.min():.2f}-{cs[cs.method=='ig'].top5_mean.max():.2f}")
pr(f"    kshap range {cs[cs.method=='kshap'].top5_mean.min():.2f}-{cs[cs.method=='kshap'].top5_mean.max():.2f}")
csmin=stoch.top5_mean.idxmin()
pr(f"    lowest cell: {cs.loc[csmin,'ds']} {cs.loc[csmin,'model']} {cs.loc[csmin,'method']} = {cs.loc[csmin,'top5_mean']:.2f}")

# ---- 5. CONFIDENCE (rho_conf_instab) ----
pr("\n=== CONFIDENCE rho_conf_instab (defined cells; deterministic LinearSHAP excluded) ===")
# Deterministic logistic-regression/LinearSHAP cells have undefined correlation.
rho=cs.loc[~((cs.model=="logreg") & (cs.method=="linshap")), "rho_conf_instab"].dropna()
pr(f"  span {rho.min():.3f} to {rho.max():.3f}; median {rho.median():.3f}; |rho|>0.3 count = {(rho.abs()>0.3).sum()} of {len(rho)}")
for m,meth,lbl in [("mlp","kshap","KernelSHAP/Adult"),("mlp","ig","IG/Adult")]:
    v=cs[(cs.ds=='adult')&(cs.model==m)&(cs.method==meth)].rho_conf_instab.values
    if len(v): pr(f"    Adult {FASTM[meth]} rho={v[0]:.2f}")
for ds,m,meth in [("bank_marketing","hgb","lime"),("bank_marketing","logreg","lime")]:
    v=cs[(cs.ds==ds)&(cs.model==m)&(cs.method==meth)].rho_conf_instab.values
    if len(v): pr(f"    {ds} {m} LIME rho={v[0]:+.2f}")

# ---- 6. MC dropout table ----
pr("\n=== TABLE 8 MC-DROPOUT ===")
mc=pd.read_csv(f"{R}/mcdropout_link.csv")
for _,r in mc.iterrows():
    pr(f"  {r.ds:14s} n={int(r.n)} pred_sd={r.pred_std_mean:.3f} rho_seed={r.rho_mc_seed_instab:.3f} rho_pert={r.rho_mc_pert_instab:.3f}")

# ---- 7. WILCOXON summary ----
pr("\n=== TABLE 5 WILCOXON (n / sig counts) ===")
wx=pd.read_csv(f"{R}/wilcoxon.csv")
pr(f"  per-cell n: {sorted(wx.n.unique())}; sig bonf={wx.sig_bonf.sum()} holm={wx.sig_holm.sum()} bh={wx.sig_bh.sum()} of {len(wx)}")
pr(f"  significant p range: {wx[wx.sig_bonf].p.max():.2e} (max among sig) ; all sig <1e-4? {(wx[wx.sig_bonf].p<1e-4).all()}")
pr(f"  smallest r among sig: {wx[wx.sig_bonf].r.min():.2f}")
pr("  non-significant cells:")
for _,r in wx[~wx.sig_bonf].iterrows():
    pr(f"    {r.ds} {r.model} {r.metric}: p={r.p:.3f} r={r.r:+.2f} CI[{r.r_ci_low:+.2f},{r.r_ci_high:+.2f}]")

# ---- 8. SPURIOUS ranges ----
pr("\n=== TABLE 7 SPURIOUS (ranges) ===")
sp=pd.read_csv(f"{R}/spurious.csv")
fast=sp[sp.method!='lime']; lime=sp[sp.method=='lime']
pr(f"  reliance share range = {sp.rel_share.min():.3f}-{sp.rel_share.max():.3f}")
pr(f"  fast salience share range = {fast.sal_share.min():.3f}-{fast.sal_share.max():.3f}")
pr(f"  fast top5 range = {fast.top5_frac.min():.2f}-{fast.top5_frac.max():.2f}; all rank 1? {(fast.attr_rank==1).all()}")
pr(f"  LIME salience share range = {lime.sal_share.min():.3f}-{lime.sal_share.max():.3f}")

# ---- 9. vardecomp / delta / broader uncertainty / hardening ----
pr("\n=== ADDITIONAL ANALYSES (analysis/results, hardening) ===")
for f,lbl in [("analysis/results/variance_decomp.csv","vardecomp"),
              ("analysis/results/delta_sweep_agg.csv","delta_agg"),
              ("analysis/results/broader_uncertainty.csv","broader_unc")]:
    if os.path.exists(f):
        pr(f"  [{lbl}] {f}")
        pr("    "+pd.read_csv(f).to_string().replace("\n","\n    "))
hd="results/q1_hardening/control_recovery_summary.csv"
if os.path.exists(hd):
    h=pd.read_csv(hd); pr(f"  [hardening control recovery]\n    "+h.to_string().replace("\n","\n    "))
sg="results/q1_hardening/synthetic_ground_truth_summary.csv"
if os.path.exists(sg):
    pr(f"  [synthetic ground truth]\n    "+pd.read_csv(sg).to_string().replace("\n","\n    "))

open("analysis/results/manuscript_numbers.txt","w").write(out.getvalue())
print(out.getvalue())
