"""Emit LaTeX table bodies for the 5-dataset manuscript from the result CSVs.
Run: python analysis/scripts/gen_tables.py"""
import pandas as pd, numpy as np
R = "results"
MN = {"logreg":"Logistic regression","rf":"Random forest","hgb":"Gradient boosting","mlp":"MLP"}
FM = {"linshap":"LinearSHAP","treeshap":"TreeSHAP","ig":"IG","kshap":"KernelSHAP","lime":"LIME"}
DS = ["adult","credit_g","bank_marketing","ccdefault","electricity"]
DSL = {"adult":"Adult","credit_g":"German Credit","bank_marketing":"Bank Marketing","ccdefault":"Default of credit card clients","electricity":"Electricity"}
DSH = {"adult":"Adult","credit_g":"German","bank_marketing":"Bank","ccdefault":"CC-default","electricity":"Electricity"}

def p3(x): return f"{x:.3f}"
def pm(m,s): return f"${m:.3f} \\pm {s:.3f}$"

print("==================== TABLE 2 perf (tab:perf) ====================")
pf = pd.read_csv(f"{R}/perf_summary.csv").set_index(["ds","model","variant"])
for ds in DS:
    print(f"\\multicolumn{{7}}{{l}}{{\\itshape {DSL[ds]}}}\\\\")
    for m in ["logreg","rf","hgb","mlp"]:
        c=pf.loc[(ds,m,"clean")]; s=pf.loc[(ds,m,"spur")]
        print(f"{MN[m]} & {pm(c.acc_mean,c.acc_std)} & {pm(c.auc_mean,c.auc_std)} & {pm(c.ece_mean,c.ece_std)} & {pm(s.acc_mean,s.acc_std)} & {pm(s.auc_mean,s.auc_std)} & {pm(s.ece_mean,s.ece_std)}\\\\")

print("\n==================== TABLE 3 damage (tab:damage) pooled top5 ====================")
dp = pd.read_csv(f"{R}/damage_curves_pooled.csv")
mm={"logreg":"linshap","rf":"treeshap","hgb":"treeshap","mlp":"ig"}
sig=[0.0,0.05,0.1,0.25,0.5]
for m in ["logreg","rf","hgb","mlp"]:
    for meth in [mm[m],"lime"]:
        cells=[]
        for s in sig:
            r=dp[(dp.model==m)&(dp.method==meth)&(np.isclose(dp.sigma,s))]
            cells.append(f"${r.top5_mean.values[0]:.2f} \\pm {r.top5_std.values[0]:.2f}$" if len(r) else "--")
        print(f"{MN[m]} & {FM[meth]} & " + " & ".join(cells) + "\\\\")

print("\n==================== TABLE 4 flip (tab:flip) pooled-by-model % ====================")
fr=pd.read_csv(f"{R}/flip_rates.csv"); sigf=[0.01,0.05,0.1,0.25,0.5]
for m in ["logreg","rf","hgb","mlp"]:
    vals=[fr[(fr.model==m)&(np.isclose(fr.sigma,s))].flip_rate.mean()*100 for s in sigf]
    print(f"{MN[m]} & " + " & ".join(f"{v:.2f}" for v in vals) + "\\\\")

print("\n==================== TABLE 5 wilcoxon (tab:wilcoxon) ====================")
wx=pd.read_csv(f"{R}/wilcoxon.csv")
def pval(p): return "$<10^{-4}$" if p<1e-4 else f"${p:.3f}$"
def rval(r): return f"{r:.2f}" if r>=0 else f"$-{abs(r):.2f}$"
for ds in DS:
    for m in ["logreg","rf","hgb","mlp"]:
        t=wx[(wx.ds==ds)&(wx.model==m)&(wx.metric=="top5")]; c=wx[(wx.ds==ds)&(wx.model==m)&(wx.metric=="cosine")]
        if t.empty: continue
        t=t.iloc[0]; c=c.iloc[0]; fmname=FM[t.method_a]
        print(f"{DSL[ds] if m=='logreg' else ''} & {MN[m]} & {fmname} & {t.mean_a:.3f} & {t.mean_b:.3f} & {pval(t.p)} & {rval(t.r)} & {c.mean_a:.3f} & {c.mean_b:.3f} & {pval(c.p)} & {rval(c.r)}\\\\")
print("-- non-sig (sig_bonf False) --")
for _,r in wx[~wx.sig_bonf].iterrows():
    print(f"   {r.ds} {r.model} {r.metric}: p={r.p:.3f} r={r.r:+.2f} holm_sig={r.sig_holm}")
print(f"sig: bonf={wx.sig_bonf.sum()} holm={wx.sig_holm.sum()} bh={wx.sig_bh.sum()} of {len(wx)}; smallest sig r={wx[wx.sig_bonf].r.min():.2f}; max sig p={wx[wx.sig_bonf].p.max():.2e}")

print("\n==================== TABLE 6 crossseed (tab:crossseed) top-5 per dataset ====================")
cs=pd.read_csv(f"{R}/crossseed.csv")
order=[("logreg","linshap"),("logreg","lime"),("rf","treeshap"),("rf","lime"),("hgb","treeshap"),("hgb","lime"),("mlp","ig"),("mlp","kshap"),("mlp","lime")]
print("header: Model & Method & " + " & ".join(DSH[d] for d in DS) + "\\\\")
for m,meth in order:
    cells=[]
    for ds in DS:
        r=cs[(cs.ds==ds)&(cs.model==m)&(cs.method==meth)]
        cells.append(f"${r.top5_mean.values[0]:.2f} \\pm {r.top5_std.values[0]:.2f}$" if len(r) else "--")
    print(f"{MN[m]} & {FM[meth]} & " + " & ".join(cells) + "\\\\")
# cosine ranges per method (for prose)
print("cosine ranges per method (prose):")
for meth in ["linshap","treeshap","ig","kshap","lime"]:
    v=cs[cs.method==meth].cosine_mean
    print(f"   {FM[meth]}: cos {v.min():.2f}-{v.max():.2f}")
print("method top5 means:", {FM[me]: round(cs[cs.method==me].top5_mean.mean(),3) for me in ["treeshap","kshap","ig","lime"]})

print("\n==================== TABLE 8 mc (tab:mc) ====================")
mc=pd.read_csv(f"{R}/mcdropout_link.csv").set_index("ds")
for ds in DS:
    r=mc.loc[ds]; rp=f"${r.rho_mc_pert_instab:.3f}$" if r.rho_mc_pert_instab>=0 else f"$-{abs(r.rho_mc_pert_instab):.3f}$"
    print(f"{DSL[ds]} & 200 & {r.pred_std_mean:.3f} & {r.rho_mc_seed_instab:.3f} & {rp}\\\\")

print("\n==================== confidence (rho_conf_instab) ====================")
# Deterministic logistic-regression/LinearSHAP cells have undefined correlation.
rho=cs.loc[~((cs.model=="logreg") & (cs.method=="linshap")), "rho_conf_instab"].dropna()
print(f"cells={len(rho)} span {rho.min():.3f}..{rho.max():.3f} median {rho.median():.3f} |rho|>0.3: {(rho.abs()>0.3).sum()}")

print("\n==================== TABLE spurious (tab:spur) ranges ====================")
sp=pd.read_csv(f"{R}/spurious.csv"); fast=sp[sp.method!='lime']; lime=sp[sp.method=='lime']
print(f"rel_share {sp.rel_share.min():.3f}-{sp.rel_share.max():.3f}; fast sal {fast.sal_share.min():.3f}-{fast.sal_share.max():.3f}; fast top5 {fast.top5_frac.min():.2f}-{fast.top5_frac.max():.2f}; all rank1 {(fast.attr_rank==1).all()}; LIME sal {lime.sal_share.min():.3f}-{lime.sal_share.max():.3f}")
