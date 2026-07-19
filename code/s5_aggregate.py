"""Stage 5: aggregate all raw outputs into result tables + results_summary.json."""
import os, sys, json, glob, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, MODELS_LIST, SEEDS, SIGMAS, EXPL, PERT, UNC, RESULTS,
                    METHODS_FOR, FAST_METHOD, N_LIME_BASE, N_KSHAP, N_EVAL,
                    load_dataset, pair_metrics, cosine)

METRICS = ["top3", "top5", "spearman", "cosine", "sign5"]

def perf_tables():
    df = pd.read_json(os.path.join(RESULTS, "perf.jsonl"), lines=True)
    df = df.drop_duplicates(subset=["ds", "model", "seed", "variant"], keep="last")
    df.to_csv(os.path.join(RESULTS, "perf_raw.csv"), index=False)
    g = df.groupby(["ds", "model", "variant"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std")).reset_index()
    g.to_csv(os.path.join(RESULTS, "perf_summary.csv"), index=False)
    return df, g

def damage_curves():
    files = glob.glob(os.path.join(PERT, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(os.path.join(RESULTS, "perturb_raw.csv"), index=False)
    # flip rate per ds/model/sigma (model-level, method-independent; use fast rows)
    fr = (raw[raw.method != "lime"].groupby(["ds", "model", "sigma"])["preserved"]
          .mean().rsub(1).rename("flip_rate").reset_index())
    fr.to_csv(os.path.join(RESULTS, "flip_rates.csv"), index=False)
    kept = raw[raw.preserved == 1].copy()
    # per-seed mean then mean/std over seeds
    per_seed = kept.groupby(["ds", "model", "method", "sigma", "seed"])[METRICS].mean().reset_index()
    agg = per_seed.groupby(["ds", "model", "method", "sigma"])[METRICS].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(RESULTS, "damage_curves.csv"), index=False)
    # pooled over datasets (per-dataset-seed means as samples)
    pooled = per_seed.groupby(["model", "method", "sigma"])[METRICS].agg(["mean", "std"])
    pooled.columns = ["_".join(c) for c in pooled.columns]
    pooled.reset_index().to_csv(os.path.join(RESULTS, "damage_curves_pooled.csv"), index=False)
    return raw, per_seed, fr

def load_base(ds, model, method, seed, variant="clean"):
    f = os.path.join(EXPL, f"base_{ds}_{model}_{method}_{seed}_{variant}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return z["attr"], z["probs"]

def crossseed_and_conf():
    rows, inst_rows = [], []
    for ds in DATASETS:
        for m in MODELS_LIST:
            for meth in METHODS_FOR[m]:
                packs = [load_base(ds, m, meth, s) for s in SEEDS]
                if any(p is None for p in packs):
                    continue
                A = np.stack([p[0] for p in packs])      # [S, n, D]
                P = np.stack([p[1] for p in packs])      # [S, n]
                n = A.shape[1]
                pairs = list(itertools.combinations(range(len(SEEDS)), 2))
                pm_all = {k: np.zeros((len(pairs), n)) for k in METRICS}
                for pi, (a, b) in enumerate(pairs):
                    for i in range(n):
                        pm = pair_metrics(A[a, i], A[b, i])
                        for k in METRICS:
                            pm_all[k][pi, i] = pm[k]
                inst_mean = {k: pm_all[k].mean(axis=0) for k in METRICS}
                conf = np.maximum(P, 1 - P).mean(axis=0)
                instab = 1 - inst_mean["cosine"]
                for i in range(n):
                    inst_rows.append(dict(ds=ds, model=m, method=meth, inst=i,
                                          conf=float(conf[i]),
                                          instab_seed=float(instab[i]),
                                          top5_seed=float(inst_mean["top5"][i])))
                row = dict(ds=ds, model=m, method=meth, n_inst=n)
                for k in METRICS:
                    row[f"{k}_mean"] = float(inst_mean[k].mean())
                    row[f"{k}_std"] = float(inst_mean[k].std())
                from scipy.stats import spearmanr
                rho = spearmanr(conf, instab).statistic
                row["rho_conf_instab"] = float(rho) if np.isfinite(rho) else np.nan
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "crossseed.csv"), index=False)
    pd.DataFrame(inst_rows).to_csv(os.path.join(RESULTS, "conf_instab_inst.csv"), index=False)

def mcdropout_link():
    rows = []
    pr = pd.read_csv(os.path.join(RESULTS, "perturb_raw.csv"))
    pr = pr[(pr.model == "mlp") & (pr.method == "ig") & (pr.sigma == 0.1) & (pr.preserved == 1)]
    inst_tab = pd.read_csv(os.path.join(RESULTS, "conf_instab_inst.csv"))
    from scipy.stats import spearmanr
    for ds in DATASETS:
        stds = []
        for s in SEEDS:
            f = os.path.join(UNC, f"mc_{ds}_{s}.npz")
            if not os.path.exists(f):
                continue
            stds.append(np.load(f)["probs"].std(axis=0))
        if not stds:
            continue
        pred_std = np.mean(np.stack(stds), axis=0)     # [n_eval]
        sub = inst_tab[(inst_tab.ds == ds) & (inst_tab.model == "mlp") & (inst_tab.method == "ig")]
        seed_instab = sub.set_index("inst")["instab_seed"]
        pert = pr[pr.ds == ds].groupby("inst")["cosine"].mean().rsub(1)
        common_idx = sorted(set(seed_instab.index) & set(pert.index))
        x_mc = pred_std[common_idx]
        r1 = spearmanr(x_mc, seed_instab.loc[common_idx]).statistic
        r2 = spearmanr(x_mc, pert.loc[common_idx]).statistic
        rows.append(dict(ds=ds, n=len(common_idx),
                         rho_mc_seed_instab=float(r1), rho_mc_pert_instab=float(r2),
                         pred_std_mean=float(pred_std.mean())))
        np.savez_compressed(os.path.join(UNC, f"mclink_{ds}.npz"),
                            pred_std=x_mc,
                            seed_instab=seed_instab.loc[common_idx].values,
                            pert_instab=pert.loc[common_idx].values)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "mcdropout_link.csv"), index=False)

def spurious():
    rows = []
    perf = pd.read_csv(os.path.join(RESULTS, "perf_raw.csv"))
    for ds in DATASETS:
        d, meta = load_dataset(ds)
        D = meta["d_orig"] + 1
        spur_j = D - 1
        for m in MODELS_LIST:
            accs_c = perf[(perf.ds == ds) & (perf.model == m) & (perf.variant == "clean")].acc
            accs_s = perf[(perf.ds == ds) & (perf.model == m) & (perf.variant == "spur")].acc
            rels = []
            for s in SEEDS:
                f = os.path.join(UNC, f"reliance_{ds}_{m}_{s}.npz")
                if os.path.exists(f):
                    rels.append(np.load(f)["reliance"])
            rel = np.mean(np.stack(rels), axis=0) if rels else None
            for meth in METHODS_FOR[m]:
                if meth == "kshap":
                    continue
                sal_shares, ranks, top3f, top5f = [], [], [], []
                for s in SEEDS:
                    pk = load_base(ds, m, meth, s, "spur")
                    if pk is None:
                        continue
                    attr = pk[0]
                    ma = np.abs(attr).mean(axis=0)
                    sal_shares.append(ma[spur_j] / ma.sum())
                    ranks.append(int((np.argsort(-ma) == spur_j).nonzero()[0][0]) + 1)
                    tk = np.argsort(-np.abs(attr), axis=1)
                    top3f.append(np.mean([spur_j in tk[i, :3] for i in range(attr.shape[0])]))
                    top5f.append(np.mean([spur_j in tk[i, :5] for i in range(attr.shape[0])]))
                if not sal_shares:
                    continue
                row = dict(ds=ds, model=m, method=meth,
                           acc_clean=float(accs_c.mean()), acc_spur=float(accs_s.mean()),
                           acc_gap=float(accs_c.mean() - accs_s.mean()),
                           sal_share=float(np.mean(sal_shares)),
                           sal_share_std=float(np.std(sal_shares)),
                           attr_rank=float(np.mean(ranks)),
                           top3_frac=float(np.mean(top3f)), top5_frac=float(np.mean(top5f)))
                if rel is not None:
                    row["rel_spur"] = float(rel[spur_j])
                    row["rel_share"] = float(rel[spur_j] / rel.sum())
                    row["rel_rank"] = int(np.argsort(-rel).tolist().index(spur_j)) + 1
                rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "spurious.csv"), index=False)

def _rank_biserial(d):
    """Matched-pairs rank-biserial correlation (Kerby 2014) from paired diffs d."""
    from scipy.stats import rankdata
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    tp = r[d > 0].sum(); tm = r[d < 0].sum()
    tot = tp + tm
    return float((tp - tm) / tot) if tot > 0 else 0.0

def _boot_ci(d, rng, B=2000):
    d = np.asarray(d, dtype=float)
    n = len(d)
    est = np.array([_rank_biserial(d[rng.integers(0, n, n)]) for _ in range(B)])
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def wilcoxon_tests():
    """Paired fast-vs-LIME Wilcoxon at sigma=0.25 on the prediction-preserving
    instances explained by both methods (the full LIME budget; no instance cap).
    Reports per-cell n, two-sided p, matched-pairs rank-biserial r with a 95%
    bootstrap CI, and Holm-Bonferroni and Benjamini-Hochberg adjusted p over the
    full family of tests."""
    from scipy.stats import wilcoxon
    raw = pd.read_csv(os.path.join(RESULTS, "perturb_raw.csv"))
    kept = raw[(raw.preserved == 1) & (raw.sigma == 0.25)]
    rng = np.random.default_rng(0)
    rows = []
    for ds in DATASETS:
        for m in MODELS_LIST:
            fm = FAST_METHOD[m]
            a = kept[(kept.ds == ds) & (kept.model == m) & (kept.method == fm)]
            b = kept[(kept.ds == ds) & (kept.model == m) & (kept.method == "lime")]
            if a.empty or b.empty:
                continue
            am = a.groupby("inst")[["top5", "cosine"]].mean()
            bm = b.groupby("inst")[["top5", "cosine"]].mean()
            idx = am.index.intersection(bm.index)   # common pred-preserving instances
            if len(idx) < 10:
                continue
            for metric in ("top5", "cosine"):
                x, y = am.loc[idx, metric].values, bm.loc[idx, metric].values
                d = x - y
                if np.allclose(x, y):
                    stat, p = np.nan, 1.0
                else:
                    stat, p = wilcoxon(x, y)
                r = _rank_biserial(d)
                lo, hi = _boot_ci(d, rng)
                rows.append(dict(ds=ds, model=m, metric=metric, method_a=fm, method_b="lime",
                                 mean_a=float(x.mean()), mean_b=float(y.mean()),
                                 n=int(len(idx)), W=float(stat) if np.isfinite(stat) else np.nan,
                                 p=float(p), r=float(r), r_ci_low=lo, r_ci_high=hi))
    df = pd.DataFrame(rows)
    pvals = df["p"].values
    mtest = len(pvals)
    order = np.argsort(pvals)
    holm = np.empty(mtest); bh = np.empty(mtest)
    run = 0.0
    for rank, i in enumerate(order):                       # Holm step-down
        run = max(run, (mtest - rank) * pvals[i]); holm[i] = min(run, 1.0)
    runbh = 1.0
    for rank in range(mtest - 1, -1, -1):                  # Benjamini-Hochberg step-up
        i = order[rank]
        runbh = min(runbh, pvals[i] * mtest / (rank + 1)); bh[i] = min(runbh, 1.0)
    df["p_holm"] = holm; df["p_bh"] = bh
    df["sig_bonf"] = df["p"] < 0.05 / mtest
    df["sig_holm"] = df["p_holm"] < 0.05
    df["sig_bh"] = df["p_bh"] < 0.05
    df.to_csv(os.path.join(RESULTS, "wilcoxon.csv"), index=False)

def summary():
    out = {}
    out["perf"] = pd.read_csv(os.path.join(RESULTS, "perf_summary.csv")).to_dict("records")
    out["damage_pooled"] = pd.read_csv(os.path.join(RESULTS, "damage_curves_pooled.csv")).to_dict("records")
    out["flip_rates"] = pd.read_csv(os.path.join(RESULTS, "flip_rates.csv")).to_dict("records")
    out["crossseed"] = pd.read_csv(os.path.join(RESULTS, "crossseed.csv")).to_dict("records")
    out["mcdropout"] = pd.read_csv(os.path.join(RESULTS, "mcdropout_link.csv")).to_dict("records")
    out["spurious"] = pd.read_csv(os.path.join(RESULTS, "spurious.csv")).to_dict("records")
    out["wilcoxon"] = pd.read_csv(os.path.join(RESULTS, "wilcoxon.csv")).to_dict("records")
    json.dump(out, open(os.path.join(RESULTS, "results_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("perf", "all"): perf_tables()
    if step in ("damage", "all"): damage_curves()
    if step in ("crossseed", "all"): crossseed_and_conf()
    if step in ("mc", "all"): mcdropout_link()
    if step in ("spur", "all"): spurious()
    if step in ("wilcoxon", "all"): wilcoxon_tests()
    if step in ("summary", "all"): summary()
    print("DONE", step)
