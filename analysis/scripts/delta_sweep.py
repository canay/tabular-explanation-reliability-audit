"""Q1-audit A-010: shortcut-recovery vs injected-signal strength (delta sweep).

Reuses code/common.py (datasets, models, explainers, metrics) and the cached
data/*.npz. For each delta it re-injects the spurious feature
    xs_train = standardize(delta * y_train + N(0, SPUR_NOISE^2))
    xs_test  = N(0, 1)            (pure noise at test time)
trains the spurious-variant model, and measures whether the MODEL-MATCHED
(fast) explainer still recovers the planted feature:
  gen_gap        clean_acc - spur_acc
  rel_share      behavioral reliance share of xs (group permutation importance)
  sal_share      mean |attribution| share of xs (fast method, eval instances)
  attr_rank      global rank of xs in mean |attribution| ordering (1 = top)
  top5_frac      fraction of eval instances with xs in the top-5

Resumable: appends one row per (delta, ds, model, seed) to the output CSV and
skips rows already present, so it can be re-invoked until complete under a
TIME_BUDGET. No OpenML access and no retraining of the clean/main models.
"""
import os, sys, csv, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from common import (load_dataset, make_model, model_path, explain_batch,
                    FAST_METHOD, SPUR_NOISE, NumpyMLP)

T0 = time.time()

def inject_xs(y_train, y_test, delta, seed):
    rng = np.random.default_rng(10_000 + seed + int(round(delta * 1000)))
    xs_tr = y_train * delta + rng.normal(0, SPUR_NOISE, len(y_train))
    sd = xs_tr.std() or 1.0
    xs_tr = (xs_tr - xs_tr.mean()) / sd
    xs_te = rng.standard_normal(len(y_test))
    return xs_tr, xs_te

def reliance(mdl, Xte_aug, groups_aug, seed, n=2000):
    rng = np.random.default_rng(7 + seed)
    n = min(n, Xte_aug.shape[0])
    idx = rng.choice(Xte_aug.shape[0], size=n, replace=False)
    Xs = Xte_aug[idx]
    p0 = mdl.predict_proba(Xs)[:, 1]
    perm = rng.permutation(n)
    rel = np.zeros(len(groups_aug))
    for j, cols in enumerate(groups_aug):
        Xp = Xs.copy(); Xp[:, cols] = Xs[perm][:, cols]
        rel[j] = np.mean(np.abs(mdl.predict_proba(Xp)[:, 1] - p0))
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="0.25,0.5,1.0,1.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--datasets", default="adult,credit_g,bank_marketing")
    ap.add_argument("--models", default="logreg,rf,hgb,mlp")
    ap.add_argument("--neval", type=int, default=100)
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "results", "delta_sweep.csv"))
    a = ap.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    datasets = a.datasets.split(","); models = a.models.split(",")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = ["delta", "ds", "model", "seed", "clean_acc", "spur_acc", "gen_gap",
            "rel_share", "rel_rank", "sal_share", "attr_rank", "top5_frac"]
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["delta"], r["ds"], r["model"], str(r["seed"])))
    new = not os.path.exists(a.out)
    fout = open(a.out, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(cols)
    cache = {}
    def accuracy_score(y, yp):
        import numpy as _np
        return float((_np.asarray(y) == _np.asarray(yp)).mean())
    n_done = 0
    for delta in deltas:
        for ds in datasets:
            if ds not in cache:
                cache[ds] = load_dataset(ds)
            d, meta = cache[ds]
            X_tr, X_te = d["X_train"], d["X_test"]
            y_tr, y_te = d["y_train"], d["y_test"]
            eval_idx = d["eval_idx"][:a.neval]
            groups = meta["groups"]
            d_enc = meta["d_enc"]
            groups_aug = groups + [[d_enc]]          # spurious col index = d_enc
            xs_tr, xs_te = inject_xs(y_tr, y_te, delta, 0)
            Xtr_aug = np.column_stack([X_tr, xs_tr])
            Xte_aug = np.column_stack([X_te, xs_te])
            ds_arrays = {"X_train": Xtr_aug}
            meta_aug = dict(meta); meta_aug["groups"] = groups_aug
            for model in models:
                key = (f"{delta}", ds, model, "")
                for s in seeds:
                    if (f"{delta}", ds, model, str(s)) in done:
                        continue
                    if time.time() - T0 > a.budget:
                        fout.flush(); fout.close()
                        print(f"TIMEUP after {n_done} cells; re-invoke to resume."); return
                    # clean acc from existing clean model (delta-independent)
                    cp = model_path(ds, model, s, "clean")
                    try:
                        import pickle
                        cm = pickle.load(open(cp, "rb"))
                        clean_acc = accuracy_score(y_te, cm.predict(X_te))
                    except Exception:
                        try:                       # no cached clean model: train one
                            cm = (NumpyMLP(X_tr.shape[1], seed=s) if model == "mlp"
                                  else make_model(model, s, X_tr.shape[1]))
                            cm.fit(X_tr, y_tr)
                            clean_acc = accuracy_score(y_te, cm.predict(X_te))
                        except Exception:
                            clean_acc = float("nan")
                    mdl = (NumpyMLP(Xtr_aug.shape[1], seed=s) if model == "mlp"
                           else make_model(model, s, Xtr_aug.shape[1]))
                    mdl.fit(Xtr_aug, y_tr)
                    spur_acc = accuracy_score(y_te, mdl.predict(Xte_aug))
                    rel = reliance(mdl, Xte_aug, groups_aug, s)
                    rel_share = rel[-1] / (rel.sum() or 1.0)
                    rel_rank = int(1 + np.sum(rel > rel[-1]))
                    Xe = Xte_aug[eval_idx]
                    attr = explain_batch(FAST_METHOD[model], model, mdl, Xe, ds_arrays, meta_aug)
                    mabs = np.abs(attr).mean(axis=0)
                    sal_share = mabs[-1] / (mabs.sum() or 1.0)
                    attr_rank = int(1 + np.sum(mabs > mabs[-1]))
                    top5 = np.mean([(attr.shape[1] - 1) in set(np.argsort(-np.abs(attr[i]))[:5])
                                    for i in range(attr.shape[0])])
                    w.writerow([delta, ds, model, s, round(clean_acc, 4), round(spur_acc, 4),
                                round(clean_acc - spur_acc, 4), round(rel_share, 4), rel_rank,
                                round(sal_share, 4), attr_rank, round(float(top5), 4)])
                    fout.flush()
                    n_done += 1
                    print("done", delta, ds, model, s, "sal_share", round(sal_share, 3),
                          "rank", attr_rank, "top5", round(float(top5), 2))
    fout.close()
    print(f"ALLDONE ({n_done} new cells)")

if __name__ == "__main__":
    main()
