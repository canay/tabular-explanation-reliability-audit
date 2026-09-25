"""Delta sweep for the LINEAR family without shap/sklearn.

For a logistic-regression model, exact interventional LinearSHAP is closed form,
phi_j(x) = w_j * (x_j - E[x_j]); the linear model is trained here with a pure
numpy L2 logistic solver so the sweep can run in the sandbox (no scipy/sklearn/
shap). Same schema as delta_sweep.py (model = "logreg"), so the two CSVs pool.
Trees (rf/hgb, TreeSHAP) still require shap+sklearn and stay a VPS task.
"""
import os, sys, csv, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from common import load_dataset, SPUR_NOISE

def inject_xs(y_train, y_test, delta, seed=0):
    rng = np.random.default_rng(10_000 + seed + int(round(delta * 1000)))
    xs_tr = y_train * delta + rng.normal(0, SPUR_NOISE, len(y_train))
    sd = xs_tr.std() or 1.0
    xs_tr = (xs_tr - xs_tr.mean()) / sd
    xs_te = rng.standard_normal(len(y_test))
    return xs_tr, xs_te

def fit_logreg(X, y, l2=1.0, iters=500, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0; yv = y.astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - yv
        w -= lr * (X.T @ g / n + (l2 / n) * w)
        b -= lr * g.mean()
    return w, b

def proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))

def reliance(w, b, Xte, groups, seed, n=2000):
    rng = np.random.default_rng(7 + seed); n = min(n, Xte.shape[0])
    idx = rng.choice(Xte.shape[0], size=n, replace=False); Xs = Xte[idx]
    p0 = proba(Xs, w, b); perm = rng.permutation(n)
    rel = np.zeros(len(groups))
    for j, cols in enumerate(groups):
        Xp = Xs.copy(); Xp[:, cols] = Xs[perm][:, cols]
        rel[j] = np.mean(np.abs(proba(Xp, w, b) - p0))
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="0.25,0.5,1.0,1.5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--datasets", default="adult,credit_g,bank_marketing")
    ap.add_argument("--neval", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "results", "delta_sweep_logreg.csv"))
    a = ap.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    datasets = a.datasets.split(",")
    cols = ["delta", "ds", "model", "seed", "clean_acc", "spur_acc", "gen_gap",
            "rel_share", "rel_rank", "sal_share", "attr_rank", "top5_frac"]
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["delta"], r["ds"], "logreg", str(r["seed"])))
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(cols)
    for delta in deltas:
        for ds in datasets:
            d, meta = load_dataset(ds)
            Xtr, Xte = d["X_train"], d["X_test"]; ytr, yte = d["y_train"], d["y_test"]
            eval_idx = d["eval_idx"][:a.neval]; groups = meta["groups"]; d_enc = meta["d_enc"]
            groups_aug = groups + [[d_enc]]
            xs_tr, xs_te = inject_xs(ytr, yte, delta, 0)
            Xtr_aug = np.column_stack([Xtr, xs_tr]); Xte_aug = np.column_stack([Xte, xs_te])
            mean_bg = Xtr_aug.mean(axis=0)
            for s in seeds:
                if (f"{delta}", ds, "logreg", str(s)) in done:
                    continue
                wc, bc = fit_logreg(Xtr, ytr)
                clean_acc = float(((proba(Xte, wc, bc) >= 0.5).astype(int) == yte).mean())
                wsp, bsp = fit_logreg(Xtr_aug, ytr)
                spur_acc = float(((proba(Xte_aug, wsp, bsp) >= 0.5).astype(int) == yte).mean())
                rel = reliance(wsp, bsp, Xte_aug, groups_aug, s)
                rel_share = rel[-1] / (rel.sum() or 1.0); rel_rank = int(1 + np.sum(rel > rel[-1]))
                Xe = Xte_aug[eval_idx]
                phi = wsp * (Xe - mean_bg)                # interventional LinearSHAP
                mabs = np.abs(phi).mean(axis=0)
                sal_share = mabs[-1] / (mabs.sum() or 1.0); attr_rank = int(1 + np.sum(mabs > mabs[-1]))
                top5 = np.mean([(phi.shape[1] - 1) in set(np.argsort(-np.abs(phi[i]))[:5])
                                for i in range(phi.shape[0])])
                w.writerow([delta, ds, "logreg", s, round(clean_acc, 4), round(spur_acc, 4),
                            round(clean_acc - spur_acc, 4), round(rel_share, 4), rel_rank,
                            round(sal_share, 4), attr_rank, round(float(top5), 4)])
                f.flush()
                print("done", delta, ds, "logreg", s, "clean", round(clean_acc, 3),
                      "sal_share", round(sal_share, 3), "rank", attr_rank, "top5", round(float(top5), 2))
    f.close(); print("ALLDONE")

if __name__ == "__main__":
    main()
