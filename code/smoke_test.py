import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NumpyMLP, integrated_gradients, explain_batch, make_model, pair_metrics, ece

rng = np.random.default_rng(0)
n, dnum, dcat = 600, 4, 2
Xnum = rng.standard_normal((n, dnum))
Xcat = np.zeros((n, 6))  # two cats with 3 levels
lv1 = rng.integers(0, 3, n); lv2 = rng.integers(0, 3, n)
for i in range(n):
    Xcat[i, lv1[i]] = 1; Xcat[i, 3 + lv2[i]] = 1
X = np.hstack([Xnum, Xcat])
y = ((Xnum[:, 0] + 0.5 * Xnum[:, 1] + 0.3 * (lv1 == 2) + 0.3 * rng.standard_normal(n)) > 0).astype(int)
groups = [[0], [1], [2], [3], [4, 5, 6], [7, 8, 9]]
meta = dict(groups=groups, num_cols_enc=[0, 1, 2, 3], cat_cols_enc=[4, 5, 6, 7, 8, 9],
            names=["n0", "n1", "n2", "n3", "c1", "c2"])
dv = dict(X_train=X, X_test=X)

for mname in ["logreg", "rf", "hgb", "mlp"]:
    mdl = make_model(mname, 0, X.shape[1])
    if mname == "mlp":
        mdl.epochs = 10
    t0 = time.time()
    mdl.fit(X, y)
    acc = (mdl.predict_proba(X)[:, 1] >= 0.5).astype(int).__eq__(y).mean()
    print(mname, "fit", round(time.time() - t0, 2), "acc", round(acc, 3))
    from common import METHODS_FOR
    for meth in METHODS_FOR[mname]:
        t0 = time.time()
        Xq = X[:8] if meth != "kshap" else X[:3]
        a = explain_batch(meth, mname, mdl, Xq, dv, meta)
        print("  ", meth, a.shape, "t", round(time.time() - t0, 2),
              "norm", round(float(np.abs(a).sum()), 3))
    if mname == "mlp":
        mc = mdl.mc_proba(X[:5], T=10, seed=0)
        print("  mc_proba", mc.shape, "std", round(float(mc.std(0).mean()), 4))
print("pair_metrics", pair_metrics(np.array([1., -2, 3, 0, 1, 0.5]), np.array([1., -2, 2.5, 0.1, 1, 0.4])))
print("ece", round(ece(y, np.clip(y * 0.8 + 0.1, 0, 1)), 3))
print("SMOKE OK")
