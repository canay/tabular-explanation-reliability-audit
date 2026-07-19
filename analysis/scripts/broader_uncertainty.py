"""Q1-audit A-010: broaden the uncertainty proxy beyond MC dropout.

For the clean MLPs this computes two additional per-instance uncertainty signals
and correlates each (Spearman, with a Fisher-z 95% CI) with Integrated Gradients
perturbation instability at sigma=0.1, so RQ3 is no longer based on MC dropout
alone:
  ens_var   deep-ensemble predictive variance over the five seed MLPs
  entropy   predictive entropy of the ensemble-mean probability
IG instability per instance = 1 - mean cosine(IG_perturbed, IG_unperturbed) at
sigma=0.1, read from results/perturb_raw.csv. Uses existing models; no retraining.
"""
import os, sys, csv, math, pickle
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from common import DATASETS, SEEDS, load_dataset, model_path
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "analysis", "results", "broader_uncertainty.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def _rankdata(a):
    a = np.asarray(a, float); order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float); ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    sa = a[order]; i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
        i = j + 1
    return ranks

def spearman(x, y):
    rx = _rankdata(x) - np.mean(_rankdata(x)); ry = _rankdata(y) - np.mean(_rankdata(y))
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0

def fisher_ci(r, n):
    if n < 5 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r)); se = 1 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))

# IG perturbation instability per (ds, inst) at sigma=0.1 (mean 1-cosine over seeds/reps)
ig = defaultdict(list)
for r in csv.DictReader(open(os.path.join(RES, "perturb_raw.csv"))):
    if r["method"] != "ig":
        continue
    try:
        if abs(float(r["sigma"]) - 0.1) > 1e-9:
            continue
        if str(r.get("preserved", "")).strip().lower() not in ("true", "1", "t", "yes"):
            continue
        ig[(r["ds"], int(r["inst"]))].append(1.0 - float(r["cosine"]))
    except Exception:
        pass

rows = [["ds", "estimator", "n", "rho", "ci_lo", "ci_hi"]]
for ds in DATASETS:
    d, meta = load_dataset(ds)
    Xe = d["X_test"][d["eval_idx"]]
    probs = []
    for s in SEEDS:
        m = pickle.load(open(model_path(ds, "mlp", s, "clean"), "rb"))
        probs.append(m.predict_proba(Xe)[:, 1])
    P = np.vstack(probs)                       # (5, n_eval)
    ens_var = P.var(axis=0)
    pbar = P.mean(axis=0).clip(1e-6, 1 - 1e-6)
    entropy = -(pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar))
    # align with IG instability by instance position in eval_idx
    inst_ids = list(range(len(d["eval_idx"])))
    igv, ev, en = [], [], []
    for pos in inst_ids:
        key = (ds, pos)
        if key in ig and ig[key]:
            igv.append(np.mean(ig[key])); ev.append(ens_var[pos]); en.append(entropy[pos])
    n = len(igv)
    for est, vals in [("ens_var", ev), ("entropy", en)]:
        if n >= 5:
            r = spearman(vals, igv); lo, hi = fisher_ci(r, n)
        else:
            r, lo, hi = float("nan"), float("nan"), float("nan")
        rows.append([ds, est, n, round(r, 3), round(lo, 3), round(hi, 3)])
        print(ds, est, "n", n, "rho", round(r, 3), "CI", round(lo, 3), round(hi, 3))

with open(OUT, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("wrote", OUT)
