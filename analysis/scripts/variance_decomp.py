"""Probe-conditioned top-5 instability for the pooled method groups (Table 2).

Uses only frozen result CSVs (no retraining). For each method group it reports
the mean top-5 instability (1 - mean top-5 overlap) under three probes:
  sampling      explainer self-disagreement at sigma=0 (identical input)
  perturbation  prediction-preserving input noise at sigma=0.25
  retraining    cross-seed disagreement (different fitted instance)
The three values are probe-conditioned totals measured on different analysis
sets; they are not additive components of one total instability. The
stochastic-trainer and per-family rows come from variance_decomp_families.py.
"""
import os, csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "analysis", "results", "variance_decomp.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def col(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    raise KeyError(names)

# pooled damage curves: model,method,sigma,top5_mean,...
damage = list(csv.DictReader(open(os.path.join(RES, "damage_curves_pooled.csv"))))
def damage_top5(method_filter, sigma):
    vals = []
    for r in damage:
        if abs(float(r["sigma"]) - sigma) < 1e-9 and method_filter(r["method"]):
            vals.append(col(r, "top5_mean"))
    return sum(vals) / len(vals) if vals else float("nan")

# cross-seed: ds,model,method,top5_mean(or top5),...
cross = list(csv.DictReader(open(os.path.join(RES, "crossseed.csv"))))
def cross_top5(method_filter):
    vals = []
    for r in cross:
        if method_filter(r.get("method", "")):
            try:
                vals.append(col(r, "top5_mean", "top5"))
            except KeyError:
                pass
    return sum(vals) / len(vals) if vals else float("nan")

fast = lambda m: m in ("linshap", "treeshap", "ig")
lime = lambda m: m == "lime"

rows = []
for name, filt in [("Model-matched (fast)", fast), ("LIME", lime)]:
    sampling = 1 - damage_top5(filt, 0.0)
    perturb = 1 - damage_top5(filt, 0.25)
    retrain = 1 - cross_top5(filt)
    rows.append((name, sampling, perturb, retrain))
    print(f"{name:22} sampling={sampling:.3f} perturbation={perturb:.3f} retraining={retrain:.3f}")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source_group", "sampling_instab", "perturbation_instab", "retraining_instab"])
    for name, s, p, r in rows:
        w.writerow([name, round(s, 4), round(p, 4), round(r, 4)])
print("wrote", OUT)
