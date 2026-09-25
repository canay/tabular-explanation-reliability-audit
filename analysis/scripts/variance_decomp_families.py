"""Probe-conditioned top-5 instability by method group and model family (Table 2).

Uses only frozen result CSVs (no retraining, no explainer calls) and the same
definitions as variance_decomp.py:
  sampling      1 - mean top-5 overlap at sigma=0 (identical input), damage_curves_pooled.csv
  perturbation  1 - mean top-5 overlap at sigma=0.25, damage_curves_pooled.csv
  retraining    1 - mean cross-seed top-5 overlap, crossseed.csv
Rows: all model-matched families pooled, the stochastic trainers pooled (random
forest, gradient boosting, MLP), one row per model-matched family, and LIME.
The three values are probe-conditioned totals on different analysis sets, not
additive components. The pooled model-matched and LIME rows must reproduce
analysis/results/variance_decomp.csv; the script stops if they do not.
"""
import os, csv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
POOLED = os.path.join(ROOT, "analysis", "results", "variance_decomp.csv")
OUT = os.path.join(ROOT, "analysis", "results", "variance_decomp_families.csv")

damage = list(csv.DictReader(open(os.path.join(RES, "damage_curves_pooled.csv"))))
cross = list(csv.DictReader(open(os.path.join(RES, "crossseed.csv"))))

FAST = ("linshap", "treeshap", "ig")
GROUPS = [
    ("Model-matched, all families", lambda m, e: e in FAST),
    ("Model-matched, stochastic trainers", lambda m, e: e in FAST and m in ("rf", "hgb", "mlp")),
    ("LinearSHAP, logistic regression", lambda m, e: m == "logreg" and e == "linshap"),
    ("TreeSHAP, random forest", lambda m, e: m == "rf" and e == "treeshap"),
    ("TreeSHAP, gradient boosting", lambda m, e: m == "hgb" and e == "treeshap"),
    ("IG, MLP", lambda m, e: m == "mlp" and e == "ig"),
    ("LIME", lambda m, e: e == "lime"),
]


def damage_top5(keep, sigma):
    vals = [float(r["top5_mean"]) for r in damage
            if abs(float(r["sigma"]) - sigma) < 1e-9 and keep(r["model"], r["method"])]
    return sum(vals) / len(vals), len(vals)


def cross_top5(keep):
    vals = [float(r["top5_mean"]) for r in cross
            if keep(r["model"], r["method"]) and r["top5_mean"] not in ("", None)]
    return sum(vals) / len(vals), len(vals)


rows = []
for name, keep in GROUPS:
    s, n_s = damage_top5(keep, 0.0)
    p, n_p = damage_top5(keep, 0.25)
    r, n_r = cross_top5(keep)
    assert n_s == n_p and n_s > 0 and n_r > 0, (name, n_s, n_p, n_r)
    rows.append((name, 1 - s, 1 - p, 1 - r, n_p, n_r))
    print(f"{name:36} sampling={1 - s:.3f} perturbation={1 - p:.3f} retraining={1 - r:.3f} "
          f"(damage rows {n_p}, cross-seed rows {n_r})")

# the pooled rows must reproduce the archived two-row summary
if os.path.exists(POOLED):
    archived = {row["source_group"]: row for row in csv.DictReader(open(POOLED))}
    for mine, theirs in [("Model-matched, all families", "Model-matched (fast)"), ("LIME", "LIME")]:
        got = next(r for r in rows if r[0] == mine)
        ref = archived[theirs]
        for value, key in zip(got[1:4], ("sampling_instab", "perturbation_instab", "retraining_instab")):
            if abs(round(value, 4) - float(ref[key])) > 1e-9:
                raise SystemExit(f"{mine} {key}: {value:.4f} does not reproduce {ref[key]}")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method_group", "sampling_instab", "perturbation_instab", "retraining_instab",
                "n_damage_rows", "n_crossseed_rows"])
    for name, s, p, r, n_p, n_r in rows:
        w.writerow([name, round(s, 4), round(p, 4), round(r, 4), n_p, n_r])
print("wrote", OUT)
