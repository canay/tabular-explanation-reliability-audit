"""Dataset-level robustness audit for the benchmark-calibrated reliability flag.

Uses only frozen per-instance artifacts; it does not train models or call an
explainer. Run from the project root. Outputs:

* analysis/results/decision_rule_by_dataset.csv
* analysis/results/decision_rule_cluster_ci.csv
* analysis/results/decision_rule_leave_one_dataset_out.csv

The cluster bootstrap resamples whole datasets. With five benchmark clusters,
the intervals are descriptive measures of between-dataset uncertainty, not
external-validation intervals.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KEY = ["ds", "model", "method", "inst"]
TAU = 0.80
# Threshold tolerance, 2026-09-21. Every top-5 overlap is an integer count
# over five, so an attainable mean can be mathematically equal to TAU while
# floating point stores it as 0.7999999999999999. Comparing raw therefore
# counted such a unit as below the threshold. TAU_TOL places a value equal
# to TAU on the stable side; it is far below the smallest nonzero distance
# between an attainable value and TAU (0.004 across the thresholds used).
TAU_TOL = 1e-9
BOOTSTRAP_REPLICATES = 20_000
SEED = 20260718


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    flag = frame["flag"].to_numpy(dtype=bool)
    unstable = frame["unreliable"].to_numpy(dtype=bool)
    tp = int(np.sum(flag & unstable))
    fp = int(np.sum(flag & ~unstable))
    fn = int(np.sum(~flag & unstable))
    tn = int(np.sum(~flag & ~unstable))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    rho = float(spearmanr(frame["cheap_overlap"], frame["top5_seed"]).statistic)
    return {
        "n": len(frame),
        "base_unreliable": float(np.mean(unstable)),
        "flag_rate": float(np.mean(flag)),
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "rho_cheap_vs_crossseed": rho,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_analysis_frame() -> pd.DataFrame:
    raw = pd.read_csv("results/perturb_raw.csv")
    cross_seed = pd.read_csv("results/conf_instab_inst.csv")
    sampling = (
        raw.loc[raw["sigma"].eq(0.0)]
        .groupby(KEY, as_index=False)["top5"]
        .mean()
        .rename(columns={"top5": "samp_overlap"})
    )
    perturbation = (
        raw.loc[raw["sigma"].eq(0.25) & raw["preserved"].eq(1)]
        .groupby(KEY, as_index=False)["top5"]
        .mean()
        .rename(columns={"top5": "pert_overlap"})
    )
    frame = sampling.merge(perturbation, on=KEY, validate="one_to_one")
    frame = frame.merge(
        cross_seed[KEY + ["top5_seed"]], on=KEY, validate="one_to_one"
    )
    frame["cheap_overlap"] = frame[["samp_overlap", "pert_overlap"]].min(axis=1)
    frame["flag"] = frame["cheap_overlap"].lt(TAU - TAU_TOL)
    frame["unreliable"] = frame["top5_seed"].lt(TAU - TAU_TOL)
    frame["group"] = np.where(frame["method"].eq("lime"), "LIME", "Model-matched")
    return frame


def cluster_bootstrap(frame: pd.DataFrame, group: str) -> list[dict[str, float | str | int]]:
    subset = frame.loc[frame["group"].eq(group)].copy()
    datasets = np.array(sorted(subset["ds"].unique()))
    clusters = {dataset: subset.loc[subset["ds"].eq(dataset)] for dataset in datasets}
    rng = np.random.default_rng(SEED + (0 if group == "LIME" else 1))
    counts = np.array(
        [
            [
                metric_row(clusters[dataset])[name]
                for name in ("tp", "fp", "fn", "tn")
            ]
            for dataset in datasets
        ],
        dtype=float,
    )
    sampled_indices = rng.integers(
        0, len(datasets), size=(BOOTSTRAP_REPLICATES, len(datasets))
    )
    sampled_counts = counts[sampled_indices].sum(axis=1)
    tp, fp, fn, tn = sampled_counts.T
    values = {
        "precision": np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0),
        "recall": np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0),
        "fpr": np.divide(fp, fp + tn, out=np.full_like(fp, np.nan), where=(fp + tn) > 0),
    }
    rows = []
    point = metric_row(subset)
    for metric, samples in values.items():
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "group": group,
                "metric": metric,
                "point_estimate": point[metric],
                "cluster_count": len(datasets),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "ci_low": float(np.percentile(finite, 2.5)),
                "ci_high": float(np.percentile(finite, 97.5)),
                "seed": SEED + (0 if group == "LIME" else 1),
            }
        )
    return rows


def main() -> None:
    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_analysis_frame()

    by_dataset = []
    for (group, dataset), subset in frame.groupby(["group", "ds"], sort=True):
        by_dataset.append({"group": group, "dataset": dataset, **metric_row(subset)})
    pd.DataFrame(by_dataset).to_csv(out_dir / "decision_rule_by_dataset.csv", index=False)

    ci_rows = []
    for group in sorted(frame["group"].unique()):
        ci_rows.extend(cluster_bootstrap(frame, group))
    pd.DataFrame(ci_rows).to_csv(out_dir / "decision_rule_cluster_ci.csv", index=False)

    leave_one_out = []
    for group, group_frame in frame.groupby("group", sort=True):
        for omitted in sorted(group_frame["ds"].unique()):
            retained = group_frame.loc[~group_frame["ds"].eq(omitted)]
            leave_one_out.append(
                {"group": group, "omitted_dataset": omitted, **metric_row(retained)}
            )
    pd.DataFrame(leave_one_out).to_csv(
        out_dir / "decision_rule_leave_one_dataset_out.csv", index=False
    )


if __name__ == "__main__":
    main()
