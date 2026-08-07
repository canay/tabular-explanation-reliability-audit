"""Reference-target sensitivity check for the clean control-recovery arm.

The clean variant of the control-recovery analysis has no behavioural ground
truth, so it scores every explanation source against a fixed placeholder target.
This script quantifies how much the resulting null depends on which placeholder
is chosen. It reads only the archived base-explanation artifacts written by
s2_base_explain.py and does not refit any model or recompute any attribution.

Target definitions
------------------
positional  first five original features in dataset column order (the placeholder
            used by run_q1_hardening.py)
random      five original feature positions drawn at random per cell
self_top5   top five features of the same cell's own real salience
loso_top5   top five features of the real salience pooled over the other seeds

Only the first two are reported in the manuscript. The last two are retained
here because they document why a salience-derived target cannot serve as a
control: self_top5 is satisfied by construction, and loso_top5 restates the
cross-seed agreement that Section 5.3 already reports.

Metrics and control vectors are identical to run_q1_hardening.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common

METHODS_FOR = common.METHODS_FOR
TARGETS = ("positional", "random", "self_top5", "loso_top5")
SOURCES = ("real", "random", "permuted", "constant", "oracle")


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    denom = float(y_true.sum())
    if denom == 0:
        return float("nan")
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].sum() / denom)


def control_vectors(scores: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> dict:
    d = scores.shape[0]
    return {
        "random": rng.random(d),
        "permuted": rng.permutation(scores),
        "constant": np.ones(d),
        "oracle": truth.astype(float),
    }


def top_k_mask(scores: np.ndarray, d: int, k: int = 5) -> np.ndarray:
    truth = np.zeros(d, dtype=int)
    truth[np.argsort(-scores)[: min(k, d)]] = 1
    return truth


def load_base_clean(ds: str, model: str, method: str, seed: int):
    path = Path(common.EXPL) / f"base_{ds}_{model}_{method}_{seed}_clean.npz"
    if not path.exists():
        return None
    return np.asarray(np.load(path)["attr"], dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(common.RESULTS) / "q1_target_sensitivity")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--datasets", default=",".join(common.DATASETS))
    parser.add_argument("--models", default=",".join(common.MODELS_LIST))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args(argv)

    datasets = [x for x in args.datasets.split(",") if x]
    models = [x for x in args.models.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rng_target = np.random.default_rng(args.seed + 1)
    rows = []

    for ds in datasets:
        _, meta = common.load_dataset(ds)
        d = len(meta["groups"])
        for model in models:
            for method in METHODS_FOR[model]:
                cell = {}
                for seed in seeds:
                    attr = load_base_clean(ds, model, method, seed)
                    if attr is None:
                        continue
                    cell[seed] = np.abs(attr).mean(axis=0)
                if not cell:
                    continue

                positional = np.zeros(d, dtype=int)
                positional[: min(5, d)] = 1
                random_truth = np.zeros(d, dtype=int)
                random_truth[rng_target.choice(d, size=min(5, d), replace=False)] = 1

                for seed, scores in cell.items():
                    others = [v for s, v in cell.items() if s != seed]
                    truths = {
                        "positional": positional,
                        "random": random_truth,
                        "self_top5": top_k_mask(scores, d),
                    }
                    if others:
                        truths["loso_top5"] = top_k_mask(np.mean(others, axis=0), d)

                    controls = control_vectors(scores, positional, rng)
                    for tname, truth in truths.items():
                        sources = {"real": scores}
                        sources.update(controls)
                        sources["oracle"] = truth.astype(float)
                        for sname, vec in sources.items():
                            absvec = np.abs(vec)
                            rows.append({
                                "target": tname,
                                "source": sname,
                                "dataset": ds,
                                "model": model,
                                "method": method,
                                "seed": seed,
                                "ap": average_precision(truth, absvec),
                                "precision_at_5": precision_at_k(truth, absvec, 5),
                                "recall_at_5": recall_at_k(truth, absvec, 5),
                            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "target_sensitivity.csv", index=False)

    summary = (df.groupby(["target", "source"], dropna=False)
                 .agg(ap_mean=("ap", "mean"),
                      p_at_5=("precision_at_5", "mean"),
                      recall_at_5=("recall_at_5", "mean"),
                      n=("ap", "size"))
                 .reset_index())
    summary.to_csv(args.out_dir / "target_sensitivity_summary.csv", index=False)

    paired = []
    key = ["dataset", "model", "method", "seed"]
    for tname in TARGETS:
        sub = df[df.target == tname]
        if sub.empty:
            continue
        real = sub[sub.source == "real"].set_index(key)
        rand = sub[sub.source == "random"].set_index(key)
        idx = real.index.intersection(rand.index)
        paired.append({
            "target": tname,
            "delta_ap_mean": float((real.loc[idx, "ap"] - rand.loc[idx, "ap"]).mean()),
            "delta_p_at_5_mean": float(
                (real.loc[idx, "precision_at_5"] - rand.loc[idx, "precision_at_5"]).mean()),
            "n_pairs": int(len(idx)),
        })
    pd.DataFrame(paired).to_csv(args.out_dir / "target_sensitivity_paired.csv", index=False)

    manifest = {
        "analysis": "clean-arm reference-target sensitivity",
        "reads": "expl/base_{dataset}_{model}_{method}_{seed}_clean.npz",
        "refits_models": False,
        "recomputes_attributions": False,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "targets": list(TARGETS),
        "sources": list(SOURCES),
        "seed": args.seed,
        "rows": int(len(df)),
    }
    (args.out_dir / "target_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print()
    print(pd.DataFrame(paired).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
