"""Q1 hardening experiments for SCI-f01 explanation reliability.

This script adds control baselines, controlled ground-truth recovery, runtime
measurements, and a bounded LIME budget audit on top of the existing fh1
artifacts. It does not rewrite the manuscript and it does not treat smoke-test
outputs as paper evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import common  # noqa: E402


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mem_snapshot() -> dict:
    out: dict[str, float | int | str] = {
        "time_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
    }
    try:
        out["loadavg"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        vals = {}
        for line in text.splitlines():
            key, rest = line.split(":", 1)
            vals[key] = int(rest.strip().split()[0])
        out["mem_total_kb"] = vals.get("MemTotal", 0)
        out["mem_available_kb"] = vals.get("MemAvailable", 0)
    except Exception:
        pass
    return out


class Recorder:
    def __init__(self, out_dir: Path, log_dir: Path):
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timing_path = self.out_dir / "timings.jsonl"
        self.log_path = self.log_dir / "q1_hardening.log"

    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def timing(self, row: dict) -> None:
        with self.timing_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    @contextmanager
    def stage(self, name: str, **meta):
        start = time.perf_counter()
        row = {"stage": name, "start_utc": utc_now(), **meta}
        try:
            yield
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            raise
        finally:
            row["end_utc"] = utc_now()
            row["wall_seconds"] = time.perf_counter() - start
            self.timing(row)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def rank_of_target(scores: np.ndarray, target: int) -> int:
    order = np.argsort(-scores)
    return int(np.where(order == target)[0][0]) + 1


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


def control_vectors(scores: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    d = scores.shape[0]
    return {
        "random": rng.random(d),
        "permuted": rng.permutation(scores),
        "constant": np.ones(d),
        "oracle": truth.astype(float),
    }


def spur_view(arrays, meta):
    d2 = {
        "X_train": np.column_stack([arrays["X_train"], arrays["xs_train"]]),
        "X_test": np.column_stack([arrays["X_test"], arrays["xs_test"]]),
    }
    m2 = dict(meta)
    m2["groups"] = meta["groups"] + [[meta["d_enc"]]]
    m2["names"] = meta["names"] + ["synthetic_spurious"]
    m2["num_cols_enc"] = meta["num_cols_enc"] + [meta["d_enc"]]
    return d2, m2


def load_base(ds: str, model: str, method: str, seed: int, variant: str):
    path = Path(common.EXPL) / f"base_{ds}_{model}_{method}_{seed}_{variant}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return np.asarray(z["attr"], dtype=float), np.asarray(z["probs"], dtype=float)


def score_recovery_rows(label: str, scores: np.ndarray, truth: np.ndarray, target: int | None, meta: dict) -> list[dict]:
    rows = []
    abs_scores = np.abs(scores)
    row = {
        **meta,
        "source": label,
        "ap": average_precision(truth, abs_scores),
        "precision_at_1": precision_at_k(truth, abs_scores, 1),
        "precision_at_3": precision_at_k(truth, abs_scores, 3),
        "precision_at_5": precision_at_k(truth, abs_scores, 5),
        "recall_at_5": recall_at_k(truth, abs_scores, 5),
        "recall_at_10": recall_at_k(truth, abs_scores, 10),
    }
    if target is not None:
        row["target_rank"] = rank_of_target(abs_scores, target)
        row["target_top1"] = float(row["target_rank"] <= 1)
        row["target_top3"] = float(row["target_rank"] <= 3)
        row["target_top5"] = float(row["target_rank"] <= 5)
        total = abs_scores.sum()
        row["target_salience_share"] = float(abs_scores[target] / total) if total > 0 else float("nan")
    rows.append(row)
    return rows


def existing_artifact_controls(args, rec: Recorder) -> dict:
    rng = np.random.default_rng(args.seed)
    rows = []
    stab_rows = []
    summary = {}
    with rec.stage("existing_artifact_controls"):
        for ds in args.datasets:
            arrays, meta0 = common.load_dataset(ds)
            for model in args.models:
                for method in common.METHODS_FOR[model]:
                    for variant in ("clean", "spur"):
                        packs = []
                        meta = dict(meta0)
                        target = None
                        if variant == "spur":
                            _, meta = spur_view(arrays, meta0)
                            target = len(meta["groups"]) - 1
                        truth = np.zeros(len(meta["groups"]), dtype=int)
                        if variant == "spur":
                            truth[target] = 1
                        else:
                            # For clean artifacts there is no single causal target; use the
                            # top five mean real-salience features as a descriptive pseudo-target
                            # only for control calibration, not as a causal claim.
                            truth[: min(5, len(truth))] = 1
                        seed_scores = []
                        for seed in args.seeds:
                            loaded = load_base(ds, model, method, seed, variant)
                            if loaded is None:
                                continue
                            attr, _ = loaded
                            scores = np.abs(attr).mean(axis=0)
                            seed_scores.append(scores)
                            base_meta = {
                                "dataset": ds,
                                "model": model,
                                "method": method,
                                "seed": seed,
                                "variant": variant,
                                "truth_type": "spurious_feature" if variant == "spur" else "descriptive_top5",
                            }
                            rows.extend(score_recovery_rows("real", scores, truth, target, base_meta))
                            for cname, cscore in control_vectors(scores, truth, rng).items():
                                rows.extend(score_recovery_rows(cname, cscore, truth, target, base_meta))
                        if len(seed_scores) >= 2:
                            for source in ("real", "random", "permuted", "constant", "oracle"):
                                src_scores = []
                                for scores in seed_scores:
                                    if source == "real":
                                        src_scores.append(scores)
                                    else:
                                        src_scores.append(control_vectors(scores, truth, rng)[source])
                                for i in range(len(src_scores)):
                                    for j in range(i + 1, len(src_scores)):
                                        pm = common.pair_metrics(src_scores[i], src_scores[j])
                                        stab_rows.append({
                                            "dataset": ds,
                                            "model": model,
                                            "method": method,
                                            "variant": variant,
                                            "source": source,
                                            "seed_a": args.seeds[i],
                                            "seed_b": args.seeds[j],
                                            **pm,
                                        })
        recovery = pd.DataFrame(rows)
        stability = pd.DataFrame(stab_rows)
        recovery.to_csv(args.out_dir / "control_recovery.csv", index=False)
        stability.to_csv(args.out_dir / "cross_seed_control_stability.csv", index=False)
        if not recovery.empty:
            control_summary = recovery.groupby(["variant", "source"], dropna=False).agg(
                ap_mean=("ap", "mean"),
                p_at_5=("precision_at_5", "mean"),
                recall_at_5=("recall_at_5", "mean"),
                target_top5=("target_top5", "mean"),
                n=("ap", "size"),
            ).reset_index()
            control_summary.to_csv(args.out_dir / "control_recovery_summary.csv", index=False)
            summary["control_recovery_rows"] = int(len(recovery))
        if not stability.empty:
            stability.groupby(["variant", "source"], dropna=False).agg(
                top5_mean=("top5", "mean"),
                cosine_mean=("cosine", "mean"),
                n=("top5", "size"),
            ).reset_index().to_csv(args.out_dir / "cross_seed_control_stability_summary.csv", index=False)
            summary["stability_rows"] = int(len(stability))
    return summary


def run_lime_budget_sweep(args, rec: Recorder) -> dict:
    rows = []
    skipped = []
    try:
        import lime  # noqa: F401
    except Exception as exc:
        rec.log(f"LIME budget sweep skipped because lime is unavailable: {exc!r}")
        pd.DataFrame([{"stage": "lime_budget_sweep", "status": "skipped", "reason": repr(exc)}]).to_csv(
            args.out_dir / "lime_budget_sweep.csv", index=False
        )
        return {"lime_budget_rows": 0, "lime_budget_skipped": repr(exc)}

    with rec.stage("lime_budget_sweep"):
        for ds in args.datasets:
            arrays0, meta0 = common.load_dataset(ds)
            for model in args.lime_models:
                fast_method = common.FAST_METHOD[model]
                for seed in args.lime_seeds:
                    for variant in ("clean", "spur"):
                        if variant == "spur":
                            arrays, meta = spur_view(arrays0, meta0)
                        else:
                            arrays = {"X_train": arrays0["X_train"], "X_test": arrays0["X_test"]}
                            meta = meta0
                        mdl_path = common.model_path(ds, model, seed, variant)
                        if not Path(mdl_path).exists():
                            skipped.append({"dataset": ds, "model": model, "seed": seed, "variant": variant, "reason": "missing_model"})
                            continue
                        mdl = pickle.load(open(mdl_path, "rb"))
                        eval_idx = arrays0["eval_idx"][: args.lime_instances]
                        x_eval = arrays["X_test"][eval_idx]
                        fast_loaded = load_base(ds, model, fast_method, seed, variant)
                        fast_attr = None
                        if fast_loaded is not None:
                            fast_attr = fast_loaded[0][: len(eval_idx)]
                        target = len(meta["groups"]) - 1 if variant == "spur" else None
                        for budget in args.lime_budgets:
                            old_budget = common.LIME_NSAMPLES
                            old_bg = common.LIME_BG
                            common.LIME_NSAMPLES = int(budget)
                            common.LIME_BG = min(common.LIME_BG, arrays["X_train"].shape[0])
                            start = time.perf_counter()
                            try:
                                attr = common.explain_batch(
                                    "lime", model, mdl, x_eval, arrays, meta, lime_state=5000 + seed + int(budget)
                                )
                                status = "ok"
                                err = ""
                            except Exception as exc:
                                attr = np.zeros((len(x_eval), len(meta["groups"])))
                                status = "error"
                                err = repr(exc)
                            finally:
                                common.LIME_NSAMPLES = old_budget
                                common.LIME_BG = old_bg
                            wall = time.perf_counter() - start
                            score = np.abs(attr).mean(axis=0)
                            row = {
                                "dataset": ds,
                                "model": model,
                                "seed": seed,
                                "variant": variant,
                                "budget": int(budget),
                                "n_instances": int(len(x_eval)),
                                "wall_seconds": float(wall),
                                "seconds_per_instance": float(wall / max(len(x_eval), 1)),
                                "status": status,
                                "error": err,
                            }
                            if fast_attr is not None and status == "ok":
                                n = min(len(fast_attr), len(attr))
                                pm = [common.pair_metrics(fast_attr[i], attr[i]) for i in range(n)]
                                row["fast_top5_mean"] = float(np.mean([p["top5"] for p in pm]))
                                row["fast_cosine_mean"] = float(np.mean([p["cosine"] for p in pm]))
                            if target is not None and status == "ok":
                                truth = np.zeros(len(meta["groups"]), dtype=int)
                                truth[target] = 1
                                row["spurious_rank"] = rank_of_target(score, target)
                                row["spurious_top5"] = float(row["spurious_rank"] <= 5)
                                row["spurious_salience_share"] = float(score[target] / score.sum()) if score.sum() > 0 else float("nan")
                            rows.append(row)
        df = pd.DataFrame(rows)
        if skipped:
            pd.DataFrame(skipped).to_csv(args.out_dir / "lime_budget_skipped.csv", index=False)
        df.to_csv(args.out_dir / "lime_budget_sweep.csv", index=False)
        if not df.empty:
            df.groupby(["model", "variant", "budget"], dropna=False).agg(
                seconds_per_instance=("seconds_per_instance", "median"),
                fast_top5=("fast_top5_mean", "mean"),
                fast_cosine=("fast_cosine_mean", "mean"),
                spurious_top5=("spurious_top5", "mean"),
                n=("status", "size"),
            ).reset_index().to_csv(args.out_dir / "lime_budget_sweep_summary.csv", index=False)
    return {"lime_budget_rows": len(rows), "lime_budget_skipped": len(skipped)}


def model_explanation(model_name: str, mdl, x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if model_name == "logreg":
        return np.abs(np.asarray(mdl.coef_[0], dtype=float)), "coefficient_abs"
    if model_name == "rf" and hasattr(mdl, "feature_importances_"):
        return np.asarray(mdl.feature_importances_, dtype=float), "gini_importance"
    if model_name == "mlp" and hasattr(mdl, "grad_input"):
        return np.abs(mdl.grad_input(x_test[:200])).mean(axis=0), "gradient_abs"
    from sklearn.inspection import permutation_importance

    start_n = min(600, len(x_test))
    pi = permutation_importance(
        mdl,
        x_test[:start_n],
        y_test[:start_n],
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
        scoring="roc_auc",
    )
    return np.maximum(pi.importances_mean, 0.0), "permutation_auc_drop"


def synthetic_ground_truth(args, rec: Recorder) -> dict:
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows = []
    rng = np.random.default_rng(args.seed + 100)
    with rec.stage("synthetic_ground_truth"):
        for seed in args.synthetic_seeds:
            x, y = make_classification(
                n_samples=args.synthetic_samples,
                n_features=20,
                n_informative=5,
                n_redundant=3,
                n_repeated=0,
                n_clusters_per_class=2,
                flip_y=0.02,
                class_sep=1.2,
                shuffle=False,
                random_state=seed,
            )
            truth = np.zeros(x.shape[1], dtype=int)
            truth[:8] = 1
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.35, stratify=y, random_state=seed
            )
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            for model_name in args.synthetic_models:
                mdl = common.make_model(model_name, seed, x_train.shape[1])
                t0 = time.perf_counter()
                mdl.fit(x_train, y_train)
                train_wall = time.perf_counter() - t0
                prob = mdl.predict_proba(x_test)[:, 1]
                acc = accuracy_score(y_test, prob >= 0.5)
                auc = roc_auc_score(y_test, prob)
                t1 = time.perf_counter()
                scores, explainer = model_explanation(model_name, mdl, x_train, x_test, y_test, seed)
                explain_wall = time.perf_counter() - t1
                meta = {
                    "dataset": "synthetic_known_features",
                    "model": model_name,
                    "seed": seed,
                    "explainer": explainer,
                    "train_wall_seconds": float(train_wall),
                    "explain_wall_seconds": float(explain_wall),
                    "accuracy": float(acc),
                    "auc": float(auc),
                }
                rows.extend(score_recovery_rows("real", scores, truth, None, meta))
                for cname, cscore in control_vectors(scores, truth, rng).items():
                    rows.extend(score_recovery_rows(cname, cscore, truth, None, meta))
        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / "synthetic_ground_truth.csv", index=False)
        if not df.empty:
            df.groupby(["model", "source"], dropna=False).agg(
                ap=("ap", "mean"),
                p_at_5=("precision_at_5", "mean"),
                recall_at_10=("recall_at_10", "mean"),
                train_wall=("train_wall_seconds", "mean"),
                explain_wall=("explain_wall_seconds", "mean"),
                n=("ap", "size"),
            ).reset_index().to_csv(args.out_dir / "synthetic_ground_truth_summary.csv", index=False)
    return {"synthetic_rows": len(rows)}


def write_benchmark_table(out_dir: Path) -> None:
    parts = []
    for name in [
        "control_recovery.csv",
        "cross_seed_control_stability.csv",
        "lime_budget_sweep.csv",
        "synthetic_ground_truth.csv",
    ]:
        path = out_dir / name
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "artifact", name)
            parts.append(df)
    if parts:
        pd.concat(parts, ignore_index=True, sort=False).to_csv(out_dir / "benchmark_results.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--datasets", default="adult,credit_g,bank_marketing")
    p.add_argument("--models", default="logreg,rf,hgb,mlp")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--lime-models", default="logreg,rf,hgb")
    p.add_argument("--lime-seeds", default="0,1")
    p.add_argument("--lime-budgets", default="250,1000,3000")
    p.add_argument("--lime-instances", type=int, default=20)
    p.add_argument("--synthetic-seeds", default="0,1,2,3,4")
    p.add_argument("--synthetic-models", default="logreg,rf,hgb,mlp")
    p.add_argument("--synthetic-samples", type=int, default=3500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.base_dir = args.base_dir.resolve()
    args.out_dir = args.out_dir or (args.base_dir / "results" / "q1_hardening")
    args.log_dir = args.log_dir or (args.base_dir / "logs")
    args.datasets = [x for x in args.datasets.split(",") if x]
    args.models = [x for x in args.models.split(",") if x]
    args.seeds = [int(x) for x in args.seeds.split(",") if x]
    args.lime_models = [x for x in args.lime_models.split(",") if x]
    args.lime_seeds = [int(x) for x in args.lime_seeds.split(",") if x]
    args.lime_budgets = [int(x) for x in args.lime_budgets.split(",") if x]
    args.synthetic_seeds = [int(x) for x in args.synthetic_seeds.split(",") if x]
    args.synthetic_models = [x for x in args.synthetic_models.split(",") if x]
    if args.smoke:
        args.datasets = args.datasets[:1]
        args.models = args.models[:2]
        args.seeds = args.seeds[:2]
        args.lime_models = args.lime_models[:1]
        args.lime_seeds = args.lime_seeds[:1]
        args.lime_budgets = args.lime_budgets[:1]
        args.lime_instances = 3
        args.synthetic_seeds = args.synthetic_seeds[:1]
        args.synthetic_models = args.synthetic_models[:2]
        args.synthetic_samples = 400
    return args


def main() -> None:
    args = parse_args()
    os.chdir(args.base_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args.out_dir, args.log_dir)
    rec.log("Starting SCI-f01 Q1 hardening benchmark.")
    (args.out_dir / "resource_report_start.json").write_text(
        json.dumps(mem_snapshot(), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "run_started_utc": utc_now(),
        "base_dir": str(args.base_dir),
        "datasets": args.datasets,
        "models": args.models,
        "seeds": args.seeds,
        "lime_models": args.lime_models,
        "lime_budgets": args.lime_budgets,
        "synthetic_models": args.synthetic_models,
        "synthetic_seeds": args.synthetic_seeds,
        "smoke": bool(args.smoke),
    }
    (args.out_dir / "q1_hardening_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary: dict[str, int | str | float] = {}
    summary.update(existing_artifact_controls(args, rec))
    summary.update(run_lime_budget_sweep(args, rec))
    summary.update(synthetic_ground_truth(args, rec))
    write_benchmark_table(args.out_dir)
    summary["run_completed_utc"] = utc_now()
    summary["benchmark_results_exists"] = (args.out_dir / "benchmark_results.csv").exists()
    (args.out_dir / "hardening_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "resource_report_end.json").write_text(
        json.dumps(mem_snapshot(), indent=2, sort_keys=True), encoding="utf-8"
    )
    rec.log(f"SCI-f01 Q1 hardening benchmark completed: {summary}")


if __name__ == "__main__":
    main()
