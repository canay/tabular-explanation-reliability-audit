# Tabular Explanation Reliability Audit

Public replication package for *Reliability Limits of Post-Hoc Explanations
for Tabular Models: Perturbation, Retraining, Sampling, and Shortcut Recovery*.

The study audits post-hoc attribution reliability across five public tabular
benchmarks and four model families. It separates prediction-preserving input
perturbation, model retraining, and explainer-sampling effects, and treats
shortcut recovery as a distinct faithfulness probe.

Author: **Özkan Canay**, Department of Information Systems and Technologies,
Faculty of Computer and Information Sciences, Sakarya University, Türkiye.

## Quick reproduction from archived artifacts

The repository includes the per-instance and cell-level result artifacts needed
to recompute the manuscript summaries, extended diagnostics, and figures without
retraining models or rerunning explainers.

```bash
python -m pip install -r requirements.txt
python reproduce.py --verify-inputs --all
```

`reproduce.py` is non-destructive: it does not delete data or model artifacts.
It verifies the frozen inputs, regenerates derived CSV/text summaries, and
rebuilds the empirical figures in `figures/`.

## Repository structure

| Path | Contents |
|---|---|
| `code/` | Data acquisition, training, explanation, perturbation, aggregation, and figure scripts |
| `results/` | Frozen main-study artifacts, including the 44.9 MB per-instance perturbation table |
| `results/q1_hardening/` | Negative/oracle controls, budget sweep, and synthetic ground-truth outputs |
| `analysis/scripts/` | Extended uncertainty, variance-decomposition, fragility, and candidate-rule analyses |
| `analysis/results/` | Frozen extended-analysis inputs and regenerated summaries |
| `unc/` | Compact frozen MC-dropout link artifacts required by the final uncertainty figure |
| `figures/` | Reproducible empirical figures in PDF and PNG |
| `provenance/` | Dataset metadata and hashes for frozen reproduction inputs |
| `environment/` | Full-run environment and provenance notes |

The manuscript source, bibliography, submission files, review logs, and private
workflow state are intentionally excluded. This repository is a replication
package, not a manuscript archive.

## Datasets

The acquisition script retrieves the datasets directly from OpenML and applies
the recorded 70/30 stratified split and preprocessing policy (seed 0).

| Dataset | OpenML ID |
|---|---:|
| Adult | 1590 |
| Bank Marketing | 1461 |
| German Credit | 31 |
| Credit Card Default | 42477 |
| Electricity | 151 |

No raw dataset is redistributed here. The metadata files under `provenance/`
record the encoded dimensions, feature groups, sample counts, and class rates.
Users should review the current OpenML dataset terms before downloading.

## Full experiment pipeline

The archived-artifact path above is the recommended verification route. A clean
from-scratch main-pipeline run can be launched in a separate workspace:

```bash
python -m pip install -r environment/requirements-full.txt
python run_full_pipeline.py --workspace full_run --execute
```

The runner refuses a non-empty workspace and never removes existing artifacts.
The complete run is computationally expensive. Q1 hardening controls are
implemented in `code/run_q1_hardening.py`; see `environment/ENVIRONMENT.md` for
the original execution split and version notes.

## Evidence map

| Manuscript evidence | Reproduction source |
|---|---|
| Predictive performance and calibration | `results/perf_summary.csv` |
| Prediction-preserving damage curves and flip rates | `results/perturb_raw.csv`, `results/damage_curves_pooled.csv`, `results/flip_rates.csv` |
| Wilcoxon tests, multiplicity corrections, and effect sizes | `results/wilcoxon.csv` |
| Cross-seed instability and confidence diagnostics | `results/crossseed.csv`, `results/conf_instab_inst.csv` |
| Shortcut reliance and attribution recovery | `results/spurious.csv`, `results/q1_hardening/` |
| Broader uncertainty and variance decomposition | `analysis/results/broader_uncertainty.csv`, `analysis/results/variance_decomp.csv` |
| LIME budget and perturbation-strength sweeps | `results/q1_hardening/lime_budget_sweep_summary.csv`, `analysis/results/delta_sweep_agg.csv` |
| Candidate report-or-flag rule | `analysis/results/decision_rule*.csv`, `analysis/scripts/decision_rule_robustness.py` |

The candidate reliability rule is benchmark-calibrated, not externally
validated. Dataset-level false-positive rates and whole-dataset cluster
bootstrap intervals are included so pooled performance is not mistaken for a
deployment guarantee.

## License

Code and package-authored documentation are released under the MIT License.
Third-party datasets retain their own terms and are fetched from OpenML rather
than redistributed.
