# Environment notes

The archived-artifact reproduction path was verified on 2026-07-18 and
re-verified on 2026-09-21 with Python 3.12.12 and the versions pinned in the
root `requirements.txt`. The 2026-09-21 run regenerated every summary artifact
byte-identically to the values reported in the manuscript.

The primary model-fitting and explanation run recorded Python 3.10.12 with
NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, SHAP 0.48.0, LIME 0.2.0.1,
pandas 2.3.3, and Matplotlib 3.10.9. These versions are pinned in
`requirements-full.txt`.

Some hardening controls were executed separately with scikit-learn 1.8.0. Their
frozen CSV/JSON outputs and timing/resource summaries are preserved under
`results/q1_hardening/`. The quick reproduction route reads those artifacts; it
does not silently rerun the hardening models under a different environment.

The NumPy MLP, Integrated Gradients, and Monte Carlo dropout implementations are
contained in `code/common.py`. All random seeds, perturbation levels, sample
budgets, and feature-grouping rules are defined in the public code.

