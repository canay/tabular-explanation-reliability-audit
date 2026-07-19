"""Aggregate the delta-sweep cells into the per (delta, model) table used by the
manuscript (Table 9). Means over datasets and seeds. Run from project root:
    python analysis/scripts/delta_agg.py [path-to-delta_sweep.csv]
Writes analysis/results/delta_sweep_agg.csv
"""
import sys, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "analysis/results/delta_sweep.csv"
df = pd.read_csv(src)
g = (df.groupby(["delta", "model"])
       .agg(n=("seed", "size"), gen_gap=("gen_gap", "mean"), rel_rank=("rel_rank", "mean"),
            sal_share=("sal_share", "mean"), attr_rank=("attr_rank", "mean"),
            top5_frac=("top5_frac", "mean"))
       .reset_index().round(3))
# order models logreg, rf, hgb, mlp for the table
order = {"logreg": 0, "rf": 1, "hgb": 2, "mlp": 3}
g["_o"] = g.model.map(order); g = g.sort_values(["delta", "_o"]).drop(columns="_o")
g.to_csv("analysis/results/delta_sweep_agg.csv", index=False)
print(g.to_string(index=False))
