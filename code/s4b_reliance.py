"""Stage 4b: behavioral reliance scores for spurious-variant models.
For each original feature group (plus the synthetic spurious feature), permute its
encoded columns jointly on the test set and record mean |delta p|."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, MODELS_LIST, SEEDS, UNC, load_dataset, model_path,
                    time_left, parse_shard, shard_of)

def main():
    si, sn = parse_shard()
    units = [(ds, m, s) for ds in DATASETS for m in MODELS_LIST for s in SEEDS]
    out_of = lambda u: os.path.join(UNC, f"reliance_{u[0]}_{u[1]}_{u[2]}.npz")
    pending = [u for u in units if shard_of("_".join(map(str, u)), sn) == si
               and not os.path.exists(out_of(u))]
    cache = {}
    done = 0
    rng = np.random.default_rng(123)
    for ds, m, s in pending:
        if time_left() < 8:
            break
        if ds not in cache:
            cache[ds] = load_dataset(ds)
        d, meta = cache[ds]
        Xte = np.column_stack([d["X_test"], d["xs_test"]])
        groups = meta["groups"] + [[meta["d_enc"]]]
        mdl = pickle.load(open(model_path(ds, m, s, "spur"), "rb"))
        # subsample test rows for speed
        n = min(2000, Xte.shape[0])
        idx = rng.choice(Xte.shape[0], size=n, replace=False)
        Xs = Xte[idx]
        p0 = mdl.predict_proba(Xs)[:, 1]
        rel = np.zeros(len(groups))
        perm = np.random.default_rng(7 + s).permutation(n)
        for j, cols in enumerate(groups):
            Xp = Xs.copy()
            Xp[:, cols] = Xs[perm][:, cols]
            rel[j] = np.mean(np.abs(mdl.predict_proba(Xp)[:, 1] - p0))
        np.savez_compressed(out_of((ds, m, s)), reliance=rel)
        done += 1
        print("reliance", ds, m, s)
    rest = len(pending) - done
    print(f"SHARD {si}/{sn}: did {done}, remaining {rest}")
    if rest == 0:
        print("ALLDONE")

if __name__ == "__main__":
    main()
