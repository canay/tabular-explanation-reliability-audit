"""Stage 1: train models. Work units = (ds, model, seed, variant). Chunked + sharded."""
import os, sys, json, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, DATASETS, MODELS_LIST, SEEDS, RESULTS, load_dataset,
                    make_model, model_path, ece, time_left, parse_shard, shard_of)

def get_xy(d, meta, variant):
    Xtr, Xte = d["X_train"], d["X_test"]
    if variant == "spur":
        Xtr = np.column_stack([Xtr, d["xs_train"]])
        Xte = np.column_stack([Xte, d["xs_test"]])
    return Xtr, Xte, d["y_train"], d["y_test"]

def main():
    si, sn = parse_shard()
    units = [(ds, m, s, v) for ds in DATASETS for m in MODELS_LIST
             for s in SEEDS for v in ("clean", "spur")]
    pending = [u for u in units if not os.path.exists(model_path(*u))
               and shard_of("_".join(map(str, u)), sn) == si]
    done = 0
    cache = {}
    for ds, m, s, v in pending:
        if time_left() < 6:
            break
        if ds not in cache:
            cache[ds] = load_dataset(ds)
        d, meta = cache[ds]
        Xtr, Xte, ytr, yte = get_xy(d, meta, v)
        mdl = make_model(m, s, Xtr.shape[1])
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        fit_t = time.time() - t0
        p = mdl.predict_proba(Xte)[:, 1]
        from sklearn.metrics import accuracy_score, roc_auc_score
        row = dict(ds=ds, model=m, seed=s, variant=v,
                   acc=float(accuracy_score(yte, (p >= 0.5).astype(int))),
                   auc=float(roc_auc_score(yte, p)), ece=ece(yte, p),
                   fit_time=round(fit_t, 2))
        with open(os.path.join(RESULTS, "perf.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        pickle.dump(mdl, open(model_path(ds, m, s, v), "wb"))
        done += 1
        print("trained", ds, m, s, v, round(fit_t, 1), "s")
    rest = len(pending) - done
    print(f"SHARD {si}/{sn}: did {done}, remaining {rest}")
    if rest == 0:
        print("ALLDONE")

if __name__ == "__main__":
    main()
