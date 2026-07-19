"""Stage 2: base explanations on eval instances. Units = (ds, model, method, seed, variant).
LIME/KernelSHAP units are resumable at instance granularity via partial files."""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, DATASETS, MODELS_LIST, SEEDS, EXPL, METHODS_FOR, N_LIME_BASE,
                    N_KSHAP, load_dataset, model_path, explain_batch, time_left,
                    parse_shard, shard_of)

def spur_view(d, meta):
    d2 = dict(X_train=np.column_stack([d["X_train"], d["xs_train"]]),
              X_test=np.column_stack([d["X_test"], d["xs_test"]]))
    m2 = dict(meta)
    m2["groups"] = meta["groups"] + [[meta["d_enc"]]]
    m2["names"] = meta["names"] + ["synthetic_spurious"]
    m2["num_cols_enc"] = meta["num_cols_enc"] + [meta["d_enc"]]
    return d2, m2

def unit_file(ds, m, meth, s, v):
    return os.path.join(EXPL, f"base_{ds}_{m}_{meth}_{s}_{v}.npz")

def main():
    si, sn = parse_shard()
    units = []
    for ds in DATASETS:
        for m in MODELS_LIST:
            for meth in METHODS_FOR[m]:
                for s in SEEDS:
                    for v in ("clean", "spur"):
                        if meth == "kshap" and v == "spur":
                            continue
                        units.append((ds, m, meth, s, v))
    pending = [u for u in units if shard_of("_".join(map(str, u)), sn) == si
               and not os.path.exists(unit_file(*u))]
    cache = {}
    done = 0
    for ds, m, meth, s, v in pending:
        if time_left() < 5:
            break
        if ds not in cache:
            cache[ds] = load_dataset(ds)
        d0, meta0 = cache[ds]
        if v == "spur":
            dv, meta = spur_view(d0, meta0)
            dv = dict(dv)
        else:
            dv, meta = dict(X_train=d0["X_train"], X_test=d0["X_test"]), meta0
        eval_idx = d0["eval_idx"]
        n_inst = {"lime": N_LIME_BASE, "kshap": N_KSHAP}.get(meth, len(eval_idx))
        Xe = dv["X_test"][eval_idx[:n_inst]]
        mdl = pickle.load(open(model_path(ds, m, s, v), "rb"))
        probs = mdl.predict_proba(Xe)[:, 1]
        parts_dir = os.path.join(ROOT, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        part = os.path.join(parts_dir,
                            os.path.basename(unit_file(ds, m, meth, s, v)) + ".part.npz")
        if meth in ("lime", "kshap"):
            if os.path.exists(part):
                pz = np.load(part)
                attr, n_done = pz["attr"], int(pz["n_done"])
            else:
                attr, n_done = np.zeros((n_inst, len(meta["groups"]))), 0
            block = 10
            while n_done < n_inst and time_left() > (8 if meth == "lime" else 12):
                e = min(n_done + block, n_inst)
                attr[n_done:e] = explain_batch(meth, m, mdl, Xe[n_done:e], dv, meta,
                                               lime_state=1000 + s)
                n_done = e
                np.savez_compressed(part, attr=attr, n_done=n_done)
            if n_done < n_inst:
                print(f"PARTIAL {ds} {m} {meth} {s} {v}: {n_done}/{n_inst}")
                break
            try:
                os.remove(part)
            except OSError:
                pass
        else:
            attr = explain_batch(meth, m, mdl, Xe, dv, meta)
        np.savez_compressed(unit_file(ds, m, meth, s, v), attr=attr, probs=probs)
        done += 1
        print("explained", ds, m, meth, s, v)
    rest = len(pending) - done
    print(f"SHARD {si}/{sn}: did {done}, remaining {rest}")
    if rest == 0:
        print("ALLDONE")

if __name__ == "__main__":
    main()
# resync marker
