"""Stage 3: explanation stability under prediction-preserving perturbation.
Fast units = (ds, model, seed, sigma) all reps vectorized.
LIME units = (ds, model, seed, sigma, rep)."""
import os, sys, json, pickle, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, MODELS_LIST, SEEDS, SIGMAS, PERT, EXPL, FAST_METHOD,
                    N_PERT_FAST, N_PERT_LIME, R_FAST, R_LIME, load_dataset,
                    model_path, explain_batch, pair_metrics, perturb, time_left,
                    parse_shard, shard_of)

def pert_rng(ds, seed, sigma, rep):
    h = int(hashlib.md5(f"{ds}|{seed}|{sigma}|{rep}".encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(h)

def fast_file(ds, m, s, sig):
    return os.path.join(PERT, f"fast_{ds}_{m}_{s}_sig{sig}.csv")

def lime_file(ds, m, s, sig, rep):
    return os.path.join(PERT, f"lime_{ds}_{m}_{s}_sig{sig}_r{rep}.csv")

HEADER = "ds,model,method,seed,sigma,rep,inst,preserved,top3,top5,spearman,cosine,sign5,conf_base\n"

def write_rows(path, rows):
    import shutil
    from common import ROOT
    parts_dir = os.path.join(ROOT, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    tmp = os.path.join(parts_dir, os.path.basename(path) + ".tmp")
    with open(tmp, "w") as f:
        f.write(HEADER)
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    shutil.copyfile(tmp, path)

def run_fast(ds, m, s, sig, cache):
    d, meta = cache[ds]
    meth = FAST_METHOD[m]
    eval_idx = d["eval_idx"]
    Xe = d["X_test"][eval_idx[:N_PERT_FAST]]
    dv = dict(X_train=d["X_train"], X_test=d["X_test"])
    mdl = pickle.load(open(model_path(ds, m, s, "clean"), "rb"))
    base = np.load(os.path.join(EXPL, f"base_{ds}_{m}_{meth}_{s}_clean.npz"))
    base_attr, base_probs = base["attr"][:N_PERT_FAST], base["probs"][:N_PERT_FAST]
    base_lab = (base_probs >= 0.5).astype(int)
    R = 1 if sig == 0 else R_FAST
    rows = []
    for rep in range(R):
        rng = pert_rng(ds, s, sig, rep)
        Xp = perturb(Xe, sig, meta["num_cols_enc"], rng)
        lab = (mdl.predict_proba(Xp)[:, 1] >= 0.5).astype(int)
        keep = lab == base_lab
        attr = explain_batch(meth, m, mdl, Xp, dv, meta)
        for i in range(len(Xe)):
            if keep[i]:
                pm = pair_metrics(base_attr[i], attr[i])
                rows.append([ds, m, meth, s, sig, rep, i, 1,
                             round(pm["top3"], 4), round(pm["top5"], 4),
                             round(pm["spearman"], 4), round(pm["cosine"], 4),
                             round(pm["sign5"], 4), round(float(base_probs[i]), 4)])
            else:
                rows.append([ds, m, meth, s, sig, rep, i, 0, "", "", "", "", "",
                             round(float(base_probs[i]), 4)])
    write_rows(fast_file(ds, m, s, sig), rows)

def run_lime(ds, m, s, sig, rep, cache):
    d, meta = cache[ds]
    eval_idx = d["eval_idx"]
    Xe = d["X_test"][eval_idx[:N_PERT_FAST]]   # same perturbation draw as fast
    dv = dict(X_train=d["X_train"], X_test=d["X_test"])
    mdl = pickle.load(open(model_path(ds, m, s, "clean"), "rb"))
    base = np.load(os.path.join(EXPL, f"base_{ds}_{m}_lime_{s}_clean.npz"))
    base_attr, base_probs = base["attr"][:N_PERT_LIME], base["probs"][:N_PERT_LIME]
    base_lab = (base_probs >= 0.5).astype(int)
    rng = pert_rng(ds, s, sig, rep)
    Xp_full = perturb(Xe, sig, meta["num_cols_enc"], rng)[:N_PERT_LIME]
    lab = (mdl.predict_proba(Xp_full)[:, 1] >= 0.5).astype(int)
    keep = lab == base_lab
    attr = np.zeros((N_PERT_LIME, len(meta["groups"])))
    if keep.any():
        attr[keep] = explain_batch("lime", m, mdl, Xp_full[keep], dv, meta,
                                   lime_state=2000 + s * 17 + rep)
    rows = []
    for i in range(N_PERT_LIME):
        if keep[i]:
            pm = pair_metrics(base_attr[i], attr[i])
            rows.append([ds, m, "lime", s, sig, rep, i, 1,
                         round(pm["top3"], 4), round(pm["top5"], 4),
                         round(pm["spearman"], 4), round(pm["cosine"], 4),
                         round(pm["sign5"], 4), round(float(base_probs[i]), 4)])
        else:
            rows.append([ds, m, "lime", s, sig, rep, i, 0, "", "", "", "", "",
                         round(float(base_probs[i]), 4)])
    write_rows(lime_file(ds, m, s, sig, rep), rows)

def main():
    si, sn = parse_shard()
    fast_units = [("fast", ds, m, s, sig) for ds in DATASETS for m in MODELS_LIST
                  for s in SEEDS for sig in SIGMAS]
    lime_units = [("lime", ds, m, s, sig, rep) for ds in DATASETS for m in MODELS_LIST
                  for s in SEEDS for sig in SIGMAS
                  for rep in range(2 if sig == 0 else R_LIME)]
    units = fast_units + lime_units
    def ufile(u):
        return fast_file(*u[1:]) if u[0] == "fast" else lime_file(*u[1:])
    pending = [u for u in units if shard_of("_".join(map(str, u)), sn) == si
               and not os.path.exists(ufile(u))]
    cache = {}
    done = 0
    for u in pending:
        need = 14 if u[0] == "fast" else 10
        if time_left() < need:
            break
        ds = u[1]
        if ds not in cache:
            cache[ds] = load_dataset(ds)
        try:
            if u[0] == "fast":
                run_fast(*u[1:], cache)
            else:
                run_lime(*u[1:], cache)
            done += 1
        except Exception as e:
            print("ERROR", u, repr(e))
            raise
    rest = len(pending) - done
    print(f"SHARD {si}/{sn}: did {done}, remaining {rest}")
    if rest == 0:
        print("ALLDONE")

if __name__ == "__main__":
    main()
