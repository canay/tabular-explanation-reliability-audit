"""Stage 4: MC-dropout predictive uncertainty for the MLP on eval instances."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, SEEDS, UNC, load_dataset, model_path, time_left

T_MC = 50

def main():
    done, rest = 0, 0
    for ds in DATASETS:
        d, meta = load_dataset(ds)
        Xe = d["X_test"][d["eval_idx"]]
        for s in SEEDS:
            out = os.path.join(UNC, f"mc_{ds}_{s}.npz")
            if os.path.exists(out):
                continue
            if time_left() < 5:
                rest += 1
                continue
            mdl = pickle.load(open(model_path(ds, "mlp", s, "clean"), "rb"))
            probs = mdl.mc_proba(Xe, T=T_MC, seed=999 + s)
            np.savez_compressed(out, probs=probs)
            done += 1
    print(f"did {done}, remaining {rest}")
    if rest == 0:
        print("ALLDONE")

if __name__ == "__main__":
    main()
