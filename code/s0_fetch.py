"""Stage 0: fetch one OpenML dataset (arg: name), preprocess, save npz + meta."""
import sys, os, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, N_EVAL, SPUR_DELTA, SPUR_NOISE

SPECS = {
    "adult": dict(openml_id=1590, max_rows=15000),
    "credit_g": dict(openml_id=31, max_rows=None),
    "bank_marketing": dict(openml_id=1461, max_rows=15000),
    "ccdefault": dict(openml_id=42477, max_rows=15000),
    "electricity": dict(openml_id=151, max_rows=15000),
}

def main(name):
    out_npz = os.path.join(DATA, f"{name}.npz")
    if os.path.exists(out_npz):
        print("already done"); return
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    spec = SPECS[name]
    ds = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
    X, y = ds.data, ds.target
    # binarize target
    y = pd.Series(pd.factorize(y)[0] if y.dtype.name == "category" else y)
    if name == "adult":
        y = (ds.target == ">50K").astype(int)
    elif name == "credit_g":
        y = (ds.target == "good").astype(int)
    elif name == "bank_marketing":
        y = (ds.target == "2").astype(int) if ds.target.dtype.name == "category" else (ds.target == 2).astype(int)
    elif name == "ccdefault":
        y = (ds.target.astype(str) == "1").astype(int)
    elif name == "electricity":
        y = (ds.target.astype(str) == "UP").astype(int)
    y = np.asarray(y, dtype=int)
    # drop rows with missing values
    mask = ~X.isna().any(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask.values]
    rng = np.random.default_rng(0)
    if spec["max_rows"] and len(X) > spec["max_rows"]:
        # stratified subsample
        idx = []
        for cls in (0, 1):
            ci = np.where(y == cls)[0]
            k = int(round(spec["max_rows"] * len(ci) / len(y)))
            idx.append(rng.choice(ci, size=k, replace=False))
        idx = np.sort(np.concatenate(idx))
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    num_cols = [c for c in X.columns if X[c].dtype.kind in "ifu"]
    cat_cols = [c for c in X.columns if c not in num_cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr[num_cols])
    def enc(df):
        parts, groups, names, cat_enc_idx = [], [], [], []
        Z = sc.transform(df[num_cols])
        col = 0
        for j, c in enumerate(num_cols):
            parts.append(Z[:, j:j+1]); groups.append([col]); names.append(c); col += 1
        for c in cat_cols:
            cats = sorted(X[c].astype(str).unique())
            M = np.zeros((len(df), len(cats)))
            vals = df[c].astype(str).values
            for k, cat in enumerate(cats):
                M[:, k] = (vals == cat)
            parts.append(M)
            groups.append(list(range(col, col + len(cats))))
            cat_enc_idx.extend(range(col, col + len(cats)))
            names.append(c); col += len(cats)
        return np.hstack(parts), groups, names, cat_enc_idx
    Xtr_enc, groups, names, cat_enc_idx = enc(Xtr)
    Xte_enc, _, _, _ = enc(Xte)
    num_cols_enc = [g[0] for g, n in zip(groups, names) if n in num_cols]
    # eval instances: stratified from test
    ev = []
    for cls in (0, 1):
        ci = np.where(yte == cls)[0]
        k = min(int(round(N_EVAL * len(ci) / len(yte))), len(ci))
        ev.append(rng.choice(ci, size=k, replace=False))
    eval_idx = np.sort(np.concatenate(ev))[:N_EVAL]
    # spurious feature (appended standardized column)
    xs_tr = (ytr * SPUR_DELTA + rng.normal(0, SPUR_NOISE, len(ytr)))
    xs_tr = (xs_tr - xs_tr.mean()) / xs_tr.std()
    xs_te = rng.standard_normal(len(yte))
    np.savez_compressed(out_npz,
        X_train=Xtr_enc.astype(np.float64), X_test=Xte_enc.astype(np.float64),
        y_train=ytr, y_test=yte, eval_idx=eval_idx,
        xs_train=xs_tr, xs_test=xs_te)
    meta = dict(name=name, openml_id=spec["openml_id"],
                n_total=int(len(X)), n_train=int(len(Xtr)), n_test=int(len(Xte)),
                d_orig=len(names), d_enc=int(Xtr_enc.shape[1]),
                names=names, groups=groups, num_cols=num_cols, cat_cols=cat_cols,
                num_cols_enc=num_cols_enc, cat_cols_enc=cat_enc_idx,
                pos_rate=float(y.mean()))
    json.dump(meta, open(os.path.join(DATA, f"{name}_meta.json"), "w"))
    print("saved", name, meta["n_train"], meta["n_test"], meta["d_orig"], meta["d_enc"])

if __name__ == "__main__":
    main(sys.argv[1])
