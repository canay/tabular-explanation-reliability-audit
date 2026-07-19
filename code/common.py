"""Shared utilities: datasets, models, NumPy MLP, explainers, metrics, chunked runner."""
import os, sys, json, time, pickle, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
EXPL = os.path.join(ROOT, "expl")
PERT = os.path.join(ROOT, "perturb")
UNC = os.path.join(ROOT, "unc")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
for d in (DATA, MODELS, EXPL, PERT, UNC, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

DATASETS = ["adult", "credit_g", "bank_marketing", "ccdefault", "electricity"]
MODELS_LIST = ["logreg", "rf", "hgb", "mlp"]
SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]
N_EVAL = 500          # eval instances for base explanations / cross-seed
N_PERT_FAST = 200     # perturbation instances, fast methods
N_PERT_LIME = 50      # perturbation instances, LIME
R_FAST = 5            # perturbation replicates fast methods
R_LIME = 5
LIME_BG = 2000        # training rows used as LIME background / sampling stats
N_LIME_BASE = 100     # LIME base explanation instances
N_KSHAP = 100          # KernelSHAP instances (MLP, cross-seed only)
LIME_NSAMPLES = 1000
KSHAP_NSAMPLES = 500
SPUR_DELTA = 1.5
SPUR_NOISE = 0.5
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", "34"))
T0 = time.time()

def time_left():
    return TIME_BUDGET - (time.time() - T0)

def shard_of(name, nshards):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % nshards

def parse_shard():
    # --shard i/n
    for a in sys.argv[1:]:
        if a.startswith("--shard"):
            v = a.split("=")[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            i, n = v.split("/")
            return int(i), int(n)
    return 0, 1

# ---------------- datasets ----------------

def load_dataset(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(DATA, f"{name}_meta.json")))
    return d, meta

def model_path(ds, model, seed, variant):
    return os.path.join(MODELS, f"{ds}_{model}_{seed}_{variant}.pkl")

# ---------------- NumPy MLP ----------------

class NumpyMLP:
    """Two-hidden-layer MLP, ReLU, inverted dropout, Adam, BCE-with-logits."""
    def __init__(self, dim_in, hidden=(64, 32), dropout=0.2, lr=1e-3,
                 epochs=40, batch=256, seed=0):
        self.dim_in = dim_in; self.hidden = hidden; self.dropout = dropout
        self.lr = lr; self.epochs = epochs; self.batch = batch; self.seed = seed
        rng = np.random.default_rng(seed)
        dims = [dim_in, hidden[0], hidden[1], 1]
        self.W = [rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i+1]))
                  for i in range(3)]
        self.b = [np.zeros(dims[i+1]) for i in range(3)]

    def _fwd(self, X, drop_rng=None):
        cache = {}
        z1 = X @ self.W[0] + self.b[0]; a1 = np.maximum(z1, 0)
        if drop_rng is not None:
            m1 = (drop_rng.random(a1.shape) >= self.dropout) / (1 - self.dropout)
            a1 = a1 * m1; cache["m1"] = m1
        z2 = a1 @ self.W[1] + self.b[1]; a2 = np.maximum(z2, 0)
        if drop_rng is not None:
            m2 = (drop_rng.random(a2.shape) >= self.dropout) / (1 - self.dropout)
            a2 = a2 * m2; cache["m2"] = m2
        z3 = a2 @ self.W[2] + self.b[2]
        cache.update(z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, X=X)
        return z3[:, 0], cache

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed + 1000)
        n = X.shape[0]
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t = 0; b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                bi = idx[s:s + self.batch]
                Xb, yb = X[bi], y[bi]
                logit, c = self._fwd(Xb, drop_rng=rng)
                p = 1 / (1 + np.exp(-logit))
                dz3 = ((p - yb) / len(bi))[:, None]
                gW3 = c["a2"].T @ dz3; gb3 = dz3.sum(0)
                da2 = dz3 @ self.W[2].T
                if "m2" in c: da2 = da2 * c["m2"]
                dz2 = da2 * (c["z2"] > 0)
                gW2 = c["a1"].T @ dz2; gb2 = dz2.sum(0)
                da1 = dz2 @ self.W[1].T
                if "m1" in c: da1 = da1 * c["m1"]
                dz1 = da1 * (c["z1"] > 0)
                gW1 = Xb.T @ dz1; gb1 = dz1.sum(0)
                grads = [(gW1, gb1), (gW2, gb2), (gW3, gb3)]
                t += 1
                for i, (gw, gb) in enumerate(grads):
                    mW[i] = b1 * mW[i] + (1 - b1) * gw
                    vW[i] = b2 * vW[i] + (1 - b2) * gw * gw
                    mb[i] = b1 * mb[i] + (1 - b1) * gb
                    vb[i] = b2 * vb[i] + (1 - b2) * gb * gb
                    self.W[i] -= self.lr * (mW[i] / (1 - b1**t)) / (np.sqrt(vW[i] / (1 - b2**t)) + eps)
                    self.b[i] -= self.lr * (mb[i] / (1 - b1**t)) / (np.sqrt(vb[i] / (1 - b2**t)) + eps)
        return self

    def decision_function(self, X):
        return self._fwd(np.asarray(X, dtype=float))[0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def grad_input(self, X):
        """d logit / d x, deterministic (no dropout)."""
        logit, c = self._fwd(np.asarray(X, dtype=float))
        dz3 = np.ones((X.shape[0], 1))
        da2 = dz3 @ self.W[2].T
        dz2 = da2 * (c["z2"] > 0)
        da1 = dz2 @ self.W[1].T
        dz1 = da1 * (c["z1"] > 0)
        return dz1 @ self.W[0].T

    def mc_proba(self, X, T=30, seed=0):
        rng = np.random.default_rng(seed)
        out = np.empty((T, X.shape[0]))
        for t in range(T):
            logit, _ = self._fwd(np.asarray(X, dtype=float), drop_rng=rng)
            out[t] = 1 / (1 + np.exp(-logit))
        return out

def integrated_gradients(mlp, X, baseline, m=32):
    X = np.asarray(X, dtype=float)
    diff = X - baseline[None, :]
    acc = np.zeros_like(X)
    for i in range(m):
        alpha = (i + 0.5) / m
        acc += mlp.grad_input(baseline[None, :] + alpha * diff)
    return diff * acc / m

# ---------------- models ----------------

def make_model(model, seed, dim_in):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    if model == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=10,
                                      min_samples_leaf=10, random_state=seed, n_jobs=2)
    if model == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=True,
                                              validation_fraction=0.15,
                                              random_state=seed)
    if model == "mlp":
        return NumpyMLP(dim_in, seed=seed)
    raise ValueError(model)

# ---------------- attribution aggregation ----------------

def aggregate(attr_enc, groups):
    """Sum signed attributions of encoded columns into original features.
    groups: list (len D_orig) of lists of encoded col indices."""
    out = np.zeros((attr_enc.shape[0], len(groups)))
    for j, cols in enumerate(groups):
        out[:, j] = attr_enc[:, cols].sum(axis=1)
    return out

# ---------------- explainers ----------------

def explain_batch(method, model_name, mdl, X_enc, ds_arrays, meta, lime_state=0):
    """Return aggregated attributions [n, D_orig] for encoded rows X_enc."""
    groups = meta["groups"]
    if method == "linshap":
        import shap
        bg = ds_arrays["X_train"][:200]
        ex = shap.LinearExplainer(mdl, shap.maskers.Independent(bg, max_samples=200))
        return aggregate(np.asarray(ex.shap_values(X_enc)), groups)
    if method == "treeshap":
        import shap
        ex = shap.TreeExplainer(mdl)
        sv = ex.shap_values(X_enc, check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, D, classes)
            sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, -1]
        return aggregate(sv, groups)
    if method == "ig":
        baseline = ds_arrays["X_train"].mean(axis=0)
        return aggregate(integrated_gradients(mdl, X_enc, baseline), groups)
    if method == "lime":
        from lime.lime_tabular import LimeTabularExplainer
        expl = LimeTabularExplainer(
            ds_arrays["X_train"][:LIME_BG], mode="classification",
            categorical_features=meta["cat_cols_enc"],
            discretize_continuous=True, random_state=lime_state,
            feature_selection="none")
        D_enc = X_enc.shape[1]
        out = np.zeros((X_enc.shape[0], len(groups)))
        pf = mdl.predict_proba
        for i in range(X_enc.shape[0]):
            e = expl.explain_instance(X_enc[i], pf, num_features=D_enc,
                                      num_samples=LIME_NSAMPLES, labels=(1,))
            w = np.zeros(D_enc)
            for fi, val in e.as_map()[1]:
                w[fi] = val
            out[i] = aggregate(w[None, :], groups)[0]
        return out
    if method == "kshap":
        import shap
        bg = shap.kmeans(ds_arrays["X_train"][:LIME_BG], 25)
        f = lambda X: mdl.predict_proba(X)[:, 1]
        ex = shap.KernelExplainer(f, bg)
        sv = ex.shap_values(X_enc, nsamples=KSHAP_NSAMPLES, silent=True)
        return aggregate(np.asarray(sv), groups)
    raise ValueError(method)

METHODS_FOR = {
    "logreg": ["linshap", "lime"],
    "rf": ["treeshap", "lime"],
    "hgb": ["treeshap", "lime"],
    "mlp": ["ig", "lime", "kshap"],
}
FAST_METHOD = {"logreg": "linshap", "rf": "treeshap", "hgb": "treeshap", "mlp": "ig"}

# ---------------- metrics ----------------

def topk_overlap(a, b, k):
    ta = set(np.argsort(-np.abs(a))[:k]); tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / k

def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))

def sign_agree_top5(a, b):
    top = np.argsort(-np.abs(a))[:5]
    return float(np.mean(np.sign(a[top]) == np.sign(b[top])))

def pair_metrics(a, b):
    return dict(top3=topk_overlap(a, b, 3), top5=topk_overlap(a, b, 5),
                spearman=spearman(a, b), cosine=cosine(a, b),
                sign5=sign_agree_top5(a, b))

def ece(y_true, p_pos, n_bins=15):
    conf = np.where(p_pos >= 0.5, p_pos, 1 - p_pos)
    pred = (p_pos >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    e, n = 0.0, len(y_true)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def perturb(X, sigma, num_cols_enc, rng):
    Xp = X.copy()
    if sigma > 0:
        Xp[:, num_cols_enc] += sigma * rng.standard_normal(Xp[:, num_cols_enc].shape)
    return Xp
