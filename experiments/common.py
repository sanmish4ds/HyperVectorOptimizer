import os
import sys
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

DATASETS = {
    # name: (N, d, generator seed)
    "syn-128": (1_000_000, 128, 11),
    "syn-768": (1_000_000, 768, 22),
    "arxiv-384": (1_000_000, 384, 33),
    "amazon-384": (990_000, 384, 44),
    "dev": (100_000, 128, 1),
}
# k and the plan set are selected per run through the environment:
#   HVO_K      top-k (default 10); k != 10 uses files suffixed _k{K}
#   HVO_PLANS  v2 (default): resumable post-filter + ACORN fused; v1: restart post-filter + faiss-filtered fused
K = int(os.environ.get("HVO_K", "10"))
KS = "" if K == 10 else f"_k{K}"
PLANSET = os.environ.get("HVO_PLANS", "v2")
PLAN_KEYS = {"v1": ("fused", "post"), "v2": ("acorn", "post_r")}[PLANSET]
QFILE, EFILE, PFILE = f"queries{KS}.pkl", f"est{KS}.pkl", f"plans{KS}.pkl"
N_TRAIN, N_TEST = 1000, 1500


def rfile(name):
    return os.path.join(RESULTS, f"{name}{KS}{'' if PLANSET == 'v2' else '_v1'}.json")


def ddir(name):
    p = os.path.join(DATA, name)
    os.makedirs(p, exist_ok=True)
    return p


def save(obj, name, fn):
    with open(os.path.join(ddir(name), fn), "wb") as f:
        pickle.dump(obj, f, protocol=5)


def load(name, fn):
    with open(os.path.join(ddir(name), fn), "rb") as f:
        return pickle.load(f)


def load_table(name):
    from hvo.data import Table
    meta = load(name, "table_meta.pkl")
    X = np.load(os.path.join(ddir(name), "X.npy"), mmap_mode=None)
    return Table(X=X, **meta)


def load_index(name):
    import faiss
    return faiss.read_index(os.path.join(ddir(name), "hnsw.faiss"))


def load_table_and_index(name):
    """Load the HNSW index and alias the table's vectors to the index's own storage (one copy in RAM)."""
    import faiss
    from hvo.data import Table
    idx = load_index(name)
    st = idx.storage
    N, d = st.ntotal, st.d
    X = faiss.rev_swig_ptr(faiss.downcast_index(st).get_xb(), N * d).reshape(N, d)
    meta = load(name, "table_meta.pkl")
    return Table(X=X, **meta), idx


def qerr(est, true, tau):
    """Q-error on local match counts, both floored at one tuple."""
    c = np.maximum(np.asarray(true) * tau, 1.0)
    e = np.maximum(np.asarray(est) * tau, 1.0)
    return np.maximum(c / e, e / c)


def rho_gls(sloc, sg):
    sl = np.clip(sloc, 1e-6, 1 - 1e-6); g = np.clip(sg, 1e-6, 1 - 1e-6)
    return np.tanh(0.5 * np.log(sl * (1 - g) / (g * (1 - sl))))


def regime(rho):
    return np.where(rho < -0.3, "anti", np.where(rho > 0.3, "pos", "indep"))
