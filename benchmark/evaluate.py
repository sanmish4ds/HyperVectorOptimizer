"""Load a filtered-search benchmark release, score results, and run reference baselines.

Needs numpy and pandas (pyarrow); the hnsw baseline also needs faiss.

  python evaluate.py DIR --results ids.npy [--k 10] [--split test]    score your (Q, k) row-id array
  python evaluate.py DIR --baseline exact|hnsw-post [--ef 64]          run a reference method
  python evaluate.py DIR --check-gt [--n 100]                          recompute ground truth on a sample

Recall@k is reported overall, as the share of queries below 0.9, and per correlation regime:
rho = tanh(0.5 * logit(s_local) - 0.5 * logit(s_global)); anti < -0.3 < indep < 0.3 < pos.
Anti-correlated queries (matches far from the query) are where post-filtering loses recall.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, d):
        self.dir = d
        self.schema = json.load(open(os.path.join(d, "schema.json")))
        self.queries = pd.read_parquet(os.path.join(d, "queries.parquet"))
        self.filters = [json.loads(f) for f in self.queries["filter"]]
        self.attrs = pd.read_parquet(os.path.join(d, "base_attributes.parquet"))
        dk = self.schema["dimension_table"]
        self.fk_col = dk["foreign_key"]
        self.dim = pd.read_parquet(os.path.join(d, f"dim_{dk['name']}.parquet"))
        self._X = self._Q = None

    @property
    def X(self):
        if self._X is None:
            self._X = np.load(os.path.join(self.dir, "base_vectors.npy"), mmap_mode="r")
        return self._X

    @property
    def Q(self):
        if self._Q is None:
            self._Q = np.load(os.path.join(self.dir, "query_vectors.npy"))
        return self._Q

    def gt(self, k):
        return np.load(os.path.join(self.dir, f"gt_k{k}.npy"))

    def mask(self, i):
        """Boolean mask over base rows matching query i's filter."""
        f, A = self.filters[i], self.attrs
        m = np.ones(len(A), dtype=bool)
        for c, (lo, hi) in f["ranges"].items():
            v = A[c].to_numpy()
            m &= (v >= np.float32(lo)) & (v <= np.float32(hi))
        for c, vals in f["in"].items():
            m &= np.isin(A[c].to_numpy(), vals)
        if f["dim_in"]:
            dm = np.ones(len(self.dim), dtype=bool)
            for c, vals in f["dim_in"].items():
                dm &= np.isin(self.dim[c].to_numpy(), vals)
            m &= dm[A[self.fk_col].to_numpy()]
        return m

    def regime(self, k):
        q = self.queries
        sl = np.clip(q[f"s_local_k{k}"].to_numpy(), 1e-6, 1 - 1e-6)
        sg = np.clip(q["s_global"].to_numpy(), 1e-6, 1 - 1e-6)
        rho = np.tanh(0.5 * np.log(sl * (1 - sg) / (sg * (1 - sl))))
        return np.where(rho < -0.3, "anti", np.where(rho > 0.3, "pos", "indep"))


def recall(found, gt):
    """Per-query recall@k; queries with fewer than k matches are scored against the matches they have."""
    out = np.empty(len(gt))
    for i, (f, g) in enumerate(zip(found, gt)):
        g = g[g >= 0]
        out[i] = len(np.intersect1d(f[f >= 0], g)) / len(g) if len(g) else 1.0
    return out


def report(ds, found, k, split, seconds=None):
    sel = np.flatnonzero(ds.queries["split"].to_numpy() == split) if split != "all" else np.arange(len(ds.queries))
    r = recall(found[sel], ds.gt(k)[sel])
    reg = ds.regime(k)[sel]
    print(f"{ds.schema['name']}  k={k}  split={split}  queries={len(sel)}")
    print(f"  recall@{k} {r.mean():.4f}   share below 0.9: {(r < 0.9).mean():.3f}")
    for g in ("anti", "indep", "pos"):
        m = reg == g
        if m.any():
            print(f"  {g:5s} n={m.sum():5d}  recall {r[m].mean():.4f}  below 0.9 {(r[m] < 0.9).mean():.3f}")
    if seconds is not None:
        print(f"  mean latency {1e3 * seconds[sel].mean():.2f} ms  p99 {1e3 * np.percentile(seconds[sel], 99):.2f} ms"
              "  (single query at a time)")


def exact(ds, k, rows):
    """Pre-filter + brute-force scan: exact by construction."""
    out = np.full((len(ds.queries), k), -1, np.int64)
    t = np.zeros(len(ds.queries))
    for i in rows:
        t0 = time.perf_counter()
        ids = np.flatnonzero(ds.mask(i))
        s = ds.X[ids] @ ds.Q[i] if len(ids) else np.empty(0)
        top = ids[np.argsort(-s)[:k]]
        t[i] = time.perf_counter() - t0
        out[i, :len(top)] = top
    return out, t


def hnsw_post(ds, k, rows, ef):
    """Unfiltered HNSW search for ef candidates, then filter: the plan that fails on anti-correlated queries."""
    import faiss
    path = os.path.join(ds.dir, "hnsw_M32_efc200.faiss")
    if os.path.exists(path):
        idx = faiss.read_index(path)
    else:
        print("building HNSW (M=32, efConstruction=200) ...", flush=True)
        idx = faiss.IndexHNSWFlat(ds.X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
        idx.add(np.ascontiguousarray(ds.X))
        faiss.write_index(idx, path)
    idx.hnsw.efSearch = ef
    faiss.omp_set_num_threads(1)
    out = np.full((len(ds.queries), k), -1, np.int64)
    t = np.zeros(len(ds.queries))
    for i in rows:
        m = ds.mask(i)                      # filter evaluation is not timed (index-side cost only)
        t0 = time.perf_counter()
        _, I = idx.search(ds.Q[i:i + 1], ef)
        I = I[0][I[0] >= 0]
        top = I[m[I]][:k]
        t[i] = time.perf_counter() - t0
        out[i, :len(top)] = top
    return out, t


def check_gt(ds, k, n):
    """Recompute exact ground truth for n random queries on the vectors in this directory."""
    rng = np.random.default_rng(0)
    rows = rng.choice(len(ds.queries), size=min(n, len(ds.queries)), replace=False)
    mine, _ = exact(ds, k, rows)
    r = recall(mine[rows], ds.gt(k)[rows])
    print(f"ground-truth agreement on {len(rows)} queries (k={k}): mean recall {r.mean():.4f}, "
          f"exact match {(r == 1).mean():.3f}")
    print("  (1.0 with the released vectors; slightly lower after re-embedding on other hardware)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--k", type=int, default=10, choices=(1, 10, 100))
    ap.add_argument("--split", default="test", choices=("train", "test", "all"))
    ap.add_argument("--results")
    ap.add_argument("--baseline", choices=("exact", "hnsw-post"))
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--check-gt", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    ds = Dataset(a.dir)
    if a.check_gt:
        return check_gt(ds, a.k, a.n)
    if a.results:
        found = np.load(a.results)
        if found.shape != (len(ds.queries), a.k):
            sys.exit(f"expected an array of shape {(len(ds.queries), a.k)}, got {found.shape}")
        return report(ds, found, a.k, a.split)
    if a.baseline:
        rows = np.flatnonzero(ds.queries["split"].to_numpy() == a.split) if a.split != "all" else np.arange(len(ds.queries))
        found, t = exact(ds, a.k, rows) if a.baseline == "exact" else hnsw_post(ds, a.k, rows, max(a.ef, a.k))
        return report(ds, found, a.k, a.split, t)
    ap.print_help()


if __name__ == "__main__":
    main()
