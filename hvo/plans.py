"""The three physical plans, executed for real on a faiss HNSW index.

Plans are adaptive (they double their search budget until k valid results are found), so a
plan's latency does not depend on the optimizer's estimate; the optimizer only chooses which
plan runs. Each round of an adaptive plan is timed separately so the runtime early-exit guard
can be replayed exactly offline.
"""
import time
import numpy as np
import faiss
from . import kernels

EF0 = 64
CAP = 1 << 17          # 131072 candidates: hard budget for graph plans


def build_hnsw(X, M=32, efc=200, threads=10):
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = efc
    idx.add(X)
    return idx


def plan_pre(T, q, pred, k, variant="simd", X8=None, q8=None):
    """Columnar predicate evaluation -> sorted row-id list -> sequential SIMD scan of survivors."""
    t0 = time.perf_counter()
    mask = pred.eval(T)
    ids = np.flatnonzero(mask)
    t_rel = time.perf_counter() - t0
    if variant == "i8":
        # INT8 SDOT scan for 4k candidates, then exact FP32 rerank of those candidates
        cand, _ = kernels.scan_topk(X8, ids, q8, 4 * k, variant="i8")
        res, _ = kernels.scan_topk(T.X, np.sort(cand), q, k, variant="simd")
    else:
        res, _ = kernels.scan_topk(T.X, ids, q, k, variant=variant)
    return res, dict(t=time.perf_counter() - t0, t_rel=t_rel, n_scan=len(ids))


def plan_post(index, T, q, pred, k, start=EF0):
    """Unfiltered HNSW search, verify predicate on candidates, double budget until k valid."""
    t0 = time.perf_counter()
    rounds = []
    kk = max(start, k)
    q2 = q[None, :]
    stats = faiss.cvar.hnsw_stats
    while True:
        stats.reset()
        params = faiss.SearchParametersHNSW(efSearch=kk)
        _, I = index.search(q2, kk, params=params)
        I = I[0]; I = I[I >= 0]
        ok = pred.eval(T, rows=I)
        nvalid = int(ok.sum())
        rounds.append(dict(t=time.perf_counter() - t0, kk=kk, nvalid=nvalid, ncand=len(I), ndis=int(stats.ndis)))
        if nvalid >= k or kk >= CAP or kk >= T.N:
            break
        kk *= 2
    res = I[ok][:k]
    return res, dict(t=time.perf_counter() - t0, rounds=rounds)


def plan_fused(index, T, q, pred, k, start=EF0):
    """Materialise the predicate bitmap, then filter inline during HNSW traversal; double ef until k found."""
    t0 = time.perf_counter()
    mask = pred.eval(T)
    bits = np.packbits(mask, bitorder="little")
    sel = faiss.IDSelectorBitmap(len(mask), faiss.swig_ptr(bits))
    t_rel = time.perf_counter() - t0
    ef = max(start, k)
    q2 = q[None, :]
    rounds = []
    prev = None
    stats = faiss.cvar.hnsw_stats
    while True:
        stats.reset()
        params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
        _, I = index.search(q2, k, params=params)
        I = I[0]; I = I[I >= 0]
        rounds.append(dict(t=time.perf_counter() - t0, ef=ef, nres=len(I), ndis=int(stats.ndis)))
        # stop once k results are found and a doubled budget no longer changes them
        cur = set(map(int, I))
        if (len(I) >= k and cur == prev) or ef >= CAP:
            break
        prev = cur if len(I) >= k else None
        ef *= 2
    return I, dict(t=time.perf_counter() - t0, t_rel=t_rel, rounds=rounds)


def recall(res, truth):
    if len(truth) == 0:
        return 1.0
    return len(set(map(int, res)) & set(map(int, truth))) / len(truth)
