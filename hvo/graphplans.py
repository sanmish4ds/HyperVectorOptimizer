"""Second-generation graph plans:
  * plan_post_resumable: post-filtering over a resumable best-first HNSW scan (kernels/graph.c), the
    behaviour of pgvector 0.8's iterative index scan: batches are verified against the predicate and
    the scan continues where it stopped, so no work is repeated.
  * plan_acorn: fused traversal with the ACORN reference implementation (ACORN-1: two-hop expansion
    over filtered neighbours), with the same adaptive efSearch rule as the first-generation fused plan.
"""
import os
import time
import ctypes as ct
import subprocess
import numpy as np
from .plans import EF0, CAP

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


class _Graph(ct.Structure):
    _fields_ = [("n", ct.c_int), ("d", ct.c_int), ("X", ct.c_void_p), ("nbr", ct.c_void_p), ("off", ct.c_void_p),
                ("cum", ct.c_void_p), ("levels", ct.c_void_p), ("entry", ct.c_int), ("max_level", ct.c_int)]


def _load_graph_lib():
    src, so = os.path.join(_HERE, "graph.c"), os.path.join(_HERE, "libgraph.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-shared", "-fPIC", "-O3", "-mcpu=native", src, "-o", so])
    lib = ct.CDLL(so)
    lib.it_new.restype = ct.c_void_p; lib.it_new.argtypes = [ct.c_void_p]
    lib.it_start.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
    lib.it_next.restype = ct.c_int; lib.it_next.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    lib.it_ndis.restype = ct.c_int64; lib.it_ndis.argtypes = [ct.c_void_p]
    return lib


class ResumableScan:
    """Owns copies of the faiss HNSW adjacency (the vectors stay in the index's storage)."""

    def __init__(self, index, X):
        import faiss
        h = index.hnsw
        self.nbr = faiss.vector_to_array(h.neighbors).astype(np.int32)
        self.off = faiss.vector_to_array(h.offsets).astype(np.uint64)
        self.cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int32)
        self.levels = faiss.vector_to_array(h.levels).astype(np.int32)
        self.X = X
        self.g = _Graph(X.shape[0], X.shape[1], X.ctypes.data, self.nbr.ctypes.data, self.off.ctypes.data,
                        self.cum.ctypes.data, self.levels.ctypes.data, int(h.entry_point), int(h.max_level))
        self.lib = _load_graph_lib()
        self.it = self.lib.it_new(ct.byref(self.g))
        self.buf = np.empty(CAP, np.int64)
        self.bufd = np.empty(CAP, np.float32)


def plan_post_resumable(S: ResumableScan, T, q, pred, k, start=EF0):
    """Verify the predicate on successive batches of the resumable scan until k matches."""
    t0 = time.perf_counter()
    q = np.ascontiguousarray(q, np.float32)
    S.lib.it_start(S.it, q.ctypes.data, max(start, k))     # window ef = max(64, k), as in the other plans
    rounds, found, emitted, B = [], [], 0, k                 # batches k, k, 2k, 4k, ... (resumed, never repeated)
    while True:
        m = S.lib.it_next(S.it, B, S.buf.ctypes.data, S.bufd.ctypes.data)
        ids = S.buf[:m]
        ok = pred.eval(T, rows=ids)
        found.extend(ids[ok].tolist())
        emitted += m
        rounds.append(dict(t=time.perf_counter() - t0, kk=emitted, nvalid=len(found), ncand=emitted,
                           ndis=int(S.lib.it_ndis(S.it))))
        if len(found) >= k or m == 0 or emitted >= CAP:
            break
        B = min(emitted, CAP - emitted)   # cumulative budget doubles each round: 64, 128, 256, ...
    return np.array(found[:k], np.int64), dict(t=time.perf_counter() - t0, rounds=rounds)


class Acorn:
    def __init__(self, path=None, X=None, M=32, gamma=1, M_beta=64, threads=10):
        so = os.path.join(_HERE, "libacorn.so")
        if not os.path.exists(so):
            subprocess.check_call([os.path.join(_HERE, "build_acorn.sh")])
        lib = ct.CDLL(so)
        lib.acorn_build.restype = ct.c_void_p
        lib.acorn_build.argtypes = [ct.c_int] * 4 + [ct.c_int64, ct.c_void_p, ct.c_int]
        lib.acorn_write.argtypes = [ct.c_void_p, ct.c_char_p]
        lib.acorn_read.restype = ct.c_void_p; lib.acorn_read.argtypes = [ct.c_char_p]
        lib.acorn_search.restype = ct.c_int
        lib.acorn_search.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p]
        lib.acorn_last_ndis.restype = ct.c_int64
        self.lib = lib
        if path and os.path.exists(path):
            self.idx = lib.acorn_read(path.encode())
        else:
            X = np.ascontiguousarray(X, np.float32)
            self.idx = lib.acorn_build(X.shape[1], M, gamma, M_beta, X.shape[0], X.ctypes.data, threads)
            if path:
                lib.acorn_write(self.idx, path.encode())


def plan_acorn(A: Acorn, T, q, pred, k, start=EF0):
    """Materialise the predicate map, then ACORN filtered search; double efSearch until k results are
    found and a doubled budget returns the same set (same stopping rule as plans.plan_fused)."""
    t0 = time.perf_counter()
    mask = pred.eval(T)
    fmap = mask.view(np.int8)
    t_rel = time.perf_counter() - t0
    q = np.ascontiguousarray(q, np.float32)
    D = np.empty(k, np.float32); I = np.empty(k, np.int64)
    ef, prev, rounds = max(start, k), None, []
    while True:
        n = A.lib.acorn_search(A.idx, q.ctypes.data, k, ef, fmap.ctypes.data, D.ctypes.data, I.ctypes.data)
        res = I[:n].copy()
        rounds.append(dict(t=time.perf_counter() - t0, ef=ef, nres=n, ndis=int(A.lib.acorn_last_ndis())))
        cur = set(res.tolist())
        if (n >= k and cur == prev) or ef >= CAP:
            break
        prev = cur if n >= k else None
        ef *= 2
    return res, dict(t=time.perf_counter() - t0, t_rel=t_rel, rounds=rounds)
