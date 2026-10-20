"""HyperVector Polytope-Ball Sketch (HPBS).

L independent partitions of the vector space into m cells. Per cell we keep
  * N_b and a small quantile profile of squared member-to-centroid distances,
  * per-attribute histograms (equi-depth global bins for numeric columns, value counts for
    categorical columns),
  * a row-weighted bottom-k (KMV) sample of foreign keys.
The local selectivity of (q, P) is estimated by (i) choosing the radius r_tau whose estimated
ball holds tau tuples, using ||q-x||^2 ~= ||q-c_b||^2 + ||x-c_b||^2 (concentration of measure),
(ii) weighting each cell by its estimated ball overlap, and (iii) multiplying per-cell predicate
fractions (independence *within* a cell only). The median over the L partitions is returned.
"""
import time
import numpy as np
import faiss
from .data import is_integral, histogram_edges, range_bounds

N_RAD = 16  # radial quantiles per cell


def _splitmix(x):
    x = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class Partition:
    def __init__(self, cent, assign, X):
        self.cent = cent.astype(np.float32)
        self.c2 = (self.cent ** 2).sum(1)
        m = len(cent)
        self.assign = assign
        self.Nb = np.bincount(assign, minlength=m).astype(np.float64)
        # radial profile: quantiles of ||x - c||^2 within each cell
        r2 = np.empty(len(assign), dtype=np.float32)
        B = 200_000
        for s in range(0, len(assign), B):
            e = min(len(assign), s + B)
            diff = X[s:e] - self.cent[assign[s:e]]
            r2[s:e] = (diff * diff).sum(1)
        order = np.argsort(assign, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(self.Nb).astype(np.int64)])
        qs = np.linspace(0, 1, N_RAD)
        self.rq = np.zeros((m, N_RAD), dtype=np.float32)
        for b in range(m):
            v = r2[order[bounds[b]:bounds[b + 1]]]
            if len(v):
                self.rq[b] = np.quantile(v, qs)
        self.order, self.bounds = order, bounds


def kmeans_partition(X, m, seed, sample=200_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    km = faiss.Kmeans(X.shape[1], m, niter=12, seed=int(seed), verbose=False, spherical=False)
    km.train(np.ascontiguousarray(X[idx]))
    _, a = km.index.search(X, 1)
    return km.centroids, a[:, 0].astype(np.int32)


def crosspolytope_partition(X, seed, d_proj=16, m_rad=8):
    """The original design: random rotation, signed arg-max cone x radial shell (2*d_proj*m_rad cells)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    U, _ = np.linalg.qr(rng.standard_normal((d, d_proj)))
    mu = X.mean(0)
    Z = (X - mu) @ U.astype(np.float32)
    j = np.abs(Z).argmax(1)
    cone = j + d_proj * (Z[np.arange(len(Z)), j] < 0)
    nrm = np.linalg.norm(Z, axis=1)
    edges = np.quantile(nrm, np.linspace(0, 1, m_rad + 1)[1:-1])
    shell = np.searchsorted(edges, nrm)
    a = (cone * m_rad + shell).astype(np.int32)
    m = 2 * d_proj * m_rad
    cent = np.zeros((m, d), dtype=np.float32)
    order = np.argsort(a, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=m))])
    for b in range(m):
        if bounds[b + 1] > bounds[b]:
            cent[b] = X[np.sort(order[bounds[b]:bounds[b + 1]])].mean(0)
    return cent, a


class HPBS:
    def __init__(self, T, L=4, m=1024, B=32, kmv=64, partitioner="kmeans", seed=0):
        t0 = time.time()
        self.L, self.m, self.B, self.kmv = L, m, B, kmv
        self.N = T.N
        self.kinds = T.kinds
        self.parts = []
        for l in range(L):
            if partitioner == "kmeans":
                cent, a = kmeans_partition(T.X, m, seed * 100 + l)
            else:
                cent, a = crosspolytope_partition(T.X, seed * 100 + l)
            self.parts.append(Partition(cent, a, T.X))
        self.m = len(self.parts[0].cent)
        # global equi-depth bin edges for numeric columns
        self.integral = {c: is_integral(v) for c, v in T.cols.items() if T.kinds[c] == "num"}
        self.edges = {c: histogram_edges(v, B).astype(np.float32)
                      for c, v in T.cols.items() if T.kinds[c] == "num"}
        self.hist = []   # per partition: col -> (m, nbins+1) cumulative fractions (num) or (m, K) fractions (cat)
        self.kmv_keys, self.kmv_cnt = [], []
        for P in self.parts:
            H = {}
            for c, v in T.cols.items():
                if T.kinds[c] == "num":
                    e = self.edges[c]
                    bins = np.clip(np.searchsorted(e, v, side="right") - 1, 0, B - 1)
                    cnt = np.bincount(P.assign.astype(np.int64) * B + bins, minlength=self.m * B).reshape(self.m, B).astype(np.float32)
                    cum = np.concatenate([np.zeros((self.m, 1), np.float32), np.cumsum(cnt, 1)], 1)
                    H[c] = (cum / np.maximum(P.Nb[:, None], 1)).astype(np.float16)
                else:
                    K = int(v.max()) + 1
                    cnt = np.bincount(P.assign.astype(np.int64) * K + v.astype(np.int64), minlength=self.m * K).reshape(self.m, K).astype(np.float32)
                    H[c] = (cnt / np.maximum(P.Nb[:, None], 1)).astype(np.float16)
            self.hist.append(H)
            # row-weighted bottom-k foreign-key sample per cell
            keys = np.full((self.m, kmv), -1, np.int32)
            cnts = np.zeros((self.m, kmv), np.float32)
            for b in range(self.m):
                rows = P.order[P.bounds[b]:P.bounds[b + 1]]
                if not len(rows):
                    continue
                u, c = np.unique(T.fk[rows], return_counts=True)
                hs = _splitmix(u)
                sel = np.argsort(hs)[:kmv]
                keys[b, :len(sel)] = u[sel]
                cnts[b, :len(sel)] = c[sel]
            self.kmv_keys.append(keys); self.kmv_cnt.append(cnts)
            del P.order, P.bounds, P.assign
        self.build_seconds = time.time() - t0

    # ------------------------------------------------------------------ sizes
    def nbytes(self):
        tot = 0
        for P, H, k, c in zip(self.parts, self.hist, self.kmv_keys, self.kmv_cnt):
            tot += P.cent.nbytes + P.rq.nbytes + P.Nb.size * 4
            tot += sum(h.nbytes for h in H.values()) + k.nbytes + c.nbytes // 2  # counts as uint16
        return tot

    # ------------------------------------------------------------ estimation
    def _cell_fraction(self, l, pred, valid_key_mask, cells=None):
        """Per-cell predicate fraction under within-cell independence, vectorised over cells."""
        H = self.hist[l]
        sl = slice(None) if cells is None else cells
        n = self.m if cells is None else len(cells)
        f = np.ones(n, np.float32)
        for c, (lo, hi) in pred.ranges.items():
            lo, hi = range_bounds(lo, hi, getattr(self, "integral", {}).get(c, False))
            e = self.edges[c]; cum = H[c][sl]
            def pos(x):
                j = int(np.clip(np.searchsorted(e, x, side="right") - 1, 0, self.B - 1))
                w = e[j + 1] - e[j]
                t = 0.0 if w <= 0 else float(np.clip((x - e[j]) / w, 0, 1))
                return cum[:, j].astype(np.float32) * (1 - t) + cum[:, j + 1].astype(np.float32) * t
            f *= np.clip(pos(hi) - pos(lo), 0, 1)
        for c, vals in pred.ins.items():
            f *= H[c][sl][:, list(vals)].astype(np.float32).sum(1)
        if valid_key_mask is not None:
            keys, cnt = self.kmv_keys[l][sl], self.kmv_cnt[l][sl]
            ok = (keys >= 0) & valid_key_mask[np.maximum(keys, 0)]
            f *= (cnt * ok).sum(1) / np.maximum(cnt.sum(1), 1)
        return f

    def global_selectivity(self, pred, valid_key_mask=None):
        P = self.parts[0]
        f = self._cell_fraction(0, pred, valid_key_mask)
        return max(float((P.Nb * f).sum() / self.N), 1.0 / self.N)

    def _ball_weights(self, P, q, tau):
        """Cells overlapping the estimated tau-NN ball and the number of ball tuples in each.

        count_b(r2) = N_b * #{j : dq_b + rq_bj <= r2} / N_RAD, with dq_b = ||q - c_b||^2; r2 is the
        smallest radius (found by bisection) whose total count reaches tau.
        """
        dq = np.maximum(P.c2 - 2.0 * (P.cent @ q) + 1.0, 0.0)
        thr = dq[:, None] + P.rq
        def count(r2):
            return ((thr <= r2).sum(1) * (P.Nb / N_RAD))
        lo, hi = 0.0, float(thr[:, -1].max())
        for _ in range(24):
            mid = np.float32(0.5 * (lo + hi))
            if count(mid).sum() < tau:
                lo = mid
            else:
                hi = mid
        w = count(np.float32(hi))
        cells = np.flatnonzero(w > 0)
        if len(cells) == 0:
            cells = np.array([int(dq.argmin())]); return cells, np.ones(1), hi
        return cells, w[cells], hi

    def estimate_k(self, q, pred, k, T=None, valid_key_mask=None):
        """k-th-match local selectivity: smallest ball holding k estimated matches; s_loc = k / |ball|."""
        if valid_key_mask is None and pred.dim_ins and T is not None:
            valid_key_mask = np.zeros(len(next(iter(T.dim_cols.values()))), bool)
            valid_key_mask[pred.valid_keys(T)] = True
        sgs, locs = [], []
        for l, P in enumerate(self.parts):
            f = self._cell_fraction(l, pred, valid_key_mask)
            sgs.append(float((P.Nb * f).sum() / self.N))
            dq = np.maximum(P.c2 - 2.0 * (P.cent @ q) + 1.0, 0.0)
            thr = dq[:, None] + P.rq
            if (P.Nb * f).sum() < k:
                locs.append(max(sgs[-1], 1.0 / self.N)); continue
            lo, hi = 0.0, float(thr[f > 0, -1].max())
            for _ in range(26):
                mid = np.float32(0.5 * (lo + hi))
                if ((thr <= mid).sum(1) * P.Nb * f).sum() / N_RAD < k:
                    lo = mid
                else:
                    hi = mid
            tot = ((thr <= np.float32(hi)).sum(1) * P.Nb).sum() / N_RAD
            locs.append(min(1.0, k / tot) if tot > 0 else 1.0)
        return max(float(np.median(sgs)), 1.0 / self.N), float(np.median(locs))

    def estimate(self, q, pred, k, T=None, valid_key_mask=None, return_features=False):
        if valid_key_mask is None and pred.dim_ins and T is not None:
            valid_key_mask = np.zeros(len(next(iter(T.dim_cols.values()))), bool)
            valid_key_mask[pred.valid_keys(T)] = True
        s_g = self.global_selectivity(pred, valid_key_mask)
        tau = int(min(self.N, max(k, np.ceil(k / s_g))))
        locs, rads, nact = [], [], []
        for l, P in enumerate(self.parts):
            cells, w, r2 = self._ball_weights(P, q, tau)
            f = self._cell_fraction(l, pred, valid_key_mask, cells)
            locs.append(float((w * f).sum() / w.sum()))
            rads.append(float(r2)); nact.append(len(cells))
        s_loc = float(np.median(locs))
        if not return_features:
            return s_g, s_loc, tau
        lg = np.log(np.maximum(locs, 1e-6))
        feats = np.array([np.log(s_g), np.log(max(s_loc, 1e-6)), float(lg.var()), float(lg.max() - lg.min()),
                          np.log(tau), pred.n_terms(), float(bool(pred.dim_ins)), len(pred.ranges), len(pred.ins),
                          np.log(np.mean(nact)), float(np.mean(rads)),
                          np.log(max(s_loc, 1e-6)) - np.log(s_g)], dtype=np.float32)
        return s_g, s_loc, tau, feats


# ---------------------------------------------------------------------------- C query path
import ctypes as _ct
import os as _os
import subprocess as _sp


class _Part(_ct.Structure):
    _fields_ = [("m", _ct.c_int), ("d", _ct.c_int), ("B1", _ct.c_int),
                ("cent", _ct.c_void_p), ("c2", _ct.c_void_p), ("Nb", _ct.c_void_p), ("rq", _ct.c_void_p),
                ("nnum", _ct.c_int), ("numcum", _ct.c_void_p),
                ("ncat", _ct.c_int), ("catf", _ct.c_void_p), ("catK", _ct.c_void_p),
                ("kmv", _ct.c_int), ("keys", _ct.c_void_p), ("kcnt", _ct.c_void_p)]


class _Query(_ct.Structure):
    _fields_ = [("nr", _ct.c_int), ("rcol", _ct.c_void_p), ("jlo", _ct.c_void_p), ("jhi", _ct.c_void_p),
                ("tlo", _ct.c_void_p), ("thi", _ct.c_void_p),
                ("nc", _ct.c_int), ("ccol", _ct.c_void_p), ("allowed", _ct.c_void_p),
                ("validkey", _ct.c_void_p)]


def _load_est():
    here = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernels")
    src, so = _os.path.join(here, "est.c"), _os.path.join(here, "libest.so")
    if not _os.path.exists(so) or _os.path.getmtime(so) < _os.path.getmtime(src):
        _sp.check_call(["cc", "-shared", "-fPIC", "-O3", "-mcpu=native", "-DACCELERATE_NEW_LAPACK", src, "-o", so, "-framework", "Accelerate"])
    lib = _ct.CDLL(so)
    lib.hpbs_global.restype = _ct.c_double
    lib.hpbs_global.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_double]
    lib.hpbs_local_k.restype = _ct.c_double
    lib.hpbs_local_k.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_double, _ct.c_double,
                                 _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p]
    lib.hpbs_all_k.restype = None
    lib.hpbs_all_k.argtypes = [_ct.c_void_p, _ct.c_int, _ct.c_void_p, _ct.c_void_p, _ct.c_double, _ct.c_double,
                               _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p]
    lib.hpbs_local.restype = _ct.c_double
    lib.hpbs_local.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_double,
                               _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p]
    return lib


class CEstimator:
    """Same estimator as HPBS.estimate, with the per-query work in C (hvo/kernels/est.c)."""

    def __init__(self, S: HPBS):
        self.S, self.lib = S, _load_est()
        self.num_cols = [c for c in S.edges]
        self.cat_cols = [c for c in S.kinds if S.kinds[c] == "cat"]
        self.keep, self.parts = [], []
        for l, P in enumerate(S.parts):
            H = S.hist[l]
            num = [np.ascontiguousarray(H[c], dtype=np.float32) for c in self.num_cols]
            cat = [np.ascontiguousarray(H[c], dtype=np.float32) for c in self.cat_cols]
            catK = np.array([a.shape[1] for a in cat], np.int32)
            nump = (_ct.c_void_p * len(num))(*[a.ctypes.data for a in num])
            catp = (_ct.c_void_p * len(cat))(*[a.ctypes.data for a in cat])
            arrs = dict(cent=np.ascontiguousarray(P.cent, np.float32), c2=P.c2.astype(np.float32),
                        Nb=P.Nb.astype(np.float32), rq=np.ascontiguousarray(P.rq, np.float32),
                        keys=np.ascontiguousarray(S.kmv_keys[l]), kcnt=np.ascontiguousarray(S.kmv_cnt[l]))
            st = _Part(S.m, P.cent.shape[1], S.B + 1, arrs["cent"].ctypes.data, arrs["c2"].ctypes.data,
                       arrs["Nb"].ctypes.data, arrs["rq"].ctypes.data, len(num), _ct.cast(nump, _ct.c_void_p),
                       len(cat), _ct.cast(catp, _ct.c_void_p), catK.ctypes.data,
                       S.kmv, arrs["keys"].ctypes.data, arrs["kcnt"].ctypes.data)
            self.keep.append((num, cat, catK, nump, catp, arrs))
            self.parts.append(st)
        self.wf = np.empty(S.m * (N_RAD + 1), np.float32)
        self.wi = np.empty(S.m * (N_RAD + 1), np.int32)
        self.r2 = np.zeros(1, np.float32); self.na = np.zeros(1, np.int32); self.sg = np.zeros(1, np.float64)
        L = len(self.parts)
        self.parr = (_Part * L)(*self.parts)
        self.o_sloc, self.o_sg = np.zeros(L), np.zeros(L)
        self.o_r2, self.o_na = np.zeros(L, np.float32), np.zeros(L, np.int32)
        # dimension-table columns as a lookup for the valid-key mask
        self._lut = np.zeros(64, bool)

    def _query(self, pred, validkey):
        S = self.S
        rc, jl, jh, tl, th = [], [], [], [], []
        for c, (lo, hi) in pred.ranges.items():
            lo, hi = range_bounds(lo, hi, getattr(S, "integral", {}).get(c, False))
            e = S.edges[c]
            for x, J, Tt in ((lo, jl, tl), (hi, jh, th)):
                j = int(min(max(np.searchsorted(e, x, side="right") - 1, 0), S.B - 1))
                w = float(e[j + 1] - e[j])
                J.append(j); Tt.append(0.0 if w <= 0 else min(max((x - float(e[j])) / w, 0.0), 1.0))
            rc.append(self.num_cols.index(c))
        cc, al = [], []
        for c, vals in pred.ins.items():
            cc.append(self.cat_cols.index(c)); al.append(sum(1 << int(v) for v in vals))
        a = dict(rc=np.array(rc, np.int32), jl=np.array(jl, np.int32), jh=np.array(jh, np.int32),
                 tl=np.array(tl, np.float32), th=np.array(th, np.float32),
                 cc=np.array(cc, np.int32), al=np.array(al, np.uint64))
        vk = validkey.ctypes.data if validkey is not None else None
        Q = _Query(len(rc), a["rc"].ctypes.data, a["jl"].ctypes.data, a["jh"].ctypes.data, a["tl"].ctypes.data,
                   a["th"].ctypes.data, len(cc), a["cc"].ctypes.data, a["al"].ctypes.data, vk)
        return Q, a

    def estimate(self, q, pred, k, T=None, return_features=False):
        S = self.S
        vk = None
        if pred.dim_ins:
            vk = np.zeros(len(next(iter(T.dim_cols.values()))), np.uint8)
            vk[pred.valid_keys(T)] = 1
        Q, _keep = self._query(pred, vk)
        qp = _ct.byref(Q)
        s_g = max(self.lib.hpbs_global(_ct.byref(self.parts[0]), qp, float(S.N)), 1.0 / S.N)
        tau = int(min(S.N, max(k, np.ceil(k / s_g))))
        q = np.ascontiguousarray(q, np.float32)
        locs, rads, nact = [], [], []
        for st in self.parts:
            locs.append(self.lib.hpbs_local(_ct.byref(st), qp, q.ctypes.data, float(tau), self.r2.ctypes.data,
                                            self.na.ctypes.data, self.wf.ctypes.data, self.wi.ctypes.data))
            rads.append(float(self.r2[0])); nact.append(int(self.na[0]))
        s_loc = float(np.median(locs))
        if not return_features:
            return s_g, s_loc, tau
        lg = np.log(np.maximum(locs, 1e-6))
        feats = np.array([np.log(s_g), np.log(max(s_loc, 1e-6)), float(lg.var()), float(lg.max() - lg.min()),
                          np.log(tau), pred.n_terms(), float(bool(pred.dim_ins)), len(pred.ranges), len(pred.ins),
                          np.log(max(np.mean(nact), 1)), float(np.mean(rads)),
                          np.log(max(s_loc, 1e-6)) - np.log(s_g)], dtype=np.float32)
        return s_g, s_loc, tau, feats

    def estimate_k(self, q, pred, k, T=None, return_features=False):
        """C version of HPBS.estimate_k."""
        S = self.S
        vk = None
        if pred.dim_ins:
            vk = np.zeros(len(next(iter(T.dim_cols.values()))), np.uint8)
            vk[pred.valid_keys(T)] = 1
        Q, _keep = self._query(pred, vk)
        qp = _ct.byref(Q)
        q = np.ascontiguousarray(q, np.float32)
        locs, sgs, rads, nact = [], [], [], []
        for st in self.parts:
            locs.append(self.lib.hpbs_local_k(_ct.byref(st), qp, q.ctypes.data, float(k), float(S.N),
                                              self.sg.ctypes.data, self.r2.ctypes.data, self.na.ctypes.data,
                                              self.wf.ctypes.data, self.wi.ctypes.data))
            sgs.append(float(self.sg[0])); rads.append(float(self.r2[0])); nact.append(int(self.na[0]))
        s_g = max(float(np.median(sgs)), 1.0 / S.N)
        s_loc = float(np.median(locs))
        if not return_features:
            return s_g, s_loc
        lg = np.log(np.maximum(locs, 1e-7))
        feats = np.array([np.log(s_g), np.log(max(s_loc, 1e-7)), float(lg.var()), float(lg.max() - lg.min()),
                          pred.n_terms(), float(bool(pred.dim_ins)), len(pred.ranges), len(pred.ins),
                          np.log(max(np.mean(nact), 1)), float(np.mean(rads)), float(np.std(rads)),
                          np.log(max(s_loc, 1e-7)) - np.log(s_g)], dtype=np.float32)
        return s_g, s_loc, feats

    def _query_fast(self, pred, validkey):
        """Same as _query, with ctypes arrays instead of NumPy temporaries."""
        S = self.S
        nr = len(pred.ranges)
        rc = (_ct.c_int * max(nr, 1))(); jl = (_ct.c_int * max(nr, 1))(); jh = (_ct.c_int * max(nr, 1))()
        tl = (_ct.c_float * max(nr, 1))(); th = (_ct.c_float * max(nr, 1))()
        B = S.B
        for i, (c, (lo, hi)) in enumerate(pred.ranges.items()):
            lo, hi = range_bounds(lo, hi, getattr(S, "integral", {}).get(c, False))
            e = S.edges[c]
            for x, J, Tt in ((lo, jl, tl), (hi, jh, th)):
                j = int(np.searchsorted(e, x, side="right")) - 1
                j = 0 if j < 0 else (B - 1 if j > B - 1 else j)
                w = float(e[j + 1] - e[j])
                t = 0.0 if w <= 0 else (x - float(e[j])) / w
                J[i] = j; Tt[i] = 0.0 if t < 0 else (1.0 if t > 1 else t)
            rc[i] = self.num_cols.index(c)
        nc = len(pred.ins)
        cc = (_ct.c_int * max(nc, 1))(); al = (_ct.c_uint64 * max(nc, 1))()
        for i, (c, vals) in enumerate(pred.ins.items()):
            cc[i] = self.cat_cols.index(c)
            m = 0
            for v in vals:
                m |= 1 << int(v)
            al[i] = m
        Q = _Query(nr, _ct.addressof(rc), _ct.addressof(jl), _ct.addressof(jh), _ct.addressof(tl), _ct.addressof(th),
                   nc, _ct.addressof(cc), _ct.addressof(al), validkey.ctypes.data if validkey is not None else None)
        return Q, (rc, jl, jh, tl, th, cc, al)

    def estimate_fast(self, q, pred, k, T=None, return_features=False):
        """estimate_k with one C call for all partitions; returns identical estimates."""
        S = self.S
        vk = None
        if pred.dim_ins:
            dm = None
            for c, vals in pred.dim_ins.items():
                lut = np.zeros(64, bool); lut[list(vals)] = True
                m = lut[T.dim_cols[c]]
                dm = m if dm is None else dm & m
            vk = dm.view(np.uint8)
        Q, _keep = self._query_fast(pred, vk)
        q = np.ascontiguousarray(q, np.float32)
        L = len(self.parts)
        self.lib.hpbs_all_k(self.parr, L, _ct.byref(Q), q.ctypes.data, float(k), float(S.N),
                            self.o_sloc.ctypes.data, self.o_sg.ctypes.data, self.o_r2.ctypes.data,
                            self.o_na.ctypes.data, self.wf.ctypes.data, self.wi.ctypes.data)
        locs = self.o_sloc
        if L == 1:
            s_g, s_loc = max(float(self.o_sg[0]), 1.0 / S.N), float(locs[0])
        else:
            s_g, s_loc = max(float(np.median(self.o_sg)), 1.0 / S.N), float(np.median(locs))
        if not return_features:
            return s_g, s_loc
        lg = np.log(np.maximum(locs, 1e-7))
        feats = np.array([np.log(s_g), np.log(max(s_loc, 1e-7)), float(lg.var()), float(lg.max() - lg.min()),
                          pred.n_terms(), float(bool(pred.dim_ins)), len(pred.ranges), len(pred.ins),
                          np.log(max(float(self.o_na.mean()), 1)), float(self.o_r2.mean()), float(self.o_r2.std()),
                          np.log(max(s_loc, 1e-7)) - np.log(s_g)], dtype=np.float32)
        return s_g, s_loc, feats
