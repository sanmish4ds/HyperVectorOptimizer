"""Baseline selectivity estimators. Each returns (s_g_hat, s_loc_hat)."""
import numpy as np
from .data import is_integral, column_kind


class HistAVI:
    """PostgreSQL-style: per-column 1-D histograms / MCV lists, attribute-value independence,
    uniform foreign-key assumption for the semi-join, and s_loc = s_g (no vector awareness)."""

    def __init__(self, T, B=100):
        self.T = T
        self.edges = {c: np.quantile(v, np.linspace(0, 1, B + 1)) for c, v in T.cols.items() if T.kinds[c] == "num"}
        # integer-valued columns: exact value frequencies (what PostgreSQL's most-common-value lists give)
        self.values, self.mcv = {}, {}
        for c, v in T.cols.items():
            if T.kinds[c] == "num" and is_integral(v):
                u, cnt = np.unique(v, return_counts=True)
                self.values[c] = (u, np.cumsum(cnt) / len(v))
            elif T.kinds[c] == "num" and column_kind(v, B) == "mass":
                # PostgreSQL-style: a most-common-value list plus an equi-depth histogram of the rest
                u, cnt = np.unique(v, return_counts=True)
                heavy = cnt / len(v) >= 1.0 / B
                rest = v[~np.isin(v, u[heavy])]
                self.mcv[c] = (u[heavy], cnt[heavy] / len(v))
                self.edges[c] = np.quantile(rest, np.linspace(0, 1, B + 1)) if len(rest) else np.zeros(B + 1)
        self.freq = {c: np.bincount(v.astype(np.int64)) / T.N for c, v in T.cols.items() if T.kinds[c] == "cat"}
        self.dfreq = {c: np.bincount(v.astype(np.int64)) / len(v) for c, v in T.dim_cols.items()}
        self.B = B

    def _range(self, c, lo, hi):
        if c in self.values:
            u, cum = self.values[c]
            def le(x):  # fraction of rows <= x
                j = np.searchsorted(u, x, side="right") - 1
                return 0.0 if j < 0 else float(cum[j])
            return max(le(np.floor(hi)) - le(np.ceil(lo) - 1), 0.0)
        if c in self.mcv:
            mu, mf = self.mcv[c]
            in_mcv = float(mf[(mu >= lo) & (mu <= hi)].sum())
            e = self.edges[c]
            def cdf_r(x):
                j = np.clip(np.searchsorted(e, x, side="right") - 1, 0, self.B - 1)
                w = e[j + 1] - e[j]
                t = 0.0 if w <= 0 else np.clip((x - e[j]) / w, 0, 1)
                return (j + t) / self.B
            return in_mcv + (1.0 - float(mf.sum())) * max(cdf_r(hi) - cdf_r(lo), 0.0)
        e = self.edges[c]
        def cdf(x):
            j = np.clip(np.searchsorted(e, x, side="right") - 1, 0, self.B - 1)
            w = e[j + 1] - e[j]
            t = 0.0 if w <= 0 else np.clip((x - e[j]) / w, 0, 1)
            return (j + t) / self.B
        return max(cdf(hi) - cdf(lo), 0.0)

    def estimate(self, q, pred, k):
        s = 1.0
        for c, (lo, hi) in pred.ranges.items():
            s *= self._range(c, lo, hi)
        for c, vals in pred.ins.items():
            s *= float(self.freq[c][list(vals)].sum())
        for c, vals in pred.dim_ins.items():
            s *= float(self.dfreq[c][list(vals)].sum())
        s = max(s, 1.0 / self.T.N)
        return s, s

    def nbytes(self):
        return sum(e.size * 4 for e in self.edges.values()) + sum(f.size * 4 for f in self.freq.values())


class ExactAVI:
    """Milvus-style: exact global selectivity from the predicate bitmap, s_loc assumed equal to s_g."""

    def __init__(self, T):
        self.T = T

    def estimate(self, q, pred, k):
        s = max(float(pred.eval(self.T).mean()), 1.0 / self.T.N)
        return s, s

    def nbytes(self):
        return 0


class UniformSample:
    """Online uniform sample: s_g from the sample; s_loc from the sample's nearest tau*rate points."""

    def __init__(self, T, rate=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.T = T
        self.rows = np.sort(rng.choice(T.N, size=int(rate * T.N), replace=False))
        self.Xs = np.ascontiguousarray(T.X[self.rows])
        self.rate = rate

    def estimate(self, q, pred, k):
        """s_g = sample match rate; s_loc = k_s / h_s with h_s the sample rank of the k_s-th sampled match,
        k_s = max(1, round(k * rate)). With rate = 1% and k = 10 this rests on a single sampled match."""
        m = pred.eval(self.T, rows=self.rows)
        sg = max(float(m.mean()), 1.0 / self.T.N)
        if not m.any():
            return sg, sg
        ks = max(1, int(round(k * self.rate)))
        sc = self.Xs @ q
        v = sc[m]
        ks = min(ks, len(v))
        thr = np.partition(v, len(v) - ks)[len(v) - ks]
        return sg, min(1.0, ks / float((sc >= thr).sum()))

    def nbytes(self):
        return self.Xs.nbytes + self.rows.nbytes


class IndexProbe:
    """Exqutor-style index probing: s_loc = valid fraction among the first P unfiltered HNSW
    neighbours of q (floored at 0.5/P); s_g from 1-D histograms."""

    def __init__(self, T, index, P=256):
        import faiss
        self.T, self.index, self.P = T, index, P
        self.params = faiss.SearchParametersHNSW(efSearch=P)
        self.hist = HistAVI(T)

    def estimate(self, q, pred, k):
        _, I = self.index.search(q[None, :], self.P, params=self.params)
        I = I[0][I[0] >= 0]
        sl = max(float(pred.eval(self.T, rows=I).mean()), 0.5 / self.P)
        sg, _ = self.hist.estimate(q, pred, k)
        return sg, sl

    def nbytes(self):
        return 0
