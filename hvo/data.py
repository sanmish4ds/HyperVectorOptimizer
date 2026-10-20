"""Synthetic vector-relational benchmark with controllable spatial-relational correlation.

Vectors come from a two-level topic mixture on the unit sphere. Relational attributes are
generated from the latent topic with different correlation strengths, and each row carries a
foreign key into a dimension table whose attributes are themselves correlated with topics.
This yields queries spanning anti-correlated, independent, and positively correlated regimes.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Table:
    X: np.ndarray                      # (N, d) float32, unit norm
    cols: dict                         # name -> (N,) array (numeric float32 or categorical int16)
    kinds: dict                        # name -> "num" | "cat"
    fk: np.ndarray                     # (N,) int32 foreign key into dim
    dim_cols: dict                     # dim attribute name -> (n_dim,) int16
    topic: np.ndarray = field(repr=False, default=None)

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def d(self):
        return self.X.shape[1]


def make_table(N, d, n_topics=64, n_sub=16, n_dim=20000, sub_spread=0.6, noise=0.55, seed=0):
    rng = np.random.default_rng(seed)
    topics = rng.standard_normal((n_topics, d)).astype(np.float32)
    subs = topics[:, None, :] + sub_spread * rng.standard_normal((n_topics, n_sub, d)).astype(np.float32)
    subs = subs.reshape(-1, d)
    # Zipf-ish topic popularity so some neighbourhoods are dense and some sparse
    pop = 1.0 / np.arange(1, n_topics * n_sub + 1) ** 0.6
    pop = rng.permutation(pop / pop.sum())
    sub_id = rng.choice(n_topics * n_sub, size=N, p=pop)
    topic = (sub_id // n_sub).astype(np.int32)

    X = np.empty((N, d), dtype=np.float32)
    B = 200_000
    for s in range(0, N, B):
        e = min(N, s + B)
        x = subs[sub_id[s:e]] + noise * rng.standard_normal((e - s, d)).astype(np.float32)
        X[s:e] = x / np.linalg.norm(x, axis=1, keepdims=True)

    cols, kinds = {}, {}
    # category: strongly tied to topic (70%)
    cat = np.where(rng.random(N) < 0.7, topic % 32, rng.integers(0, 32, N))
    cols["category"], kinds["category"] = cat.astype(np.int16), "cat"
    # language: tied to sub-topic (50%)
    lang = np.where(rng.random(N) < 0.5, (sub_id * 7) % 12, rng.integers(0, 12, N))
    cols["lang"], kinds["lang"] = lang.astype(np.int16), "cat"
    # year: topic-specific mean, moderate correlation
    tmean = rng.uniform(2002, 2023, n_topics)
    year = np.clip(tmean[topic] + 4.0 * rng.standard_normal(N), 1995, 2026)
    cols["year"], kinds["year"] = year.astype(np.float32), "num"
    # price: independent of the vectors
    cols["price"], kinds["price"] = rng.lognormal(3.0, 1.0, N).astype(np.float32), "num"
    # rating: weak sub-topic effect
    smean = rng.normal(3.4, 0.6, n_topics * n_sub)
    cols["rating"], kinds["rating"] = np.clip(smean[sub_id] + 0.9 * rng.standard_normal(N), 1, 5).astype(np.float32), "num"
    # popularity: independent heavy tail
    cols["views"], kinds["views"] = rng.pareto(1.5, N).astype(np.float32), "num"

    # dimension table (e.g. sellers): region and tier; rows pick sellers correlated with topic
    region = rng.integers(0, 10, n_dim).astype(np.int16)
    tier = rng.integers(0, 4, n_dim).astype(np.int16)
    by_region = [np.flatnonzero(region == r) for r in range(10)]
    pref = rng.integers(0, 10, n_topics)
    # 60% of rows pick a seller in the topic's preferred region; seller popularity is Zipfian
    fk = rng.integers(0, n_dim, N)
    m = rng.random(N) < 0.6
    idx = np.flatnonzero(m)
    for r in range(10):
        sel = idx[pref[topic[idx]] == r]
        pool = by_region[r]
        w = 1.0 / np.arange(1, len(pool) + 1) ** 0.8
        fk[sel] = rng.choice(pool, size=len(sel), p=w / w.sum())
    return Table(X=X, cols=cols, kinds=kinds, fk=fk.astype(np.int32),
                 dim_cols={"region": region, "tier": tier}, topic=topic)


# ---------------------------------------------------------------- histogram helpers
def is_integral(v):
    """True for integer-valued numeric columns (counts, years), which have many tied values."""
    return bool(np.all(v == np.round(v)))


def histogram_edges(v, B):
    """Equi-depth bin edges. For integer-valued columns, edges sit at half-integers so that no
    integer value straddles a bin boundary (ties make plain quantile edges collapse)."""
    e = np.quantile(v, np.linspace(0, 1, B + 1))
    if is_integral(v):
        e = np.floor(e) + 0.5
        e[0], e[-1] = float(v.min()) - 0.5, float(v.max()) + 0.5
        e = np.maximum.accumulate(e)
    return e


def range_bounds(lo, hi, integral):
    """Inclusive [lo, hi] on an integer column covers the continuous interval [ceil(lo)-1/2, floor(hi)+1/2]."""
    if integral:
        return float(np.ceil(lo)) - 0.5, float(np.floor(hi)) + 0.5
    return lo, hi


# ---------------------------------------------------------------- predicates
@dataclass
class Pred:
    """Conjunction of per-column constraints plus an optional semi-join on the dimension table."""
    ranges: dict      # num col -> (lo, hi)
    ins: dict         # cat col -> sorted tuple of allowed values
    dim_ins: dict     # dim col -> tuple of allowed values

    def n_terms(self):
        return len(self.ranges) + len(self.ins) + len(self.dim_ins)

    def eval(self, T: Table, rows=None):
        """Vectorized columnar evaluation; returns a boolean mask over T (or over `rows`)."""
        def col(c):
            v = T.cols[c]
            return v if rows is None else v[rows]
        n = T.N if rows is None else len(rows)
        m = np.ones(n, dtype=bool)
        for c, (lo, hi) in self.ranges.items():
            v = col(c)
            m &= (v >= lo) & (v <= hi)
        for c, vals in self.ins.items():
            lut = np.zeros(64, dtype=bool); lut[list(vals)] = True
            m &= lut[col(c)]
        if self.dim_ins:
            dm = np.ones(len(next(iter(T.dim_cols.values()))), dtype=bool)
            for c, vals in self.dim_ins.items():
                lut = np.zeros(64, dtype=bool); lut[list(vals)] = True
                dm &= lut[T.dim_cols[c]]
            fk = T.fk if rows is None else T.fk[rows]
            m &= dm[fk]
        return m

    def valid_keys(self, T: Table):
        if not self.dim_ins:
            return None
        dm = np.ones(len(next(iter(T.dim_cols.values()))), dtype=bool)
        for c, vals in self.dim_ins.items():
            lut = np.zeros(64, dtype=bool); lut[list(vals)] = True
            dm &= lut[T.dim_cols[c]]
        return np.flatnonzero(dm)


def random_pred(T: Table, rng, quant):
    """Draw 1-3 constraints with widths chosen so global selectivity spans ~0.05%-60%."""
    ranges, ins, dim_ins = {}, {}, {}
    names = list(T.cols)
    nterm = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2])
    chosen = rng.choice(names + ["__dim__"], size=nterm, replace=False)
    for c in chosen:
        if c == "__dim__":
            d0, d1 = list(T.dim_cols)[:2]
            if rng.random() < 0.5:
                K0 = int(T.dim_cols[d0].max()) + 1
                dim_ins[d0] = tuple(sorted(rng.choice(K0, size=rng.integers(1, min(6, K0)), replace=False).tolist()))
            else:
                dim_ins[d1] = (int(rng.integers(0, int(T.dim_cols[d1].max()) + 1)),)
        elif T.kinds[c] == "cat":
            K = int(T.cols[c].max()) + 1
            ins[c] = tuple(sorted(rng.choice(K, size=int(rng.integers(1, max(2, K // 3))), replace=False).tolist()))
        else:
            q = quant[c]
            w = rng.uniform(0.02, 0.9)
            lo = rng.uniform(0, 1 - w)
            ranges[c] = (float(q[int(lo * (len(q) - 1))]), float(q[int((lo + w) * (len(q) - 1))]))
    return Pred(ranges, ins, dim_ins)


def column_quantiles(T: Table, n=1001):
    return {c: np.quantile(v, np.linspace(0, 1, n)).astype(np.float32)
            for c, v in T.cols.items() if T.kinds[c] == "num"}


def make_queries(T: Table, n, seed, noise=0.35):
    """Query vectors are perturbed data points; predicates are independent draws."""
    rng = np.random.default_rng(seed)
    quant = column_quantiles(T)
    base = rng.integers(0, T.N, n)
    Q = T.X[base] + noise * rng.standard_normal((n, T.d)).astype(np.float32) / np.sqrt(T.d)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    preds = [random_pred(T, rng, quant) for _ in range(n)]
    return Q.astype(np.float32), preds


def make_queries_from_pool(T: Table, Qpool, n, seed, offset=0):
    """Query vectors are held-out real vectors (not in T); predicates are independent draws."""
    rng = np.random.default_rng(seed)
    quant = column_quantiles(T)
    Q = np.ascontiguousarray(Qpool[offset:offset + n], dtype=np.float32)
    preds = [random_pred(T, rng, quant) for _ in range(len(Q))]
    return Q, preds
