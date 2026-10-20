"""Run every cardinality estimator on every query; record estimates, features and planning time.

usage: python experiments/run_est.py syn-128
"""
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
from common import *  # noqa: F401,F403
import numpy as np
import faiss
from hvo.sketch import CEstimator
from hvo.estimators import HistAVI, ExactAVI, UniformSample, IndexProbe


def timed(fn, reps=3):
    fn()
    best = None
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return r, best


def main(name):
    faiss.omp_set_num_threads(1)
    T, idx = load_table_and_index(name)
    G = load(name, "queries.pkl")
    ests = {"pg-hist": HistAVI(T), "exact-avi": ExactAVI(T), "sample-1%": UniformSample(T, 0.01),
            "probe": IndexProbe(T, idx, P=256)}
    sk = {}
    for s in ("hpbs", "hpbs-L1", "hpbs-m256", "hpbs-cp"):
        S = load(name, f"sketch_{s}.pkl")
        sk[s] = (CEstimator(S), S.nbytes())
    rows = []
    t0 = time.time()
    for i, (q, p) in enumerate(zip(G["Q"], G["P"])):
        r = {}
        for n_, E in ests.items():
            (sg, sl), dt = timed(lambda: E.estimate(q, p, K))
            r[n_] = (sg, sl, dt)
        for n_, (C, _) in sk.items():
            (sg, sl, f), dt = timed(lambda: C.estimate_fast(q, p, K, T, return_features=True))
            r[n_] = (sg, sl, dt); r[n_ + "_feat"] = f
        rows.append(r)
        if i % 250 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    meta = {n_: E.nbytes() for n_, E in ests.items()}
    meta.update({n_: b for n_, (_, b) in sk.items()})
    save(dict(rows=rows, bytes=meta), name, "est.pkl")


if __name__ == "__main__":
    main(sys.argv[1])
