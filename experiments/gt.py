"""Exact k-th-match local selectivity: s_loc = k / h_k, where h_k is the number of tuples (matching
or not) at least as close to q as its k-th nearest matching tuple.

usage: python experiments/gt.py syn-128
"""
import sys
import time
from common import *  # noqa: F401,F403
import numpy as np


def main(name):
    T = load_table(name)
    G = load(name, "queries.pkl")          # base query set (vectors, predicates, split)
    hk, topk, t0 = [], [], time.time()
    for i, (q, p) in enumerate(zip(G["Q"], G["P"])):
        m = p.eval(T)
        sc = T.X @ q
        v = sc[m]
        kk = min(K, len(v))
        thr = np.partition(v, len(v) - kk)[len(v) - kk]      # k-th largest score among matches
        hk.append(int((sc >= thr).sum()))
        ids = np.flatnonzero(m)
        topk.append(ids[np.argsort(-sc[ids])[:K]])
        if i % 500 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    G = dict(G)
    G["hk"] = np.array(hk)
    if "sloc_tau" not in G:
        G["sloc_tau"] = G["sloc"]
    G["sloc"] = np.minimum(1.0, K / G["hk"])
    G["topk"] = topk
    save(G, name, QFILE)


if __name__ == "__main__":
    main(sys.argv[1])
