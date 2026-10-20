"""Generate a dataset, its HNSW index, HPBS sketches, train/test queries, and exact ground truth.

usage: python experiments/build.py syn-128
"""
import os
import sys
import time
import json
from common import *  # noqa: F401,F403
import numpy as np
import faiss
from hvo.data import make_table, make_queries, make_queries_from_pool
from hvo.sketch import HPBS
from hvo import plans, kernels


def main(name):
    N, d, seed = DATASETS[name]
    D = ddir(name)
    log = {}
    t = time.time()
    if not os.path.exists(os.path.join(D, "X.npy")):
        assert not name.startswith(("arxiv", "amazon")), "run the dataset's prep script first"
        T = make_table(N, d, seed=seed)
        np.save(os.path.join(D, "X.npy"), T.X)
        save(dict(cols=T.cols, kinds=T.kinds, fk=T.fk, dim_cols=T.dim_cols, topic=T.topic), name, "table_meta.pkl")
    T = load_table(name)
    log["gen_s"] = time.time() - t
    print("table", T.N, T.d, flush=True)

    if not os.path.exists(os.path.join(D, "hnsw.faiss")):
        t = time.time()
        idx = plans.build_hnsw(T.X, M=32, efc=200, threads=10)
        faiss.write_index(idx, os.path.join(D, "hnsw.faiss"))
        log["hnsw_build_s"] = time.time() - t
        print("hnsw", log["hnsw_build_s"], flush=True)
        del idx

    faiss.omp_set_num_threads(10)
    sketches = {"hpbs": dict(L=4, m=1024, partitioner="kmeans"),
                "hpbs-L1": dict(L=1, m=1024, partitioner="kmeans"),
                "hpbs-m256": dict(L=4, m=256, partitioner="kmeans"),
                "hpbs-cp": dict(L=4, m=256, partitioner="crosspolytope")}
    for sk, kw in sketches.items():
        fn = f"sketch_{sk}.pkl"
        if os.path.exists(os.path.join(D, fn)):
            continue
        S = HPBS(T, B=32, kmv=64, seed=seed, **kw)
        log[f"{sk}_build_s"] = S.build_seconds
        log[f"{sk}_bytes"] = S.nbytes()
        save(S, name, fn)
        print(sk, S.build_seconds, S.nbytes(), flush=True)

    if not os.path.exists(os.path.join(D, "queries.pkl")):
        if name.startswith("arxiv"):
            # real data: queries are held-out papers that are not in the table
            Qpool = np.load(os.path.join(D, "Qpool.npy"))
            Qtr, Ptr = make_queries_from_pool(T, Qpool, N_TRAIN, seed=seed + 1000, offset=0)
            Qte, Pte = make_queries_from_pool(T, Qpool, N_TEST, seed=seed + 2000, offset=N_TRAIN)
        elif name.startswith("amazon"):
            # real data with query-anchored predicates built by prep_amazon.py
            Qpool = np.load(os.path.join(D, "Qpool.npy"))
            pp = load(name, "pool_preds.pkl")["preds"]
            Qtr, Ptr = np.ascontiguousarray(Qpool[:N_TRAIN]), pp[:N_TRAIN]
            Qte, Pte = np.ascontiguousarray(Qpool[N_TRAIN:N_TRAIN + N_TEST]), pp[N_TRAIN:N_TRAIN + N_TEST]
        else:
            Qtr, Ptr = make_queries(T, N_TRAIN, seed=seed + 1000)
            Qte, Pte = make_queries(T, N_TEST, seed=seed + 2000)
        Q = np.concatenate([Qtr, Qte]); P = Ptr + Pte
        split = np.array(["train"] * N_TRAIN + ["test"] * N_TEST)
        # ground truth: exact selectivities and exact filtered top-k
        t = time.time()
        sg, sloc, tau, topk = [], [], [], []
        for i, (q, p) in enumerate(zip(Q, P)):
            m = p.eval(T)
            g = m.mean()
            ta = int(min(T.N, max(K, np.ceil(K / max(g, 1.0 / T.N)))))
            sc = T.X @ q
            nn = np.argpartition(-sc, ta - 1)[:ta] if ta < T.N else np.arange(T.N)
            sg.append(g); tau.append(ta); sloc.append(m[nn].mean())
            ids = np.flatnonzero(m)
            topk.append(ids[np.argsort(-sc[ids])[:K]] if len(ids) else np.empty(0, np.int64))
            if i % 500 == 0:
                print("gt", i, time.time() - t, flush=True)
        keep = np.array(sg) > 0          # a query with an empty answer is not interesting
        save(dict(Q=Q[keep], P=[p for p, k in zip(P, keep) if k], split=split[keep],
                  sg=np.array(sg)[keep], sloc=np.array(sloc)[keep], tau=np.array(tau)[keep],
                  topk=[t_ for t_, k in zip(topk, keep) if k]), name, "queries.pkl")
        log["gt_s"] = time.time() - t
    with open(os.path.join(D, "build_log.json"), "a") as f:
        f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
