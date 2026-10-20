"""Execute every physical plan on every query (single thread), record latency, rounds and recall.

usage: python experiments/run_plans.py syn-128 [reps]
"""
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
from common import *  # noqa: F401,F403
import numpy as np
import faiss
from hvo import plans, kernels


def med_run(fn, reps):
    fn()                                   # warm-up, discarded
    outs = [fn() for _ in range(reps)]
    ts = [o[1]["t"] for o in outs]
    j = int(np.argsort(ts)[len(ts) // 2])
    return outs[j]


def main(name, reps=3, only=None):
    faiss.omp_set_num_threads(1)
    T, idx = load_table_and_index(name)
    G = load(name, "queries.pkl")
    X8, s8 = kernels.quantize_i8(T.X)
    prev = load(name, "plans.pkl") if only else None
    out = []
    t0 = time.time()
    for i, (q, p, truth) in enumerate(zip(G["Q"], G["P"], G["topk"])):
        q8 = np.clip(np.round(q * s8), -127, 127).astype(np.int8)
        r = dict(prev[i]) if only else {}
        for v in ("simd", "scalar", "i8"):
            if only and f"pre_{v}" not in only:
                continue
            res, info = med_run(lambda: plans.plan_pre(T, q, p, K, variant=v, X8=X8, q8=q8), reps)
            r[f"pre_{v}"] = dict(info, recall=plans.recall(res, truth))
        if not only:
            res, info = med_run(lambda: plans.plan_post(idx, T, q, p, K), reps)
            r["post"] = dict(info, recall=plans.recall(res, truth))
            res, info = med_run(lambda: plans.plan_fused(idx, T, q, p, K), reps)
            r["fused"] = dict(info, recall=plans.recall(res, truth))
        out.append(r)
        if i % 100 == 0:
            print(i, f"{time.time() - t0:.0f}s", {k: round(v['t'] * 1e3, 2) for k, v in r.items()}, flush=True)
    save(out, name, "plans.pkl")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         sys.argv[3].split(",") if len(sys.argv) > 3 else None)
