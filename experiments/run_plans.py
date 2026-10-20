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


def main(name, reps=3, which=None):
    """which: plan keys to (re)run, e.g. "post_r,acorn"; results are merged into the existing file."""
    from hvo.graphplans import ResumableScan, plan_post_resumable, Acorn, plan_acorn
    faiss.omp_set_num_threads(1)
    which = which or ["pre_simd", "pre_scalar", "pre_i8", "post", "fused", "post_r", "acorn"]
    if any(w in which for w in ("post", "fused", "post_r")):
        T, idx = load_table_and_index(name)
    else:                     # ACORN keeps its own copy of the vectors: don't also hold the faiss index
        T, idx = load_table(name), None
    G = load(name, QFILE)
    X8 = s8 = None
    if "pre_i8" in which:
        X8, s8 = kernels.quantize_i8(T.X)
    S = ResumableScan(idx, T.X) if "post_r" in which else None
    A = None
    if "acorn" in which:
        faiss.omp_set_num_threads(10)
        # requires the two ACORN memory-safety fixes in hvo/kernels/acorn_fixes.patch
        # built in memory each run (15-60 s); not cached on disk to save space
        A = Acorn(None, T.X, M=32, gamma=1, M_beta=64, threads=10)
        faiss.omp_set_num_threads(1)
    path = os.path.join(ddir(name), PFILE)
    prev = load(name, PFILE) if os.path.exists(path) else None
    out = []
    t0 = time.time()
    for i, (q, p, truth) in enumerate(zip(G["Q"], G["P"], G["topk"])):
        r = dict(prev[i]) if prev else {}
        q8 = np.clip(np.round(q * s8), -127, 127).astype(np.int8) if X8 is not None else None
        for v in ("simd", "scalar", "i8"):
            if f"pre_{v}" in which:
                res, info = med_run(lambda: plans.plan_pre(T, q, p, K, variant=v, X8=X8, q8=q8), reps)
                r[f"pre_{v}"] = dict(info, recall=plans.recall(res, truth))
        if "post" in which:
            res, info = med_run(lambda: plans.plan_post(idx, T, q, p, K), reps)
            r["post"] = dict(info, recall=plans.recall(res, truth))
        if "fused" in which:
            res, info = med_run(lambda: plans.plan_fused(idx, T, q, p, K), reps)
            r["fused"] = dict(info, recall=plans.recall(res, truth))
        if "post_r" in which:
            res, info = med_run(lambda: plan_post_resumable(S, T, q, p, K), reps)
            r["post_r"] = dict(info, recall=plans.recall(res, truth))
        if "acorn" in which:
            res, info = med_run(lambda: plan_acorn(A, T, q, p, K), reps)
            r["acorn"] = dict(info, recall=plans.recall(res, truth))
        out.append(r)
        if i % 100 == 0:
            print(i, f"{time.time() - t0:.0f}s", {k: round(v['t'] * 1e3, 2) for k, v in r.items()}, flush=True)
    save(out, name, PFILE)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         sys.argv[3].split(",") if len(sys.argv) > 3 else None)
