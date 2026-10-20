"""Time HVO's planning decision when it runs entirely in C (hvo/kernels/est.c: hvo_plan): sketch
estimate, residual calibrator, recall admissibility and minimax cost routing in one call.
Uses the parameters analyze.py fitted on the training split; checks the native choices agree with it.

usage: python experiments/native_plan.py syn-128      (after analyze.py)
writes results/<name>[_kK]_native.json
"""
import os
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
import json
import time
import ctypes as ct
from common import *  # noqa: F401,F403
import numpy as np
from hvo.sketch import CEstimator, _Part, _Query


class CalW(ct.Structure):
    _fields_ = [("nin", ct.c_int), ("nh", ct.c_int), ("mu", ct.c_void_p), ("sd", ct.c_void_p),
                ("W0", ct.c_void_p), ("b0", ct.c_void_p), ("W1", ct.c_void_p), ("b1", ct.c_void_p),
                ("W2", ct.c_void_p), ("b2", ct.c_float)]


class CostP(ct.Structure):
    _fields_ = [("rel0", ct.c_double), ("rel1", ct.c_double), ("rel2", ct.c_double), ("a0", ct.c_double),
                ("cscan", ct.c_double), ("post", ct.c_double * 3), ("fused", ct.c_double * 3), ("cf", ct.c_double),
                ("beta", ct.c_double), ("ef0", ct.c_double), ("cap", ct.c_double), ("rec_w0", ct.c_double * 2),
                ("rec_w", (ct.c_double * 4) * 2), ("rec_c", ct.c_double * 2), ("tau_rec", ct.c_double),
                ("eta", ct.c_double), ("n_iso", ct.c_int), ("iso_x", ct.c_void_p), ("iso_y", ct.c_void_p)]


def main(name):
    P = json.load(open(rfile(name).replace(".json", "_planner.json")))
    T = load_table(name)
    G = load(name, QFILE)
    C = CEstimator(load(name, "sketch_hpbs.pkl"))
    lib = C.lib
    lib.hvo_plan.restype = ct.c_int
    lib.hvo_plan.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_double, ct.c_double,
                             ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_void_p, ct.c_void_p,
                             ct.c_void_p, ct.c_void_p, ct.c_void_p]
    c = P["cal"]
    arr = lambda x: np.ascontiguousarray(np.array(x, np.float32))
    keep = [arr(c["mu"]), arr(c["sd"]), arr(c["W"][0]), arr(c["b"][0]), arr(c["W"][1]), arr(c["b"][1]),
            arr(c["W"][2]).ravel()]
    cw = CalW(12, keep[2].shape[1], *[k.ctypes.data for k in keep], float(c["b"][2][0]))
    co = P["cost"]
    cp = CostP()
    cp.rel0, cp.rel1, cp.rel2 = co["rel"]
    cp.a0, cp.cscan = co["pre_c"]
    cp.post[:] = co["post"]; cp.fused[:] = co["fused"]
    cp.cf, cp.beta, cp.ef0, cp.cap, cp.eta = co["cf"], co["beta"], 64.0, float(1 << 17), P["eta"]
    iso_x = np.ascontiguousarray(co.get("iso_x", []), np.float64)
    iso_y = np.ascontiguousarray(co.get("iso_y", []), np.float64)
    cp.n_iso, cp.iso_x, cp.iso_y = len(iso_x), iso_x.ctypes.data, iso_y.ctypes.data
    for j, (w0, w) in enumerate(P["recall_model"]):
        if w is None:
            cp.rec_w0[j] = float("nan"); cp.rec_c[j] = w0
        else:
            cp.rec_w0[j] = w0
            for i in range(4):
                cp.rec_w[j][i] = w[i]
    est = np.zeros(3)
    out = {}
    for mode, tau, ref in (("hvo", 0.0, P["choice_hvo"]), ("hvo-recall", P["tau_rec"], P["choice_hvo_recall"])):
        cp.tau_rec = tau
        times, agree, choices = [], 0, []
        for j, i in enumerate(P["test_idx"]):
            q, p = np.ascontiguousarray(G["Q"][i], np.float32), G["P"][i]
            vk = None
            if p.dim_ins:
                dm = None
                for col, vals in p.dim_ins.items():
                    lut = np.zeros(64, bool); lut[list(vals)] = True
                    m = lut[T.dim_cols[col]]
                    dm = m if dm is None else dm & m
                vk = dm.view(np.uint8)
            Q, _keep = C._query_fast(p, vk)
            args = (C.parr, len(C.parts), ct.byref(Q), q.ctypes.data, float(K), float(T.N), float(p.n_terms()),
                    float(bool(p.dim_ins)), float(len(p.ranges)), float(len(p.ins)), ct.byref(cw), ct.byref(cp),
                    est.ctypes.data, C.wf.ctypes.data, C.wi.ctypes.data)
            lib.hvo_plan(*args)
            best = None
            for _ in range(5):
                t = time.perf_counter(); ch = lib.hvo_plan(*args); dt = time.perf_counter() - t
                best = dt if best is None else min(best, dt)
            times.append(best); choices.append(ch); agree += int(ch == ref[j])
        out[mode] = dict(native_us=float(np.median(times) * 1e6), native_us_p99=float(np.percentile(times, 99) * 1e6),
                         agreement=agree / len(times), choices=choices, times=times)
        print(name, mode, f"native planning {np.median(times)*1e6:.1f} us (p99 {np.percentile(times,99)*1e6:.1f}),"
              f" agreement with Python planner {agree / len(times):.4f}")
    with open(rfile(name).replace(".json", "_native.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
