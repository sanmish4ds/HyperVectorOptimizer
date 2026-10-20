"""Calibrator transfer: train the residual calibrator (and the interval width eta) on one dataset's
training split and evaluate HPBS + HVO routing on another dataset's test split.

The cost model is always calibrated on the target dataset (it is a property of the machine and the
data layout); only the learned estimator correction is transferred.

usage: python experiments/transfer.py syn-128 syn-768 arxiv-384
writes results/transfer.json
"""
import sys
import json
from common import *  # noqa: F401,F403
import numpy as np
from analyze import load_all, Calibrator, CostModel, plan_times, oracle, evaluate, pct, floor_s


def main(names):
    A = {n: load_all(n) for n in names}
    cal, eta = {}, {}
    for s in names:
        a = A[s]; tr = a["split"] == "train"
        c = Calibrator().fit(a["feat"]["hpbs"][tr], a["est"]["hpbs"][tr, 1], a["sloc"][tr], a["tau"][tr])
        cal[s] = c
        s_hat = c.apply(a["feat"]["hpbs"][tr], a["est"]["hpbs"][tr, 1], a["tau"][tr])
        eta[s] = float(np.quantile(qerr(s_hat, a["sloc"][tr], a["tau"][tr]), 0.9))
    out = {"pairs": [], "eta": eta}
    for t in names:
        a = A[t]; N = DATASETS[t][0]
        tr, te = a["split"] == "train", a["split"] == "test"
        CM = CostModel().fit(a, tr, N)
        tt, rr = plan_times(a)
        oc, ot = oracle(tt, rr)
        tt, rr, oc, ot = tt[te], rr[te], oc[te], ot[te]
        args = (a["tau"][te], a["nterm"][te], a["join"][te])
        raw = a["est"]["hpbs"]
        t_plan = raw[te, 2]
        for s in names + ["none"]:
            if s == "none":
                s_hat = floor_s(raw[te, 1], a["tau"][te])
                tr_hat = floor_s(raw[tr, 1], a["tau"][tr])
                e = float(np.quantile(qerr(tr_hat, a["sloc"][tr], a["tau"][tr]), 0.9))
            else:
                s_hat = cal[s].apply(a["feat"]["hpbs"][te], raw[te, 1], a["tau"][te])
                e = eta[s]
            q = qerr(s_hat, a["sloc"][te], a["tau"][te])
            C = np.maximum(CM.costs(raw[te, 0], np.clip(s_hat / e, 1e-7, 1), *args),
                           CM.costs(raw[te, 0], np.clip(s_hat * e, 1e-7, 1), *args))
            r = evaluate(C.argmin(1), tt, rr, t_plan, ot, oc)
            out["pairs"].append(dict(source=s, target=t, qerr=pct(q), eta=e, lat_ms=r["lat_ms"],
                                     regret=r["regret"], plan_acc=r["plan_acc"]))
            print(f"{s:>10s} -> {t:10s}  Q p50 {np.median(q):.2f} p99 {np.percentile(q, 99):7.2f}  eta {e:5.2f}"
                  f"  mean {r['lat_ms']['mean']:.2f} p99 {r['lat_ms']['p99']:.2f} reg99 {r['regret']['p99']:.2f}")
    with open(os.path.join(RESULTS, "transfer.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
