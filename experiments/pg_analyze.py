"""Summarise the PostgreSQL + pgvector runs (test split): latency, recall, low-recall share, regret
against an oracle that runs both forced plans (pre / iterative post) and keeps the faster one with
Recall@k >= 0.9, and HVO's planning time.

usage: python experiments/pg_analyze.py arxiv-384 amazon-384
writes results/pg.json
"""
import sys
import json
from common import *  # noqa: F401,F403
import numpy as np

MODES = ["stock", "iter", "exqutor", "pre", "post", "hvo", "hvo_norec"]
RECALL_OK = 0.9


def pct(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main(names):
    out = {}
    for name in names:
        G = load(name, QFILE)
        runs = {m: load(name, f"pg_{m}{KS}.pkl") for m in MODES if os.path.exists(os.path.join(ddir(name), f"pg_{m}{KS}.pkl"))}
        te = [i for i in range(len(G["Q"])) if G["split"][i] == "test" and all(i in r for r in runs.values())]
        t = {m: np.array([r[i]["t"] for i in te]) for m, r in runs.items()}
        rec = {m: np.array([r[i]["recall"] for i in te]) for m, r in runs.items()}
        tp = np.where(rec["pre"] >= RECALL_OK, t["pre"], np.inf)
        tq = np.where(rec["post"] >= RECALL_OK, t["post"], np.inf)
        best = np.minimum(tp, tq)
        oc = np.where(tq < tp, 1, 0)                         # oracle choice: 0 pre, 1 post
        none = ~np.isfinite(best)
        best[none] = t["pre"][none]; oc[none] = 0           # pre is exact: the fallback
        oracle_rec = np.where(oc == 1, rec["post"], rec["pre"])
        d = {"n_test": len(te), "oracle": {"lat_ms": pct(best * 1e3), "recall": float(oracle_rec.mean()),
                                            "low_recall": float((oracle_rec < RECALL_OK).mean()),
                                            "post_share": float((oc == 1).mean())}}
        for m in runs:
            d[m] = {"lat_ms": pct(t[m] * 1e3), "recall": float(rec[m].mean()),
                    "low_recall": float((rec[m] < RECALL_OK).mean()), "regret": pct(t[m] / best)}
            if m.startswith("hvo"):
                ch = np.array([runs[m][i]["choice"] for i in te])
                pl = np.array([runs[m][i]["plan_us"] for i in te])
                d[m].update(plan_us=pct(pl), post_share=float((ch == 2).mean()),
                            plan_acc=float(((ch == 2).astype(int) == oc).mean()))
        out[name] = d
        print(name, "n_test", len(te))
        for m in ["oracle"] + list(runs):
            x = d[m]
            extra = f" plan {x['plan_us']['p50']:.0f}us post {x['post_share']:.2f} acc {x['plan_acc']:.3f}" if "plan_us" in x else ""
            print(f"  {m:10s} mean {x['lat_ms']['mean']:8.2f} p50 {x['lat_ms']['p50']:7.2f} p99 {x['lat_ms']['p99']:8.2f} "
                  f"rec {x['recall']:.3f} low {x['low_recall']:.3f}" + (f" reg99 {x['regret']['p99']:.2f}" if "regret" in x else "") + extra)
    with open(os.path.join(RESULTS, f"pg{KS}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
