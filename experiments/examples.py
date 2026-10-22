"""Select and describe the running example of the paper (two real ARXIV-384 test queries), and compute the
aggregate statistics that put the example in context.

Selection rule (fixed in code, applied to the k=10 test split): among pairs of queries with 1-2 predicate
terms and no join whose global selectivities are within 15% of each other (and between 2% and 40%), one
anti-correlated and one positively correlated, whose best (oracle) plans differ, that the tuned global
router nevertheless sends to the same plan and that HVO routes as the oracle does, take the pair on which the global router's misrouting costs the most (ratio of
its latency to the oracle's on the misrouted query). The example is chosen to illustrate the failure mode;
the aggregate statistics report how often it occurs.

usage: python experiments/examples.py      (after analyze.py arxiv-384 / amazon-384)
writes results/examples.json
"""
import gzip
import json
import os
import pickle
import numpy as np
from common import DATA, RESULTS, rho_gls, regime

NAME = "arxiv-384"
PLAN_NAMES = ["pre", "fused", "post"]


def describe(P, info):
    arch = info["archives"]
    terms = []
    for c, (lo, hi) in P.ranges.items():
        c = str(c)
        terms.append(f"{c} = {int(lo)}" if lo == hi else f"{c} in [{int(lo)}, {int(hi)}]")
    for c, vals in P.ins.items():
        c = str(c)
        if c == "archive":
            terms.append("archive in {" + ", ".join(arch[v] for v in vals) + "}")
        elif c == "published":
            terms.append("published = " + ("yes" if vals == (1,) else "no"))
        else:
            terms.append(f"{c} in {list(vals)}")
    return terms


def context(ex, sg0, tol=0.15):
    """How much local selectivity and the best plan vary among queries with about the same global selectivity."""
    sg, sl, oc = (np.array(ex[k]) for k in ("sg", "sloc", "oracle"))
    m = np.abs(np.log(sg / sg0)) < np.log(1 + tol)
    return dict(n=int(m.sum()), sloc_min=float(sl[m].min()), sloc_max=float(sl[m].max()),
                share_pre=float((oc[m] == 0).mean()), share_post=float((oc[m] == 2).mean()),
                share_fused=float((oc[m] == 1).mean()))


def stats(ex):
    t = np.array(ex["t"])
    gc, hc, oc = (np.array(ex[k]) for k in ("global_choice", "hvo_choice", "oracle"))
    n = np.arange(len(gc))
    dis = gc != hc
    lg, lh = t[n, gc], t[n, hc] + np.array(ex["plan_s"])
    return dict(n=len(gc), disagree=float(dis.mean()), hvo_faster_when_disagree=float((lh < lg)[dis].mean()),
                global_wrong=float((gc != oc).mean()), hvo_wrong=float((hc != oc).mean()))


def main():
    ex = json.load(open(os.path.join(RESULTS, f"{NAME}_examples.json")))
    G = pickle.load(open(os.path.join(DATA, NAME, "queries.pkl"), "rb"))
    info = pickle.load(open(os.path.join(DATA, NAME, "arxiv_info.pkl"), "rb"))
    Qpool = np.load(os.path.join(DATA, NAME, "Qpool.npy"), mmap_mode="r")
    idx = np.array(ex["idx"])
    sg, sl, t = np.array(ex["sg"]), np.array(ex["sloc"]), np.array(ex["t"])
    oc, gc, hc = (np.array(ex[k]) for k in ("oracle", "global_choice", "hvo_choice"))
    r = rho_gls(sl, sg)
    simple = np.array([G["P"][i].n_terms() <= 2 and not G["P"][i].dim_ins for i in idx])
    ok = simple & (sg > 0.02) & (sg < 0.4) & (hc == oc)
    anti = [i for i in np.flatnonzero(ok & (r < -0.3))]
    pos = [i for i in np.flatnonzero(ok & (r > 0.3))]
    best = None
    for a in anti:
        for b in pos:
            if abs(np.log(sg[a] / sg[b])) >= np.log(1.15) or gc[a] != gc[b] or oc[a] == oc[b]:
                continue
            for w in (a, b):
                if gc[w] != oc[w]:
                    score = t[w, gc[w]] / t[w, oc[w]]
                    if best is None or score > best[0]:
                        best = (score, a, b)
    _, a, b = best

    # titles of the two query papers
    pool_pos = {}
    for i in (a, b):
        q = G["Q"][idx[i]]
        pool_pos[i] = int(np.flatnonzero((np.asarray(Qpool) == q).all(1))[0])
    want = {info["pool_ids"][pool_pos[i]]: i for i in (a, b)}
    meta = {}
    with gzip.open(os.path.join(DATA, "raw", "arxiv", "arxiv-abstracts.jsonl.gz"), "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] in want:
                meta[want[rec["id"]]] = dict(id=rec["id"], title=" ".join(rec["title"].split()),
                                             primary=" ".join(rec["categories"]).split()[0])
            if len(meta) == len(want):
                break

    out = {"rule": __doc__.split("\n\n")[1], "eta": ex["eta"], "global_threshold": ex["global_threshold"],
           "stats": {NAME: stats(ex)}}
    for tag, i in (("anti", a), ("pos", b)):
        P = G["P"][idx[i]]
        out[tag] = dict(query=int(idx[i]), **meta[i], filter=describe(P, info), sg=float(sg[i]), sloc=float(sl[i]),
                        hk=int(G["hk"][idx[i]]), rho=float(r[i]), regime=str(regime(r[i])),
                        t_ms={p: 1e3 * float(t[i, j]) for j, p in enumerate(PLAN_NAMES)},
                        recall={p: float(ex["rec"][i][j]) for j, p in enumerate(PLAN_NAMES)},
                        oracle=PLAN_NAMES[oc[i]], global_choice=PLAN_NAMES[gc[i]], hvo_choice=PLAN_NAMES[hc[i]],
                        sloc_raw=float(ex["sloc_raw"][i]), sloc_hat=float(ex["sloc_hat"][i]),
                        sloc_probe=float(ex["sloc_probe"][i]), sg_hat=float(ex["sg_hat"][i]),
                        cost_lo_ms={p: 1e3 * float(ex["cost_lo"][i][j]) for j, p in enumerate(PLAN_NAMES)},
                        cost_hi_ms={p: 1e3 * float(ex["cost_hi"][i][j]) for j, p in enumerate(PLAN_NAMES)},
                        cost_true_ms={p: 1e3 * float(ex["cost_true"][i][j]) for j, p in enumerate(PLAN_NAMES)},
                        plan_us=1e6 * float(ex["plan_s"][i]))
    out["context"] = context(ex, (sg[a] + sg[b]) / 2)
    am = os.path.join(RESULTS, "amazon-384_examples.json")
    if os.path.exists(am):
        out["stats"]["amazon-384"] = stats(json.load(open(am)))
    json.dump(out, open(os.path.join(RESULTS, "examples.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
