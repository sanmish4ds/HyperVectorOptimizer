"""Fit HVO's cost and recall models to PostgreSQL + pgvector execution times on the TRAINING queries
(forced pre-filter and forced iterative post-filter runs from pg_bench.py) and write the binary model
the hvo_pg planner hook loads: HPBS sketch, dimension table, residual calibrator, cost model, interval
width and recall model.

usage: python experiments/pg_model.py arxiv-384
writes data/<name>/hvo_pg_model[_kK].bin
"""
import sys
import struct
import json
from pg_common import *  # noqa: F401,F403
import numpy as np
from scipy.optimize import nnls, least_squares
from analyze import Calibrator, RecallModel, recall_features, staircase, k_exp, floor_s, RECALL_OK

DT = {np.dtype(np.float32): 0, np.dtype(np.float64): 1, np.dtype(np.int32): 2, np.dtype(np.uint8): 3,
      np.dtype(np.int16): 4}


def write_model(path, entries):
    with open(path, "wb") as f:
        f.write(b"HVO1" + struct.pack("<I", len(entries)))
        for name, arr in entries:
            arr = np.ascontiguousarray(arr)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<B", DT[arr.dtype]) + struct.pack("<I", arr.ndim))
            for s in arr.shape:
                f.write(struct.pack("<q", s))
            f.write(arr.tobytes())


def main(name):
    T = load_table(name)
    G = load(name, QFILE)
    E = load(name, EFILE)
    S = load(name, "sketch_hpbs.pkl")
    N = T.N
    split = G["split"]
    tr = split == "train"
    pre, post = load(name, f"pg_pre{KS}.pkl"), load(name, f"pg_post{KS}.pkl")
    idx = np.flatnonzero(tr)
    idx = np.array([i for i in idx if i in pre and i in post])
    t_pre = np.array([pre[i]["t"] for i in idx]); t_post = np.array([post[i]["t"] for i in idx])
    rec_post = np.array([post[i]["recall"] for i in idx])
    sg, sl = G["sg"][idx], floor_s(G["sloc"][idx], N / K)
    nterm = np.array([G["P"][i].n_terms() for i in idx], float)
    join = np.array([bool(G["P"][i].dim_ins) for i in idx], float)

    # pre: a0 + c1 N terms + c2 N join + cscan s_g N   (relative-error NNLS)
    Mx = np.stack([np.ones(len(idx)), N * nterm, N * join, sg * N], 1)
    w = 1.0 / t_pre
    c = nnls(Mx * w[:, None], t_pre * w)[0]
    # post: staircase in the k-th-match horizon
    need = k_exp(sl)
    def res(p):
        return np.log(staircase(p, need, 0)) - np.log(t_post)
    x0 = np.clip([np.log(t_post.min() / 2), np.log(t_post.mean() / need.mean()), 0.9], -39, 39)
    post_p = least_squares(res, x0=x0, bounds=([-40, -40, 0.3], [40, 40, 2.5])).x

    # estimator: calibrator + interval width from the training split (engine independent)
    rows = E["rows"]
    raw = np.array([[r["hpbs"][0], r["hpbs"][1]] for r in rows])
    F = np.array([r["hpbs_feat"] for r in rows])
    tau = np.full(len(rows), N / K)
    cal = Calibrator().fit(F[tr], raw[tr, 1], G["sloc"][tr], tau[tr])
    s_cal = cal.apply(F, raw[:, 1], tau)
    eta = float(np.quantile(qerr(s_cal[tr], G["sloc"][tr], tau[tr]), 0.9))
    # recall model for pgvector's iterative post-filter
    RM = RecallModel().fit(recall_features(raw[idx, 0], s_cal[idx], nterm, join), (rec_post >= RECALL_OK)[:, None])
    (w0, wv), = RM.coefs()
    TAU_REC = 0.9
    # cost[10] = logistic intercept (NaN: constant probability in cost[11]); cost[11:15] = coefficients
    rec = [np.nan, float(w0)] if wv is None else [float(w0)] + list(map(float, wv))
    cost = np.zeros(16)
    cost[:10] = [0.0, c[1], c[2], c[0], c[3], *post_p, eta, TAU_REC]
    cost[10:10 + len(rec)] = rec

    ent = [("L", np.array([S.L], np.int32)), ("m", np.array([S.m], np.int32)),
           ("d", np.array([T.d], np.int32)), ("B1", np.array([S.B + 1], np.int32)), ("N", np.array([N], np.float64))]
    num = list(S.edges)
    cat = [c_ for c_ in S.kinds if S.kinds[c_] == "cat"]
    dimc = list(T.dim_cols)
    enc = lambda xs: np.frombuffer("\n".join(xs).encode(), np.uint8)
    ent += [("num_names", enc([pgname(x) for x in num])), ("cat_names", enc([pgname(x) for x in cat])),
            ("dim_names", enc([pgname(x) for x in dimc])),
            # column kind per numeric column: 0 continuous, 1 integer, 2 point mass
            ("integral", np.array([{"int": 1, True: 1, "mass": 2}.get(S.integral.get(x, False), 0) for x in num],
                                  np.uint8))]
    ent += [(f"edges.{j}", S.edges[x].astype(np.float32)) for j, x in enumerate(num)]
    ent += [(f"dim.{j}", T.dim_cols[x].astype(np.int16)) for j, x in enumerate(dimc)]
    for l, P in enumerate(S.parts):
        H = S.hist[l]
        ent += [(f"p{l}.cent", P.cent.astype(np.float32)), (f"p{l}.c2", P.c2.astype(np.float32)),
                (f"p{l}.Nb", P.Nb.astype(np.float32)), (f"p{l}.rq", P.rq.astype(np.float32))]
        ent += [(f"p{l}.num{j}", H[x].astype(np.float32)) for j, x in enumerate(num)]
        ent += [(f"p{l}.cat{j}", H[x].astype(np.float32)) for j, x in enumerate(cat)]
        ent += [(f"p{l}.keys", S.kmv_keys[l].astype(np.int32)), (f"p{l}.kcnt", S.kmv_cnt[l].astype(np.float32))]
    W, b = cal.m.coefs_, cal.m.intercepts_
    ent += [("cal.mu", cal.mu.astype(np.float32)), ("cal.sd", cal.sd.astype(np.float32)),
            ("cal.W0", W[0].astype(np.float32)), ("cal.b0", b[0].astype(np.float32)),
            ("cal.W1", W[1].astype(np.float32)), ("cal.b1", b[1].astype(np.float32)),
            ("cal.W2", W[2].ravel().astype(np.float32)), ("cal.b2", b[2].astype(np.float32)),
            ("cost", cost)]
    path = os.path.join(ddir(name), f"hvo_pg_model{KS}.bin")
    write_model(path, ent)
    info = dict(pre=dict(a0_us=c[0] * 1e6, ns_per_row_term=c[1] * 1e9, ns_per_row_join=c[2] * 1e9,
                         scan_ns_per_vec=c[3] * 1e9),
                post=dict(b0_us=float(np.exp(post_p[0]) * 1e6), b1_ns=float(np.exp(post_p[1]) * 1e9),
                          gamma=float(post_p[2])), eta=eta, recall_model=RM.coefs(), n_train=len(idx))
    with open(os.path.join(RESULTS, f"{name}{KS}_pg_model.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
