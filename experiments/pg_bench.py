"""Run the benchmark queries through PostgreSQL + pgvector under one configuration.

Each query is executed once as a warm-up and twice timed (client wall time, which includes planning
and therefore HVO's own planning cost); the slower timed run is kept. Recall@k is measured against the
exact filtered top-k. In "hvo" mode the hook's decision and planning time are read back with hvo_last().

usage: python experiments/pg_bench.py arxiv-384 MODE [split] [port]
       MODE in stock, iter, pre, post, hvo, exqutor; split in train, test, all (default all)
writes data/<name>/pg_<mode>[_kK].pkl
"""
import sys
import time
from pg_common import *  # noqa: F401,F403
import numpy as np
from hvo import plans


def main(name, mode, split="all", port=5432):
    T = load_table(name)
    G = load(name, QFILE)
    conn = connect(name, port)
    cur = conn.cursor()
    for s in COMMON + MODES.get(mode, []):
        cur.execute(s)
    if mode.startswith("hvo"):
        cur.execute("CREATE EXTENSION IF NOT EXISTS hvo_pg")
        cur.execute(f"SET hvo.model = '{os.path.join(ddir(name), f'hvo_pg_model{KS}.bin')}'")
    idx = [i for i in range(len(G["Q"])) if split == "all" or G["split"][i] == split]
    out = {}
    t0 = time.time()
    for n_, i in enumerate(idx):
        sql = to_sql(T, G["P"][i], G["Q"][i], K)
        cur.execute(sql); cur.fetchall()                      # warm-up
        ts, res = [], None
        for _ in range(2):
            t = time.perf_counter()
            cur.execute(sql)
            rows = cur.fetchall()
            ts.append(time.perf_counter() - t)
            res = [r[0] for r in rows]
        r = dict(t=max(ts), recall=plans.recall(np.array(res, np.int64), G["topk"][i]), n=len(res))
        if mode.startswith("hvo"):
            cur.execute("SELECT hvo_last()")
            ch, us, sg, sl = cur.fetchone()[0].split()
            r.update(choice=int(ch), plan_us=float(us))
        out[i] = r
        if n_ % 100 == 0:
            print(mode, n_, f"{time.time() - t0:.0f}s", f"{r['t'] * 1e3:.2f}ms rec {r['recall']:.2f}", flush=True)
    path_name = f"pg_{mode}{KS}.pkl"
    prev = load(name, path_name) if os.path.exists(os.path.join(ddir(name), path_name)) else {}
    prev.update(out)
    save(prev, name, path_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all",
         int(sys.argv[4]) if len(sys.argv) > 4 else 5432)
