"""Load a benchmark into PostgreSQL + pgvector: items(id, embedding, attributes..., fk) and dims(id, ...),
an HNSW index with pgvector's defaults (m = 16, ef_construction = 64, inner product) and B-tree
indexes on every attribute and on fk.

usage: python experiments/pg_load.py arxiv-384 [port]
"""
import io
import sys
import time
import subprocess
from pg_common import *  # noqa: F401,F403
import numpy as np


def main(name, port=5432):
    T = load_table(name)
    db = dbname(name)
    subprocess.call([f"{PG_BIN}/createdb", "-p", str(port), "-h", "localhost", db])
    conn = connect(name, port)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("SELECT to_regclass('items') IS NOT NULL")
    loaded = cur.fetchone()[0]
    if loaded:
        cur.execute("SELECT count(*) FROM items")
        loaded = cur.fetchone()[0] == T.N
    if loaded:
        print("items already loaded; building missing indexes only", flush=True)
        return build_indexes(cur, T, time.time())
    cur.execute("DROP TABLE IF EXISTS items; DROP TABLE IF EXISTS dims")
    cols = [(pgname(c), "real" if T.kinds[c] == "num" else "smallint") for c in T.cols]
    cur.execute("CREATE TABLE items (id int PRIMARY KEY, embedding vector(%d), %s, fk int)"
                % (T.d, ", ".join(f"{c} {t}" for c, t in cols)))
    dcols = [pgname(c) for c in T.dim_cols]
    cur.execute("CREATE TABLE dims (id int PRIMARY KEY, %s)" % ", ".join(f"{c} smallint" for c in dcols))
    nd = len(next(iter(T.dim_cols.values())))
    buf = io.StringIO()
    for i in range(nd):
        buf.write("\t".join([str(i)] + [str(int(v[i])) for v in T.dim_cols.values()]) + "\n")
    buf.seek(0)
    cur.copy_expert("COPY dims FROM STDIN", buf)
    t0 = time.time()
    names = list(T.cols)
    B = 50_000
    for s in range(0, T.N, B):
        e = min(T.N, s + B)
        buf = io.StringIO()
        Xs = T.X[s:e]
        vals = [T.cols[c][s:e] for c in names]
        for j in range(e - s):
            row = [str(s + j), vec_literal(Xs[j])] + [repr(float(v[j])) if T.kinds[c] == "num" else str(int(v[j]))
                                                       for c, v in zip(names, vals)] + [str(int(T.fk[s + j]))]
            buf.write("\t".join(row) + "\n")
        buf.seek(0)
        cur.copy_expert("COPY items FROM STDIN", buf)
        print("loaded", e, f"{time.time() - t0:.0f}s", flush=True)
    build_indexes(cur, T, t0)


def build_indexes(cur, T, t0):
    """HNSW (pgvector defaults) + B-tree indexes; skips any that already exist."""
    cur.execute("SET maintenance_work_mem = '3GB'")
    cur.execute("SET max_parallel_maintenance_workers = 8")
    t = time.time()
    cur.execute("CREATE INDEX IF NOT EXISTS items_hnsw ON items USING hnsw (embedding vector_ip_ops)")
    hnsw_s = time.time() - t
    for c in T.cols:
        cur.execute(f"CREATE INDEX IF NOT EXISTS items_{pgname(c)} ON items ({pgname(c)})")
    cur.execute("CREATE INDEX IF NOT EXISTS items_fk ON items (fk)")
    cur.execute("VACUUM ANALYZE items")
    cur.execute("VACUUM ANALYZE dims")
    print("hnsw build", f"{hnsw_s:.0f}s", "total", f"{time.time() - t0:.0f}s", flush=True)
    name = [n for n in DATASETS if dbname(n) == cur.connection.info.dbname][0]
    with open(os.path.join(ddir(name), "pg_load.json"), "w") as f:
        f.write(f'{{"hnsw_build_s": {hnsw_s}}}\n')


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5432)
