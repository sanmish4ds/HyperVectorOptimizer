"""Shared helpers for the PostgreSQL / pgvector experiments."""
import os
import numpy as np
import psycopg2
from common import *  # noqa: F401,F403

PG_BIN = os.environ.get("HVO_PG_BIN", "/opt/homebrew/opt/postgresql@16/bin")
RESERVED = {"group": "grp", "size": "sz"}       # SQL keywords used as attribute names


def pgname(c):
    return RESERVED.get(c, c)


def dbname(name):
    return "hvo_" + name.replace("-", "_")


def connect(name, port=5432):
    conn = psycopg2.connect(dbname=dbname(name), host="localhost", port=port)
    conn.autocommit = True
    return conn


def vec_literal(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def to_sql(T, pred, q, k):
    """The benchmark predicate as SQL against items(...) and dims(...)."""
    w = []
    for c, (lo, hi) in pred.ranges.items():
        w.append(f"{pgname(c)} BETWEEN {lo!r} AND {hi!r}")
    for c, vals in pred.ins.items():
        w.append(f"{pgname(c)} IN ({','.join(map(str, vals))})")
    if pred.dim_ins:
        dw = " AND ".join(f"{pgname(c)} IN ({','.join(map(str, v))})" for c, v in pred.dim_ins.items())
        w.append(f"fk IN (SELECT id FROM dims WHERE {dw})")
    where = (" WHERE " + " AND ".join(w)) if w else ""
    return f"SELECT id FROM items{where} ORDER BY embedding <#> '{vec_literal(q)}' LIMIT {k}"


# session settings per configuration; every configuration runs single-threaded
COMMON = ["SET max_parallel_workers_per_gather = 0", "SET jit = off"]
MODES = {
    # pgvector as shipped: no iterative scan, ef_search 40
    "stock": [],
    # pgvector's recommended setting for filtered queries
    "iter": ["SET hnsw.iterative_scan = strict_order"],
    # forced plans (the oracle takes the faster one with Recall@10 >= 0.9)
    "pre": ["SET enable_indexscan = off"],
    "post": ["SET enable_seqscan = off", "SET enable_bitmapscan = off", "SET hnsw.iterative_scan = strict_order",
             "SET hnsw.max_scan_tuples = 131072", "SET hnsw.ef_search = 64"],
    # HVO planner hook: decides pre vs post per query, post uses the same iterative settings
    "hvo": ["LOAD 'hvo_pg'", "SET hvo.enabled = on", "SET hnsw.iterative_scan = strict_order",
            "SET hnsw.max_scan_tuples = 131072", "SET hnsw.ef_search = 64"],
    # ablation: HVO without the recall constraint (pure minimax latency routing)
    "hvo_norec": ["LOAD 'hvo_pg'", "SET hvo.enabled = on", "SET hvo.recall_tau = 0",
                  "SET hnsw.iterative_scan = strict_order", "SET hnsw.max_scan_tuples = 131072",
                  "SET hnsw.ef_search = 64"],
}
