"""Export the real-data filtered-search benchmarks (ARXIV-384, AMAZON-384) in a tool-neutral format.

usage: python benchmark/export.py arxiv-384 amazon-384      (writes data/release/<name>/)

Per dataset:
  queries.parquet       query_id, split, source_id, filter (JSON), s_global, s_local_k{1,10,100}, h_k{1,10,100}
  gt_k{1,10,100}.npy    exact filtered top-k row ids (int32, -1 padded), maximum inner product
  schema.json           columns, kinds, category codes, filter semantics, sizes, checksums
ARXIV-384 (source CC0) additionally ships the data itself:
  base_vectors.npy      float32 (N, 384), unit norm; base_attributes.parquet; dim_categories.parquet
  query_vectors.npy     float32 (Q, 384)
AMAZON-384 ships only product ids (the source states no license); rebuild_amazon.py reconstructs
vectors and attributes from the user's own copy of Amazon Reviews 2023 and checks the checksums.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, ROOT)
from common import DATA, DATASETS, ddir, load  # noqa: E402

KS = {1: "_k1", 10: "", 100: "_k100"}
AMAZON_DEPTS = ["Video_Games", "Musical_Instruments", "Baby_Products", "Industrial_and_Scientific"]
FILTER_SEMANTICS = ("A row matches when every term holds: ranges {col: [lo, hi]} are closed intervals "
                    "lo <= value <= hi; in {col: [codes]} requires the row's code to be listed; dim_in "
                    "{col: [codes]} is a semi-join: the row's foreign key must point to a dimension row "
                    "whose code is listed.")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def filter_json(p):
    return json.dumps({"ranges": {str(c): [float(lo), float(hi)] for c, (lo, hi) in p.ranges.items()},
                       "in": {str(c): [int(v) for v in vals] for c, vals in p.ins.items()},
                       "dim_in": {str(c): [int(v) for v in vals] for c, vals in p.dim_ins.items()}},
                      sort_keys=True)


def pool_index(Q, Qpool):
    """Position of each query vector in the held-out pool (queries are pool rows, some dropped)."""
    idx = np.full(len(Q), -1)
    j = 0
    for i, q in enumerate(Q):
        while not np.array_equal(Qpool[j], q):
            j += 1
        idx[i] = j
        j += 1
    return idx


def amazon_ids():
    """parent_asin of every parsed record, in prep_amazon.py's order, and the seeded table/pool split."""
    ids = []
    for dep in AMAZON_DEPTS:
        with open(os.path.join(DATA, "raw", "amazon", f"meta_{dep}.jsonl")) as f:
            for line in f:
                ids.append(json.loads(line)["parent_asin"])
    rng = np.random.default_rng(DATASETS["amazon-384"][2])
    n_table = DATASETS["amazon-384"][0]
    pick = rng.choice(len(ids), size=n_table + 3_000, replace=False)
    table, pool = np.sort(pick[:n_table]), np.sort(pick[n_table:])
    return [ids[i] for i in table], [ids[i] for i in pool]


def main(name):
    out = os.path.join(DATA, "release", name)
    os.makedirs(out, exist_ok=True)
    D = ddir(name)
    meta = load(name, "table_meta.pkl")
    G = {k: load(name, f"queries{s}.pkl") for k, s in KS.items()}
    Q = G[10]["Q"]
    for k, g in G.items():
        assert np.array_equal(g["Q"], Q), f"query set differs at k={k}"
    Qpool = np.load(os.path.join(D, "Qpool.npy"), mmap_mode="r")
    pidx = pool_index(Q, Qpool)

    if name == "arxiv-384":
        info = load(name, "arxiv_info.pkl")
        base_ids, pool_ids = info["table_ids"], info["pool_ids"]
        codes = {"archive": info["archives"], "published": ["no", "yes"]}
        dim = {"name": "categories", "key": "category_id", "names": info["categories"],
               "codes": {"group": info["groups"], "size": ["q1 (rarest)", "q2", "q3", "q4 (most popular)"]}}
        source = {"name": "gfissore/arxiv-abstracts-2021", "license": "CC0-1.0",
                  "text": "title + '. ' + abstract", "id": "arXiv identifier"}
    else:
        base_ids, pool_ids = amazon_ids()
        pp = load(name, "pool_preds.pkl")
        codes = {"dept": AMAZON_DEPTS, "main_category": pp["main_names"], "subcat": pp["sub_names"]}
        dim = {"name": "stores", "key": "store_id", "names": None,
               "codes": {"size": ["q1 (fewest items)", "q2", "q3", "q4 (most items)"],
                         "quality": ["q1 (lowest mean rating)", "q2", "q3", "q4 (highest mean rating)"]}}
        source = {"name": "McAuley-Lab/Amazon-Reviews-2023 (item metadata)", "license": "not stated by the source",
                  "text": "title + features + description, whitespace-joined", "id": "parent_asin",
                  "departments": AMAZON_DEPTS}
    N = len(base_ids)
    assert N == len(meta["fk"]) == DATASETS[name][0]

    # queries
    q = pd.DataFrame({"query_id": np.arange(len(Q)), "split": G[10]["split"].astype(str),
                      "source_id": [pool_ids[i] for i in pidx],
                      "filter": [filter_json(p) for p in G[10]["P"]],
                      "s_global": G[10]["sg"].astype(np.float64)})
    for k, g in G.items():
        q[f"s_local_k{k}"] = g["sloc"].astype(np.float64)
        q[f"h_k{k}"] = g["hk"].astype(np.int64)
        gt = np.full((len(Q), k), -1, np.int32)
        for i, t in enumerate(g["topk"]):
            gt[i, :len(t)] = t
        np.save(os.path.join(out, f"gt_k{k}.npy"), gt)
    q.to_parquet(os.path.join(out, "queries.parquet"), index=False)

    cols = {c: v for c, v in meta["cols"].items()}
    checks = {f"attr:{c}": sha(v) for c, v in cols.items()}
    checks["fk"] = sha(meta["fk"])
    checks.update({f"dim:{c}": sha(v) for c, v in meta["dim_cols"].items()})

    if name == "arxiv-384":
        dst = os.path.join(out, "base_vectors.npy")
        if not os.path.exists(dst):
            os.link(os.path.join(D, "X.npy"), dst)          # same bytes, no extra disk
        np.save(os.path.join(out, "query_vectors.npy"), np.ascontiguousarray(Q, np.float32))
        b = pd.DataFrame({"row_id": np.arange(N), "source_id": base_ids})
        for c, v in cols.items():
            b[c] = v
        b[dim["key"]] = meta["fk"]
        b.to_parquet(os.path.join(out, "base_attributes.parquet"), index=False)
        d = pd.DataFrame({dim["key"]: np.arange(len(dim["names"])), "category": dim["names"]})
        for c, v in meta["dim_cols"].items():
            d[c] = v
        d.to_parquet(os.path.join(out, "dim_categories.parquet"), index=False)
    else:
        with open(os.path.join(out, "base_ids.txt"), "w") as f:
            f.write("\n".join(base_ids) + "\n")
        with open(os.path.join(out, "pool_ids.txt"), "w") as f:     # all held-out products (store keys)
            f.write("\n".join(pool_ids) + "\n")
        checks["base_vectors_rows"] = N

    schema = {
        "name": name, "n_base": N, "n_queries": int(len(Q)), "dim": 384,
        "split_counts": {s: int((q["split"] == s).sum()) for s in ("train", "test")},
        "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "max_seq_length": 128,
                      "normalized": True},
        "distance": "maximum inner product (equivalently cosine; vectors are unit norm)",
        "source": source,
        "attributes": {c: {"kind": meta["kinds"][c], "dtype": str(v.dtype),
                           **({"codes": codes[c]} if c in codes else {})} for c, v in cols.items()},
        "dimension_table": {"name": dim["name"], "foreign_key": dim["key"],
                            "n_rows": int(len(next(iter(meta["dim_cols"].values())))),
                            "columns": {c: {"codes": dim["codes"].get(c)} for c in meta["dim_cols"]}},
        "filter_semantics": FILTER_SEMANTICS,
        "ground_truth": "gt_k{k}.npy: exact top-k among matching rows by inner product, row ids into the base "
                        "table, -1 where fewer than k rows match. h_k{k}: number of base rows (matching or not) "
                        "scoring at least the k-th best matching row, i.e. its rank in the unfiltered order "
                        "(k-th = last match when fewer than k match); s_local_k{k} = min(1, k / h_k{k}), "
                        "the k-th-match local selectivity.",
        "sha256": checks,
    }
    with open(os.path.join(out, "schema.json"), "w") as f:
        json.dump(schema, f, indent=1)
    print(name, "queries", len(Q), "->", out)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
