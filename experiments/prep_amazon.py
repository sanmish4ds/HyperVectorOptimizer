"""Build the second real-data benchmark AMAZON-384 from Amazon Reviews 2023 item metadata
(McAuley Lab; departments Video Games, Musical Instruments, Baby Products, Industrial & Scientific).

Vectors: all-MiniLM-L6-v2 (384-d, unit norm) of "title. features. description", truncated to 128 tokens.
Fact attributes (real metadata): dept, main_category, subcat (cat); price, avg_rating (num, continuous);
n_ratings, n_features, desc_words (num, integer).
Foreign key: store -> dimension table of stores with size (quartile of #items) and quality (quartile
of mean rating).
990,000 products form the table; 3,000 further products are queries.

Query predicates are *anchored on the query product* rather than drawn independently: a shopper
looking at a product filters by its department, a price band around its price, a minimum rating,
popularity, or a seller tier. With probability 0.2 the department filter names another department
(cross-category recommendation), which produces anti-correlated queries.

usage: python experiments/prep_amazon.py   (needs data/raw/amazon/meta_*.jsonl)
"""
import os
import json
import time
from collections import Counter
from common import *  # noqa: F401,F403
import numpy as np
from hvo.data import Pred

NAME = "amazon-384"
DEPTS = ["Video_Games", "Musical_Instruments", "Baby_Products", "Industrial_and_Scientific"]
N_TABLE, N_POOL = 990_000, 3_000


def text_of(r):
    parts = [r.get("title") or ""]
    parts += r.get("features") or []
    parts += r.get("description") or []
    return " ".join(" ".join(parts).split())


def main():
    t0 = time.time()
    recs = []
    for di, dep in enumerate(DEPTS):
        with open(os.path.join(DATA, "raw", "amazon", f"meta_{dep}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                cats = r.get("categories") or []
                recs.append(dict(dept=di, main=r.get("main_category") or "none",
                                 sub=cats[1] if len(cats) > 1 else "none",
                                 price=float(r["price"]) if isinstance(r.get("price"), (int, float)) else 0.0,
                                 rating=float(r.get("average_rating") or 0.0),
                                 n_ratings=float(r.get("rating_number") or 0),
                                 n_features=float(len(r.get("features") or [])),
                                 desc_words=float(len(" ".join(r.get("description") or []).split())),
                                 store=r.get("store") or "none", text=text_of(r)))
    print("parsed", len(recs), f"{time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(DATASETS[NAME][2])
    pick = rng.choice(len(recs), size=N_TABLE + N_POOL, replace=False)
    table, pool = np.sort(pick[:N_TABLE]), np.sort(pick[N_TABLE:])

    def topmap(key, n=31):
        top = [v for v, _ in Counter(recs[i][key] for i in table).most_common(n)]
        return {v: j for j, v in enumerate(top)}, top + ["other"]
    mmap, mnames = topmap("main")
    smap, snames = topmap("sub")
    stores = sorted(set(recs[i]["store"] for i in pick))
    stmap = {s: j for j, s in enumerate(stores)}

    def attrs(rows):
        return {"dept": np.array([recs[i]["dept"] for i in rows], np.int16),
                "main_category": np.array([mmap.get(recs[i]["main"], 31) for i in rows], np.int16),
                "subcat": np.array([smap.get(recs[i]["sub"], 31) for i in rows], np.int16),
                "price": np.array([recs[i]["price"] for i in rows], np.float32),
                "avg_rating": np.array([recs[i]["rating"] for i in rows], np.float32),
                "n_ratings": np.array([recs[i]["n_ratings"] for i in rows], np.float32),
                "n_features": np.array([recs[i]["n_features"] for i in rows], np.float32),
                "desc_words": np.array([recs[i]["desc_words"] for i in rows], np.float32)}
    cols = attrs(table)
    kinds = {"dept": "cat", "main_category": "cat", "subcat": "cat", "price": "num", "avg_rating": "num",
             "n_ratings": "num", "n_features": "num", "desc_words": "num"}
    fk = np.array([stmap[recs[i]["store"]] for i in table], np.int32)
    n_items = np.bincount(fk, minlength=len(stores))
    rsum = np.bincount(fk, weights=cols["avg_rating"], minlength=len(stores))
    mean_r = np.where(n_items > 0, rsum / np.maximum(n_items, 1), 0)
    q4 = lambda v: np.searchsorted(np.quantile(v, [0.25, 0.5, 0.75]), v, side="right").astype(np.int16)
    dim_cols = {"size": q4(n_items), "quality": q4(mean_r)}

    # query-anchored predicates for the pool
    P = attrs(pool)
    pool_fk = np.array([stmap[recs[i]["store"]] for i in pool], np.int32)
    pq = lambda c, u: float(np.quantile(cols[c], u))
    preds = []
    for j in range(len(pool)):
        ranges, ins, dim_ins = {}, {}, {}
        kinds_ = ["dept", "price", "rating", "popular", "main", "seller"]
        nterm = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2])
        for t in rng.choice(kinds_, size=nterm, replace=False):
            if t == "dept":
                d = int(P["dept"][j]) if rng.random() < 0.8 else int(rng.choice([x for x in range(4) if x != P["dept"][j]]))
                ins["dept"] = (d,)
            elif t == "main":
                ins["main_category"] = (int(P["main_category"][j]),)
            elif t == "price":
                p0 = float(P["price"][j])
                if p0 > 0:
                    w = rng.uniform(0.2, 1.0)
                    ranges["price"] = (p0 * (1 - w / 2), p0 * (1 + w))
                else:
                    ranges["price"] = (0.01, pq("price", rng.uniform(0.5, 0.95)))
            elif t == "rating":
                ranges["avg_rating"] = (float(np.clip(P["avg_rating"][j] - rng.uniform(0, 1), 1, 4.8)), 5.0)
            elif t == "popular":
                ranges["n_ratings"] = (pq("n_ratings", rng.uniform(0.3, 0.9)), float(cols["n_ratings"].max()))
            elif t == "seller":
                if rng.random() < 0.5:
                    dim_ins["size"] = tuple(range(int(rng.integers(1, 4)), 4))
                else:
                    dim_ins["quality"] = tuple(range(int(rng.integers(1, 4)), 4))
        preds.append(Pred(ranges, ins, dim_ins))

    D = ddir(NAME)
    save(dict(cols=cols, kinds=kinds, fk=fk, dim_cols=dim_cols, topic=None), NAME, "table_meta.pkl")
    save(dict(preds=preds, main_names=mnames, sub_names=snames, n_stores=len(stores)), NAME, "pool_preds.pkl")
    print("attributes saved", f"{time.time() - t0:.0f}s", flush=True)
    if os.path.exists(os.path.join(D, "X.npy")) and os.path.exists(os.path.join(D, "Qpool.npy")):
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
    model.max_seq_length = 128

    def embed(rows, path):
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), 384))
        B = 50_000
        for s in range(0, len(rows), B):
            e = model.encode([recs[i]["text"] or "untitled" for i in rows[s:s + B]], batch_size=256,
                             normalize_embeddings=True)
            out[s:s + len(e)] = e
            print("embedded", s + len(e), f"{time.time() - t0:.0f}s", flush=True)
        out.flush()
    embed(pool, os.path.join(D, "Qpool.npy"))
    embed(table, os.path.join(D, "X.npy"))
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
