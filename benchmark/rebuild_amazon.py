"""Rebuild AMAZON-384 from your own copy of the Amazon Reviews 2023 item metadata.

The release ships only product ids (parent_asin), filters and ground truth. This script reads the four
department files (meta_<Dept>.jsonl from https://amazon-reviews-2023.github.io), recomputes the
attributes and the store dimension table exactly as the benchmark did, checks them against the SHA-256
sums in schema.json, and (with --embed) recomputes the vectors with all-MiniLM-L6-v2.

usage: python rebuild_amazon.py RAW_DIR RELEASE_DIR [--embed] [--device cpu|mps|cuda]
writes into RELEASE_DIR: base_attributes.parquet, dim_stores.parquet, and with --embed
base_vectors.npy, query_vectors.npy
"""
import os
import sys
import json
import hashlib
import argparse
import numpy as np
import pandas as pd


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def text_of(r):
    parts = [r.get("title") or ""]
    parts += r.get("features") or []
    parts += r.get("description") or []
    return " ".join(" ".join(parts).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("release")
    ap.add_argument("--embed", action="store_true"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.release, "schema.json")))
    base_ids = open(os.path.join(a.release, "base_ids.txt")).read().split()
    queries = pd.read_parquet(os.path.join(a.release, "queries.parquet"))
    pool_ids = list(queries["source_id"])
    sample_pool = open(os.path.join(a.release, "pool_ids.txt")).read().split()   # all 3,000 held-out products
    want = set(base_ids) | set(sample_pool)
    depts = S["source"]["departments"]

    recs = {}
    for di, dep in enumerate(depts):
        with open(os.path.join(a.raw, f"meta_{dep}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["parent_asin"] in want:
                    cats = r.get("categories") or []
                    recs[r["parent_asin"]] = dict(
                        dept=di, main=r.get("main_category") or "none", sub=cats[1] if len(cats) > 1 else "none",
                        price=float(r["price"]) if isinstance(r.get("price"), (int, float)) else 0.0,
                        rating=float(r.get("average_rating") or 0.0), n_ratings=float(r.get("rating_number") or 0),
                        n_features=float(len(r.get("features") or [])),
                        desc_words=float(len(" ".join(r.get("description") or []).split())),
                        store=r.get("store") or "none", text=text_of(r))
        print("read", dep, len(recs), flush=True)
    missing = want - set(recs)
    if missing:
        sys.exit(f"{len(missing)} ids not found in {a.raw} (wrong files or version?)")

    A = S["attributes"]
    mmap = {v: j for j, v in enumerate(A["main_category"]["codes"][:-1])}
    smap = {v: j for j, v in enumerate(A["subcat"]["codes"][:-1])}
    rows = [recs[i] for i in base_ids]
    cols = {"dept": np.array([r["dept"] for r in rows], np.int16),
            "main_category": np.array([mmap.get(r["main"], 31) for r in rows], np.int16),
            "subcat": np.array([smap.get(r["sub"], 31) for r in rows], np.int16),
            "price": np.array([r["price"] for r in rows], np.float32),
            "avg_rating": np.array([r["rating"] for r in rows], np.float32),
            "n_ratings": np.array([r["n_ratings"] for r in rows], np.float32),
            "n_features": np.array([r["n_features"] for r in rows], np.float32),
            "desc_words": np.array([r["desc_words"] for r in rows], np.float32)}
    # store keys: sorted store names of the whole sample (table + all held-out products)
    stores = sorted(set(r["store"] for r in rows) | set(recs[i]["store"] for i in sample_pool))
    n_dim = S["dimension_table"]["n_rows"]
    stmap = {s: j for j, s in enumerate(stores)}
    fk = np.array([stmap[r["store"]] for r in rows], np.int32)

    ok = True
    for c, v in cols.items():
        good = sha(v) == S["sha256"][f"attr:{c}"]
        ok &= good
        print(f"  {c:13s} {'ok' if good else 'MISMATCH'}")
    if len(stores) == n_dim and sha(fk) == S["sha256"]["fk"]:
        n_items = np.bincount(fk, minlength=n_dim)
        rsum = np.bincount(fk, weights=cols["avg_rating"], minlength=n_dim)
        mean_r = np.where(n_items > 0, rsum / np.maximum(n_items, 1), 0)
        q4 = lambda v: np.searchsorted(np.quantile(v, [0.25, 0.5, 0.75]), v, side="right").astype(np.int16)
        dim = {"size": q4(n_items), "quality": q4(mean_r)}
        for c, v in dim.items():
            good = sha(v) == S["sha256"][f"dim:{c}"]
            ok &= good
            print(f"  dim:{c:9s} {'ok' if good else 'MISMATCH'}")
        print("  fk            ok")
    else:
        ok = False
        print("  fk            MISMATCH (store set differs from the benchmark's)")
    if not ok:
        sys.exit("checksums differ: your metadata files are not the ones the benchmark was built from")

    b = pd.DataFrame({"row_id": np.arange(len(base_ids)), "source_id": base_ids})
    for c, v in cols.items():
        b[c] = v
    b["store_id"] = fk
    b.to_parquet(os.path.join(a.release, "base_attributes.parquet"), index=False)
    pd.DataFrame({"store_id": np.arange(n_dim), **dim}).to_parquet(
        os.path.join(a.release, "dim_stores.parquet"), index=False)
    print("attributes verified and written")

    if a.embed:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=a.device)
        model.max_seq_length = 128
        for ids, fn in ((pool_ids, "query_vectors.npy"), (base_ids, "base_vectors.npy")):
            out = np.lib.format.open_memmap(os.path.join(a.release, fn), mode="w+", dtype=np.float32,
                                            shape=(len(ids), 384))
            for s in range(0, len(ids), 50_000):
                out[s:s + 50_000] = model.encode([recs[i]["text"] or "untitled" for i in ids[s:s + 50_000]], batch_size=256,
                                                 normalize_embeddings=True)
                print("embedded", fn, min(s + 50_000, len(ids)), flush=True)
            out.flush()
        print("vectors written; run evaluate.py --check-gt to confirm the ground truth on your vectors")


if __name__ == "__main__":
    main()
