"""Build the real-data benchmark ARXIV-384 from gfissore/arxiv-abstracts-2021 (CC0).

Vectors: all-MiniLM-L6-v2 embeddings (384-d, unit norm) of "title. abstract", truncated to 128 tokens.
Fact attributes (all real metadata):
  archive    (cat) top-level archive of the primary category (top 31 + "other")
  published  (cat) has a journal reference
  year       (num) submission year, from the arXiv identifier
  n_cats     (num) number of categories the paper is listed in
  n_authors  (num) number of authors
  abs_words  (num) abstract length in words
Foreign key: primary category -> dimension table of categories with
  group (subject group: physics, math, cs, ...) and size (quartile of category popularity).
1,000,000 papers form the table; 3,000 further papers (disjoint) are the query pool.

usage: python experiments/prep_arxiv.py   (needs data/raw/arxiv/arxiv-abstracts.jsonl.gz)
"""
import os
import re
import gzip
import json
import time
from collections import Counter
from common import *  # noqa: F401,F403
import numpy as np

NAME = "arxiv-384"
SRC = os.path.join(DATA, "raw", "arxiv", "arxiv-abstracts.jsonl.gz")
N_TABLE, N_POOL = 1_000_000, 3_000
GROUPS = {"cs": "cs", "math": "math", "stat": "stat", "q-bio": "q-bio", "q-fin": "q-fin", "eess": "eess",
          "econ": "econ"}  # every other archive is physics


def year_of(aid):
    m = re.match(r"^(\d{2})(\d{2})\.", aid)            # new style YYMM.NNNNN
    if not m:
        m = re.match(r"^[a-z\-\.A-Z]+/(\d{2})(\d{2})", aid)  # old style archive/YYMMNNN
    yy = int(m.group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


def n_authors(s):
    s = re.sub(r"\(.*?\)", "", s.replace("\n", " "))
    return max(1, len([a for a in re.split(r",| and ", s) if a.strip()]))


def main():
    t0 = time.time()
    meta = []
    with gzip.open(SRC, "rt") as f:
        for line in f:
            r = json.loads(line)
            # categories come as a list holding one space-separated string, e.g. ["math.CO cs.CG"]
            cats = " ".join(r["categories"] if isinstance(r["categories"], list) else [r["categories"]]).split()
            meta.append((r["id"], cats[0], len(cats), n_authors(r["authors"] or ""), len(r["abstract"].split()),
                         int(bool(r.get("journal-ref")))))
    print("parsed", len(meta), f"{time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(DATASETS[NAME][2])
    pick = rng.choice(len(meta), size=N_TABLE + N_POOL, replace=False)
    table_rows, pool_rows = np.sort(pick[:N_TABLE]), np.sort(pick[N_TABLE:])
    want = set(pick.tolist())

    D = ddir(NAME)
    need_embed = not (os.path.exists(os.path.join(D, "X.npy")) and os.path.exists(os.path.join(D, "Qpool.npy")))
    texts = {}
    if need_embed:
        with gzip.open(SRC, "rt") as f:
            for i, line in enumerate(f):
                if i in want:
                    r = json.loads(line)
                    texts[i] = r["title"].replace("\n", " ").strip() + ". " + " ".join(r["abstract"].split())
        print("texts", len(texts), f"{time.time() - t0:.0f}s", flush=True)

    # attributes of the table rows
    prim = [meta[i][1] for i in table_rows]
    arch = [p.split(".")[0] for p in prim]
    top = [a for a, _ in Counter(arch).most_common(31)]
    amap = {a: j for j, a in enumerate(top)}
    archive = np.array([amap.get(a, 31) for a in arch], np.int16)
    cats = sorted(set(meta[i][1] for i in pick))
    cmap = {c: j for j, c in enumerate(cats)}
    fk = np.array([cmap[p] for p in prim], np.int32)
    gnames = sorted(set(GROUPS.values()) | {"physics"})
    group = np.array([gnames.index(GROUPS.get(c.split(".")[0], "physics")) for c in cats], np.int16)
    popularity = np.bincount(fk, minlength=len(cats))
    size = np.searchsorted(np.quantile(popularity, [0.25, 0.5, 0.75]), popularity, side="right").astype(np.int16)
    cols = {"archive": archive,
            "published": np.array([meta[i][5] for i in table_rows], np.int16),
            "year": np.array([year_of(meta[i][0]) for i in table_rows], np.float32),
            "n_cats": np.array([meta[i][2] for i in table_rows], np.float32),
            "n_authors": np.array([meta[i][3] for i in table_rows], np.float32),
            "abs_words": np.array([meta[i][4] for i in table_rows], np.float32)}
    kinds = {"archive": "cat", "published": "cat", "year": "num", "n_cats": "num", "n_authors": "num",
             "abs_words": "num"}

    # embeddings (skipped when X.npy and Qpool.npy already exist; the sample is seeded)
    if need_embed:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
        model.max_seq_length = 128

    def embed(rows, path):
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), 384))
        B = 50_000
        for s in range(0, len(rows), B):
            e = model.encode([texts[i] for i in rows[s:s + B]], batch_size=256, normalize_embeddings=True)
            out[s:s + len(e)] = e
            print("embedded", s + len(e), f"{time.time() - t0:.0f}s", flush=True)
        out.flush()
        return out

    if need_embed:
        embed(pool_rows, os.path.join(D, "Qpool.npy"))
        embed(table_rows, os.path.join(D, "X.npy"))
    save(dict(cols=cols, kinds=kinds, fk=fk, dim_cols={"group": group, "size": size}, topic=None), NAME,
         "table_meta.pkl")
    save(dict(table_ids=[meta[i][0] for i in table_rows], pool_ids=[meta[i][0] for i in pool_rows],
              archives=top + ["other"], categories=cats, groups=gnames), NAME, "arxiv_info.pkl")
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
