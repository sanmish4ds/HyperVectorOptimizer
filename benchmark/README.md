---
license: cc0-1.0
pretty_name: Correlated Filtered Vector Search (ARXIV-384, AMAZON-384)
task_categories:
- feature-extraction
tags:
- vector-search
- filtered-search
- approximate-nearest-neighbor
- benchmark
size_categories:
- 1M<n<10M
---

# Correlated Filtered Vector Search Benchmark

These are two real-data benchmarks for **filtered top-k vector search**: queries of the form "the k nearest
vectors among rows that satisfy a predicate". The predicates are *correlated* with the query vector.

- Sometimes the matching rows lie right around the query (positive correlation).
- Sometimes they lie far from it (anti-correlation).

Most existing filtered-ANN benchmarks draw predicates independently of the vectors, so they miss the case
that breaks real systems. When a filter is anti-correlated with the query, an index scan followed by a filter
returns too few correct results. With HNSW and a fixed candidate budget, recall drops to 0.26–0.29 on these
queries (see the baselines below).

The benchmarks were built for the HyperVector-Optimizer (HVO) study of plan routing for filtered vector
search. Every query has exact ground truth at k ∈ {1, 10, 100}, together with its global and *local*
selectivity, so methods can be compared by correlation regime.

| | ARXIV-384 | AMAZON-384 |
|---|---|---|
| Base vectors | 1,000,000 arXiv papers | 990,000 Amazon products |
| Dimension / metric | 384, inner product (unit norm) | 384, inner product (unit norm) |
| Embedding | all-MiniLM-L6-v2, 128 tokens, title + abstract | all-MiniLM-L6-v2, 128 tokens, title + features + description |
| Attributes | archive, published, year, n_cats, n_authors, abs_words | dept, main_category, subcat, price, avg_rating, n_ratings, n_features, desc_words |
| Dimension table (semi-join) | 172 primary categories: subject group, popularity quartile | 143,933 stores: size quartile, rating quartile |
| Queries (train / test) | 2,456 (986 / 1,470), held-out papers | 2,425 (970 / 1,455), held-out products |
| Predicates | 1–3 terms, independent of the query | 1–3 terms, anchored on the query product (department, price band, rating, popularity, seller tier) |
| Regimes at k=10 (anti / indep / pos) | 1,037 / 783 / 636 | 458 / 521 / 1,446 |
| What is shipped | everything (source is CC0) | ids, filters, ground truth; a rebuild script (see below) |

## Files (per dataset directory)

| File | Content |
|---|---|
| `schema.json` | columns, kinds, category codes → names, filter semantics, source, SHA-256 checksums |
| `queries.parquet` | `query_id`, `split`, `source_id`, `filter` (JSON), `s_global`, `s_local_k{1,10,100}`, `h_k{1,10,100}` |
| `gt_k{1,10,100}.npy` | int32 `(n_queries, k)`: exact filtered top-k base row ids, `-1` padded when fewer than k rows match |
| `base_vectors.npy`, `query_vectors.npy` | float32, row-aligned with `base_attributes.parquet` / `queries.parquet` (arXiv only) |
| `base_attributes.parquet` | `row_id`, `source_id`, attribute columns, foreign key (arXiv only) |
| `dim_categories.parquet` / `dim_stores.parquet` | dimension table for the semi-join (arXiv only) |
| `base_ids.txt`, `pool_ids.txt` | Amazon `parent_asin` of the base rows / of all 3,000 held-out products |

**Filters.** A filter is `{"ranges": {col: [lo, hi]}, "in": {col: [codes]}, "dim_in": {col: [codes]}}`.
A row matches when every term holds:
- `ranges` are closed intervals;
- `in` lists the allowed category codes;
- `dim_in` is a semi-join: the row's foreign key must point to a dimension row whose code is listed.

For example, `{"ranges": {"year": [2015, 2021]}, "in": {"archive": [1]}, "dim_in": {"size": [0]}}` means
"cs papers from 2015–2021 in one of the rarest categories". In SQL:
`... WHERE year BETWEEN 2015 AND 2021 AND archive IN (1) AND category_id IN (SELECT category_id FROM categories WHERE size IN (0)) ORDER BY embedding <#> q LIMIT k`.

**Selectivity and regimes.**
- `s_global` is the fraction of all rows that match.
- `h_k` is the rank of the k-th best matching row in the *unfiltered* similarity order.
- `s_local = min(1, k / h_k)` is the match rate among the rows a search must touch. It is the quantity
  that decides whether filtering after an index scan works.
- Queries are bucketed by ρ = tanh(½·logit(s_local) − ½·logit(s_global)): *anti* when ρ < −0.3,
  *pos* when ρ > 0.3, *indep* otherwise.

## Usage

```bash
python evaluate.py arxiv-384 --baseline exact             # pre-filter + brute force (recall 1 by construction)
python evaluate.py arxiv-384 --baseline hnsw-post --ef 64 # unfiltered HNSW, then filter (needs faiss)
python evaluate.py arxiv-384 --results my_ids.npy --k 10  # score your method: (n_queries, k) int array
```

```python
from evaluate import Dataset, recall
ds = Dataset("arxiv-384")
X, Q, gt = ds.X, ds.Q, ds.gt(10)        # base vectors (memory-mapped), query vectors, ground truth
mask = ds.mask(0)                       # rows matching query 0's filter
```

Train and test splits come from the original study, which fit estimators and cost models on `train`.
Report results on `test`.

### AMAZON-384: rebuild from the source

The Amazon Reviews 2023 metadata states no license, so this release ships no Amazon text, attributes or
embeddings. To rebuild them:

1. Download `meta_Video_Games.jsonl`, `meta_Musical_Instruments.jsonl`, `meta_Baby_Products.jsonl` and
   `meta_Industrial_and_Scientific.jsonl` from https://amazon-reviews-2023.github.io.
2. Run:

```bash
python rebuild_amazon.py RAW_DIR amazon-384 --embed --device cpu   # or mps / cuda
```

The script recomputes the attributes and the store table, checks all of them against the SHA-256 sums in
`schema.json`, and re-embeds the 993,000 texts. A CPU re-embedding of 2,000 sampled rows matched the
benchmark's vectors to within 3×10⁻⁷, so the shipped ground truth stays exact. Run
`python evaluate.py amazon-384 --check-gt` to confirm it on your own copy.

## Reference baselines (test split, k = 10)

Measured on an Apple M4, single-threaded, faiss HNSW with M=32 and efConstruction=200.
Latency covers the index search only, not filter evaluation.

| Dataset | Method | Recall@10 | Below 0.9 | Anti | Indep | Pos | Mean latency |
|---|---|---|---|---|---|---|---|
| ARXIV-384 | exact (pre-filter + scan, numpy) | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 75.9 ms |
| ARXIV-384 | HNSW post-filter, ef=64 | 0.589 | 0.487 | 0.256 | 0.751 | 0.930 | 0.28 ms |
| ARXIV-384 | HNSW post-filter, ef=400 | 0.740 | 0.294 | 0.455 | 0.924 | 0.983 | 1.17 ms |
| AMAZON-384 | HNSW post-filter, ef=64 | 0.730 | 0.344 | 0.288 | 0.651 | 0.891 | 0.19 ms |

The gap between the *anti* and *pos* columns is what this benchmark measures. A good filtered-search
system should reach high recall in every regime without paying exact-scan cost on every query.

## Provenance and license

- **ARXIV-384:** built from [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021)
  (CC0). 1,003,000 papers sampled with a fixed seed; 3,000 are held out as queries, of which those with a
  non-empty answer are kept.
- **AMAZON-384:** built from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) item metadata
  (Hou et al., 2024), four departments. Its terms apply to anything you rebuild from it.
- **Embeddings:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (Apache-2.0).
- **This release** (ids, filters, ground truth, scripts, and the arXiv vectors and attributes):
  CC0-1.0.
- **Code:** https://github.com/sanmish4ds/HyperVectorOptimizer (`benchmark/`). The full pipeline that
  produced the data is in `experiments/prep_arxiv.py`, `experiments/prep_amazon.py`,
  `experiments/build.py` and `experiments/gt.py`.

## Citation

If you use this benchmark, please cite the HyperVector-Optimizer paper (Mishra, 2026,
under submission) and the source datasets.
