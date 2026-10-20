# HyperVector-Optimizer (HVO)

Correlation-aware selectivity estimation (HPBS) and hardware-calibrated plan routing for filtered
top-k vector search. This repository contains the implementation and every experiment in the paper
(`paper/hvo.tex`).

## Layout

```
hvo/
  data.py         synthetic vector-relational generator with controllable correlation; predicates
  sketch.py       HPBS construction (Python) + query path (Python reference and C via ctypes)
  estimators.py   baselines: 1-D histograms (PG-style), exact-s_g (Milvus-style), 1% sample, index probe
  plans.py        pre-filter scan; first-generation graph plans (restart post-filter, faiss bitmap fused)
  graphplans.py   second-generation graph plans: resumable post-filter, ACORN-1 fused
  kernels/graph.c resumable ef-window HNSW scan
  kernels/acorn_wrap.cpp, build_acorn.sh, acorn_fixes.patch   ACORN reference code behind a C ABI
                                                                (with two heap-overflow fixes)
  kernels/scan.c  id-list-guided top-k scan: NEON FP32, scalar FP32, NEON SDOT INT8
  kernels/est.c   HPBS query-time estimator and the native planner hvo_plan()
pg/
  hvo_pg.c        PostgreSQL 16 planner_hook extension for pgvector (shares hvo_plan with the benchmark)
experiments/
  prep_arxiv.py   real-data benchmark: arXiv metadata + MiniLM embeddings (1M papers + query pool)
  build.py        dataset + HNSW + sketches + queries (disjoint train/test) + exact ground truth
  gt.py           exact k-th-match local selectivity (s_loc = k / h_k)
  run_est.py      every estimator on every query (estimates, features, planning time)
  run_plans.py    every plan on every query, single thread, warm-up + median of repetitions
  analyze.py      fits calibrator, cost and recall models on train, evaluates on test -> results/<name>.json
  transfer.py     calibrator trained on one dataset, used on another
  native_plan.py  times the native planner and checks it agrees with analyze.py
  pg_*.py         PostgreSQL: load, benchmark configurations, fit/export the hook's model, summarise
  prep_amazon.py  second real dataset (Amazon Reviews 2023 metadata, query-anchored predicates)
  make_tables.py  LaTeX tables / figure data from results/*.json
```

## Reproduce

Requirements: Python 3.11+, numpy, scipy, scikit-learn, faiss-cpu (1.13), a C compiler.
The kernels use ARM NEON (`-mcpu=native`); on x86 replace them with AVX equivalents.

```bash
./run_all.sh            # everything, or step by step:
cd experiments
python build.py syn-128 && python gt.py syn-128
python run_est.py syn-128 && python run_plans.py syn-128 2
python analyze.py syn-128
python make_tables.py
```

`syn-768` is built the same way. The real-data benchmark `arxiv-384` first needs
`python prep_arxiv.py` (downloads nothing itself; expects `data/raw/arxiv/arxiv-abstracts.jsonl.gz`
from huggingface.co/datasets/gfissore/arxiv-abstracts-2021, CC0, 940 MB; embedding 1M abstracts
with all-MiniLM-L6-v2 on Apple MPS takes ~32 min, and its plan run ~4.5 h because real data has many
rare local matches). Expect ~25 min (128-d) and ~60 min (768-d) on a 10-core Apple M4
with 16 GB of RAM. All latency measurements are single-threaded (`OMP_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`); do not run other heavy jobs at the same time.
