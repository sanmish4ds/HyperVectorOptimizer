#!/usr/bin/env bash
# Reproduce every experiment and regenerate all numbers, tables and figures in paper/hvo.tex.
# Timing runs are single-threaded; keep the machine otherwise idle (no other heavy jobs).
# Prerequisites (see README): faiss-cpu, scikit-learn, sentence-transformers (MiniLM cached),
# ACORN built with hvo/kernels/acorn_fixes.patch, PostgreSQL 16 + pgvector 0.8.7 + pg/hvo_pg.
set -euo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../data/raw/arxiv ../data/raw/amazon
[ -f ../data/raw/arxiv/arxiv-abstracts.jsonl.gz ] || curl -L -o ../data/raw/arxiv/arxiv-abstracts.jsonl.gz \
  https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021/resolve/main/arxiv-abstracts.jsonl.gz
for c in Video_Games Musical_Instruments Baby_Products Industrial_and_Scientific; do
  [ -f ../data/raw/amazon/meta_$c.jsonl ] || curl -L -o ../data/raw/amazon/meta_$c.jsonl \
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_$c.jsonl
done
python prep_arxiv.py
python prep_amazon.py
DS="syn-128 syn-768 arxiv-384 amazon-384"
for ds in $DS; do python build.py $ds; done
for k in 10 1 100; do
  for ds in $DS; do
    HVO_K=$k python gt.py $ds
    HVO_K=$k python run_est.py $ds
    if [ $k = 10 ]; then HVO_K=$k python run_plans.py $ds 2 pre_simd,pre_scalar,pre_i8,post_r
    else HVO_K=$k python run_plans.py $ds 2 pre_simd,post_r; fi
    HVO_K=$k python run_plans.py $ds 2 acorn
    HVO_K=$k python analyze.py $ds
  done
done
for ds in syn-128 syn-768 arxiv-384; do                 # first-generation plans, for Table 10
  python run_plans.py $ds 2 post,fused && HVO_PLANS=v1 python analyze.py $ds
done
python transfer.py arxiv-384 amazon-384 syn-128 syn-768
for ds in $DS; do python native_plan.py $ds; done
# PostgreSQL + pgvector (+ the hvo_pg extension); Exqutor runs on its own cluster (see README)
for ds in arxiv-384 amazon-384; do
  python pg_load.py $ds
  python pg_bench.py $ds pre train && python pg_bench.py $ds post train && python pg_model.py $ds
  for mode in stock iter pre post hvo hvo_norec; do python pg_bench.py $ds $mode test; done
done
python pg_analyze.py arxiv-384 amazon-384
python make_tables.py
cd ../paper && latexmk -pdf -interaction=nonstopmode hvo.tex
