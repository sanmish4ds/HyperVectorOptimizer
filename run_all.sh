#!/usr/bin/env bash
# Reproduce every experiment and regenerate all numbers, tables and figures in paper/hvo.tex.
# Timing runs are single-threaded; keep the machine otherwise idle.
set -euo pipefail
cd "$(dirname "$0")/experiments"
# real data: download the CC0 arXiv dump (940 MB) once, then embed a seeded 1M sample with MiniLM
mkdir -p ../data/raw/arxiv
[ -f ../data/raw/arxiv/arxiv-abstracts.jsonl.gz ] || curl -L -o ../data/raw/arxiv/arxiv-abstracts.jsonl.gz \
  https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021/resolve/main/arxiv-abstracts.jsonl.gz
python prep_arxiv.py
for ds in syn-128 syn-768 arxiv-384; do
  python build.py "$ds"
  python gt.py "$ds"
  python run_est.py "$ds"
  python run_plans.py "$ds" 2
  python analyze.py "$ds"
done
python make_tables.py
cd ../paper && latexmk -pdf -interaction=nonstopmode hvo.tex
