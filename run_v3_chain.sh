#!/usr/bin/env bash
# Third round: after (i) the point-mass histogram fix, (ii) the ef-window resumable post-filter and
# (iii) ACORN distance counting. Every timing step runs alone, single-threaded; swap is logged.
set -uo pipefail
cd "$(dirname "$0")/experiments"
LOG=../data/v3_chain.log
step() { echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1 || echo "!!! FAILED: $*" | tee -a "$LOG"; }
( while true; do echo "$(date +%H:%M:%S) $(vm_stat | awk '/Swapins/{gsub(/\./,"",$2); print $2}')"; sleep 60; done ) \
  > ../data/v3_swap.log 2>&1 &
SWAP_PID=$!
brew services stop postgresql@16 >/dev/null 2>&1

DS="syn-128 syn-768 arxiv-384 amazon-384"
for ds in syn-128 syn-768 arxiv-384; do                 # amazon's sketches were rebuilt already
  rm -f ../data/$ds/sketch_*.pkl
  step python3 build.py $ds
done
for ds in syn-128 syn-768 arxiv-384; do step python3 run_est.py $ds; done
for k in 1 100; do for ds in $DS; do step env HVO_K=$k python3 run_est.py $ds; done; done
for k in 10 1 100; do for ds in $DS; do step env HVO_K=$k python3 run_plans.py $ds 2 post_r; done; done
for ds in $DS; do step python3 run_plans.py $ds 2 acorn; done

# PostgreSQL: refit the HVO model with the new sketches, rerun only the HVO configurations
brew services start postgresql@16 >/dev/null 2>&1; sleep 5
/opt/homebrew/opt/postgresql@16/bin/psql -d postgres -Atc "ALTER SYSTEM SET max_wal_size = '1GB'" >> "$LOG" 2>&1
/opt/homebrew/opt/postgresql@16/bin/psql -d postgres -Atc "SELECT pg_reload_conf()" >> "$LOG" 2>&1
/opt/homebrew/opt/postgresql@16/bin/psql -d postgres -Atc "CHECKPOINT" >> "$LOG" 2>&1
for ds in arxiv-384 amazon-384; do
  step python3 pg_model.py $ds
  step python3 pg_bench.py $ds hvo test
  step python3 pg_bench.py $ds hvo_norec test
done
kill $SWAP_PID
echo "=== $(date +%H:%M:%S) chain done" | tee -a "$LOG"
