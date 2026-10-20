#!/usr/bin/env bash
# Timing chain for the second round of experiments. Every step is single-threaded and runs alone:
# do not start other heavy jobs on the machine while this runs. Swap activity is logged.
set -uo pipefail
cd "$(dirname "$0")/experiments"
LOG=../data/v2_chain.log
step() { echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1 || echo "!!! FAILED: $*" | tee -a "$LOG"; }

( while true; do echo "$(date +%H:%M:%S) $(vm_stat | awk '/Swapins/{gsub(/\./,"",$2); print $2}')"; sleep 60; done ) \
  > ../data/v2_swap.log 2>&1 &
SWAP_PID=$!

# ---- phase A: faiss-based benchmarks (PostgreSQL stopped to free its shared memory)
brew services stop postgresql@16 >/dev/null 2>&1
for ds in syn-128 syn-768 arxiv-384; do                # new graph plans at k = 10 (pre-filter runs reused)
  case " ${SKIP_POSTR:-} " in *" $ds "*) ;; *) step python3 run_plans.py $ds 2 post_r ;; esac
  step python3 run_plans.py $ds 2 acorn
done
step python3 build.py amazon-384                        # second real dataset, full set of plans
step python3 gt.py amazon-384
step python3 run_est.py amazon-384
step python3 run_plans.py amazon-384 2 pre_simd,pre_scalar,pre_i8,post_r
step python3 run_plans.py amazon-384 2 acorn
for k in 1 100; do                                      # k sweep
  for ds in syn-128 syn-768 arxiv-384 amazon-384; do
    step env HVO_K=$k python3 gt.py $ds
    step env HVO_K=$k python3 run_est.py $ds
    step env HVO_K=$k python3 run_plans.py $ds 2 pre_simd,post_r
    step env HVO_K=$k python3 run_plans.py $ds 2 acorn
  done
done

# ---- phase B: PostgreSQL + pgvector
brew services start postgresql@16 >/dev/null 2>&1; sleep 5
step python3 pg_load.py amazon-384
for ds in arxiv-384 amazon-384; do
  step python3 pg_bench.py $ds pre train
  step python3 pg_bench.py $ds post train
  step python3 pg_model.py $ds
  for mode in stock iter pre post hvo hvo_norec; do
    step python3 pg_bench.py $ds $mode test
  done
done

# ---- phase C: Exqutor (patched PostgreSQL 16 + pgvector 0.7.1) on its own cluster, same settings
EXQ=$HOME/hvo-ext/exq-pg
brew services stop postgresql@16 >/dev/null 2>&1
if [ ! -d "$EXQ/data" ]; then
  "$EXQ/bin/initdb" -D "$EXQ/data" >> "$LOG" 2>&1
  cat >> "$EXQ/data/postgresql.conf" <<CONF
port = 5433
shared_buffers = 4GB
effective_cache_size = 8GB
work_mem = 64MB
maintenance_work_mem = 6GB
max_parallel_maintenance_workers = 8
max_wal_size = 8GB
max_parallel_workers_per_gather = 0
jit = off
CONF
fi
"$EXQ/bin/pg_ctl" -D "$EXQ/data" -l "$EXQ/log.log" start >> "$LOG" 2>&1; sleep 5
step env HVO_PG_BIN="$EXQ/bin" python3 pg_load.py arxiv-384 5433
step python3 pg_bench.py arxiv-384 exqutor test 5433
"$EXQ/bin/pg_ctl" -D "$EXQ/data" stop >> "$LOG" 2>&1
brew services start postgresql@16 >/dev/null 2>&1

kill $SWAP_PID
echo "=== $(date +%H:%M:%S) chain done" | tee -a "$LOG"
