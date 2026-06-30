#!/usr/bin/env bash
# Same-run corpus comparison: RSLAB auto-tuned default vs faer / PARDISO / SuperLU.
# Truncates corpus.jsonl, runs the Rust time+mem passes (auto,faer,pardiso) at the
# historical 24 threads, then appends SuperLU (SciPy) over the cached .mtx files.
# All four solvers measured in one sitting -> drift-free cross-solver comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=$(ls -t target/release/deps/bench_suite-*.exe | head -1)
OUT=benches/bench_out/corpus.jsonl
export RAYON_NUM_THREADS=24
export RLA_BENCH_FAMILY=corpus
export RLA_BENCH_SOLVERS=auto,faer,pardiso
export RLA_BENCH_OUT=$OUT

echo "[corpus] binary: $BIN"
: > "$OUT"   # truncate -> fresh same-run dataset

echo "[corpus] === TIME pass ==="
RLA_BENCH_MEM=0 "$BIN"

echo "[corpus] === MEM pass ==="
RLA_BENCH_MEM=1 "$BIN"

echo "[corpus] === SuperLU pass ==="
python benches/superlu_corpus.py

echo "[corpus] records: $(wc -l < "$OUT")"
