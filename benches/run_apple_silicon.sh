#!/usr/bin/env bash
# Apple-Silicon bench: RSLAB shipped default (auto) vs Apple Accelerate Sparse
# Solvers (macOS 15.5+), same-run over the full corpus - the two generator
# families (sym 8-class complex-symmetric EM/FEM, unsym convection-diffusion +
# MoM) on the historical 1k-110k log grid, plus the SuiteSparse corpus. Time and
# memory passes per dataset; sequential, one solver at a time, so the ratios are
# drift-free. Outputs benches/bench_out/apple_{sym,unsym,corpus}.jsonl.
#
# RSLAB runs its shipped heuristic pick (calibrated via `cargo xtask calibrate`);
# Accelerate runs its vendor defaults (no thread knob - internal parallelism).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo bench --bench bench_suite --features matgen,matgen-download --no-run 2>/dev/null
BIN=$(ls -t target/release/deps/bench_suite-* | grep -v '\.d$' | head -1)
echo "[apple] binary: $BIN"

SIZES=$(python3 -c "
import math
lo, hi, k = 1000, 110000, 63
print(','.join(str(round(lo*(hi/lo)**(i/(k-1)))) for i in range(k)))")

export RAYON_NUM_THREADS=8
export RLA_BENCH_SOLVERS=auto,accel

for fam in sym unsym; do
  OUT=benches/bench_out/apple_${fam}.jsonl
  : > "$OUT"
  echo "[apple] === $fam TIME ==="
  RLA_BENCH_FAMILY=$fam RLA_BENCH_SIZES=$SIZES RLA_BENCH_MEM=0 RLA_BENCH_OUT=$OUT "$BIN"
  echo "[apple] === $fam MEM ==="
  RLA_BENCH_FAMILY=$fam RLA_BENCH_SIZES=$SIZES RLA_BENCH_MEM=1 RLA_BENCH_OUT=$OUT "$BIN"
done

OUT=benches/bench_out/apple_corpus.jsonl
: > "$OUT"
echo "[apple] === corpus TIME ==="
RLA_BENCH_FAMILY=corpus RLA_BENCH_MEM=0 RLA_BENCH_OUT=$OUT "$BIN"
echo "[apple] === corpus MEM ==="
RLA_BENCH_FAMILY=corpus RLA_BENCH_MEM=1 RLA_BENCH_OUT=$OUT "$BIN"

echo "[apple] records: sym=$(wc -l < benches/bench_out/apple_sym.jsonl)" \
     "unsym=$(wc -l < benches/bench_out/apple_unsym.jsonl)" \
     "corpus=$(wc -l < benches/bench_out/apple_corpus.jsonl)"
