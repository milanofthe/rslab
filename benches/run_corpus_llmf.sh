#!/usr/bin/env bash
# Append the two raw RSLAB kernels (left-looking / multifrontal) to corpus.jsonl
# for the RSLAB-internal memory / stage breakdown figures (corpus_breakdown.py),
# and refresh the a-priori memory-estimate file. fit_scaling.py ignores ll/mf
# (its ORDER is auto/faer/pardiso/superlu), so the same-run comparison is intact.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=$(ls -t target/release/deps/bench_suite-*.exe | head -1)
OUT=benches/bench_out/corpus.jsonl
export RAYON_NUM_THREADS=24 RLA_BENCH_FAMILY=corpus RLA_BENCH_SOLVERS=ll,mf RLA_BENCH_OUT=$OUT

echo "[llmf] TIME"; RLA_BENCH_MEM=0 "$BIN"
echo "[llmf] MEM";  RLA_BENCH_MEM=1 "$BIN"
echo "[estimate] a-priori estimates -> corpus_estimate.jsonl"
RLA_BENCH_ESTIMATE=1 RLA_BENCH_OUT=benches/bench_out/corpus_estimate.jsonl "$BIN"
echo "[llmf] done: $(grep -c '"ll"' $OUT) ll, $(grep -c '"mf"' $OUT) mf records"
