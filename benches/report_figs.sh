#!/usr/bin/env bash
# Regenerate every technical-report figure as a PDF with black axes into
# docs/report/figures/. Paper mode is selected by RSLAB_REPORT=1, which makes
# bench_style redirect each save to docs/report/figures/<stem>.pdf (white page,
# black axes, serif/CM font, in-figure titles stripped). Same plot code as the
# README PNGs - one figure definition, two skins.
set -euo pipefail
cd "$(dirname "$0")/.."
export RSLAB_REPORT=1
OUT=benches/bench_out

echo "[report] scaling / memory / residual (vs faer/PARDISO/SuperLU)"
python benches/fit_scaling.py $OUT/corpus.jsonl

echo "[report] auto-tuner end-to-end"
python benches/autotune_plot.py $OUT/autotune.jsonl

echo "[report] thread scaling per solver"
python benches/agg_thread_scaling_solvers.py $OUT/corpus_threads.jsonl

echo "[report] block GMRES BCGS2 scaling (needs $OUT/block_gmres_bcgs2.jsonl; MGS ref is committed)"
python benches/block_gmres_plot.py

echo "[report] preconditioner + GMRES trade-off (needs $OUT/precond_gmres.jsonl)"
python benches/precond_gmres_plot.py

echo "[report] memory + runtime breakdown (RSLAB LL/MF)"
python benches/corpus_breakdown.py $OUT/corpus.jsonl $OUT/corpus_estimate.jsonl

echo "[report] figures written to docs/report/figures/"
ls -1 docs/report/figures/*.pdf 2>/dev/null || true
