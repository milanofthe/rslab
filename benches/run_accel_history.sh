#!/usr/bin/env bash
# Release history vs Apple Accelerate: run the *same* matrices through the
# `auto` path of every release binary, so the head-to-head ratio can be tracked
# over the project's history instead of only at HEAD.
#
# Why this works: the bench harness emits the same JSONL, calls the same public
# `tuned()` entry point, and `build_family` plus `src/matgen` are byte-identical
# from v0.19.1 on, so every version sees exactly the same matrices. Accelerate is
# an external library and is measured once per round by the newest binary.
#
# Two phases:
#   1. build - one worktree + `cargo bench --no-run` per tag, binaries stashed in
#      $WORK/bin/<tag> (a shared CARGO_TARGET_DIR keeps the dependency builds).
#   2. run   - ROUNDS round-robin passes over all versions on the same sizes, so
#      thermal drift hits every version equally; the aggregate takes the minimum
#      per (version, matrix, solver).
#
# Output: benches/bench_out/accel_history.jsonl (committed - it needs the old
# binaries and is not regeneratable from HEAD alone).
#
# Usage: benches/run_accel_history.sh          # build + run + aggregate
#        ROUNDS=3 benches/run_accel_history.sh
#        SKIP_BUILD=1 benches/run_accel_history.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
WORK=${WORK:-${TMPDIR:-/tmp}/rslab-accel-history}
ROUNDS=${ROUNDS:-2}
OUT=$WORK/out
mkdir -p "$WORK/bin" "$WORK/wt" "$OUT"

# One entry per release that moved the numeric paths (patch releases that only
# touched wasm/CI are folded into their line). Append the new tag on release.
TAGS=${TAGS:-"v0.20.0 v0.21.0 v0.22.0 v0.23.0 v0.24.0 v0.25.0 v0.26.0 v0.26.4 v0.27.0"}
HEAD_TAG=${HEAD_TAG:-v$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')}

# The reduced head-to-head grid (9 log-spaced sizes 4k-110k for the generator
# families, 8 sizes 4k-200k for the circuit family) - the same points the README
# head-to-head table is computed on.
SIZES=${SIZES:-4000,6053,9160,13861,20976,31743,48035,72690,110000}
CSIZES=${CSIZES:-4000,6995,12232,21389,37402,65405,114372,200000}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  export CARGO_TARGET_DIR=$WORK/target
  for t in $TAGS; do
    [ -x "$WORK/bin/$t" ] && { echo "[build] $t cached"; continue; }
    [ -d "$WORK/wt/$t" ] || git worktree add --detach "$WORK/wt/$t" "$t" >/dev/null
    echo "[build] $t"
    ( cd "$WORK/wt/$t" && cargo bench --bench bench_suite --features matgen --no-run -j 6 ) \
      >"$WORK/bin/$t.log" 2>&1 || { echo "[build] $t FAILED, see $WORK/bin/$t.log"; continue; }
    cp "$(ls -t "$CARGO_TARGET_DIR"/release/deps/bench_suite-* | grep -v '\.d$\|\.o$' | head -1)" "$WORK/bin/$t"
  done
  unset CARGO_TARGET_DIR
  cargo bench --bench bench_suite --features matgen --no-run -j 6 2>/dev/null
  cp "$(ls -t target/release/deps/bench_suite-* | grep -v '\.d$\|\.o$' | head -1)" "$WORK/bin/$HEAD_TAG"
fi

export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export RLA_BENCH_MEM=0
run() { # tag family sizes solvers round
  local out="$OUT/$1_r$5_$2.jsonl"
  : > "$out"
  RLA_BENCH_FAMILY=$2 RLA_BENCH_SIZES=$3 RLA_BENCH_SOLVERS=$4 RLA_BENCH_OUT=$out \
    "$WORK/bin/$1" >/dev/null 2>&1 || echo "  ! $1 $2 failed"
  echo "  $1 $2 r$5: $(wc -l < "$out" | tr -d ' ') records"
}

for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r ==="
  for v in $TAGS $HEAD_TAG; do
    [ -x "$WORK/bin/$v" ] || { echo "  ! no binary for $v"; continue; }
    # Accelerate is measured by the newest binary only (external library, and the
    # shim's variant pick is itself part of the harness, not of rslab).
    if [ "$v" = "$HEAD_TAG" ]; then gen=auto,accel; circ=auto,klu,accel
    else gen=auto; circ=auto,klu; fi
    run "$v" sym "$SIZES" "$gen" "$r"
    run "$v" unsym "$SIZES" "$gen" "$r"
    # The circuit family entered the harness in v0.23.0.
    case $v in v0.20.0|v0.21.0|v0.22.0) ;; *) run "$v" circuit "$CSIZES" "$circ" "$r" ;; esac
  done
done

python3 - "$OUT" "$REPO/benches/bench_out/accel_history.jsonl" <<'PY'
"""Fold the round files into one record per (version, matrix, solver), keeping
the fastest factor time - the minimum over rounds is the drift-free estimate."""
import json, re, sys
from pathlib import Path

best = {}
for p in sorted(Path(sys.argv[1]).glob("*_r*_*.jsonl")):
    ver = re.match(r"(v[\d.]+)_r\d+_", p.name).group(1)
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["version"] = ver
        k = (ver, r["family"], r["name"], r["solver"])
        if k not in best or r["fac_ms"] < best[k]["fac_ms"]:
            best[k] = r

out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
key = lambda r: (r["version"], r["family"], r["n"], r["solver"])
with out.open("w") as f:
    for r in sorted(best.values(), key=key):
        f.write(json.dumps(r) + "\n")
print(f"wrote {out} ({len(best)} records, "
      f"{len({r['version'] for r in best.values()})} versions)")
PY
