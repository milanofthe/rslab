#!/usr/bin/env bash
# The pool entry of RSLAB's benchmark corpus, run from an rslab checkout.
#
#   benches/corpus/pool.sh corpus SOURCE [corpus.py build options]
#       One source of the corpus (benches/corpus/corpus.py) built into the cache of each
#       machine that holds a family it feeds (corpus.py PLACEMENT): the source ships at its
#       pinned commit, archived from a local clone (CORPUS_SRC_<NAME>=DIR, e.g.
#       CORPUS_SRC_SANE) or from a fresh clone of GitHub. A path /cache/<name>/... among the
#       options mounts that cache too (--reuse). CORPUS_INPUTS=DIR ships DIR into the
#       corpus' inputs/ (the MoM project cases).
#
#   benches/corpus/pool.sh ab REF_A REF_B [REF ...] [pardiso_corpus.py options]
#       RSLAB revisions against REF_A (benches/pardiso_corpus.py --rslab) with MKL PARDISO
#       beside them: one job per family (--family, default all) on the machine that holds
#       it, as many cores as the measurement has threads and the memory of PLACEMENT; e.g.
#       `pool.sh ab main main feat/x` (main twice: the noise next to the change). Results:
#       falllow fetch RUN (ab.jsonl).
#
# Env: CORPUS_CACHE (rslab-corpus), CORPUS_CPU (6, the physical cores of the pool machines),
# CORPUS_DISK. Every job is submitted detached; follow it with falllow logs -f RUN.
set -euo pipefail

CACHE=${CORPUS_CACHE:-rslab-corpus}
HERE=$(cd "$(dirname "$0")" && pwd)
mode=${1:?usage: pool.sh corpus SOURCE [options] | ab REF_A REF_B [options]}
shift

# --- inside the job ------------------------------------------------------------------------

if [ "$mode" = job-corpus ]; then
  source=$1; shift
  [ -d inputs ] && mkdir -p "/cache/$CACHE/inputs" && cp -rn inputs/. "/cache/$CACHE/inputs/"
  uv venv -q /scratch/venv-corpus
  # rslab for the routing of SANE's systems (rslab.analyze(path='auto').route)
  uv pip install -q --python /scratch/venv-corpus/bin/python numpy scipy rslab==1.12.0
  exec /scratch/venv-corpus/bin/python -u rslab/benches/corpus/corpus.py build "$source" --src src \
    --out "/cache/$CACHE" --work /scratch/corpus "$@"
fi

if [ "$mode" = job-ab ]; then
  uv venv -q /scratch/venv-ab
  uv pip install -q --python /scratch/venv-ab/bin/python numpy scipy psutil mkl
  revs=()
  for d in revs/*/; do revs+=("$(cut -d' ' -f2 "$d/REVISION")=$PWD/$d"); done
  inputs=()
  printf '%s\n' "$@" | grep -q '^/cache/' || inputs=("/cache/$CACHE")
  echo "== $(lscpu | grep 'Model name' | tr -s ' '), $(nproc) threads, ${FALLLOW_MEMORY_MIB:-?} MiB"
  exec /scratch/venv-ab/bin/python -u rslab/benches/pardiso_corpus.py ${inputs[@]+"${inputs[@]}"} --rslab "${revs[@]}" \
    --build-dir /scratch/revisions --out /artifacts/ab.jsonl "$@"
fi

# --- on this machine: stage and submit --------------------------------------------------------

REPO=$(git -C "$HERE" rev-parse --show-toplevel)
stage=$(mktemp -d "${TMPDIR:-/tmp}/rslab-corpus.XXXXXX")
trap 'rm -rf "$stage"' EXIT
# the tool as it is in this checkout, uncommitted changes included
tool=$(git -C "$REPO" stash create)
tool=${tool:-$(git -C "$REPO" rev-parse HEAD)}
mkdir -p "$stage/rslab"
git -C "$REPO" archive --format=tar "$tool" -- benches ':(exclude)benches/bench_out' | tar -x -C "$stage/rslab"

caches=(--cache "$CACHE")
for a in "$@"; do
  if [[ $a =~ ^/cache/([^/]+) ]] && [ "${BASH_REMATCH[1]}" != "$CACHE" ]; then
    caches+=(--cache "${BASH_REMATCH[1]}")
  fi
done

case $mode in
corpus)
  source=${1:?usage: pool.sh corpus SOURCE [options]}; shift
  while read -r name repo commit specs; do
    var=CORPUS_SRC_$(echo "$name" | tr '[:lower:]' '[:upper:]')
    clone=${!var:-}
    if [ -z "$clone" ]; then
      clone=$stage/clone-$name
      git clone -q --filter=blob:none --no-checkout "https://github.com/$repo" "$clone"
    fi
    mkdir -p "$stage/src/$name"
    # shellcheck disable=SC2086 # the pathspecs are words
    git -C "$clone" archive --format=tar "$commit" -- $specs | tar -x -C "$stage/src/$name"
    echo "shipping $name $repo@${commit:0:10} ($(du -sh "$stage/src/$name" | cut -f1))"
    rm -rf "$stage/clone-$name"
  done < <(python3 "$HERE/corpus.py" pins "$source")
  [ -n "${CORPUS_INPUTS:-}" ] && cp -R "$CORPUS_INPUTS" "$stage/inputs"
  cd "$stage"
  while read -r family machine memory; do
    falllow run --machine "$machine" "${caches[@]}" --name "corpus-$source-$family-$(date +%H%M%S)" \
      --cpu "${CORPUS_CPU:-6}" --memory "$memory" --disk "${CORPUS_DISK:-40G}" --detach \
      -- bash rslab/benches/corpus/pool.sh job-corpus "$source" --family "$family" "$@" < /dev/null
  done < <(python3 "$HERE/corpus.py" placement "$source")
  ;;
ab)
  refs=()
  while [ $# -gt 0 ] && [[ $1 != -* ]]; do refs+=("$1"); shift; done
  [ ${#refs[@]} -ge 2 ] || { echo "usage: pool.sh ab REF_A REF_B [REF ...] [options]" >&2; exit 2; }
  # the families asked for (--family), the rest of the options passed on
  families=() rest=()
  while [ $# -gt 0 ]; do
    if [ "$1" = --family ]; then
      shift
      while [ $# -gt 0 ] && [[ $1 != -* ]]; do families+=("$1"); shift; done
    else
      rest+=("$1"); shift
    fi
  done
  k=0
  for ref in "${refs[@]}"; do
    sha=$(git -C "$REPO" rev-parse --verify "$ref^{commit}")
    mkdir -p "$stage/revs/$k"
    git -C "$REPO" archive --format=tar "$sha" -- . ':(exclude)docs' ':(exclude)benches/bench_out' \
      | tar -x -C "$stage/revs/$k"
    echo "$sha $ref" > "$stage/revs/$k/REVISION"
    echo "revision $k: $ref ($sha)"
    k=$((k + 1))
  done
  cd "$stage"
  while read -r family machine memory; do
    if [ ${#families[@]} -gt 0 ] && [[ " ${families[*]} " != *" $family "* ]]; then continue; fi
    falllow run --machine "$machine" "${caches[@]}" --name "corpus-ab-$family-$(date +%H%M%S)" \
      --cpu "${CORPUS_CPU:-6}" --memory "$memory" --disk "${CORPUS_DISK:-40G}" --detach \
      -- bash rslab/benches/corpus/pool.sh job-ab --family "$family" ${rest[@]+"${rest[@]}"} < /dev/null
  done < <(python3 "$HERE/corpus.py" placement)
  ;;
*)
  echo "unknown mode $mode: corpus or ab" >&2
  exit 2
  ;;
esac
