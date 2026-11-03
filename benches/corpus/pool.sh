#!/usr/bin/env bash
# The pool entry of RSLAB's benchmark corpus, run from an rslab checkout.
#
#   benches/corpus/pool.sh corpus SOURCE [corpus.py build options]
#       One source of the corpus (benches/corpus/corpus.py) built on the corpus machine into
#       its cache: the source ships at its pinned commit, archived from a local clone
#       (CORPUS_SRC_<NAME>=DIR, e.g. CORPUS_SRC_SANE) or from a fresh clone of GitHub.
#       A path /cache/<name>/... among the options mounts that cache too (--reuse).
#       CORPUS_INPUTS=DIR ships DIR into the corpus' inputs/ (the MoM project cases).
#
#   benches/corpus/pool.sh ab REF_A REF_B [pardiso_corpus.py options and inputs]
#       The A/B of two RSLAB revisions (benches/pardiso_corpus.py --rslab), REF_A the base,
#       with MKL PARDISO beside them, on the whole corpus unless inputs under /cache/ are
#       given; e.g. `pool.sh ab main feat/x --family fem`. The machine to itself (all its
#       cores), so the timings see no other job. Results: falllow fetch RUN (ab.jsonl).
#
# Env: CORPUS_MACHINE (default milan-box-01), CORPUS_CACHE (rslab-corpus), CORPUS_CPU,
# CORPUS_MEMORY, CORPUS_DISK (the job's resources), CORPUS_DETACH=1 (submit, do not follow).
set -euo pipefail

MACHINE=${CORPUS_MACHINE:-milan-box-01}
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
follow=()
[ "${CORPUS_DETACH:-0}" = 1 ] && follow=(--detach)

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
  falllow run --machine "$MACHINE" "${caches[@]}" --name "corpus-$source-$(date +%H%M%S)" \
    --cpu "${CORPUS_CPU:-6..12}" --memory "${CORPUS_MEMORY:-32G}" --disk "${CORPUS_DISK:-80G}" \
    ${follow[@]+"${follow[@]}"} -- bash rslab/benches/corpus/pool.sh job-corpus "$source" "$@"
  ;;
ab)
  a=${1:?usage: pool.sh ab REF_A REF_B [options]}; b=${2:?usage: pool.sh ab REF_A REF_B [options]}; shift 2
  k=0
  for ref in "$a" "$b"; do
    sha=$(git -C "$REPO" rev-parse --verify "$ref^{commit}")
    mkdir -p "$stage/revs/$k"
    git -C "$REPO" archive --format=tar "$sha" -- . ':(exclude)docs' ':(exclude)benches/bench_out' \
      | tar -x -C "$stage/revs/$k"
    echo "$sha $ref" > "$stage/revs/$k/REVISION"
    echo "revision $k: $ref ($sha)"
    k=$((k + 1))
  done
  cd "$stage"
  falllow run --machine "$MACHINE" "${caches[@]}" --name "corpus-ab-$(date +%H%M%S)" \
    --cpu "${CORPUS_CPU:-12}" --memory "${CORPUS_MEMORY:-40G}" --disk "${CORPUS_DISK:-60G}" \
    ${follow[@]+"${follow[@]}"} -- bash rslab/benches/corpus/pool.sh job-ab "$@"
  ;;
*)
  echo "unknown mode $mode: corpus or ab" >&2
  exit 2
  ;;
esac
