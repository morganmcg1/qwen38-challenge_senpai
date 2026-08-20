#!/usr/bin/env bash
# E64 rung 0b instrument check: does the isolated cell reproduce the NA ladder
# step it is supposed to explain? Without this, a null on `forced` cannot be
# read as "the alloca is not the cause"; it could equally mean the cell never
# contained the phenomenon.
#
# One gated session per NA, so every session enters at the same real 40C gate.
#
#   research/e64_ladder.sh [--reps N] [--shape NAME] [--na-list "2 3 4 5 6 7"]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo}"

reps="7"
shape="mlp.gate_up_fused"
na_list="2 3 4 5 6 7"
artifacts="research/e64-artifacts"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --reps) reps="$2"; shift 2 ;;
    --shape) shape="$2"; shift 2 ;;
    --na-list) na_list="$2"; shift 2 ;;
    *) echo "e64_ladder: unknown argument $1" >&2; exit 2 ;;
  esac
done

for na in ${na_list}; do
  echo "e64_ladder na=${na} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  research/e64_rung0b.sh --na "${na}" --reps "${reps}" --shape "${shape}" \
    --skip-census \
    --out "${artifacts}/rung0b-ladder-na${na}.json" \
    --log "${artifacts}/rung0b-ladder-na${na}.log" || exit 1
done

python3 research/e64_analyze.py --ladder \
  --out "${artifacts}/rung0b-ladder-analysis.json" \
  $(for na in ${na_list}; do echo "${artifacts}/rung0b-ladder-na${na}.json"; done)
