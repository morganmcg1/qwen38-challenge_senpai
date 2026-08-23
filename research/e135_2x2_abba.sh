#!/usr/bin/env bash
# E135 rungs 2 and 3 in one traced session: dispatch table x launch grid.
#
#   usage: research/e135_2x2_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# Four cells, both selected at run time so one worker times all of them:
#
#   A  MLX_E120_QMV_TABLE=shipped   grid wide    ipg 3,4,5,3,4,4,3
#   B  table unset (onepass67)      grid wide    ipg 3,4,5,6,7,4,3
#   C  MLX_E120_QMV_TABLE=shipped   grid tight
#   D  table unset (onepass67)      grid tight
#
# ORDER  B D C A A C D B, one palindrome per replicate. Code the table +1 for
# onepass67 and -1 for shipped, the grid +1 for tight and -1 for wide, and the
# interaction as their product: all three codes and the centred leg index are
# mutually orthogonal over the eight positions, so a linear drift in leg index
# cannot leak into any of the three effects.
#
# WHY BOTH TABLES. Rung 2 needs the per-column launch cost `a`. Holding the
# table fixed holds the work per column fixed, so a wide-minus-tight difference
# at one width is `a * empty_columns` with no work term to fit. The two tables
# put DIFFERENT empty-column counts on the same widths - shipped gives
# 2,3,4,4,5,6,6 for M = 3..9 against onepass67's 2,3,4,5,6,6,6 - so the shipped
# pairs are an independent replicate of `a` on the same rounds, not a repeat.
#
# WHY TRACED. Rung 2 pairs on ROUND INDEX, not width. The grid is a host-side
# launch argument that never enters the Metal source, so the two arms are
# bit-identical and round i of the wide leg is the same round as round i of the
# tight leg: same width, same draft and accept counts, same KV length, same
# position in the leg. Differencing on round index removes the KV-length growth
# that a width stratum would eat as residual.
#
# Adjacent legs are the pairs: (1,2) and (7,8) are the onepass67 wide/tight
# pairs, (3,4) and (5,6) the shipped ones. Each pair is time-adjacent and the
# two arm orders are balanced across the four pairs of a replicate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-s2}"
first="${4:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_2x2_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

cell_table() { case "$1" in A|C) echo shipped ;; *) echo "" ;; esac; }
cell_grid()  { case "$1" in A|B) echo wide ;; *) echo tight ;; esac; }

# Both selectors are set EXPLICITLY, never left to the fallback. Rung 4 moves
# the compiled default from `.wide` to `.tight`, so an unset grid no longer
# means wide and a session that relied on the fallback would silently time the
# same arm twice. The witness phase reads the launched column count back off
# the dispatch argument, so a selector that fails to arrive stops the session.
apply_cell() {
  local t
  t="$(cell_table "$1")"
  if [[ -n "${t}" ]]; then export MLX_E120_QMV_TABLE="${t}"
  else unset MLX_E120_QMV_TABLE; fi
  export MLX_E120_QMV_GRID="$(cell_grid "$1")"
}

# PHASE 1: witness every cell before timing any of it.
#
# Both selectors fall back silently to their compiled default, so a selector
# that never arrives is indistinguishable from one that arrives and is ignored,
# and the compiled grid default is now `.tight`. The
# witness reads `plan`, which names the table the worker actually built its
# dispatch from, and `columns_by_width`, which is read back off the argument
# handed to Metal. `e135_columns_check.py` derives the expected map from the
# leg's OWN plan, so it fails if either selector is wrong.
#
# Rule 101: each wide cell is checked a second time against the tight map and
# MUST fail, and each cell's `plan` is required to name the intended table.
witness_tokens="${E135_WITNESS_TOKENS:-16}"
for cell in A B C D; do
  tag="e135${label}w${cell}"
  want_grid="$(cell_grid "${cell}")"
  want_table="$(cell_table "${cell}")"
  want_table="${want_table:-onepass67}"
  echo "=== witness ${tag}: table=${want_table} grid=${want_grid} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  apply_cell "${cell}"
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG MLX_E120_QMV_TABLE MLX_E120_QMV_GRID
  {
    echo "e135_cell=${cell}"
    echo "e135_table=${want_table}"
    echo "e135_grid=${want_grid}"
    echo "e135_leg_exit=${status}"
  } >> "${out}/meta.txt"

  logged_table="$(python3 -c "
import json,sys
print(json.load(open(sys.argv[1]))['table'])" "${out}/pipelines.json")"
  if [[ "${logged_table}" != "${want_table}" ]]; then
    echo "e135_2x2_abba: cell ${cell} built table '${logged_table}', wanted" \
         "'${want_table}'; not timing" >&2
    exit 3
  fi

  python3 research/e135_columns_check.py "${out}/pipelines.json" \
    --want "${want_grid}" | tee "${out}/columns-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_2x2_abba: cell ${cell} did not launch a ${want_grid} grid;" \
         "not timing" >&2
    exit 4
  fi

  if [[ "${want_grid}" == "wide" ]]; then
    echo "--- Rule 101 control: the same check must FAIL on this leg ---"
    python3 research/e135_columns_check.py "${out}/pipelines.json" \
      --want tight | tee "${out}/columns-check-control.txt"
    if [[ "${PIPESTATUS[0]}" == "0" ]]; then
      echo "e135_2x2_abba: cell ${cell} passed the TIGHT check on a WIDE" \
           "leg, so the witness cannot fail and proves nothing" >&2
      exit 5
    fi
    echo "control ok: the witness fails when it should"
  fi
done

# PHASE 2: the traced, counterbalanced session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for cell in B D C A A C D B; do
    position=$((position + 1))
    tag="e135${label}k${rep}p${position}${cell}"
    table="$(cell_table "${cell}")"
    echo "=== ${tag}: cell=${cell} table=${table:-onepass67}" \
         "grid=$(cell_grid "${cell}") tokens=${tokens} ==="
    apply_cell "${cell}"
    research/e79_trace_leg.sh "${tag}" "${tokens}"
    status=$?
    unset MLX_E120_QMV_TABLE MLX_E120_QMV_GRID
    {
      echo "e135_cell=${cell}"
      echo "e135_table=${table:-onepass67}"
      echo "e135_grid=$(cell_grid "${cell}")"
      echo "e135_replicate=${rep}"
      echo "e135_position=${position}"
      echo "e135_leg_index=$(( (rep - first) * 8 + position ))"
      echo "e135_session_commit=${session_commit}"
      echo "e135_session_worker_sha256=${session_worker}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e135_2x2_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e135_2x2_abba: ${failures} failed legs"
python3 research/e135_per_width.py --label "${label}"
exit $((failures > 0))
