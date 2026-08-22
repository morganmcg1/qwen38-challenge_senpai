#!/usr/bin/env bash
# E136: the QMV launch-column ladder, timed in one counterbalanced session.
#
#   usage: research/e136_ladder_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# WHAT THIS ANSWERS. E135 deleted no-op threadgroup columns and moved absolute
# candidate MTP seconds per token by 1.806 % +- 0.066 %. It cannot say what a
# column costs, because `wide -> tight` changed the column count by a different
# factor at every width at once. Three laws fit that single point:
#
#   flat     the cost is per drafting round and does not see the column count
#   linear   the cost is proportional to the launched column count
#   log      the cost is proportional to ln(columns)
#
# THE LADDER SEPARATES THEM. `tightN` launches N times the working column count
# at every width, so the arithmetic, the buffers, the pipelines and the emitted
# bytes are all held fixed and only the launched threadgroup count moves. On
# this fixture the launched columns per 512-token leg are
#
#   tight 137   tight2 274   tight4 548   tight8 1096   wide 572
#
# so the three laws predict three different signatures over the rungs:
#
#   flat     no change at any rung
#   linear   increments of +137, +411, +959 columns, that is 1x, 3x, 7x
#   log      equal increments per doubling
#
# WHY `wide` IS IN THE SESSION. It anchors the ladder to E135 finding 182 on
# this worker, on this day, in this session, instead of across two commits.
#
# EXACTNESS. Every padded column starts at `first_m = column * ipg >= M` and
# takes the early return already in `qwen_e120_qmv_m`, so no rung can change a
# token. `paddedColumnsAllTakeTheKernelEarlyReturn` mechanizes that argument and
# `ladderRungsAreBitIdenticalToTight` checks it on real buffers with a positive
# control.
#
# COUNTERBALANCE. One replicate is the arm list followed by its reverse, so
# every arm has mean position 5.5 and a monotone drift in leg index cancels to
# first order in every contrast. Entry and exit temperature are recorded per leg
# and every leg keeps `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false`, because an ungated reading is directional
# evidence inside its own counterbalanced session and nothing more.
#
# NO REBUILD BETWEEN LEGS. Every rung is compiled into one worker and selected
# at run time by `MLX_E120_QMV_GRID`, so every leg times the same bytes and
# `worker_sha256` is asserted equal across the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-L1}"
first="${4:-1}"

arms=(wide tight tight2 tight4 tight8)
order=(wide tight tight2 tight4 tight8 tight8 tight4 tight2 tight wide)
legs_per_rep="${#order[@]}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e136_ladder_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e136_ladder_abba: commit=${session_commit} worker=${session_worker}"
echo "e136_ladder_abba: ${replicates} replicates x ${legs_per_rep} legs" \
     "+ 4 traced diagnostic legs, ${tokens} tokens"

# PHASE 1: prove every selector reaches the worker AND changes the launch.
#
# `Grid(rawValue:)` falls back to the compiled default on any unrecognized
# string, so a selector that never arrives is indistinguishable from one that
# arrives and is ignored: both run `tight`. The witness therefore reads
# `columns_by_width`, which the dispatch records from the argument it actually
# handed to Metal.
witness_tokens="${E136_WITNESS_TOKENS:-16}"
for arm in "${arms[@]}"; do
  tag="e136${label}w${arm}"
  echo "=== witness ${tag}: grid=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  export MLX_E120_QMV_GRID="${arm}"
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG MLX_E120_QMV_GRID
  {
    echo "e136_grid=${arm}"
    echo "e136_leg_exit=${status}"
  } >> "${out}/meta.txt"

  python3 research/e135_columns_check.py "${out}/pipelines.json" --want "${arm}" \
    | tee "${out}/columns-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e136_ladder_abba: ${arm} leg did not launch a ${arm} grid; not timing" >&2
    exit 3
  fi

  # Rule 101. A witness that cannot fail is not a witness. Each padded rung is
  # also checked against the unpadded map, which is exactly what the leg would
  # have recorded had the selector been dropped, and that check MUST fail.
  if [[ "${arm}" != "tight" && "${arm}" != "wide" ]]; then
    echo "--- Rule 101 control: the tight check must FAIL on this leg ---"
    python3 research/e135_columns_check.py "${out}/pipelines.json" --want tight \
      | tee "${out}/columns-check-control.txt"
    if [[ "${PIPESTATUS[0]}" == "0" ]]; then
      echo "e136_ladder_abba: the ${arm} leg passed the TIGHT check, so the" \
           "witness cannot fail and proves nothing" >&2
      exit 4
    fi
    echo "control ok: the witness fails when it should"
  fi
done

run_leg() {
  local tag="$1" arm="$2" toks="$3" traceflag="$4"
  shift 4
  export MLX_E120_QMV_GRID="${arm}"
  # `e79_trace_leg.sh` traces by default and rejects an unknown argument, so a
  # traced leg passes no flag at all.
  if [[ "${traceflag}" == "--no-trace" ]]; then
    research/e79_trace_leg.sh "${tag}" "${toks}" --no-trace
  else
    research/e79_trace_leg.sh "${tag}" "${toks}"
  fi
  local status=$?
  unset MLX_E120_QMV_GRID
  {
    echo "e136_grid=${arm}"
    echo "e136_session_commit=${session_commit}"
    echo "e136_session_worker_sha256=${session_worker}"
    printf '%s\n' "$@"
  } >> "research/out/${tag}/meta.txt"
  return "${status}"
}

# PHASE 2: the counterbalanced timed session, untraced.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in "${order[@]}"; do
    position=$((position + 1))
    tag="e136${label}k${rep}p$(printf '%02d' "${position}")${arm}"
    echo "=== ${tag}: grid=${arm} replicate=${rep} tokens=${tokens} ==="
    run_leg "${tag}" "${arm}" "${tokens}" --no-trace \
      "e136_replicate=${rep}" \
      "e136_position=${position}" \
      "e136_leg_index=$(( (rep - first) * legs_per_rep + position ))" \
      "e136_phase=ladder"
    status=$?
    if ((status != 0)); then
      echo "e136_ladder_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

# PHASE 3: attribution. Four traced legs in one palindrome at the extremes of
# the ladder say WHICH per-round phase pays for the extra launches. Tracing adds
# its own per-round cost, so these legs are read only against each other and
# never mixed into the phase-2 headline.
tposition=0
for arm in tight tight8 tight8 tight; do
  tposition=$((tposition + 1))
  tag="e136${label}t$(printf '%02d' "${tposition}")${arm}"
  echo "=== ${tag}: TRACED grid=${arm} tokens=${tokens} ==="
  run_leg "${tag}" "${arm}" "${tokens}" --trace \
    "e136_position=${tposition}" \
    "e136_leg_index=${tposition}" \
    "e136_phase=trace"
  status=$?
  if ((status != 0)); then
    echo "e136_ladder_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e136_ladder_abba: ${failures} failed legs"
python3 research/e136_ladder_report.py --label "${label}"
exit $((failures > 0))
