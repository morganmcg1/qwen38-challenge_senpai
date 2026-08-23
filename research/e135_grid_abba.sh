#!/usr/bin/env bash
# E135: the wide QMV launch grid timed against the tight one in one session.
#
#   usage: research/e135_grid_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# W = wide   `m` threadgroup columns, of which `m - ceil(m/ipg)` return before
#            any weight load. This is the shipped default.
# T = tight  `ceil(m/ipg)` columns, which is exactly the working set.
#
# Order inside a replicate is W T T W, so both arms have mean position 2.5 and
# a monotone drift in leg index cancels exactly in the contrast. This is the
# counterbalance `program.md` requires before `MLXFAST_LOCAL_COOL_GATE=0` is a
# permitted mode; entry and exit temperature are recorded per leg and the legs
# keep `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
#
# BOTH ARMS ARE SET EXPLICITLY. Rung 4 moved the compiled default from
# `.wide` to `.tight`, so an unset selector no longer means wide.
#
# NO REBUILD BETWEEN LEGS. Both grids are compiled into one worker and the arm
# is chosen at run time by `MLX_E120_QMV_GRID`, so every leg times the same
# bytes and `worker_sha256` is asserted equal across the session.
#
# WHAT IS MEASURED. `mtp_seconds_per_token`, absolute, is the headline. The
# local ratio is a valid second instrument for THIS change only: widths 3...9
# are the routed set and the serial leg decodes at M = 1, which reaches
# `default: break` and launches no routed QMV at all. The serial leg is
# therefore untouched and cannot cancel the effect.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
tokens="${2:-512}"
label="${3:-s1}"
first="${4:-1}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_grid_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

# PHASE 1: prove the selector reaches the worker AND changes the launch.
#
# `Grid(rawValue:)` falls back to `.wide` on any unrecognized string, so a
# selector that never arrives is indistinguishable from one that arrives and is
# ignored: both run wide. The witness therefore reads `columns_by_width`, which
# the dispatch records from the argument it actually handed to Metal.
#
# Rule 101. The `wide` leg is checked twice: once against its own map, which
# must pass, and once against the `tight` map, which MUST fail. A witness that
# cannot fail is not a witness.
witness_tokens="${E135_WITNESS_TOKENS:-16}"
for arm in wide tight; do
  tag="e135${label}w${arm}"
  echo "=== witness ${tag}: grid=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  export MLX_E120_QMV_GRID="${arm}"
  export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E120_QMV_PIPELINE_LOG MLX_E120_QMV_GRID
  {
    echo "e135_grid=${arm}"
    echo "e135_leg_exit=${status}"
  } >> "${out}/meta.txt"

  python3 research/e135_columns_check.py "${out}/pipelines.json" --want "${arm}" \
    | tee "${out}/columns-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_grid_abba: ${arm} leg did not launch a ${arm} grid; not timing" >&2
    exit 3
  fi

  if [[ "${arm}" == "wide" ]]; then
    echo "--- Rule 101 control: the same check must FAIL on this leg ---"
    python3 research/e135_columns_check.py "${out}/pipelines.json" --want tight \
      | tee "${out}/columns-check-control.txt"
    if [[ "${PIPESTATUS[0]}" == "0" ]]; then
      echo "e135_grid_abba: the wide leg passed the TIGHT check, so the" \
           "witness cannot fail and proves nothing" >&2
      exit 4
    fi
    echo "control ok: the witness fails when it should"
  fi
done

# PHASE 2: the counterbalanced timed session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in wide tight tight wide; do
    position=$((position + 1))
    tag="e135${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: grid=${arm} replicate=${rep} tokens=${tokens} ==="
    export MLX_E120_QMV_GRID="${arm}"
    research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
    status=$?
    unset MLX_E120_QMV_GRID
    {
      echo "e135_grid=${arm}"
      echo "e135_replicate=${rep}"
      echo "e135_position=${position}"
      echo "e135_leg_index=$(( (rep - first) * 4 + position ))"
      echo "e135_session_commit=${session_commit}"
      echo "e135_session_worker_sha256=${session_worker}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e135_grid_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e135_grid_abba: ${failures} failed legs"
python3 research/e135_report.py --label "${label}"
exit $((failures > 0))
