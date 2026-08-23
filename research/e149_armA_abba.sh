#!/usr/bin/env bash
# E149 Arm A: does 16 rows per derived-cluster leaf beat the shipped 8 on the
# SHIPPED draft vocabulary, at the `.ship` depth price?
#
#   usage: research/e149_armA_abba.sh [REPLICATES] [LABEL] [FIRST]
#   env:   E149_PHASES=0,1,2,3   which phases to run (default all)
#          E149_NO_BUILD=1       skip the compile inside phase 0, keep the
#                                selector assertion
#
# A = shipped   rowsPerLeaf 8, the compiled default, exports no leaf variable
# B = leaf16    rowsPerLeaf 16, selected by MLX_E141_ROWS_PER_LEAF
#
# BOTH ARMS RUN AT `MLX_E134_DEPTH_PRICE_ARM=ship` (advisor F1). `pb6` is being
# retired from the base, and Rule 125 makes the depth price load-bearing for
# this arm specifically: changing the leaf width changes which candidates reach
# the rerank, which changes the drafted token, which changes acceptance, and
# the depth price is what turns an acceptance change into a schedule change.
# Measuring at `pb6` and reporting for `.ship` would be an unstated
# extrapolation.
#
# WHY THIS DIFFERS FROM research/e141_rung2_abba.sh, which must not be reused
# here. Its `leaf16` arm spec is `98304@1536:16`, which PINS MLX_E141_PROBES to
# 1536. E149 needs the probe count to DERIVE from the leaf count, because the
# saving under test is exactly "half the leaves, therefore half the coarse
# probes". The derived value is ceil(0.25 * 6146) = 1537, not 1536. This script
# exports the leaf width and nothing else. It never exports MLX_E141_PROBES and
# never touches qwen35DerivedClusterProbeFraction, which belongs to another
# student.
#
# REAL 40 C GATE, not the ungated counterbalanced mode. The effect under test
# is at the same scale as the ungated noise penalty (0.052 % gated against
# 0.091 % ungated), so the gate is taken before every timed leg through
# benchmark.sh's own `--local-cool-gate-only` entry point. That entry point
# returns before benchmark.sh acquires the resident-model lock, so it does not
# fight the lock research/e128_session.sh takes for the leg itself.
#
# `research/e128_session.sh` hardcodes `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false` in every leg's meta.txt. That file is not
# in this experiment's ownership set, so this script does not edit it; it
# appends `e149_cool_gate_*` and `e149_gate_qualified_for_timing` lines AFTER
# those two, and the report reads only the `e149_` spellings.
#
# ORDER. Inside one replicate the order per prompt is A B B A, so both arms sit
# at mean position 2.5 and monotone drift cancels to first order even with the
# gate taken. The gate removes the drift; the palindrome removes what the gate
# leaves.
#
# ARM WITNESS (Rule 114). Decoding is deterministic for a fixed prompt, golden,
# depth, build and depth price, so `round_count` and `effective_mean_draft_len`
# are an exact behavioural signature. Two independent witnesses run here:
#
#   1. Depth price. On `benchfixture` the `.ship` arm must report exactly 78
#      rounds and effective_mean_draft_len 6.358974358974359; `pb6` reports 82
#      and 5.853658536585366. A shipped-arm leg that lands on the pb6 pair
#      proves the environment did not take, and every leg of that session is
#      void.
#   2. Leaf width. Within one prompt the two shipped legs must agree exactly
#      with each other and the two leaf16 legs must agree exactly with each
#      other. The leaf16 pair MAY legitimately differ from the shipped pair,
#      because the mechanism changes which candidates reach the rerank. That
#      difference is a measurement, not a defect, and the report prices it.
#
# NO REBUILD BETWEEN LEGS. Phase 0 builds once and asserts the selector is in
# the binary that is about to run (HARNESS DEFECT 36: the benchmark wrapper
# never rebuilds the worker on its own).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-1}"
label="${2:-s1}"
first="${3:-1}"
phases="${E149_PHASES:-0,1,2,3}"

tokens=512
depth=8
# benchfixture carries the F1 depth-price tripwire. beagle_a and
# essays_montaigne are the two prompts E141 rung 2 priced leaf16 on, so the
# -264.38 us/round prior is comparable only on them.
prompts=(benchfixture beagle_a essays_montaigne)
runs_parent=".mlxfast-private/e128/runs-e149a"

# F1, calibrated across three QMV arms and two independent sessions. The two
# constants are DOUBLES, and the report and the advisor's note print them at
# different precision: `jq` renders the same value as 6.3589743589743586 and F1
# quotes 6.358974358974359. `same_double` compares the numbers, not the text,
# so a repr difference cannot void a correct leg.
ship_bench_rounds=78
ship_bench_edl=6.358974358974359
pb6_bench_edl=5.853658536585366

has_phase() { [[ ",${phases}," == *",$1,"* ]]; }

same_double() {
  [[ -n "$1" && -n "$2" ]] || return 1
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 == b + 0) }'
}

arm_leaf() { case "$1" in leaf16) echo 16 ;; *) echo "" ;; esac; }

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e149_armA: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

if has_phase 0; then
  echo "=== e149_armA phase 0: worker build and selector assertion ==="
  # `deriveCompactCoarseTable`, which research/e141_rung2_abba.sh asserts, is
  # only a comment on this base; asserting it would fail a correct build.
  # `activeClusterRowsPerLeaf` is the function that applies the override and
  # `qwen35E141RowsPerLeafOverride` is the one that reads it, so both are real
  # symbols and both must be present.
  build_args=(--require MLX_E141_ROWS_PER_LEAF
    --require MLX_E134_DEPTH_PRICE_ARM
    --require-symbol activeClusterRowsPerLeaf
    --require-symbol qwen35E141RowsPerLeafOverride)
  [[ "${E149_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
  senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
    echo "e149_armA: the worker does not carry both selectors; not timing" >&2
    exit 3; }
  CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift \
    || { echo "e149_armA: CLI build failed" >&2; exit 3; }
fi

session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
session_cli="$(
  shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
echo "e149_armA: session_commit=${session_commit}"
echo "e149_armA: session_worker_sha256=${session_worker}"
echo "e149_armA: session_cli_sha256=${session_cli}"

# Phase 1 -- Rule 137 warmup, reference-row generation and the traced arm
# witness, in one pass. These legs are TRACED, so they can never be mistaken
# for timing evidence: research/e128_session.sh writes timing_valid=false on
# them and the trace itself carries `arm=<rawValue>` from
# Qwen36MTPBlockSession.snapshotScheduleSignal, which is a direct read of the
# depth price the run actually used rather than the variable it was asked with.
# Reference rows must exist BEFORE the gated phase: generating a golden inside
# a gated leg would heat the GPU between the gate and the clock.
if has_phase 1; then
  for id in "${prompts[@]}"; do
    echo "=== e149_armA phase 1: warmup, goldens and arm witness for ${id} ==="
    env MLX_E134_DEPTH_PRICE_ARM=ship \
        E128_FORCE=1 \
        E128_TOKENS="${tokens}" \
        E128_DEPTH="${depth}" \
        E128_RUNS_DIR="runs-e149a/warm-${id}" \
      research/e128_session.sh "${id}"
    out="${runs_parent}/warm-${id}/${id}"
    traced_arm="$(sed -n 's/.*arm=\([a-z0-9]*\).*/\1/p' "${out}/trace.txt" \
      2>/dev/null | sort -u | tr '\n' ',')"
    echo "e149_armA: ${id} traced_arm=${traced_arm:-none}" \
         "rounds=$(sed -n 's/^round_count=//p' "${out}/meta.txt" 2>/dev/null)" \
         "edl=$(sed -n 's/^effective_mean_draft_len=//p' "${out}/meta.txt" \
                2>/dev/null)"
    {
      echo "e149_leg_role=warmup-discarded"
      echo "e149_traced_depth_price_arm=${traced_arm:-none}"
      echo "e149_session_commit=${session_commit}"
    } >> "${out}/meta.txt"
  done
fi

failures=0
voided=0
if has_phase 2; then
 for ((rep = first; rep < first + replicates; rep++)); do
  for id in "${prompts[@]}"; do
    position=0
    for arm in shipped leaf16 leaf16 shipped; do
      position=$((position + 1))
      slot="${label}k${rep}p${position}${arm}"
      out="${runs_parent}/${slot}/${id}"
      nleaf="$(arm_leaf "${arm}")"

      echo "=== e149_armA cool gate before ${slot} ${id} (real 40C) ==="
      if ./benchmark.sh --local-cool-gate-only; then
        gate_status=passed
      else
        gate_status=stalled_above_40C
      fi

      echo "=== e149_armA ${slot}: prompt=${id} arm=${arm}" \
           "leaf=${nleaf:-8} depth_price=ship replicate=${rep}" \
           "position=${position} gate=${gate_status} ==="
      env MLX_E134_DEPTH_PRICE_ARM=ship \
          ${nleaf:+MLX_E141_ROWS_PER_LEAF=${nleaf}} \
          E128_FORCE=1 \
          E128_NO_TRACE=1 \
          E128_TOKENS="${tokens}" \
          E128_DEPTH="${depth}" \
          E128_RUNS_DIR="runs-e149a/${slot}" \
        research/e128_session.sh "${id}"
      status=$?

      got_rounds="$(sed -n 's/^round_count=//p' "${out}/meta.txt" 2>/dev/null)"
      got_edl="$(sed -n 's/^effective_mean_draft_len=//p' "${out}/meta.txt" \
        2>/dev/null)"
      witness=ok
      if [[ "${id}" == "benchfixture" && "${arm}" == "shipped" ]]; then
        if [[ "${got_rounds}" != "${ship_bench_rounds}" ]] \
            || ! same_double "${got_edl}" "${ship_bench_edl}"; then
          witness=DEPTH_PRICE_MISMATCH
          voided=$((voided + 1))
          echo "e149_armA: ${slot} ${id} asked for .ship but ran" \
               "${got_rounds:-<none>} rounds at edl ${got_edl:-<none>};" \
               "wanted ${ship_bench_rounds} at ${ship_bench_edl}" >&2
        fi
      fi
      if same_double "${got_edl}" "${pb6_bench_edl}"; then
        witness=PB6_DETECTED
        voided=$((voided + 1))
        echo "e149_armA: ${slot} ${id} ran the pb6 depth price" >&2
      fi
      {
        echo "e149_leg_role=timed"
        echo "e149_arm_requested=${arm}"
        echo "e149_leaf_exported=${nleaf:-8}"
        echo "e149_probes_exported=derived"
        echo "e149_depth_price_arm_exported=ship"
        echo "e149_cool_gate_real_gate_invoked=1"
        echo "e149_cool_gate_status=${gate_status}"
        echo "e149_gate_qualified_for_timing=$(
          [[ "${gate_status}" == passed ]] && echo true || echo false)"
        echo "e149_witness=${witness}"
        echo "e149_replicate=${rep}"
        echo "e149_position=${position}"
        echo "e149_session_commit=${session_commit}"
        echo "e149_session_worker_sha256=${session_worker}"
        echo "e149_session_cli_sha256=${session_cli}"
      } >> "${out}/meta.txt"

      if ((status != 0)); then
        echo "e149_armA: ${slot} ${id} exited ${status}" >&2
        failures=$((failures + 1))
      fi
    done
  done
 done
 echo "e149_armA: ${failures} failed legs, ${voided} voided witnesses"
fi

if has_phase 3; then
  python3 research/e149_armA_report.py --label "${label}" \
    --runs "${runs_parent}" --out research/e149-armA.json
fi
exit $(( failures > 0 ))
