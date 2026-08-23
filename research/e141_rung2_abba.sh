#!/usr/bin/env bash
# E141 rung 2: what does the widened draft vocabulary cost per round, and does
# the extra acceptance repay it?
#
#   usage: research/e141_rung2_abba.sh [REPLICATES] [LABEL] [FIRST]
#
# A = shipped  compact prefix 98,304, the compiled default, exports nothing
# B = full     compact prefix 248,320, selected by MLX_E141_DRAFT_PREFIX
#
# Order inside a replicate is A B B A per prompt, so both arms have mean
# position 2.5 and monotone thermal drift cancels to first order. That
# counterbalance is what `program.md` requires before `MLXFAST_LOCAL_COOL_GATE`
# is left ungated. `research/e128_session.sh` records entry and exit GPU
# temperature per leg and writes `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false` verbatim. These legs are directional causal
# evidence inside one counterbalanced session, never a ranked score.
#
# WHAT IS MEASURED, and why it is the whole answer. Widening changes two things
# at once: it adds coarse-scoring work per draft, and it removes a class of
# guaranteed rejects. `parent_measured_seconds_per_token`, absolute, already
# contains both. Because the ranked numerator is the runner's own prebuilt
# serial baseline and no candidate edit can move it
# (`senpai/verify-ranked-score-boundary.sh`), no psi_serial share is ever
# subtracted and there is no local-ratio cancellation term.
#
# That does NOT make the local percentage the ranked percentage. CAMPAIGN
# RULE 115 applies, because half of this mechanism is a fixed absolute cost per
# round. `research/e141_rung2_report.py` splits the contrast into a
# deterministic round-count ratio, which transfers unchanged, and an absolute
# per-round cost, which it re-expresses over the 52,726 us ranked round rather
# than the ~195,000 us local one. This script only has to deliver clean matched
# absolute microseconds per round; the conversion belongs to the report.
#
# ARM WITNESS (Rule 114). Decoding is deterministic for a fixed prompt, golden,
# depth and build, so `round_count` is an exact behavioural signature of the
# arm that actually ran. The expected counts come from the INDEPENDENT untimed
# rung 3 session in research/e141-rung3.json, never from the variable this
# script exports. A leg whose round count does not match its arm is discarded.
#
# NO REBUILD BETWEEN LEGS. Phase 0 builds once and asserts the selector is in
# the binary that is about to run; HARNESS DEFECT 36 means the benchmark
# wrapper never rebuilds the worker on its own.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-2}"
label="${2:-s1}"
first="${3:-1}"
# Candidate arms, one ABBA block each against the same shipped control. A
# comma list keeps every block inside one session, which Rule 119 requires,
# and gives the byte model more than one calibration point.
candidates="${4:-full}"

tokens=512
depth=8
prompts=(beagle_a essays_montaigne)
runs_parent=".mlxfast-private/e128/runs-e141"
rung3=research/e141-rung3.json

# Same arm grammar as research/e141_session.sh: `<prefix>`, `<prefix>@<probes>`
# or `<prefix>@<probes>:<rowsPerLeaf>`.
arm_spec() {
  case "$1" in
    shipped) echo "" ;;
    full)    echo "248320" ;;
    armA)    echo "248320@3073" ;;
    armB20)  echo "248320@1229:20" ;;
    armB16)  echo "248320@1536:16" ;;
    leaf16)  echo "98304@1536:16" ;;
    *)       echo "$1" ;;
  esac
}
arm_prefix() { local s; s="$(arm_spec "$1")"; echo "${s%%@*}"; }
arm_probes() {
  local s
  s="$(arm_spec "$1")"
  s="${s#*@}"
  [[ "$(arm_spec "$1")" == *@* ]] && echo "${s%%:*}" || echo ""
}
arm_leaf() {
  local s
  s="$(arm_spec "$1")"
  [[ "${s}" == *:* ]] && echo "${s##*:}" || echo ""
}

# Expected round count per (arm, prompt), read from the untimed rung 3 run.
witness_rounds_for() {
  [[ -s "${rung3}" ]] || { echo ""; return 0; }
  python3 -c "
import json,sys
b=json.load(open('${rung3}'))
try: print(b['arms']['$1']['seeds']['$2']['round_count'])
except KeyError: print('')
"
}

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e141_rung2: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

echo "=== e141_rung2 phase 0: worker build and selector assertion ==="
build_args=(--require MLX_E141_DRAFT_PREFIX --require MLX_E141_ROWS_PER_LEAF \
  --require-symbol deriveCompactCoarseTable)
[[ "${E141_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
  echo "e141_rung2: the worker does not carry the arm selector; not timing" >&2
  exit 3; }
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e141_rung2: session_commit=${session_commit}"
echo "e141_rung2: session_worker_sha256=${session_worker}"

failures=0
discarded=0
for ((rep = first; rep < first + replicates; rep++)); do
 for cand in ${candidates//,/ }; do
  for id in "${prompts[@]}"; do
    position=0
    for arm in shipped "${cand}" "${cand}" shipped; do
      position=$((position + 1))
      slot="${label}k${rep}${cand}p${position}${arm}"
      out="${runs_parent}/${slot}/${id}"
      prefix="$(arm_prefix "${arm}")"
      nprobes="$(arm_probes "${arm}")"
      nleaf="$(arm_leaf "${arm}")"
      want_rounds="$(witness_rounds_for "${arm}" "${id}")"

      echo "=== e141_rung2 ${slot}: prompt=${id} arm=${arm}" \
           "prefix=${prefix:-unset} probes=${nprobes:-declared}" \
           "leaf=${nleaf:-8} replicate=${rep} position=${position} ==="
      env ${prefix:+MLX_E141_DRAFT_PREFIX=${prefix}} \
          ${nprobes:+MLX_E141_PROBES=${nprobes}} \
          ${nleaf:+MLX_E141_ROWS_PER_LEAF=${nleaf}} \
          E128_FORCE=1 \
          E128_NO_TRACE=1 \
          E128_TOKENS="${tokens}" \
          E128_DEPTH="${depth}" \
          E128_RUNS_DIR="runs-e141/${slot}" \
        research/e128_session.sh "${id}"
      status=$?

      got_rounds="$(sed -n 's/^round_count=//p' "${out}/meta.txt" 2>/dev/null)"
      witness="ok"
      if [[ -z "${want_rounds}" ]]; then
        witness="unpinned"
      elif [[ "${got_rounds}" != "${want_rounds}" ]]; then
        witness="MISMATCH"
        discarded=$((discarded + 1))
        echo "e141_rung2: ${slot} ${id} requested ${arm} but ran" \
             "${got_rounds:-<none>} rounds, wanted ${want_rounds}" >&2
      fi
      {
        echo "e141_arm_requested=${arm}"
        echo "e141_prefix_exported=${prefix:-unset}"
        echo "e141_probes_exported=${nprobes:-declared}"
        echo "e141_leaf_exported=${nleaf:-8}"
        echo "e141_block_candidate=${cand}"
        echo "e141_witness_rounds_want=${want_rounds:-unpinned}"
        echo "e141_witness_rounds_got=${got_rounds:-none}"
        echo "e141_witness=${witness}"
        echo "e141_replicate=${rep}"
        echo "e141_position=${position}"
        echo "e141_session_commit=${session_commit}"
        echo "e141_session_worker_sha256=${session_worker}"
      } >> "${out}/meta.txt"

      if ((status != 0)); then
        echo "e141_rung2: ${slot} ${id} exited ${status}" >&2
        failures=$((failures + 1))
      fi
    done
  done
 done
done

echo "e141_rung2: ${failures} failed legs, ${discarded} witness mismatches"
for cand in ${candidates//,/ }; do
  out=research/e141-rung2.json
  [[ "${cand}" == "full" ]] || out="research/e141-rung2-${cand}.json"
  python3 research/e141_rung2_report.py --label "${label}" --candidate "${cand}" \
    --runs "${runs_parent}" --out "${out}"
done
exit $(( failures > 0 ))
