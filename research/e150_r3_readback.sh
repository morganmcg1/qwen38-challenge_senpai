#!/usr/bin/env bash
# E150 R3: what does the host round trip cost that a SEQUENTIAL stopping rule
# has to pay before it earns anything?
#
#   usage: research/e150_r3_readback.sh [REPLICATES] [LABEL] [FIRST] [CANDS]
#
# WHY THIS IS THE DECIDING NUMBER. R2's demand curve says a rule that stops on
# the head's own confidence is worth +1.5042 pp over one-shot at the PAD
# operating point, which buys a break-even host readback of 775.0 us/round.
# The R3 stop rule is 100 us/round. So the whole L5/L6 axis turns on one
# scalar, and the scalar is measurable without building the rule: any
# sequential design must make the stop decision visible on the HOST, because
# the verify width is a launched grid shape.
#
# A = off        the compiled default, exports nothing
# B = firstOnly  one sync per round, after draft 1  (the cheap L5 design)
# B = perStep    one sync per draft step            (the full L6 design)
#
# WHAT IS MEASURED. The arm reads one Int32 the drafting loop has already
# computed and discards it into a sink. It adds no arithmetic and launches no
# kernel, so the contrast is the round trip alone. That is a FLOOR under every
# sequential design, not one design's cost, which is the only form of this
# measurement that can retire the axis rather than one candidate.
#
# RULE 79 DOES NOT BIND. Rule 79 forbids a local timing leg that publishes a
# depth-price or schedule-policy contrast. It permits cost measurement with the
# policy held fixed. Nothing downstream reads the sink, so both arms run an
# identical schedule by construction, and phase 1 proves it from the run's own
# trace rather than asserting it: the d= sequence must be byte-identical across
# arms while rb= differs. If phase 1 fails, phase 2 must not be published.
#
# ARM WITNESS (Rule 114). rb= is the witness, not the variable this script
# exports. An arm leg that reports rb=0 ran the control. Note that arm= in the
# trace is the LEGACY depth-price arm and reads `ship` on every leg here.
#
# THERMAL. Order inside a replicate is A B B A per prompt, so both arms have
# mean position 2.5 and monotone drift cancels to first order. e128_session.sh
# records entry and exit GPU temperature and writes
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false
# verbatim. Directional causal evidence inside one counterbalanced session,
# never a ranked score.
#
# NO REBUILD BETWEEN LEGS. Phase 0 builds once and asserts the selector is in
# the binary about to run; the benchmark wrapper never rebuilds the worker.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-1}"
label="${2:-s1}"
first="${3:-1}"
candidates="${4:-firstOnly,perStep}"

tokens=512
depth=8
prompts=(beagle_a essays_montaigne)
runs_parent=".mlxfast-private/e128/runs-e150r3"
witness_json=research/e150-artifacts/r3_witness.json

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e150_r3: scored surface is dirty; refusing to time over uncommitted" \
       "work" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

echo "=== e150_r3 phase 0: worker build and selector assertion ==="
build_args=(--require MLX_E150_READBACK_ARM)
[[ "${E150_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
  echo "e150_r3: the worker does not carry the arm selector; not timing" >&2
  exit 3; }
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e150_r3: session_commit=${session_commit}"
echo "e150_r3: session_worker_sha256=${session_worker}"

# ---------------------------------------------------------------- phase 1
# Traced witness legs. One prompt, one leg per arm. Establishes that the arm
# fires, that it fires the right number of times, and that the schedule is
# untouched. Traced legs report timing_valid=false and are never timed.
if [[ "${E150_SKIP_WITNESS:-0}" != "1" ]]; then
  echo "=== e150_r3 phase 1: traced arm witness ==="
  for arm in off firstOnly perStep; do
    slot="${label}w${arm}"
    echo "=== e150_r3 witness ${slot}: arm=${arm} ==="
    env ${arm:+MLX_E150_READBACK_ARM=${arm}} \
        MLXFAST_QWEN_MTP_TRACE=1 \
        E128_FORCE=1 \
        E128_TOKENS="${tokens}" \
        E128_DEPTH="${depth}" \
        E128_RUNS_DIR="runs-e150r3/${slot}" \
      research/e128_session.sh "${prompts[0]}"
    echo "e150_r3_arm_requested=${arm}" \
      >> "${runs_parent}/${slot}/${prompts[0]}/meta.txt"
  done
  python3 research/e150_r3_witness.py \
    --runs "${runs_parent}" --label "${label}" --prompt "${prompts[0]}" \
    --out "${witness_json}" || {
      echo "e150_r3: WITNESS FAILED. Rule 79 clearance is not established;" \
           "refusing to run timed legs." >&2
      exit 4; }
fi

# ---------------------------------------------------------------- phase 2
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
 for cand in ${candidates//,/ }; do
  for id in "${prompts[@]}"; do
    position=0
    for arm in off "${cand}" "${cand}" off; do
      position=$((position + 1))
      slot="${label}k${rep}${cand}p${position}${arm}"
      out="${runs_parent}/${slot}/${id}"

      echo "=== e150_r3 ${slot}: prompt=${id} arm=${arm}" \
           "replicate=${rep} position=${position} ==="
      env MLX_E150_READBACK_ARM="${arm}" \
          E128_FORCE=1 \
          E128_NO_TRACE=1 \
          E128_TOKENS="${tokens}" \
          E128_DEPTH="${depth}" \
          E128_RUNS_DIR="runs-e150r3/${slot}" \
        research/e128_session.sh "${id}"
      status=$?

      {
        echo "e150_r3_arm_requested=${arm}"
        echo "e150_r3_block_candidate=${cand}"
        echo "e150_r3_replicate=${rep}"
        echo "e150_r3_position=${position}"
        echo "e150_r3_session_commit=${session_commit}"
        echo "e150_r3_session_worker_sha256=${session_worker}"
      } >> "${out}/meta.txt"

      if ((status != 0)); then
        echo "e150_r3: ${slot} ${id} exited ${status}" >&2
        failures=$((failures + 1))
      fi
    done
  done
 done
done

echo "e150_r3: ${failures} failed legs"
python3 research/e150_r3_report.py --runs "${runs_parent}" --label "${label}" \
  --candidates "${candidates}" --witness "${witness_json}" \
  --replicates "${replicates}" --first "${first}" \
  --prompts "$(IFS=,; echo "${prompts[*]}")" \
  --out research/e150-artifacts/r3_readback.json
exit $(( failures > 0 ))
