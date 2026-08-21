#!/usr/bin/env bash
# E109 rung 0 -- the campaign A/B protocol that resolves a 0.20 % arm.
#
#   usage: research/e109_ab_session.sh TAG CONTROL_SPEC ARM_SPEC [ARM_SPEC ...]
#
#   TAG          output directory name under research/out/
#   *_SPEC       label=ENV=VALUE[,ENV=VALUE ...]   (`label=` alone runs the
#                                                    unmodified build)
#   The FIRST spec is the control. Every other arm is reported as a paired
#   contrast against it.
#
# WHY THIS DESIGN. E105 measured the campaign's end-to-end resolution with the
# instrument every experiment was using: one ABBA pair of
# `--local-iterate` legs. Median identical-leg repeatability came out at 445 us
# MTP against a 135,309 us decode-only round -- 0.33 % -- so a single pair
# cannot see a 0.20 % arm, and the E105 dose ladder proved it in the worst
# possible way: its 0.40 % MTP dose point came back with the WRONG SIGN.
# Campaign rule 35 followed. This protocol is the fix, and it changes four
# things at once.
#
#   1. THE ENDPOINT IS A ROUND, NOT A LEG SCALAR. `mtp-timed` already reports
#      `block_request_seconds`, one parent-measured wall-clock time per decode
#      round. A 512-token leg carries about 79 of them. The leg statistic is a
#      10 %-trimmed mean over those rounds after dropping round 0 (a measured
#      one-time post-prefill warmup that the box wrapper's own stall guardrail
#      also excludes). Trimming was fixed before the data and removes OS
#      stalls, which add variance and carry no signal.
#
#   2. THE DENOMINATOR IS MEASURED, NOT MODELLED. A round time IS the
#      decode-only round, so campaign rule 34 holds by construction: no
#      `wall / rounds` denominator, no `spt(n) = P/n + D` seed model, and no
#      4.1x understatement.
#
#   3. ARMS ARE PAIRED INSIDE A BLOCK, NOT ACROSS A SESSION. Every arm runs
#      once per block, so the contrast is a within-block difference and any
#      thermal or clock state shared by the block cancels. E105's two worst
#      pairs were its palindrome ENDPOINTS -- the two legs separated by the
#      whole session. Here no paired legs are ever more than one block apart.
#
#   4. POSITION IS COUNTERBALANCED, NOT RANDOMISED. Block orders are a fixed
#      rotation-plus-mirror schedule: over 2n blocks each arm occupies each
#      within-block position exactly twice, and consecutive blocks are
#      time-reversed so a linear drift inside the pair cancels. Nothing is
#      randomised, so the schedule is reproducible from the tag alone.
#
# BLOCK 0 IS A THERMAL CONDITIONING BLOCK and is EXCLUDED from every estimate.
# It is not discarded data: it exists so the estimate blocks start from a
# steady thermal and clock state instead of from an idle machine. That
# exclusion rule, the trim fraction, the round-0 drop and the block count were
# all fixed before this script was first run against a dosed build.
#
# THERMAL MODE. Ungated by construction (see research/e109_ab_leg.sh),
# counterbalanced within one session, entry and exit GPU temperature recorded
# per leg, honesty flags preserved. This host asymptotes at 40.55 C and cannot
# reach the 40 C cool gate, so no leg here is gate-qualified and no number
# here is a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e109_ab_session.sh TAG CONTROL_SPEC ARM_SPEC ...}"
shift
if [[ "$#" -lt 2 ]]; then
  echo "e109_ab_session: need a control spec and at least one arm spec" >&2
  exit 2
fi
specs=("$@")

blocks="${E109_BLOCKS:-8}"
tokens="${E109_TOKENS:-512}"
depth="${E109_DEPTH:-8}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

# --- one model-holding run at a time -----------------------------------------
# Reuse benchmark.sh's own lock and orphan scan rather than reimplementing
# them, exactly as benchmark-qwen-mtp.sh and benchmark-dflash.sh do, so this
# session and a local benchmark exclude each other in both directions.
LOCAL_ITERATE=1
LOCAL_SUBMIT=0
lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_guard_enabled\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${lock_definitions}"; then
  echo "e109_ab_session: could not reuse benchmark.sh's run-lock definitions" >&2
  exit 1
fi
for fn in local_run_guard_enabled local_run_lock_path acquire_local_run_lock \
          release_local_run_lock list_resident_model_processes \
          abort_if_model_already_resident; do
  if ! declare -F "${fn}" >/dev/null 2>&1; then
    echo "e109_ab_session: could not reuse benchmark.sh's ${fn}()" >&2
    exit 1
  fi
done
trap 'release_local_run_lock' EXIT
acquire_local_run_lock
abort_if_model_already_resident

# --- preflight ---------------------------------------------------------------
for tool in jq python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "e109_ab_session: missing ${tool}" >&2; exit 1; }
done
[[ -x "${swift_bin}" ]] || { echo "e109_ab_session: missing ${swift_bin}" >&2; exit 1; }
[[ -f "${weights}/config.json" ]] || {
  echo "e109_ab_session: missing ${weights}/config.json; run the local wrapper once first" >&2
  exit 1; }

eval "$(./setup-qwen-mtp.sh --print-paths)"
head_dir="${E109_HEAD_DIR:-${MLXFAST_QWEN_MTP_HEAD_DIR:?no MTP head}}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e109_ab_session: no MTP head at ${head_dir}" >&2; exit 1; }

# --- reference rows, generated ONCE for the whole session --------------------
# Same verb, same fixture, same row-count preflight the local wrapper uses.
# Cached across sessions: the rows depend on the target, the head and the
# decode chain, none of which an arm may change (an arm that does change them
# fails `all_tokens_matched` in every leg, which is the correctness gate).
golden="${E109_GOLDEN:-research/out/e109-golden-${tokens}.json}"
public_golden="correctness_prompts/public_longcopy_gate_english_512_256.json"
if ! jq -e --argjson tokens "${tokens}" '
      .reference_self_consistent == true and (.rows | length) >= ($tokens + 1)
    ' "${golden}" >/dev/null 2>&1; then
  echo "e109_ab_session: generating ${tokens} + 1 reference rows into ${golden}"
  mkdir -p "$(dirname "${golden}")"
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
    "${public_golden}" > "${out_dir}/plan.json"
  MLXFAST_NO_SANDBOX=1 "${swift_bin}" mtp-verify \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --emitted "${out_dir}/plan.json" \
    --generate "$(( tokens + 1 ))" \
    --mtp-depth "${depth}" \
    --output "${golden}" \
    --plan-output "${out_dir}/generated-plan.json" \
    > "${out_dir}/verify.log" 2>&1 || {
      echo "e109_ab_session: reference generation failed" >&2
      tail -20 "${out_dir}/verify.log" >&2
      exit 1; }
  jq -e --argjson tokens "${tokens}" '
      . as $g
      | $g.reference_self_consistent == true
        and ($g.rows | length) >= ($tokens + 1)
        and ([range(0; ($g.rows | length))
              | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)]
             | length) == 0
    ' "${golden}" >/dev/null || {
      echo "e109_ab_session: generated reference rows are not usable" >&2
      exit 1; }
fi

# --- the counterbalanced schedule --------------------------------------------
labels=()
envs=()
for spec in "${specs[@]}"; do
  labels+=("${spec%%=*}")
  rest="${spec#*=}"
  [[ "${rest}" == "${spec}" ]] && rest=""
  envs+=("${rest}")
done
n_arms="${#labels[@]}"

schedule="$(python3 - "${n_arms}" "${blocks}" <<'PY'
import sys
n, blocks = int(sys.argv[1]), int(sys.argv[2])
# Block 0 conditions the machine and is excluded from every estimate; blocks
# 1..blocks carry it. Rotation j on the forward pass, its mirror on the next
# block: over 2n estimate blocks each arm sits at each position exactly twice.
for b in range(blocks + 1):
    j = (b // 2) % n
    order = [(i + j) % n for i in range(n)]
    if b % 2 == 1:
        order.reverse()
    print(b, " ".join(str(i) for i in order))
PY
)"

session_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "e109 session ${tag} start=${session_start} arms=${labels[*]}" \
     "blocks=${blocks}(+1 conditioning) tokens=${tokens} depth=${depth}"

{
  echo "{"
  echo "  \"tag\": \"${tag}\","
  echo "  \"protocol\": \"e109-blocked-counterbalanced-round-paired\","
  echo "  \"control_arm\": \"${labels[0]}\","
  echo "  \"tokens\": ${tokens},"
  echo "  \"offered_depth\": ${depth},"
  echo "  \"estimate_blocks\": ${blocks},"
  echo "  \"conditioning_blocks\": 1,"
  echo "  \"round_zero_dropped\": true,"
  echo "  \"trim_fraction_each_tail\": 0.10,"
  echo "  \"outlier_rule\": \"none beyond the fixed trim; every leg enters the estimate\","
  echo "  \"randomised\": false,"
  echo "  \"counterbalanced\": \"rotation plus mirror over blocks\","
  echo "  \"session_start_utc\": \"${session_start}\""
  echo "}"
} > "${out_dir}/design.json"

failed=()
while read -r block order; do
  [[ -n "${block}" ]] || continue
  echo
  echo "=== block ${block} order ${order} ==="
  for idx in ${order}; do
    label="${labels[idx]}"
    leg_dir="${out_dir}/b${block}-${label}"
    printf '  leg %-6s ' "${label}"
    arm_env=()
    if [[ -n "${envs[idx]}" ]]; then
      IFS=',' read -r -a arm_env <<<"${envs[idx]}"
    fi
    E109_GOLDEN="${golden}" E109_HEAD_DIR="${head_dir}" \
    E109_TOKENS="${tokens}" E109_DEPTH="${depth}" \
      research/e109_ab_leg.sh "${leg_dir}" "${label}" "${arm_env[@]}" \
      || failed+=("b${block}-${label}")
    echo "block=${block}" >> "${leg_dir}/meta.txt"
    echo "arm_index=${idx}" >> "${leg_dir}/meta.txt"
  done
done <<<"${schedule}"

echo
echo "e109 session ${tag} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "e109_ab_session: failed legs: ${failed[*]}" >&2
  exit 1
fi

python3 research/e109_ab_report.py "${out_dir}" \
  --json "${out_dir}/ab-report.json"
