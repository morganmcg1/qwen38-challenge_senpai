#!/usr/bin/env bash
# E134: the shipped `pb6` depth price timed against `ship` in one session, on
# one worker binary.
#
#   usage: research/e134_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# A = ship   the pre-arm depth price
# B = pb6    the shipped arm, one priced step at the width-6 pass boundary
#
# Order inside a replicate is A B B A, so both arms have mean position 2.5 and
# monotone thermal drift cancels to first order. That counterbalance is what
# `program.md` requires before `MLXFAST_LOCAL_COOL_GATE=0` is a permitted mode.
# Entry and exit GPU temperature are recorded per leg, and every leg keeps
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
#
# WHY THIS SESSION EXISTS. The first `pb6` evidence compared archived legs from
# two different worker binaries collected up to eleven hours apart, and the
# commit range between those binaries carried an unrelated change as well as
# the arm flip. That confound is bounded at about 0.04 %, two orders below the
# effect, but it does not have to exist at all. Here both arms run the same
# bytes and the arm is chosen at run time by `MLX_E134_DEPTH_PRICE_ARM`.
#
# WHAT IS MEASURED. `parent_measured_seconds_per_token`, absolute. This is the
# DECODE side only; the report script also prints the seed-inclusive figure,
# because the ranked leg times seed processing and decoding together and no
# depth policy can move the seed.
#
# THE RULING IS ONE-SIDED. On this host `na6` clamps and `na7` spills, so the
# local width-6 cliff is steeper than the ranked one and `pb6` is biased to
# look better here than it will on the runner. A local win therefore proves
# nothing on its own. A local loss blocks.
#
# NO REBUILD BETWEEN LEGS, and the build happens once, first. HARNESS DEFECT 36
# means `./benchmark-qwen-mtp.sh` never rebuilds the worker, so the arm
# selector can be missing from the binary while every wrapper check passes.
# Phase 0 asserts the selector string is present in the built worker before any
# leg runs. `E134_NO_BUILD=1` skips the compile but keeps the assertion.
#
# FIRST numbers the first replicate, so a later session extends this estimate
# instead of restarting it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-3}"
tokens="${2:-512}"
label="${3:-s1}"
first="${4:-1}"

readonly PROMPT_ID="benchfixture"
readonly RUNS_PARENT=".mlxfast-private/e128/runs-abba"

# The arm witness. `DepthPriceArm(rawValue:)` falls back to the compiled
# default on any unrecognised string, so a selector that never arrives is
# indistinguishable from one that arrives and is honoured: both run `pb6`.
# Decoding is deterministic for a fixed fixture, token budget and build, so the
# round count is an exact behavioural signature of the arm that actually ran.
# These two values are from the archived 512-token `benchfixture` legs.
readonly WITNESS_ROUNDS_SHIP=78
readonly WITNESS_ROUNDS_PB6=82

witness_rounds_for() {
  case "$1" in
    ship) echo "${WITNESS_ROUNDS_SHIP}" ;;
    pb6)  echo "${WITNESS_ROUNDS_PB6}" ;;
    *) echo "e134_abba: unknown arm $1" >&2; return 2 ;;
  esac
}

if [[ "${tokens}" != "512" ]]; then
  echo "e134_abba: witness round counts are pinned at 512 tokens;" \
       "re-derive them before timing ${tokens}" >&2
  exit 2
fi

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e134_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  echo "${dirty}" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"

# PHASE 0: build once, then prove the selector is IN the binary about to run.
build_args=(--require MLX_E134_DEPTH_PRICE_ARM)
[[ "${E134_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
echo "=== e134_abba phase 0: worker build and selector assertion ==="
senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
  echo "e134_abba: the worker does not carry the arm selector; not timing" >&2
  exit 3; }

session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e134_abba: session_commit=${session_commit}"
echo "e134_abba: session_worker_sha256=${session_worker}"

# PHASE 1: the counterbalanced timed session.
failures=0
discarded=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in ship pb6 pb6 ship; do
    position=$((position + 1))
    slot="${label}k${rep}p${position}${arm}"
    out="${RUNS_PARENT}/${slot}/${PROMPT_ID}"
    want_rounds="$(witness_rounds_for "${arm}")" || exit 2

    echo "=== e134_abba ${slot}: arm=${arm} replicate=${rep}" \
         "position=${position} tokens=${tokens} ==="
    env MLX_E134_DEPTH_PRICE_ARM="${arm}" \
        E128_FORCE=1 \
        E128_NO_TRACE=1 \
        E128_TOKENS="${tokens}" \
        E128_DEPTH=8 \
        E128_RUNS_DIR="runs-abba/${slot}" \
      research/e128_session.sh "${PROMPT_ID}"
    status=$?

    got_rounds="$(sed -n 's/^round_count=//p' "${out}/meta.txt" 2>/dev/null)"
    got_draft="$(
      sed -n 's/^effective_mean_draft_len=//p' "${out}/meta.txt" 2>/dev/null)"
    witness="ok"
    if [[ "${got_rounds}" != "${want_rounds}" ]]; then
      witness="MISMATCH"
      discarded=$((discarded + 1))
      echo "e134_abba: ${slot} requested ${arm} but ran" \
           "${got_rounds:-<no round count>} rounds, wanted ${want_rounds}" >&2
    fi
    {
      echo "e134_arm_requested=${arm}"
      echo "e134_arm_witness_rounds_want=${want_rounds}"
      echo "e134_arm_witness_rounds_got=${got_rounds:-none}"
      echo "e134_arm_witness_mean_draft_got=${got_draft:-none}"
      echo "e134_arm_witness=${witness}"
      echo "e134_replicate=${rep}"
      echo "e134_position=${position}"
      echo "e134_session_commit=${session_commit}"
      echo "e134_session_worker_sha256=${session_worker}"
      echo "e134_ruling=one-sided: a local win proves nothing, a local loss blocks"
    } >> "${out}/meta.txt"

    if ((status != 0)); then
      echo "e134_abba: ${slot} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e134_abba: ${failures} failed legs, ${discarded} witness mismatches"
python3 research/e134_abba_report.py --label "${label}" \
  --json "research/e134-artifacts/abba-${label}.json"
exit $(( failures > 0 || discarded > 0 ))
