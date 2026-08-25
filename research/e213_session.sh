#!/usr/bin/env bash
# E213 Stage-1: decisive local timing of the staged QMV rows-per-group retune
# and of the register-boundary control.
#
#   research/e213_session.sh TAG_PREFIX [TOKENS]
#
# ONE binary, THREE arms selected at process start by DARKBLOOM_E213_QMV_ARM:
#
#   off      shipped staged plan: (6,3) (7,4) (8,4) (9,5)
#   retuned  the assigned family: (6,4) (7,5) (8,5) (9,5)
#            (6,5) is not buildable, because 6 % 5 == 1 makes a one-input tail
#            group, so m = 6 takes the widest legal first group, 4
#   na6      the register-boundary control: (6,3) (7,4) (8,4) (9,6)
#
# G(m) = ceil(m / IPG) is 2 at m = 6, 7, 8 and 9 in ALL THREE arms, so no arm
# changes the number of weight passes. `retuned` therefore tests the row split
# alone and is predicted null. `na6` holds the pass count and takes the widest
# compiled body from NA = 5 (125 lane-weighted live values) to NA = 6 (144),
# across the Apple 128-register boundary, so it can fail and it prices the
# register term that FINDING 500 named.
#
# SIX legs in the palindrome off retuned na6 na6 retuned off, so monotone
# session drift cancels to first order in the two orientations.
#
# The arms run the identical cap-8 depth schedule. The contrast is kernel
# dispatch only, so the round `(d, acc)` trajectory is a function of the head
# and the schedule alone and must be identical across arms. The analysis
# asserts that identity before it pairs rounds (RULE 396(b)).
#
# Decision statistics (RULE 394), both round-endpoint or leg-absolute:
#   (a) trusted-parent `block_request_seconds`, paired by round index, split by
#       served width;
#   (b) whole-leg absolute candidate `mtp_seconds_per_token` from score.json.
# Candidate-side `round_us` and the phase timers attribute only.
#
# Arm witness (RULE 391(b)): every traced round carries `qmv_plan=`, built from
# the selected plan table rather than from the environment. A leg whose witness
# does not match its requested arm fails the session.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 under the standing three
# conditions: palindrome arm order, entry and exit GPU temperature recorded per
# leg, and cool_gate_passed_real_gate=false / gate_qualified_for_timing=false
# kept verbatim.
#
# Exactness still runs on every leg: --local-iterate checks both timed legs
# against reference rows this build generates for the full window, and the
# trusted parent closes the row ledger.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: research/e213_session.sh TAG_PREFIX [TOKENS]}"
tokens="${2:-512}"
order="${E213_ORDER:-off retuned na6 na6 retuned off}"

out_root="research/out/${prefix}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"

research/e168_build.sh || exit 1

head_dir="${E213_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e213: declared head tree missing at ${head_dir}" >&2
  exit 2
}
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_NO_SANDBOX=1

digest() { shasum -a 256 "${worker}" | cut -d' ' -f1; }
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

baseline_digest="$(digest)"
echo "e213: worker ${baseline_digest}"
echo "e213: head $(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
echo "e213: tokens ${tokens}, cool gate OFF (ungated, counterbalanced)"

read -r -a session <<< "${order}"
echo "e213: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  case "${arm}" in
    off)
      unset DARKBLOOM_E213_QMV_ARM
      expect_plan="selective-m6+ipg9-5" ;;
    retuned)
      export DARKBLOOM_E213_QMV_ARM=retuned
      expect_plan="selective-m6+ipg9-5+e213-4-5-5-5" ;;
    na6)
      export DARKBLOOM_E213_QMV_ARM=na6
      expect_plan="selective-m6+ipg9-6+e213-3-4-4-6" ;;
    *) echo "e213: unknown arm ${arm}" >&2; status=2; break ;;
  esac

  out="${out_root}/leg$(printf '%02d' "${leg}")-${arm}"
  rm -rf "${out}"; mkdir -p "${out}/reports"
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace.txt"
  export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
  export MLXFAST_CAPTURE_DIR="${PWD}/${out}/reports"
  export MLXFAST_CAPTURE_REAL_BIN="${PWD}/.build/release/mlxfast-swift"
  export MLXFAST_SWIFT_BIN="${PWD}/research/capture-cli.sh"

  before="$(digest)"
  [[ "${before}" == "${baseline_digest}" ]] || {
    echo "e213: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e213: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate \
    > "${out}/wrapper.out" 2> "${out}/wrapper.err"
  rc=$?

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  plans="$(LC_ALL=C sed -n 's/.*qmv_plan=\([^ ]*\).*/\1/p' "${out}/trace.txt" \
    2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')"
  {
    echo "tag=${prefix}"
    echo "experiment=e213-ipg5-retune-m678"
    echo "harness=local"
    echo "e213_arm=${arm}"
    echo "e213_leg=${leg}"
    echo "e213_session_order=${session[*]}"
    echo "e213_arm_env=DARKBLOOM_E213_QMV_ARM=${DARKBLOOM_E213_QMV_ARM:-<unset>}"
    echo "e213_expected_plan=${expect_plan}"
    echo "e213_observed_plans=${plans:-<none>}"
    echo "gpu_temp_entry_c=${entry:-unavailable}"
    echo "gpu_temp_exit_c=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
    echo "candidate_sha=$(git rev-parse HEAD)"
    echo "campaign_base_sha=523da3be9474973b99fc3bbada550f78ab862406"
    echo "dirty_candidate_paths=$(
      git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "os=$(sw_vers -productVersion)"
    echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
    echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
    echo "sandbox=off"
    echo "trace=1"
    echo "cool_gate=0"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "official_or_ranked_score=false"
    echo "timing_source=trusted-parent-report"
    echo "tokens=${tokens}"
    echo "mode=qwen-mtp-local-iterate"
    echo "exit=${rc}"
    echo "rounds_traced=$(
      grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
    echo "started=${started}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e213: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -20 "${out}/wrapper.err" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e213: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
  [[ "${plans}" == "${expect_plan}" ]] || {
    echo "e213: leg ${leg} arm=${arm} witness '${plans}' != '${expect_plan}'" >&2
    status=1; break
  }
  echo "e213: leg ${leg} arm=${arm} plan=${plans} exit=${rc} exit_temp=${exit_temp}C"
done

echo "e213_session: exit ${status}"
exit "${status}"
