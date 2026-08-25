#!/usr/bin/env bash
# E208 Stage-1: decisive local timing of the staged QMV (9,3) -> (9,5) entry.
#
#   research/e208_session.sh TAG_PREFIX [TOKENS]
#
# ONE binary, TWO arms selected at process start by DARKBLOOM_E208_QMV_ARM:
#
#   off   shipped staged plan, IPG = 3 at m = 9, G(9) = 3 weight passes
#   on    stagedWide9,         IPG = 5 at m = 9, G(9) = 2 weight passes
#
# FOUR legs in the palindrome off, on, on, off, so monotone session drift
# cancels to first order in the two orientations.
#
# The arms run the identical cap-8 depth schedule. The contrast is kernel
# dispatch only, so the round `(d, acc)` trajectory is a function of the head
# and the schedule alone and must be identical across arms. The analysis
# asserts that identity before it pairs rounds (RULE 396(b)).
#
# Decision statistics (RULE 394), both round-endpoint or leg-absolute:
#   (a) trusted-parent `block_request_seconds`, paired by round index, on the
#       m = 9 rounds;
#   (b) whole-leg absolute candidate `mtp_seconds_per_token` from score.json.
# Candidate-side `round_us` and the phase timers attribute only.
#
# Arm witness (RULE 391(b)): every traced round carries `qmv_plan=`, which is
# built from the selected plan table rather than from the environment. A leg
# whose witness does not match its requested arm fails the session.
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

prefix="${1:?usage: research/e208_session.sh TAG_PREFIX [TOKENS]}"
tokens="${2:-512}"
order="${E208_ORDER:-off on on off}"

out_root="research/out/${prefix}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"

research/e168_build.sh || exit 1

head_dir="${E208_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e208: declared head tree missing at ${head_dir}" >&2
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
echo "e208: worker ${baseline_digest}"
echo "e208: head $(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
echo "e208: tokens ${tokens}, cool gate OFF (ungated, counterbalanced)"

read -r -a session <<< "${order}"
echo "e208: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  case "${arm}" in
    off) unset DARKBLOOM_E208_QMV_ARM; expect_plan="selective-m6+ipg9-3" ;;
    on) export DARKBLOOM_E208_QMV_ARM=on; expect_plan="selective-m6+ipg9-5" ;;
    *) echo "e208: unknown arm ${arm}" >&2; status=2; break ;;
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
    echo "e208: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e208: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
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
    echo "experiment=e208-qmv-9row-ipg5"
    echo "harness=local"
    echo "e208_arm=${arm}"
    echo "e208_leg=${leg}"
    echo "e208_session_order=${session[*]}"
    echo "e208_arm_env=DARKBLOOM_E208_QMV_ARM=${DARKBLOOM_E208_QMV_ARM:-<unset>}"
    echo "e208_expected_plan=${expect_plan}"
    echo "e208_observed_plans=${plans:-<none>}"
    echo "gpu_temp_entry_c=${entry:-unavailable}"
    echo "gpu_temp_exit_c=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
    echo "base_sha=$(git rev-parse HEAD)"
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
    echo "e208: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -20 "${out}/wrapper.err" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e208: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
  [[ "${plans}" == "${expect_plan}" ]] || {
    echo "e208: leg ${leg} arm=${arm} witness '${plans}' != '${expect_plan}'" >&2
    status=1; break
  }
  echo "e208: leg ${leg} arm=${arm} plan=${plans} exit=${rc} exit_temp=${exit_temp}C"
done

echo "e208_session: exit ${status}"
exit "${status}"
