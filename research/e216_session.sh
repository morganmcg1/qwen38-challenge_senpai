#!/usr/bin/env bash
# E216: decisive local timing of the cooperative weight stream at the two-group
# QMV widths.
#
#   research/e216_session.sh TAG_PREFIX [TOKENS] [ORDER]
#
# ONE binary. The arm is selected at process start by DARKBLOOM_E216_QMV_ARM:
#
#   off    shipped `split` grid: first_m = tid.x * IPG, out_row = tid.y*8 + sgid*4
#   coop   paired grid:          first_m = sgid * IPG,  out_row = tid.y*4
#
# Both mappings launch the same threadgroup count with the same threadgroup
# shape (32, 2, 1) and the same per-simdgroup geometry (NA <= 5, rows 4). The
# only difference is WHICH simdgroup owns which token-column group. At a width
# where G(m) = ceil(m / IPG) is 2, `split` places the two column groups of one
# output-row slice in DIFFERENT threadgroups, so the same weight rows stream
# from device memory twice. `coop` places both groups in ONE threadgroup, one
# per simdgroup, so the slice streams once.
#
# Moved widths are the staged G == 2 entries: m = 6 (IPG 3, mlp.down cell only),
# 7 (IPG 4), 8 (IPG 4) and 9 (IPG 5). m <= 5 is the flat control band: those
# widths are G == 1 and keep the shipped mapping in both arms.
#
# FOUR legs in the palindrome off coop coop off, so monotone session drift
# cancels to first order in the two orientations. E216_ORDER overrides it.
#
# The arms run the identical cap-8 depth schedule. The contrast is kernel index
# mapping only, so the round `(d, acc)` trajectory is a function of the head and
# the schedule alone and must be identical across arms. The analysis asserts
# that identity before it pairs rounds (RULE 396(b)).
#
# Decision statistics (RULE 394), round endpoint or leg absolute only:
#   (a) trusted-parent `block_request_seconds`, paired by round index and split
#       by served width;
#   (b) whole-leg absolute candidate `mtp_seconds_per_token` from score.json.
# Candidate-side `round_us` and the phase timers attribute only.
#
# Arm witness (RULE 391(b)): every traced round carries `qmv_plan=` built from
# the selected plan table, plus the cumulative launch counters `qmv_coop=`,
# `qmv_split=` and `qmv_split_g2=`. A coop leg must end with qmv_coop > 0 AND
# qmv_split_g2 == 0: every G == 2 dispatch took the paired grid. An off leg must
# end with qmv_coop == 0 AND qmv_split_g2 > 0. A leg that fails its witness
# fails the session.
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

prefix="${1:?usage: research/e216_session.sh TAG_PREFIX [TOKENS] [ORDER]}"
tokens="${2:-512}"
order="${3:-${E216_ORDER:-off coop coop off}}"

out_root="research/out/${prefix}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"

research/e168_build.sh || exit 1

head_dir="${E216_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e216: declared head tree missing at ${head_dir}" >&2
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
# The launch counters are cumulative over the process, so the LAST traced round
# carries the whole-leg total.
last_counter() {
  LC_ALL=C sed -n "s/.*[[:space:]]$1=\([0-9]*\).*/\1/p" "$2" 2>/dev/null | tail -1
}

baseline_digest="$(digest)"
echo "e216: worker ${baseline_digest}"
echo "e216: head $(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
echo "e216: tokens ${tokens}, cool gate OFF (ungated, counterbalanced)"

read -r -a session <<< "${order}"
echo "e216: ${#session[@]} legs, order ${session[*]}"

status=0
leg=0
for arm in "${session[@]}"; do
  leg=$((leg + 1))
  case "${arm}" in
    off)
      unset DARKBLOOM_E216_QMV_ARM
      expect_plan="selective-m6+ipg9-5" ;;
    coop)
      export DARKBLOOM_E216_QMV_ARM=coop
      expect_plan="selective-m6+ipg9-5+e216-coop-g2" ;;
    *) echo "e216: unknown arm ${arm}" >&2; status=2; break ;;
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
    echo "e216: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }
  entry="$(gpu_temp)"
  echo "=== e216: leg ${leg}/${#session[@]} arm=${arm} entry=${entry}C ==="
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate \
    > "${out}/wrapper.out" 2> "${out}/wrapper.err"
  rc=$?

  exit_temp="$(gpu_temp)"
  after="$(digest)"
  plans="$(LC_ALL=C sed -n 's/.*qmv_plan=\([^ ]*\).*/\1/p' "${out}/trace.txt" \
    2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')"
  coop_launches="$(last_counter qmv_coop "${out}/trace.txt")"
  split_launches="$(last_counter qmv_split "${out}/trace.txt")"
  split_g2_launches="$(last_counter qmv_split_g2 "${out}/trace.txt")"
  {
    echo "tag=${prefix}"
    echo "experiment=e216-coop-weight-stream"
    echo "harness=local"
    echo "e216_arm=${arm}"
    echo "e216_leg=${leg}"
    echo "e216_session_order=${session[*]}"
    echo "e216_arm_env=DARKBLOOM_E216_QMV_ARM=${DARKBLOOM_E216_QMV_ARM:-<unset>}"
    echo "e216_expected_plan=${expect_plan}"
    echo "e216_observed_plans=${plans:-<none>}"
    echo "e216_qmv_coop_launches=${coop_launches:--1}"
    echo "e216_qmv_split_launches=${split_launches:--1}"
    echo "e216_qmv_split_g2_launches=${split_g2_launches:--1}"
    echo "gpu_temp_entry_c=${entry:-unavailable}"
    echo "gpu_temp_exit_c=${exit_temp:-unavailable}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
    echo "candidate_sha=$(git rev-parse HEAD)"
    echo "campaign_base_sha=1f978983694342095f2fb7a8c5398d68fb6e1ee1"
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
    echo "e216: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -20 "${out}/wrapper.err" >&2
    status=1; break
  fi
  [[ "${before}" == "${after}" ]] || {
    echo "e216: worker changed DURING leg ${leg}" >&2
    status=1; break
  }
  [[ "${plans}" == "${expect_plan}" ]] || {
    echo "e216: leg ${leg} arm=${arm} witness '${plans}' != '${expect_plan}'" >&2
    status=1; break
  }
  if [[ "${arm}" == "coop" ]]; then
    if [[ "${coop_launches:-0}" -le 0 || "${split_g2_launches:--1}" -ne 0 ]]; then
      echo "e216: leg ${leg} coop witness failed: qmv_coop=${coop_launches} qmv_split_g2=${split_g2_launches}" >&2
      status=1; break
    fi
  else
    if [[ "${coop_launches:--1}" -ne 0 || "${split_g2_launches:-0}" -le 0 ]]; then
      echo "e216: leg ${leg} off witness failed: qmv_coop=${coop_launches} qmv_split_g2=${split_g2_launches}" >&2
      status=1; break
    fi
  fi
  echo "e216: leg ${leg} arm=${arm} plan=${plans} coop=${coop_launches}" \
    "split=${split_launches} split_g2=${split_g2_launches} exit=${rc}" \
    "exit_temp=${exit_temp}C"
done

echo "e216_session: exit ${status}"
exit "${status}"
