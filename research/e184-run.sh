#!/usr/bin/env bash
# E184 prefill cost decomposition: run_job entry points (argv only, no env
# plumbing at the call site).
#
#   research/e184-run.sh probe                       # isolated scan probe, no model
#   research/e184-run.sh profile TAG MODE TOKENS GATE
#   research/e184-run.sh session TOKENS GATE         # ABBA: off,coarse,fine,fine,coarse,off
#
# `profile` runs ONE matched serial/MTP pair through the unmodified
# benchmark-qwen-mtp.sh wrapper. MODE selects the research profiler:
#   off     the instrument is inert; the run measures the natural prefill and
#           the trusted `seed_prefill_seconds` field is the anchor
#   coarse  layer-level eval boundaries only (64 per forward)
#   fine    every sub-phase boundary (~400 per forward)
#
# GATE=1 keeps the real 40 C cool-down gate; GATE=0 is the standing permitted
# ungated measurement mode -- allowed only ABBA-counterbalanced within one
# session with entry/exit temperatures recorded, and never reported as a
# gate-qualified or ranked number.
#
# research/capture-cli.sh keeps every trusted CLI report after the measured
# phase ends, so `seed_prefill_seconds`, `decode_seconds` and
# `block_request_seconds` survive the wrapper's cleanup trap. No trusted file is
# edited and the timed window is untouched.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

export PATH="${HOME}/bin:${HOME}/.local/bin:${PATH}"

gpu_temp() {
  if command -v macmon >/dev/null 2>&1; then
    macmon pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg' 2>/dev/null || echo "unavailable"
  else
    echo "macmon-missing"
  fi
}

run_profile() {
  local tag="$1" mode="$2" tokens="$3" gate="$4"
  local capture="${PWD}/research/capture-e184-${tag}"
  local score="research/score-e184-${tag}.json"
  local profile_out="${PWD}/research/e184-phases-${tag}.jsonl"

  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_SWIFT_BIN="research/capture-cli.sh"
  export MLXFAST_CAPTURE_REAL_BIN="${PWD}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${capture}"
  export MLXFAST_SCORE_PATH="${score}"
  export MLXFAST_LOCAL_COOL_GATE="${gate}"
  if [[ "${mode}" == "off" ]]; then
    unset DARKBLOOM_E184_PREFILL_PROFILE
    unset DARKBLOOM_E184_PREFILL_PROFILE_OUT
  else
    export DARKBLOOM_E184_PREFILL_PROFILE="${mode}"
    export DARKBLOOM_E184_PREFILL_PROFILE_OUT="${profile_out}"
  fi

  rm -rf "${capture}" "${score}" "${profile_out}"
  echo "e184: HEAD $(git rev-parse HEAD)"
  echo "e184: worker sha256 $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "e184: cli sha256 $(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "e184: tag=${tag} mode=${mode} tokens=${tokens} cool_gate=${gate}"
  echo "e184: gpu_temp_entry=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  ./benchmark-qwen-mtp.sh --local-iterate
  local status=$?

  echo "e184: gpu_temp_exit=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "e184: wrapper_exit=${status} tag=${tag}"
  return "${status}"
}

verb="${1:-}"
case "${verb}" in
  probe)
    export MLXFAST_RUN_E184_PROBE=1
    export MLXFAST_E184_PROBE_OUT="${PWD}/research/e184-scan-probe.json"
    echo "e184: gpu_temp_entry=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    swift test --force-resolved-versions \
      --filter E184GatedDeltaScanCostTests
    status=$?
    echo "e184: gpu_temp_exit=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit "${status}"
    ;;
  profile)
    run_profile "${2:?tag}" "${3:?mode}" "${4:?tokens}" "${5:?gate}"
    exit $?
    ;;
  session)
    tokens="${2:?tokens}"
    gate="${3:?gate}"
    overall=0
    for arm in a1-off a2-coarse a3-fine a4-fine a5-coarse a6-off; do
      mode="${arm##*-}"
      if ! run_profile "${arm}" "${mode}" "${tokens}" "${gate}"; then
        overall=1
        echo "e184: arm ${arm} FAILED; continuing so the session stays counterbalanced" >&2
      fi
    done
    exit "${overall}"
    ;;
  *)
    echo "usage: research/e184-run.sh probe|profile TAG MODE TOKENS GATE|session TOKENS GATE" >&2
    exit 2
    ;;
esac
