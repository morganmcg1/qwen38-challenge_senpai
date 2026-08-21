#!/usr/bin/env bash
# E87 rung 3: run ./benchmark-qwen-mtp.sh --local-submit for one or more head
# arms and record the identity of each leg.
#
#   usage: research/e87_submit_gate.sh PREFIX [ARM ...]     (default: declared)
#
# --local-submit checks the candidate against the 1024-step public golden
# (correctness_prompts/public_longcopy_gate_english_512_1024.json) instead of
# the 256-step fixture the edit loop uses. E87_TOKENS raises the decode window
# from the wrapper default of 128 to the full 512, so one leg is both the
# pre-submit check and the 512-token exactness check, including post-EOS
# continuation and row-ledger closure.
#
# The legs are UNGATED, like the rung-2 timing session. These legs decide
# correctness, not speed, so the thermal gate has no bearing on the verdict;
# meta.txt still records gate_qualified_for_timing=false so no score taken here
# can be mistaken for a gated measurement.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e87_submit_gate.sh PREFIX [ARM ...]}"
cache="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
tokens="${E87_TOKENS:-512}"

shift || true
if (($#)); then
  declare -a arms=("$@")
else
  declare -a arms=(declared)
fi

dir_for() {
  case "$1" in
    declared|dense|derived|derived25|derived15) echo "${cache}/mtp-head-declared-run" ;;
    armc) echo "${cache}/e87/built/e87-armC-plain-k12292-p25-run" ;;
    *) echo "e87_submit_gate.sh: unknown arm $1" >&2; exit 2 ;;
  esac
}

# `derived*` is option B: the DECLARED head plus the cluster index the runtime
# builds from it during the untimed warm. Every other arm pins the gate off so
# a shipped index or the dense readout is what actually runs.
index_for() {
  case "$1" in
    derived|derived25|derived15) echo "1" ;;
    *) echo "0" ;;
  esac
}

probe_for() {
  case "$1" in
    derived15) echo "0.15" ;;
    *) echo "${E87_PROBE_FRACTION:-0.25}" ;;
  esac
}

dirty="$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
if [[ "${dirty}" != "0" ]]; then
  echo "e87_submit_gate.sh: ${dirty} dirty candidate path(s)" >&2
  exit 1
fi

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

for arm in "${arms[@]}"; do
  head_dir="$(dir_for "${arm}")"
  probe_fraction="$(probe_for "${arm}")"
  if [[ ! -s "${head_dir}/config.json" ]]; then
    echo "e87_submit_gate.sh: no head at ${head_dir}" >&2
    exit 1
  fi

  out="research/out/${prefix}-${arm}"
  rm -rf "${out}"
  mkdir -p "${out}"

  {
    echo "tag=${prefix}-${arm}"
    echo "experiment=e87-coarse-draft-shortlist-traffic"
    echo "e87_arm=${arm}"
    echo "harness=local"
    echo "local_mode=--local-submit"
    echo "tokens=${tokens}"
    echo "golden=correctness_prompts/public_longcopy_gate_english_512_1024.json"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "official_or_ranked_score=false"
    echo "base_sha=$(git rev-parse HEAD)"
    echo "dirty_candidate_paths=${dirty}"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
    echo "head_dir=${head_dir}"
    echo "e87_derived_index=$(index_for "${arm}")"
    echo "e87_probe_fraction=${probe_fraction}"
    echo "worker_sha256=$(
      shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
    echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
    echo "gpu_temp_entry_c=$(gpu_temp)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  # The gate runs the worker under benchmark.sh's Seatbelt profile, which
  # denies every file write except /dev/null. The two research sinks below
  # therefore stay empty here, and that is the evidence that the derived index
  # touches no file on a scored run. Read them from a trace leg instead:
  # research/e79_trace_leg.sh sets MLXFAST_NO_SANDBOX=1.
  MLX_E87_DERIVED_INDEX="$(index_for "${arm}")" \
  MLX_E87_PROBE_FRACTION="${probe_fraction}" \
  MLX_E87_DERIVED_DUMP="${E87_DERIVED_DUMP:-${PWD}/${out}/derived-order.bin}" \
  MLX_E87_DERIVED_LOG="${PWD}/${out}/derived.log" \
  MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}" \
  MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS="${tokens}" \
  MLXFAST_SCORE_PATH="${PWD}/${out}/score.json" \
  MLXFAST_LOCAL_COOL_GATE=0 \
  MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}" \
    ./benchmark-qwen-mtp.sh --local-submit \
      > "${out}/wrapper.out" 2> "${out}/wrapper.err"
  status=$?

  {
    echo "gpu_temp_exit_c=$(gpu_temp)"
    echo "post_run_worker_sha256=$(
      shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
    echo "exit=${status}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"

  echo "submit-gate ${prefix}-${arm} exit=${status}"
  ((status == 0)) || exit "${status}"
done
