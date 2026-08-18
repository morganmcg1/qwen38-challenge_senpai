#!/usr/bin/env bash
# Research-only (qwen38-r1-e29): run one --local-iterate arm and keep the
# per-round phase trace plus the CLI reports the harness would otherwise
# delete with its scratch directory.
#
#   research/e29-run.sh LABEL=TRACE:DEPTH [...]
#
#     TRACE  0 | 1 | 2
#              0  no trace (control that prices the trace perturbation)
#              1  MLX_QWEN_MTP_TRACE (per-round six-way host/GPU split)
#              2  also MLX_QWEN_MTP_TRACE_SYNC_HEAD: drain the head chain
#                 before the verify-build window so verify_build_us stops
#                 absorbing head-chain GPU time. Attribution probe only.
#     DEPTH  offered draft depth (MLXFAST_QWEN_MTP_DEPTH)
#
# e.g. research/e29-run.sh T1=1:8 N1=0:8
#
# The label names the MEASUREMENT, the pair names the CONFIG. TRACE=1 adds
# host-side string formatting and file writes inside the round, so a TRACE=0
# arm is required to price that perturbation rather than assume it away.
#
# Every gate benchmark-qwen-mtp.sh owns (drift tripwire, orphan scan, run lock,
# report seals) runs unmodified. The 40C cool gate is the one exception and it
# is disabled deliberately: idle GPU on this host sits at ~42.9C, above
# COOL_GATE_TEMP_C=40, so the real gate can never be satisfied. Entry and exit
# temperatures are sampled per arm instead, and the resulting numbers are
# reported with cool_gate_passed_real_gate=false / gate_qualified_for_timing=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repo_root="${PWD}"
runs_root="${E29_RUNS_ROOT:-${repo_root}/.mlxfast-private/e29/runs}"
head_dir="${E29_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared}"
tokens="${E29_TOKENS:-256}"

# Leaked overrides would silently answer a different question under the same
# label, and the trace gate is exactly the variable under test.
for v in $(env | sed -n 's/^\(MLX_QWEN_[A-Z_]*\)=.*/\1/p'); do unset "${v}"; done

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W all_power=\(.all_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

for bin in .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker; do
  [[ -x "${bin}" ]] || { echo "e29-run: missing ${bin}; run research/rebuild.sh" >&2; exit 1; }
done

status=0
for spec in "$@"; do
  label="${spec%%=*}"
  cfg="${spec#*=}"
  trace="${cfg%%:*}"
  depth="${cfg#*:}"
  out="${runs_root}/${label}"

  rm -rf "${out}"; mkdir -p "${out}/reports"

  export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_QWEN_MTP_DEPTH="${depth}"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
  export MLXFAST_MACMON_BIN="${macmon_bin}"
  export MLXFAST_LOCAL_COOL_GATE=0

  if [[ "${trace}" == "1" || "${trace}" == "2" ]]; then
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${out}/rounds.trace"
    # The worker sandbox denies file-write*, and mtp-timed swallows worker
    # stderr, so the file sink needs the documented local relaxation.
    export MLXFAST_NO_SANDBOX=1
  else
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH MLXFAST_NO_SANDBOX
  fi
  if [[ "${trace}" == "2" ]]; then
    export MLX_QWEN_MTP_TRACE_SYNC_HEAD=1
  else
    unset MLX_QWEN_MTP_TRACE_SYNC_HEAD
  fi

  {
    echo "label=${label}"
    echo "trace=${trace}"
    echo "offered_depth=${depth}"
    echo "tokens=${tokens}"
    echo "head_dir=${head_dir}"
    echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
    echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    echo "host=$(hostname)"
    echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "no_sandbox=${MLXFAST_NO_SANDBOX:-0}"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "mlx_qwen_env=$(env | sed -n 's/^\(MLX_QWEN_[A-Z_]*\)=.*/\1/p' | sort | tr '\n' ',')"
    echo "thermal_before=$(sample_thermal)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  echo "=== e29-run: ${label} (trace ${trace} depth ${depth} tokens ${tokens}) ==="
  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?
  {
    echo "exit=${rc}"
    echo "thermal_after=$(sample_thermal)"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"
  [[ ${rc} -eq 0 ]] || status=1
done

exit "${status}"
