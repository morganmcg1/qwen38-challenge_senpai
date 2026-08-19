#!/usr/bin/env bash
# Run one timed E60 leg at the ranked command-buffer geometry.
#
# usage: research/e60_run_leg.sh TAG ARM TOKENS
#
# The arm is selected by staging its prebuilt binaries into the canonical build
# paths, so the metallib and its fingerprint sidecar keep resolving exactly as
# they do in an ordinary local run and the working tree stays clean.
#
# MLX_MAX_MB_PER_BUFFER=512 and MLX_MAX_OPS_PER_BUFFER=50 are the ranked
# command-buffer geometry. RuntimeStartupMemoryPolicy force-sets the same pair,
# but only at 96 GiB or more of physical memory, so it never fires on this
# 48 GiB host and the export is the only way to reach the ranked geometry here.
# `MLX_` is one of the prefixes the trusted worker environment allowlist keeps
# (QwenRuntimeWorker.swift sanitizedRuntimeWorkerEnvironment), so the pair
# reaches the worker; the leg records the worker's own environment as proof.
#
# MLXFAST_LOCAL_COOL_GATE=0 is set: the session is ABBA/palindrome
# counterbalanced, entry and exit GPU temperature are recorded per leg, and the
# result carries cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e60_run_leg.sh TAG ARM TOKENS}"
arm="${2:?usage: e60_run_leg.sh TAG ARM TOKENS}"
tokens="${3:?usage: e60_run_leg.sh TAG ARM TOKENS}"

bin_dir="research/out/e60/bin/${arm}"
for product in mlxfast-swift mlxfast-runtime-worker; do
  if [[ ! -x "${bin_dir}/${product}" ]]; then
    echo "e60: missing prebuilt ${product} for arm ${arm}" >&2
    exit 1
  fi
done

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

cp "${bin_dir}/mlxfast-swift" .build/release/mlxfast-swift
cp "${bin_dir}/mlxfast-runtime-worker" .build-worker/release/mlxfast-runtime-worker

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50
export MLXFAST_LOCAL_COOL_GATE=0

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

{
  echo "tag=${tag}"
  echo "arm=${arm}"
  echo "tokens=${tokens}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "cool_gate=0"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain | grep -v '^?? research/' | wc -l | tr -d ' ')"
  echo "staged_cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "staged_worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  cat "${bin_dir}/provenance.txt"
} > "${out}/meta.txt"

{
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate > "${out}/wrapper.out" 2> "${out}/wrapper.err" &
bench_pid=$!

# One-shot proof that the ranked geometry reached the worker process itself.
# Bounded to the load phase, and it stops at the first successful capture, so it
# cannot touch either timed window.
(
  for _ in $(seq 1 30); do
    worker_pid="$(pgrep -f 'mlxfast-runtime-worker' | head -1)"
    if [[ -n "${worker_pid}" ]]; then
      ps eww -o command= -p "${worker_pid}" 2>/dev/null \
        | tr ' ' '\n' | grep -E '^MLX_(MAX|METAL)' > "${out}/worker-env.txt"
      if [[ -s "${out}/worker-env.txt" ]]; then exit 0; fi
    fi
    sleep 3
  done
) &
env_probe_pid=$!

wait "${bench_pid}"
status=$?
wait "${env_probe_pid}" 2>/dev/null

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"
exit "${status}"
