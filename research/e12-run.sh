#!/bin/bash
# e12 seed-prefill charge: run_job entry points (argv only, no env plumbing).
#
#   research/e12-run.sh build
#   research/e12-run.sh iterate TOKENS TRACE TAG
#
# `iterate` measures one matched serial/MTP pair through the unmodified
# benchmark-qwen-mtp.sh wrapper, behind research/await-lock-then-run.sh, and keeps
# the per-arm payloads via research/capture-cli.sh. TRACE=1 sets
# MLX_QWEN_MTP_TRACE=1, which forwards the session's worker-stderr trace; a traced
# run carries decode-side trace overhead and is never a timed headline.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

gpu_temp() {
  if command -v macmon >/dev/null 2>&1; then
    macmon pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg' 2>/dev/null || echo "unavailable"
  else
    echo "macmon-missing"
  fi
}

case "${1:-}" in
build)
  export MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
  export MLXFAST_SKIP_MLX_METALLIB=1
  export MLXFAST_SKIP_MACMON_INSTALL=1
  export MLXFAST_SKIP_HOMEBREW_INSTALL=1
  export MLXFAST_SKIP_CMAKE_INSTALL=1
  export MLXFAST_SKIP_METAL_TOOLCHAIN_INSTALL=1
  echo "e12: HEAD $(git rev-parse HEAD)"
  echo "e12: build start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ./setup.sh
  status=$?
  echo "e12: build end $(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
  for bin in .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker; do
    if [[ -x "${bin}" ]]; then
      echo "e12: sha256 ${bin} $(shasum -a 256 "${bin}" | awk '{print $1}')"
      echo "e12: mtime ${bin} $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%SZ' "${bin}")"
    else
      echo "e12: MISSING ${bin}"
      status=1
    fi
  done
  exit "${status}"
  ;;
iterate)
  tokens="${2:?tokens}"
  trace="${3:?trace 0|1}"
  tag="${4:?tag}"
  capture="${PWD}/research/capture-e12-${tag}"
  score="research/score-e12-${tag}.json"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLX_QWEN_MTP_TRACE="${trace}"
  export MLXFAST_SWIFT_BIN="research/capture-cli.sh"
  export MLXFAST_CAPTURE_REAL_BIN="${PWD}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${capture}"
  export MLXFAST_SCORE_PATH="${score}"
  rm -rf "${capture}" "${score}"
  echo "e12: HEAD $(git rev-parse HEAD)"
  echo "e12: worker sha256 $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "e12: cli sha256 $(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "e12: tag=${tag} tokens=${tokens} trace=${trace} depth=${MLXFAST_QWEN_MTP_DEPTH:-8(default)}"
  echo "e12: gpu_temp_before=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  research/await-lock-then-run.sh 420 ./benchmark-qwen-mtp.sh --local-iterate
  status=$?
  echo "e12: lock_wrapper_exit=${status} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "e12: gpu_temp_after=$(gpu_temp)"
  for f in "${score}" "${capture}"/*.json; do
    if [[ -s "${f}" ]]; then
      echo "e12: === ${f}"
      jq -S 'del(.row_ledger, .block_request_seconds, .effective_draft_lengths, .tokens, .reference_rows)' "${f}"
    else
      echo "e12: === ${f} EMPTY-OR-MISSING"
    fi
  done
  exit "${status}"
  ;;
*)
  echo "usage: research/e12-run.sh build | iterate TOKENS TRACE TAG" >&2
  exit 2
  ;;
esac
