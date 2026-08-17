#!/bin/bash
# e12 seed-prefill charge: run_job entry points (argv only, no env plumbing).
#
#   research/e12-run.sh build
#   research/e12-run.sh iterate TOKENS TRACE TAG
#
# `iterate` measures one matched serial/MTP pair through the unmodified
# benchmark-qwen-mtp.sh wrapper, behind research/await-lock-then-run.sh, and keeps
# the per-arm payloads via research/e12-swift-shim.sh. TRACE=1 sets
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
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLX_QWEN_MTP_TRACE="${trace}"
  export MLXFAST_SWIFT_BIN="research/e12-swift-shim.sh"
  export MLXFAST_E12_TAG="${tag}"
  export MLXFAST_SCORE_PATH="research/score-e12-${tag}-summary.json"
  rm -f "research/score-e12-${tag}-depth"*.json "research/score-e12-${tag}-summary.json"
  echo "e12: HEAD $(git rev-parse HEAD)"
  echo "e12: worker sha256 $(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "e12: cli sha256 $(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "e12: tag=${tag} tokens=${tokens} trace=${trace} depth=${MLXFAST_QWEN_MTP_DEPTH:-8(default)}"
  echo "e12: gpu_temp_before=$(gpu_temp) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  research/await-lock-then-run.sh 420 ./benchmark-qwen-mtp.sh --local-iterate
  status=$?
  echo "e12: lock_wrapper_exit=${status} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "e12: gpu_temp_after=$(gpu_temp)"
  for f in "research/score-e12-${tag}-depth"*.json "research/score-e12-${tag}-summary.json"; do
    if [[ -s "${f}" ]]; then
      echo "e12: === ${f}"
      jq -S 'del(.row_ledger, .block_request_seconds, .effective_draft_lengths)' "${f}"
    else
      echo "e12: === ${f} MISSING"
    fi
  done
  exit "${status}"
  ;;
*)
  echo "usage: research/e12-run.sh build | iterate TOKENS TRACE TAG" >&2
  exit 2
  ;;
esac
