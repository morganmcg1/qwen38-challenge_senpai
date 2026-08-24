#!/usr/bin/env bash
# Research-only (qwen38-r1-e181-local-q-pair-jit-channel): build ONE arm tree
# and certify which mechanism is inside the built worker.
#
#   research/e181_build_arm.sh P|PQ|PQS
#
# The arm trees are separate directories, not one tree rebuilt between legs:
# an ABBA session must alternate two FIXED binaries, and a rebuild-in-place
# ladder cannot prove that the binary timed in leg 4 is the binary timed in
# leg 1.
#
# WITNESSES (advisor F1, PR 180). `shift_dst` is NOT a Q witness: organizer
# fp_quantized_nax carries it too. The discriminating needles are
#
#   qmm_t   mma_op.mma(Xs, Ws + cur * Ws_tile);   campaign 1, organizer 0
#   nax     qmm_t_nax_ws_halves                   campaign 4, organizer 0
#
# and MLX_QWEN_MTP_TRACE is the positive control that proves the string probe
# can see into this binary at all, so a zero count is evidence and not a
# broken probe.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: research/e181_build_arm.sh P|PQ|PQS}"
case "${arm}" in P|PQ|PQS) ;; *) echo "e181_build_arm: arm must be P, PQ or PQS" >&2; exit 2 ;; esac

worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1
tools/build-mlx-metallib.sh || exit 1

count_in_worker() { strings -a "${worker}" | grep -c -F -- "$1" || true; }

qmm_needle='mma_op.mma(Xs, Ws + cur * Ws_tile);'
nax_needle='qmm_t_nax_ws_halves'
control_needle='MLX_QWEN_MTP_TRACE'

qmm_count="$(count_in_worker "${qmm_needle}")"
nax_count="$(count_in_worker "${nax_needle}")"
control_count="$(count_in_worker "${control_needle}")"

case "${arm}" in
  P)  want_qmm=0 ;;
  PQ) want_qmm=1 ;;
  PQS) want_qmm=1 ;;
esac

status=0
if [[ "${control_count}" -eq 0 ]]; then
  echo "e181_build_arm: positive control '${control_needle}' absent; the string probe is broken" >&2
  status=1
fi
if [[ "${qmm_count}" -ne "${want_qmm}" ]]; then
  echo "e181_build_arm: arm ${arm} wants ${want_qmm} qmm_t pipelined needle, worker has ${qmm_count}" >&2
  status=1
fi
if [[ "${arm}" == "P" && "${nax_count}" -ne 0 ]]; then
  echo "e181_build_arm: arm P must carry no nax needle, worker has ${nax_count}" >&2
  status=1
fi
if [[ "${arm}" != "P" && "${nax_count}" -eq 0 ]]; then
  echo "e181_build_arm: arm ${arm} must carry the nax needle, worker has 0" >&2
  status=1
fi

{
  echo "arm=${arm}"
  echo "tree=${PWD}"
  echo "organizer_sha=$(git rev-parse upstream/main)"
  echo "assignment_head=$(git rev-parse HEAD)"
  echo "scored_diff_vs_organizer=$(git diff --numstat upstream/main -- Sources Vendor | awk '{a+=$1; d+=$2; n+=1} END {printf "%d files +%d -%d", n, a, d}')"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "metallib_sha256=$(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
  echo "metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "needle_qmm_pipelined=${qmm_count}"
  echo "needle_nax_ws_halves=${nax_count}"
  echo "needle_control_trace=${control_count}"
  echo "twin_audit<<EOF"
  python3 research/twin_audit.py 2>&1 | tail -5
  echo "EOF"
} | tee "research/e181-arm-${arm}-build.txt"

exit "${status}"
