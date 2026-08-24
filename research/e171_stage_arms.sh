#!/usr/bin/env bash
# E171: build and stage the two arm workers for the instrumentation-strip ABBA.
#
#   usage: research/e171_stage_arms.sh S0_COMMIT S1_COMMIT
#
# S0 = base plus the E165 default flip, research instrumentation still present.
# S1 = S0 plus the strip: Qwen35.swift back to organizer-main bytes and the two
#      session-side consumers of the deleted counters removed.
#
# The arm is a COMPILE-TIME property, so one binary cannot carry both. The
# staged workers are selected per leg through MLXFAST_RUNTIME_WORKER_EXECUTABLE
# (main.swift:2252), which is the same mechanism research/e162_abba.sh used, so
# an ABBA schedule costs two builds rather than four.
#
# The CLI is built ONCE, from S1, and both arms run under it. Under
# MLXFAST_USE_RUNTIME_WORKER=1 the CLI is the trusted parent: it drives the
# protocol and owns the clock, and the worker holds the model and runs the
# session. Holding the parent fixed keeps the measured quantity attributable to
# the worker.
#
# WITNESS. `qwen35XSumsSidecarHits` is one of the five globals the strip
# deletes. It must appear in the S0 symbol table and be absent from S1.
# `prefetchHeadStep` is the positive control: it is present in both arms, so a
# zero count for the forbidden symbol means "absent", not "nm found nothing".
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

s0="${1:?usage: e171_stage_arms.sh S0_COMMIT S1_COMMIT}"
s1="${2:?usage: e171_stage_arms.sh S0_COMMIT S1_COMMIT}"

root="${MLXFAST_E171_WORKERS:-$(cd ../.. && pwd)/e171-workers}"
surface=(Sources Vendor)

if [[ -n "$(git status --porcelain -- "${surface[@]}")" ]]; then
  echo "e171_stage_arms: the scored surface is dirty; commit first" >&2
  exit 1
fi

build_worker() {
  CLANG_MODULE_CACHE_PATH="$PWD/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker
}

stage() {
  local label="$1" dir="${root}/$1"
  mkdir -p "${dir}"
  cp .build-worker/release/mlxfast-runtime-worker "${dir}/mlxfast-runtime-worker"
  cp .build-worker/release/mlx.metallib "${dir}/mlx.metallib"
  echo "staged ${label} worker_sha256=$(shasum -a 256 "${dir}/mlxfast-runtime-worker" | cut -d' ' -f1)"
  echo "staged ${label} metallib_sha256=$(shasum -a 256 "${dir}/mlx.metallib" | cut -d' ' -f1)"
}

echo "== S1 ${s1}: worker =="
git checkout "${s1}" -- "${surface[@]}" || exit 1
build_worker || exit 1
stage s1

echo "== S1 ${s1}: CLI (one parent for both arms) =="
CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1

echo "== S0 ${s0}: worker =="
git checkout "${s0}" -- "${surface[@]}" || exit 1
build_worker || exit 1
stage s0

echo "== restore S1 working tree =="
git checkout "${s1}" -- "${surface[@]}" || exit 1
if [[ -n "$(git status --porcelain -- "${surface[@]}")" ]]; then
  echo "e171_stage_arms: the surface did not return to ${s1}" >&2
  exit 1
fi

echo
echo "== arm witnesses =="
status=0
for label in s0 s1; do
  w="${root}/${label}/mlxfast-runtime-worker"
  strip_sym="$(nm -a "${w}" 2>/dev/null | grep -c -F 'qwen35XSumsSidecarHits')"
  ctrl_sym="$(nm -a "${w}" 2>/dev/null | grep -c -F 'prefetchHeadStep')"
  echo "${label} qwen35XSumsSidecarHits=${strip_sym} prefetchHeadStep=${ctrl_sym}"
  [[ "${ctrl_sym}" -ge 1 ]] || { echo "FAIL ${label}: positive control absent"; status=1; }
  if [[ "${label}" == "s0" ]]; then
    [[ "${strip_sym}" -ge 1 ]] || { echo "FAIL s0: instrumentation symbol absent"; status=1; }
  else
    [[ "${strip_sym}" -eq 0 ]] || { echo "FAIL s1: instrumentation symbol still present"; status=1; }
  fi
done
[[ "$(shasum -a 256 "${root}/s0/mlxfast-runtime-worker" | cut -d' ' -f1)" \
   != "$(shasum -a 256 "${root}/s1/mlxfast-runtime-worker" | cut -d' ' -f1)" ]] \
  || { echo "FAIL: the two staged workers are byte-identical"; status=1; }

echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
[[ "${status}" -eq 0 ]] && echo "e171_stage_arms: PASS" || echo "e171_stage_arms: FAIL"
exit "${status}"
