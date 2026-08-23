#!/usr/bin/env bash
# E147 rung A positive control (Rule 101): break the pipelined schedule and
# prove the exactness leg reports it.
#
# WHAT IS BROKEN, AND WHY THAT IS THE RIGHT THING TO BREAK. The pipelined
# k-loop keeps exactly one barrier per iteration, at the top. That barrier
# carries two obligations: it publishes the tile staged by the previous
# iteration to every mma reader, and it proves that every reader of the half
# about to be overwritten has finished. The first obligation is the one the
# prologue depends on, because the prologue stages tile 0 outside the loop.
# Suppressing the barrier on the first iteration therefore leaves the k = 0 mma
# reading a threadgroup buffer whose stores are not yet visible across
# simdgroups, and nothing else about the kernel changes.
#
# A check that has never failed is not a check. This script builds that broken
# worker, runs one 512-token exactness leg against the same base-provenance
# reference rows the ABBA session used, and requires the leg to report a
# mismatch. It restores the tree and rebuilds the real candidate worker before
# it exits, whatever the outcome.
#
# The broken kernel is never committed and never times anything.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

header=Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h
twin=Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
worker=".build-worker/release/mlxfast-runtime-worker"
prompt_id="${E147_CONTROL_PROMPT:-beagle_a}"
out=".mlxfast-private/e147/runs/control/${prompt_id}"

CONTROL_NEEDLE='if (k > 0) { threadgroup_barrier(mem_flags::mem_threadgroup); }'

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e147_control: scored surface is dirty; refusing to start" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"

restore() {
  git checkout "${session_commit}" -- "${header}" "${twin}"
  echo "=== e147_control: rebuilding the real candidate worker ==="
  senpai/rebuild-and-assert-worker.sh \
    --require 'mma_op.mma(Xs + cur * Xs_tile, Ws + cur * Ws_tile);' \
    --forbid "${CONTROL_NEEDLE}"
}
trap restore EXIT

echo "=== e147_control: suppressing the first-iteration barrier ==="
python3 - "${header}" <<'PY' || exit 1
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = ("      for (int k = 0; k < K_eff; k += BK) {\n"
       "        threadgroup_barrier(mem_flags::mem_threadgroup);\n")
new = ("      for (int k = 0; k < K_eff; k += BK) {\n"
       "        if (k > 0) { threadgroup_barrier(mem_flags::mem_threadgroup); }\n")
count = text.count(old)
if count != 4:
    print(f"e147_control: expected 4 pipelined k-loops, found {count}",
          file=sys.stderr)
    raise SystemExit(1)
path.write_text(text.replace(old, new))
print("e147_control: suppressed the barrier in 4 k-loops")
PY

research/e147_port_twin.py "${header}" "${twin}" "${session_commit}" || exit 1

senpai/rebuild-and-assert-worker.sh --require "${CONTROL_NEEDLE}" || {
  echo "e147_control: the broken worker did not build" >&2; exit 3; }

echo "=== e147_control: one 512-token exactness leg ==="
E128_FORCE=1 \
E128_NO_TRACE=1 \
E128_TOKENS=512 \
E128_DEPTH=8 \
E128_ROOT=".mlxfast-private/e147" \
E128_GOLDENS_DIR="${E147_GOLDENS_DIR:-.mlxfast-private/e128/goldens}" \
E128_RUNS_DIR="runs/control" \
  research/e128_session.sh "${prompt_id}"
leg_status=$?

fired=1
matched="none"
divergences="none"
if [[ -s "${out}/report.json" ]]; then
  matched="$(jq -r '.all_tokens_matched' "${out}/report.json")"
  divergences="$(jq -r '.residual_divergence_count' "${out}/report.json")"
  if [[ "${matched}" == "true" && "${divergences}" == "0" ]]; then
    fired=0
  fi
fi

{
  echo "e147_control_prompt=${prompt_id}"
  echo "e147_control_leg_exit=${leg_status}"
  echo "e147_control_all_tokens_matched=${matched}"
  echo "e147_control_residual_divergence_count=${divergences}"
  echo "e147_rungA_positive_control_failed=$([[ ${fired} -eq 1 ]] && echo 1.0 || echo 0.0)"
  echo "e147_control_session_commit=${session_commit}"
} | tee "research/e147-rungA-control.txt"

if ((fired == 1)); then
  echo "e147_control: PASS, the broken schedule was caught"
  exit 0
fi
echo "e147_control: FAIL, the broken schedule passed the exactness leg" >&2
exit 1
