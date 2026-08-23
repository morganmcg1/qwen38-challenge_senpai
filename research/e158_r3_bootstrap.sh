#!/usr/bin/env bash
# Bring a fresh role checkout back to a runnable state without re-downloading
# the 14 GB target checkpoint or the two head artifacts.
#
# The launcher created a new role directory for this generation. The transformed
# target checkpoint, the organizer-pinned head and the declared head already
# exist in the previous generation's role directory on the same APFS volume, so
# `cp -c` clones them at zero extra disk cost. Everything else is rebuilt from
# source in this checkout.
#
# Research-only. Not a submitted path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREV_ROLE="${E158_PREV_ROLE:-/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd}"
CACHE_ROOT="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1"
PREV_CACHE="${PREV_ROLE}/home/.cache/mlxfast/qwen3.8-27b-mtp-v1"

step() { printf '\n=== %s === %s\n' "$1" "$(date -u +%H:%M:%SZ)"; }

step "target checkpoint"
if [[ -f weights/config.json ]]; then
  echo "weights/ already present"
else
  [[ -f "${PREV_ROLE}/workspace/target/weights/config.json" ]] || {
    echo "FATAL: no previous transformed checkpoint at ${PREV_ROLE}" >&2
    exit 1
  }
  cp -Rc "${PREV_ROLE}/workspace/target/weights/." weights/ || exit 1
  ls -1 weights | head
fi

step "head artifacts"
mkdir -p "${CACHE_ROOT}"
for d in mtp-head mtp-head-declared mtp-head-declared-run; do
  if [[ -e "${CACHE_ROOT}/${d}" ]]; then
    echo "${d} already present"
  elif [[ -e "${PREV_CACHE}/${d}" ]]; then
    cp -Rc "${PREV_CACHE}/${d}" "${CACHE_ROOT}/${d}" || exit 1
    echo "${d} cloned"
  else
    echo "WARNING: ${d} missing in previous role cache" >&2
  fi
done
du -sh "${CACHE_ROOT}"/* 2>/dev/null

step "reference_weights"
# benchmark.sh hashes the reference repo revision and its config.json into the
# transform-source digest. With reference_weights/ empty the digest cannot
# match the stamp inside a cloned weights/ tree, so the harness decides the
# cache is stale and tries to redo the ~15 GB transform. These two links make
# the digest reproducible; they cost no disk.
hf_snapshot="${PREV_ROLE}/workspace/target/reference_weights/Qwen3.8-27B-4bit"
if [[ -e reference_weights/Qwen3.8-27B-4bit ]]; then
  echo "reference_weights already linked"
else
  model_dir="${HOME}/.cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit"
  if [[ ! -d "${model_dir}" ]]; then
    prev_model="$(cd "$(readlink "${hf_snapshot}")/../.." && pwd -P)"
    mkdir -p "$(dirname "${model_dir}")"
    cp -Rc "${prev_model}" "$(dirname "${model_dir}")/" || exit 1
  fi
  ln -sfn "${model_dir}/snapshots/$(basename "$(readlink "${hf_snapshot}")")" \
    reference_weights/Qwen3.8-27B-4bit
  ln -sfn "${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head" \
    reference_weights/Qwen3.6-27B-MTP-4bit
  echo "reference_weights linked"
fi

step "head digests"
python3 research/e158_head_census.py --digest-only 2>/dev/null || true

step "swift build"
# The rebuild guard refuses to run without an assertion. `Qwen35IslandArm` is
# the island selector this branch reads at warm-up. `installExactQKVRows` is
# fully inlined by the release compiler and leaves no symbol of its own, so it
# cannot be used as a witness.
senpai/rebuild-and-assert-worker.sh --require-symbol Qwen35IslandArm || exit 1

step "mlx.metallib"
# `swift build` does not produce the metallib, and `benchmark-qwen-mtp.sh`
# refuses to start without it. Reuse the previous role's copy only when this
# tree computes the same vendored-source fingerprint; otherwise compile it.
metallib=.build-worker/release/mlx.metallib
want="$(tools/build-mlx-metallib.sh --print-fingerprint | tail -1)"
prev="${PREV_ROLE}/workspace/target/${metallib}"
if [[ -f "${metallib}" ]] \
  && grep -qF "${want}" "${metallib}.fingerprint" 2>/dev/null; then
  echo "metallib already present and current"
elif [[ -f "${prev}" ]] \
  && grep -qF "${want}" "${prev}.fingerprint" 2>/dev/null; then
  cp -c "${prev}" "${prev}.fingerprint" .build-worker/release/ || exit 1
  echo "metallib cloned from ${PREV_ROLE} at fingerprint ${want}"
else
  tools/build-mlx-metallib.sh || exit 1
fi
grep -qF "${want}" "${metallib}.fingerprint" || {
  echo "metallib fingerprint does not match this tree" >&2
  exit 1
}

step "done"
ls -l .build/release/mlxfast-swift .build-worker/release/mlxfast-runtime-worker
