#!/usr/bin/env bash
# E166 step 0 -- the island prelude, one binary, two arms, ABBA.
#
#   arm S   DARKBLOOM_QWEN_MTP_ISLAND_ARM=all    MLX_E159_FIXED_DRAFT_DEPTH=4
#   arm P   DARKBLOOM_QWEN_MTP_ISLAND_ARM=none   MLX_E159_FIXED_DRAFT_DEPTH=4
#
# The arm is a load-time selection inside ONE build (Qwen35.swift
# `Qwen35IslandArm.fromEnvironment`), so the two legs differ by one environment
# variable and by nothing else. Depth is pinned so the schedule cannot confound
# the acceptance contrast.
#
# It decides FINDING 372 mechanism (a) -- the islands change WHICH tokens are
# proposed, so the accept ledger moves -- against mechanism (b) -- the islands
# only change round runtime and the ledger is invariant.
#
# The base moved to 4b6eb4f5, so this rebuilds the CLI and the runtime worker
# first and records both digests before any timed leg.
#
# THERMAL. Ungated by construction (research/e109_ab_leg.sh), counterbalanced
# inside one session, entry and exit GPU temperature per leg,
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false kept
# verbatim. Step 0 is a ledger comparison, not a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-e166s0}"
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

echo "=== build at $(git rev-parse HEAD) ==="
swift build -c release --force-resolved-versions --product mlxfast-swift \
  > "${out_dir}/build-cli.log" 2>&1 || {
    echo "e166_step0: CLI build failed" >&2
    tail -30 "${out_dir}/build-cli.log" >&2
    exit 1; }
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker \
  > "${out_dir}/build-worker.log" 2>&1 || {
    echo "e166_step0: worker build failed" >&2
    tail -30 "${out_dir}/build-worker.log" >&2
    exit 1; }

{
  echo "base_sha=$(git rev-parse HEAD)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
} | tee "${out_dir}/digests.txt"

head_dir="${E166_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e166_step0: no declared head at ${head_dir}" >&2; exit 1; }

E109_BLOCKS="${E166_BLOCKS:-1}" \
E109_TOKENS="${E166_TOKENS:-128}" \
E109_DEPTH=8 \
E109_HEAD_DIR="${head_dir}" \
E109_GOLDEN="research/out/e166-golden-${E166_TOKENS:-128}.json" \
  research/e109_ab_session.sh "${tag}" \
    "S=DARKBLOOM_QWEN_MTP_ISLAND_ARM=all,MLX_E159_FIXED_DRAFT_DEPTH=4" \
    "P=DARKBLOOM_QWEN_MTP_ISLAND_ARM=none,MLX_E159_FIXED_DRAFT_DEPTH=4"
