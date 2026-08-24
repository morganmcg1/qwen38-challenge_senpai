#!/usr/bin/env bash
# E166 step 0b -- repeat the island prelude WITH the arm witness, then take one
# traced adaptive leg for the step 1 deliverable 3 histogram.
#
# The first step 0 session (`e166s0`) proved the accept ledger is invariant
# across the two island arms, but it set no `MLX_QWEN_MTP_TRACE_PATH`, and
# `Qwen35IslandArm.writeWitness` (Qwen35.swift:2925-2942) writes only to that
# path or to a stderr stream the `mtp-timed` parent discards. Without the
# witness the null is unwitnessed, so this session repeats it with the path
# set on every leg.
#
# The second session runs the SHIPPED adaptive schedule with the phase trace
# on. `costModelDepth` then emits one `sched=` line per round carrying the
# margin, the streak, the cap and the whole per-position EMA vector, which is
# every input the counterfactual price replay needs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

head_dir="${E166_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e166_step0b: no declared head at ${head_dir}" >&2; exit 1; }

echo "=== session 1: island prelude with witness, 128 tokens, ABBA ==="
E109_BLOCKS=1 E109_TOKENS=128 E109_DEPTH=8 \
E109_HEAD_DIR="${head_dir}" E109_GOLDEN="research/out/e166-golden-128.json" \
  research/e109_ab_session.sh e166s0w \
    "S=DARKBLOOM_QWEN_MTP_ISLAND_ARM=all,MLX_E159_FIXED_DRAFT_DEPTH=4,MLX_QWEN_MTP_TRACE_PATH=@LEG@/trace.txt" \
    "P=DARKBLOOM_QWEN_MTP_ISLAND_ARM=none,MLX_E159_FIXED_DRAFT_DEPTH=4,MLX_QWEN_MTP_TRACE_PATH=@LEG@/trace.txt"
status1=$?

echo
echo "=== session 2: traced adaptive leg, 512 tokens ==="
E109_BLOCKS=0 E109_TOKENS=512 E109_DEPTH=8 \
E109_HEAD_DIR="${head_dir}" E109_GOLDEN="research/out/e166-golden-512.json" \
  research/e109_ab_session.sh e166trace \
    "adapt=MLX_QWEN_MTP_TRACE=1,MLX_QWEN_MTP_TRACE_PATH=@LEG@/trace.txt" \
    "adaptc="
status2=$?

echo "e166_step0b: session1=${status1} session2=${status2}"
# A single-block session has no estimate block, so its report step exits
# non-zero by design. The legs are what this script delivers.
for leg in research/out/e166s0w/b*-* research/out/e166trace/b*-*; do
  [[ -s "${leg}/report.json" ]] || { echo "missing ${leg}/report.json" >&2; exit 1; }
done
echo "e166_step0b: every leg produced a report"
