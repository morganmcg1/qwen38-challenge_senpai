#!/usr/bin/env bash
# Rung 5f: the submission-grade local gate for Route B.
#
#   usage: research/e120_rung5f.sh [OUT_DIR]
#
# `MLXFAST_QWEN_MTP_HEAD_DIR` is unset on purpose. Campaign ledger rule 64:
# the declared proposal head is resolved runner-side before the sandbox, so
# pinning it here would test a path the ranked runner never takes.
#
# `MLX_E120_QMV_ARM` is left unset so the run takes the shipped default, which
# is the `sumtable` candidate arm. Setting it would prove nothing about what a
# submission actually executes.
set -euo pipefail

OUT_DIR="${1:-research/out/e120-5f-local-submit}"
mkdir -p "$OUT_DIR"

unset MLXFAST_QWEN_MTP_HEAD_DIR
unset MLX_E120_QMV_ARM

export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512
export MLX_E58_BUFFER_LIMIT_OPS=0

echo "== worker certificate before the timed run =="
senpai/rebuild-and-assert-worker.sh \
    --require "qwen35_custom_affine4_g64_qmv_wide_v1" \
    --require "qwen35_custom_affine4_g64_qmv_wide_sums_v1" \
    --require "qwen35_custom_affine4_g64_xsums_v1" \
    --require "MLX_E120_QMV_ARM" \
    --require "SHARE_SUMS" \
    --forbid "MLXFAST_QWEN_E120_QMV" \
    2>&1 | tee "$OUT_DIR/worker-assert.txt"

WORKER=".build-worker/release/mlxfast-runtime-worker"
shasum -a 256 "$WORKER" | tee "$OUT_DIR/worker-sha256.txt"

echo "== commit under test =="
git rev-parse HEAD | tee "$OUT_DIR/commit.txt"
git status --porcelain | tee "$OUT_DIR/worktree.txt"

echo "== local-submit, 512 decode tokens =="
./benchmark-qwen-mtp.sh --local-submit 2>&1 | tee "$OUT_DIR/local-submit.log"

echo "== worker certificate after the timed run =="
shasum -a 256 "$WORKER" | tee "$OUT_DIR/worker-sha256-after.txt"

diff "$OUT_DIR/worker-sha256.txt" "$OUT_DIR/worker-sha256-after.txt" \
    && echo "worker binary unchanged across the run"
