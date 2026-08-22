#!/usr/bin/env bash
# Rung 5f: the submission-grade local gate for Route B.
#
#   usage: research/e120_rung5f.sh [OUT_DIR]
#
# THE HEAD MUST BE THE DECLARED ONE, NOT THE PINNED ONE. `mtp-head.manifest.json`
# declares `hf:amal-david/qwen38-mtp-head-q2-q4-rerank-v1@ae628274`, and the
# ranked runner attaches that head to the CANDIDATE leg; the pinned head serves
# only the serial leg, which the runner builds itself and which never runs our
# code. `setup-qwen-mtp.sh` defaults `MLXFAST_QWEN_MTP_HEAD_DIR` to the pinned
# head, and the shipped index needs tensors that exist only in the declared one,
# so leaving the default in place tests a head the candidate leg never sees. A
# first attempt that unset the variable failed for exactly this reason.
#
# `MLX_E120_QMV_ARM` is left unset so the run takes the shipped default, which
# is the `sumtable` candidate arm. Setting it would prove nothing about what a
# submission actually executes.
set -euo pipefail

OUT_DIR="${1:-research/out/e120-5f-local-submit}"
mkdir -p "$OUT_DIR"

HEAD_DIR="${E120_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${HEAD_DIR}/config.json" ]]; then
    echo "e120_rung5f.sh: no declared MTP head at ${HEAD_DIR}" >&2
    exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${HEAD_DIR}"
unset MLX_E120_QMV_ARM

export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512
export MLX_E58_BUFFER_LIMIT_OPS=0

echo "== worker certificate before the timed run =="
senpai/rebuild-and-assert-worker.sh \
    --require "qwen35_custom_affine4_g64_qmv_wide_v2" \
    --require "qwen35_custom_affine4_g64_qmv_wide_sums_v2" \
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
