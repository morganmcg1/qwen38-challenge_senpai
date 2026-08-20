#!/usr/bin/env bash
# Research-only (qwen38-r1-e55): provision this fresh workspace for timing.
#
# The 15:02Z launch retag left the checkout with no model cache and no build
# roots, so E55 must provision from zero before any arm can run:
#
#   1. ./setup.sh              toolchain and Swift dependency graph
#   2. ./setup-qwen-mtp.sh     transformed target checkpoint + ORGANIZER-PINNED head
#   3. research/fetch-declared-head.sh
#                              the head mtp-head.manifest.json DECLARES, which is
#                              the head the ranked CANDIDATE leg executes.
#                              setup-qwen-mtp.sh never reads the declaration, so
#                              without this step every timed candidate leg runs
#                              the wrong head and any constant fitted to it is
#                              fitted to a head no ranked leg runs.
#
# Step 3 stages `<dest>-run`, which is the default E42_HEAD_DIR.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

echo "=== e55_setup: step 1/3 ./setup.sh ==="
./setup.sh || { echo "e55_setup: setup.sh failed" >&2; exit 1; }

echo "=== e55_setup: step 2/3 ./setup-qwen-mtp.sh ==="
./setup-qwen-mtp.sh || { echo "e55_setup: setup-qwen-mtp.sh failed" >&2; exit 1; }

echo "=== e55_setup: step 3/3 research/fetch-declared-head.sh ==="
research/fetch-declared-head.sh \
  || { echo "e55_setup: fetch-declared-head.sh failed" >&2; exit 1; }

head_run="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
echo "=== e55_setup: provisioned ==="
echo "head_run=${head_run}"
shasum -a 256 "${head_run}/model.safetensors" | awk '{print "head_safetensors_sha256=" $1}'
echo "done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
