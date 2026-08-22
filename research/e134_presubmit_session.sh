#!/usr/bin/env bash
# E134 item 6 -- the full pre-submit chain, run at the commit that ships.
#
#   usage: research/e134_presubmit_session.sh [RUNS_DIR]
#
# The earlier pre-submit evidence was collected at `a9714610`, before the arm
# selector moved to run time. The source that ships is not that source, so the
# chain runs again here rather than being carried forward.
#
# ORDER MATTERS. HARNESS DEFECT 36 means `./benchmark-qwen-mtp.sh` never
# rebuilds `.build-worker/release/mlxfast-runtime-worker`, so the worker is
# rebuilt and asserted FIRST. FM7 then reads the COMMITTED tree, because Yukon
# packages commits and not the checkout, and a worktree grep cannot see a
# missing arm flip.
#
# The exactness legs keep the per-round phase trace, so they record
# `timing_valid=false`. That is correct: this chain is an exactness and
# gross-regression gate, not a timing session. The ABBA session owns timing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

runs_dir="${1:-runs-pb6-final}"
art="research/e134-artifacts"

echo "=== e134 pre-submit phase 0: worker build and selector assertion ==="
senpai/rebuild-and-assert-worker.sh --require 'MLX_E134_DEPTH_PRICE_ARM' || exit 2

echo "=== e134 pre-submit phase 1: FM7 packaged-snapshot assertions ==="
python3 research/e134_fm7_snapshot.py --json "${art}/fm7-snapshot.json" || exit 3

echo "=== e134 pre-submit phase 2: 512-token exactness legs ==="
E128_FORCE=1 E128_TOKENS=512 E128_DEPTH=8 E128_RUNS_DIR="${runs_dir}" \
  research/e128_session.sh \
    beagle_a benchfixture essays_montaigne medicine_hist
legs=$?

echo "=== e134 pre-submit phase 3: --local-submit ==="
./benchmark-qwen-mtp.sh --local-submit
submit=$?

echo "=== e134 pre-submit phase 4: collect the evidence ==="
python3 research/e134_item5_presubmit.py \
  --pb6 ".mlxfast-private/e128/${runs_dir}" \
  --ship .mlxfast-private/e128/runs-shipped \
  --json "${art}/item5-presubmit.json"
collect=$?

echo "e134_presubmit: legs=${legs} local_submit=${submit} collect=${collect}"
exit $(( legs != 0 || submit != 0 || collect != 0 ))
