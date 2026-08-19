#!/usr/bin/env bash
# Research-only (qwen38-r1-e55, revision e55-r2): run the ONE documented
# pre-submit gate the advisor asked for, on the composed candidate.
#
#   research/e55_local_submit.sh
#
# The revision asks for `./benchmark-qwen-mtp.sh --local-submit` and nothing
# else, so this script adds no arm, no counterfactual and no new timing
# protocol. It only does what must happen before that command can be trusted:
#
#   1. rebuild BOTH build roots from the candidate source. The E55 timing
#      session left the worker holding the base2 arm (M=9 NA=3), and
#      benchmark-qwen-mtp.sh never rebuilds -- it consumes .build/release and
#      .build-worker/release as prerequisites. Running the gate against a stale
#      worker would gate the wrong binary.
#   2. prove BY CONTENT that the worker embeds THIS arm's JIT dispatch literal
#      (research/e55_binary_assert.sh). mtimes cannot witness this; see that
#      script's header.
#   3. run the twin audit, so a divergence between the readable header and the
#      runtime-effective JIT string cannot reach the gate.
#   4. run the documented command with its documented defaults.
#
# The wrapper's own cool gate is left exactly as shipped: --local-submit runs
# ./benchmark.sh --local-cool-gate-only before each model-resident leg, so this
# leg IS real-gate qualified, unlike the ungated ABBA timing session. That is a
# property of the wrapper, not something this script asserts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== e55-local-submit: start $(stamp) ==="
echo "head_commit=$(git rev-parse HEAD)"
echo "dirty=$(git status --porcelain -- Sources/ Vendor/ mtp-head.manifest.json mtp-head/ | wc -l | tr -d ' ')"

echo "=== step 0/4: GPU occupancy gate $(stamp) ==="
python3 research/e49_gpu_gate.py || { echo "e55-local-submit: GPU is not idle" >&2; exit 1; }

echo "=== step 1/4: rebuild both build roots $(stamp) ==="
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift \
  || { echo "e55-local-submit: mlxfast-swift build failed" >&2; exit 1; }
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker \
  || { echo "e55-local-submit: mlxfast-runtime-worker build failed" >&2; exit 1; }
echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"

echo "=== step 2/4: binary content assert $(stamp) ==="
research/e55_binary_assert.sh || { echo "e55-local-submit: binary assert failed" >&2; exit 1; }

echo "=== step 3/4: twin audit $(stamp) ==="
python3 research/twin_audit.py
echo "twin_audit_rc=$?"

echo "=== step 4/4: ./benchmark-qwen-mtp.sh --local-submit $(stamp) ==="
./benchmark-qwen-mtp.sh --local-submit
rc=$?
echo "local_submit_rc=${rc}"
echo "=== e55-local-submit: score payload $(stamp) ==="
if [[ -s score.json ]]; then
  cp score.json research/e55-local-submit-score.json
  cat score.json
else
  echo "e55-local-submit: no score.json was written" >&2
fi
echo "=== e55-local-submit: end $(stamp) rc=${rc} ==="
exit "${rc}"
