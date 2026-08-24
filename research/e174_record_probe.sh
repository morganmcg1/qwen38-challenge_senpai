#!/usr/bin/env bash
# E174 step 2b: run the CPU-side host-kernel-record probe.
#
# The probe is an opt-in runtime test, so it needs MLXFAST_RUN_MLX_RUNTIME_TESTS
# in the environment. It loads no model and holds no model process, but it does
# touch the GPU on its eval phase, so run it with nothing else resident.
#
# Writes research/out/e174-record-probe.json.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_RUN_MLX_RUNTIME_TESTS=1
exec swift test --force-resolved-versions \
  --filter xsumsRecordConstructionCostSplitsTheFillBracket
