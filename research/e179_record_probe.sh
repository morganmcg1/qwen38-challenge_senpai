#!/usr/bin/env bash
# E179 step 1: price the launch-config cache with the host record probe.
#
# `Qwen35KernelConfigCache` reads its arm once per process, so each arm needs
# its own process. The arms run ABBA in one session so host drift cancels to
# first order.
#
# The probe loads no model and holds no model process. It does touch the GPU on
# its eval phase, so run it with nothing else resident.
#
# Writes research/out/e179-record-probe-<tag>.json, one file per leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MLXFAST_RUN_MLX_RUNTIME_TESTS=1

for leg in on1 off1 off2 on2; do
  case "$leg" in
    on*) arm=on ;;
    off*) arm=off ;;
  esac
  echo "== leg $leg (arm=$arm) =="
  # The default debug configuration, as in the E174 record probe, so the two
  # probes are directly comparable. Unoptimised Swift can only overstate the
  # Swift-side share the mechanism removes, so the measured saving is an UPPER
  # BOUND on what the release worker can recover. A ceiling below the minimum
  # useful effect therefore settles the question; a ceiling above it only
  # licenses the end-to-end release screen.
  MLX_E179_CFG_CACHE_ARM="$arm" E179_PROBE_TAG="$leg" \
    swift test --force-resolved-versions \
    --filter kernelConfigCacheRecordCostSplitsTheRecordBracket \
    2>&1 | tail -3
done
