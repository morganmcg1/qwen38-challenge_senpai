#!/usr/bin/env bash

# Fail closed when the ranked workflow no longer proves that candidate edits
# can affect only the candidate MTP denominator.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOW="$ROOT/.github/workflows/qwen-mtp-ranked-benchmark.yml"

require() {
  local pattern="$1"
  local description="$2"
  if ! grep -Fq -- "$pattern" "$WORKFLOW"; then
    echo "ERROR: ranked score boundary is unproven: $description" >&2
    echo "Re-read $WORKFLOW and re-derive every score model before pricing work." >&2
    exit 1
  fi
}

[ -f "$WORKFLOW" ] || {
  echo "ERROR: ranked workflow not found: $WORKFLOW" >&2
  exit 2
}

require 'MLXFAST_QWEN_MTP_BASELINE_WS: /opt/bench-runner/baseline/' \
  'baseline workspace is not pinned outside the candidate tree'
require '--candidate "${MLXFAST_JOB_WS}"' \
  'ranked candidate does not resolve to the submitted workspace'
require '--baseline "${MLXFAST_QWEN_MTP_BASELINE_RESOLVED}"' \
  'ranked serial leg does not resolve to the pinned baseline workspace'
require 'baseline_serial_seconds_per_token_mean / .aggregate.candidate_mtp_seconds_per_token_mean' \
  'score is not pinned-baseline serial divided by candidate MTP time'

echo "PASS: ranked numerator is pinned baseline; candidate edits affect the MTP denominator only"
