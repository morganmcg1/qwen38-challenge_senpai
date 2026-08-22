#!/usr/bin/env bash
# Pre-submit occupancy-cliff gate.
#
# Compiles every scored Metal entry point for `applegpu_g16s` and
# `applegpu_g17s` from a base git ref and from the candidate, then fails when
# the candidate loses a resident simdgroup on `applegpu_g17s`, the ranked
# architecture. E121 raised the wide-QMV entry point from 101 registers to 102,
# which costs one of 39 resident simdgroups at EVERY dispatch of that kernel,
# and the submission that carried it scored below the base. This gate is the
# check that would have stopped it.
#
# The gate needs no GPU, loads no model and runs no benchmark. It runs the real
# AGX backend through `xcrun metal-tt`, so it is a compile-time census.
#
# Rule 89: every simdgroup number this gate reports is DERIVED from the
# register count through `floor(BUDGET / registers)`. It is a model output, not
# a measurement. Registers, spill bytes and ISA text sizes are measurements.
#
# Usage:
#   senpai/entry-point-cliff-census.sh --base <ref> [--candidate <ref>]
#                                      [--json <path>]
#
# `--candidate` defaults to the working tree, which is what a pre-submit check
# wants. Exit status 0 means no scored entry point lost residency; 1 means at
# least one did; 2 means the gate could not run.

set -uo pipefail

BASE=""
CANDIDATE=""
JSON=""

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 ;;
    --candidate) CANDIDATE="${2:-}"; shift 2 ;;
    --json) JSON="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "entry-point-cliff-census: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$BASE" ]; then
  echo "entry-point-cliff-census: --base <ref> is required" >&2
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

if ! command -v xcrun >/dev/null 2>&1; then
  echo "entry-point-cliff-census: xcrun is required" >&2
  exit 2
fi

ARGS=(--base "$BASE")
[ -n "$CANDIDATE" ] && ARGS+=(--candidate "$CANDIDATE")
[ -n "$JSON" ] && ARGS+=(--json "$JSON")

python3 research/e131_cliff_gate.py "${ARGS[@]}"
STATUS=$?
if [ "$STATUS" -gt 1 ]; then
  echo "entry-point-cliff-census: the census itself failed to run" >&2
  exit 2
fi
exit "$STATUS"
