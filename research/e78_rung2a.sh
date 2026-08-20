#!/usr/bin/env bash
# E78 rung 2a: absolute per-cell time at both group counts, every scored shape.
#
#   research/e78_rung2a.sh [--reps N]
#
# The advisor asked for a per-family in-situ contrast. That contrast cannot
# exist in one binary: `IPG` is a compile-time template parameter, so a single
# build runs each width at exactly one group count. Bridging two builds is a
# cross-session contrast, and E74 measured cross-session per-family ratios
# spreading 0.911 to 1.031, which is larger than the effect under test.
#
# E73's cell harness is the instrument that does answer the question. It
# compiles every `(M, IPG)` arm into ONE process, walks the scored shapes
# shape-major, and times every arm on a shape back to back in a palindrome
# order, so each cell is measured against its alternative under the same
# thermal state and the same allocation. The numbers are `isolated`, not
# in situ: they carry no surrounding round and no cache state from the model.
#
# Widths: 5, 6 and 9 are the three the two tables disagree on. 4 is added
# because it carries 14.2 % of ranked time and sits just below the operating
# point of the two prompts that set the published score; both tables ship
# IPG 4 there, so the contrast is against the only other legal partition.
#
# n = 98336 `head.compact_draft_vocab` is deliberately NOT measured. It is the
# draft head's 2-bit readout at M = 1, so it never reaches the `bits == 4`
# affine gate this dispatch table lives behind, and `xgroup_census.py` prices
# it at 0 calls per verify round. A 4-bit cell at that n would be fiction.
#
# The real 40 C cool gate is skipped because it provably aborts on this host:
# a probe from 42.9 C asymptoted at 40.1 C and aborted after 340 s against a
# 40 C target. The harness counterbalances arm position within each shape and
# records entry and exit temperature per shape instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

reps="31"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --reps) reps="$2"; shift 2 ;;
    *) echo "e78_rung2a: unknown argument $1" >&2; exit 2 ;;
  esac
done

arms="m4_ipg2,m4_ipg4,m5_ipg3,m5_ipg5,m6_ipg3,m6_ipg6,m9_ipg3,m9_ipg5"
artifacts="research/e78-artifacts"
mkdir -p "${artifacts}"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-$(command -v macmon)}"
export E73_WANDB_GROUP="${E73_WANDB_GROUP:-e78-width-dependent-inner-group-count}"

research/e73_rung1.sh \
  --arms "${arms}" \
  --reps "${reps}" \
  --tag "-e78-rung2a" \
  --log "${artifacts}/rung2a.log" \
  --skip-gate
status=$?
((status == 0)) || { echo "e78_rung2a: cell harness exited ${status}" >&2; exit "${status}"; }

cp research/e73-artifacts/rung1-e78-rung2a.json "${artifacts}/rung2a-cells-raw.json"
python3 research/e78_cell_table.py \
  --cells "${artifacts}/rung2a-cells-raw.json" \
  --config /tmp/e73-build/config-e78-rung2a.json \
  --out "${artifacts}/rung2a-cells.json" \
  --markdown "${artifacts}/rung2a-cells.md"
