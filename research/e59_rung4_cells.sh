#!/usr/bin/env bash
# E59 rung 4 stage A: the whole-table M=5 cell measurement, then the extreme
# command-buffer dose.
#
#   research/e59_rung4_cells.sh
#
# Stage A answers the advisor's primary question directly. Rung 3 already
# measured the same mechanism, but it measured it in ISOLATED builds, where
# `t55`'s cell carries a +38 register entry-point dose that the shipped table
# does not pay. The whole-table census says `t55` moves the entry point by only
# +2 and `m5_rbx` by 0, so this session measures the transfer-relevant form.
#
# Six legs, palindrome, matching the form askeladd used on E61 rung 1:
#
#   shipped  t55  m5_rbx  m5_rbx  t55  shipped
#
# A palindrome cancels monotone drift to first order for every arm at once,
# which an ABBA pair only does for one pair at a time. Every leg passes the
# real 40 C gate, so drift is small to begin with; the ordering is insurance,
# not a substitute for the gate.
#
# Widths 1..10 give eight untreated control widths. `t55` and `m5_rbx` change
# only `case 5`, so any movement at the other widths is the instrument, not the
# mechanism.
#
# Stage B is the extreme command-buffer dose. The moderate 8-vs-50 dose measured
# a null, which two stories explain: the export never reaches MLX, or commits
# are free because decode sits at 97 % of the DRAM roofline. A cap of 1 commits
# about every second operation and separates them.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

date -u "+e59_rung4_cells: === %Y-%m-%dT%H:%M:%SZ stage A, whole-table cells ==="
research/e59_session.sh \
  shipped:e59-r4c-a1 \
  t55:e59-r4c-a2 \
  m5_rbx:e59-r4c-a3 \
  m5_rbx:e59-r4c-a4 \
  t55:e59-r4c-a5 \
  shipped:e59-r4c-a6 \
  --widths 1,2,3,4,5,6,7,8,9,10 --reps 21 --inner 10
cells_rc=$?
echo "e59_rung4_cells: stage A rc=${cells_rc}"

date -u "+e59_rung4_cells: === %Y-%m-%dT%H:%M:%SZ stage B, extreme dose ==="
research/e59_geometry_proof.sh --skip-probes --extreme
dose_rc=$?
echo "e59_rung4_cells: stage B rc=${dose_rc}"

date -u "+e59_rung4_cells: === %Y-%m-%dT%H:%M:%SZ done ==="
# Stage B is an instrument check. It must not mask a stage A failure, and a
# failing dose verdict is itself a reportable result rather than a broken run.
exit "${cells_rc}"
