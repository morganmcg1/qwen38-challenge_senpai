#!/usr/bin/env bash
# E87 draw 2 pre-submit chain: arm C + section 8, no Q-row rider.
#
#   usage: research/e87_draw2_presubmit.sh PREFIX
#
# Runs the standing chain in one job so the whole gate is one durable record.
# `swift test` is expected to report the standing floor of named organizer
# failures, so its exit code is recorded rather than treated as fatal; the name
# set is judged afterwards from the captured log.
#
# The worker rebuild carries the section 8 witness `qwen_mtp_probe_sort` in
# addition to the four standing witnesses. Without it a stale worker would pass
# every later step while running the mbsort chain this draw is meant to replace.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e87_draw2_presubmit.sh PREFIX}"
base_sha="770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
out="research/out/${prefix}-presubmit"
mkdir -p "${out}"

log() { echo "=== $* ===" | tee -a "${out}/chain.log"; }
rc_of() { echo "$1=$2" >> "${out}/status.txt"; }

: > "${out}/chain.log"
: > "${out}/status.txt"

log "step 1 rebuild worker with the section 8 witness"
senpai/rebuild-and-assert-worker.sh \
  --require qwen_mtp_probe_sort \
  --require qwen35_dual_rms_norm_concat_bf16_v1 \
  --forbid qwen35_dual_rms_norm_bf16_v1 \
  --require-symbol snapshotScheduleSignal \
  >> "${out}/chain.log" 2>&1
rc=$?; rc_of rebuild "${rc}"
if [[ "${rc}" -ne 0 ]]; then
  log "rebuild failed rc=${rc}; stopping before any measurement"
  exit "${rc}"
fi

log "step 2 twin audit"
python3 research/twin_audit.py >> "${out}/chain.log" 2>&1
rc_of twin_audit $?

log "step 3 editable budget against ${base_sha}"
senpai/check-editable-budget.sh "${base_sha}" >> "${out}/chain.log" 2>&1
rc_of budget $?

log "step 4 assignment scope"
senpai/validate-assignment-scope.sh "${base_sha}" \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift \
  Sources/MLXFastModel >> "${out}/chain.log" 2>&1
rc_of scope $?

log "step 5 ranked score boundary"
senpai/verify-ranked-score-boundary.sh >> "${out}/chain.log" 2>&1
rc_of boundary $?

log "step 6 swift test (standing floor expected, exit code recorded not gated)"
swift test --force-resolved-versions > "${out}/swift-test.log" 2>&1
rc_of swift_test $?

# The arm names the probe fraction the shipped source builds, and the gate
# rejects `declared` because the derived index is now unconditional.
log "step 7 local-submit at 512 tokens, derived index at probe fraction 0.25"
E87_TOKENS=512 research/e87_submit_gate.sh "${prefix}" derived25 \
  >> "${out}/chain.log" 2>&1
rc_of submit_gate $?

# The bare exit code is not the gate. The gate is the failing name set and the
# issue count, so record the decomposition next to the log rather than leaving
# the next reader to re-derive it.
log "swift test decomposition"
{
  echo "total_issues=$(grep -cE 'recorded an issue' "${out}/swift-test.log")"
  grep -oE 'Test [a-zA-Z0-9_]+\(\) recorded an issue' "${out}/swift-test.log" \
    | grep -oE 'Test [a-zA-Z0-9_]+' | sed 's/^Test //' | sort | uniq -c \
    | awk '{print $2"="$1}'
} | tee -a "${out}/chain.log" > "${out}/swift-test-names.txt"

log "chain finished"
cat "${out}/status.txt"
exit 0
