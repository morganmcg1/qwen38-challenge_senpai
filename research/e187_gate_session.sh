#!/usr/bin/env bash
# E187 Stage 1: run the whole offline gate session.
#
#   usage: research/e187_gate_session.sh [TOKENS] [ARM...]
#
# Each arm is an independent `mtp-verify --generate` chain on the public
# fixture. None of them is timed. The worker is rebuilt and asserted once
# before the first arm, because a stale worker cannot witness a Swift edit.
#
#   fp32     the pinned baseline
#   bf16     the treatment: every stored GDN recurrent state rounded to bf16
#   fp16     the fallback treatment, same store sites
#   fp32b    determinism control: the baseline repeated
#   ulpbf16  positive control: +1 bf16 ulp at ONE state cell, once
#   ulp1     positive control: +1 fp32 ulp at ONE state cell, once
#   cell2x   positive control: ONE state cell doubled, once
#   cellbig  positive control: ONE state cell set to 1e30, once
#
# A run keeps going after a failing arm, because the ulp arms are expected to
# fail their R5 self-consistency replay (the probe fires once, so the rebuilt
# frame does not carry it) and their golden is still written.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
shift || true
arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
  arms=(fp32 bf16 fp32b ulpbf16 ulp1)
fi

echo "session_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "tokens=${tokens}"
echo "arms=${arms[*]}"

senpai/rebuild-and-assert-worker.sh \
  --require 'MLX_E187_STATE_STORE' \
  --require 'qwen35-e187-state-store-arm-engaged='
echo "rebuild_rc=$?"

for arm in "${arms[@]}"; do
  mode="${arm%b}"
  echo
  echo "=== arm ${arm} (MLX_E187_STATE_STORE=${mode}) ==="
  research/e187_gate_arm.sh "${mode}" "${tokens}" "${arm}"
  echo "arm_${arm}_rc=$?"
done

echo
echo "session_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
