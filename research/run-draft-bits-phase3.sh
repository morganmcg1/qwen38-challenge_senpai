#!/usr/bin/env bash
# Research-only driver: E15 Phase 3 counterbalanced timing pairs.
#
#   research/run-draft-bits-phase3.sh TAG_PREFIX [TOKENS] [BASE_SHA] [ORDER] [POS_OFFSET]
#
# Phase 2 ran one control->candidate pair and the candidate started 7.96C hotter
# than the control (42.41C vs 50.37C), so its -0.727% carried an uncontrolled
# thermal term of unknown sign. Phase 3 removes that two ways at once.
#
# ORDER defaults to the ABBA sequence 4,3,3,4. Arm 4 then occupies positions 1
# and 4 and arm 3 occupies 2 and 3, so both arms have mean position 2.5 and any
# drift that is linear in position cancels exactly in the arm means. Pairing
# adjacent runs (1,2) and (4,3) additionally gives two independent same-direction
# pair deltas whose spread bounds the residual.
#
# POS_OFFSET lets the ABBA sequence be split across two supervised jobs while
# keeping one global position numbering: `... 4,3 0` then `... 3,4 2` labels
# positions 1,2 then 3,4, which is what the analyzer's slot accounting expects.
#
# Each arm is preceded by a bounded settle whose default target is 40.0C: the
# same numeric threshold as benchmark.sh's COOL_GATE_TEMP_C and the ranked
# runner's gate. When the settle reaches it, the driver also runs the real
# `./benchmark.sh --local-cool-gate-only` gate as a witness (it passes on its
# first poll at that point, so it cannot burn budget). The arms themselves still
# run with MLXFAST_LOCAL_COOL_GATE=0 for a budget reason, not a thermal one:
# benchmark-qwen-mtp.sh invokes the gate three times per arm and each invocation
# may wait up to COOL_GATE_MAX_WAIT_SECONDS=900, which no 30-minute job can
# contain for four arms.
set -euo pipefail

prefix="${1:?usage: run-draft-bits-phase3.sh TAG_PREFIX [TOKENS] [BASE_SHA] [ORDER] [POS_OFFSET]}"
tokens="${2:-256}"
base_sha="${3:-}"
order="${4:-4,3,3,4}"
pos_offset="${5:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export MLX_QWEN_MTP_TRACE="${MLX_QWEN_MTP_TRACE:-1}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-qwen38-r1-e15-draft-readout-3bit}"

settle_target_c="${MLXFAST_SETTLE_TARGET_C:-40.0}"
settle_max_s="${MLXFAST_SETTLE_MAX_S:-300}"
lock_wait_s="${MLXFAST_LOCK_WAIT_S:-300}"
export MLXFAST_SETTLE_TARGET_C="${settle_target_c}"

gpu_temp_now() {
  local macmon
  macmon="$(command -v macmon || true)"
  [[ -n "${macmon}" ]] || return 0
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

# Exports the settle outcome for run-draft-bits-arm.sh to seal into identity.txt.
# It cannot be written into the arm directory here: the arm rm -rf's its own out
# dir before it starts.
settle() {
  local pos="$1" waited=0 t min=""
  export MLXFAST_SETTLE_REACHED_C="" MLXFAST_SETTLE_MIN_C="" \
    MLXFAST_SETTLE_WAITED_S="" MLXFAST_COOL_GATE_STATUS="no_temperature_reader"
  while :; do
    t="$(gpu_temp_now)"
    if [[ -z "${t}" ]]; then
      echo "phase3-settle: pos=${pos} no macmon reading; proceeding ungated"
      return 0
    fi
    if [[ -z "${min}" ]] || awk -v a="${t}" -v b="${min}" 'BEGIN{exit !(a<b)}'; then
      min="${t}"
    fi
    export MLXFAST_SETTLE_REACHED_C="${t}" MLXFAST_SETTLE_MIN_C="${min}" \
      MLXFAST_SETTLE_WAITED_S="${waited}"
    if awk -v a="${t}" -v b="${settle_target_c}" 'BEGIN{exit !(a<=b)}'; then
      echo "phase3-settle: pos=${pos} reached=${t}C target=${settle_target_c}C min=${min}C waited=${waited}s"
      if MLXFAST_LOCAL_COOL_GATE=1 ./benchmark.sh --local-cool-gate-only; then
        export MLXFAST_COOL_GATE_STATUS="passed_real_gate"
      else
        export MLXFAST_COOL_GATE_STATUS="real_gate_failed_after_settle"
      fi
      echo "phase3-settle: pos=${pos} cool_gate=${MLXFAST_COOL_GATE_STATUS}"
      return 0
    fi
    if [[ "${waited}" -ge "${settle_max_s}" ]]; then
      echo "phase3-settle: pos=${pos} STALLED at=${t}C target=${settle_target_c}C min=${min}C waited=${waited}s"
      export MLXFAST_COOL_GATE_STATUS="stalled_above_${settle_target_c}C"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

# Build once up front so the per-arm build is a near no-op and does not reheat
# the package between the settle and the clock. run-draft-bits-arm.sh still runs
# its own build and its own MLX_QWEN_MTP_DRAFT_BITS strings tripwire per arm.
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker

echo "run-draft-bits-phase3: prefix=${prefix} tokens=${tokens} order=${order} pos_offset=${pos_offset}"
echo "run-draft-bits-phase3: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
echo "run-draft-bits-phase3: settle_target_c=${settle_target_c} settle_max_s=${settle_max_s}"

pos="${pos_offset}"
IFS=',' read -r -a arm_order <<<"${order}"
for bits in "${arm_order[@]}"; do
  pos=$((pos + 1))
  settle "${pos}"
  echo "=== run-draft-bits-phase3: arm pos=${pos} bits=${bits} tokens=${tokens} ==="
  research/await-lock-then-run.sh "${lock_wait_s}" \
    research/run-draft-bits-arm.sh "${bits}" "${prefix}-p${pos}-b${bits}" \
    "${tokens}" "${base_sha}"
done

echo "=== run-draft-bits-phase3: replicate summary ==="
pos="${pos_offset}"
for bits in "${arm_order[@]}"; do
  pos=$((pos + 1))
  python3 - "${repo_root}/.mlxfast-private/draft-bits/${prefix}-p${pos}-b${bits}" \
    "${pos}" "${bits}" <<'PY'
import json, os, sys
d, pos, bits = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(os.path.join(d, "amdahl.json")))
ident = {}
for line in open(os.path.join(d, "identity.txt")):
    for tok in line.replace("run-draft-bits-arm:", "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            ident[k] = v
for leg in ("serial_leg", "mtp_leg"):
    x = r.get(leg, {})
    print("phase3 pos=%s bits=%s %s matched=%s emitted=%s spt=%s "
          "temp_before=%s temp_after=%s" % (
              pos, bits, leg, x.get("all_tokens_matched"),
              x.get("emitted_token_total"),
              x.get("parent_measured_seconds_per_token"),
              ident.get("gpu_temp_c_before"), ident.get("gpu_temp_c_after")))
PY
done
