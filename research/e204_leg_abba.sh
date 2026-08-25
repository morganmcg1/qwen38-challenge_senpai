#!/usr/bin/env bash
# E204 stage 3: leg-level ABBA over the chain-prefetch arm, ONE build.
#
# Stage 2 measured `round_us` and found the ON arm 2532.5 +- 792.5 us faster.
# The first gated leg-level pair pointed the other way by 0.08%, which is
# inside single-run noise but can reverse the promotion decision, so this
# session replicates the comparison at LEG level with the real 40C gate.
#
# One armed build serves every leg, so no rebuild sits between the arms. The
# serial K=1 leg of each run is an internal control: at depth 0 no drafting
# round exists, so the chain prefetch never runs and the serial time measures
# only the session's host and thermal state.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
arms=(off on on off)
i=0
for arm in "${arms[@]}"; do
  i=$((i + 1))
  out="research/out/e204-abba-${i}-${arm}"
  rm -rf "${out}"
  mkdir -p "${out}"
  echo "=== leg ${i}: arm=${arm} ==="
  env DARKBLOOM_E204_CHAIN_ARM="${arm}" \
      MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
      MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}" \
      MLXFAST_SCORE_PATH="${PWD}/${out}/score.json" \
      ./benchmark-qwen-mtp.sh --local-submit 2>&1 | tee "${out}/run.log"
  {
    echo "arm=${arm}"
    echo "leg_index=${i}"
    echo "order=${arms[*]}"
  } > "${out}/meta.txt"
done
