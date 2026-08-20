#!/usr/bin/env bash
# Prove that both E85 fused paths really execute, and price them on the GPU.
#
#   usage: research/e85_proof_session.sh [TOKENS]
#
# Four census legs from ONE census binary, so no build difference can explain a
# difference between them:
#
#   d1-base  d1-ab   forced draft width 1
#   d5-base  d5-ab   forced draft width 5
#
# The pair at each width is the execution proof. If the guards in `preFcConcat`
# and `draftTokenIDWithDeclaredRerank` really pass, then the `ab` legs lose
# these five kernels from the `draft_head` phase:
#
#   gather_frontuint32_int32_int_1      arm (a) packed embedding row
#   gather_frontbfloat16_int32_int_2    arm (a) scales and biases
#   affine_dequantize_bfloat16_t_gs_64_b_4   arm (a) dequantized row
#   gather_frontuint32_uint32_int_1     arm (b) gathered packed rows
#   gather_frontbfloat16_uint32_int_2   arm (b) gathered scales and biases
#
# and gain `qwen35_embed_dual_rms_norm_concat_bf16_v1` plus a gather_qmm
# kernel. A guard that fails silently leaves the five kernels in place, which
# is exactly the failure mode a null timing result could otherwise hide.
#
# The two widths give the per-draft slope of every count, which is immune to
# the phase-attribution smear MLX's asynchronous encoding creates.
#
# These are census builds. Their WALL timings are invalid and are never
# reported. GPU time is read from Metal's own command-buffer timestamps, so the
# per-phase GPU nanoseconds remain meaningful even though wall time does not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-128}"

# Metal command-buffer timestamps, so the census also prices the head phase.
export MLX_E80_GPU_TIME=1
export MLX_E80_SNAPSHOT_ROUNDS=8

status=0
for width in 1 5; do
  for arm in base ab; do
    tag="e85-proof-d${width}-${arm}"
    echo "=== ${tag} ==="
    research/e85_census_leg.sh "${tag}" "${width}" "${tokens}" "${arm}"
    leg_status=$?
    if ((leg_status != 0)); then
      echo "e85_proof_session.sh: ${tag} exited ${leg_status}" >&2
      status="${leg_status}"
      break
    fi
    matched="$(python3 -c "
import json
print(json.load(open('research/out/${tag}/score.json'))['metrics']['all_tokens_matched'])
")"
    echo "${tag}: all_tokens_matched=${matched}"
    if [[ "${matched}" != "True" ]]; then
      echo "e85_proof_session.sh: ${tag} broke token matching" >&2
      status=3
      break
    fi
  done
  ((status == 0)) || break
done

exit "${status}"
