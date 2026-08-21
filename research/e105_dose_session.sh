#!/usr/bin/env bash
# E105 rung 1+2 -- one counterbalanced in-situ dispatch-dose session.
#
#   usage: research/e105_dose_session.sh [PREFIX]
#
# Runs the palindrome
#
#   A0  Z0@32  B1  C4  D4tiny  E4op  |  E4op  D4tiny  C4  B1  Z0@32  A0
#
# inside one thermal session, so monotone thermal drift cancels to first
# order across each arm pair. Every leg is `research/e105_dose_leg.sh`, which
# asserts the arm inside `.build-worker/release/mlxfast-runtime-worker` before
# it times anything, records entry and exit GPU temperature, and keeps
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`.
#
# WHY THESE DOSES. The decision needs F, the marginal cost of one dispatch
# boundary, in the regime where a fusion removes 80 to 96 dispatches per
# round. Dose 1 and dose 4 per layer are 64 and 256 extra dispatches per
# target forward, which brackets that regime instead of extrapolating into
# it from far outside. A dose-16 smoke leg at 32 tokens showed the response
# is large, so a small dose is enough and a large dose only risks leaving the
# linear regime.
#
# WHY THREE SHAPES. Removing a real dispatch removes both its GPU boundary
# and MLX's CPU cost for that graph node, and the two are not the same for
# every op. `op` is a plain MLX multiply, the analogue of the KV write's
# `slice_update`. `tiny` is a one-threadgroup custom Metal kernel, the
# analogue of a custom kernel with no width. `prework` is the same custom
# kernel at the live 400-threadgroup width of `qwen35_packed_gdn_prework`.
# `op` to `tiny` prices MLX's custom-kernel call path; `tiny` to `prework`
# prices dispatch width. 48 of the 80 removable dispatches are custom
# kernels and 32 are `slice_update`s, so both numbers are load-bearing.
#
# WHY A SECOND TOKEN COUNT AT DOSE 0. `--local-iterate` reports seconds per
# token over a leg that also processes a 512-token seed, so
#
#   spt(n) = P / n + D
#
# where P is the fixed seed and warmup cost and D is the true marginal decode
# cost per token. Every dosed leg runs at the same n, so P cancels out of the
# slope and F is unaffected. But P does NOT cancel out of the DENOMINATOR, and
# the e105-f1 promotion bar is a percentage of the local round. Two dose-0
# legs at n = 32 and n = 64 solve the two-point system for D exactly, which
# gives an honest decode-only local round to price the ceiling against. The
# dose is gated off during prefill, so it cannot contaminate P.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:-e105r12}"
tokens="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"
probe_tokens="${MLXFAST_E105_PROBE_TOKENS:-32}"

# leg-label dose shape tokens
legs=(
  "a1 0 prework ${tokens}"
  "z1 0 prework ${probe_tokens}"
  "b1 1 prework ${tokens}"
  "c1 4 prework ${tokens}"
  "d1 4 tiny ${tokens}"
  "e1 4 op ${tokens}"
  "e2 4 op ${tokens}"
  "d2 4 tiny ${tokens}"
  "c2 4 prework ${tokens}"
  "b2 1 prework ${tokens}"
  "z2 0 prework ${probe_tokens}"
  "a2 0 prework ${tokens}"
)

session_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "e105 dose session ${prefix} start=${session_start} tokens=${tokens}"
tags=()
failed=()

for spec in "${legs[@]}"; do
  read -r label dose shape leg_tokens <<<"${spec}"
  tag="${prefix}-${label}-d${dose}-${shape}-n${leg_tokens}"
  tags+=("${tag}")
  echo
  echo "=== leg ${label}: dose=${dose} shape=${shape} tokens=${leg_tokens} " \
       "tag=${tag} ==="
  MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${leg_tokens}" \
    research/e105_dose_leg.sh "${tag}" "${dose}" "${shape}"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "e105_dose_session: leg ${label} failed with ${status}" >&2
    failed+=("${tag}")
  fi
done

echo
echo "e105 dose session ${prefix} finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "tags: ${tags[*]}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "failed legs: ${failed[*]}" >&2
  exit 1
fi

python3 research/e105_dose_report.py "${tags[@]}" \
  --json "research/out/${prefix}/dose-report.json"
