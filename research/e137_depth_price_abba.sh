#!/usr/bin/env bash
# E137: pb6 against ship, timed in one counterbalanced same-binary session.
#
#   usage: research/e137_depth_price_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# WHY. Ranked receipt `e003a86d` scored 3.57502547497774, which is 2.380 %
# below our own previous receipt `572b2cc4` at 3.66218563656629. Three changes
# separated them. Two carry ranked isolations that together predict +0.52 to
# +0.65 %: removing the one-pass table (`0b2f0014`, +0.2646 % slower when
# restored) and lowering the probe fraction. The third, `pb6`, has no ranked
# isolation, so the whole -2.9 to -3.0 % residual sits on it.
#
# The mechanism is circular. `passBoundaryTierFactor = 1.45` at
# `passBoundaryVerifyWidth = 6` is fitted from
# `research/e134-artifacts/item2-measured-curve.json`, whose header reads
# `receipt 623e77af, harness ranked`. That receipt is the promotion of the
# one-pass table. Its 5-to-6 step is 16,566.7 us against an ordinary step of
# 3,446.1 us. Under the one-pass table width 6 runs `na6` at 105 registers and
# 37 simdgroups on g17s; under the reverted two-pass plan it runs `na3` at 94
# registers and 42 simdgroups. `e003a86d` deleted the table that made width 6
# dear and kept the price fitted to it.
#
# WHY THE LOCAL RATIO IS ADMISSIBLE HERE. `program.md` allows the local
# serial-to-MTP ratio as direct evidence for changes whose causal path is
# confined to the candidate MTP leg, and names schedule policy as such a case.
# A depth price changes only how many drafts the candidate proposes. It cannot
# touch the serial leg. Absolute candidate seconds per token is still the
# headline and the serial leg is still reported as a null diagnostic.
#
# COUNTERBALANCE. One replicate is `pb6 ship ship pb6`, so both arms have mean
# position 2.5 and a monotone drift in leg index cancels to first order. Entry
# and exit temperature are recorded per leg. Every leg keeps
# `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false`:
# this is directional evidence inside its own session and nothing more.
#
# NO REBUILD BETWEEN LEGS. `MLX_E134_DEPTH_PRICE_ARM` selects the arm at run
# time, so every leg times the same bytes and `worker_sha256` is asserted
# equal across the session.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-3}"
tokens="${2:-512}"
label="${3:-D1}"
first="${4:-1}"

order=(pb6 ship ship pb6)
legs_per_rep="${#order[@]}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e137_depth_price_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
echo "e137_depth_price_abba: commit=${session_commit} worker=${session_worker}"
echo "e137_depth_price_abba: ${replicates} replicates x ${legs_per_rep} legs," \
     "${tokens} tokens"

edl_of() {
  python3 -c "
import json,sys
print(repr(json.load(open(sys.argv[1]))['metrics']['effective_mean_draft_len']))
" "$1"
}

# PHASE 1: CAMPAIGN RULE 114. An unrecognized value of
# `MLX_E134_DEPTH_PRICE_ARM` falls back to the compiled default, so a selector
# that never arrives is indistinguishable from one that arrives and is ignored:
# both run `pb6`. The witness therefore reads the realised schedule out of the
# leg's own `score.json`, never the variable the leg was asked with.
#
# Three short legs settle it:
#   unset  -> the compiled default
#   pb6    -> must equal the unset leg, proving pb6 IS the compiled default
#   ship   -> must differ, proving the selector arrives and moves the schedule
witness_tokens="${E137_WITNESS_TOKENS:-64}"
declare -a witness_edl
wi=0
for arm in unset pb6 ship; do
  tag="e137${label}w${arm}"
  echo "=== witness ${tag}: arm=${arm} tokens=${witness_tokens} ==="
  out="research/out/${tag}"
  mkdir -p "${out}"
  if [[ "${arm}" != "unset" ]]; then
    export MLX_E134_DEPTH_PRICE_ARM="${arm}"
  fi
  research/e79_trace_leg.sh "${tag}" "${witness_tokens}" --no-trace
  status=$?
  unset MLX_E134_DEPTH_PRICE_ARM
  if ((status != 0)); then
    echo "e137_depth_price_abba: witness leg ${tag} exited ${status}" >&2
    exit 2
  fi
  e="$(edl_of "${out}/score.json")"
  witness_edl[$wi]="${e}"
  wi=$((wi + 1))
  {
    echo "e137_arm=${arm}"
    echo "e137_phase=witness"
    echo "e137_effective_mean_draft_len=${e}"
  } >> "${out}/meta.txt"
  echo "  realised effective_mean_draft_len = ${e}"
done

echo "--- Rule 114 witness ---"
echo "  unset ${witness_edl[0]}"
echo "  pb6   ${witness_edl[1]}"
echo "  ship  ${witness_edl[2]}"
if [[ "${witness_edl[0]}" != "${witness_edl[1]}" ]]; then
  echo "e137_depth_price_abba: the unset leg and the pb6 leg realised" \
       "different schedules, so pb6 is not the compiled default and this" \
       "session does not measure what it claims" >&2
  exit 3
fi
if [[ "${witness_edl[1]}" == "${witness_edl[2]}" ]]; then
  echo "e137_depth_price_abba: pb6 and ship realised the SAME schedule, so" \
       "the selector never arrived or the arms are identical; not timing" >&2
  exit 4
fi
echo "control ok: pb6 is the compiled default and ship moves the schedule"

run_leg() {
  local tag="$1" arm="$2" toks="$3"
  shift 3
  if [[ "${arm}" != "unset" ]]; then
    export MLX_E134_DEPTH_PRICE_ARM="${arm}"
  fi
  research/e79_trace_leg.sh "${tag}" "${toks}" --no-trace
  local status=$?
  unset MLX_E134_DEPTH_PRICE_ARM
  {
    echo "e137_arm=${arm}"
    echo "e137_session_commit=${session_commit}"
    echo "e137_session_worker_sha256=${session_worker}"
    printf '%s\n' "$@"
  } >> "research/out/${tag}/meta.txt"
  return "${status}"
}

# PHASE 2: the counterbalanced timed session.
failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in "${order[@]}"; do
    position=$((position + 1))
    tag="e137${label}k${rep}p$(printf '%02d' "${position}")${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    run_leg "${tag}" "${arm}" "${tokens}" \
      "e137_replicate=${rep}" \
      "e137_position=${position}" \
      "e137_leg_index=$(( (rep - first) * legs_per_rep + position ))" \
      "e137_phase=ladder"
    status=$?
    if ((status != 0)); then
      echo "e137_depth_price_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e137_depth_price_abba: ${failures} failed legs"
python3 research/e137_depth_price_report.py "${label}" \
  | tee "research/out/e137${label}-report.txt"
exit "${failures}"
