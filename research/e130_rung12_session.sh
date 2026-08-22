#!/usr/bin/env bash
# E130 rung 12, F21 section 4: a fresh base level on the 35d8cf58 tree.
#
#   usage: research/e130_rung12_session.sh PREFIX TOKENS
#
# WHY THIS EXISTS. The promoted one-pass QMV table `{6:6, 7:7}` reached the
# campaign base in `ea546d1f` and is unconditional, so it now runs on this
# `applegpu_g16s` host. F21 section 4 records that the host compiles NA=6 and
# NA=7 to a different, more tightly clamped kernel than the ranked runner does,
# and that NA=7 also spills 32 bytes here. Every absolute candidate seconds per
# token this experiment measured at `770a3ff2` is therefore void as a base.
#
# Two questions, one session:
#
#   1. What is the candidate leg time on an unchanged `35d8cf58` tree, under
#      the exact identity tuple a candidate would be measured with?
#   2. Does the wired-slack arm still read null on that tree? F21 section 4
#      notes a clamped kernel changes the resident scratch footprint, so the
#      arm could interact with the new base even though rung 11 refuted it at
#      `cbf87ee8`.
#
# ARMS. `none` is the genuine unchanged base: no environment override, so the
# compiled `wiredZHDefaultSlackMB = 64` applies. `s512` sets
# `DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB=512`. That override is base code, not a
# research patch, so this ladder needs no source change at all and the
# submitted surface stays byte-identical to the base on every leg.
#
# ORDER. Twelve legs, `none s512 s512 none` repeated three times. Both arms
# take mean leg index 6.5, so a linear session trend cancels exactly. Fitting
# `y ~ arm + leg_index` leaves 12 - 2 - 1 = 9 degrees of freedom and
# t_crit = 2.262. At the rung 11 residual sd of 0.0650 % that resolves about
# 0.060 % at 2 sigma, against the +0.0179 % +/- 0.0531 % that rung 11 measured
# for the same contrast.
#
# WARMUP LEG. Rung 11 leg 1 entered at 37.67 C while legs 2-13 entered at 58.55
# to 60.94 C, a 23.27 C spread carried entirely by one leg, and that leg was an
# arm leg. Leg 00 here is a discarded warmup that absorbs the cold start. It is
# excluded from the fit by the reader, which refuses to score it.
#
# TRACE OFF on every leg. Rung 11 measured the trace tax at +0.0410 % +/-
# 0.0835, not resolvable from zero but inside the absolute level. This session
# reports an absolute base level that a `--local-submit` run must match, and
# `--local-submit` does not trace, so the matched choice is untraced. The cost
# is that channels B and C are unavailable; rung 11 already read both and found
# both null.
#
# THERMAL. `MLXFAST_LOCAL_COOL_GATE=0` under the standing permitted mode, set
# by the leg script. Entry and exit GPU temperature are recorded per leg and
# every result carries `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false`.
#
# A failed leg does not stop the session. A missing leg is better than a
# half-counterbalanced one, and the reader reports which legs are present.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e130_rung12_session.sh PREFIX TOKENS}"
tokens="${2:?usage: e130_rung12_session.sh PREFIX TOKENS}"

warmup_tag="${prefix}-00-warmup"
echo "############ leg 00 WARMUP (discarded)  arm=none  tag=${warmup_tag} ############"
E130_LEG_TRACE=0 research/e130_rung10a_leg.sh "${warmup_tag}" none "${tokens}"
echo "############ leg 00 exit=$? ############"

order=(
  none s512 s512 none
  none s512 s512 none
  none s512 s512 none
)

total="${#order[@]}"
index=0
for arm in "${order[@]}"; do
  index=$((index + 1))
  tag="${prefix}-$(printf '%02d' "${index}")-${arm}"
  echo "############ leg ${index}/${total}  arm=${arm}  tag=${tag} ############"
  E130_LEG_TRACE=0 research/e130_rung10a_leg.sh "${tag}" "${arm}" "${tokens}"
  echo "############ leg ${index}/${total} exit=$? ############"
done

echo "=== session complete: ${prefix} ==="
for tag in "${warmup_tag}" ; do
  echo "--- ${tag} (discarded) ---"
  grep -E "^(arm|exit|gpu_temp_entry_c|gpu_temp_exit_c)=" \
    "research/out/${tag}/meta.txt" 2>/dev/null || echo "missing"
done
index=0
for arm in "${order[@]}"; do
  index=$((index + 1))
  tag="${prefix}-$(printf '%02d' "${index}")-${arm}"
  echo "--- ${tag} ---"
  grep -E "^(arm|exit|base_sha|worker_sha256|gpu_temp_entry_c|gpu_temp_exit_c|wired_residency_active|wired_clamped_count|wired_apply_failures|wired_slack_mb)=" \
    "research/out/${tag}/meta.txt" 2>/dev/null || echo "missing"
done
