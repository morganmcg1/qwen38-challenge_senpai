#!/usr/bin/env bash
# E130 rung 11, F16: the wired-slack ladder.
#
#   usage: research/e130_rung11_session.sh PREFIX TOKENS
#
# Rung 10a measured s64 -> s512 at -0.1968 % candidate seconds per token, with
# admission responding one for one in bytes and no sign of saturation. F16 asks
# whether the response keeps going. Four arms:
#
#   s64    the rung 10a anchor
#   s512   the currently shipped value
#   s1024  untimed
#   s2048  untimed, and the F16 bound-C hard ceiling
#
# DESIGN. A bare palindrome over four arms leaves 2 degrees of freedom and
# t_crit = 4.303, which is what made interim 12's serial channel useless. The
# order below is a palindrome over eight legs plus one uniformly random
# permutation of the same four arms:
#
#   legs 1-4    A B C D
#   legs 5-8    D C B A     position-balanced, so a linear trend cancels
#   legs 9-12   random      adds power without breaking the balance
#
# Fitting `y ~ arm + leg_index` then leaves 12 - 4 - 1 = 7 degrees of freedom
# and t_crit = 2.365. At the rung 10a residual sd of 0.0498 % that resolves
# 0.06 % at 2 sigma, about ten times finer than the smallest predicted step.
#
# THE RANDOM PERMUTATION IS PRE-REGISTERED. It was drawn before any leg ran,
# from a published seed, so it cannot be reselected after seeing the data:
#
#   seed_material = "e130-rung11-slack-ladder-F16"
#   seed          = int(sha256(seed_material)[:16], 16) = 14383609076371482244
#   python3 -c "import random,hashlib;
#     s=int(hashlib.sha256(b'e130-rung11-slack-ladder-F16').hexdigest()[:16],16)
#     a=['s64','s512','s1024','s2048']; random.Random(s).shuffle(a); print(a)"
#   -> ['s2048', 's512', 's64', 's1024']
#
# HEADLINE. Absolute candidate seconds per token. Not the local ratio: finding
# 160 showed the ratio adds 0.4788 % of serial noise to a 0.0498 % measurement,
# a 9.6x variance penalty, on a causally wrong estimand.
#
# THERMAL. MLXFAST_LOCAL_COOL_GATE=0 under the standing permitted mode, set by
# the leg script. Entry and exit GPU temperature are recorded per leg and every
# result carries cool_gate_passed_real_gate=false and
# gate_qualified_for_timing=false.
#
# BOUND C. The leg script refuses any arm above 2048 MiB. Live scratch peaks
# were 2,369.9 to 2,495.4 MiB above sizing, so this ladder stays below the
# region where the pool would start competing with the large transients.
#
# A failed leg does not stop the session. A missing leg is better than a
# half-counterbalanced one, and the reader reports which legs are present.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e130_rung11_session.sh PREFIX TOKENS}"
tokens="${2:?usage: e130_rung11_session.sh PREFIX TOKENS}"

order=(
  s64 s512 s1024 s2048
  s2048 s1024 s512 s64
  s2048 s512 s64 s1024
)

total="${#order[@]}"
index=0
for arm in "${order[@]}"; do
  index=$((index + 1))
  tag="${prefix}-$(printf '%02d' "${index}")-${arm}"
  echo "############ leg ${index}/${total}  arm=${arm}  tag=${tag} ############"
  research/e130_rung10a_leg.sh "${tag}" "${arm}" "${tokens}"
  echo "############ leg ${index}/${total} exit=$? ############"
done

echo "=== session complete: ${prefix} ==="
index=0
for arm in "${order[@]}"; do
  index=$((index + 1))
  tag="${prefix}-$(printf '%02d' "${index}")-${arm}"
  echo "--- ${tag} ---"
  grep -E "^(arm|exit|gpu_temp_entry_c|gpu_temp_exit_c|wired_residency_active|wired_clamped_count|wired_apply_failures|trace_anchor_lines|trace_pids)=" \
    "research/out/${tag}/meta.txt" 2>/dev/null || echo "missing"
done
