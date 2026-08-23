#!/usr/bin/env bash
# E152 R2: the derived index's leaf width on the crown surface, ABBA
# counterbalanced inside one session, with the real 40 C gate on every leg.
#
#   usage: research/e152_r2_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# Arms come from ONE binary. `leaf8` sets MLX_E141_ROWS_PER_LEAF=8 and
# reproduces the crown's own index geometry; `leaf16` leaves the variable unset
# and takes the compiled default, which is what a ranked host runs because the
# runner sets no MLX_ variable. Both arms therefore carry the same census
# counters and the same trace fields, so the contrast is the leaf width alone.
#
# Order per replicate is leaf8, leaf16, leaf16, leaf8. Both arms have mean
# position 2.5, so monotone thermal drift inside a replicate cancels to first
# order, and the two leaf8 legs bracket the replicate.
#
# Prep is identical in kind for both arms: nothing rebuilds, and the derived
# index is built inside the untimed warm on every leg. The real cool gate runs
# immediately before each timed phase, so a prep-length difference cannot reach
# the timed window as a thermal offset.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-3}"
tokens="${2:-512}"
label="${3:-r2}"
first="${4:-1}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e152_r2_abba: worktree is dirty; refusing to time over uncommitted work" >&2
  git status --porcelain >&2
  exit 1
fi

failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in leaf8 leaf16 leaf16 leaf8; do
    position=$((position + 1))
    tag="e152${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    if [[ "${arm}" == "leaf8" ]]; then
      MLX_E141_ROWS_PER_LEAF=8 research/e79_trace_leg.sh "${tag}" "${tokens}" --cool-gate
    else
      research/e79_trace_leg.sh "${tag}" "${tokens}" --cool-gate
    fi
    status=$?
    {
      echo "e152_arm=${arm}"
      echo "e152_replicate=${rep}"
      echo "e152_position=${position}"
      echo "e152_rows_per_leaf_env=${arm/leaf/}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e152_r2_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e152_r2_abba: ${failures} failed legs"
E152_LABEL="${label}" E152_FIRST="${first}" E152_REPLICATES="${replicates}" \
  python3 research/e152_r2_abba_analyse.py
exit "${failures}"
