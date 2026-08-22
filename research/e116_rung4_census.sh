#!/usr/bin/env bash
# E116 rung 4 -- the measured wide-QMV share at the REALISED verify width.
#
#   usage: research/e116_rung4_census.sh
#
# One 512-token census leg with the schedule left alone and one dispatch per
# command buffer. Two properties matter and they pull in opposite directions:
#
#   REALISED WIDTH. `MLX_E80_FORCE_DRAFTS` stays unset, so this leg reports the
#   histogram the shipped schedule actually chooses. Finding 47 says the local
#   fixture runs at mean verify width 6.9 while the scored point is 5.4 to 7.1,
#   so a forced-width share is a share of a width the scored worker may never
#   use. The 64-token rung 1 leg only reached 10 MTP rounds; 512 tokens reaches
#   about 79 and is the window the ranked contract uses.
#
#   ISOLATION. `exclusive_kernels` is a per-kernel exclusive GPU time only when
#   one command buffer holds one dispatch, so `MLX_E58_BUFFER_LIMIT_OPS=0` is
#   required. Rung 1 measured what isolation costs: the width-1 round GPU busy
#   went from 64,343 us in situ to 67,297 us isolated, +4.6 %. That inflation
#   applies to every kernel in the leg, so a RATIO of two kernels within the
#   leg is far more robust to it than either absolute number, and a share is a
#   ratio.
#
# The dose is off. This leg measures the model, not the instrument.
#
# A census leg is never a timing leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${E116_RUNG4_TAG:-e116r4-realised-512}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e116_rung4_census: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  exit 1
fi

research/e116_census_leg.sh "${tag}" realised 512 0 0
status=$?

python3 research/e116_qmv_share.py "${tag}" \
  --json "research/e116-artifacts/rung4-qmv-share-512.json" \
  || status=1
python3 research/e116_round_switch_witness.py "${tag}" \
  --json "research/e116-artifacts/rung0b-round-switch-witness.json" \
  || status=1

exit "${status}"
