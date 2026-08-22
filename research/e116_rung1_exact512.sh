#!/usr/bin/env bash
# E116 rung 1 -- the 512-token exactness pair.
#
#   usage: research/e116_rung1_exact512.sh
#
# Two traced legs at the full 512-token window, dose 0 and dose 12, and one
# digest comparison over their ordered `mtp-row:` lines. The pinned E101
# reference is 1025 rows at
# sha256 719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e
# (`research/e101-results.md:123`). Both legs must reproduce it.
#
# 512 is the only representative window: the seed is 512 tokens, so a shorter
# leg never walks the key length past 1024 and never exercises the boundary
# behaviour the row ledger has to close over.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e116_rung1_exact512: worktree is dirty; refusing to measure over" \
       "uncommitted work" >&2
  exit 1
fi

failures=0
for arm in 0 12; do
  echo
  echo "=== e116x512k${arm}: MLX_E116_DOSE=${arm} tokens=512 ==="
  research/e116_exactness_leg.sh "e116x512k${arm}" 512 "${arm}" \
    || { echo "e116_rung1_exact512: leg k=${arm} failed" >&2
         failures=$((failures + 1)); }
done

echo
python3 research/e101_row_digest.py e116x512k0 e116x512k12 || failures=$((failures + 1))
python3 research/e116_row_digest_check.py e116x512k0 e116x512k12 \
  --pin 719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e \
  --expect-rows 1025 \
  --negative-control e116r1-insitu-k0 \
  --json research/e116-artifacts/row-digest-512.json \
  || failures=$((failures + 1))

echo
echo "e116_rung1_exact512: ${failures} failures"
exit "${failures}"
