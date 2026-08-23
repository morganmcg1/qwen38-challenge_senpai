#!/usr/bin/env bash
# E145 R4, both cost models, one job.
#
# The measured run is the headline. The replayed run is the control that
# answers the actual six-point question: if you tune the price plane on the
# replayed curve, which cell do you pick, and what does that cell really cost
# once the curve is measured? Both runs sweep the same grid with the same
# seeds, so the replayed argmax can be read straight out of the measured grid
# without any extra fitting.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

seeds="${1:-6}"
windows="${2:-200}"
out="research/e145-artifacts"
mkdir -p "${out}"

failures=0
for cost in measured replayed; do
  echo "=== e145 r4 cost=${cost} seeds=${seeds} windows=${windows} ==="
  python3 research/e145_r4.py \
    --cost "${cost}" \
    --seeds "${seeds}" \
    --windows "${windows}" \
    --json "${out}/r4-${cost}.json" || {
      echo "e145_r4_both: cost=${cost} failed" >&2
      failures=$((failures + 1)); }
done

echo "=== e145 r4 cross-evaluation: tune on one curve, pay on the other ==="
python3 research/e145_r4_cross.py "${out}/r4-measured.json" \
  "${out}/r4-replayed.json" || failures=$((failures + 1))

echo "e145_r4_both: ${failures} failures"
exit $(( failures > 0 ))
