#!/usr/bin/env bash
# E130 rung 9: measure how much memory the scored window allocates AFTER the
# wired residency ticket is sized.
#
#   usage: research/e130_rung9_leg.sh TAG TOKENS
#
# The ticket is `active_at_sizing + 64 MiB`, capped below the recommended
# working set, and it is never resized. If the run allocates more than the
# 64 MiB allowance after that instant, the ticket is under-sized and the driver
# must drop something from the residency set.
#
# THIS IS NOT A TIMING LEG. It reads allocator counters only. The probe writes
# one line per second to stderr, which the wrapper captures in wrapper.err, and
# that write itself perturbs wall time. Never report a duration from this leg.
#
# The probe is opt-in and runs on both sides of the 96 GiB wired gate, so a
# 48 GiB host measures the same growth the 128 GiB ranked host would.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e130_rung9_leg.sh TAG TOKENS}"
tokens="${2:?usage: e130_rung9_leg.sh TAG TOKENS}"

export MLX_E130_RESIDENCY_PROBE=1
research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
status=$?

out="research/out/${tag}"
{
  echo "experiment=e130-rung9"
  echo "instrument=allocator-counters"
  echo "instrument_is_timing_safe=false"
  echo "residency_probe=1"
  echo "sizing_events=$(grep -c 'phase=sizing' "${out}/wrapper.err" 2>/dev/null || true)"
  echo "probe_samples=$(grep -c 'phase=sample' "${out}/wrapper.err" 2>/dev/null || true)"
} >> "${out}/meta.txt"

echo "=== e130 rung 9 sizing events, ${tag} ==="
grep 'phase=sizing' "${out}/wrapper.err" 2>/dev/null || echo "none"
echo "=== last 3 samples ==="
grep 'phase=sample' "${out}/wrapper.err" 2>/dev/null | tail -3 || echo "none"

exit "${status}"
