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
# one line per second, and that write itself perturbs wall time. Never report a
# duration from this leg.
#
# The probe writes to a FILE, not to stderr. The `mtp-timed` parent builds its
# runtime worker through `runtimeWorkerOptions` without `forwardsWorkerStderr`,
# so `WorkerStderrDrain` installs a swallowing emitter and throws every worker
# stderr line away. A first attempt that wrote to stderr produced an empty
# result for exactly that reason. The file sink needs `MLXFAST_NO_SANDBOX=1`,
# which `research/e79_trace_leg.sh` already exports.
#
# Every worker process of the leg -- reference, serial control and MTP -- opens
# the same file O_APPEND and tags its lines with its own pid, so a reader must
# group by pid before it reads a growth series.
#
# The probe is opt-in and runs on both sides of the 96 GiB wired gate, so a
# 48 GiB host measures the same growth the 128 GiB ranked host would.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e130_rung9_leg.sh TAG TOKENS}"
tokens="${2:?usage: e130_rung9_leg.sh TAG TOKENS}"

out="research/out/${tag}"
mkdir -p "${out}"
probe="${PWD}/${out}/residency.log"
: > "${probe}"

export MLX_E130_RESIDENCY_PROBE=1
export MLX_E130_RESIDENCY_PROBE_PATH="${probe}"
research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace
status=$?

{
  echo "experiment=e130-rung9"
  echo "instrument=allocator-counters"
  echo "instrument_is_timing_safe=false"
  echo "residency_probe=1"
  echo "residency_probe_path=${probe}"
  echo "sizing_events=$(grep -c 'phase=sizing' "${probe}" 2>/dev/null || true)"
  echo "probe_samples=$(grep -c 'phase=sample' "${probe}" 2>/dev/null || true)"
} >> "${out}/meta.txt"

echo "=== e130 rung 9 sizing events, ${tag} ==="
grep 'phase=sizing' "${probe}" 2>/dev/null || echo "none"
echo "=== last 3 samples per pid ==="
for pid in $(grep -o 'pid=[0-9]*' "${probe}" 2>/dev/null | sort -u); do
  echo "--- ${pid} ---"
  grep "${pid} phase=sample" "${probe}" | tail -3
done

exit "${status}"
