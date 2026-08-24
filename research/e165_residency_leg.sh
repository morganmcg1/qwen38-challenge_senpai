#!/usr/bin/env bash
# Run one leg with a hardware GPU residency sampler attached, for E165.
#
#   usage: research/e165_residency_leg.sh TAG TOKENS [e79_trace_leg args...]
#
# ADVISOR F1 asks for the GPU-idle fraction MEASURED rather than inferred. The
# host-clock census can only bound it: a host clock cannot see a bubble that
# opens after the command buffer is committed and closes before it completes,
# so `gpu_covered` is an UPPER bound on busy and the host-visible gap is a
# LOWER bound on idle. `powermetrics --samplers gpu_power` reads the GPU's own
# residency counters, so it sees those intra-queue bubbles directly.
#
# CONTROLS. The sampler runs against an idle GPU before and after the leg. An
# instrument that cannot tell a decoding GPU from an idle one is caught by
# those two samples, not assumed away. The parser refuses to report a leg whose
# idle-baseline residency is not high.
#
# The sampler is system-wide, not per-process. The idle baselines also bound
# how much of the leg's activity belongs to other processes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e165_residency_leg.sh TAG TOKENS [e79 args...]}"
tokens="${2:?usage: e165_residency_leg.sh TAG TOKENS [e79 args...]}"
shift 2

interval_ms="${E165_RESIDENCY_INTERVAL_MS:-100}"
out="research/out/${tag}"

sudo -n true 2>/dev/null || {
  echo "e165_residency_leg.sh: powermetrics needs passwordless sudo" >&2
  exit 1
}

sample_idle() {
  sudo -n powermetrics --samplers gpu_power -n 1 -i 300 2>/dev/null
}

# e79_trace_leg.sh removes and recreates the output directory, so the baseline
# has to be captured somewhere else first and moved in afterwards.
pre="$(mktemp -t e165res)"
sample_idle > "${pre}"

# Bound the sampler by its own sample count as well as by the kill below. An
# unbounded sampler that survives the kill would otherwise hold the whole job
# allocation open.
max_samples="${E165_RESIDENCY_MAX_SAMPLES:-9000}"
sampler_log="$(mktemp -t e165pm)"
sudo -n powermetrics --samplers gpu_power -i "${interval_ms}" -n "${max_samples}" \
  > "${sampler_log}" 2>/dev/null &
sleep 1

research/e79_trace_leg.sh "${tag}" "${tokens}" "$@"
status=$?

sleep 1
sudo -n pkill -INT -f 'powermetrics --samplers gpu_power -i' 2>/dev/null
sleep 1
sudo -n pkill -KILL -f 'powermetrics --samplers gpu_power -i' 2>/dev/null

mv "${sampler_log}" "${out}/powermetrics.txt"
mv "${pre}" "${out}/powermetrics-idle-pre.txt"
sample_idle > "${out}/powermetrics-idle-post.txt"

{
  echo "e165_residency_interval_ms=${interval_ms}"
  echo "e165_residency_sampler=powermetrics gpu_power"
  echo "e165_stage=0-census"
  echo "e165_leg_role=gpu-residency"
} >> "${out}/meta.txt"

exit "${status}"
