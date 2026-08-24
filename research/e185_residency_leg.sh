#!/usr/bin/env bash
# E185 stage 1: sample the GPU hardware idle counter across one decode leg.
#
#   usage: research/e185_residency_leg.sh TAG TOKENS [e90_leg args...]
#
# E165 sampled `powermetrics` in its human text form at 100 ms and read the
# percentage field, so its window had to be found by thresholding whole
# samples. This leg reads the plist form instead. Every sample carries
# `elapsed_ns` and `idle_ns` from the same hardware counter, so the idle share
# of an arbitrary window is a ratio of two nanosecond sums rather than a mean
# of rounded percentages, and a sample that straddles a phase boundary is
# visible as one sample rather than silently rounded into the window.
#
# ALIGNMENT. `mtp-anchor:` stamps every round phase on the mach uptime clock
# and the plist stamps every sample on the wall clock at one-second
# resolution. The offset between the two clocks is captured here, before and
# after the leg, so the reader can place the decode rounds inside the sample
# stream instead of guessing the window from a busy threshold.
#
# OBSERVATION ONLY. This leg is not a timed contrast. It runs ungated, it runs
# with the in-repo round trace on, and it carries a system-wide sampler, so its
# absolute seconds per token are not comparable with a gated production leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e185_residency_leg.sh TAG TOKENS [e90 args...]}"
tokens="${2:?usage: e185_residency_leg.sh TAG TOKENS [e90 args...]}"
shift 2

interval_ms="${E185_RESIDENCY_INTERVAL_MS:-20}"
max_samples="${E185_RESIDENCY_MAX_SAMPLES:-60000}"
out="research/out/${tag}"

sudo -n true 2>/dev/null || {
  echo "e185_residency_leg.sh: powermetrics needs passwordless sudo" >&2
  exit 1
}

clock_pair() {
  python3 -c 'import time; print("%.9f %d" % (time.time(), time.monotonic_ns()))'
}

# e79_trace_leg.sh removes and recreates the output directory, so anything
# captured before it runs has to live outside the directory first.
pre_clock="$(clock_pair)"
idle_pre="$(mktemp -t e185res)"
sudo -n powermetrics --samplers gpu_power -n 1 -i 300 -f plist \
  > "${idle_pre}" 2>/dev/null

sampler_log="$(mktemp -t e185pm)"
sudo -n powermetrics --samplers gpu_power -i "${interval_ms}" \
  -n "${max_samples}" -f plist > "${sampler_log}" 2>/dev/null &
sleep 1

research/e90_leg.sh "${tag}" "${tokens}" "$@"
status=$?

sleep 1
post_clock="$(clock_pair)"
sudo -n pkill -INT -f 'powermetrics --samplers gpu_power -i' 2>/dev/null
sleep 1
sudo -n pkill -KILL -f 'powermetrics --samplers gpu_power -i' 2>/dev/null

mv "${sampler_log}" "${out}/powermetrics.plist"
mv "${idle_pre}" "${out}/powermetrics-idle-pre.plist"
sudo -n powermetrics --samplers gpu_power -n 1 -i 300 -f plist \
  > "${out}/powermetrics-idle-post.plist" 2>/dev/null

{
  echo "e185_residency_interval_ms=${interval_ms}"
  echo "e185_residency_sampler=powermetrics gpu_power plist"
  echo "e185_clock_pre_wall_mono=${pre_clock}"
  echo "e185_clock_post_wall_mono=${post_clock}"
  echo "e185_leg_role=gpu-residency-observation"
  echo "e185_timed_contrast=false"
} >> "${out}/meta.txt"

exit "${status}"
