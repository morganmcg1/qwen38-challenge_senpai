#!/usr/bin/env bash
# E158 R1 F8 -- external resident-set sampler for the gated ABBA palindrome.
#
# `Memory.peakMemory` is only reachable through the worker's phase_diagnostics
# request and QwenRuntimeMTPDriver never issues it, so the MTP path prints no
# peak_ram_gb at all (see research/e62_run_leg.sh:212-215). This sampler runs
# for the whole gated session, so its cost is identical across all four legs
# and cancels in every arm contrast. It reads `ps` only; it never touches the
# GPU and never writes inside the repository checkout.
#
# Output is one line per sample: RFC3339 UTC, pid, rss_kb. Legs are attributed
# afterwards from each leg's `started` and `finished` stamps in meta.txt.
set -uo pipefail

out="${1:?usage: sample_worker_rss.sh OUTFILE [SECONDS]}"
deadline=$(( $(date +%s) + ${2:-7000} ))
: > "${out}"

while (( $(date +%s) < deadline )); do
  for pid in $(pgrep -f 'mlxfast-runtime-worker' 2>/dev/null); do
    rss="$(ps -o rss= -p "${pid}" 2>/dev/null | tr -d ' ')"
    [[ -n "${rss}" ]] \
      && echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${pid} ${rss}" >> "${out}"
  done
  sleep 2
done
