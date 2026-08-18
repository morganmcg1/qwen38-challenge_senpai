#!/usr/bin/env bash
# Research-only (qwen38-r1-e24-constant-scalar-dispatch-tax): wait for this
# host's GPU to fall to a target temperature, recording a thermal trace.
#
#   research/e24-cool-watch.sh [TARGET_C] [MAX_WAIT_S] [OUT_TSV]
#
# WHY THIS EXISTS. benchmark.sh's cool gate is hardcoded `readonly
# COOL_GATE_TEMP_C=40` (benchmark.sh:28) and aborts once no NEW minimum has been
# seen for COOL_GATE_STALL_SECONDS=90 (benchmark.sh:31,978). On 2026-08-18 an
# E24 arm hit that abort at 42.7C while the GPU drew 0.008W at 1% usage with no
# benchmark process resident -- i.e. the die had already reached this chassis's
# soak floor, not "something else is loading the GPU" as the gate's hint guesses.
#
# The gate is NOT bypassed or weakened here: this script runs BEFORE a measured
# arm and only decides WHEN to start it, so every arm still passes the
# unmodified 40C gate on its own. The trace it records is the evidence for how
# far the idle floor actually is from the gate, which is a reportable property
# of the host rather than a nuisance to wait out silently.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

target="${1:-39.0}"
max_wait="${2:-1500}"
out="${3:-.mlxfast-private/e24/thermal/cool-watch-$(date -u +%Y%m%dT%H%M%SZ).tsv}"
interval=15

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
[[ -x "${macmon_bin}" ]] || { echo "e24-cool-watch: no macmon at ${macmon_bin}" >&2; exit 2; }

mkdir -p "$(dirname "${out}")"
printf 'iso_time\twaited_s\tgpu_temp_c\tcpu_temp_c\tgpu_power_w\tall_power_w\n' >"${out}"

waited=0
min_temp=""
while :; do
  sample="$("${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '[(.temp.gpu_temp_avg // 0), (.temp.cpu_temp_avg // 0),
              (.gpu_power // 0), (.all_power // 0)] | @tsv')"
  [[ -n "${sample}" ]] || { echo "e24-cool-watch: macmon unreadable" >&2; exit 2; }
  temp="$(cut -f1 <<<"${sample}")"
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${waited}" "${sample}" >>"${out}"

  if [[ -z "${min_temp}" ]] || awk -v t="${temp}" -v m="${min_temp}" 'BEGIN{exit !(t<m)}'; then
    min_temp="${temp}"
  fi
  if awk -v t="${temp}" -v g="${target}" 'BEGIN{exit !(t<=g)}'; then
    echo "e24-cool-watch: reached ${temp}C <= ${target}C after ${waited}s (trace ${out})"
    exit 0
  fi
  if ((waited >= max_wait)); then
    echo "e24-cool-watch: STILL ${temp}C after ${waited}s (min seen ${min_temp}C, target ${target}C)" >&2
    echo "e24-cool-watch: this host's idle GPU floor is above the target; trace ${out}" >&2
    exit 3
  fi
  sleep "${interval}"
  waited=$((waited + interval))
done
