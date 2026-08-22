#!/usr/bin/env bash
# E130 rung 10, F13 section 4: the admission probe. NOT A TIMED EXPERIMENT.
#
#   usage: research/e130_admission_leg.sh TAG ARM TOKENS
#
# Asks one question with no clock in it: when the greedy fill in
# `ResidencySet::resize` runs, what does it admit and what does it leave in
# `unwired_set_`? The probe reports page-rounded byte counts, so the page
# rounding tax stops being a bound and becomes a measurement.
#
# ONE BINARY SERVES BOTH ARMS, exactly as in rung 10a. They differ only by
# `DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB`.
#
# The binary carries a research probe in `resident.cpp`, which is NOT in
# benchmark.json editablePaths and therefore never reaches a submission. It
# does file I/O under the allocator mutex, so this build must never be used
# for a timed leg. No cool gate is taken and none is claimed: there is no
# timing here to protect.
#
# Each `--local-iterate` leg starts several model-holding worker processes,
# and hash order over pointers differs per process, so one leg yields several
# independent draws of the admission lottery.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e130_admission_leg.sh TAG ARM TOKENS}"
arm="${2:?usage: e130_admission_leg.sh TAG ARM TOKENS}"
tokens="${3:?usage: e130_admission_leg.sh TAG ARM TOKENS}"

case "${arm}" in
  s64|s512) : ;;
  *) echo "e130 admission: unknown arm ${arm}" >&2; exit 2 ;;
esac

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full
export MLX_MAX_MB_PER_BUFFER=512
export MLX_MAX_OPS_PER_BUFFER=50
export MLXFAST_LOCAL_COOL_GATE=0
export MLXFAST_NO_SANDBOX=1

# The wired path is gated at 96 GiB and this host has 48, so the research
# override is what makes the arithmetic observable at all.
export MLX_E130_WIRED_GATE_GIB=32
export DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB="${arm#s}"

# The session probe reports active bytes and the live buffer count at sizing.
# The admission probe reports what the residency set did with them. Together
# they give the page tax as a difference rather than an assumption.
export MLX_E130_RESIDENCY_PROBE=1
export MLX_E130_RESIDENCY_PROBE_PATH="${PWD}/${out}/residency.log"
export MLX_E130_ADMISSION_PROBE_PATH="${PWD}/${out}/admission.log"
: > "${MLX_E130_RESIDENCY_PROBE_PATH}"
: > "${MLX_E130_ADMISSION_PROBE_PATH}"

{
  echo "experiment=e130-admission"
  echo "tag=${tag}"
  echo "arm=${arm}"
  echo "slack_mb=${DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB}"
  echo "tokens=${tokens}"
  echo "timed=false"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "admission_resize_events=$(grep -c 'phase=resize' "${MLX_E130_ADMISSION_PROBE_PATH}")"
  echo "admission_steady_events=$(grep -c 'phase=steady' "${MLX_E130_ADMISSION_PROBE_PATH}")"
  echo "sizing_events=$(grep -c 'phase=sizing' "${MLX_E130_RESIDENCY_PROBE_PATH}")"
} >> "${out}/meta.txt"

echo "=== ${tag} arm=${arm} exit=${status} ==="
grep -E "^(admission_|sizing_)" "${out}/meta.txt"
exit "${status}"
