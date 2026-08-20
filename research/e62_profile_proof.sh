#!/usr/bin/env bash
# Rung 0 instrument proof 1: DARKBLOOM_STARTUP_MEMORY_PROFILE really reaches
# RuntimeStartupMemoryPolicy.resolve inside the worker process.
#
# The positive control is a deliberate crash. `=bogus` takes the default branch
# of the switch in resolve() and hits preconditionFailure
# (RuntimeStartupMemoryPolicy.swift:101-104), which traps the process. `=full`
# resolves cleanly and the worker instead fails later on the missing
# config.json. Two different failures from the same command line prove the
# variable is read.
#
# Neither launch touches the GPU: both end before any weight load. This runs on
# the stock binary, which is the binary every timed leg of rungs 1-3 uses.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/out/e62-rung0-profile-proof"
rm -rf "${out}"
mkdir -p "${out}"

cp research/out/e62/bin/stock/mlxfast-runtime-worker \
   .build-worker/release/mlxfast-runtime-worker

probe() {
  local profile="$1" log="$2"
  (
    cd .build-worker/release \
      && DARKBLOOM_STARTUP_MEMORY_PROFILE="${profile}" \
         ./mlxfast-runtime-worker mtp-runtime-worker \
           --weights /nonexistent-e62 --mtp-head /nonexistent-e62
  ) > "${log}" 2>&1
  echo $?
}

bogus_status="$(probe bogus "${out}/bogus.log")"
full_status="$(probe full "${out}/full.log")"
auto_status="$(probe auto "${out}/auto.log")"

bogus_precondition="$(grep -c "must be auto, full, or low" "${out}/bogus.log")"
full_notice="$(grep -c "low-memory startup profile engaged" "${out}/full.log")"
auto_notice="$(grep -c "low-memory startup profile engaged" "${out}/auto.log")"

{
  echo "{"
  echo " \"proof\": \"startup memory profile reaches resolve() in the worker\","
  echo " \"binary\": \"stock\","
  echo " \"bogus_exit\": ${bogus_status},"
  echo " \"bogus_precondition_lines\": ${bogus_precondition},"
  echo " \"full_exit\": ${full_status},"
  echo " \"full_low_memory_notice_lines\": ${full_notice},"
  echo " \"auto_exit\": ${auto_status},"
  echo " \"auto_low_memory_notice_lines\": ${auto_notice},"
  echo " \"host_physical_memory_gib\": $(( $(sysctl -n hw.memsize) >> 30 )),"
  # A release build drops the preconditionFailure message, so the signature is
  # the trap itself: 128 + SIGTRAP(5) = 133.
  echo " \"bogus_expected_exit_sigtrap\": 133,"
  echo " \"passed\": $(
    if [[ "${bogus_status}" -eq 133 && "${full_status}" -eq 1 \
          && "${full_notice}" -eq 0 && "${auto_notice}" -ge 1 ]]; then
      echo true
    else
      echo false
    fi
  )"
  echo "}"
} | tee "${out}/proof.json"

echo "--- bogus tail ---"; tail -5 "${out}/bogus.log"
echo "--- full tail ---"; tail -3 "${out}/full.log"
echo "--- auto tail ---"; tail -3 "${out}/auto.log"
