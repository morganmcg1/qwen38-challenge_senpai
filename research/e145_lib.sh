#!/usr/bin/env bash
# E145 shared leg runner. One gated, timed 512-token decode leg per call.
#
# WHY THIS EXISTS AND NOT `research/e134_abba.sh`. E134's session took no
# thermal gate: every leg it wrote carries `cool_gate_passed_real_gate=false`
# and leans on the ABBA palindrome to cancel drift. E145 R1 has to reproduce an
# E134 number AND compare three fixtures whose legs are minutes apart, so the
# palindrome alone is not enough. This runner takes the REAL 40 C gate through
# the unmodified `benchmark.sh --local-cool-gate-only` entry point immediately
# before every leg, and records the gate's own evidence beside the leg.
#
# `e128_session.sh` writes `cool_gate_passed_real_gate=false` unconditionally,
# because it never takes a gate itself. That line stays in the file exactly as
# it wrote it. This runner appends its own `e145_*` gate fields afterwards, so
# a reader sees both the sub-session's default and the gate this session
# actually took, and can tell which is which.
#
# HARNESS DEFECT 36: `./benchmark-qwen-mtp.sh` never rebuilds the worker, so a
# session must build and witness the worker ONCE, before the first leg. The
# session scripts do that; this file only runs legs.
set -uo pipefail

readonly E145_WORKER=".build-worker/release/mlxfast-runtime-worker"

# The GPU temperature reader `benchmark.sh` itself uses. Reported per leg so an
# entry-temperature spread can be put beside every measured effect.
e145_gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

# Take the real gate. Returns non-zero if the gate refuses, which is the
# correct behaviour: a leg that starts on a hot GPU is not a gated leg.
#
# `benchmark.sh` gives up after 180 s of its own waiting. Late in a long
# session the GPU can sit just above 40 C and cool slower than that, so a
# single refusal is not evidence that something else is loading the GPU. The
# gate is therefore re-entered up to `E145_GATE_ATTEMPTS` times, and each
# attempt is the SAME real gate at the SAME real 40 C threshold. Nothing here
# lowers, bypasses or fakes the gate: a leg still runs only after the
# unmodified `benchmark.sh` says the GPU is cool enough.
e145_cool_gate() {
  local log="$1" started elapsed attempt rc=1
  started="$(date +%s)"
  for (( attempt = 1; attempt <= ${E145_GATE_ATTEMPTS:-3}; attempt++ )); do
    ./benchmark.sh --local-cool-gate-only > "${log}" 2>&1
    rc=$?
    E145_GATE_ATTEMPTS_USED="${attempt}"
    ((rc == 0)) && break
    echo "e145_leg: the real cool gate refused on attempt ${attempt};" \
         "re-entering the same gate" >&2
  done
  elapsed=$(( $(date +%s) - started ))
  E145_GATE_WAIT_S="${elapsed}"
  E145_GATE_LINE="$(grep -h 'cool-down gate passed' "${log}" | tail -1)"
  return "${rc}"
}

# e145_leg SLOT FIXTURE ARM PIN TOKENS
#
#   ARM    a `MLX_E134_DEPTH_PRICE_ARM` value, or `serial` for the depth-0
#          control, which reads no depth price at all.
#   PIN    a `MLX_E145_PIN_DEPTH` value, or `none` for the shipped estimator.
#
# Writes to `.mlxfast-private/e128/e145/SLOT/FIXTURE/`. Sets `E145_LEG_OUT`.
e145_leg() {
  local slot="$1" fixture="$2" arm="$3" pin="$4" tokens="$5"
  local out=".mlxfast-private/e128/e145/${slot}/${fixture}"
  local gate_log=".mlxfast-private/e128/e145/${slot}-${fixture}-gate.log"
  E145_LEG_OUT="${out}"
  # Created BEFORE the gate so a refused gate can still record why it refused.
  mkdir -p "${out}"

  echo "=== e145 ${slot}: fixture=${fixture} arm=${arm} pin=${pin}" \
       "tokens=${tokens} ==="
  if ! e145_cool_gate "${gate_log}"; then
    echo "e145_leg: the real cool gate refused before ${slot}/${fixture}" >&2
    tail -5 "${gate_log}" >&2
    {
      echo "e145_slot=${slot}"
      echo "e145_real_cool_gate_taken=false"
      echo "e145_gate_refused=true"
      echo "e145_gate_attempts=${E145_GATE_ATTEMPTS_USED}"
      echo "e145_gate_wait_s=${E145_GATE_WAIT_S}"
      echo "e145_leg_exit=4"
    } >> "${out}/meta.txt"
    return 4
  fi
  local entry_c
  entry_c="$(e145_gpu_temp)"
  echo "e145_leg: gate passed in ${E145_GATE_WAIT_S}s, entry ${entry_c}C"

  # CAMPAIGN RULE 128: two arms are comparable only if their warm phase left
  # the process in the same residency and cache state, so every leg tries to
  # record `wired-zh` and `warm`.
  #
  # THIS CANNOT SUCCEED ON THE TIMED PATH, and R1b proved it. Both lines are
  # written by the WORKER to its own stderr
  # (`Qwen36MTPBlockSession.swift:271` and `:354`). The `mtp-timed` parent
  # calls `runtimeWorkerOptions` WITHOUT `forwardsWorkerStderr`, so
  # `QwenRuntimeWorker.swift:2046` installs a swallowing emitter and the
  # stream never reaches any file. `MLX_DFLASH_TRACE_CACHE_SEAM` does not
  # help: it is read only inside the DFlash subcommand
  # (`MLXFastCLI/main.swift:1409`), which `mtp-timed` never enters. The
  # source says so directly at `Qwen36MTPBlockSession.swift:800-806`.
  #
  # Capturing this telemetry needs a change in the trusted parent, which is
  # outside the E145 assignment scope. The variable is still set on every leg
  # so the probe stays uniform across arms, and the reader records
  # `warm_telemetry_present=false` rather than skipping it, because a leg with
  # no telemetry cannot support an arm attribution claim and that fact must be
  # visible.
  local -a leg_env=(E128_FORCE=1 E128_NO_TRACE=1
                    MLX_DFLASH_TRACE_CACHE_SEAM=1
                    "E128_TOKENS=${tokens}"
                    "E128_RUNS_DIR=e145/${slot}")
  if [[ "${arm}" == "serial" ]]; then
    leg_env+=(E128_DEPTH=0)
  else
    leg_env+=(E128_DEPTH=8 "MLX_E134_DEPTH_PRICE_ARM=${arm}")
  fi
  [[ "${pin}" != "none" ]] && leg_env+=("MLX_E145_PIN_DEPTH=${pin}")

  env "${leg_env[@]}" research/e128_session.sh "${fixture}"
  local rc=$?
  local exit_c
  exit_c="$(e145_gpu_temp)"

  {
    echo "e145_experiment=e145-live-width-cost-curve"
    echo "e145_slot=${slot}"
    echo "e145_arm_requested=${arm}"
    echo "e145_pin_requested=${pin}"
    echo "e145_real_cool_gate_taken=true"
    echo "e145_cool_gate_source=benchmark.sh --local-cool-gate-only"
    echo "e145_cool_gate_line=${E145_GATE_LINE}"
    echo "e145_gate_wait_s=${E145_GATE_WAIT_S}"
    echo "e145_gate_attempts=${E145_GATE_ATTEMPTS_USED}"
    echo "e145_gate_entry_temp_c=${entry_c}"
    echo "e145_leg_exit_temp_c=${exit_c}"
    echo "e145_session_commit=${E145_SESSION_COMMIT}"
    echo "e145_session_worker_sha256=${E145_SESSION_WORKER}"
    echo "e145_leg_exit=${rc}"
    e145_warm_telemetry "${out}/${fixture}/stderr.log"
  } >> "${out}/meta.txt"
  return "${rc}"
}

# Pull one `key=value` field out of a whitespace-separated telemetry line.
e145_field() {
  sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" <<< "$1"
}

# Rule 128 evidence for one leg, written as `e145_warm_*` meta keys. Absence is
# recorded rather than skipped: a leg with no telemetry cannot support an arm
# attribution claim, and the reader must be able to see that.
e145_warm_telemetry() {
  local log="$1" wired warm key
  wired="$(grep -h 'qwen-mtp wired-zh' "${log}" 2>/dev/null | tail -1)"
  warm="$(grep -h 'qwen-mtp warm ' "${log}" 2>/dev/null | tail -1)"
  echo "e145_warm_telemetry_present=$(
    [[ -n "${wired}" && -n "${warm}" ]] && echo true || echo false)"
  echo "e145_warm_wired_line=${wired}"
  echo "e145_warm_warm_line=${warm}"
  for key in request applied active slack_mb fraction maxrec; do
    echo "e145_warm_wired_${key}=$(e145_field "${wired}" "${key}")"
  done
  for key in shapes_ms wnorm wprefetch refill refill_ms emulated_clear \
             cache_after_sizing cache_end active_end wired_gate_fired \
             physmem; do
    echo "e145_warm_${key}=$(e145_field "${warm}" "${key}")"
  done
}

# Build the worker once and prove BOTH E145 gates are inside the bytes that are
# about to run. A same-binary env-gated arm study whose selector never reached
# the binary measures nothing, and both selectors are longer than the 16-byte
# floor that `rebuild-and-assert-worker.sh` needs to read a string literal.
e145_prepare_session() {
  local dirty
  dirty="$(git status --porcelain -- Sources Vendor Package.swift \
    Package.resolved mtp-head.manifest.json)"
  if [[ -n "${dirty}" ]]; then
    echo "e145: the scored surface is dirty; refusing to time over" \
         "uncommitted work" >&2
    echo "${dirty}" >&2
    return 1
  fi
  E145_SESSION_COMMIT="$(git rev-parse HEAD)"

  local -a build_args=(--require MLX_E145_PIN_DEPTH
                       --require MLX_E134_DEPTH_PRICE_ARM)
  [[ "${E145_NO_BUILD:-0}" == "1" ]] && build_args+=(--no-build)
  echo "=== e145 phase 0: worker build and selector assertion ==="
  senpai/rebuild-and-assert-worker.sh "${build_args[@]}" || {
    echo "e145: the worker does not carry both selectors; not timing" >&2
    return 3; }

  E145_SESSION_WORKER="$(
    shasum -a 256 "${E145_WORKER}" | awk '{print $1}')"
  echo "e145: session_commit=${E145_SESSION_COMMIT}"
  echo "e145: session_worker_sha256=${E145_SESSION_WORKER}"
  return 0
}
