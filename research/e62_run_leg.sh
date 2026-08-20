#!/usr/bin/env bash
# Run one E62 leg of ./benchmark-qwen-mtp.sh.
#
# usage:
#   research/e62_run_leg.sh TAG ARM TOKENS [options]
#
# options:
#   --mb N            MLX_MAX_MB_PER_BUFFER      (default 512, the ranked value)
#   --ops N           MLX_MAX_OPS_PER_BUFFER     (default 50, the ranked value)
#   --wired on|off    DARKBLOOM_QWEN_MTP_WIRED_ZH (default off)
#   --wired-fraction F DARKBLOOM_QWEN_MTP_WIRED_ZH_FRACTION
#   --census          turn the E58 in-process dispatch census on. Only valid on
#                     the census binary, and never valid in a timed contrast.
#   --submit          use --local-submit instead of --local-iterate
#   --label TEXT      free-text arm label carried into meta.txt and W&B
#
# THE MEASUREMENT, STATED PLAINLY
#
# On the ranked 128 GiB M5 the editable setenv at
# RuntimeStartupMemoryPolicy.swift:75-76 fires with overwrite=1 and installs
# MLX_MAX_MB_PER_BUFFER / MLX_MAX_OPS_PER_BUFFER itself. On this 48 GiB host
# installQwenMTPFullProfileCommandBufferDefaults returns at its 96 GiB gate, so
# that setenv never runs and the shell export below IS the whole mechanism.
# A shell export here therefore emulates the ranked source constant exactly, but
# the source edit itself is NOT locally observable. Never report an export as a
# measured source change.
#
# DARKBLOOM_STARTUP_MEMORY_PROFILE=full is required, not cosmetic. Under the
# default "auto" profile this 48 GiB host resolves lowMemory = true,
# applyQwenMTPStartupMemoryProfile passes its guard and setenv()s 128/64 with
# overwrite=1, which would clobber the export. `=full` makes the guard return.
#
# MLXFAST_LOCAL_COOL_GATE=0 is set. Every session is palindrome-counterbalanced,
# entry and exit GPU temperature are recorded per leg, and every result carries
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e62_run_leg.sh TAG ARM TOKENS [options]}"
arm="${2:?usage: e62_run_leg.sh TAG ARM TOKENS [options]}"
tokens="${3:?usage: e62_run_leg.sh TAG ARM TOKENS [options]}"
shift 3

mb=512
ops=50
wired=off
wired_fraction=
census=0
mode=--local-iterate
label=
while (($#)); do
  case "$1" in
    --mb) mb="$2"; shift 2 ;;
    --ops) ops="$2"; shift 2 ;;
    --wired) wired="$2"; shift 2 ;;
    --wired-fraction) wired_fraction="$2"; shift 2 ;;
    --census) census=1; shift ;;
    --submit) mode=--local-submit; shift ;;
    --label) label="$2"; shift 2 ;;
    *) echo "e62_run_leg.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [[ "${wired}" == "on" && "${arm}" != "wired" ]]; then
  echo "e62: --wired on needs the throwaway 'wired' binary" >&2
  exit 2
fi
if ((census)) && [[ "${arm}" != "census" ]]; then
  echo "e62: --census needs the 'census' binary" >&2
  exit 2
fi

bin_dir="research/out/e62/bin/${arm}"
for product in mlxfast-swift mlxfast-runtime-worker; do
  if [[ ! -x "${bin_dir}/${product}" ]]; then
    echo "e62: missing prebuilt ${product} for arm ${arm}" >&2
    exit 1
  fi
done

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

cp "${bin_dir}/mlxfast-swift" .build/release/mlxfast-swift
cp "${bin_dir}/mlxfast-runtime-worker" .build-worker/release/mlxfast-runtime-worker

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export DARKBLOOM_STARTUP_MEMORY_PROFILE=full
export MLX_MAX_MB_PER_BUFFER="${mb}"
export MLX_MAX_OPS_PER_BUFFER="${ops}"
export MLXFAST_LOCAL_COOL_GATE=0

# The wired-residency file sink and the census both write from inside the
# worker, whose generated sandbox denies file-write*. Both wiring arms set this,
# so the relaxation is matched across the rung 1b contrast rather than confounded
# with it.
wired_sink="${PWD}/${out}/wired.txt"
if [[ "${arm}" == "wired" ]]; then
  export MLXFAST_NO_SANDBOX=1
  : > "${wired_sink}"
  export MLX_E62_WIRED_OUT="${wired_sink}"
  export DARKBLOOM_QWEN_MTP_WIRED_ZH=$([[ "${wired}" == "on" ]] && echo 1 || echo 0)
  [[ -n "${wired_fraction}" ]] \
    && export DARKBLOOM_QWEN_MTP_WIRED_ZH_FRACTION="${wired_fraction}"
fi

census_path="${PWD}/${out}/census.jsonl"
if ((census)); then
  export MLXFAST_NO_SANDBOX=1
  : > "${census_path}"
  export MLX_E58_DISPATCH_CENSUS=1
  export MLX_E58_DISPATCH_CENSUS_PATH="${census_path}"
fi

# Runtime control for the profile clobber, run with THIS leg's worker binary and
# THIS leg's environment. `mtp-timed` builds its worker options without
# forwardsWorkerStderr, so the notice can never reach wrapper.err on a timed leg;
# launching the worker directly against a nonexistent tree reaches
# applyQwenMTPStartupMemoryProfile and then fails on the missing config.json,
# which costs about one second and touches no GPU.
profile_notice_count() {
  (
    cd .build-worker/release \
      && ./mlxfast-runtime-worker mtp-runtime-worker \
           --weights /nonexistent-e62 --mtp-head /nonexistent-e62 2>&1
  ) | grep -c "low-memory startup profile engaged"
}

notice_count="$(profile_notice_count)"
if [[ "${notice_count}" != "0" ]]; then
  echo "e62: leg ${tag} aborted: the worker still engages the low-memory" >&2
  echo "e62: startup profile, so it would clobber MLX_MAX_* with 128/64" >&2
  exit 1
fi

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

{
  echo "tag=${tag}"
  echo "arm=${arm}"
  echo "label=${label:-${arm}}"
  echo "tokens=${tokens}"
  echo "local_mode=${mode}"
  echo "darkbloom_startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "wired_residency_requested=${wired}"
  echo "wired_residency_fraction=${wired_fraction:-<default-1.0>}"
  echo "census=${census}"
  echo "low_memory_notice_count=${notice_count}"
  echo "low_memory_notice_count_under_auto=$(
    DARKBLOOM_STARTUP_MEMORY_PROFILE=auto profile_notice_count
  )"
  echo "cool_gate=0"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "staged_cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "staged_worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  cat "${bin_dir}/provenance.txt"
} > "${out}/meta.txt"

{
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

./benchmark-qwen-mtp.sh "${mode}" > "${out}/wrapper.out" 2> "${out}/wrapper.err" &
bench_pid=$!

# One-shot proof that this leg's geometry reached the worker PROCESS, bounded to
# the load phase so it cannot touch either timed window.
(
  for _ in $(seq 1 30); do
    worker_pid="$(pgrep -f 'mlxfast-runtime-worker' | head -1)"
    if [[ -n "${worker_pid}" ]]; then
      ps eww -o command= -p "${worker_pid}" 2>/dev/null \
        | tr ' ' '\n' \
        | grep -E '^(MLX_(MAX|METAL|E62)|DARKBLOOM_)' > "${out}/worker-env.txt"
      if [[ -s "${out}/worker-env.txt" ]]; then exit 0; fi
    fi
    sleep 3
  done
) &
env_probe_pid=$!

# Peak worker resident set. `Memory.peakMemory` is only reachable through the
# worker's phase_diagnostics request and QwenRuntimeMTPDriver never issues it,
# so the MTP path prints no peak_ram_gb at all. This external sampler runs on
# every leg, so its cost is matched across arms and cancels in every contrast.
(
  peak_kb=0
  : > "${out}/rss-trace.txt"
  while kill -0 "${bench_pid}" 2>/dev/null; do
    for pid in $(pgrep -f 'mlxfast-runtime-worker'); do
      rss="$(ps -o rss= -p "${pid}" 2>/dev/null | tr -d ' ')"
      if [[ -n "${rss}" ]]; then
        echo "$(date -u +%H:%M:%S) ${pid} ${rss}" >> "${out}/rss-trace.txt"
        if ((rss > peak_kb)); then peak_kb="${rss}"; fi
      fi
    done
    sleep 2
  done
  echo "${peak_kb}" > "${out}/peak-rss-kb.txt"
) &
rss_probe_pid=$!

wait "${bench_pid}"
status=$?
wait "${env_probe_pid}" 2>/dev/null
wait "${rss_probe_pid}" 2>/dev/null

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  peak_kb="$(cat "${out}/peak-rss-kb.txt" 2>/dev/null || echo 0)"
  echo "worker_peak_rss_gb=$(
    awk -v kb="${peak_kb}" 'BEGIN { printf "%.3f", kb / 1048576 }'
  )"
  echo "worker_peak_rss_source=external-ps-sampler-2s"
  if [[ -s "${wired_sink}" ]]; then
    echo "wired_residency_active=true"
    echo "wired_outcome_line=$(tr '\n' ' ' < "${wired_sink}")"
  else
    echo "wired_residency_active=false"
  fi
} >> "${out}/meta.txt"
exit "${status}"
