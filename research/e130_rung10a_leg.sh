#!/usr/bin/env bash
# E130 rung 10a: time one arm of the wired-residency ladder.
#
#   usage: research/e130_rung10a_leg.sh TAG ARM TOKENS
#
#   ARM = none    no wiring. The shipped behaviour on this 48 GiB host, where
#                 the 96 GiB gate at Qwen36MTPBlockSession.swift returns.
#         s64     wiredLimit = active + 64 MiB. The shipped M5 formula.
#         s512    wiredLimit = active + 512 MiB. The proposed fix.
#
# ONE BINARY SERVES ALL THREE ARMS. They differ only by environment, which is
# the E62 discipline: `MLX_E130_WIRED_GATE_GIB` lowers the research gate and
# `DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB` selects the slack. Both prefixes are
# on the worker environment allowlist in `sanitizedRuntimeWorkerEnvironment`.
#
# `none` against `s512` is the POSITIVE CONTROL. If wiring the tower changes
# nothing at all on this hardware, the instrument cannot see wiring effects and
# `s64` against `s512` is uninterpretable. Report the control first.
#
# Matched across every arm, so nothing is confounded with the treatment:
#   - MLXFAST_NO_SANDBOX=1 and the outcome sink path, set even on `none`, which
#     simply writes no wiring line;
#   - MLX_MAX_MB_PER_BUFFER=512 and MLX_MAX_OPS_PER_BUFFER=50, the ranked
#     command-buffer geometry. This 48 GiB host never installs them itself,
#     because installQwenMTPFullProfileCommandBufferDefaults returns at its own
#     96 GiB gate, so the export IS the mechanism here and is not a source
#     change;
#   - DARKBLOOM_STARTUP_MEMORY_PROFILE=full, which stops the low-memory profile
#     clobbering MLX_MAX_* with 128/64 on this host.
#
# THERMAL. MLXFAST_LOCAL_COOL_GATE=0 under the standing permitted mode. The
# caller must counterbalance the arms within one session. Entry and exit GPU
# temperature are recorded here, and every result carries
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false.
#
# PAGING. Arm s512 asks the driver to wire about 24.23 GiB on a 48 GiB box.
# vm_stat swap counters are recorded on both sides of every leg; a leg that
# swaps is worthless and the caller must discard it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e130_rung10a_leg.sh TAG ARM TOKENS}"
arm="${2:?usage: e130_rung10a_leg.sh TAG ARM TOKENS}"
tokens="${3:?usage: e130_rung10a_leg.sh TAG ARM TOKENS}"

# Rung 11 generalises the arm set from {none, s64, s512} to any `s<MiB>` rung.
# 2048 MiB is the F16 bound-C hard ceiling and is refused here rather than in
# the caller, so no session order can walk past it by accident.
readonly E130_BOUND_C_CEILING_MB=2048
case "${arm}" in
  none) : ;;
  s[0-9]*)
    slack_mb="${arm#s}"
    if [[ ! "${slack_mb}" =~ ^[0-9]+$ ]]; then
      echo "e130 slack leg: malformed arm ${arm}" >&2; exit 2
    fi
    if (( slack_mb > E130_BOUND_C_CEILING_MB )); then
      echo "e130 slack leg: arm ${arm} exceeds the bound-C ceiling of" >&2
      echo "e130 slack leg: ${E130_BOUND_C_CEILING_MB} MiB. Refusing to run." >&2
      exit 2
    fi
    ;;
  *) echo "e130 slack leg: unknown arm ${arm}" >&2; exit 2 ;;
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

# Matched on every arm so the sandbox relaxation is never confounded with the
# treatment. The residency sampler stays OFF: it writes once per second from a
# background thread and would perturb the very timing this leg measures.
wired_sink="${PWD}/${out}/wired.txt"
: > "${wired_sink}"
export MLXFAST_NO_SANDBOX=1
export MLX_E130_RESIDENCY_PROBE_PATH="${wired_sink}"
unset MLX_E130_RESIDENCY_PROBE

# F18 round decomposition. `MLX_QWEN_MTP_TRACE=1` emits one `mtp-round:` line
# carrying `round_us=` and one `mtp-anchor:` line per round, so the reader can
# split a leg mean into its round-1 excess and its steady-state median.
#
# COST. Under this flag a round pays about twelve DispatchTime.now() reads,
# two formatted lines, and one `mtp-row:` line per accepted row. Against a
# ~237,000 us MTP round and a ~73,700 us serial round that is order 0.04 %,
# and it is identical on every arm, so it cancels in each arm contrast. It
# still sits inside the absolute headline, so the ladder's absolute seconds
# per token is not directly comparable with an untraced receipt.
export MLX_QWEN_MTP_TRACE=1
export MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/trace.txt"
: > "${MLX_QWEN_MTP_TRACE_PATH}"
# Never on a timed leg: draining the head chain destroys the head/verify
# overlap the round is built around.
unset MLX_QWEN_MTP_TRACE_SYNC_HEAD

if [[ "${arm}" == "none" ]]; then
  unset MLX_E130_WIRED_GATE_GIB
  unset DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB
else
  export MLX_E130_WIRED_GATE_GIB=32
  export DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB="${arm#s}"
fi

# The low-memory startup profile would clobber MLX_MAX_* with 128/64. Prove it
# is disengaged with THIS leg's worker binary and THIS leg's environment. The
# worker is launched against a nonexistent tree, so it reaches the profile and
# then fails on the missing config.json: about one second, no GPU.
profile_notice_count() {
  (
    cd .build-worker/release \
      && ./mlxfast-runtime-worker mtp-runtime-worker \
           --weights /nonexistent-e130 --mtp-head /nonexistent-e130 2>&1
  ) | grep -c "low-memory startup profile engaged"
}

notice_count="$(profile_notice_count)"
if [[ "${notice_count}" != "0" ]]; then
  echo "e130 rung10a: leg ${tag} aborted: the low-memory startup profile is" >&2
  echo "e130 rung10a: engaged and would clobber MLX_MAX_* with 128/64" >&2
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

swap_counters() {
  vm_stat | awk -F: '
    /Swapins/    { gsub(/[ .]/, "", $2); printf "swapins=%s ", $2 }
    /Swapouts/   { gsub(/[ .]/, "", $2); printf "swapouts=%s ", $2 }
    /occupied by compressor/ { gsub(/[ .]/, "", $2); printf "compressor_pages=%s", $2 }
  '
}

{
  echo "experiment=e130-rung10a"
  echo "tag=${tag}"
  echo "arm=${arm}"
  echo "tokens=${tokens}"
  echo "local_mode=--local-iterate"
  echo "wired_gate_gib=${MLX_E130_WIRED_GATE_GIB:-<shipped-96>}"
  echo "wired_slack_mb=${DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB:-<shipped-default>}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "darkbloom_startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "low_memory_notice_count=${notice_count}"
  echo "cool_gate=0"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "physmem_bytes=$(sysctl -n hw.memsize)"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "swap_entry=$(swap_counters)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "swap_exit=$(swap_counters)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # `mtp-anchor:` is the only line that carries both a pid and every phase
  # stamp, so the F18 reader joins on it alone. `mtp-trace:` has no pid, and
  # O_APPEND from several workers can interleave whole lines between the two.
  echo "trace_anchor_lines=$(grep -c '^mtp-anchor:' "${MLX_QWEN_MTP_TRACE_PATH}" 2>/dev/null || echo 0)"
  echo "trace_round_lines=$(grep -c '^mtp-trace:' "${MLX_QWEN_MTP_TRACE_PATH}" 2>/dev/null || echo 0)"
  echo "trace_row_lines=$(grep -c '^mtp-row:' "${MLX_QWEN_MTP_TRACE_PATH}" 2>/dev/null || echo 0)"
  echo "trace_pids=$(grep -o 'pid=[0-9]*' "${MLX_QWEN_MTP_TRACE_PATH}" 2>/dev/null | sort -u | tr '\n' ',')"
  if grep -q "wired-zh" "${wired_sink}" 2>/dev/null; then
    echo "wired_residency_active=true"
    echo "wired_outcome_line=$(grep 'wired-zh' "${wired_sink}" | head -1)"
    echo "wired_outcome_count=$(grep -c 'wired-zh' "${wired_sink}")"
    # F16 bound-C safety. Both counters must stay at zero on every rung.
    echo "wired_clamped_count=$(grep -c 'clamped=true' "${wired_sink}")"
    echo "wired_apply_failures=$(grep -cE 'applied=(-1|0) ' "${wired_sink}")"
  else
    echo "wired_residency_active=false"
  fi
} >> "${out}/meta.txt"

echo "=== ${tag} arm=${arm} ==="
grep -E "gpu_temp|swap_|wired_residency_active|wired_outcome_line|wired_clamped_count|wired_apply_failures|exit=" "${out}/meta.txt"
exit "${status}"
