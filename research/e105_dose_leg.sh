#!/usr/bin/env bash
# E105 rung 1/2 -- one in-situ dispatch-boundary dose leg.
#
#   usage: research/e105_dose_leg.sh TAG DOSE SHAPE
#
#   TAG    output directory under research/out/
#   DOSE   extra dependent dispatches injected per decoder layer, 0 disables
#   SHAPE  `tiny` (grid 1x1x1, 1 threadgroup) or `prework`
#          (grid 32x5x80 tg 32x1x1, 400 threadgroups, the live
#          `qwen35_packed_gdn_prework` width)
#
# WHY A DOSE LADDER AND NOT A METAL HARNESS. E105 rung 0 showed the three
# target families are already one dispatch per layer, so the whole fusion
# prize is bounded by (dispatches removed) x F, where F is the marginal cost
# of one dispatch. A synthetic harness measures F in an isolated frame and
# then needs the 1.65x-2.59x isolation discount to reach the in-situ answer.
# A dose ladder measures F in situ directly and needs no discount at all.
#
# The instrument is `e105ApplyDispatchDose` in
# `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`. It inserts DOSE
# dependent dispatches per decoder layer into the decode residual chain and
# nothing else. 64 decoder layers, so a round carries 64 x DOSE extra
# dispatches. The chained value is identically zero, so `all_tokens_matched`
# must still hold and any leg where it does not is void.
#
# Finding 28. `benchmark-qwen-mtp.sh` never rebuilds the worker, so the arm is
# asserted inside `.build-worker/release/mlxfast-runtime-worker` before the
# leg. The needle is the Metal kernel NAME, which is a real string literal in
# the JIT source, so `--require` on the string table is the correct witness.
# No bracket characters: a bracket is a regex character class and would make
# the guard pass on a stale build.
#
# MLXFAST_LOCAL_COOL_GATE=0 is the standing permitted local measurement mode.
# This host asymptotes at 40.55 C and cannot reach the 40 C gate. The arms are
# counterbalanced within one session, entry and exit GPU temperature are
# recorded per leg, and the honesty flags below stay false. Directional causal
# evidence only, never a ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e105_dose_leg.sh TAG DOSE SHAPE}"
dose="${2:?usage: e105_dose_leg.sh TAG DOSE SHAPE}"
shape="${3:?usage: e105_dose_leg.sh TAG DOSE SHAPE}"

case "${shape}" in
  tiny|prework) ;;
  *) echo "e105_dose_leg: unknown shape '${shape}'" >&2; exit 2 ;;
esac

out_dir="research/out/${tag}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}"

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

senpai/rebuild-and-assert-worker.sh \
  --require 'e105_dispatch_dose_probe' \
  --require-symbol 'e105ApplyDispatchDose' \
  > "${out_dir}/worker-pre.txt" 2>&1
status=$?
cat "${out_dir}/worker-pre.txt"
if [[ "${status}" -ne 0 ]]; then
  echo "e105_dose_leg: refusing to time; the worker does not carry the arm." >&2
  exit 3
fi

worker=".build-worker/release/mlxfast-runtime-worker"
worker_mtime_pre="$(date -u -r "${worker}" +%Y-%m-%dT%H:%M:%SZ)"
worker_sha_pre="$(shasum -a 256 "${worker}" | awk '{print $1}')"

head_dir="${E85_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e105_dose_leg: no head at ${head_dir}" >&2
  exit 1
fi

entry_c="$(gpu_temp)"
start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

depth="${MLXFAST_QWEN_MTP_DEPTH:-8}"
tokens="${MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS:-64}"

MLX_E105_DOSE="${dose}" \
MLX_E105_DOSE_SHAPE="${shape}" \
MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}" \
MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}" \
MLXFAST_NO_SANDBOX=1 \
MLXFAST_LOCAL_COOL_GATE=0 \
MLXFAST_QWEN_MTP_DEPTH="${depth}" \
MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}" \
MLXFAST_SCORE_PATH="${PWD}/${out_dir}/score.json" \
./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee "${out_dir}/run.log"
status="${PIPESTATUS[0]}"

exit_c="$(gpu_temp)"

{
  echo "experiment=e105-latency-class-dispatch-family"
  echo "leg_kind=in-situ-dispatch-dose-ladder"
  echo "harness=local"
  echo "dose_per_layer=${dose}"
  echo "dose_shape=${shape}"
  echo "decoder_layers=64"
  echo "extra_dispatches_per_target_forward=$((dose * 64))"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty_build=$(git status --porcelain -- Sources Vendor Package.swift \
    Package.resolved tools mtp-head.manifest.json | wc -l | tr -d ' ')"
  echo "worker_mtime_pre=${worker_mtime_pre}"
  echo "worker_sha256_pre=${worker_sha_pre}"
  echo "worker_mtime_post=$(date -u -r "${worker}" +%Y-%m-%dT%H:%M:%SZ)"
  echo "worker_sha256_post=$(shasum -a 256 "${worker}" | awk '{print $1}')"
  echo "worker_dose_probe=$(strings -a "${worker}" | grep -c 'e105_dispatch_dose_probe')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "gpu_cores=20"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "head_dir=${head_dir}"
  echo "gpu_temp_entry_c=${entry_c}"
  echo "gpu_temp_exit_c=${exit_c}"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "offered_depth=${depth}"
  echo "decode_tokens=${tokens}"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
jq -r '.metrics | "mtp_spt=\(.mtp_seconds_per_token) serial_spt=\(.serial_seconds_per_token) speedup=\(.mtp_decode_speedup) mean_draft=\(.effective_mean_draft_len) matched=\(.all_tokens_matched)"' \
  "${out_dir}/score.json" 2>/dev/null || true
exit "${status}"
