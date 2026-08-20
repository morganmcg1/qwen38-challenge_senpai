#!/usr/bin/env bash
# Research-only driver: measure the verify-width cost curve of
# `quantized_matmul` at the scored Qwen 3.8 27B shapes, through the vendored
# MLX the scored worker links and through stock pip MLX, then summarize.
#
#   research/run-qmv-curve.sh TAG [BASE_SHA] [--widths L] [--shapes-only]
#                             [--reps N] [--inner N] [--skip-stock]
#
# Holds benchmark.sh's own local run lock for the whole measurement and passes
# benchmark.sh's own 40C cool gate before each timed process, so this never
# overlaps a model-holding run and never times a hot GPU.
set -euo pipefail

tag="${1:?usage: run-qmv-curve.sh TAG [BASE_SHA]}"
shift
base_sha=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  base_sha="${1}"
  shift
fi

sweep_widths=""
shapes_only=""
sweep_reps=""
sweep_inner=""
skip_stock=""
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --widths) sweep_widths="${2:?--widths needs a comma-separated list}"; shift 2 ;;
    --shapes-only) shapes_only=1; shift ;;
    --reps) sweep_reps="${2:?--reps needs a count}"; shift 2 ;;
    --inner) sweep_inner="${2:?--inner needs a count}"; shift 2 ;;
    --skip-stock) skip_stock=1; shift ;;
    *) echo "run-qmv-curve.sh: unknown argument ${1}" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# --- reuse benchmark.sh's local run guard -------------------------------------
# Same extraction contract as benchmark-qwen-mtp.sh: take the definitions, then
# verify each name really got defined, so a refactor of benchmark.sh fails
# closed here instead of running unguarded.
LOCAL_RUN_LOCK_OWNED=""
local_run_guard_enabled() {
  [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]
}
run_lock_definitions="$(
  awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh
  awk '/^local_run_lock_path\(\) \{/,/^\}/' benchmark.sh
  awk '/^acquire_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^release_local_run_lock\(\) \{/,/^\}/' benchmark.sh
  awk '/^list_resident_model_processes\(\) \{/,/^\}/' benchmark.sh
  awk '/^abort_if_model_already_resident\(\) \{/,/^\}/' benchmark.sh
)"
if ! eval "${run_lock_definitions}"; then
  echo "run-qmv-curve.sh: could not evaluate benchmark.sh's local run guard; refusing to run unguarded" >&2
  exit 1
fi
for reused in \
  local_run_lock_path acquire_local_run_lock release_local_run_lock \
  list_resident_model_processes abort_if_model_already_resident
do
  if ! declare -F "${reused}" >/dev/null 2>&1; then
    echo "run-qmv-curve.sh: could not reuse benchmark.sh's ${reused}(); refusing to run unguarded" >&2
    exit 1
  fi
done
if [[ -z "${RESIDENT_MODEL_PROCESS_PATTERN:-}" ]]; then
  echo "run-qmv-curve.sh: benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN is empty; refusing to run unguarded" >&2
  exit 1
fi

cleanup() {
  release_local_run_lock
}
trap cleanup EXIT

acquire_local_run_lock
abort_if_model_already_resident

out_dir="${repo_root}/.mlxfast-private/qmv-curve/${tag}"
rm -rf -- "${out_dir}"
mkdir -p "${out_dir}"

{
  echo "run-qmv-curve: tag=${tag} base_sha=${base_sha:-unset}"
  echo "run-qmv-curve: head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "run-qmv-curve: host=$(sysctl -n machdep.cpu.brand_string) mem=$(sysctl -n hw.memsize)"
  echo "run-qmv-curve: widths=${sweep_widths:-default} shapes_only=${shapes_only:-0} reps=${sweep_reps:-default} inner=${sweep_inner:-default} skip_stock=${skip_stock:-0}"
  date -u '+run-qmv-curve: started_utc=%Y-%m-%dT%H:%M:%SZ'
} | tee "${out_dir}/identity.txt" >&2

# --- proposal-head provenance -------------------------------------------------
# The advisor requires the exact head bytes behind every number, and
# setup-qwen-mtp.sh stages the organizer-pinned head by model id, not from
# mtp-head.manifest.json. Record what is actually on disk.
head_dir="${MLXFAST_QWEN_MTP_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head}"
{
  echo "{"
  printf '  "head_dir": "%s",\n' "${head_dir}"
  printf '  "repo_declares_proposal_head": %s,\n' \
    "$(compgen -G 'mtp-head/*.safetensors' >/dev/null && echo true || echo false)"
  echo '  "files": ['
  first=1
  for f in "${head_dir}"/.gitattributes "${head_dir}"/config.json \
           "${head_dir}"/model.safetensors "${head_dir}"/model.safetensors.index.json
  do
    [[ -f "${f}" ]] || continue
    [[ ${first} -eq 1 ]] || echo ","
    first=0
    printf '    {"name": "%s", "bytes": %s, "sha256": "%s"}' \
      "$(basename "${f}")" \
      "$(stat -f %z "${f}")" \
      "$(shasum -a 256 "${f}" | cut -d' ' -f1)"
  done
  echo ""
  echo "  ]"
  echo "}"
} >"${out_dir}/head-provenance.json"

# --- build the vendored-kernel bench ------------------------------------------
# Release, because the curve at the small shapes is only tens of microseconds
# and a debug host would add its own dispatch overhead to every point. Release
# builds omit `-enable-testing`, which the suite's `@testable` imports require.
swift build -c release --build-tests --force-resolved-versions -Xswiftc -enable-testing
# The xctest bundle is a third location Cmlx does not search by default; without
# this the first MLXArray fails to load the default metallib.
tools/build-mlx-metallib.sh --all-build-roots

# Read the arm back out of the bundle that is about to be timed. llbuild signs
# C and C++ inputs by content, so a rewrite with identical bytes relinks
# nothing, and a bundle left over from an earlier arm is otherwise
# indistinguishable from a correct one. A bundle carrying an `m5_rbx` routing
# was found on disk while the checkout held base bytes, so this is a real
# failure mode and not a hypothetical one. `set -e` makes it fail closed, ahead
# of the timing rather than after it.
if [[ -n "${LEG_ARM_JSON:-}" ]]; then
  xctest_bin="$(ls -1 .build/*/release/*PackageTests.xctest/Contents/MacOS/*PackageTests \
    2>/dev/null | head -1)"
  [[ -n "${xctest_bin}" ]] || {
    echo "run-qmv-curve: cannot find the built test bundle to probe" >&2; exit 2; }
  python3 research/e59_binary_probe.py "${LEG_ARM_JSON}" "${xctest_bin}" \
    | tee "${out_dir}/binary-probe.log"
fi

eval "$(
  awk '/^find_macmon\(\) \{/,/^\}/' benchmark.sh
  awk '/^local_gpu_temp\(\) \{/,/^\}/' benchmark.sh
)"
COOL_GATE_MACMON_BIN="$(find_macmon || true)"

# This host's idle GPU floor sits above benchmark.sh's 40C target, so the gate's
# stall detector aborts before reaching it. Every width in the sweep is timed
# round-robin inside one process, so a thermal floor biases all widths equally
# and cancels in the C(M)/C(1) ratio the sweep exists to measure. The gate still
# runs first; a stalled cool-down downgrades to a recorded thermal state instead
# of discarding the measurement. Same contract as run-draft-bits-sweep.sh.
cool_gate() {
  echo "run-qmv-curve: cool gate before ${1}" >&2
  if ./benchmark.sh --local-cool-gate-only; then
    echo "cool_gate_${1}=passed" | tee -a "${out_dir}/identity.txt" >&2
  else
    echo "cool_gate_${1}=stalled_above_40C" | tee -a "${out_dir}/identity.txt" >&2
  fi
  local temp
  temp="$(local_gpu_temp || true)"
  echo "gpu_temp_c_before_${1}=${temp:-unknown}" | tee -a "${out_dir}/identity.txt" >&2
  printf '%s %s\n' "${1}" "${temp:-unknown}" >>"${out_dir}/start-temps.txt"
}

# --- vendored sweep (authoritative: these are the scored kernels) --------------
cool_gate vendored
MLXFAST_RUN_QMV_COST_CURVE=1 \
MLXFAST_QMV_COST_CURVE_OUT="${out_dir}/vendored.json" \
MLXFAST_QMV_COST_CURVE_WIDTHS="${sweep_widths}" \
MLXFAST_QMV_COST_CURVE_SHAPES_ONLY="${shapes_only}" \
MLXFAST_QMV_COST_CURVE_REPS="${sweep_reps}" \
MLXFAST_QMV_COST_CURVE_INNER="${sweep_inner}" \
  swift test -c release --force-resolved-versions -Xswiftc -enable-testing \
  --filter QwenQMVCostCurveTests 2>&1 | tee "${out_dir}/vendored.log"
echo "gpu_temp_c_after_vendored=$(local_gpu_temp || true)" | tee -a "${out_dir}/identity.txt" >&2

# --- stock pip MLX control ----------------------------------------------------
python_bin="${MLXFAST_MLX_PYTHON_BIN:-/opt/homebrew/bin/python3}"
if [[ -n "${skip_stock}" ]]; then
  echo "run-qmv-curve: --skip-stock; not running the stock control" >&2
elif "${python_bin}" -c 'import mlx.core' >/dev/null 2>&1; then
  cool_gate stock
  "${python_bin}" research/qmv_cost_curve.py --out "${out_dir}/stock.json" \
    2>&1 | tee "${out_dir}/stock.log"
else
  echo "run-qmv-curve: ${python_bin} has no mlx; skipping the stock control" >&2
fi

# --- summarize ----------------------------------------------------------------
summary_python="${MLXFAST_PYTHON_BIN:-/Users/ec2-user/.senpai/aws-mac-runners/qwen38-mlx-senpai-r1/venv/bin/python3}"
[[ -x "${summary_python}" ]] || summary_python="$(command -v python3)"

# macOS ships bash 3.2, where `set -u` treats an empty array's `[@]` expansion as
# unbound. The `${a[@]+...}` guard is the only portable way to pass an optional
# flag list; without it --skip-stock kills the summary after a good measurement.
stock_flag=()
[[ -f "${out_dir}/stock.json" ]] && stock_flag=(--stock "${out_dir}/stock.json")
wandb_flag=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb_flag=(--wandb)
else
  echo "run-qmv-curve: WANDB_API_KEY unset; skipping W&B logging" >&2
fi

WANDB_PROJECT="${WANDB_PROJECT:-qwen38-mlx-challenge-senpai}" \
WANDB_ENTITY="${WANDB_ENTITY:-wandb-applied-ai-team}" \
"${summary_python}" research/qmv_cost_curve_summary.py \
  --vendored "${out_dir}/vendored.json" \
  ${stock_flag[@]+"${stock_flag[@]}"} \
  --head-provenance "${out_dir}/head-provenance.json" \
  --out "${out_dir}/summary.json" \
  --tag "${tag}" \
  --host "$(sysctl -n machdep.cpu.brand_string)" \
  --base-sha "${base_sha:-unset}" \
  --na-max "${MLXFAST_QMV_NA_MAX:-4}" \
  ${wandb_flag[@]+"${wandb_flag[@]}"} 2>&1 | tee "${out_dir}/summary.log"

date -u '+run-qmv-curve: finished_utc=%Y-%m-%dT%H:%M:%SZ' >&2
echo "run-qmv-curve: artifacts in ${out_dir}" >&2
