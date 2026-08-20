#!/usr/bin/env bash
# One E59 rung 4 end-to-end leg: patch -> commit -> build -> measure -> unwind.
#
#   research/e59_e2e_run.sh ARM TAG [--tokens N] [--hot] [--ops N] [--warmup]
#
# The arm patch is committed while it is compiled, so every timed leg records
# dirty=0 and names an exact commit, and the commit is unwound on every exit
# path. The branch's scored surface therefore stays byte-identical to the
# campaign base between legs.
#
# The crossrow kernel ships as a source string inside mlx-generated/quantized.cpp,
# a C++ translation unit rather than the metallib, so an arm needs a real Swift
# rebuild of BOTH build roots. `benchmark-qwen-mtp.sh --local-iterate` uses
# prebuilt binaries and does not build them.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

arm="${1:?usage: e59_e2e_run.sh ARM TAG [--tokens N] [--hot]}"
tag="${2:?usage: e59_e2e_run.sh ARM TAG [--tokens N] [--hot]}"
shift 2

tokens="${E59_TOKENS:-512}"
hot=0
warmup=0
ops_override=""
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --hot) hot=1; shift ;;
    # Declared in advance and excluded from the analysis. A cold first leg does
    # not cancel in a palindrome, so the session pays for one throwaway leg
    # instead of carrying a cold leg into the counterbalanced set.
    --warmup) warmup=1; shift ;;
    # Command-buffer geometry dose. Only the geometry proof uses this; every
    # timed leg runs at the ranked 50.
    --ops) ops_override="$2"; shift 2 ;;
    *) echo "e59_e2e_run: unknown argument $1" >&2; exit 2 ;;
  esac
done

readonly SCORED_FILES=(
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
)
base_sha="${E59_BASE_SHA:-$(git rev-parse origin/senpai/qwen38-mtp-r1)}"
arms_module="${E59_ARMS_MODULE:-research/e59_arms.py}"
root="${E59_E2E_ROOT:-${repo_root}/.mlxfast-private/e59-e2e}"
fixture="${E59_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
head_dir="${E59_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"

# benchmark.sh derives the run lock from $HOME, which differs per role while the
# uid does not, so the default lock gives zero mutual exclusion against another
# student timing on the same box. The shared parent restores it for both.
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"

# Ranked command-buffer geometry.
#
# `MLX_MAX_*_PER_BUFFER` alone does NOT reach MLX on this host. The exports do
# survive the worker's environment sanitizer, but
# `applyQwenMTPStartupMemoryProfile` (QwenRuntimeMTPWorker.swift:479) runs
# before the first MLX device access and force-sets both names with
# `overwrite=1` whenever the resolved profile is the low-memory one. This host
# has 48 GiB, below the 64 GiB full-profile minimum, so the low-memory profile
# is selected by default and MLX would see 128 MiB / 64 ops no matter what the
# parent exports.
#
# `DARKBLOOM_STARTUP_MEMORY_PROFILE=full` is the lever that works. It makes
# `resolve` return the full profile, so the worker's `guard policy.isLowMemory
# else { return }` returns before either force-set and the exported values
# survive.
#
# The exports are also the ONLY source of the ranked values here.
# `installQwenMTPFullProfileCommandBufferDefaults`
# (RuntimeStartupMemoryPolicy.swift:63) opens with
# `guard physicalMemoryBytes >= 96 GiB else { return }`, so on a 48 GiB host it
# installs nothing and warns about nothing. The ranked 128 GiB box passes that
# guard and force-sets exactly 512 and 50 for itself. Same resolved state, two
# different routes to it.
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${ops_override:-${MLX_MAX_OPS_PER_BUFFER:-50}}"

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

unwind() {
  if [[ -n "${transient_sha}" ]]; then
    if [[ "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
      git reset -q "${pre_patch_sha}"
    else
      echo "e59_e2e_run: HEAD moved during the leg; restoring files only" >&2
    fi
  fi
  git checkout -q "${pre_patch_sha}" -- "${SCORED_FILES[@]}" 2>/dev/null || true
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e59_e2e_run: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi
if ! git diff --quiet "${base_sha}" -- "${SCORED_FILES[@]}"; then
  echo "e59_e2e_run: scored kernel files differ from ${base_sha}; refusing to stack patches" >&2
  exit 1
fi
[[ -s "${fixture}" ]] || { echo "e59_e2e_run: missing fixture ${fixture}" >&2; exit 2; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] \
  || { echo "e59_e2e_run: declared head tree missing at ${head_dir}" >&2; exit 2; }

out="${root}/runs/${tag}"
rm -rf "${out}"; mkdir -p "${out}/reports"

# benchmark.sh's own `find_macmon` prefers an override, then PATH, then the
# usual install locations, which is why its cool gate reads temperatures here
# while this leg record did not: `${HOME}/bin/macmon` is the ranked boxes'
# layout and does not exist in this role home. macmon is on PATH at
# /opt/homebrew/bin/macmon. Search the same order so entry and exit temperature
# are recorded for every leg.
macmon_bin="${MLXFAST_MACMON_BIN:-}"
if [[ -z "${macmon_bin}" ]]; then
  macmon_bin="$(command -v macmon 2>/dev/null || true)"
fi
[[ -n "${macmon_bin}" ]] || macmon_bin="${HOME}/bin/macmon"
gpu_temp() {
  [[ -x "${macmon_bin}" ]] || { echo ""; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
}

# --- who else is on this GPU --------------------------------------------------
gate_json="${out}/gpu-gate.json"
python3 research/e49_gpu_gate.py --samples 5 --out "${gate_json}"
case "$?" in
  0) echo "e59_e2e_run: GPU gate idle" >&2 ;;
  1) echo "e59_e2e_run: GPU gate reports BUSY; not timing." >&2; exit 3 ;;
  *) echo "e59_e2e_run: GPU utilization counter unavailable; not timing blind." >&2; exit 4 ;;
esac

# --- patch --------------------------------------------------------------------
python3 "${arms_module}" "${arm}" --out "${out}/arm.json" || exit 2
python3 research/twin_audit.py quantized || exit 2

# The kernel ships as a source string inside a C++ translation unit, so a Metal
# syntax or template-arity error survives `swift build` and only surfaces as a
# JIT failure minutes into the run. quantized.metal carries the
# instantiate_quantized_* macros, so compiling it exercises every width case.
if ! xcrun -sdk macosx metal -std=metal3.1 -O2 -c \
    Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal \
    -I Vendor/mlx-swift/Source/Cmlx/mlx -o /dev/null 2>"${out}/metalcc.err"; then
  echo "e59_e2e_run: quantized.metal does not compile at ${arm}; not spending a run" >&2
  grep 'error:' "${out}/metalcc.err" | head -5 >&2
  exit 2
fi

git add -- "${SCORED_FILES[@]}"
git commit -q --allow-empty -m "E59 leg ${tag}: TRANSIENT ${arm} arm bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical to ${base_sha}. This commit exists
only so the bytes the compiler saw are reachable while the leg runs."
transient_sha="$(git rev-parse HEAD)"

# --- build both roots ---------------------------------------------------------
status=0
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift \
  > "${out}/build-cli.log" 2>&1 || status=1
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker \
  > "${out}/build-worker.log" 2>&1 || status=1
if ((status == 0)); then
  tools/build-mlx-metallib.sh --all-build-roots > "${out}/build-metallib.log" 2>&1 || status=1
fi
# Content, not mtime: read this arm's routing back out of the image that will
# be timed. The QMV family is JIT-compiled from a source string inside
# mlx-generated/quantized.cpp, so every width's dispatch call is linked into
# the worker verbatim and the binary can be asked directly which arm it is.
#
# An mtime comparison cannot do this job. llbuild signs C and C++ inputs by
# content, so rewriting a file with identical bytes correctly relinks nothing
# and leaves a product older than its sources. That is exactly the `shipped`
# arm, whose bytes equal the base bytes by construction.
#
# Only the worker is probed. mlxfast-swift embeds no Metal text at all; it is a
# driver that spawns the worker, so its build status is the whole of its
# contribution here.
if ((status == 0)); then
  python3 research/e59_binary_probe.py "${out}/arm.json" \
    .build-worker/release/mlxfast-runtime-worker \
    > "${out}/binary-probe.log" 2>&1 || status=1
  cat "${out}/binary-probe.log"
fi
((status == 0)) || { echo "e59_e2e_run: build failed for ${arm}" >&2; exit 5; }

# --- measure ------------------------------------------------------------------
{
  echo "tag=${tag}"
  echo "arm=${arm}"
  echo "base_sha=${base_sha}"
  echo "measured_commit_unwound=${transient_sha}"
  echo "branch_commit=${pre_patch_sha}"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "tokens=${tokens}"
  echo "offered_depth=${MLXFAST_QWEN_MTP_DEPTH:-8}"
  echo "fixture=${fixture}"
  echo "fixture_sha256=$(shasum -a 256 "${fixture}" | cut -d' ' -f1)"
  echo "head_dir=${head_dir}"
  # Three numbers, because they answer three different questions and one of
  # them was already mistaken for another. The file digest identifies the
  # weights. The single-file tree digest is the only one comparable with
  # mtp-head.manifest.json, whose `sha256` is a TREE digest under the rule in
  # QwenMTPHeadDeclaration.swift:181-233. The run-directory tree digest differs
  # from both, because this directory adds the config.json the local loader
  # needs, and it is what the worker reports as head_provenance_sha256.
  echo "head_safetensors_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "head_manifest_declared_sha256=$(python3 -c 'import json,sys;print(json.load(open("mtp-head.manifest.json"))["sha256"])' 2>/dev/null)"
  echo "head_single_file_tree_sha256=$(python3 research/e59_head_tree_digest.py "${head_dir}" --only model.safetensors)"
  echo "head_run_dir_tree_sha256=$(python3 research/e59_head_tree_digest.py "${head_dir}")"
  echo "twin_digests=$(shasum -a 256 "${SCORED_FILES[@]}" | awk '{printf "%s ", $1}')"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
  echo "worker_binary_probe=$(head -1 "${out}/binary-probe.log" | cut -d' ' -f1-3)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "cool_gate_requested=$((1 - hot))"
  echo "startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "physical_memory_gib=$(( $(sysctl -n hw.memsize) >> 30 ))"
  # `wireResidentWeightsIfEnabled` (Qwen36MTPBlockSession.swift:222-226) needs
  # the opt-out unset and >= 96 GiB of physical memory. The ranked 128 GiB box
  # wires the weights; a 48 GiB host cannot, so the two hosts decode from
  # different residency states and this flag has to travel with every leg.
  echo "wired_residency_active=$(
    if [[ "${DARKBLOOM_QWEN_MTP_WIRED_ZH:-}" != "0" ]] \
      && (( $(sysctl -n hw.memsize) >= (96 << 30) )); then
      echo true
    else
      echo false
    fi)"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${fixture}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${out}/score.json"
export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
export MLXFAST_CAPTURE_DIR="${out}/reports"
((hot)) && export MLXFAST_LOCAL_COOL_GATE=0

./benchmark-qwen-mtp.sh --local-iterate > "${out}/run.log" 2>&1
rc=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "wrapper_exit=${rc}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

# A metallib built from another arm's sources would silently time the wrong
# kernel, so this is a hard failure rather than a warning.
stale_metal="$(grep -c 'built from different vendored Metal sources' "${out}/run.log")"
echo "stale_metallib_warnings=${stale_metal}" >> "${out}/meta.txt"
if ((stale_metal > 0)); then
  echo "e59_e2e_run: ${tag} saw ${stale_metal} stale-metallib warnings; discarding the leg" >&2
  rc=6
fi

# WITHDRAWN: grepping this log for `low-memory startup profile` was not a
# check. The worker writes that notice to its own stderr
# (QwenRuntimeMTPWorker.swift:504). `mtp-timed` builds its worker options at
# main.swift:1799 without `forwardsWorkerStderr`, the default is `false`
# (QwenRuntime.swift:304), and the drain then installs a swallowing emitter,
# `emit: options.forwardsWorkerStderr ? nil : { _ in }`
# (QwenRuntimeWorker.swift:1985). The only call site that can enable forwarding
# is the `dflash` subcommand at main.swift:1409, which a Qwen MTP leg never
# reaches. The notice could never appear, so a zero count meant nothing and the
# `rc=7` gate could never fire.
#
# The replacement is two falsifiable proofs, both outside this script:
#   1. `research/e59_geometry_proof.sh` launches one worker with
#      `DARKBLOOM_STARTUP_MEMORY_PROFILE=bogus`, which must crash at the
#      `preconditionFailure` in `resolve` (RuntimeStartupMemoryPolicy.swift:104).
#      A worker that starts at all under `=full` therefore resolved to the full
#      profile, because `case "full"` sets `lowMemory = false` unconditionally.
#   2. The same script times one leg at `--ops 8` and one at `--ops 50`. If the
#      export reaches MLX the small cap must be materially slower, because MLX
#      latches the value once (utils.h:178) and commits a command buffer every
#      N operations against a round that issues roughly a thousand.
# Both proofs are session evidence, so record the geometry the leg requested and
# let the proof decide whether the request landed.
echo "startup_memory_profile_resolves_full=1" >> "${out}/meta.txt"
echo "geometry_proof=research/e59-artifacts/e59-geometry-proof.json" \
  >> "${out}/meta.txt"
echo "warmup_discarded=${warmup}" >> "${out}/meta.txt"

echo "status=${rc}" >> "${out}/meta.txt"

# Log while measuring, never once at session end: a session that dies on leg 4
# must still leave legs 1-3 on the board.
if ((rc == 0)); then
  python3 research/e59_wandb_log.py \
    --stage "${E59_LEG_STAGE:-rung4-leg}" --leg "${out}" \
    >> "${out}/wandb.log" 2>&1 \
    || echo "e59_e2e_run: W&B logging failed for ${tag}; see ${out}/wandb.log" >&2
fi

exit "${rc}"
