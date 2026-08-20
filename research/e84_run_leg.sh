#!/usr/bin/env bash
# One E84 leg: install an arm -> commit -> build -> witness -> measure -> unwind.
#
#   research/e84_run_leg.sh ARM TAG [--tokens N] [--hot] [--warmup]
#
# ARM is base | a | b | ab. research/e84_arm.py owns the arm bytes and refuses
# to write an arm that does not reproduce its reference blob.
#
# The witness matches the language of each mechanism, per the runbook. Mechanism
# A is ordinary Swift, so it witnesses in the SYMBOL table through
# `--require-symbol` / `--forbid-symbol`. Mechanism B adds a Metal JIT source
# string, so its kernel name witnesses in the STRING table through `--require` /
# `--forbid`. The worker is asserted before and after the measured run, and a
# digest that moves between the two reads invalidates the leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

arm="${1:?usage: e84_run_leg.sh ARM TAG [--tokens N] [--hot] [--warmup]}"
tag="${2:?usage: e84_run_leg.sh ARM TAG [--tokens N] [--hot] [--warmup]}"
shift 2

tokens="${E84_TOKENS:-64}"
hot=0
warmup=0
build_only=0
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --hot) hot=1; shift ;;
    --warmup) warmup=1; shift ;;
    # Prove the arm compiles and witnesses before spending a timed session.
    --build-only) build_only=1; shift ;;
    *) echo "e84_run_leg: unknown argument $1" >&2; exit 2 ;;
  esac
done

readonly SCORED_FILE="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
# Mechanism A's Swift entry point and mechanism B's Metal kernel name. Both are
# absent from `base` and present in `ab`, so every leg proves its own arm.
readonly A_SYMBOL="islandFastPathReady"
readonly B_STRING="qwen35_gated_delta_replay_state"

base_sha="${E84_BASE_SHA:-07c75a708c2347021d3148d7bc87b246ba2aec73}"
root="${E84_ROOT:-${repo_root}/.mlxfast-private/e84}"
fixture="${E84_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
head_dir="${E84_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

unwind() {
  if [[ -n "${transient_sha}" && "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
    git reset -q "${pre_patch_sha}"
  fi
  git checkout -q "${pre_patch_sha}" -- "${SCORED_FILE}" 2>/dev/null || true
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e84_run_leg: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi
[[ -s "${fixture}" ]] || { echo "e84_run_leg: missing fixture ${fixture}" >&2; exit 2; }
[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] \
  || { echo "e84_run_leg: declared head tree missing at ${head_dir}" >&2; exit 2; }

out="${root}/runs/${tag}"
rm -rf "${out}"; mkdir -p "${out}/reports"

macmon_bin="${MLXFAST_MACMON_BIN:-}"
[[ -n "${macmon_bin}" ]] || macmon_bin="$(command -v macmon 2>/dev/null || true)"
[[ -n "${macmon_bin}" ]] || macmon_bin="${HOME}/bin/macmon"
gpu_temp() {
  [[ -x "${macmon_bin}" ]] || { echo ""; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
}

if ((build_only == 0)); then
  python3 research/e49_gpu_gate.py --samples 5 --out "${out}/gpu-gate.json"
  case "$?" in
    0) echo "e84_run_leg: GPU gate idle" >&2 ;;
    1) echo "e84_run_leg: GPU gate reports BUSY; not timing." >&2; exit 3 ;;
    *) echo "e84_run_leg: GPU utilization counter unavailable; not timing blind." >&2; exit 4 ;;
  esac
fi

python3 research/e84_arm.py "${arm}" --base "${base_sha}" --tip "${pre_patch_sha}" \
  > "${out}/arm.log" 2>&1 || {
  echo "e84_run_leg: arm ${arm} could not be materialised" >&2
  cat "${out}/arm.log" >&2
  exit 2
}
cp "${out}/arm.log" "${out}/arm.txt"

if [[ "${arm}" == "ab" ]] && ! git diff --quiet -- "${SCORED_FILE}"; then
  echo "e84_run_leg: arm ab differs from the branch tip; not a tip control" >&2
  exit 2
fi

git add -- "${SCORED_FILE}"
git commit -q --allow-empty -m "E84 leg ${tag}: TRANSIENT ${arm} arm bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical between legs. This commit exists
only so the bytes the compiler saw are reachable while the leg runs."
transient_sha="$(git rev-parse HEAD)"

mkdir -p .build/clang-module-cache .build-worker/clang-module-cache

# The worker rebuild, the CLI rebuild and the arm witness in one step, so no leg
# can time a stale worker. `--require`/`--forbid` read strings; `*-symbol` read
# `nm -a`.
witness=()
if [[ "${arm}" == "a" || "${arm}" == "ab" ]]; then
  witness+=(--require-symbol "${A_SYMBOL}")
else
  witness+=(--forbid-symbol "${A_SYMBOL}")
fi
if [[ "${arm}" == "b" || "${arm}" == "ab" ]]; then
  witness+=(--require "${B_STRING}")
else
  witness+=(--forbid "${B_STRING}")
fi
senpai/rebuild-and-assert-worker.sh "${witness[@]}" \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e84_run_leg: worker assert failed before the leg" >&2
  tail -40 "${out}/worker-assert-pre.txt" >&2
  exit 5
}

if ((build_only)); then
  echo "e84_run_leg: ${tag} (${arm}) built and witnessed; --build-only, not timing" >&2
  grep -E '^(ok|worker_sha256|worker_mtime) ' "${out}/worker-assert-pre.txt" >&2
  exit 0
fi

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
  echo "head_provenance_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "head_run_dir_tree_sha256=$(python3 research/e59_head_tree_digest.py "${head_dir}")"
  echo "scored_source_sha256=$(shasum -a 256 "${SCORED_FILE}" | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "worker_sha256=$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
  echo "worker_mtime=$(awk '/^worker_mtime /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
  echo "metallib_sha256=$(shasum -a 256 .build-worker/release/mlx.metallib | cut -d' ' -f1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  echo "cool_gate_requested=$((1 - hot))"
  echo "cool_gate_passed_real_gate=$( ((hot)) && echo false || echo true )"
  echo "gate_qualified_for_timing=$( ((hot)) && echo false || echo true )"
  echo "startup_memory_profile=${DARKBLOOM_STARTUP_MEMORY_PROFILE}"
  echo "mlx_max_mb_per_buffer=${MLX_MAX_MB_PER_BUFFER}"
  echo "mlx_max_ops_per_buffer=${MLX_MAX_OPS_PER_BUFFER}"
  echo "physical_memory_gib=$(( $(sysctl -n hw.memsize) >> 30 ))"
  echo "host_chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ -n "${LEG_EXTRA_META:-}" ]] && printf '%s\n' "${LEG_EXTRA_META}"
} > "${out}/meta.txt"

export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${fixture}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${out}/score.json"
export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"
export MLXFAST_CAPTURE_DIR="${out}/reports"
((hot)) && export MLXFAST_LOCAL_COOL_GATE=0

# Per-round records for the paired-median analysis. The path must be per leg:
# the sink is opened O_APPEND once per process, so a shared path would
# interleave every leg into one file with no way to attribute a round.
#
# MLX_QWEN_MTP_TRACE_SYNC_HEAD is deliberately NOT set. It drains the head
# chain before the verify-build window, which is what makes it useful for
# attributing draft cost, but its own doc comment records that it destroys the
# head/verify overlap the round is built around. That would change round_us,
# which is the quantity being compared here.
if [[ "${E84_TRACE:-0}" == "1" ]]; then
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${out}/round-trace.txt"
fi

./benchmark-qwen-mtp.sh --local-iterate > "${out}/run.log" 2>&1
rc=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "wrapper_exit=${rc}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

senpai/rebuild-and-assert-worker.sh --no-build "${witness[@]}" \
  > "${out}/worker-assert-post.txt" 2>&1
post_rc=$?
post_sha="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"
pre_sha="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
echo "worker_sha256_post=${post_sha}" >> "${out}/meta.txt"
echo "worker_assert_post_exit=${post_rc}" >> "${out}/meta.txt"
if ((post_rc != 0)) || [[ "${post_sha}" != "${pre_sha}" ]]; then
  echo "e84_run_leg: ${tag} worker changed or failed its post-assert; discarding" >&2
  rc=7
fi

stale_metal="$(grep -c 'built from different vendored Metal sources' "${out}/run.log")"
echo "stale_metallib_warnings=${stale_metal}" >> "${out}/meta.txt"
if ((stale_metal > 0)); then
  echo "e84_run_leg: ${tag} saw ${stale_metal} stale-metallib warnings; discarding" >&2
  rc=6
fi

echo "warmup_discarded=${warmup}" >> "${out}/meta.txt"
echo "status=${rc}" >> "${out}/meta.txt"

if ((rc == 0)); then
  python3 research/e84_wandb_log.py --leg "${out}" >> "${out}/wandb.log" 2>&1 \
    || echo "e84_run_leg: W&B logging failed for ${tag}; see ${out}/wandb.log" >&2
fi

exit "${rc}"
