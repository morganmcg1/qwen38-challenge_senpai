#!/usr/bin/env bash
# One gated E162 timing leg, run against a NAMED prebuilt worker binary.
#
#   usage: research/e162_leg.sh TAG WORKER_BIN [TOKENS] [DEPTH] [GOLDEN]
#
# WHY THIS EXISTS. `benchmark-qwen-mtp.sh` never reports the quantity E162 is
# about. It redirects the timed verb's stdout into `${run_dir}/mtp-decode.json`
# (benchmark-qwen-mtp.sh:678) and its EXIT trap `rm -rf`s that directory
# (:525), so `seed_prefill_seconds` and `decode_seconds` -- both published by
# `Sources/MLXFastCLI/main.swift:2011-2020` -- are computed, written and then
# deleted. The wrapper's surviving artifact, `score.json`, carries neither.
# This script runs the same trusted verbs with the same arguments and keeps the
# report.
#
# WHY IT TAKES A WORKER PATH. The candidate lives entirely in the `quantized`
# JIT source string compiled into `mlxfast-runtime-worker`; ledger 202(H)
# records that `mlxfast-swift` does not carry that string. `main.swift:2252`
# reads `MLXFAST_RUNTIME_WORKER_EXECUTABLE` and `:2323` reads
# `MLXFAST_MLX_METALLIB`, so an arm is selected by pointing at a prebuilt
# worker. That makes an ABBA schedule cost two builds instead of one per leg.
#
# THE ARM IS WITNESSED, NOT ASSUMED. Ledger 202(H) is the trap where a stale
# worker is timed and still passes. Every leg records the sha256, the mtime and
# the `qmm_t_pipelined_k_loop` string count OF THE BINARY IT ACTUALLY RAN. The
# base worker must report 0 and the candidate worker must report > 0; a leg
# whose witness disagrees with its tag is void.
#
# THE GATE IS REAL. `./benchmark.sh --local-cool-gate-only` is the same 40C
# gate the wrapper uses. Entry and exit GPU temperature are recorded per leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e162_leg.sh TAG WORKER_BIN [TOKENS] [DEPTH] [GOLDEN]}"
worker_bin="${2:?usage: e162_leg.sh TAG WORKER_BIN [TOKENS] [DEPTH] [GOLDEN]}"
tokens="${3:-512}"
depth="${4:-8}"
golden="${5:-}"

swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_256.json"

out="research/out/e162/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

abs() { python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

for f in "${swift_bin}" "${worker_bin}"; do
  [[ -x "${f}" ]] || { echo "e162_leg: missing executable ${f}" >&2; exit 1; }
done

worker_abs="$(abs "${worker_bin}")"
metallib="$(dirname "${worker_abs}")/mlx.metallib"
[[ -s "${metallib}" ]] || {
  echo "e162_leg: no mlx.metallib beside ${worker_abs}" >&2; exit 1; }

# One model-holding process at a time. This script does not take the wrapper's
# lock, so it refuses to start beside a live worker rather than racing it.
if pgrep -f "mlxfast-runtime-worker" >/dev/null 2>&1; then
  echo "e162_leg: a runtime worker is already resident:" >&2
  pgrep -lf "mlxfast-runtime-worker" >&2
  exit 1
fi

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"

export MLXFAST_USE_RUNTIME_WORKER=1
export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${worker_abs}"
export MLXFAST_MLX_METALLIB="${metallib}"
export MLXFAST_NO_SANDBOX=1

gpu_temp() {
  local macmon
  macmon="$(command -v macmon || echo "${HOME}/.local/bin/macmon")"
  [[ -x "${macmon}" ]] || { echo "unavailable"; return; }
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // "unavailable"' 2>/dev/null \
    || echo "unavailable"
}

{
  echo "tag=${tag}"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "worker=${worker_abs}"
  echo "worker_sha256=$(shasum -a 256 "${worker_abs}" | cut -d' ' -f1)"
  echo "worker_mtime=$(stat -f %m "${worker_abs}")"
  echo "worker_bytes=$(stat -f %z "${worker_abs}")"
  # THE ARM WITNESS. Literal, >= 16 bytes, and unique to the pipelined helper.
  echo "witness_qmm_t_pipelined_k_loop=$(strings -a "${worker_abs}" \
    | grep -c -F 'qmm_t_pipelined_k_loop' || true)"
  echo "metallib_sha256=$(shasum -a 256 "${metallib}" | cut -d' ' -f1)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  echo "host=$(sysctl -n machdep.cpu.brand_string)"
  echo "mem_bytes=$(sysctl -n hw.memsize)"
} > "${out}/meta.txt"

# --- reference rows ----------------------------------------------------------
# Generated ONCE from the base build and then reused by every arm, which is
# stricter than the wrapper's own scheme: the wrapper regenerates rows from the
# candidate build, so a candidate that changed the tokens would still match
# itself. Reusing base-generated rows makes `all_tokens_matched` a real
# cross-arm bit-exactness test.
if [[ -z "${golden}" ]]; then
  golden="${out}/golden-rows.json"
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
    "${public_golden_path}" > "${out}/seed-plan.json"
  ./benchmark.sh --local-cool-gate-only >> "${out}/gate.log" 2>&1
  "${swift_bin}" mtp-verify \
    --weights "${weights_path}" \
    --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    --emitted "${out}/seed-plan.json" \
    --generate "$(( tokens + 1 ))" \
    --mtp-depth "${depth}" \
    --output "${golden}" \
    --plan-output "${out}/generated-plan.json" \
    > "${out}/verify.out" 2> "${out}/verify.err"
  echo "generate_exit=$?" >> "${out}/meta.txt"
fi
echo "golden=${golden}" >> "${out}/meta.txt"
echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)" >> "${out}/meta.txt"

# --- gated timed leg ---------------------------------------------------------
./benchmark.sh --local-cool-gate-only >> "${out}/gate.log" 2>&1
gate_status=$?
echo "cool_gate_exit=${gate_status}" >> "${out}/meta.txt"
echo "cool_gate_passed_real_gate=$([[ ${gate_status} == 0 ]] && echo true || echo false)" \
  >> "${out}/meta.txt"
echo "gpu_temp_entry_c=$(gpu_temp)" >> "${out}/meta.txt"

"${swift_bin}" mtp-timed \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --golden "${golden}" \
  --tokens "${tokens}" \
  --mtp-depth "${depth}" \
  --output "${out}/report.json" \
  > "${out}/mtp-decode.json" 2> "${out}/mtp.err"
timed_status=$?
echo "timed_exit=${timed_status}" >> "${out}/meta.txt"
echo "gpu_temp_exit_c=$(gpu_temp)" >> "${out}/meta.txt"

if [[ "${timed_status}" != "0" ]]; then
  echo "e162_leg: ${tag}: mtp-timed exited ${timed_status}" >&2
  tail -5 "${out}/mtp.err" >&2
  exit "${timed_status}"
fi

# `decode_seconds` and `seed_prefill_seconds` share one clock origin
# (QwenRuntimeMTPDriver.swift:94 and :197), so the subtraction below is exact
# rather than an estimate, and prefill is charged to the same window.
jq -r '
  "tokens=\(.decode_token_count)",
  "matched=\(.all_tokens_matched)",
  "seed_prefill_seconds=\(.seed_prefill_seconds)",
  "prefill_seconds_per_token=\(.prefill_seconds_per_token)",
  "decode_seconds=\(.decode_seconds)",
  "parent_measured_seconds_per_token=\(.parent_measured_seconds_per_token)",
  "prefill_share=\(.seed_prefill_seconds / .decode_seconds)",
  "decode_only_seconds=\(.decode_seconds - .seed_prefill_seconds)",
  "decode_only_spt=\((.decode_seconds - .seed_prefill_seconds) / .decode_token_count)",
  "effective_mean_draft_len=\(.effective_mean_draft_len // "absent")",
  "accepted_draft_total=\(.accepted_draft_total // "absent")",
  "round_count=\(.round_count // "absent")",
  "emitted_token_total=\(.emitted_token_total // "absent")",
  "reference_checked_rows=\(.reference_checked_rows // "absent")"
' "${out}/report.json" >> "${out}/meta.txt" 2>/dev/null

cat "${out}/meta.txt"
