#!/usr/bin/env bash
# E162: one gated, counterbalanced ABBA session over the whole 512-token
# window, run as a single job so no thermal state is lost between legs.
#
#   usage: research/e162_abba.sh [TOKENS] [DEPTH] [SCHEDULE]
#   e.g.   research/e162_abba.sh 512 8 "P C C P P C C P"
#
# WHY THIS SCRIPT EXISTS AT ALL. `benchmark-qwen-mtp.sh` computes the number
# E162 is about and then deletes it: it redirects the timed verb's stdout into
# `${run_dir}/mtp-decode.json` (benchmark-qwen-mtp.sh:678) and its EXIT trap
# `rm -rf`s `run_dir` (:525). `seed_prefill_seconds` and `decode_seconds` are
# published by `Sources/MLXFastCLI/main.swift:2010-2019`; the wrapper's only
# surviving artifact, `score.json`, carries neither. This runs the same trusted
# verbs with the same arguments and keeps the report.
#
# HOW AN ARM IS SELECTED. The candidate lives entirely inside the `quantized`
# JIT source string that is compiled into `mlxfast-runtime-worker`.
# `main.swift:2252` reads `MLXFAST_RUNTIME_WORKER_EXECUTABLE` and `:2323` reads
# `MLXFAST_MLX_METALLIB`, so an arm is chosen by pointing at a prebuilt worker.
# An eight-leg schedule therefore costs two builds, not eight.
#
# THE ARM IS WITNESSED, NOT ASSUMED. Every leg records the sha256 and the
# `qmm_t_pipelined_k_loop` string count OF THE BINARY IT RAN. P must report 0
# and C must report > 0. A leg whose witness disagrees with its label is void.
#
# WHY THE REFERENCE ROWS COME FROM P. The wrapper regenerates rows from the
# build under test, so a candidate that silently changed the tokens would still
# match itself. Generating once from P and reusing them for both arms turns
# `all_tokens_matched` into a real cross-arm bit-exactness test.
#
# WHY THE SCHEDULE IS DOUBLED. `P C C P P C C P` is two nested ABBA blocks.
# Each block cancels linear thermal drift to first order, and running two of
# them separates drift from a genuine arm effect.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
depth="${2:-8}"
schedule="${3:-P C C P P C C P}"

# The staged workers live beside the checkout, not inside it: they are 200 MB
# each and must never enter Git, and they must survive a `git checkout` of a
# different arm.
workers_root="${MLXFAST_E162_WORKERS:-$(cd ../.. && pwd)/e162-workers}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_256.json"

out="research/out/e162/abba"
rm -rf "${out}"
mkdir -p "${out}"

# macOS ships bash 3.2, which has no associative arrays.
arm_dir() {
  case "$1" in
    P) echo "${workers_root}/base" ;;
    C) echo "${workers_root}/armA" ;;
    *) echo "e162_abba: unknown arm label '$1'" >&2; exit 1 ;;
  esac
}

for label in P C; do
  for f in "$(arm_dir "${label}")/mlxfast-runtime-worker" "$(arm_dir "${label}")/mlx.metallib"; do
    [[ -s "${f}" ]] || { echo "e162_abba: missing ${f}" >&2; exit 1; }
  done
done
[[ -x "${swift_bin}" ]] || { echo "e162_abba: missing ${swift_bin}" >&2; exit 1; }

if pgrep -f "mlxfast-runtime-worker" >/dev/null 2>&1; then
  echo "e162_abba: a runtime worker is already resident; refusing to race it" >&2
  pgrep -lf "mlxfast-runtime-worker" >&2
  exit 1
fi

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"
export MLXFAST_USE_RUNTIME_WORKER=1
export MLXFAST_NO_SANDBOX=1

gpu_temp() {
  local macmon
  macmon="$(command -v macmon || echo "${HOME}/.local/bin/macmon")"
  [[ -x "${macmon}" ]] || { echo "unavailable"; return; }
  "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // "unavailable"' 2>/dev/null \
    || echo "unavailable"
}

select_arm() {
  local d
  d="$(arm_dir "$1")"
  export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${d}/mlxfast-runtime-worker"
  export MLXFAST_MLX_METALLIB="${d}/mlx.metallib"
}

witness_of() {
  strings -a "$1" | grep -c -F 'qmm_t_pipelined_k_loop' || true
}

{
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l | tr -d ' ')"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "schedule=${schedule}"
  echo "host=$(sysctl -n machdep.cpu.brand_string)"
  echo "mem_bytes=$(sysctl -n hw.memsize)"
  echo "swift_bin_sha256=$(shasum -a 256 "${swift_bin}" | cut -d' ' -f1)"
  echo "swift_bin_witness=$(witness_of "${swift_bin}")"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  for label in P C; do
    d="$(arm_dir "${label}")"
    echo "worker_${label}_sha256=$(shasum -a 256 "${d}/mlxfast-runtime-worker" | cut -d' ' -f1)"
    echo "worker_${label}_witness=$(witness_of "${d}/mlxfast-runtime-worker")"
    echo "metallib_${label}_sha256=$(shasum -a 256 "${d}/mlx.metallib" | cut -d' ' -f1)"
  done
} | tee "${out}/session.txt"

# --- reference rows, generated ONCE from P -----------------------------------
golden="${out}/golden-rows.json"
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${out}/seed-plan.json"

select_arm P
echo "e162_abba: cool gate before reference generation" >&2
./benchmark.sh --local-cool-gate-only >> "${out}/gate.log" 2>&1
echo "e162_abba: generating $(( tokens + 1 )) reference rows from P (depth=${depth})" >&2
"${swift_bin}" mtp-verify \
  --weights "${weights_path}" \
  --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
  --emitted "${out}/seed-plan.json" \
  --generate "$(( tokens + 1 ))" \
  --mtp-depth "${depth}" \
  --output "${golden}" \
  --plan-output "${out}/generated-plan.json" \
  > "${out}/verify.out" 2> "${out}/verify.err"
gen_status=$?
echo "reference_generate_exit=${gen_status}" >> "${out}/session.txt"
if [[ "${gen_status}" != "0" ]]; then
  echo "e162_abba: reference generation failed" >&2; tail -20 "${out}/verify.err" >&2; exit 1
fi

# The wrapper's own reference assertion (benchmark-qwen-mtp.sh:632-654), copied
# so a short or self-inconsistent reference aborts here instead of producing a
# raw range error in the middle of a measured leg.
if ! jq -e --argjson tokens "${tokens}" '
    . as $g
    | ($g.rows | length) as $rows
    | $g.reference_self_consistent == true
      and ($g.seed_tokens | type == "array") and ($g.seed_tokens | length) > 0
      and ($g.rows | type == "array") and $rows > 0
      and ($g.emitted_tokens | type == "array")
      and ($g.emitted_tokens | length) == $rows
      and $rows >= ($tokens + 1)
      and ([range(0; $rows)
            | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
  ' "${golden}" >/dev/null 2>&1; then
  echo "e162_abba: the reference rows are not usable" >&2; exit 1
fi
{
  echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
  echo "golden_rows=$(jq -r '.rows | length' "${golden}")"
  echo "golden_seed_tokens=$(jq -r '.seed_tokens | length' "${golden}")"
} >> "${out}/session.txt"

# --- the counterbalanced legs ------------------------------------------------
printf 'leg\tarm\tworker_sha12\twitness\tentry_c\texit_c\tmatched\tspt\tdecode_s\tprefill_s\tdecode_only_spt\tedl\taccepted\trounds\n' \
  > "${out}/legs.tsv"

leg=0
for label in ${schedule}; do
  leg=$(( leg + 1 ))
  select_arm "${label}"
  worker="${MLXFAST_RUNTIME_WORKER_EXECUTABLE}"
  legdir="${out}/leg${leg}-${label}"
  mkdir -p "${legdir}"

  echo "e162_abba: leg ${leg} arm ${label}: cool gate" >&2
  ./benchmark.sh --local-cool-gate-only >> "${legdir}/gate.log" 2>&1
  gate_status=$?
  entry_c="$(gpu_temp)"

  "${swift_bin}" mtp-timed \
    --weights "${weights_path}" \
    --mtp-head "${MLXFAST_QWEN_MTP_HEAD_DIR}" \
    --golden "${golden}" \
    --tokens "${tokens}" \
    --mtp-depth "${depth}" \
    --output "${legdir}/report.json" \
    > "${legdir}/stdout.json" 2> "${legdir}/stderr.txt"
  timed_status=$?
  exit_c="$(gpu_temp)"

  {
    echo "leg=${leg}"
    echo "arm=${label}"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "worker_sha256=$(shasum -a 256 "${worker}" | cut -d' ' -f1)"
    echo "witness=$(witness_of "${worker}")"
    echo "cool_gate_exit=${gate_status}"
    echo "cool_gate_passed_real_gate=$([[ ${gate_status} == 0 ]] && echo true || echo false)"
    echo "gate_qualified_for_timing=$([[ ${gate_status} == 0 ]] && echo true || echo false)"
    echo "gpu_temp_entry_c=${entry_c}"
    echo "gpu_temp_exit_c=${exit_c}"
    echo "timed_exit=${timed_status}"
  } > "${legdir}/meta.txt"

  if [[ "${timed_status}" != "0" ]]; then
    echo "e162_abba: leg ${leg} (${label}) FAILED exit ${timed_status}" >&2
    tail -10 "${legdir}/stderr.txt" >&2
    continue
  fi

  # `decode_seconds` and `seed_prefill_seconds` share one clock origin
  # (QwenRuntimeMTPDriver.swift:94 and :197), so this subtraction is exact
  # rather than an estimate and charges prefill to the same window.
  jq -r --arg leg "${leg}" --arg arm "${label}" \
     --arg sha "$(shasum -a 256 "${worker}" | cut -c1-12)" \
     --arg wit "$(witness_of "${worker}")" \
     --arg ec "${entry_c}" --arg xc "${exit_c}" '
    [ $leg, $arm, $sha, $wit, $ec, $xc,
      (.all_tokens_matched | tostring),
      (.parent_measured_seconds_per_token | tostring),
      (.decode_seconds | tostring),
      ((.seed_prefill_seconds // 0) | tostring),
      (((.decode_seconds - (.seed_prefill_seconds // 0)) / .decode_token_count) | tostring),
      ((.effective_mean_draft_len // "absent") | tostring),
      ((.accepted_draft_total // "absent") | tostring),
      ((.round_count // "absent") | tostring)
    ] | @tsv' "${legdir}/report.json" >> "${out}/legs.tsv"
done

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out}/session.txt"
cat "${out}/session.txt"
echo
column -t -s $'\t' "${out}/legs.tsv"
