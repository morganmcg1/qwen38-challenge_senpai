#!/usr/bin/env bash
# E171: price the research instrumentation on the submitted surface with one
# gated, counterbalanced ABBA session over the whole 512-token window.
#
#   usage: research/e171_abba.sh [TOKENS] [DEPTH] [SCHEDULE]
#   e.g.   research/e171_abba.sh 512 8 "s0 s1 s1 s0"
#
# ARMS. Both carry the E165 cross-round head-chain prefetch as the shipped
# default. They differ only in the research instrumentation:
#
#   s0  Qwen35.swift as the campaign base carries it: two counter increments in
#       the custom QMV router, five counter globals, the MLX_E141_ROWS_PER_LEAF
#       override, three counter writes in buildDerivedClusterIndex; plus the
#       five session trace fields and the E159 fixed-depth pin.
#   s1  Qwen35.swift restored to organizer-main bytes at 0863b06a, with the two
#       session-side consumers of the deleted counters removed.
#
# The arm is compile-time, so it is selected per leg by pointing
# MLXFAST_RUNTIME_WORKER_EXECUTABLE at a prebuilt worker
# (research/e171_stage_arms.sh). Every leg records the sha256 and the
# `qwen35XSumsSidecarHits` symbol count OF THE BINARY IT RAN, with
# `prefetchHeadStep` as the positive control that the symbol table was read at
# all. A leg whose witness disagrees with its label is void.
#
# WHY THE REFERENCE ROWS COME FROM s0. The wrapper regenerates rows from the
# build under test, so a candidate that silently changed the token stream would
# still match itself. Generating once from s0 and reusing them for both arms
# turns `all_tokens_matched` into a real cross-arm bit-exactness test of the
# strip.
#
# PALINDROME. s0 s1 s1 s0 gives both arms mean leg position 2.5, so a monotone
# thermal or clock drift in leg index cancels to first order.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
depth="${2:-8}"
schedule="${3:-s0 s1 s1 s0}"

workers_root="${MLXFAST_E171_WORKERS:-$(cd ../.. && pwd)/e171-workers}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_1024.json"

out="research/out/e171/abba"
rm -rf "${out}"
mkdir -p "${out}"

arm_dir() {
  case "$1" in
    s0|s1) echo "${workers_root}/$1" ;;
    *) echo "e171_abba: unknown arm label '$1'" >&2; exit 1 ;;
  esac
}

for label in s0 s1; do
  for f in "$(arm_dir "${label}")/mlxfast-runtime-worker" "$(arm_dir "${label}")/mlx.metallib"; do
    [[ -s "${f}" ]] || { echo "e171_abba: missing ${f}" >&2; exit 1; }
  done
done
[[ -x "${swift_bin}" ]] || { echo "e171_abba: missing ${swift_bin}" >&2; exit 1; }

if pgrep -f "mlxfast-runtime-worker" >/dev/null 2>&1; then
  echo "e171_abba: a runtime worker is already resident; refusing to race it" >&2
  pgrep -lf "mlxfast-runtime-worker" >&2
  exit 1
fi

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"
export MLXFAST_USE_RUNTIME_WORKER=1

find_macmon() {
  local candidate
  if [[ -n "${MLXFAST_MACMON_BIN:-}" && -x "${MLXFAST_MACMON_BIN}" ]]; then
    printf '%s\n' "${MLXFAST_MACMON_BIN}"; return 0
  fi
  if candidate="$(command -v macmon 2>/dev/null)"; then
    printf '%s\n' "${candidate}"; return 0
  fi
  for candidate in /opt/homebrew/bin/macmon /usr/local/bin/macmon "${HOME}/bin/macmon"; do
    [[ -x "${candidate}" ]] && { printf '%s\n' "${candidate}"; return 0; }
  done
  return 1
}
MACMON_BIN="$(find_macmon || true)"

gpu_temp() {
  [[ -n "${MACMON_BIN}" ]] || { echo "unavailable"; return; }
  local t
  t="$("${MACMON_BIN}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null)"
  [[ -n "${t}" ]] && echo "${t}" || echo "unavailable"
}

gate_temp() {
  grep -o 'gate passed (current [0-9.]*C' "$1" 2>/dev/null \
    | tail -1 | sed 's/.*current //; s/C$//'
}

select_arm() {
  local d
  d="$(arm_dir "$1")"
  export MLXFAST_RUNTIME_WORKER_EXECUTABLE="${d}/mlxfast-runtime-worker"
  export MLXFAST_MLX_METALLIB="${d}/mlx.metallib"
}

witness_of() { nm -a "$1" 2>/dev/null | grep -c -F 'qwen35XSumsSidecarHits' || true; }
control_of() { nm -a "$1" 2>/dev/null | grep -c -F 'prefetchHeadStep' || true; }

{
  echo "utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain -- Sources Vendor | wc -l | tr -d ' ')"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "schedule=${schedule}"
  echo "host=$(sysctl -n machdep.cpu.brand_string)"
  echo "mem_bytes=$(sysctl -n hw.memsize)"
  echo "swift_bin_sha256=$(shasum -a 256 "${swift_bin}" | cut -d' ' -f1)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
  for label in s0 s1; do
    d="$(arm_dir "${label}")"
    echo "worker_${label}_sha256=$(shasum -a 256 "${d}/mlxfast-runtime-worker" | cut -d' ' -f1)"
    echo "worker_${label}_witness=$(witness_of "${d}/mlxfast-runtime-worker")"
    echo "worker_${label}_control=$(control_of "${d}/mlxfast-runtime-worker")"
    echo "metallib_${label}_sha256=$(shasum -a 256 "${d}/mlx.metallib" | cut -d' ' -f1)"
  done
} | tee "${out}/session.txt"

# --- reference rows, generated ONCE from s0 ----------------------------------
golden="${out}/golden-rows.json"
jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' \
  "${public_golden_path}" > "${out}/seed-plan.json"

select_arm s0
echo "e171_abba: cool gate before reference generation" >&2
./benchmark.sh --local-cool-gate-only >> "${out}/gate.log" 2>&1
echo "e171_abba: generating $(( tokens + 1 )) reference rows from s0 (depth=${depth})" >&2
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
  echo "e171_abba: reference generation failed" >&2; tail -20 "${out}/verify.err" >&2; exit 1
fi

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
  echo "e171_abba: the reference rows are not usable" >&2; exit 1
fi
{
  echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
  echo "golden_rows=$(jq -r '.rows | length' "${golden}")"
  echo "golden_seed_tokens=$(jq -r '.seed_tokens | length' "${golden}")"
} >> "${out}/session.txt"

# --- the counterbalanced legs ------------------------------------------------
printf 'leg\tarm\tworker_sha12\twitness\tcontrol\tentry_c\texit_c\tmatched\tspt\tdecode_s\tprefill_s\tdecode_only_spt\tedl\taccepted\trounds\taccept_rate\n' \
  > "${out}/legs.tsv"

leg=0
for label in ${schedule}; do
  leg=$(( leg + 1 ))
  select_arm "${label}"
  worker="${MLXFAST_RUNTIME_WORKER_EXECUTABLE}"
  legdir="${out}/leg${leg}-${label}"
  mkdir -p "${legdir}"

  echo "e171_abba: leg ${leg} arm ${label}: cool gate" >&2
  ./benchmark.sh --local-cool-gate-only >> "${legdir}/gate.log" 2>&1
  gate_status=$?
  gate_c="$(gate_temp "${legdir}/gate.log")"
  entry_c="${gate_c:-$(gpu_temp)}"

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
    echo "control=$(control_of "${worker}")"
    echo "cool_gate_exit=${gate_status}"
    echo "cool_gate_passed_real_gate=$([[ ${gate_status} == 0 ]] && echo true || echo false)"
    echo "gate_qualified_for_timing=$([[ ${gate_status} == 0 ]] && echo true || echo false)"
    echo "gpu_temp_entry_c=${entry_c}"
    echo "gpu_temp_entry_source=$([[ -n "${gate_c}" ]] && echo cool_gate || echo macmon_sample)"
    echo "gpu_temp_exit_c=${exit_c}"
    echo "macmon_bin=${MACMON_BIN:-none}"
    echo "timed_exit=${timed_status}"
  } > "${legdir}/meta.txt"

  if [[ "${timed_status}" != "0" ]]; then
    echo "e171_abba: leg ${leg} (${label}) FAILED exit ${timed_status}" >&2
    tail -10 "${legdir}/stderr.txt" >&2
    continue
  fi

  jq -r --arg leg "${leg}" --arg arm "${label}" \
     --arg sha "$(shasum -a 256 "${worker}" | cut -c1-12)" \
     --arg wit "$(witness_of "${worker}")" \
     --arg ctl "$(control_of "${worker}")" \
     --arg ec "${entry_c}" --arg xc "${exit_c}" '
    [ $leg, $arm, $sha, $wit, $ctl, $ec, $xc,
      (.all_tokens_matched | tostring),
      (.parent_measured_seconds_per_token | tostring),
      (.decode_seconds | tostring),
      ((.seed_prefill_seconds // 0) | tostring),
      (((.decode_seconds - (.seed_prefill_seconds // 0)) / .decode_token_count) | tostring),
      ((.effective_mean_draft_len // "absent") | tostring),
      ((.accepted_draft_total // "absent") | tostring),
      ((.round_count // "absent") | tostring),
      ((.accepted_draft_rate // "absent") | tostring)
    ] | @tsv' "${legdir}/report.json" >> "${out}/legs.tsv"
done

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out}/session.txt"
cat "${out}/session.txt"
echo
column -t -s $'\t' "${out}/legs.tsv"
