#!/usr/bin/env bash
# E167: one gated, counterbalanced session over the whole 512-token window for
# an ARBITRARY NUMBER OF ARMS, run as a single job so no thermal state is lost
# between legs. Generalised from research/e162_abba.sh, which hard-codes two
# arms named P and C.
#
#   usage: research/e167_arms_abba.sh [TOKENS] [DEPTH] [SCHEDULE]
#   e.g.   research/e167_arms_abba.sh 512 8 \
#            "B0 B1 B2 B2 B1 B0 B0 B1 B2 B2 B1 B0"
#
# An arm label maps to ${workers_root}/<label>, so adding an arm costs one
# staged directory and no edit to this file.
#
# EVERY LEG IS PUBLISHED TO W&B AS IT FINISHES. On 2026-08-24T07:30Z this host
# was reprovisioned mid-session and all 16 legs of the four-arm counter-strip
# session were lost, because leg results only existed under the role directory.
# research/e167_wandb_leg.py resumes one run by id and appends the leg, so a
# second reset costs only the legs that had not yet run. Set
# MLXFAST_E167_WANDB=0 to disable.
#
# WHY THIS SCRIPT EXISTS AT ALL. `benchmark-qwen-mtp.sh` computes the decode
# and prefill numbers and then deletes them: it redirects the timed verb's
# stdout into `${run_dir}/mtp-decode.json` (benchmark-qwen-mtp.sh:678) and its
# EXIT trap `rm -rf`s `run_dir` (:525). `seed_prefill_seconds` and
# `decode_seconds` are published by `Sources/MLXFastCLI/main.swift:2010-2019`;
# the wrapper's only surviving artifact, `score.json`, carries neither. This
# runs the same trusted verbs with the same arguments and keeps the report.
#
# HOW AN ARM IS SELECTED. `main.swift:2252` reads
# `MLXFAST_RUNTIME_WORKER_EXECUTABLE` and `:2323` reads `MLXFAST_MLX_METALLIB`,
# so an arm is chosen by pointing at a prebuilt worker. A twelve-leg schedule
# over three arms therefore costs three builds, not twelve.
#
# THE ARM IS WITNESSED, NOT ASSUMED. Every leg records the sha256 of the binary
# it ran and a `nm -a` symbol witness with a positive control. A leg whose
# witness disagrees with its label is void. See `witness_of` below for what
# `nm` can and cannot separate for these particular arms.
#
# WHY THE REFERENCE ROWS COME FROM ONE ARM. The wrapper regenerates rows from
# the build under test, so an arm that silently changed the tokens would still
# match itself. Generating once from the first arm of the schedule and reusing
# those rows for every arm turns `all_tokens_matched` into a real N-way
# cross-arm bit-exactness test.
#
# WHY THE SCHEDULE IS REPEATED. `B0 B1 B2 B2 B1 B0` is one palindromic block in
# which every arm has the same position sum, so a monotone thermal or clock
# drift cancels to first order. Repeating the block separates drift from a
# genuine arm effect and gives the within-arm replicates the interval needs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
depth="${2:-8}"
schedule="${3:-B0 B1 B2 B2 B1 B0 B0 B1 B2 B2 B1 B0}"

# The staged workers live beside the checkout, not inside it: they are 200 MB
# each and must never enter Git, and they must survive a `git checkout` of a
# different arm.
workers_root="${MLXFAST_E167_WORKERS:-$(cd ../.. && pwd)/e167-workers}"
swift_bin="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
weights_path="${MLXFAST_WEIGHTS_PATH:-weights}"
public_golden_path="correctness_prompts/public_longcopy_gate_english_512_256.json"

out="research/out/e167/arms"
rm -rf "${out}"
mkdir -p "${out}"

# One W&B run for the whole session, resumed once per finished leg.
wandb_enabled="${MLXFAST_E167_WANDB:-1}"
wandb_run_id="${MLXFAST_E167_WANDB_RUN_ID:-e167-arms-$(date -u +%Y%m%dT%H%M%SZ)}"

# macOS ships bash 3.2, which has no associative arrays, so an arm label maps
# to a directory by name rather than through a table.
arm_dir() {
  echo "${workers_root}/$1"
}

# The distinct labels of the schedule, in first-appearance order.
arms=""
for label in ${schedule}; do
  case " ${arms} " in
    *" ${label} "*) ;;
    *) arms="${arms}${label} " ;;
  esac
done
[[ -n "${arms}" ]] || { echo "e167_arms_abba: empty schedule" >&2; exit 1; }

for label in ${arms}; do
  for f in "$(arm_dir "${label}")/mlxfast-runtime-worker" "$(arm_dir "${label}")/mlx.metallib"; do
    [[ -s "${f}" ]] || { echo "e167_arms_abba: missing ${f}" >&2; exit 1; }
  done
done
[[ -x "${swift_bin}" ]] || { echo "e167_arms_abba: missing ${swift_bin}" >&2; exit 1; }

# Every arm must appear the same number of times, or the schedule is not
# balanced and a monotone drift does not cancel.
balance=""
for label in ${arms}; do
  n=0
  for s in ${schedule}; do [[ "${s}" == "${label}" ]] && n=$(( n + 1 )); done
  balance="${balance}${label}=${n} "
  if [[ -z "${expect_n:-}" ]]; then expect_n="${n}"; fi
  if [[ "${n}" != "${expect_n}" ]]; then
    echo "e167_arms_abba: unbalanced schedule (${balance})" >&2; exit 1
  fi
done

if pgrep -f "mlxfast-runtime-worker" >/dev/null 2>&1; then
  echo "e167_arms_abba: a runtime worker is already resident; refusing to race it" >&2
  pgrep -lf "mlxfast-runtime-worker" >&2
  exit 1
fi

eval "$(./setup-qwen-mtp.sh --print-paths)"
: "${MLXFAST_QWEN_MTP_HEAD_DIR:?setup-qwen-mtp.sh did not provide the MTP head path}"
export MLXFAST_USE_RUNTIME_WORKER=1
export MLXFAST_NO_SANDBOX=1

# Same reader and same search order as benchmark.sh:419-454, including the
# ${HOME}/bin drop used on the ranked boxes, which is where macmon actually
# lives here. An earlier version searched only PATH and ~/.local/bin and
# recorded "unavailable" for every leg.
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

# The gate's own last reading, taken by the trusted gate itself rather than by
# a second sample after it returned. This is the temperature the leg actually
# started from.
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

# RULE 197. The arm is witnessed, not assumed, and the witness carries a
# positive control so a silently empty instrument cannot read as a stripped
# arm. `xsums` counts the counter symbols, `addr` counts the unsafe mutable
# addressors that the linkage hypothesis is about, and `anchor` must be
# non-zero in EVERY arm.
#
# `nm` cannot separate an arm that keeps the declarations and drops the writes
# from the unmodified base. That contrast is witnessed functionally instead, by
# the per-round census signature each arm prints in a traced run.
witness_of() {
  local xsums addr anchor
  xsums="$(nm -a "$1" 2>/dev/null | grep -c 'qwen35XSums' || true)"
  addr="$(nm -a "$1" 2>/dev/null | grep 'qwen35XSums' | grep -c 'Sivau' || true)"
  anchor="$(nm -a "$1" 2>/dev/null | grep -c 'Qwen36MTPBlockSession' || true)"
  echo "xsums=${xsums},addr=${addr},anchor=${anchor}"
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
  echo "arms=${arms}"
  echo "arm_balance=${balance}"
  echo "wandb_run_id=$([[ "${wandb_enabled}" == "1" ]] && echo "${wandb_run_id}" || echo disabled)"
  for label in ${arms}; do
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

# The rows come from the FIRST arm of the schedule and are reused by every
# other arm, which turns `all_tokens_matched` into a real N-way cross-arm
# bit-exactness test instead of each arm matching itself.
ref_arm="$(echo ${arms} | awk '{print $1}')"
select_arm "${ref_arm}"
echo "reference_arm=${ref_arm}" >> "${out}/session.txt"
echo "e167_arms_abba: cool gate before reference generation" >&2
./benchmark.sh --local-cool-gate-only >> "${out}/gate.log" 2>&1
echo "e167_arms_abba: generating $(( tokens + 1 )) reference rows from ${ref_arm} (depth=${depth})" >&2
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
  echo "e167_arms_abba: reference generation failed" >&2; tail -20 "${out}/verify.err" >&2; exit 1
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
  echo "e167_arms_abba: the reference rows are not usable" >&2; exit 1
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

  echo "e167_arms_abba: leg ${leg} arm ${label}: cool gate" >&2
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
    echo "e167_arms_abba: leg ${leg} (${label}) FAILED exit ${timed_status}" >&2
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

  # Publish this leg now, not at the end of the session. A host reset then
  # costs only the legs that have not run. This happens after the timed verb
  # has exited and before the next leg's cool gate, so it cannot perturb a
  # measurement.
  if [[ "${wandb_enabled}" == "1" ]]; then
    python3 research/e167_wandb_leg.py \
      --run-id "${wandb_run_id}" \
      --leg-dir "${legdir}" \
      --session "${out}/session.txt" \
      >> "${out}/wandb.log" 2>&1 \
      || echo "e167_arms_abba: W&B publish failed for leg ${leg}; continuing" >&2
  fi
done

echo "utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out}/session.txt"
cat "${out}/session.txt"
echo
column -t -s $'\t' "${out}/legs.tsv"
