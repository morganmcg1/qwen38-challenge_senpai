#!/usr/bin/env bash
# Research-only driver for E26 (stop-token continuation defect).
#
#   research/e26-legs.sh golden
#   research/e26-legs.sh legs ARM TOKENS:DEPTH [TOKENS:DEPTH ...]
#
# The wrapper cannot express what this experiment needs: several token windows
# against ONE reference golden, so the 128/256 windows (provably stop-free) and
# the 512 window (crosses the first stop token at emitted index 301) are checked
# against the same reference stream instead of against three windows' worth of
# separately regenerated rows.
#
# `golden` generates 513 reference rows from the public fixture seed through
# Qwen36MTPReferenceSession, which this experiment does not touch, so one golden
# is valid for the base arm and the candidate arm alike.
#
# Every leg writes the trusted CLI's own timed report, including its row_ledger,
# under E26_ROOT (outside Git). Legs are independent: a failing leg is recorded
# and the next one still runs, because a leg that ABORTS is itself the E26
# observation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

phase="${1:?usage: e26-legs.sh golden | legs ARM TOKENS:DEPTH ...}"

root="${E26_ROOT:-${HOME}/e26-stop-token}"
fixture="${E26_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_1024.json}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
head_dir="${MLXFAST_QWEN_MTP_HEAD_DIR:?MLXFAST_QWEN_MTP_HEAD_DIR must be set}"
cli="${repo_root}/.build/release/mlxfast-swift"
worker="${repo_root}/.build-worker/release/mlxfast-runtime-worker"
plan="${root}/seed-plan.json"
golden="${root}/golden-513.json"
session_src="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

gpu_temp() {
  local bin="${MLXFAST_MACMON_BIN:-}"
  if [[ -x "${bin}" ]]; then
    "${bin}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // "unknown"' 2>/dev/null || echo unknown
  else
    echo unknown
  fi
}

case "${phase}" in
  golden) armdir="${root}/golden" ;;
  legs) armdir="${root}/${2:?usage: e26-legs.sh legs ARM TOKENS:DEPTH ...}" ;;
  *) echo "e26-legs: unknown phase ${phase}" >&2; exit 2 ;;
esac
mkdir -p "${armdir}"
log="${armdir}/legs.log"

say() { echo "e26-legs: $*" | tee -a "${log}"; }

for path in "${cli}" "${worker}"; do
  [[ -x "${path}" ]] || { echo "e26-legs: missing ${path}" >&2; exit 2; }
done
[[ -d "${weights}" ]] || { echo "e26-legs: missing weights dir ${weights}" >&2; exit 2; }
[[ -d "${head_dir}" ]] || { echo "e26-legs: missing head dir ${head_dir}" >&2; exit 2; }

say "phase=${phase} armdir=${armdir} started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"
say "session_blob=$(git hash-object "${session_src}") session_lines=$(wc -l <"${session_src}" | tr -d ' ')"
say "reached_stop_token_occurrences=$(grep -c reachedStopToken "${session_src}")"
say "cli_mtime=$(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "${cli}") worker_mtime=$(stat -f %Sm -t %Y-%m-%dT%H:%M:%S "${worker}")"
say "gpu_temp_c=$(gpu_temp) cool_gate=BYPASSED_this_host_idles_above_40C"

if [[ "${phase}" == "golden" ]]; then
  jq -e '(.cases[0].prompt_tokens | type == "array") and (.cases[0].prompt_tokens | length) > 0' \
    "${fixture}" >/dev/null || { say "fixture ${fixture} carries no seed tokens"; exit 3; }
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "${fixture}" >"${plan}" || exit 3
  say "fixture=${fixture} seed_tokens=$(jq '.seed_tokens | length' "${plan}")"
  say "generating reference rows=513 depth=2"
  t0=$(date +%s)
  "${cli}" mtp-verify \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --emitted "${plan}" \
    --generate 513 \
    --mtp-depth 2 \
    --output "${golden}" \
    --plan-output "${root}/generated-plan.json" >"${armdir}/verify-stdout.json" 2>>"${log}"
  rc=$?
  say "golden_exit=${rc} wall_s=$(( $(date +%s) - t0 ))"
  (( rc == 0 )) || exit "${rc}"
  # Same preflight the wrapper applies, pinned at the longest window this
  # experiment measures: reference_self_consistent alone does not prove the
  # recorded rows agree with the chain they were generated against.
  if ! jq -e --argjson tokens 512 '
      . as $g
      | ($g.rows | length) as $rows
      | $g.reference_self_consistent == true
        and ($g.seed_tokens | type == "array") and ($g.seed_tokens | length) > 0
        and ($g.rows | type == "array") and $rows > 0
        and ($g.emitted_tokens | type == "array")
        and ($g.emitted_tokens | length) == $rows
        and $rows >= ($tokens + 1)
        and ([range(0; $rows) | select($g.emitted_tokens[.] != $g.rows[.].sequential_argmax)] | length) == 0
    ' "${golden}" >/dev/null 2>&1; then
    say "golden preflight FAILED"
    exit 4
  fi
  say "golden_ok rows=$(jq '.rows | length' "${golden}")"
  # The reason this experiment exists: where the reference itself puts the first
  # stop token, and how many tokens it keeps generating after that.
  jq -r '
    ([.emitted_tokens | to_entries[] | select(.value == 248044 or .value == 248046) | .key] | sort) as $hits
    | "e26-legs: reference first_stop_index=\($hits[0] // -1) stop_hits=\($hits | length) tokens_after_first_stop=\((.emitted_tokens | length) - (($hits[0] // -1) + 1))"
  ' "${golden}" | tee -a "${log}"
  jq -c '.emitted_tokens[296:308]' "${golden}" | sed 's/^/e26-legs: reference emitted[296:308]=/' | tee -a "${log}"
  exit 0
fi

shift 2
(( $# > 0 )) || { say "no legs requested"; exit 2; }
[[ -s "${golden}" ]] || { say "missing golden ${golden}; run the golden phase first"; exit 2; }
say "golden=${golden} rows=$(jq '.rows | length' "${golden}")"

status=0
for spec in "$@"; do
  tokens="${spec%%:*}"
  depth="${spec##*:}"
  out="${armdir}/leg-t${tokens}-d${depth}.json"
  err="${armdir}/leg-t${tokens}-d${depth}.stderr"
  say "leg tokens=${tokens} depth=${depth} gpu_temp_c=$(gpu_temp) start_utc=$(date -u +%H:%M:%SZ)"
  t0=$(date +%s)
  "${cli}" mtp-timed \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --golden "${golden}" \
    --tokens "${tokens}" \
    --mtp-depth "${depth}" >"${out}" 2>"${err}"
  rc=$?
  wall=$(( $(date +%s) - t0 ))
  say "leg tokens=${tokens} depth=${depth} exit=${rc} wall_s=${wall} stdout_bytes=$(wc -c <"${out}" | tr -d ' ')"
  echo "${tokens} ${depth} ${rc} ${wall}" >>"${armdir}/exit-codes.txt"
  if (( rc != 0 )); then
    status=1
    say "leg tokens=${tokens} depth=${depth} stderr_tail: $(tail -c 600 "${err}" | tr '\n' ' ')"
    continue
  fi
  jq -r '
    "e26-legs:   matched=\(.all_tokens_matched) residual_div=\(.residual_divergence_count)"
    + " first_div=\(.first_divergence_index // "null") spt=\(.parent_measured_seconds_per_token)"
    + " decode_s=\(.decode_seconds) prefill_s=\(.seed_prefill_seconds // "unmeasured")"
    + " rounds=\(.round_count) acc_rate=\(.accepted_draft_rate) eff_depth=\(.effective_mean_draft_len // 0)"
    + " acc=\(.accepted_draft_total) rej=\(.rejected_draft_total) tails=\(.target_tail_total)"
    + " emitted=\(.emitted_token_total) declared_rows=\(.declared_rows_total)"
    + " ref_checked=\(.reference_checked_row_total) cache_off=\(.target_cache_offset_final)"
  ' "${out}" 2>/dev/null | tee -a "${log}"
  # The per-row fingerprint, so base and candidate are diffed bitwise instead of
  # trusting two independently computed match flags. Row kinds are `draft` and
  # `targetTail` (QwenRuntimeMTP.swift:180).
  fp="${armdir}/ledger-t${tokens}-d${depth}.json"
  jq -c '[.row_ledger[] | [.kind, .accepted, .token, .reference_token]]' "${out}" >"${fp}" 2>/dev/null
  jq -c '[.row_ledger[] | select(.accepted == true) | .token]' \
    "${out}" >"${armdir}/accepted-t${tokens}-d${depth}.json" 2>/dev/null
  say "  ledger_rows=$(jq 'length' "${fp}" 2>/dev/null) ledger_sha256=$(shasum -a 256 "${fp}" | cut -c1-16) accepted_len=$(jq 'length' "${armdir}/accepted-t${tokens}-d${depth}.json" 2>/dev/null)"
done

say "phase=legs finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status}"
exit "${status}"
