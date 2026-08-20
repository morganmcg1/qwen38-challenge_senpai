#!/usr/bin/env bash
# One E84 exactness pass: install an arm, build it, witness it, and run the
# untimed 512-token gate against a golden the BASE arm produced.
#
#   research/e84_ledger_run.sh ARM [--tokens N] [--depth D]
#
# `mtp-verify --golden` (no `--generate`) is the only verb that runs the
# candidate session with `retainLedger: true`, so it is the only producer of
# `row_ledger`, which carries `top2_tokens` and `top2_logits` for every row the
# session actually evaluated. The benchmark pipeline never invokes that mode.
#
# The golden is generated once by the unchanged base arm, so it is not a
# candidate-generated reference for the arms under test: every arm is judged
# against one byte-identical set of reference rows produced by code that has
# neither mechanism in it.
#
# Untimed and produces no score, so it needs no thermal gate and is not a
# replicate of the timed measurement.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

arm="${1:?usage: e84_ledger_run.sh ARM [--tokens N] [--depth D]}"
shift

tokens="${E84_LEDGER_TOKENS:-512}"
depth="${E84_LEDGER_DEPTH:-8}"
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    *) echo "e84_ledger_run: unknown argument $1" >&2; exit 2 ;;
  esac
done

readonly SCORED_FILE="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
readonly A_SYMBOL="islandFastPathReady"
readonly B_STRING="qwen35_gated_delta_replay_state"

base_sha="${E84_BASE_SHA:-07c75a708c2347021d3148d7bc87b246ba2aec73}"
root="${E84_ROOT:-${repo_root}/.mlxfast-private/e84}"
head_dir="${E84_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
golden="${E84_GOLDEN:-${root}/runs/exact-base-512/reports/02-mtp-verify-output.json}"
# A label lets the same arm run twice against different goldens without one
# pass overwriting the other's evidence.
label="${E84_LEDGER_LABEL:-${arm}}"
# The positive control runs a known-good arm against a golden with one token
# changed. It passes only when the gate reports a mismatch, which is what
# proves the gate is not vacuous.
expect_mismatch="${E84_EXPECT_MISMATCH:-0}"
out_dir="${root}/ledgers"
out="${out_dir}/${label}.json"
meta="${out_dir}/${label}-meta.txt"

fail() { echo "e84_ledger_run: $*" >&2; exit 1; }

[[ -d "${head_dir}" ]] || fail "missing head tree ${head_dir}"
[[ -r "${golden}" ]] || fail "missing golden ${golden}"
mkdir -p "${out_dir}"

export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
# Same command-buffer geometry as the timed arms: a different geometry can move
# evaluation boundaries and therefore which rows are batched together.
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

[[ -z "$(git status --porcelain)" ]] || fail "worktree is dirty"

python3 research/e84_arm.py "${arm}" --base "${base_sha}" --tip "${pre_patch_sha}" \
  || fail "arm ${arm} could not be materialised"
git add -- "${SCORED_FILE}"
git commit -q --allow-empty -m "E84 ledger ${arm}: TRANSIENT arm bytes under exactness test

Unwound to ${pre_patch_sha} when the run exits."
transient_sha="$(git rev-parse HEAD)"

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
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
senpai/rebuild-and-assert-worker.sh "${witness[@]}" > "${meta}.assert" 2>&1 \
  || { tail -40 "${meta}.assert" >&2; fail "worker assert failed for ${arm}"; }

rows_available="$(python3 -c "
import json
print(len(json.load(open('${golden}'))['rows']))
")"
((rows_available >= tokens + 1)) \
  || fail "golden carries ${rows_available} rows; need $((tokens + 1))"

{
  echo "arm=${arm}"
  echo "label=${label}"
  echo "expect_mismatch=${expect_mismatch}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "base_sha=${base_sha}"
  echo "branch_commit=${pre_patch_sha}"
  echo "measured_commit_unwound=${transient_sha}"
  echo "golden=${golden}"
  echo "golden_rows=${rows_available}"
  echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
  echo "head_dir=${head_dir}"
  echo "head_provenance_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "scored_source_sha256=$(shasum -a 256 "${SCORED_FILE}" | cut -d' ' -f1)"
  echo "worker_sha256=$(awk '/^worker_sha256 /{print $2}' "${meta}.assert" | tail -1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
} > "${meta}"

echo "=== e84_ledger_run: ${label}: mtp-verify --golden (${tokens} tokens, depth ${depth}) ==="
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
.build/release/mlxfast-swift mtp-verify \
  --mtp-head "${head_dir}" \
  --golden "${golden}" \
  --tokens "${tokens}" \
  --mtp-depth "${depth}" > "${out}" 2> "${out_dir}/${label}-stderr.txt"
status=$?

{
  echo "verify_exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${meta}"
grep '^mtp-verify: ' "${out_dir}/${label}-stderr.txt" >> "${meta}"

python3 - "${out}" "${label}" "${expect_mismatch}" "${status}" <<'PY'
import json
import sys

path, label, expect_mismatch, verify_exit = (
    sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4]))

report = None
try:
    report = json.load(open(path))
except Exception as exc:
    read_error = str(exc)

if report is None:
    if expect_mismatch and verify_exit != 0:
        print("e84_ledger_run: %s: positive control OK, the gate refused the "
              "corrupted golden and wrote no report (verify_exit=%d)"
              % (label, verify_exit))
        raise SystemExit(0)
    raise SystemExit(
        "e84_ledger_run: %s: unreadable report (%s)" % (label, read_error))

ledger = report.get("row_ledger", [])
matched = report.get("all_tokens_matched")
print("e84_ledger_run: %s: rows=%d declared=%s matched=%s parity=%s "
      "verify_exit=%d" % (label, len(ledger), report.get("declared_rows_total"),
                          matched, report.get("parity_all_ok"), verify_exit))

if expect_mismatch:
    if matched is False or verify_exit != 0:
        print("e84_ledger_run: %s: positive control OK, the gate rejects a "
              "corrupted golden" % label)
        raise SystemExit(0)
    raise SystemExit(
        "e84_ledger_run: %s: POSITIVE CONTROL FAILED, the gate accepted a "
        "corrupted golden (all_tokens_matched=%s, verify_exit=%d)"
        % (label, matched, verify_exit))

if not ledger:
    raise SystemExit("e84_ledger_run: %s: NO row_ledger in the report" % label)
if matched is not True:
    raise SystemExit(
        "e84_ledger_run: %s: all_tokens_matched=%s" % (label, matched))
PY
check=$?

# In positive-control mode a non-zero CLI exit is the expected outcome, so the
# Python verdict is the only one that counts.
if ((expect_mismatch)) || ((status == 0)); then
  status="${check}"
fi

echo "e84_ledger_run: ${label}: status=${status} ledger=${out}"
exit "${status}"
