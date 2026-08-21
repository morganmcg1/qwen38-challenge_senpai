#!/usr/bin/env bash
# E90 rung 0a: composed-tree exactness for the merged E84 + E85 + E86 base.
#
#   research/e90_exact_session.sh [--tokens N] [--label NAME]
#
# E84 and E85 both edit Qwen35.swift and have never been verified together.
# This session runs the untimed 512-token exact-token and row-ledger gate on
# the tree as it stands, then proves the gate can fail.
#
# 1. Reference rows. `mtp-verify --emitted PLAN --generate N+1` runs the SERIAL
#    reference session (depth 0, one token per target forward) over the public
#    fixture's seed. Those rows are the truth the MTP pass is judged against.
# 2. Gate pass. `mtp-verify --golden` is the only verb that runs the candidate
#    session with `retainLedger: true`, so it is the only producer of
#    `row_ledger` with per-row top-2 tokens and logits.
# 3. Positive control. One reference token in the middle of the window is
#    changed and the same gate is run again. A gate that cannot fail proves
#    nothing about the passes above.
#
# Untimed and produces no score, so it needs no thermal gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

tokens=512
label="rung0a"
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    *) echo "e90_exact_session: unknown argument $1" >&2; exit 2 ;;
  esac
done

root="${E90_ROOT:-${repo_root}/.mlxfast-private/e90}"
head_dir="${E90_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
fixture="${E90_FIXTURE:-correctness_prompts/public_longcopy_gate_english_512_256.json}"
weights="${MLXFAST_WEIGHTS_PATH:-weights}"
cli="${MLXFAST_SWIFT_BIN:-.build/release/mlxfast-swift}"
depth="${E90_DEPTH:-8}"

out_dir="${root}/${label}"
mkdir -p "${out_dir}"
golden="${out_dir}/golden-${tokens}.json"
plan="${out_dir}/seed-plan.json"
ledger="${out_dir}/ledger.json"
mutated="${out_dir}/golden-mutated-row256.json"
control="${out_dir}/ledger-mutated.json"
meta="${out_dir}/meta.txt"

fail() { echo "e90_exact_session: $*" >&2; exit 1; }

[[ -s "${head_dir}/config.json" ]] || fail "no declared head at ${head_dir}"
[[ -x "${cli}" ]] || fail "no CLI at ${cli}"
[[ -s "${fixture}" ]] || fail "no fixture at ${fixture}"

# Same command-buffer geometry as every other leg in this campaign: a different
# geometry moves evaluation boundaries and therefore which rows batch together.
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export DARKBLOOM_STARTUP_MEMORY_PROFILE="${DARKBLOOM_STARTUP_MEMORY_PROFILE:-full}"
export MLX_MAX_MB_PER_BUFFER="${MLX_MAX_MB_PER_BUFFER:-512}"
export MLX_MAX_OPS_PER_BUFFER="${MLX_MAX_OPS_PER_BUFFER:-50}"
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

{
  echo "label=${label}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "commit=$(git rev-parse HEAD)"
  echo "dirty_candidate_paths=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "fixture=${fixture}"
  echo "head_dir=${head_dir}"
  echo "head_provenance_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 "${cli}" | cut -d' ' -f1)"
  echo "vendored_metal_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
} > "${meta}"

if [[ ! -s "${golden}" ]]; then
  echo "=== e90 rung 0a: reference rows ($((tokens + 1)) rows) ===" >&2
  jq -c '{seed_tokens: .cases[0].prompt_tokens, emitted: []}' "${fixture}" > "${plan}"
  "${cli}" mtp-verify \
    --weights "${weights}" \
    --mtp-head "${head_dir}" \
    --emitted "${plan}" \
    --generate "$((tokens + 1))" \
    --mtp-depth "${depth}" \
    --output "${golden}" \
    --plan-output "${out_dir}/generated-plan.json" \
    > "${out_dir}/generate-stdout.json" 2> "${out_dir}/generate-stderr.txt" \
    || fail "reference generation failed; see ${out_dir}/generate-stderr.txt"
fi
[[ -s "${golden}" ]] || fail "no golden at ${golden}"
echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)" >> "${meta}"

echo "=== e90 rung 0a: gate pass (${tokens} tokens, depth ${depth}) ===" >&2
"${cli}" mtp-verify \
  --weights "${weights}" \
  --mtp-head "${head_dir}" \
  --golden "${golden}" \
  --tokens "${tokens}" \
  --mtp-depth "${depth}" > "${ledger}" 2> "${out_dir}/ledger-stderr.txt"
gate_status=$?
echo "gate_exit=${gate_status}" >> "${meta}"

echo "=== e90 rung 0a: positive control ===" >&2
python3 - "${golden}" "${mutated}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
g = json.load(open(src))
i = 256
old = g["rows"][i]["sequential_argmax"]
new = old + 1
g["rows"][i]["sequential_argmax"] = new
g["rows"][i]["top2_tokens"][0] = new
g["emitted_tokens"][i] = new
json.dump(g, open(dst, "w"))
print("e90_exact_session: mutated golden row %d token %d -> %d" % (i, old, new))
PY

"${cli}" mtp-verify \
  --weights "${weights}" \
  --mtp-head "${head_dir}" \
  --golden "${mutated}" \
  --tokens "${tokens}" \
  --mtp-depth "${depth}" > "${control}" 2> "${out_dir}/control-stderr.txt"
control_status=$?
echo "control_exit=${control_status}" >> "${meta}"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${meta}"

python3 research/e90_exact_report.py \
  --golden "${golden}" \
  --ledger "${ledger}" \
  --gate-exit "${gate_status}" \
  --control "${control}" \
  --control-exit "${control_status}" \
  --tokens "${tokens}" \
  --meta "${meta}" \
  --output "${out_dir}/rung0a-summary.json"
exit $?
