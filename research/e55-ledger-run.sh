#!/usr/bin/env bash
# E55 PATH C: capture the candidate's WIDE-dispatch per-row top-2 evidence.
#
# `mtp-verify --golden` (no `--generate`) is the only verb that runs the
# candidate session with `retainLedger: true`, so it is the only producer of
# `row_ledger`, which carries `top2_tokens` and `top2_logits` for every one of
# the 567 rows the wide multi-row dispatch actually evaluated. The benchmark
# pipeline never invokes that mode: `--local-iterate` runs
# `mtp-verify --generate` (reference rows, M=1) and then `mtp-timed` twice with
# `retainLedger: false`. Without this script there is no direct bitwise reading
# of the code this experiment changed.
#
# This verb is NOT timed and produces no score, so it needs no thermal gate and
# is not a replicate of the timed measurement. It reuses the golden that the
# timed run already produced, so the reference side is held fixed across arms.
#
#   research/e55-ledger-run.sh ARM
#
# ARM names the arm whose twins are checked out and built; the ledger is written
# to .mlxfast-private/e55/ledgers/ARM.json.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="$PWD"

arm="${1:?research/e55-ledger-run.sh: pass an arm name}"
tokens="${E55_LEDGER_TOKENS:-512}"
depth="${E55_LEDGER_DEPTH:-8}"

head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
# One shared golden for every arm: the timed run proved all three arms generate a
# byte-identical reference, so pinning one file removes the reference as a
# variable and makes any ledger difference attributable to the wide dispatch.
golden="${repo_root}/.mlxfast-private/e55/runs/base/reports/leg-1/02-mtp-verify-output.json"
out_dir="${repo_root}/.mlxfast-private/e55/ledgers"
out="${out_dir}/${arm}.json"
meta="${out_dir}/${arm}-meta.txt"

fail() { echo "e55-ledger-run: $*" >&2; exit 1; }

[[ -d "${head_dir}" ]] || fail "missing head tree ${head_dir}"
[[ -r "${golden}" ]] || fail "missing golden ${golden}"
mkdir -p "${out_dir}"

# The worker is the scored binary and the only one that embeds the kernel JIT
# source. Reuse e42-run.sh's two-root recipe verbatim: a plain `swift build`
# does not touch the .build-worker twin, so a single-root build would leave the
# worker holding the previous arm's kernel.
echo "=== e55-ledger-run: ${arm}: build ==="
mkdir -p .build/clang-module-cache .build-worker/clang-module-cache
CLANG_MODULE_CACHE_PATH="${PWD}/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift \
  || fail "mlxfast-swift build failed"
CLANG_MODULE_CACHE_PATH="${PWD}/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker \
  || fail "mlxfast-runtime-worker build failed"

# `quantized.h` is a Metal header: `quantized.metal` and `fp_quantized.metal`
# both include it, so an edit to it changes BOTH the JIT twin and the metallib.
# benchmark-qwen-mtp.sh detects the resulting staleness and rebuilds, which is
# why every timed arm ran with a metallib matching its own sources. A direct CLI
# invocation gets no such rebuild, so without this step the run would mix this
# arm's JIT source with the previous arm's metallib -- a configuration no timed
# arm was in. Rebuild unconditionally: it is idempotent and cheap when current.
echo "=== e55-ledger-run: ${arm}: rebuild mlx.metallib for this arm ==="
tools/build-mlx-metallib.sh --all-build-roots || fail "metallib rebuild failed"

echo "=== e55-ledger-run: ${arm}: assert the worker holds this arm's source ==="
research/e55_binary_assert.sh | tee "${meta}" || fail "binary assert failed"

rows_available="$(python3 -c "
import json,sys
print(len(json.load(open('${golden}'))['rows']))
")"
(( rows_available >= tokens + 1 )) \
  || fail "golden carries ${rows_available} rows; need $((tokens + 1))"

{
  echo "arm=${arm}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tokens=${tokens}"
  echo "depth=${depth}"
  echo "golden=${golden}"
  echo "golden_rows=${rows_available}"
  echo "golden_sha256=$(shasum -a 256 "${golden}" | cut -d' ' -f1)"
  echo "head_dir=${head_dir}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "worker_sha256=$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | cut -d' ' -f1)"
  echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | cut -d' ' -f1)"
  # Both the source fingerprint and the built artefact, so a reader can tell a
  # matched arm from a mixed one without rerunning anything.
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
  for lib in .build-worker/release/mlx.metallib \
             .build-worker/arm64-apple-macosx/release/mlx.metallib; do
    [[ -e "${lib}" ]] && echo "metallib_sha256[${lib}]=$(shasum -a 256 "${lib}" | cut -d' ' -f1)"
  done
} >> "${meta}"

echo "=== e55-ledger-run: ${arm}: mtp-verify --golden (${tokens} tokens, depth ${depth}) ==="
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
.build/release/mlxfast-swift mtp-verify \
  --mtp-head "${head_dir}" \
  --golden "${golden}" \
  --tokens "${tokens}" \
  --mtp-depth "${depth}" > "${out}"
status=$?

{
  echo "verify_exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${meta}"

if ((status == 0)); then
  python3 -c "
import json
d = json.load(open('${out}'))
led = d.get('row_ledger', [])
print('e55-ledger-run: ${arm}: rows=%d declared=%s matched=%s parity=%s' % (
    len(led), d.get('declared_rows_total'), d.get('all_tokens_matched'),
    d.get('parity_all_ok')))
if not led:
    raise SystemExit('e55-ledger-run: ${arm}: NO row_ledger in the report')
"
  status=$?
fi

echo "e55-ledger-run: ${arm}: status=${status} ledger=${out}"
exit "${status}"
