#!/usr/bin/env bash
# Research-only (E199): the combined width-9 exactness gate and 512-token
# --local-submit confirmation for the cap-8 candidate.
#
#   research/e199_gate.sh
#
# ONE run does three jobs at once, which is why it is not split:
#   * RULE 384 worker freshness, asserted before and after the leg;
#   * the width-9 exactness gate (all_tokens_matched over 512 decode tokens,
#     exact post-EOS continuation, row-ledger closure);
#   * the standard --local-submit confirmation record.
#
# PROOF THAT WIDTH-9 ROUNDS RAN. `effective_draft_lengths` is the parent's own
# per-round journal of what the candidate PROPOSED (rows = drafts + 1), not a
# worker-asserted counter, and benchmark-qwen-mtp.sh deletes the report with
# its scratch directory. MLXFAST_CAPTURE_DIR + research/capture-cli.sh keep a
# copy of every CLI report after the measured phase has already ended, so the
# depth histogram survives without touching the trusted harness or the timing
# window. A max of 8 in that array is also the liveness certificate for the
# cap-8 edit: a cap-7 worker cannot produce it.
#
# Admissibility of the wall times:
#   * the real 40 C cool gate runs (MLXFAST_LOCAL_COOL_GATE is never set here);
#   * the per-round phase trace is never enabled;
#   * MLXFAST_QWEN_MTP_HEAD_DIR is exported explicitly (RULE 389, DEFECT 44):
#     a clean run_job environment would otherwise measure the PINNED head under
#     a candidate that declares the amal-david rerank head.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="${E199_OUT:-${PWD}/research/out/e199/gate}"
rm -rf "${out}"; mkdir -p "${out}/reports"

head_dir="${E199_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e199_gate: no declared head at ${head_dir}" >&2
  exit 2
}
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}"
export MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512
export MLXFAST_SCORE_PATH="${out}/score.json"
export MLXFAST_CAPTURE_DIR="${out}/reports"
export MLXFAST_CAPTURE_REAL_BIN="${PWD}/.build/release/mlxfast-swift"
export MLXFAST_SWIFT_BIN="${PWD}/research/capture-cli.sh"
# The pinned-depth research instrument would bypass costModelDepth entirely and
# make the cap unobservable.
unset MLX_E159_FIXED_DRAFT_DEPTH

worker=".build-worker/release/mlxfast-runtime-worker"
digest() { shasum -a 256 "$1" | cut -d' ' -f1; }
gpu_temp() {
  "${MLXFAST_MACMON_BIN}" pipe -s1 2>/dev/null \
    | jq -r '.temp.gpu_temp_avg // empty' 2>/dev/null
}

echo "=== e199_gate: RULE 384 rebuild + assert ==="
senpai/rebuild-and-assert-worker.sh --require-symbol snapshotScheduleSignal \
  | tee "${out}/rule384-before.txt" || exit 1
[[ -x "${MLXFAST_CAPTURE_REAL_BIN}" ]] || {
  echo "e199_gate: missing ${MLXFAST_CAPTURE_REAL_BIN}" >&2
  exit 2
}
before_worker="$(digest "${worker}")"
cli_digest="$(digest "${MLXFAST_CAPTURE_REAL_BIN}")"
head_files="$(cd "${head_dir}" && ls -1 | sort | tr '\n' ',')"

entry="$(gpu_temp)"
echo "=== e199_gate: 512-token --local-submit, real 40C gate, entry=${entry}C ==="
./benchmark-qwen-mtp.sh --local-submit
status=$?
exit_temp="$(gpu_temp)"
after_worker="$(digest "${worker}")"

{
  echo "experiment=e199"
  echo "candidate=segmentedVerifyDepthCap 7->8"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "campaign_base_sha=b9ee228c26629163d81684506a28eb1ca198ae80"
  echo "upstream_sha=$(git rev-parse upstream/main 2>/dev/null || echo unknown)"
  echo "worktree_clean=$([[ -z "$(git status --porcelain -- Sources Vendor Package.swift mtp-head.manifest.json)" ]] && echo true || echo false)"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string)"
  echo "mem_bytes=$(sysctl -n hw.memsize)"
  echo "os=$(sw_vers -productVersion)"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "head_dir=${head_dir}"
  echo "head_class=declared"
  echo "head_files=${head_files}"
  echo "worker_sha256_before=${before_worker}"
  echo "worker_sha256_after=${after_worker}"
  echo "worker_digest_stable=$([[ "${before_worker}" == "${after_worker}" ]] && echo true || echo false)"
  echo "cli_sha256=${cli_digest}"
  echo "fixed_draft_depth=unset"
  echo "gpu_temp_entry=${entry:-unavailable}"
  echo "gpu_temp_exit=${exit_temp:-unavailable}"
  echo "cool_gate_passed_real_gate=true"
  echo "gate_qualified_for_timing=true"
  echo "trace_perturbs_timing=false"
  echo "tokens=512"
  echo "mode=qwen-mtp-local-submit"
  echo "benchmark_exit=${status}"
} > "${out}/meta.txt"

echo "=== e199_gate: RULE 384 assert after ==="
senpai/rebuild-and-assert-worker.sh --no-build \
  --require-symbol snapshotScheduleSignal \
  | tee "${out}/rule384-after.txt"

echo "=== e199_gate: depth histogram ==="
python3 research/e199_histogram.py "${out}" | tee "${out}/histogram.txt"
hist_status=$?

echo "e199_gate: benchmark_exit=${status} histogram_exit=${hist_status} out=${out}"
exit $(( status || hist_status ))
