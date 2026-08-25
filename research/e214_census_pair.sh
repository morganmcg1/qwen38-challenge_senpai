#!/usr/bin/env bash
# Research-only (qwen38-r1-e214): the SAME-HOST base/candidate schedule census
# pair the advisor asked for.
#
#   usage: research/e214_census_pair.sh [TOKENS]      # default 512
#          research/e214_stage_arms.sh must have run first.
#
# WHY. This experiment's headline IS the schedule delta. FINDING 545's base
# census (76 rounds, EDL 6.855263, acc 0.836852) was measured on another M4
# Pro, so on this host it is INFERRED (RULE 386). Both arms are measured here,
# back to back, in one session, so the delta carries no cross-host label.
#
# WHAT EACH ARM IS. `e214_stage_arms.sh` built one binary from the branch tree
# and one from a tree whose submitted surface is byte-identical to BASE_SHA.
# This script only INSTALLS a staged binary and runs a leg, so the worktree
# stays clean from start to finish. Three assertions stop a leg from being
# labelled as an arm it did not run:
#
#   1. the installed worker digest must equal the staged manifest digest;
#   2. the binary must PRINT the price arm the leg claims;
#   3. the worker digest must be unchanged after the leg, so a rebuild inside
#      the wrapper cannot silently swap the arm mid-leg.
#
# NOT A PRICE (RULE 79). Both legs run traced, unsandboxed and ungated, so
# every second in them is discarded. The legs count rounds, depth and
# acceptance. Timing evidence comes from the separate gated confirmation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
stage="${MLXFAST_E214_WORKERS:-$(cd ../.. && pwd)/e214-workers}"
worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
cli=".build/release/mlxfast-swift"
out_root="research/out"

head_dir="${E214_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e214_census_pair.sh: no head at ${head_dir}" >&2; exit 2; }
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e214_census_pair.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }

for arm in base cand; do
  [[ -s "${stage}/${arm}/manifest.txt" ]] || {
    echo "e214_census_pair.sh: no staged ${arm} arm; run e214_stage_arms.sh" >&2
    exit 2; }
done

digest() { shasum -a 256 "$1" | awk '{print $1}'; }
manifest_field() { grep "^$2=" "${stage}/$1/manifest.txt" | tail -1 | cut -d= -f2; }

install_arm() {
  local arm="$1" want got
  cp "${stage}/${arm}/mlxfast-runtime-worker" "${worker}"
  cp "${stage}/${arm}/mlx.metallib" "${metallib}"
  cp "${stage}/${arm}/mlxfast-swift" "${cli}"
  want="$(manifest_field "${arm}" worker_sha256)"
  got="$(digest "${worker}")"
  [[ "${want}" == "${got}" ]] || {
    echo "e214_census_pair.sh: installed ${arm} worker ${got} != staged ${want}" >&2
    return 3; }
}

# The candidate binaries must be back in place however this script ends.
trap 'install_arm cand >/dev/null 2>&1 || true' EXIT

gpu_temp() {
  local macmon
  for macmon in "${MLXFAST_MACMON_BIN:-}" "${HOME}/bin/macmon" \
                /opt/homebrew/bin/macmon /usr/local/bin/macmon; do
    [[ -n "${macmon}" && -x "${macmon}" ]] || continue
    "${macmon}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
    return 0
  done
  echo ""
}

run_leg() {
  local arm="$1" want_arm="$2" out="${out_root}/e214-${1}-census"
  rm -rf "${out}"
  mkdir -p "${out}"

  local trace_path="${PWD}/${out}/trace.txt"
  : > "${trace_path}"

  (
    export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
    export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
    export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
    export MLXFAST_NO_SANDBOX=1
    export MLXFAST_LOCAL_COOL_GATE=0
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"

    {
      echo "tag=e214-${arm}-census"
      echo "experiment=e214-step-price-implementation"
      echo "arm=${arm}"
      echo "expected_price_arm=${want_arm}"
      echo "harness=local"
      echo "tokens=${tokens}"
      echo "local_mode=--local-iterate"
      echo "sandbox=off"
      echo "trace=1"
      echo "cool_gate=0"
      echo "cool_gate_passed_real_gate=false"
      echo "gate_qualified_for_timing=false"
      echo "official_or_ranked_score=false"
      echo "seconds_per_token_is_a_price=false"
      echo "branch_sha=$(git rev-parse HEAD)"
      echo "arm_session_blob=$(manifest_field "${arm}" session_blob)"
      echo "arm_surface_equals_base=$(manifest_field "${arm}" surface_equals_base)"
      echo "host=$(hostname)"
      echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
      echo "memory_bytes=$(sysctl -n hw.memsize)"
      echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
      echo "worker_sha256=$(digest "${worker}")"
      echo "metallib_sha256=$(digest "${metallib}")"
      echo "cli_sha256=$(digest "${cli}")"
      echo "gpu_temp_entry_c=$(gpu_temp)"
      echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${out}/meta.txt"

    ./benchmark-qwen-mtp.sh --local-iterate \
      > "${out}/wrapper.out" 2> "${out}/wrapper.err"
    echo $? > "${out}/exit"
  )

  local status printed post
  status="$(cat "${out}/exit")"
  printed="$(grep -m1 -o 'mtp-price: arm=[a-z0-9]*' "${trace_path}" \
             | sed 's/.*arm=//' || true)"
  post="$(digest "${worker}")"
  {
    echo "gpu_temp_exit_c=$(gpu_temp)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
    echo "printed_price_arm=${printed}"
    echo "post_run_worker_sha256=${post}"
    echo "exit=${status}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"

  [[ "${printed}" == "${want_arm}" ]] || {
    echo "e214_census_pair.sh: arm ${arm} printed '${printed}', want '${want_arm}'" >&2
    return 3; }
  [[ "${post}" == "$(manifest_field "${arm}" worker_sha256)" ]] || {
    echo "e214_census_pair.sh: the ${arm} worker changed during the leg" >&2
    return 3; }
  return "${status}"
}

echo "=== arm base"
install_arm base || exit $?
run_leg base ship || exit $?

echo "=== arm cand"
install_arm cand || exit $?
run_leg cand stepq || exit $?

echo "=== done"
grep -H '^printed_price_arm=\|^worker_sha256=\|^trace_rounds=' \
  "${out_root}"/e214-{base,cand}-census/meta.txt
