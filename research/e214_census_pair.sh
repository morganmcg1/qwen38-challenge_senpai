#!/usr/bin/env bash
# Research-only (qwen38-r1-e214): the SAME-HOST base/candidate schedule census
# pair the advisor asked for.
#
#   usage: research/e214_census_pair.sh [TOKENS]      # default 512
#
# WHY. This experiment's headline IS the schedule delta. FINDING 545's base
# census (76 rounds, EDL 6.855263, acc 0.836852) was measured on another M4
# Pro, so on this host it is INFERRED (RULE 386). Both arms are measured here,
# back to back, in one session, so the delta carries no cross-host label.
#
# WHAT EACH ARM IS. The only submitted file this candidate changes is
# `Qwen36MTPBlockSession.swift`. Arm `base` checks that one file out at
# BASE_SHA, which makes the whole submitted surface byte-identical to the base,
# and rebuilds. Arm `cand` restores the branch file and rebuilds. The script
# asserts the surface, the built symbol, and the arm the binary PRINTS in each
# leg, so a leg can never be labelled as an arm it did not run.
#
# NOT A PRICE (RULE 79). Both legs run traced, unsandboxed and ungated, so
# every second in them is discarded. The legs exist to count rounds, depth and
# acceptance. Timing evidence comes from the separate gated confirmation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
base_sha="e0c7a026aebc9dbdd579577964ad413a11ede6fc"
session="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
surface=(Sources Vendor Package.swift mtp-head.manifest.json)
out_root="research/out"

head_dir="${E214_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e214_census_pair.sh: no head at ${head_dir}" >&2; exit 2; }
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

# A patched arm is only the arm it claims if the tree was clean before it.
[[ -z "$(git status --porcelain)" ]] || {
  echo "e214_census_pair.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }

changed="$(git diff --name-only "${base_sha}" -- "${surface[@]}")"
[[ "${changed}" == "${session}" ]] || {
  echo "e214_census_pair.sh: submitted surface differs from ${base_sha} in more" >&2
  echo "  than the session file, so the base arm cannot be made by one checkout:" >&2
  echo "${changed}" >&2
  exit 2; }

restore() {
  git checkout -- "${session}" 2>/dev/null || true
}
trap restore EXIT

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
      echo "base_sha=${base_sha}"
      echo "session_blob_sha=$(git hash-object "${session}")"
      echo "surface_equals_base=$(
        git diff --quiet "${base_sha}" -- "${surface[@]}" && echo true || echo false)"
      echo "host=$(hostname)"
      echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
      echo "memory_bytes=$(sysctl -n hw.memsize)"
      echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
      echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
      echo "worker_sha256=$(
        shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
      echo "cli_sha256=$(shasum -a 256 .build/release/mlxfast-swift | awk '{print $1}')"
      echo "gpu_temp_entry_c=$(gpu_temp)"
      echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${out}/meta.txt"

    ./benchmark-qwen-mtp.sh --local-iterate \
      > "${out}/wrapper.out" 2> "${out}/wrapper.err"
    echo $? > "${out}/exit"
  )

  local status
  status="$(cat "${out}/exit")"
  {
    echo "gpu_temp_exit_c=$(gpu_temp)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
    echo "printed_price_arm=$(
      grep -m1 -o 'mtp-price: arm=[a-z0-9]*' "${trace_path}" \
        | sed 's/.*arm=//' || true)"
    echo "exit=${status}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"

  local got
  got="$(grep '^printed_price_arm=' "${out}/meta.txt" | tail -1 | cut -d= -f2)"
  if [[ "${got}" != "${want_arm}" ]]; then
    echo "e214_census_pair.sh: arm ${arm} printed '${got}', expected '${want_arm}'" >&2
    return 3
  fi
  return "${status}"
}

echo "=== arm base: checking out ${session} at ${base_sha}"
git checkout "${base_sha}" -- "${session}"
git diff --quiet "${base_sha}" -- "${surface[@]}" || {
  echo "e214_census_pair.sh: base arm surface is not byte-identical to base" >&2
  exit 3; }
senpai/rebuild-and-assert-worker.sh \
  --forbid-symbol makeThresholdDepthPrice > "${out_root}/e214-base-build.log" 2>&1 || {
  echo "e214_census_pair.sh: base arm build failed; see ${out_root}/e214-base-build.log" >&2
  exit 3; }
run_leg base ship || exit $?

echo "=== arm cand: restoring the branch file"
git checkout -- "${session}"
[[ -z "$(git status --porcelain)" ]] || {
  echo "e214_census_pair.sh: tree not restored after the base arm" >&2; exit 3; }
senpai/rebuild-and-assert-worker.sh \
  --require-symbol makeThresholdDepthPrice > "${out_root}/e214-cand-build.log" 2>&1 || {
  echo "e214_census_pair.sh: candidate arm build failed" >&2
  exit 3; }
run_leg cand stepq || exit $?

echo "=== done"
grep -H '^printed_price_arm=\|^worker_sha256=\|^trace_rounds=' \
  "${out_root}"/e214-{base,cand}-census/meta.txt
