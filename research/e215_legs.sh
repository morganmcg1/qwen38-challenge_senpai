#!/usr/bin/env bash
# Research-only (qwen38-r1-e215): the untimed 512-token evidence legs.
#
#   usage: research/e215_legs.sh [PROMPT ...]      # default: all five
#          research/e215_stage_arms.sh must have run first.
#
# Each leg is one `--local-iterate` run at 512 decode tokens with the declared
# head (HARNESS DEFECT 44), traced, unsandboxed and ungated. Every leg produces
# both kinds of evidence this experiment needs:
#
#   Q1 exactness -- `all_tokens_matched`, `residual_divergence_count`,
#      `target_cache_offset_final` and the row ledger from the captured CLI
#      reports, plus the position of every stop token in the reference chain;
#   Q2 schedule census -- rounds, effective draft lengths and acceptance.
#
# NOT A PRICE (RULE 79). Ungated, traced, unsandboxed: every second measured in
# these legs is discarded. Entry and exit GPU temperature are still recorded,
# and `cool_gate_passed_real_gate=false` / `gate_qualified_for_timing=false`
# are written verbatim.
#
# ARM CERTIFICATION (RULE 402). A leg is only labelled with the arm the binary
# PRINTS. Three assertions: the installed worker digest equals the staged
# manifest digest; the trace prints the expected `arm=`; the worker digest is
# unchanged after the leg, so a rebuild inside the wrapper cannot swap the arm
# mid-leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${E215_TOKENS:-512}"
arms="${E215_ARMS:-stepq ship}"
stage="${MLXFAST_E215_WORKERS:-$(cd ../.. && pwd)/e215-workers}"
worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
cli=".build/release/mlxfast-swift"
out_root="research/out"

prompts=("$@")
if [[ ${#prompts[@]} -eq 0 ]]; then
  prompts=(public e215-eos-short e215-eos-para e215-code e215-story)
fi

fixture_for() {
  case "$1" in
    public) echo "correctness_prompts/public_longcopy_gate_english_512_256.json" ;;
    e215-eos-para) echo "research/e215-artifacts/fixtures/e215-eos-para_512_192.json" ;;
    *) echo "research/e215-artifacts/fixtures/$1_512_64.json" ;;
  esac
}

head_dir="${E215_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
[[ -s "${head_dir}/config.json" ]] || {
  echo "e215_legs.sh: no declared head at ${head_dir}" >&2; exit 2; }
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"

[[ -z "$(git status --porcelain)" ]] || {
  echo "e215_legs.sh: worktree is dirty; commit or discard first" >&2
  exit 2; }

for arm in ${arms}; do
  [[ -s "${stage}/${arm}/manifest.txt" ]] || {
    echo "e215_legs.sh: no staged ${arm} arm; run research/e215_stage_arms.sh" >&2
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
    echo "e215_legs.sh: installed ${arm} worker ${got} != staged ${want}" >&2
    return 3; }
}

trap 'install_arm stepq >/dev/null 2>&1 || true' EXIT

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
  local prompt="$1" arm="$2" fixture out
  fixture="$(fixture_for "${prompt}")"
  out="${out_root}/e215-${prompt}-${arm}"
  [[ -s "${fixture}" ]] || {
    echo "e215_legs.sh: missing fixture ${fixture}" >&2; return 2; }
  rm -rf "${out}"
  mkdir -p "${out}/reports"

  local trace_path="${PWD}/${out}/trace.txt"
  : > "${trace_path}"

  (
    export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
    export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
    export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${fixture}"
    export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
    export MLXFAST_CAPTURE_DIR="${PWD}/${out}/reports"
    export MLXFAST_CAPTURE_REAL_BIN="${PWD}/${cli}"
    export MLXFAST_SWIFT_BIN="${PWD}/research/capture-cli.sh"
    export MLXFAST_NO_SANDBOX=1
    export MLXFAST_LOCAL_COOL_GATE=0
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"

    {
      echo "tag=e215-${prompt}-${arm}"
      echo "experiment=e215-schedule-evidence"
      echo "prompt=${prompt}"
      echo "fixture=${fixture}"
      echo "arm=${arm}"
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
      echo "base_sha=32a53a58fc7bb33f996299b63c49882db673c8cc"
      echo "arm_session_blob=$(manifest_field "${arm}" session_blob)"
      echo "host=$(hostname)"
      echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
      echo "memory_bytes=$(sysctl -n hw.memsize)"
      echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR}"
      echo "head_dir_sha256=$(
        find "${MLXFAST_QWEN_MTP_HEAD_DIR}" -type f -exec shasum -a 256 {} + \
          | awk '{print $1}' | sort | shasum -a 256 | awk '{print $1}')"
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
  printed="$(grep -m1 -o 'arm=[a-z0-9]*' "${trace_path}" | sed 's/.*arm=//' || true)"
  post="$(digest "${worker}")"
  {
    echo "gpu_temp_exit_c=$(gpu_temp)"
    echo "trace_rounds=$(grep -c '^mtp-trace: round=' "${trace_path}" || true)"
    echo "printed_price_arm=${printed}"
    echo "post_run_worker_sha256=${post}"
    echo "exit=${status}"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"

  [[ "${printed}" == "${arm}" ]] || {
    echo "e215_legs.sh: leg ${prompt}/${arm} printed '${printed}'" >&2
    return 3; }
  [[ "${post}" == "$(manifest_field "${arm}" worker_sha256)" ]] || {
    echo "e215_legs.sh: the ${arm} worker changed during ${prompt}" >&2
    return 3; }
  return "${status}"
}

for prompt in "${prompts[@]}"; do
  for arm in ${arms}; do
    echo "=== ${prompt} / ${arm}"
    install_arm "${arm}" || exit $?
    run_leg "${prompt}" "${arm}" || exit $?
    grep -H '^printed_price_arm=\|^trace_rounds=\|^exit=' \
      "${out_root}/e215-${prompt}-${arm}/meta.txt"
  done
done

echo "=== done"
