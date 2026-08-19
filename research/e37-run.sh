#!/usr/bin/env bash
# Research-only (qwen38-r1-e37-draft-width-census-beagle-medicine): drive the
# NON-TIMED dispatched-width census at the scored geometry.
#
#   research/e37-run.sh --golden ID [ID ...]   512-step serial goldens
#   research/e37-run.sh --census ID [ID ...]   512-token --local-iterate census
#
# COUNTS ONLY, and MLXFAST_LOCAL_COOL_GATE=0. No number produced by this script
# is a timing measurement; meta.txt records that verbatim so no downstream
# reader can mistake one for a gate-qualified figure.
#
# E37_TRACE=1 additionally turns on the per-round phase trace, which buys the
# live cap/streak/EMA signal at the cost of file I/O inside the round. The
# per-round width census does NOT need it: `effective_draft_lengths` is written
# by the TRUSTED parent (QwenRuntimeMTPDriver.swift:289) one element per round,
# so the default is trace OFF and the histogram carries no perturbation caveat.
#
# ID `benchfixture` censuses the benchmark's OWN default --local-iterate fixture
# (correctness_prompts/public_longcopy_gate_english_512_256.json) instead of a
# research prose proxy. Its 512-token seed is used verbatim; the timed legs are
# checked against reference rows this build generates for the full census
# window, exactly as for a proxy.
#
# Scored geometry, and where each part of it comes from:
#   512 decode tokens        MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS below
#   offered depth 8          benchmark-qwen-mtp.sh:141 default, = the ranked
#                            workflow's MLXFAST_QWEN_MTP_DEPTH: "8"
#   sdpaWidthWallDepthCap 5  shipped literal, Qwen36MTPBlockSession.swift:662
#   segmentedStreakGate 2    shipped literal, :697
#   segmentedVerifyDepthCap 8 shipped literal, :669
#   declared proposal head   mtp-head.manifest.json, staged by
#                            research/fetch-declared-head.sh (the ranked
#                            candidate leg's head, not the organizer-pinned one)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

repo_root="${PWD}"
root="${E37_ROOT:-${repo_root}/.mlxfast-private/e37}"
goldens_dir="${root}/goldens"
runs_root="${root}/runs"
tokens="${E37_TOKENS:-512}"
trace="${E37_TRACE:-0}"
# Decoupled from the census window so a short reachability probe can reuse the
# full-length golden instead of paying a second serial reference pass.
golden_steps="${E37_GOLDEN_STEPS:-512}"
head_dir="${E37_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"

bench_fixture="correctness_prompts/public_longcopy_gate_english_512_256.json"

prompt_file_for() {
  case "$1" in
    benchfixture) echo "${bench_fixture}" ;;
    english) echo "research/e11_prose_gate_english_512.txt" ;;
    *) echo "research/e17_prose_$1_512.txt" ;;
  esac
}

golden_for() {
  local stem
  case "$1" in
    benchfixture) echo "${bench_fixture}"; return ;;
  esac
  stem="$(basename "$(prompt_file_for "$1")" .txt)"
  echo "${goldens_dir}/${stem}_${golden_steps}.json"
}

macmon_bin="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"
sample_thermal() {
  [[ -x "${macmon_bin}" ]] || { echo "unavailable"; return 0; }
  "${macmon_bin}" pipe -s1 2>/dev/null \
    | jq -r '"gpu_temp=\(.temp.gpu_temp_avg // "?")C cpu_temp=\(.temp.cpu_temp_avg // "?")C gpu_power=\(.gpu_power // "?")W"' \
      2>/dev/null || echo "unreadable"
}

mode="${1:?usage: research/e37-run.sh --golden|--census ID [ID ...]}"
shift
(($#)) || { echo "e37-run: need at least one prompt id" >&2; exit 2; }

for id in "$@"; do
  [[ -s "$(prompt_file_for "${id}")" ]] || {
    echo "e37-run: no prompt text for ${id}" >&2; exit 2; }
done

if [[ "${mode}" == "--golden" ]]; then
  files=()
  for id in "$@"; do
    [[ "${id}" == "benchfixture" ]] && {
      echo "e37-run: benchfixture ships its own fixture; no golden step" >&2; exit 2; }
    files+=("$(prompt_file_for "${id}")")
  done
  E11_GOLDEN_DIR="${goldens_dir}" E11_GOLDEN_STEPS="${golden_steps}" \
    research/e11-golden.sh "${files[@]}"
  exit $?
fi

[[ "${mode}" == "--census" ]] || {
  echo "e37-run: unknown mode ${mode}" >&2; exit 2; }

[[ -s "${head_dir}/config.json" && -s "${head_dir}/model.safetensors" ]] || {
  echo "e37-run: declared head run tree missing at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 2; }

status=0
for id in "$@"; do
  golden="$(golden_for "${id}")"
  [[ -s "${golden}" ]] || {
    echo "e37-run: missing golden ${golden}; run --golden ${id} first" >&2
    status=2; break; }
  covered="$(python3 -c 'import json,sys; print(min(len(c["expected_tokens"]) for c in json.load(open(sys.argv[1]))["cases"]))' "${golden}")"
  # expected_tokens feeds ONLY benchmark-qwen-mtp.sh:519's drift tripwire. Both
  # timed legs are checked against reference rows the build generates for the
  # full window (:556, --generate tokens+1), so short coverage weakens the
  # tripwire, not the census exactness. The shipped fixture is deliberately
  # short and is the one thorfinn's four E33 arms already passed on this box.
  if ((covered < tokens)); then
    if [[ "${id}" == "benchfixture" ]]; then
      echo "e37-run: ${id}: tripwire covers ${covered} of ${tokens} decode tokens (shipped fixture)" >&2
    else
      echo "e37-run: golden ${golden} covers ${covered} of ${tokens} decode tokens" >&2
      status=2; break
    fi
  fi

  out="${runs_root}/${id}"
  rm -rf "${out}"; mkdir -p "${out}/reports"

  if ((trace)); then
    # The worker sandbox denies file-write*, and the parent swallows worker
    # stderr, so the phase trace needs the documented local relaxation.
    export MLX_QWEN_MTP_TRACE=1
    export MLX_QWEN_MTP_TRACE_PATH="${out}/trace.txt"
    export MLXFAST_NO_SANDBOX=1
  else
    unset MLX_QWEN_MTP_TRACE MLX_QWEN_MTP_TRACE_PATH
  fi
  export MLXFAST_LOCAL_COOL_GATE=0
  export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
  export MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE="${golden}"
  export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
  export MLXFAST_SCORE_PATH="${out}/score.json"
  export MLXFAST_CAPTURE_REAL_BIN="${repo_root}/.build/release/mlxfast-swift"
  export MLXFAST_CAPTURE_DIR="${out}/reports"
  export MLXFAST_SWIFT_BIN="${repo_root}/research/capture-cli.sh"

  {
    echo "prompt_id=${id}"
    echo "prompt_file=$(prompt_file_for "${id}")"
    echo "prompt_sha256=$(shasum -a 256 "$(prompt_file_for "${id}")" | cut -d' ' -f1)"
    echo "tokens=${tokens}"
    echo "offered_depth=${MLXFAST_QWEN_MTP_DEPTH:-8}"
    echo "golden=${golden}"
    echo "golden_steps_covered=${covered}"
    echo "head_dir=${head_dir}"
    echo "head_safetensors_sha256=$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"
    echo "head_sha=$(git rev-parse HEAD)"
    echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
    # Read-only: NOT rebuilt here. A rebuild would change the resident kernel
    # under a sibling's timing lock, and it would break the kernel control that
    # makes these count runs comparable with each other and with E33's arms.
    echo "metallib_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null | tail -1)"
    echo "pass=$( ((trace)) && echo traced-census || echo untraced-census)"
    echo "phase_trace=${trace}"
    echo "cool_gate_passed_real_gate=false"
    echo "gate_qualified_for_timing=false"
    echo "timing_claims_permitted=false"
    echo "trace_perturbs_timing=$( ((trace)) && echo true || echo n/a)"
    echo "thermal_before=$(sample_thermal)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/meta.txt"

  echo "=== e37-run: census ${id} (${tokens} tokens, depth ${MLXFAST_QWEN_MTP_DEPTH:-8}) ==="
  ./benchmark-qwen-mtp.sh --local-iterate
  rc=$?
  {
    echo "exit=${rc}"
    echo "cli_sha256=$(shasum -a 256 "${repo_root}/.build/release/mlxfast-swift" | cut -d' ' -f1)"
    echo "worker_sha256=$(shasum -a 256 "${repo_root}/.build-worker/release/mlxfast-runtime-worker" | cut -d' ' -f1)"
    echo "thermal_after=$(sample_thermal)"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "${out}/meta.txt"
  if ((rc != 0)); then
    echo "e37-run: ${id}: benchmark exited ${rc}" >&2
    status=1; break
  fi
done
exit "${status}"
