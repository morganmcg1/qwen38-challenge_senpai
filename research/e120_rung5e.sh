#!/usr/bin/env bash
# E120 rung 5e -- end-to-end native-MTP decode with and without the
# candidate-owned QMV dispatch, ABBA counterbalanced inside one session.
#
#   usage: research/e120_rung5e.sh TAG [tokens] [depth] [order]
#
#   TAG     output directory name under research/out/
#   tokens  decode tokens per leg, default 512
#   depth   offered draft ceiling, default 8 (the ranked workflow's value,
#           fixtures/qwen3_8_27b_mtp_track.json offered_draft_depth_ceiling)
#   order   arm sequence of Qwen35CustomQMV.Arm raw values,
#           default `off,sumtable,sumtable,off`
#
# Both arms run the SAME binary. `off` makes Qwen35CustomQMV.matmul decline
# every cell, so the routed call sites fall through to MLX's own quantizedMM
# and the arm reproduces BASE_SHA behaviour on the QMV path. That removes the
# build-identity confound a two-worktree comparison would carry.
#
# The wrapper's own 40C cool gate runs before every timed phase and is NOT
# bypassed here, so these arms are gate qualified. They are still local: one
# public fixture, candidate-generated reference rows, no hidden pool, so they
# are never an official or ranked score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e120_rung5e.sh TAG [tokens] [depth] [order]}"
tokens="${2:-512}"
depth="${3:-8}"
order="${4:-off,sumtable,sumtable,off}"

# `Qwen35CustomQMV.arm` falls back to `sumtable` for any raw value it does not
# know, so a typo here would silently time the shipped arm twice.
for arm in ${order//,/ }; do
  case "${arm}" in
    off|replica|fill_noconsume|sumtable) ;;
    *) echo "e120_rung5e.sh: unknown arm '${arm}'" >&2; exit 2 ;;
  esac
done
out_dir="research/out/${tag}"
mkdir -p "${out_dir}"

# The wrapper defaults MLXFAST_QWEN_MTP_HEAD_DIR to setup-qwen-mtp.sh's
# organizer-pinned bf16 cache. The current base declares the promoted remote
# head in mtp-head.manifest.json, and the promoted attachment code cannot load
# the pinned tree: the worker dies with
# `keyNotFound(["mtp","pre_fc_norm_hidden","weight"])` before it answers the
# parent's first request. Point the wrapper at the declared head every other
# campaign leg uses.
head_dir="${E120_HEAD_DIR:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run}"
if [[ ! -s "${head_dir}/config.json" ]]; then
  echo "e120_rung5e.sh: no head at ${head_dir}; run research/fetch-declared-head.sh" >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_HEAD_DIR="${head_dir}"
head_sha256="$(shasum -a 256 "${head_dir}/model.safetensors" | cut -d' ' -f1)"

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

start_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
head_sha="$(git rev-parse HEAD)"
dirty="$(git status --porcelain | wc -l | tr -d ' ')"

index=0
status=0
IFS=',' read -r -a arms <<< "${order}"
for arm in "${arms[@]}"; do
  index=$((index + 1))
  label="$(printf '%02d-%s' "${index}" "${arm}")"
  entry_c="$(gpu_temp)"
  arm_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== rung 5e arm ${label}: MLX_E120_QMV_ARM=${arm} tokens=${tokens} depth=${depth} entry=${entry_c}C"

  MLX_E120_QMV_ARM="${arm}" \
  MLXFAST_QWEN_MTP_DEPTH="${depth}" \
  MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}" \
  MLXFAST_SCORE_PATH="${PWD}/${out_dir}/score.${label}.json" \
  ./benchmark-qwen-mtp.sh --local-iterate 2>&1 | tee "${out_dir}/leg.${label}.log"
  arm_status="${PIPESTATUS[0]}"
  [[ "${arm_status}" -eq 0 ]] || status="${arm_status}"

  exit_c="$(gpu_temp)"
  arm_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  jq -n --arg arm "${arm}" --arg label "${label}" \
        --arg entry "${entry_c}" --arg exit "${exit_c}" \
        --arg started "${arm_start}" --arg finished "${arm_end}" \
        --argjson status "${arm_status}" \
    '{arm: $arm, label: $label, gpu_temp_entry_c: ($entry | tonumber? // null),
      gpu_temp_exit_c: ($exit | tonumber? // null),
      started_utc: $started, finished_utc: $finished, status: $status}' \
    >> "${out_dir}/arms.jsonl"
done

{
  echo "experiment=qwen38-r1-e120-own-the-qmv-dispatch"
  echo "rung=5e"
  echo "harness=local"
  echo "tokens=${tokens}"
  echo "offered_draft_depth=${depth}"
  echo "order=${order}"
  echo "head_dir=${head_dir}"
  echo "head_safetensors_sha256=${head_sha256}"
  echo "started_utc=${start_iso}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=${head_sha}"
  echo "git_dirty=${dirty}"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "memory_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "toolchain=$(swift --version 2>&1 | head -1)"
  echo "cool_gate_passed_real_gate=true"
  echo "gate_qualified_for_timing=true"
  echo "official_or_ranked_score=false"
  echo "status=${status}"
} > "${out_dir}/meta.txt"

cat "${out_dir}/meta.txt"
exit "${status}"
