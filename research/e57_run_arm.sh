#!/usr/bin/env bash
# Research-only launcher for one E57 arm of ./benchmark-qwen-mtp.sh --local-iterate.
#
# run_job takes an argv list with no environment field, so the per-arm
# environment has to be established inside a script. Everything expensive still
# goes through ./benchmark-qwen-mtp.sh, so the run lock, the orphan check and
# the cool gate are never bypassed.
#
# usage:
#   research/e57_run_arm.sh TAG --chunk-arm base|narrow|off [--tokens N]
#                               [--row-trace] [--sdpa-trace] [--hot]
#
# --row-trace  turns on MLX_QWEN_MTP_TRACE, which emits the per-position
#              top-two row evidence the exactness rungs compare. It also adds
#              host-side timing snapshots, so it must stay off in a timed arm.
# --sdpa-trace turns on the E57 per-call SDPA fact trace.
# --hot        sets MLXFAST_LOCAL_COOL_GATE=0. Permitted for local timed arms
#              only inside one ABBA-counterbalanced session, and the result must
#              carry cool_gate_passed_real_gate=false and
#              gate_qualified_for_timing=false verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e57_run_arm.sh TAG --chunk-arm base|narrow|off [options]}"
shift

chunk_arm=""
tokens=512
row_trace=0
sdpa_trace=0
hot=0
while (($#)); do
  case "$1" in
    --chunk-arm) chunk_arm="$2"; shift 2 ;;
    --tokens) tokens="$2"; shift 2 ;;
    --row-trace) row_trace=1; shift ;;
    --sdpa-trace) sdpa_trace=1; shift ;;
    --hot) hot=1; shift ;;
    *) echo "e57_run_arm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "${chunk_arm}" in
  base|narrow|off) ;;
  *) echo "e57_run_arm.sh: --chunk-arm must be base, narrow or off" >&2; exit 2 ;;
esac

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

# The per-role $HOME shards the local run lock into one lock per student, which
# is worse than no lock because both peers then believe they hold the machine.
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"
export MLX_E57_SDPA_CHUNK_ARM="${chunk_arm}"

trace_path="${PWD}/${out}/trace.txt"
if ((row_trace || sdpa_trace)); then
  # The generated worker sandbox denies file-write*, and the mtp-timed parent
  # swallows worker stderr, so a readable trace needs the documented local
  # relaxation plus an O_APPEND path the three legs can share.
  export MLXFAST_NO_SANDBOX=1
  : > "${trace_path}"
fi
if ((row_trace)); then
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"
fi
if ((sdpa_trace)); then
  export MLX_E57_SDPA_TRACE=1
  export MLX_E57_SDPA_TRACE_PATH="${trace_path}"
fi
if ((hot)); then
  export MLXFAST_LOCAL_COOL_GATE=0
fi

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

# A previous job can leave a metallib built from another arm's patched sources
# in every build root, and nothing downstream of here rebuilds it.
tools/build-mlx-metallib.sh --all-build-roots

{
  echo "tag=${tag}"
  echo "chunk_arm=${chunk_arm}"
  echo "tokens=${tokens}"
  echo "row_trace=${row_trace}"
  echo "sdpa_trace=${sdpa_trace}"
  echo "cool_gate=$((1 - hot))"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "candidate_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
} > "${out}/meta.txt"

# Read the entry temperature AFTER the metallib check, so a Metal compile does
# not sit between the reading and the measured window.
{
  echo "gpu_temp_entry_c=$(gpu_temp)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"

./benchmark-qwen-mtp.sh --local-iterate > "${out}/wrapper.out" 2> "${out}/wrapper.err"
status=$?

{
  echo "gpu_temp_exit_c=$(gpu_temp)"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "${out}/meta.txt"
exit "${status}"
