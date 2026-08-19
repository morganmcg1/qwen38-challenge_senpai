#!/usr/bin/env bash
# Research-only launcher for one E58 arm of ./benchmark-qwen-mtp.sh --local-iterate.
#
# run_job takes an argv list with no environment field, so the per-arm
# environment has to be established inside a script. Everything expensive still
# goes through ./benchmark-qwen-mtp.sh, so the run lock, the orphan check and
# the cool gate are never bypassed.
#
# usage:
#   research/e58_run_arm.sh TAG [--census] [--census-shapes] [--sync-head]
#                               [--tax N] [--tax-mode metal|mlx]
#                               [--tax-ops-per-buffer N] [--tokens N]
#                               [--trace] [--hot]
#
# --census      counts every GPU dispatch by round, verify width and phase. The
#               swizzle's lock cost makes the run unfit for timing: use it for
#               counts only.
# --sync-head   drains the head chain before the verify graph is built, so the
#               head's dispatches cannot be encoded inside the verify window.
#               Required for a clean head-versus-target split, and never valid
#               in a timed arm.
# --tax N       adds N trivial dispatches per round. Use in TIMED arms, with the
#               census OFF, to regress the marginal cost of one dispatch.
# --hot         sets MLXFAST_LOCAL_COOL_GATE=0. Permitted for local timed arms
#               only inside one ABBA-counterbalanced session, and the result must
#               carry cool_gate_passed_real_gate=false and
#               gate_qualified_for_timing=false verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: e58_run_arm.sh TAG [options]}"
shift

census=0
census_shapes=0
sync_head=0
tax=0
tax_mode=metal
tax_ops_per_buffer=64
tokens=512
row_trace=0
hot=0
while (($#)); do
  case "$1" in
    --census) census=1; shift ;;
    --census-shapes) census=1; census_shapes=1; shift ;;
    --sync-head) sync_head=1; shift ;;
    --tax) tax="$2"; shift 2 ;;
    --tax-mode) tax_mode="$2"; shift 2 ;;
    --tax-ops-per-buffer) tax_ops_per_buffer="$2"; shift 2 ;;
    --tokens) tokens="$2"; shift 2 ;;
    --trace) row_trace=1; shift ;;
    --hot) hot=1; shift ;;
    *) echo "e58_run_arm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "${tax_mode}" in
  metal|mlx) ;;
  *) echo "e58_run_arm.sh: --tax-mode must be metal or mlx" >&2; exit 2 ;;
esac
if ((census)) && ((tax > 0)); then
  echo "e58_run_arm.sh: the census perturbs timing, so a taxed arm must run" \
       "with the census OFF" >&2
  exit 2
fi

out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

# The per-role $HOME shards the local run lock into one lock per student, which
# is worse than no lock because both peers then believe they hold the machine.
export MLXFAST_LOCAL_RUN_LOCK_DIR="${MLXFAST_LOCAL_RUN_LOCK_DIR:-/tmp/mlxfast-shared}"
export MLXFAST_QWEN_MTP_LOCAL_ITERATE_TOKENS="${tokens}"
export MLXFAST_SCORE_PATH="${PWD}/${out}/score.json"

census_path="${PWD}/${out}/census.jsonl"
trace_path="${PWD}/${out}/trace.txt"
if ((census || row_trace)); then
  # The generated worker sandbox denies file-write*, and the mtp-timed parent
  # swallows worker stderr, so a readable dump needs the documented local
  # relaxation plus an O_APPEND path the three legs can share.
  export MLXFAST_NO_SANDBOX=1
fi
if ((census)); then
  : > "${census_path}"
  export MLX_E58_DISPATCH_CENSUS=1
  export MLX_E58_DISPATCH_CENSUS_PATH="${census_path}"
  ((census_shapes)) && export MLX_E58_DISPATCH_CENSUS_SHAPES=1
fi
if ((sync_head)); then
  export MLX_QWEN_MTP_TRACE_SYNC_HEAD=1
fi
if ((tax > 0)); then
  export MLX_E58_DISPATCH_TAX="${tax}"
  export MLX_E58_DISPATCH_TAX_MODE="${tax_mode}"
  export MLX_E58_DISPATCH_TAX_OPS_PER_BUFFER="${tax_ops_per_buffer}"
fi
if ((row_trace)); then
  : > "${trace_path}"
  export MLX_QWEN_MTP_TRACE=1
  export MLX_QWEN_MTP_TRACE_PATH="${trace_path}"
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

{
  echo "tag=${tag}"
  echo "census=${census}"
  echo "census_shapes=${census_shapes}"
  echo "sync_head=${sync_head}"
  echo "tax=${tax}"
  echo "tax_mode=${tax_mode}"
  echo "tax_ops_per_buffer=${tax_ops_per_buffer}"
  echo "tokens=${tokens}"
  echo "row_trace=${row_trace}"
  echo "cool_gate=$((1 - hot))"
  echo "metallib_source_fingerprint=$(tools/build-mlx-metallib.sh --print-fingerprint)"
  echo "head_dir=${MLXFAST_QWEN_MTP_HEAD_DIR:-<setup-default>}"
  echo "candidate_sha=$(git rev-parse HEAD)"
  echo "dirty=$(git status --porcelain | wc -l | tr -d ' ')"
} > "${out}/meta.txt"

# Read the entry temperature last, so nothing expensive sits between the
# reading and the measured window.
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
