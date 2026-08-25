#!/usr/bin/env bash
# Research-only (qwen38-r1-e217-staged-dequant): one counterbalanced sweep of
# the `G == 2` wide-QMV grid mapping on the CURRENT composed surface.
#
#   research/e217_session.sh SESSION_TAG ARM [ARM ...]
#
# Each ARM selects `MLX_E217_QMV_MAP` for one leg of the SAME binary:
#   off      the shipped `split` mapping. One threadgroup owns eight output
#            rows and ONE token column group, so a two-pass width launches the
#            same weight bytes twice in two different threadgroups.
#   coop     both column groups in one threadgroup over four output rows,
#            sharing nothing explicitly. E216 measured this and it LOST
#            (FINDING 565). This arm is a cross-host replication only.
#   staged   the shipped eight output rows over four simdgroups, with one raw
#            packed eight-row weight tile in threadgroup memory
#            (staticThreadgroupMemoryLength = 2048). This is the arm under test.
#
# The caller passes the whole session order, so a palindrome such as
#   off coop staged staged coop off
# places both legs of every arm symmetrically about the session midpoint and
# monotone thermal drift cancels to first order.
#
# The mapping is read once per process from the environment, so every leg runs
# the SAME worker binary. The session digest-guards that binary before and
# after every leg (HARNESS DEFECT 46) and stops on the first mismatch.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 under the standing three
# conditions: counterbalanced order, entry/exit GPU temperature per leg, and
# cool_gate_passed_real_gate=false / gate_qualified_for_timing=false preserved
# verbatim in each leg's meta.txt. Tracing perturbs the round on top of that.
#
# The scored surface is UNCHANGED: with `MLX_E217_QMV_MAP` unset the launch
# takes the shipped `split` mapping. This session builds nothing and edits
# nothing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: research/e217_session.sh SESSION_TAG ARM [ARM ...]}"
shift
(($#)) || { echo "e217: no arms given" >&2; exit 2; }
arms=("$@")

tokens="${E217_TOKENS:-512}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"

digest() { shasum -a 256 "${worker}" | cut -d' ' -f1; }
baseline_digest="$(digest)"

echo "e217: session=${session} tokens=${tokens} legs=${#arms[@]}"
echo "e217: order ${arms[*]}"
echo "e217: worker ${baseline_digest}"
echo "e217: head $(git rev-parse HEAD)"

status=0
leg=0
for arm in "${arms[@]}"; do
  leg=$((leg + 1))
  tag="$(printf 'e217/%s/leg%02d-%s' "${session}" "${leg}" "${arm}")"
  out="research/out/${tag}"

  unset MLX_E217_QMV_MAP
  case "${arm}" in
    off) mapping="split" ;;
    coop) mapping="coop"; export MLX_E217_QMV_MAP=coop ;;
    staged) mapping="staged"; export MLX_E217_QMV_MAP=staged ;;
    *) echo "e217: unknown arm ${arm}" >&2; status=2; break ;;
  esac

  before="$(digest)"
  [[ "${before}" == "${baseline_digest}" ]] || {
    echo "e217: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }

  echo "=== e217: leg ${leg}/${#arms[@]} arm=${arm} map=${mapping} $(date -u +%H:%M:%SZ) ==="
  research/e79_trace_leg.sh "${tag}" "${tokens}"
  rc=$?

  after="$(digest)"
  {
    echo "experiment=e217-staged-dequant"
    echo "e217_session=${session}"
    echo "e217_arm=${arm}"
    echo "e217_leg=${leg}"
    echo "e217_session_order=${arms[*]}"
    echo "qmv_group_mapping=${mapping}"
    echo "MLX_E217_QMV_MAP=${MLX_E217_QMV_MAP:-unset}"
    echo "worker_sha256_before=${before}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "timing_claims_permitted=false"
    echo "trace_perturbs_timing=true"
  } >> "${out}/meta.txt"

  if [[ "${before}" != "${after}" ]]; then
    echo "e217: worker digest moved across leg ${leg}: ${before} -> ${after}" >&2
    status=1; break
  fi

  if ((rc != 0)); then
    echo "e217: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -5 "${out}/wrapper.err" 2>/dev/null >&2
    status=1; break
  fi
  matched="$(jq -r '.metrics.all_tokens_matched' "${out}/score.json" 2>/dev/null)"
  if [[ "${matched}" != "true" ]]; then
    echo "e217: leg ${leg} (${arm}) all_tokens_matched=${matched}; stopping" >&2
    status=1; break
  fi

  # The mapping witness: only a `G == 2` width can leave the shipped mapping,
  # so these counters prove which mapping every two-pass dispatch really took.
  counters="$(grep -o 'qmv_split_g2=[0-9]* qmv_coop=[0-9]* qmv_staged=[0-9]*' \
    "${out}/trace.txt" 2>/dev/null | tail -1)"
  echo "e217_last_round_mapping_counters=${counters}" >> "${out}/meta.txt"
  echo "e217: leg ${leg} ok map=${mapping} mtp_spt=$(
    jq -r '.metrics.mtp_seconds_per_token' "${out}/score.json") edl=$(
    jq -r '.metrics.effective_mean_draft_len' "${out}/score.json") rounds=$(
    grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0) ${counters}"
done
exit "${status}"
