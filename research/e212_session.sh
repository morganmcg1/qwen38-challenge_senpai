#!/usr/bin/env bash
# Research-only (qwen38-r1-e212-width-cost-law): one counterbalanced sweep of
# the per-round verify cost law R_local(m) on the CURRENT composed surface
# (cap-8 + staged (9,5)).
#
#   research/e212_session.sh SESSION_TAG ARM [ARM ...]
#
# Each ARM is either:
#   wN   pin the proposed draft count at N (MLX_E159_FIXED_DRAFT_DEPTH=N), so
#        every drafting round of the MTP leg verifies width m = N + 1.
#   ref  the shipped adaptive schedule, unpinned. This is the same-host
#        shipped-schedule reference leg and it supplies the served-width
#        distribution the step test is round-weighted with.
#   bN   wN plus MLX_QWEN_MTP_TRACE_SYNC_HEAD=1 and MLX_E182_BAND_SYNC=1: the
#        head chain drains before the verify build and the verify forward
#        drains at every mixer and MLP boundary, so `band_*_us` are per-layer
#        family device times. ATTRIBUTION ONLY -- 129 extra syncs per forward
#        inflate the round, so a b-arm round time is never a round cost.
#
# The caller passes the whole session order, so a palindrome such as
#   ref w1 w2 ... w9 w9 ... w2 w1 ref
# places both legs of every width symmetrically about the session midpoint and
# monotone thermal drift cancels to first order.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 under the standing three
# conditions: counterbalanced order, entry/exit GPU temperature per leg, and
# cool_gate_passed_real_gate=false / gate_qualified_for_timing=false preserved
# verbatim in each leg's meta.txt. Tracing perturbs the round on top of that.
#
# The scored surface is UNCHANGED: the forced-width pin and the phase trace are
# both research instruments that already exist in the base and are off unless
# their environment variables are set. This session builds nothing and edits
# nothing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: research/e212_session.sh SESSION_TAG ARM [ARM ...]}"
shift
(($#)) || { echo "e212: no arms given" >&2; exit 2; }
arms=("$@")

tokens="${E212_TOKENS:-512}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"

digest() { shasum -a 256 "${worker}" | cut -d' ' -f1; }
baseline_digest="$(digest)"

echo "e212: session=${session} tokens=${tokens} legs=${#arms[@]}"
echo "e212: order ${arms[*]}"
echo "e212: worker ${baseline_digest}"
echo "e212: head $(git rev-parse HEAD)"

status=0
leg=0
for arm in "${arms[@]}"; do
  leg=$((leg + 1))
  tag="$(printf 'e212/%s/leg%02d-%s' "${session}" "${leg}" "${arm}")"
  out="research/out/${tag}"

  unset MLX_E159_FIXED_DRAFT_DEPTH MLX_QWEN_MTP_TRACE_SYNC_HEAD MLX_E182_BAND_SYNC
  declare -a legopts=()
  case "${arm}" in
    ref) width="adaptive" ;;
    w[0-9])
      export MLX_E159_FIXED_DRAFT_DEPTH="${arm#w}"
      width="$((${arm#w} + 1))"
      ;;
    b[0-9])
      export MLX_E159_FIXED_DRAFT_DEPTH="${arm#b}"
      export MLX_E182_BAND_SYNC=1
      legopts+=(--sync-head)
      width="$((${arm#b} + 1))"
      ;;
    *) echo "e212: unknown arm ${arm}" >&2; status=2; break ;;
  esac

  before="$(digest)"
  [[ "${before}" == "${baseline_digest}" ]] || {
    echo "e212: worker changed before leg ${leg}: ${before}" >&2
    status=1; break
  }

  echo "=== e212: leg ${leg}/${#arms[@]} arm=${arm} m=${width} $(date -u +%H:%M:%SZ) ==="
  research/e79_trace_leg.sh "${tag}" "${tokens}" "${legopts[@]+"${legopts[@]}"}"
  rc=$?

  after="$(digest)"
  {
    echo "experiment=e212-width-cost-law"
    echo "e212_session=${session}"
    echo "e212_arm=${arm}"
    echo "e212_leg=${leg}"
    echo "e212_session_order=${arms[*]}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH:-unset}"
    echo "verify_width_m=${width}"
    echo "band_sync=${MLX_E182_BAND_SYNC:-0}"
    echo "worker_sha256_after=${after}"
    echo "worker_digest_stable=$([[ "${before}" == "${after}" ]] && echo true || echo false)"
    echo "timing_claims_permitted=false"
    echo "trace_perturbs_timing=true"
  } >> "${out}/meta.txt"

  if ((rc != 0)); then
    echo "e212: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -5 "${out}/wrapper.err" 2>/dev/null >&2
    status=1; break
  fi
  matched="$(jq -r '.metrics.all_tokens_matched' "${out}/score.json" 2>/dev/null)"
  if [[ "${matched}" != "true" ]]; then
    echo "e212: leg ${leg} (${arm}) all_tokens_matched=${matched}; stopping" >&2
    status=1; break
  fi
  echo "e212: leg ${leg} ok m=${width} mtp_spt=$(
    jq -r '.metrics.mtp_seconds_per_token' "${out}/score.json") edl=$(
    jq -r '.metrics.effective_mean_draft_len' "${out}/score.json") rounds=$(
    grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
done
exit "${status}"
