#!/usr/bin/env bash
# Research-only (qwen38-r1-e218-width-tax-census): pinned-width sessions that
# measure WHO charges for each extra draft row outside the QMV pass structure.
#
#   research/e218_session.sh SESSION_TAG ARM [ARM ...]
#
# Each ARM is one 512-token `--local-iterate` leg of the SHIPPED surface:
#   wN   pin the proposed draft count at N (MLX_E159_FIXED_DRAFT_DEPTH=N), so
#        every drafting round verifies width m = N + 1. Phase trace on, band
#        sync OFF. These legs carry the round-cost law R_local(m).
#   bN   wN plus MLX_E182_BAND_SYNC=1 and --sync-head: the verify forward
#        drains at every mixer and MLP boundary, so `band_*_us` are per-layer
#        family device times. ATTRIBUTION ONLY -- the 129 extra drains per
#        forward inflate the round, so a b-arm round time is never a round
#        cost. Paired w/b legs at the same width bound that perturbation.
#   ref  the shipped adaptive schedule, unpinned (served-width census).
#
# The caller passes the whole session order, so a palindrome such as
#   w0 w2 w3 ... w9 w9 ... w3 w2 w0
# places both legs of every width symmetrically about the session midpoint and
# monotone thermal drift cancels to first order. The w0 legs are the serial
# null control (FINDING 503 per-session noise instrument) and the m=1 anchor.
#
# NOT GATE-QUALIFIED. MLXFAST_LOCAL_COOL_GATE=0 under the standing three
# conditions: counterbalanced order, entry/exit GPU temperature per leg, and
# cool_gate_passed_real_gate=false / gate_qualified_for_timing=false preserved
# verbatim in each leg's meta.txt.
#
# HARNESS DEFECT 46 discipline: the worker binary is digest-guarded before AND
# after every leg against the session baseline. A mid-session rebuild stops the
# session instead of silently retiming another binary.
#
# The scored surface is UNCHANGED: the forced-width pin and the band trace are
# research instruments already in the base, off unless their environment
# variables are set. This session builds nothing and edits nothing.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

session="${1:?usage: research/e218_session.sh SESSION_TAG ARM [ARM ...]}"
shift
(($#)) || { echo "e218: no arms given" >&2; exit 2; }
arms=("$@")

tokens="${E218_TOKENS:-512}"
worker="${PWD}/.build-worker/release/mlxfast-runtime-worker"
cli="${PWD}/.build/release/mlxfast-swift"
export MLXFAST_MACMON_BIN="${MLXFAST_MACMON_BIN:-${HOME}/bin/macmon}"

digest() { shasum -a 256 "$1" | cut -d' ' -f1; }
baseline_worker="$(digest "${worker}")"
baseline_cli="$(digest "${cli}")"

echo "e218: session=${session} tokens=${tokens} legs=${#arms[@]}"
echo "e218: order ${arms[*]}"
echo "e218: worker ${baseline_worker}"
echo "e218: cli    ${baseline_cli}"
echo "e218: head   $(git rev-parse HEAD)"
echo "e218: dirty scored paths $(
  git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"

status=0
leg=0
for arm in "${arms[@]}"; do
  leg=$((leg + 1))
  tag="$(printf 'e218/%s/leg%02d-%s' "${session}" "${leg}" "${arm}")"
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
    *) echo "e218: unknown arm ${arm}" >&2; status=2; break ;;
  esac

  before_worker="$(digest "${worker}")"
  before_cli="$(digest "${cli}")"
  if [[ "${before_worker}" != "${baseline_worker}" \
        || "${before_cli}" != "${baseline_cli}" ]]; then
    echo "e218: binary changed before leg ${leg}: worker ${before_worker}" >&2
    status=1; break
  fi

  echo "=== e218: leg ${leg}/${#arms[@]} arm=${arm} m=${width} $(date -u +%H:%M:%SZ) ==="
  research/e79_trace_leg.sh "${tag}" "${tokens}" "${legopts[@]+"${legopts[@]}"}"
  rc=$?

  after_worker="$(digest "${worker}")"
  after_cli="$(digest "${cli}")"
  {
    echo "experiment=e218-width-tax-census"
    echo "e218_session=${session}"
    echo "e218_arm=${arm}"
    echo "e218_leg=${leg}"
    echo "e218_session_order=${arms[*]}"
    echo "fixed_draft_depth=${MLX_E159_FIXED_DRAFT_DEPTH:-unset}"
    echo "verify_width_m=${width}"
    echo "band_sync=${MLX_E182_BAND_SYNC:-0}"
    echo "worker_sha256_after=${after_worker}"
    echo "cli_sha256_after=${after_cli}"
    echo "worker_digest_stable=$(
      [[ "${before_worker}" == "${after_worker}" ]] && echo true || echo false)"
    echo "cli_digest_stable=$(
      [[ "${before_cli}" == "${after_cli}" ]] && echo true || echo false)"
    echo "timing_claims_permitted=false"
    echo "trace_perturbs_timing=$([[ "${arm}" == b* ]] && echo true || echo false)"
  } >> "${out}/meta.txt"

  if [[ "${after_worker}" != "${baseline_worker}" \
        || "${after_cli}" != "${baseline_cli}" ]]; then
    echo "e218: binary changed DURING leg ${leg} (HARNESS DEFECT 46)" >&2
    status=1; break
  fi

  if ((rc != 0)); then
    echo "e218: leg ${leg} (${arm}) exited ${rc}" >&2
    tail -5 "${out}/wrapper.err" 2>/dev/null >&2
    status=1; break
  fi
  matched="$(jq -r '.metrics.all_tokens_matched' "${out}/score.json" 2>/dev/null)"
  if [[ "${matched}" != "true" ]]; then
    echo "e218: leg ${leg} (${arm}) all_tokens_matched=${matched}; stopping" >&2
    status=1; break
  fi
  echo "e218: leg ${leg} ok m=${width} mtp_spt=$(
    jq -r '.metrics.mtp_seconds_per_token' "${out}/score.json") edl=$(
    jq -r '.metrics.effective_mean_draft_len' "${out}/score.json") rounds=$(
    grep -c '^mtp-trace: round=' "${out}/trace.txt" 2>/dev/null || echo 0)"
done
exit "${status}"
