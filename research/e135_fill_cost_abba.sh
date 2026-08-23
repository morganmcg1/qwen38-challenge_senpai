#!/usr/bin/env bash
# F41 prep: price the `xsums_v1` fill dispatch with the schedule held fixed.
#
#   usage: research/e135_fill_cost_abba.sh [TOKENS] [LABEL]
#
# TWO ARMS, one worker, one thermal session, no rebuild between legs. Both arms
# hold the shipped composition (tight grid, onepass67 plan, p15 probe, width 2
# routed, depth arm ship) and differ only in `MLX_E120_QMV_ARM`:
#
#   repl   replica         the wide kernel recomputes the chunk sums itself
#   fill   fill_noconsume  the same kernel, plus a live fill dispatch whose
#                          output is bound and never read
#
# Both arms are bit-exact replicas of the incumbent, so they emit the same
# tokens and the same draft schedule. The contrast is therefore a pure cost
# measurement and Rule 79 does not bind on it. The report refuses to price the
# contrast if the two arms disagree on `effective_mean_draft_len`.
#
# PALINDROME. repl fill fill repl gives both arms mean position 2.5, so a
# monotone thermal or clock drift in leg index cancels to first order.
#
# GATED. Every timed leg runs the real 40 C gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
label="${2:-fill}"

if [[ -n "$(git status --porcelain -- Sources Vendor Package.swift)" ]]; then
  echo "e135_fill_cost_abba: scored surface is dirty; refusing to time over" \
       "uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
session_worker="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

apply_arm() {
  export MLX_E120_QMV_GRID=tight
  export MLX_E120_QMV_TABLE=onepass67
  export MLX_E135_PROBE_ARM=p15
  export MLX_E120_QMV_WIDTH2=1
  export MLX_E134_DEPTH_PRICE_ARM=ship
  case "$1" in
    repl) export MLX_E120_QMV_ARM=replica ;;
    fill) export MLX_E120_QMV_ARM=fill_noconsume ;;
    *) echo "e135_fill_cost_abba: unknown arm $1" >&2; exit 2 ;;
  esac
}

clear_arm() {
  unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E135_PROBE_ARM \
        MLX_E120_QMV_WIDTH2 MLX_E134_DEPTH_PRICE_ARM MLX_E120_QMV_ARM
}

anti_arm() {
  case "$1" in repl) echo fill ;; fill) echo repl ;; esac
}

# PHASE 0: warmup (Rule 137), ungated, and the `fill` witness at full length.
warm_tag="e135${label}warm"
echo "=== warmup ${warm_tag}: arm=fill tokens=${tokens} ungated ==="
apply_arm fill
export MLX_E120_QMV_PIPELINE_LOG="${PWD}/research/out/${warm_tag}/pipelines.json"
mkdir -p "research/out/${warm_tag}"
research/e79_trace_leg.sh "${warm_tag}" "${tokens}" --no-trace
warm_status=$?
unset MLX_E120_QMV_PIPELINE_LOG
clear_arm
echo "e135_arm=fill" >> "research/out/${warm_tag}/meta.txt"
echo "e135_leg_role=warmup" >> "research/out/${warm_tag}/meta.txt"
if ((warm_status != 0)); then
  echo "e135_fill_cost_abba: the warmup leg exited ${warm_status}" >&2
  exit 5
fi

# PHASE 1: prove the selector reaches the worker, on both arms, with a control
# that must fail. Rule 101.
for arm in fill repl; do
  if [[ "${arm}" == "fill" ]]; then
    out="research/out/${warm_tag}"
  else
    tag="e135${label}w${arm}"
    out="research/out/${tag}"
    echo "=== witness ${tag}: arm=${arm} tokens=16 ==="
    mkdir -p "${out}"
    apply_arm "${arm}"
    export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
    research/e79_trace_leg.sh "${tag}" 16 --no-trace
    status=$?
    unset MLX_E120_QMV_PIPELINE_LOG
    clear_arm
    echo "e135_arm=${arm}" >> "${out}/meta.txt"
    if ((status != 0)); then
      echo "e135_fill_cost_abba: witness ${tag} exited ${status}" >&2
      exit 5
    fi
  fi
  echo "--- Rule 110 witness for ${arm} ---"
  python3 research/e135_fill_cost.py check "${out}/pipelines.json" \
    --want "${arm}" | tee "${out}/fill-arm-check.txt"
  if [[ "${PIPESTATUS[0]}" != "0" ]]; then
    echo "e135_fill_cost_abba: the ${arm} leg did not run the ${arm} arm" >&2
    exit 3
  fi
  anti="$(anti_arm "${arm}")"
  echo "--- Rule 101 control: the same check must FAIL against ${anti} ---"
  python3 research/e135_fill_cost.py check "${out}/pipelines.json" \
    --want "${anti}" | tee "${out}/fill-arm-control.txt"
  if [[ "${PIPESTATUS[0]}" == "0" ]]; then
    echo "e135_fill_cost_abba: the ${arm} leg also passed the ${anti} check" >&2
    exit 4
  fi
  echo "control ok: the witness fails when it should"
done

# PHASE 2: the counterbalanced gated session.
failures=0
position=0
for arm in repl fill fill repl; do
  position=$((position + 1))
  tag="e135${label}k1p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} gated ==="
  apply_arm "${arm}"
  research/e79_trace_leg.sh "${tag}" "${tokens}" --no-trace --cool-gate
  status=$?
  clear_arm
  out="research/out/${tag}"
  {
    echo "e135_arm=${arm}"
    echo "e135_position=${position}"
    echo "e135_session_commit=${session_commit}"
    echo "e135_session_worker_sha256=${session_worker}"
  } >> "${out}/meta.txt"
  if ((status != 0)); then
    echo "e135_fill_cost_abba: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
  fi
done

echo "e135_fill_cost_abba: ${failures} failed legs"
python3 research/e135_fill_cost.py report --label "${label}"
exit $((failures > 0))
