#!/usr/bin/env bash
# E89 rung 0b: run identical drafting legs with the host-state probe on.
#
#   usage: research/e89_probe_session.sh PREFIX TOKENS LEGSPEC ...
#
# LEGSPEC is `name=qos[/parts]`. `qos` is the value of MLX_E89_FORCE_QOS for
# that leg, `none` to leave the thread policy exactly as the worker inherits
# it, or `off` to also disable the probe itself (MLX_E89_PROBE=0). An `off`
# leg is the instrument-cost control: same worker binary, no per-round probe
# work. `parts` is a `+`-separated subset of the probe components
# `marks probe rusage thread mem`, which becomes MLX_E89_PARTS; omit it for
# every component. The pseudo-part `synchead` is not a probe component: it
# passes --sync-head to the leg, which drains the head chain every round. The
# `name` becomes the arm label in the tag; a repeated name gets the next
# repeat index.
#
# Rung 0b had no arm to balance, because it measured the natural rate at which
# a leg lands in the slow host state. Rung 0c does have one: forced
# userInteractive QoS is a candidate fix, so its legs and their controls are
# ABBA-counterbalanced within the session and every other setting stays at the
# shipped configuration, which is the default ladder, production overlap and
# the declared head.
#
# The legs are UNGATED (program.md permits an ungated, counterbalanced local
# arm). e79_trace_leg.sh records entry and exit GPU temperature per leg and
# keeps cool_gate_passed_real_gate=false and gate_qualified_for_timing=false.
# Nothing here is a score.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e89_probe_session.sh PREFIX TOKENS LEGSPEC ...}"
tokens="${2:?usage: e89_probe_session.sh PREFIX TOKENS LEGSPEC ...}"
shift 2
(($#)) || { echo "e89_probe_session.sh: no legs given" >&2; exit 2; }

declare -a order=("$@")
declare -a tags=()
for i in "${!order[@]}"; do
  spec="${order[$i]}"
  name="${spec%%=*}"
  [[ "${name}" != "${spec}" ]] || {
    echo "e89_probe_session.sh: leg '${spec}' is not name=qos" >&2
    exit 2
  }
  n=1
  for ((j = 0; j < i; j++)); do
    [[ "${order[$j]%%=*}" == "${name}" ]] && n=$((n + 1))
  done
  tags+=("${prefix}-${name}-${n}")
done

echo "plan: ${#order[@]} legs, ${tokens} decode tokens, probe on"
for i in "${!order[@]}"; do
  echo "  pos=${i} tag=${tags[$i]} force_qos=${order[$i]#*=}"
done

export MLXFAST_QWEN_MTP_HEAD_DIR="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"

for i in "${!order[@]}"; do
  value="${order[$i]#*=}"
  qos="${value%%/*}"
  parts=""
  [[ "${value}" == */* ]] && parts="${value#*/}"
  sync=0
  if [[ "+${parts}+" == *"+synchead+"* || "${parts}" == "synchead" ]]; then
    sync=1
    parts="${parts//synchead/}"
    parts="${parts//++/+}"
    parts="${parts#+}"
    parts="${parts%+}"
  fi
  probe=1
  unset MLX_E89_FORCE_QOS MLX_E89_PARTS
  case "${qos}" in
    off) probe=0 ;;
    none) ;;
    *) export MLX_E89_FORCE_QOS="${qos}" ;;
  esac
  [[ -n "${parts}" ]] && export MLX_E89_PARTS="${parts//+/,}"
  export MLX_E89_PROBE="${probe}"
  if ((sync)); then
    research/e79_trace_leg.sh "${tags[$i]}" "${tokens}" --sync-head
  else
    research/e79_trace_leg.sh "${tags[$i]}" "${tokens}"
  fi
  status=$?
  {
    echo "e89_force_qos=${qos}"
    echo "e89_parts=${parts:-all}"
    echo "e89_position=${i}"
    echo "e89_probe=${probe}"
    echo "e89_sync_head=${sync}"
  } >> "research/out/${tags[$i]}/meta.txt"
  bytes=$(wc -c < "research/out/${tags[$i]}/trace.txt" | tr -d ' ')
  echo "leg ${tags[$i]} pos=${i} qos=${qos} exit=${status} trace_bytes=${bytes}"
  # A --golden leg runs under a seatbelt that denies file writes, so an empty
  # dump can accompany a leg that reports success. Fail loudly instead.
  ((bytes > 1000)) || { echo "e89: empty trace dump" >&2; exit 1; }
  ((status == 0)) || exit "${status}"
done

echo "session ${prefix}: ${#order[@]} legs complete"
