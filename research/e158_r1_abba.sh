#!/usr/bin/env bash
# E158 R1.C -- price the precision islands on the candidate leg.
#
#   usage: research/e158_r1_abba.sh PHASE [PHASE ...]
#          PHASE is one of: default-witness, perprompt, canary, gated
#
# `perprompt` runs an ABBA palindrome (all, none, none, all) of TIMED
# `mtp-timed` legs for each prompt, so the candidate-leg seconds per token and
# the effective mean draft length are reported PER PROMPT and never pooled.
# `mtp-timed` runs the candidate MTP leg only, which is the leg RULE 177 prices.
# The cool gate is not taken; the palindrome is the counterbalance and every leg
# records its entry and exit GPU temperature, `cool_gate_passed_real_gate=false`
# and `gate_qualified_for_timing=false`.
#
# `gated` runs the same palindrome through `benchmark-qwen-mtp.sh
# --local-iterate` at 512 tokens with the real 40 C gate, on the one public
# fixture that harness can run. Those legs are gate-qualified and also carry the
# serial control leg, so they give the local serial-to-MTP ratio as well as the
# absolute candidate time. The ratio is admissible here because the causal path
# is confined to the candidate MTP leg: the serial control leg never drafts and
# therefore never reads a precision-island tensor.
#
# `canary` runs one leg per arm on `plutarch_lives`. Advisor F5 measured that a
# head change moves plutarch `edl` about 128x more than it moves beagle `edl`,
# so plutarch is the highest-gain detector of a head-quality change on the
# board. It carries median weight 0.0033 and needs +181 % before the published
# median moves, so it is a canary and never a target. Acceptance is
# deterministic under greedy decoding, so one leg per arm is enough and no
# counterbalancing is required.
#
# `default-witness` is the Rule 101 control. It runs one short leg with NO
# selector in the environment and asserts that the shipped default is now
# `none`, and one short leg with the selector set to `all` and asserts the
# witness changes. A comparison that cannot fail is not a comparison.
#
# Every leg drafts with the head `mtp-head.manifest.json` DECLARES, because the
# pinned head carries no `precision_islands.*` tensors at all and would make
# every arm a silent null.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

(($#)) || {
  echo "usage: research/e158_r1_abba.sh PHASE [PHASE ...]" >&2
  echo "  PHASE: default-witness | perprompt | canary | gated" >&2
  exit 2
}

tokens="${E158_TOKENS:-512}"
prompts=(beagle_a essays_montaigne benchfixture)
canary_prompt="${E158_CANARY_PROMPT:-plutarch_lives}"
arms=(all none none all)
head_dir="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run"
witness_root=".mlxfast-private/e158r1"
mkdir -p "${witness_root}"
status=0

assert_witness() {
  local path="$1" want="$2" label="$3" line
  line="$(grep -h '^qwen-mtp-island-arm: ' "${path}" 2>/dev/null | sort -u)"
  if [[ -z "${line}" ]]; then
    echo "e158_r1_abba: ${label}: NO ISLAND WITNESS in ${path}" >&2
    return 1
  fi
  if [[ "$(printf '%s\n' "${line}" | wc -l | tr -d ' ')" != "1" ]]; then
    echo "e158_r1_abba: ${label}: conflicting witnesses: ${line//$'\n'/ | }" >&2
    return 1
  fi
  if [[ "${line}" != "qwen-mtp-island-arm: ${want} "* ]]; then
    echo "e158_r1_abba: ${label}: wanted arm ${want}, leg ran: ${line}" >&2
    return 1
  fi
  echo "e158_r1_abba: ${label}: ${line}"
  return 0
}

for phase in "$@"; do
case "${phase}" in
  default-witness)
    for spec in "unset:none" "all:all"; do
      requested="${spec%%:*}"
      want="${spec##*:}"
      out="${witness_root}/default-${requested}"
      rm -rf "${out}"; mkdir -p "${out}"
      env_args=(E128_HEAD_DIR="${head_dir}"
                E128_RUNS_DIR="runs-e158r1-witness-${requested}"
                E128_TOKENS="${E158_WITNESS_TOKENS:-24}"
                E128_DEPTH=8
                E128_FORCE=1
                E128_NO_TRACE=1
                MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/witness.txt")
      [[ "${requested}" == "unset" ]] \
        || env_args+=("DARKBLOOM_QWEN_MTP_ISLAND_ARM=${requested}")
      : > "${out}/witness.txt"
      echo "=== e158r1 witness leg: selector=${requested} expect=${want} ==="
      env -u DARKBLOOM_QWEN_MTP_ISLAND_ARM "${env_args[@]}" \
        research/e128_session.sh beagle_a || status=1
      assert_witness "${out}/witness.txt" "${want}" \
        "selector=${requested}" || status=1
    done
    ;;

  perprompt)
    for id in "${prompts[@]}"; do
      for i in "${!arms[@]}"; do
        arm="${arms[$i]}"
        slot="$((i + 1))"
        out="${witness_root}/perprompt-${id}-${slot}-${arm}"
        rm -rf "${out}"; mkdir -p "${out}"
        : > "${out}/witness.txt"
        echo "=== e158r1 perprompt ${id} slot ${slot} arm ${arm} ==="
        env DARKBLOOM_QWEN_MTP_ISLAND_ARM="${arm}" \
            E128_HEAD_DIR="${head_dir}" \
            E128_RUNS_DIR="runs-e158r1-${id}-${slot}-${arm}" \
            E128_TOKENS="${tokens}" \
            E128_DEPTH=8 \
            E128_FORCE=1 \
            E128_NO_TRACE=1 \
            MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/witness.txt" \
          research/e128_session.sh "${id}" || status=1
        assert_witness "${out}/witness.txt" "${arm}" \
          "${id} slot ${slot}" || status=1
      done
    done
    ;;

  canary)
    for arm in all none; do
      out="${witness_root}/canary-${canary_prompt}-${arm}"
      rm -rf "${out}"; mkdir -p "${out}"
      : > "${out}/witness.txt"
      echo "=== e158r1 canary ${canary_prompt} arm ${arm} ==="
      env DARKBLOOM_QWEN_MTP_ISLAND_ARM="${arm}" \
          E128_HEAD_DIR="${head_dir}" \
          E128_RUNS_DIR="runs-e158r1-canary-${arm}" \
          E128_TOKENS="${tokens}" \
          E128_DEPTH=8 \
          E128_FORCE=1 \
          E128_NO_TRACE=1 \
          MLX_QWEN_MTP_TRACE_PATH="${PWD}/${out}/witness.txt" \
        research/e128_session.sh "${canary_prompt}" || status=1
      assert_witness "${out}/witness.txt" "${arm}" \
        "canary ${canary_prompt} ${arm}" || status=1
    done
    ;;

  gated)
    for i in "${!arms[@]}"; do
      arm="${arms[$i]}"
      slot="$((i + 1))"
      tag="e158r1g-${slot}-${arm}"
      out="${witness_root}/gated-${slot}-${arm}"
      rm -rf "${out}"; mkdir -p "${out}"
      echo "=== e158r1 gated slot ${slot} arm ${arm} (${tokens} tokens) ==="
      env DARKBLOOM_QWEN_MTP_ISLAND_ARM="${arm}" \
          E79_WITNESS_PATH="${PWD}/${out}/witness.txt" \
        research/e79_trace_leg.sh "${tag}" "${tokens}" \
          --cool-gate --no-trace || status=1
      assert_witness "${out}/witness.txt" "${arm}" \
        "gated slot ${slot}" || status=1
    done
    ;;

  *)
    echo "e158_r1_abba: unknown phase '${phase}'" >&2
    exit 2
    ;;
esac
done

exit "${status}"
