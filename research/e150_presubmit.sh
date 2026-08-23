#!/usr/bin/env bash
# E150 R4 -- the pre-submit chain for the linearised depth schedule.
#
#   usage: research/e150_presubmit.sh
#
# WHAT THIS CHAIN IS. An exactness, scope and gross-regression gate for the
# `linearised` schedule arm. It is NOT a timing session. Every decode leg keeps
# the per-round phase trace, which writes twice inside every round, so each leg
# records `timing_valid=false`. CAMPAIGN RULE 79 forbids a local timing leg
# from publishing a depth-price or schedule-policy contrast, so the chain does
# not try to produce one. The offline replay owns the price; this chain owns
# correctness.
#
# ORDER. `--local-submit` takes the real 40C cool gate, so it runs before the
# sixteen ungated decode legs heat the die rather than after them. The full
# `swift test` runs last because it needs no GPU.
#
# WHY THE CONTROL ARM RUNS BEFORE THE CANDIDATE. The 513-row reference goldens
# were generated on an earlier campaign commit. If the serial path had moved
# under them, both arms would fail exactness for a reason that has nothing to
# do with this experiment. `shipped` is the arm that is already promoted, so it
# reads the goldens first.
#
# WHY `benchfixture` IS FIRST IN THE PROMPT LIST. It is the one local prompt
# whose 513-row golden contains an EOS token (id 248044, emitted index 300).
# A 512-token leg on it therefore generates 211 tokens AFTER EOS and is the
# only local evidence available for fixed-window post-EOS continuation.
#
# EXIT STATUS. The static gates, the exactness legs and the collector are hard
# steps and set the exit status. The full `swift test` is a SOFT step: the
# suite already fails on this base with a documented issue set that this
# experiment does not touch, so it is recorded and reported but does not decide
# the chain. Its log is the receipt for that accounting.
#
# HARNESS DEFECT 28: only `MLX_`, `DARKBLOOM_`, `METAL_`, `MTL_`, `DYLD_` and
# `LC_` prefixed names survive `sanitizedRuntimeWorkerEnvironment`, which is
# why the arm selector is spelled `MLX_E150_SCHEDULE_ARM`.
#
# HARNESS DEFECT 36: `./benchmark-qwen-mtp.sh` never rebuilds the runtime
# worker, so the worker is rebuilt and asserted before anything reads it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONTRACT_SHA="770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
PROMPTS=(benchfixture beagle_a essays_montaigne medicine_hist
         plutarch_lives republic_jowett botany_andrews travel_eothen)
out_dir="research/out/e150-presubmit"
mkdir -p "${out_dir}"

hard_rc=0
declare -a report=()

record() {
  local name="$1" status="$2" kind="$3"
  report+=("${kind} ${name}=${status}")
  echo "EXIT ${name}=${status} (${kind})"
  [[ "${kind}" == "hard" && "${status}" -ne 0 ]] && hard_rc="${status}"
  return 0
}

step() {
  local kind="$1" name="$2"; shift 2
  echo
  echo "################ ${name} ################"
  "$@" 2>&1 | tee "${out_dir}/${name}.log"
  record "${name}" "${PIPESTATUS[0]}" "${kind}"
}

echo "e150_presubmit: commit $(git rev-parse HEAD)"
echo "e150_presubmit: worktree dirty paths $(git status --porcelain | wc -l | tr -d ' ')"

step hard worker-assert senpai/rebuild-and-assert-worker.sh \
  --require 'MLX_E150_SCHEDULE_ARM' \
  --require-symbol linearisedDepth \
  --forbid 'MLXFAST_E150_SCHEDULE_ARM'
step hard twin-audit python3 research/twin_audit.py
step hard scope senpai/validate-assignment-scope.sh "${CONTRACT_SHA}" \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift
step hard budget senpai/check-editable-budget.sh "${CONTRACT_SHA}"
step hard boundary senpai/verify-ranked-score-boundary.sh
step hard swift-test-e150 swift test --force-resolved-versions --filter E150

echo
echo "################ local-submit-512 ################"
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
MLXFAST_SCORE_PATH="${PWD}/${out_dir}/local-submit-512.json" \
  ./benchmark-qwen-mtp.sh --local-submit 2>&1 \
  | tee "${out_dir}/local-submit-512.log"
record local-submit-512 "${PIPESTATUS[0]}" hard

for arm in shipped linearised; do
  echo
  echo "################ exactness-512-${arm} ################"
  MLX_E150_SCHEDULE_ARM="${arm}" \
  E128_FORCE=1 E128_TOKENS=512 E128_DEPTH=8 \
  E128_RUNS_DIR="runs-e150-${arm}" \
    research/e128_session.sh "${PROMPTS[@]}" 2>&1 \
    | tee "${out_dir}/exactness-512-${arm}.log"
  record "exactness-512-${arm}" "${PIPESTATUS[0]}" hard
done

step soft swift-test swift test --force-resolved-versions

step hard collect python3 research/e150_presubmit_collect.py \
  --shipped .mlxfast-private/e128/runs-e150-shipped \
  --linearised .mlxfast-private/e128/runs-e150-linearised \
  --local-submit "${out_dir}/local-submit-512.json" \
  --chain-rc "${hard_rc}"

echo
echo "################ SUMMARY ################"
printf '%s\n' "${report[@]}"
echo "hard rc=${hard_rc}"
exit "${hard_rc}"
