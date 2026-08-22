#!/bin/bash
# E135 composition gate chain (advisor F13): rebuild, assert the arm witnesses,
# run the Swift suite, audit the generated Metal twins, run every campaign gate,
# then take ONE bare 512-token exactness leg on the composed commit.
#
# The leg is the pb6 tripwire. The composed tree must show edward's pb6 round
# schedule (about 82 rounds, mean draft about 5.854). If it shows the pre-pb6
# schedule (78 rounds, 6.359) the rebase dropped pb6 and the chain fails.
#
# The leg exports no MLX_E120_QMV_GRID and no MLX_E120_QMV_TABLE, so it takes
# the route the ranked runner takes, and every arm witness is read back off the
# run's own trace (CAMPAIGN RULE 114), never off an environment variable.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

tag="e135c512"
out="research/out/${tag}"
log_dir="research/out/e135compose"
mkdir -p "${log_dir}"
rc=0

digest() {
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker 2>/dev/null | awk '{print $1}'
}

step() {
  local name="$1"; shift
  echo
  echo "################ ${name} ################"
  "$@" 2>&1 | tee "${log_dir}/${name}.log"
  local s="${PIPESTATUS[0]}"
  echo "EXIT ${name}=${s}"
  [[ "${s}" -eq 0 ]] || rc=1
  return "${s}"
}

echo "################ identity ################"
echo "head_commit $(git rev-parse HEAD)"
echo "host        $(hostname)"
git status --porcelain -- Sources Vendor Package.swift mtp-head.manifest.json \
  | sed 's/^/dirty: /'

# A bare invocation exits 2 and asserts nothing, so every arm witness is named
# here. `noteLaunch` is a Swift identifier and lives in the symbol table, not
# the string table. Every string needle is at least 16 bytes (HARNESS DEFECT
# 38): `strings` splits shorter runs and a short needle can match by accident.
#
# The width-plan literal is NOT a discriminator, whatever the source comment
# claims. All four `Table.witness` literals survive into the built worker at
# exactly one copy each, so requiring one of them passes for a worker built
# from any table. It is asserted only to prove the witness table is intact.
# The route witness IS a discriminator: it is a single literal that names both
# compiled defaults, `defaultRouteWitnessNamesTheCompiledDefaults` fails if it
# drifts from them, and no other literal contains it. The table a leg actually
# ran is read off the run's own `plan` trace field (CAMPAIGN RULE 114).
step rebuild senpai/rebuild-and-assert-worker.sh \
  --require 'e135_default_grid/tight' \
  --require 'e135_default_probe/p10' \
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4' \
  --require 'e120_default_route/tiered_switch/shipped' \
  --require 'columns_by_width' \
  --forbid 'e135_default_grid/wide' \
  --forbid 'e135_default_probe/p15' \
  --forbid 'e135_default_probe/p25' \
  --forbid 'e120_default_route/tiered_switch/onepass67' \
  --require-symbol 'noteLaunch'
echo "worker_sha256 $(digest)"

# The suite never exits 0 on this host: ten tests fail on the unchanged base.
# Nine are the long-documented organizer contract and documentation tests and
# the tenth, `theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax`, is
# inherited from the composed base, whose `Qwen36MTPBlockSession.swift` and
# `E130WiredResidencySlackTests.swift` are byte-identical here. The verdict is
# therefore the failing SET, not the exit code: any name outside the inherited
# list is a campaign-added failure and stops the chain.
echo
echo "################ swift-test ################"
swift test --force-resolved-versions 2>&1 | tee "${log_dir}/swift-test.log"
echo "EXIT swift-test=${PIPESTATUS[0]} (exit code is not the verdict)"
grep -oE '✘ Test [A-Za-z0-9_]+\(\)' "${log_dir}/swift-test.log" \
  | sed 's/✘ Test //' | sort -u > "${log_dir}/failing-tests.txt"
cat > "${log_dir}/inherited-tests.txt" <<'INHERITED'
contestantDocsCommandBlocksKeepTheDependencyGraphFrozen()
participantDocsExposeDefaultCLIInstallDirectory()
qwen36ConfigContractDigestMatchesTheReferenceManifest()
startupMemoryPolicyKeepsRanked128GiBProfile()
submissionStaticReviewPromptCoversMeasurementStructureExploitation()
theCheckedInDeclarationSelectsThePinnedHead()
theEvenMedianRuleIsTheMeanOfTheTwoCentralValues()
theQwenMTPTrackIsArmedOnQwen38()
theSeededCalibrationExpectationMatchesItsRecordedProvenance()
theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax()
INHERITED
added="$(comm -23 "${log_dir}/failing-tests.txt" "${log_dir}/inherited-tests.txt")"
echo "failing tests: $(wc -l < "${log_dir}/failing-tests.txt" | tr -d ' ')"
if [[ -n "${added}" ]]; then
  echo "FAIL campaign-added test failures:"
  echo "${added}"
  rc=1
else
  echo "ok   every failing test is on the inherited list"
fi

step twin-audit python3 research/twin_audit.py
# Both gates exit 2 when invoked bare, and a gate that did not run is not
# evidence. `base` is the composition base; the only submitted path this
# experiment changes is the Qwen runtime file.
base="c6e32050f5a6b715df63aec7790a851ce1d9163c"
step scope senpai/validate-assignment-scope.sh "${base}" \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
step budget senpai/check-editable-budget.sh \
  770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf
step score-boundary senpai/verify-ranked-score-boundary.sh
step cliff-census senpai/entry-point-cliff-census.sh --base "${base}" \
  --json "${log_dir}/cliff-census.json"

if [[ "${rc}" -ne 0 ]]; then
  echo
  echo "GATES FAILED (rc=${rc}); the 512-token leg is NOT run."
  exit "${rc}"
fi

echo
echo "################ bare 512-token exactness leg ################"
unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE
mkdir -p "${out}"
export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${out}/pipelines.json"
research/e79_trace_leg.sh "${tag}" 512 2>&1 | tee "${log_dir}/bare-512.log"
b="${PIPESTATUS[0]}"
unset MLX_E120_QMV_PIPELINE_LOG
echo "EXIT bare-512=${b}"
echo "worker_after_leg $(digest)"
[[ "${b}" -eq 0 ]] || rc=3

echo
echo "--- witness: the leg's own plan and route trace ---"
# `e135_columns_check.py` derives its expectation FROM the recorded plan, so it
# is self-consistent and cannot notice the wrong table. This names the table.
python3 - "${out}/pipelines.json" <<'PY'
import json
import sys

WANT_PLAN = "e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4"
WANT_ROUTE = "e120_default_route/tiered_switch/shipped"
WANT_GRID = "e135_default_grid/tight"
WANT_PROBE = "e135_default_probe/p10"

trace = json.load(open(sys.argv[1]))
ok = True
for key, want in (("plan", WANT_PLAN), ("default_route", WANT_ROUTE),
                  ("default_grid", WANT_GRID), ("default_probe", WANT_PROBE)):
    got = trace.get(key)
    print("%-14s %s" % (key, got))
    if got != want:
        print("FAIL %s is %r, expected %r" % (key, got, want))
        ok = False
print("PASS table and grid witness" if ok else "FAIL table and grid witness")
sys.exit(0 if ok else 1)
PY
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=8

echo
echo "--- witness: columns_by_width must be tight ---"
python3 research/e135_columns_check.py "${out}/pipelines.json" --want tight
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=4
echo "--- Rule 101 control: the same check must FAIL against wide ---"
python3 research/e135_columns_check.py "${out}/pipelines.json" --want wide
[[ "${PIPESTATUS[0]}" -ne 0 ]] || rc=5

echo
echo "--- witness: probe fraction and probe count off this leg's own trace ---"
python3 research/e135_probe_check.py "${out}/pipelines.json"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=6

echo
echo "--- witness: pb6 round schedule ---"
python3 research/e135_pb6_check.py "${out}/score.json"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=7

echo
echo "################ SUMMARY ################"
jq -c '{passed, m:(.metrics|{decode_tokens, all_tokens_matched,
    residual_divergence_count, public_drift_tripwire_passed,
    mtp_seconds_per_token, serial_seconds_per_token, mtp_decode_speedup,
    effective_mean_draft_len, accepted_draft_rate})}' "${out}/score.json" \
  2>/dev/null || cat "${out}/score.json"
echo "chain rc=${rc}"
exit "${rc}"
