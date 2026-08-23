#!/bin/bash
# E135 composition gate chain (advisor F13): rebuild, assert the arm witnesses,
# run the Swift suite, audit the generated Metal twins, run every campaign gate,
# then take ONE bare 512-token exactness leg on the composed commit.
#
# The leg is the depth-arm tripwire. Advisor F23 reverts the shipped arm from
# pb6 back to ship, so the composed tree must show the ship round schedule
# (78 rounds, mean draft 6.359). If it shows the pb6 schedule (82 rounds,
# 5.854) the revert did not reach the binary and the chain fails.
#
# The leg exports no MLX_E120_QMV_GRID, no MLX_E120_QMV_TABLE and no
# MLX_E135_PROBE_ARM, so it takes the route the ranked runner takes, and every
# arm witness is read back off the run's own trace (CAMPAIGN RULE 114), never
# off an environment variable. Advisor F20 orders probe rung p15, whose derived
# integer 1844 the leg must report, and advisor F18 orders width 2 into the
# routed set, which the plan witness and the `columns_by_width` census carry.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

tag="e135ship"
out="research/out/${tag}"
log_dir="research/out/e135shipgate"
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
#
# The width-2 route IS witnessed here, and unusually the plan literal does
# discriminate it. Every `Table.witness` string carries the `e120_width_plan/`
# prefix, so the pre-F18 literal is not a substring of the F18 one: forbidding
# `e120_width_plan/3:3:4,...` fails on any worker built before width 2 entered
# the four tables, and passes only after all four moved.
step rebuild senpai/rebuild-and-assert-worker.sh \
  --require 'e135_default_grid/tight' \
  --require 'e135_default_probe/p15' \
  --require 'e120_width_plan/2:2:4,3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4' \
  --require 'e120_default_route/tiered_switch/onepass67' \
  --require 'columns_by_width' \
  --forbid 'e135_default_grid/wide' \
  --forbid 'e135_default_probe/p10' \
  --forbid 'e135_default_probe/p25' \
  --forbid 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:3:4,7:4:4,8:4:4,9:3:4' \
  --forbid 'e120_default_route/tiered_switch/shipped' \
  --require-symbol 'noteLaunch'
echo "worker_sha256 $(digest)"

# The suite never exits 0 on this host, so the verdict is the failing SET, not
# the exit code. Any name outside the two known lists below stops the chain.
#
# HARNESS DEFECT 39. Until this edit the extractor was
# `grep -oE '✘ Test [A-Za-z0-9_]+\(\)'`, which reads only tests that carry no
# `@Test("display name")`. swift-testing prints the quoted display name instead
# of the function name for every test that has one, so the whole display-named
# population was invisible to the gate. The p10 chain reported ten failures and
# passed while the log actually held twelve. Both hidden failures are recorded
# below. The extractor now takes both forms.
echo
echo "################ swift-test ################"
swift test --force-resolved-versions 2>&1 | tee "${log_dir}/swift-test.log"
echo "EXIT swift-test=${PIPESTATUS[0]} (exit code is not the verdict)"
grep -oE '✘ Test ([A-Za-z0-9_]+\(\)|"[^"]*") failed' "${log_dir}/swift-test.log" \
  | sed -E 's/^✘ Test //; s/ failed$//' | sort -u > "${log_dir}/failing-tests.txt"

# Failures present on the unchanged base. Most are the long-documented
# organizer contract and documentation tests.
# `theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax` is inherited from
# the composed base, whose `Qwen36MTPBlockSession.swift` and
# `E130WiredResidencySlackTests.swift` are byte-identical here.
# `theSubmissionTemplateNamesTheLocalSubmitCommand` is advisor F19's tenth name,
# reported pre-existing on `BASE_SHA`; it is listed so a base that carries it
# does not stop the chain, and the printed failing set below states whether
# this tree actually reaches it.
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
theSubmissionTemplateNamesTheLocalSubmitCommand()
theWiredSlackCoversTheMeasuredGrowthAndItsPageRoundingTax()
INHERITED

# This branch caused no test failure of its own. The compiled route is back on
# `.onePass67`, so `E134PassBoundaryPriceTests` and the E120 route witness both
# describe the compiled tree again and any name outside the inherited list is a
# real regression.
sort -u "${log_dir}/inherited-tests.txt" > "${log_dir}/known-tests.txt"
added="$(comm -23 "${log_dir}/failing-tests.txt" "${log_dir}/known-tests.txt")"
echo "failing tests: $(wc -l < "${log_dir}/failing-tests.txt" | tr -d ' ')"
cat "${log_dir}/failing-tests.txt" | sed 's/^/  fail: /'
if [[ -n "${added}" ]]; then
  echo "FAIL campaign-added test failures:"
  echo "${added}"
  rc=1
else
  echo "ok   every failing test is inherited or a recorded route consequence"
fi

# The plain suite above skips every `MLXFAST_RUN_MLX_RUNTIME_TESTS` gate, so it
# never touches the GPU. Advisor F18 requires bit exactness at the new width
# against the library kernel the route displaces, with a control that can fail.
# `E135Width2RouteTests` compares m = 2 with `quantizedMM` on six scored shapes,
# and `E120CustomQMVProbeTests` now sweeps `Qwen35CustomQMV.widths`, so its
# x_hit, meta_hit and table_hit controls cover width 2 as well.
echo
echo "################ runtime-exactness ################"
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
  --filter 'E135Width2RouteTests' --filter 'E120CustomQMVProbeTests' \
  2>&1 | tee "${log_dir}/runtime-exactness.log"
r="${PIPESTATUS[0]}"
echo "EXIT runtime-exactness=${r}"
[[ "${r}" -eq 0 ]] || rc=1
grep -cE '✔ Test .* passed' "${log_dir}/runtime-exactness.log" \
  | sed 's/^/runtime tests passed: /'

step twin-audit python3 research/twin_audit.py
# Both gates exit 2 when invoked bare, and a gate that did not run is not
# evidence. `base` is the composition base; the only submitted path this
# experiment changes is the Qwen runtime file.
# The live advisor head. It is NOT merged into this branch: askeladd's tip and
# this branch both add a file at `research/e136_wandb_log.py`, and F19 states
# his tip changes zero candidate bytes. `git diff bdba19f6 5800f88e` over
# `Sources`, `Vendor`, `Package.swift`, `Package.resolved`,
# `mtp-head.manifest.json` and `mtp-head/` is empty, so the scope check gives
# the same answer against either commit and is run against the newer one.
base="5800f88e1666f1a01895ef03ab5dfacfbbb729e8"
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
unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E135_PROBE_ARM
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

WANT_PLAN = "e120_width_plan/2:2:4,3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:4:4,9:3:4"
WANT_ROUTE = "e120_default_route/tiered_switch/onepass67"
WANT_GRID = "e135_default_grid/tight"
WANT_PROBE = "e135_default_probe/p15"

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

# Advisor F19 item 5. `e135_columns_check.py` derives its expectation from the
# recorded plan, so it cannot notice a plan that is itself wrong. This states
# the map the leg must show under `.onePass67` + tight + width 2 as a literal,
# computed as `ceil(m / ipg)` over the onePass67 plan. Widths 6 and 7 fall from
# 2 columns to 1, which is the mechanism T29-A rung 1 priced.
echo "--- witness: the launched column map, stated as a literal ---"
python3 - "${out}/pipelines.json" <<'PY'
import json
import sys

WANT = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 3}
trace = json.load(open(sys.argv[1]))
seen = {int(k): int(v) for k, v in trace.get("columns_by_width", {}).items()}
print("columns_by_width %s" % dict(sorted(seen.items())))
print("expected         %s" % dict(sorted(WANT.items())))
missing = sorted(set(WANT) - set(seen))
wrong = sorted(m for m in seen if WANT.get(m) != seen[m])
if missing:
    print("FAIL the leg never launched widths %s" % missing)
if wrong:
    print("FAIL wrong column count at widths %s" % wrong)
print("PASS launched column map" if not missing and not wrong else "FAIL")
sys.exit(0 if not missing and not wrong else 1)
PY
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=9
echo "--- Rule 101 control: the same check must FAIL against wide ---"
python3 research/e135_columns_check.py "${out}/pipelines.json" --want wide
[[ "${PIPESTATUS[0]}" -ne 0 ]] || rc=5

echo
echo "--- witness: probe fraction and probe count off this leg's own trace ---"
python3 research/e135_probe_check.py "${out}/pipelines.json"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=6

echo
echo "--- witness: pb6 round schedule ---"
python3 research/e135_arm_check.py "${out}/score.json" --want pb6
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=7
echo "--- Rule 101 control: the same check must FAIL against ship ---"
python3 research/e135_arm_check.py "${out}/score.json" --want ship
[[ "${PIPESTATUS[0]}" -ne 0 ]] || rc=10

echo
echo "################ SUMMARY ################"
jq -c '{passed, m:(.metrics|{decode_tokens, all_tokens_matched,
    residual_divergence_count, public_drift_tripwire_passed,
    mtp_seconds_per_token, serial_seconds_per_token, mtp_decode_speedup,
    effective_mean_draft_len, accepted_draft_rate})}' "${out}/score.json" \
  2>/dev/null || cat "${out}/score.json"
echo "chain rc=${rc}"
exit "${rc}"
