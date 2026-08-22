#!/usr/bin/env bash
# E135 rung 4 -- the pre-submit chain for the tight QMV launch grid.
#
#   usage: research/e135_presubmit.sh
#
# The candidate is one word: `Grid.compiledDefault` moves from `.wide` to
# `.tight`, so the routed QMV dispatch launches `ceil(m / ipg)` threadgroup
# columns instead of `m`. The columns it stops launching are exactly the ones
# that return at `qwen_e120_qmv_m:1548` before any load of `w`, `scales`,
# `biases`, `x` or `xsums` and before any store to `y`, so the arm is
# bit-identical by construction and the whole question is a timing question.
#
# Steps run in cost order: static checks, then the Swift suite, then two GPU
# legs. The GPU cools during the suite, which is why the legs run last.
#
# THE LOAD-BEARING WITNESS IS STEP 8. Every other check would still pass if the
# default had not moved. The ranked runner exports nothing, so the only proof
# that it takes the tight grid is a leg that exports nothing either and still
# records the tight launched-column map. Rule 101: the same leg is checked a
# second time against the wide map and MUST fail.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_SHA="ea546d1f8ab0d1de2169e5fbf19bdef2aa79003d"
CONTRACT_SHA="770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
out_dir="research/out/e135-presubmit"
mkdir -p "${out_dir}"

rc=0
step() {
  local name="$1"; shift
  echo
  echo "################ ${name} ################"
  "$@" 2>&1 | tee "${out_dir}/${name}.log"
  local s="${PIPESTATUS[0]}"
  echo "EXIT ${name}=${s}"
  [[ "${s}" -eq 0 ]] || rc="${s}"
  return 0
}

# `--require` and `--forbid` read the string table, which is the right witness
# for `defaultGridWitness`: it is one plain literal that exists only for the
# case actually compiled, so the `--forbid` line can fail. `noteLaunch` is a
# Swift symbol and reaches only the symbol table.
step worker-assert senpai/rebuild-and-assert-worker.sh \
  --require 'e135_default_grid/tight' \
  --forbid  'e135_default_grid/wide' \
  --require 'e120_width_plan/3:3:4,4:4:4,5:5:4,6:6:4,7:7:4,8:4:4,9:3:4' \
  --require 'e120_default_route/tiered_switch/onepass67' \
  --require 'columns_by_width' \
  --require-symbol noteLaunch

step twin-audit python3 research/twin_audit.py
step scope senpai/validate-assignment-scope.sh "${BASE_SHA}" \
  Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
step budget senpai/check-editable-budget.sh "${CONTRACT_SHA}"
step boundary senpai/verify-ranked-score-boundary.sh
step cliff-census senpai/entry-point-cliff-census.sh --base "${BASE_SHA}" \
  --json "${out_dir}/cliff-census.json"
step swift-test swift test --force-resolved-versions

# STEP 8. The bare-leg default-grid witness.
#
# No `MLX_E120_QMV_GRID` and no `MLX_E120_QMV_TABLE` are exported, so this leg
# takes exactly the route the ranked runner takes. `columns_by_width` is read
# back off the argument handed to Metal, so it witnesses the compiled fallback
# and not merely a parsed enum.
echo
echo "################ bare-leg-grid-witness ################"
witness_dir="research/out/e135-bare-default"
mkdir -p "${witness_dir}"
unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE
export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${witness_dir}/pipelines.json"
research/e79_trace_leg.sh e135-bare-default 16 --no-trace \
  2>&1 | tee "${out_dir}/bare-leg-grid-witness.log"
s="${PIPESTATUS[0]}"
unset MLX_E120_QMV_PIPELINE_LOG
[[ "${s}" -eq 0 ]] || rc="${s}"

python3 research/e135_columns_check.py "${witness_dir}/pipelines.json" \
  --want tight | tee -a "${out_dir}/bare-leg-grid-witness.log"
if [[ "${PIPESTATUS[0]}" != "0" ]]; then
  echo "e135_presubmit: a BARE leg did not launch a tight grid, so the" \
       "compiled default did not move" >&2
  rc=6
fi
echo "--- Rule 101 control: the same check must FAIL against wide ---"
python3 research/e135_columns_check.py "${witness_dir}/pipelines.json" \
  --want wide | tee -a "${out_dir}/bare-leg-grid-witness.log"
if [[ "${PIPESTATUS[0]}" == "0" ]]; then
  echo "e135_presubmit: the bare leg passed the WIDE check as well, so the" \
       "witness cannot fail and proves nothing" >&2
  rc=7
fi
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('default_grid witness in the log:', d.get('default_grid'))
print('parsed grid enum:              ', d['grid'])
sys.exit(0 if d.get('default_grid') == 'e135_default_grid/tight'
         and d['grid'] == 'tight' else 8)
" "${witness_dir}/pipelines.json" | tee -a "${out_dir}/bare-leg-grid-witness.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=8
echo "EXIT bare-leg-grid-witness=${rc}"

echo
echo "################ local-submit-512 ################"
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
MLXFAST_SCORE_PATH="${PWD}/${out_dir}/local-submit-512.json" \
  ./benchmark-qwen-mtp.sh --local-submit 2>&1 \
  | tee "${out_dir}/local-submit-512.log"
s="${PIPESTATUS[0]}"
echo "EXIT local-submit-512=${s}"
[[ "${s}" -eq 0 ]] || rc="${s}"

echo
echo "################ SUMMARY ################"
git rev-parse HEAD
shasum -a 256 .build-worker/release/mlxfast-runtime-worker
jq -c '{passed, m:(.metrics|{decode_tokens,all_tokens_matched,residual_divergence_count,public_drift_tripwire_passed,mtp_seconds_per_token,serial_seconds_per_token,mtp_decode_speedup,effective_mean_draft_len,accepted_draft_rate})}' \
  "${out_dir}/local-submit-512.json" 2>/dev/null || true
echo "chain rc=${rc}"
exit "${rc}"
