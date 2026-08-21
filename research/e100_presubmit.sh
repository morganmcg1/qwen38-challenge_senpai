#!/usr/bin/env bash
# E100 -- the standing pre-submit chain for the M = 5 collapse candidate.
#
#   usage: research/e100_presubmit.sh
#
# Steps run in cost order: the cheap static checks first, then the Swift test
# suite, then the 512-token exactness gate. The GPU cools during the test suite,
# which is why the gated leg runs last.
#
# The worker rebuild comes first because FINDING 28 makes the worker binary the
# only witness of which kernel will run. `--require` and `--forbid` are regexes
# passed to `grep -c --`, so the needles must be free of metacharacters.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_SHA="cd0a89dadf543261a91eb6cae07c57b3f3282519"
CONTRACT_SHA="770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
out_dir="research/out/e100-presubmit"
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

step worker-assert senpai/rebuild-and-assert-worker.sh \
  --require 'qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>' \
  --forbid  'qmv_fast_crossrow_affine4_g64_m<T, 5, 3, true>' \
  --require-symbol snapshotScheduleSignal

step metallib tools/build-mlx-metallib.sh --all-build-roots
step twin-audit python3 research/twin_audit.py
step scope senpai/validate-assignment-scope.sh "${BASE_SHA}" \
  Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h \
  Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp
step budget senpai/check-editable-budget.sh "${CONTRACT_SHA}"
step boundary senpai/verify-ranked-score-boundary.sh
step swift-test swift test --force-resolved-versions

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
for f in "${out_dir}"/*.log; do
  printf '%-24s %s\n' "$(basename "${f}" .log)" \
    "$(grep -c . "${f}" 2>/dev/null) lines"
done
jq -c '{passed, m:(.metrics|{decode_tokens,all_tokens_matched,residual_divergence_count,public_drift_tripwire_passed,mtp_seconds_per_token,serial_seconds_per_token,mtp_decode_speedup})}' \
  "${out_dir}/local-submit-512.json" 2>/dev/null || true
echo "chain rc=${rc}"
exit "${rc}"
