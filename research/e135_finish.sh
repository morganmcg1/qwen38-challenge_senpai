#!/usr/bin/env bash
# E135 tail chain: retry the 512-token --local-submit gate, then run the
# rung 2 / rung 3 traced 2x2. The second stage runs even if the first fails,
# because the 2x2 does not depend on the submit gate.
set -u

cd "$(dirname "$0")/.." || exit 1
out_dir="research/out/e135-presubmit"
mkdir -p "${out_dir}"

unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E120_QMV_PIPELINE_LOG

echo "################ local-submit-512 (retry) ################"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
env MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
  ./benchmark-qwen-mtp.sh --local-submit \
  2>&1 | tee "${out_dir}/local-submit-512-retry.log"
submit_rc="${PIPESTATUS[0]}"
echo "EXIT local-submit-512-retry=${submit_rc}"
date -u +"end %Y-%m-%dT%H:%M:%SZ"

# The 512-token window is the campaign headline, but the documented default for
# this mode is 128. If the long window dies twice on this 48 GiB host, the
# shorter window still exercises the same exactness and row-ledger gate.
submit128_rc="skipped"
if [[ "${submit_rc}" -ne 0 ]]; then
  echo
  echo "################ local-submit-128 (fallback) ################"
  date -u +"start %Y-%m-%dT%H:%M:%SZ"
  ./benchmark-qwen-mtp.sh --local-submit \
    2>&1 | tee "${out_dir}/local-submit-128.log"
  submit128_rc="${PIPESTATUS[0]}"
  echo "EXIT local-submit-128=${submit128_rc}"
  date -u +"end %Y-%m-%dT%H:%M:%SZ"
fi

echo
echo "################ rung 2 + 3 traced 2x2 ################"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
research/e135_2x2_abba.sh 2 512 s2 1 2>&1 | tee "${out_dir}/2x2-s2.log"
abba_rc="${PIPESTATUS[0]}"
echo "EXIT 2x2-s2=${abba_rc}"
date -u +"end %Y-%m-%dT%H:%M:%SZ"

echo
echo "################ SUMMARY ################"
echo "local-submit-512-retry=${submit_rc}"
echo "local-submit-128=${submit128_rc}"
echo "2x2-s2=${abba_rc}"
exit 0
