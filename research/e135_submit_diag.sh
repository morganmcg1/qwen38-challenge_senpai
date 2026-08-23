#!/usr/bin/env bash
# Why does `--local-submit` lose its worker in the reference pass?
#
# Both modes feed the reference pass a byte-identical 512-token seed, so the
# generation input is not the difference. Two things do differ: the grid this
# experiment ships, and the public drift tripwire, which runs 1024 steps before
# --local-submit and only 256 before --local-iterate. One cell isolates each.
#
#   A  wide grid, stock 1024-step tripwire   -> does the tight grid cause it?
#   B  tight grid, 256-step tripwire         -> does the heavy tripwire cause it?
#
# Cell A is the Rule 101 control that matters: the only difference from the
# failing run is the launch grid, and the launch grid cannot reach the reference
# worker's startup.
set -u

cd "$(dirname "$0")/.." || exit 1
out_dir="research/out/e135-presubmit"
mkdir -p "${out_dir}"

unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE MLX_E120_QMV_PIPELINE_LOG

echo "################ A: wide grid, stock 1024-step tripwire ################"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
env MLX_E120_QMV_GRID=wide \
  ./benchmark-qwen-mtp.sh --local-submit \
  2>&1 | tee "${out_dir}/diag-a-wide.log"
a_rc="${PIPESTATUS[0]}"
echo "EXIT diag-a-wide=${a_rc}"
date -u +"end %Y-%m-%dT%H:%M:%SZ"

echo
echo "################ B: tight grid, 256-step tripwire ################"
date -u +"start %Y-%m-%dT%H:%M:%SZ"
env MLX_E120_QMV_GRID=tight \
  MLXFAST_QWEN_MTP_LOCAL_GOLDEN_FIXTURE=correctness_prompts/public_longcopy_gate_english_512_256.json \
  ./benchmark-qwen-mtp.sh --local-submit \
  2>&1 | tee "${out_dir}/diag-b-light-tripwire.log"
b_rc="${PIPESTATUS[0]}"
echo "EXIT diag-b-light-tripwire=${b_rc}"
date -u +"end %Y-%m-%dT%H:%M:%SZ"

echo
echo "################ SUMMARY ################"
echo "A wide grid,  1024-step tripwire = ${a_rc}"
echo "B tight grid,  256-step tripwire = ${b_rc}"
exit 0
