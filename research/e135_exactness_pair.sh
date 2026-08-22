#!/bin/bash
# E135 pre-submit exactness evidence on the ship commit (advisor F7 section 5).
#
# Leg A gives the 512-token `--local-submit` exactly ONE more attempt. The
# worker assertion (HARNESS DEFECT 36) runs before this script, not inside it.
#
# If leg A dies with the same `runtime worker closed stdout ... exit_status=15`
# transient, the script does NOT retry the long window a third time. It runs
# leg A2, the 128-token `--local-submit`, and leg B, one BARE 512-token
# exactness leg. Both are labelled.
#
# Leg B exports no `MLX_E120_QMV_GRID` and no `MLX_E120_QMV_TABLE`, so it takes
# the route the ranked runner takes, and it carries its own `columns_by_width`
# witness read off the dispatch argument (CAMPAIGN RULE 114: witness the arm
# from the run's own trace, never from the environment variable).
#
# The worker digest is recorded before and after every leg.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

out_dir="research/out/e135exact"
mkdir -p "${out_dir}"
rc=0

digest() {
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}'
}

echo "################ identity ################"
echo "head_commit          $(git rev-parse HEAD)"
echo "ship_sources_commit  1efb1916"
echo "host                 $(hostname)"
echo "worker_pre           $(digest)"
git status --porcelain -- Sources Vendor Package.swift mtp-head.manifest.json \
  | sed 's/^/dirty: /'

echo
echo "################ A: --local-submit, 512-token window, one attempt ################"
echo "head_dir=unset (Rule 64)  grid/table env unset  cool_gate=real"
env -u MLXFAST_QWEN_MTP_HEAD_DIR \
    -u MLX_E120_QMV_GRID \
    -u MLX_E120_QMV_TABLE \
    MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 \
    MLXFAST_SCORE_PATH="${PWD}/${out_dir}/local-submit-512.json" \
  ./benchmark-qwen-mtp.sh --local-submit 2>&1 \
  | tee "${out_dir}/local-submit-512.log"
a="${PIPESTATUS[0]}"
echo "EXIT local-submit-512=${a}"
echo "worker_post_A ${a} $(digest)"

if [[ "${a}" -ne 0 ]]; then
  echo
  echo "######## A2: 128-token --local-submit, the F7 fallback ########"
  echo "The long window failed again. Per F7 there is no third 512-token"
  echo "attempt. This leg is a 128-token screen and is labelled as one."
  env -u MLXFAST_QWEN_MTP_HEAD_DIR \
      -u MLX_E120_QMV_GRID \
      -u MLX_E120_QMV_TABLE \
      MLXFAST_SCORE_PATH="${PWD}/${out_dir}/local-submit-128.json" \
    ./benchmark-qwen-mtp.sh --local-submit 2>&1 \
    | tee "${out_dir}/local-submit-128.log"
  a2="${PIPESTATUS[0]}"
  echo "EXIT local-submit-128=${a2}"
  echo "worker_post_A2 ${a2} $(digest)"
  [[ "${a2}" -eq 0 ]] || rc=2
fi

echo
echo "################ B: bare 512-token exactness leg on the ship commit ################"
witness_dir="research/out/e135x512"
mkdir -p "${witness_dir}"
unset MLX_E120_QMV_GRID MLX_E120_QMV_TABLE
export MLX_E120_QMV_PIPELINE_LOG="${PWD}/${witness_dir}/pipelines.json"
research/e79_trace_leg.sh e135x512 512 2>&1 | tee "${out_dir}/bare-512.log"
b="${PIPESTATUS[0]}"
unset MLX_E120_QMV_PIPELINE_LOG
echo "EXIT bare-512=${b}"
echo "worker_post_B ${b} $(digest)"
[[ "${b}" -eq 0 ]] || rc=3

echo
echo "--- B witness: columns_by_width off this leg's own trace ---"
python3 research/e135_columns_check.py "${witness_dir}/pipelines.json" \
  --want tight | tee -a "${out_dir}/bare-512.log"
[[ "${PIPESTATUS[0]}" -eq 0 ]] || rc=4
echo "--- Rule 101 control: the same check must FAIL against wide ---"
python3 research/e135_columns_check.py "${witness_dir}/pipelines.json" \
  --want wide | tee -a "${out_dir}/bare-512.log"
[[ "${PIPESTATUS[0]}" -ne 0 ]] || rc=5

echo
echo "################ SUMMARY ################"
for f in "${out_dir}/local-submit-512.json" \
         "${out_dir}/local-submit-128.json" \
         "${witness_dir}/score.json"; do
  [[ -f "${f}" ]] || continue
  echo "--- ${f}"
  jq -c '{passed, m:(.metrics|{decode_tokens, all_tokens_matched,
      residual_divergence_count, public_drift_tripwire_passed,
      mtp_seconds_per_token, serial_seconds_per_token, mtp_decode_speedup,
      effective_mean_draft_len, accepted_draft_rate})}' "${f}" 2>/dev/null \
    || cat "${f}"
done
echo "worker_final $(digest)"
echo "chain rc=${rc}"
exit "${rc}"
