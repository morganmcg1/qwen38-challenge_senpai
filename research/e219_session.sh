#!/usr/bin/env bash
# E219: run one pass-anatomy phase of the QMV census.
#
#   research/e219_session.sh TAG PHASE [EXTRA_ENV_ASSIGNMENTS...]
#
# Phases:
#   sanity   prove the launch shim is bit-exact against the shipped routed
#            dispatch. No timing. Must pass before any sweep is believed.
#   ipg      NA (= inputs-per-group = activation width) 2..5 at G = 1, cold and
#            hot, on the three top scored cells.
#   groups   G in {1,2,3} at fixed NA -> per-pass fixed cost and the
#            FINDING 559 second-pass ratio.
#   kext     k ladder at n = 5120 -> weight-stream coefficient b.
#   next     n ladder at k = 5120 -> output and threadgroup coefficients.
#   dsplit   identical total work in 1/2/4/8 dispatches -> dispatch overhead a.
#   rows     rows_per_simd in {4 shipped, 4 parameterized, 2} at NA in {4,5}
#            and G in {1,2} -> E221 step 1, the register-pressure premise
#            against the activation-re-read premise (FINDING 576).
#   xdtype   activation dtype in {bfloat16 shipped, float32} at NA 2..5 and
#            G in {1,2} -> E223 step 1, the price of one removed
#            bfloat16-to-float conversion per column per k-block against the
#            doubled activation bytes that removing it costs.
#
# Each phase runs in its OWN process so the replica rings of one phase are
# released before the next builds its own. Peak resident device memory is
# bounded by MLX_E219_REPLICA_CAP_MB.
#
# NOT GATE-QUALIFIED and never a whole-leg or ranked number (RULE 79). This is
# a standalone kernel microbenchmark: it holds no model, so the benchmark
# wrapper's process lock and 40C cool gate do not apply and cannot be claimed.
# Entry and exit GPU temperature are recorded per block instead, and every
# report carries harness=local-microbench with
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e219_session.sh TAG PHASE [VAR=VALUE...]}"
phase="${2:?usage: research/e219_session.sh TAG PHASE [VAR=VALUE...]}"
shift 2

out_dir="research/out/e219/${tag}"
mkdir -p "${out_dir}"

for assignment in "$@"; do
    export "${assignment?}"
done

export MLX_E219_PHASE="${phase}"
export MLX_E219_OUT="${out_dir}/${phase}.json"

case "${phase}" in
    sanity) filter="E219InstrumentSanityTests" ;;
    *) filter="E219PassAnatomyTests" ;;
esac

{
    echo "tag=${tag}"
    echo "phase=${phase}"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base_sha=$(git rev-parse HEAD)"
    echo "worktree_clean=$([ -z "$(git status --porcelain)" ] && echo true || echo false)"
    echo "host=$(sysctl -n machdep.cpu.brand_string)"
    echo "memsize_bytes=$(sysctl -n hw.memsize)"
    echo "os=$(sw_vers -productVersion)"
    echo "swift=$(swift --version 2>&1 | head -1)"
    for name in MLX_E219_BLOCKS MLX_E219_CHAINS MLX_E219_WARMUP \
        MLX_E219_TARGET_US MLX_E219_REPLICA_TARGET_MB \
        MLX_E219_REPLICA_CAP_MB MLX_E219_NA MLX_E219_GROUPS \
        MLX_E219_GROUP_NA MLX_E219_K MLX_E219_N MLX_E219_SPLITS \
        MLX_E219_DSPLIT_CELLS MLX_E219_DSPLIT_NA \
        MLX_E219_ROWS_NA MLX_E219_ROWS_GROUPS \
        MLX_E219_XDTYPE_NA MLX_E219_XDTYPE_GROUPS; do
        echo "${name}=${!name-}"
    done
} > "${out_dir}/${phase}.meta.txt"

swift test --force-resolved-versions --filter "${filter}" \
    2>&1 | tee "${out_dir}/${phase}.log"
status="${PIPESTATUS[0]}"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out_dir}/${phase}.meta.txt"
echo "exit_status=${status}" >> "${out_dir}/${phase}.meta.txt"
exit "${status}"
