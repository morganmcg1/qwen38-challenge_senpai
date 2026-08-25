#!/usr/bin/env bash
# E222: run one phase of the dequant-value-sharing screen.
#
#   research/e222_session.sh TAG PHASE [EXTRA_ENV_ASSIGNMENTS...]
#
# Phases:
#   exact    every arm must be bit-exact against
#            Qwen35CustomQMV.matmulWithTable at every timed (cell, width), and
#            every fused instantiation must pass the RULE 408 write-coverage
#            census with its two positive controls. Writes exact.json and
#            write-coverage.json. No timing. Must pass before any sweep is
#            believed.
#   screen   the decisive session: m in {6, 7, 8, 9} against
#            {staged_r4, single_r4, fused_r2, fused_r4} on the three top scored
#            cells. single_r4 at m = 6 calibrates the fusion saving at the one
#            width where both geometries are register-legal.
#   widths   what the lazy load and the late scale/bias read cost on the clock.
#   shape    what the split VF0/VF1 accumulator buys, at m = 9.
#
# Each phase runs in its OWN process so the replica rings of one phase are
# released before the next builds its own. Peak resident device memory is
# bounded by MLX_E222_REPLICA_CAP_MB.
#
# NOT GATE-QUALIFIED and never a whole-leg or ranked number (RULE 79). This is
# a standalone kernel microbenchmark: it holds no model, so the benchmark
# wrapper's process lock and 40C cool gate do not apply and cannot be claimed.
# Entry and exit GPU temperature are recorded per block instead, and every
# report carries harness=local-microbench with
# cool_gate_passed_real_gate=false and gate_qualified_for_timing=false verbatim.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e222_session.sh TAG PHASE [VAR=VALUE...]}"
phase="${2:?usage: research/e222_session.sh TAG PHASE [VAR=VALUE...]}"
shift 2

out_dir="research/out/e222/${tag}"
mkdir -p "${out_dir}"

for assignment in "$@"; do
    export "${assignment?}"
done

export MLX_E222_PHASE="${phase}"
export MLX_E222_OUT_DIR="${out_dir}"

case "${phase}" in
    exact) filter="E222ExactnessTests" ;;
    *) filter="E222ValueSharingTests" ;;
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
    for name in MLX_E222_BLOCKS MLX_E222_CHAINS MLX_E222_WARMUP \
        MLX_E222_TARGET_US MLX_E222_REPLICA_TARGET_MB \
        MLX_E222_REPLICA_CAP_MB MLX_E222_WIDTHS MLX_E222_ARMS MLX_E222_OUT_DIR; do
        echo "${name}=${!name-}"
    done
} > "${out_dir}/${phase}.meta.txt"

swift test --force-resolved-versions --filter "${filter}" \
    2>&1 | tee "${out_dir}/${phase}.log"
status="${PIPESTATUS[0]}"

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out_dir}/${phase}.meta.txt"
echo "exit_status=${status}" >> "${out_dir}/${phase}.meta.txt"
exit "${status}"
