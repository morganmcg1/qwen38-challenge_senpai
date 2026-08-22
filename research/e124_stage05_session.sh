#!/usr/bin/env bash
# E124 stage 0.5 -- read the acceptance regime of every candidate seed.
#
#   usage: research/e124_stage05_session.sh
#
# NOT A TIMING SESSION. Every leg runs with the per-round phase trace on, so no
# seconds figure produced here measures anything. The legs record
# `timing_valid=false`, `cool_gate_passed_real_gate=false` and
# `gate_qualified_for_timing=false` verbatim.
#
# WHAT IT ANSWERS. F92: the published median's marginal weight sits entirely on
# hidden prompts accepting 0.83-0.90 at depth 4.4-6.1, and every local prose
# fixture accepts 0.44-0.52 at depth ~2.5. This runs one shipped-schedule
# 512-token leg per candidate seed so stage 1 can be stratified instead of
# pooled. Seeds accepting >= 0.80 become stratum H.
#
# ARM. The shipped default. `DARKBLOOM_QWEN_MTP_ISLAND_ARM` is left unset, so
# `Qwen35IslandArm.fromEnvironment` returns `.all`, which is byte-identical to
# the pre-E124 install path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=(
  beagle_a beagle_b
  medicine_hippoc medicine_hist
  essays_montaigne essays_bacon
  botany_andrews
  republic_jowett
  plutarch_lives
  drama_dollhouse
  travel_eothen
)

if [[ "${E124_SKIP_BUILD:-0}" != "1" ]]; then
  # Stage 0.5 and stage 1 must share one CLI build, and `mtp-timed` runs the
  # model inside `mlxfast-swift`, not inside the runtime worker.
  senpai/rebuild-and-assert-worker.sh \
    --require-symbol Qwen35IslandArm \
    --require DARKBLOOM_QWEN_MTP_ISLAND_ARM || exit 1
fi

# `benchfixture` is the one local prompt already known to sit in the
# median-carrying band, so it is re-read in this session as the in-session
# reference point rather than quoted from an older build.
E122_RUNS_DIR=runs-e124 research/e122_rung0_session.sh benchfixture "${SEEDS[@]}"
status=$?

python3 research/e124_regime.py --runs-dir .mlxfast-private/e122/runs-e124 \
  --extra benchfixture --extra-runs-dir .mlxfast-private/e122/runs-e124

exit "${status}"
