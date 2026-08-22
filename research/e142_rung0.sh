#!/usr/bin/env bash
# E142 rung 0, stage 1: build the instrumented binaries and capture the scored
# verify rows for a domain-spread seed set.
#
# HOME is set here because `run_job` starts a supervised process with
# HOME=/Users/ec2-user, while the declared head, the E133 goldens and the
# thermal state this campaign uses all live under the role home.
set -uo pipefail
export HOME=/Users/ec2-user/.senpai/native/qwen38-mlx-senpai-r2/roles/student-qwen-askeladd/home
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# beagle carries the lowest per-step acceptance and holds a ranked median order
# statistic in 97.9 % of runs (FINDING 153), so it leads. The other five cover
# the remaining domain families in the E133 corpus.
seeds="beagle_a,beagle_f,essays_montaigne,medicine_hippoc,republic_jowett,travel_eothen"

research/e142_rebuild.sh || exit $?
research/e142_capture.sh --only "${seeds}" || exit $?
