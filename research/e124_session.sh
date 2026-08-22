#!/usr/bin/env bash
# E124 -- run a sequence of island-arm legs inside ONE session.
#
#   usage: research/e124_session.sh LABEL TOKENS ARM [ARM ...]
#
# The arm is an environment variable, so no leg needs a rebuild and every
# inter-leg gap is the same gap. That removes harness defect 25 by
# construction: with no rebuild anywhere in the schedule, no arm can inherit a
# cooling gap that another arm does not get.
#
# Legs are tagged `<LABEL>p<position><arm>` so position and arm are both
# recoverable from the tag alone.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

label="${1:?usage: e124_session.sh LABEL TOKENS ARM [ARM ...]}"
tokens="${2:?usage: e124_session.sh LABEL TOKENS ARM [ARM ...]}"
shift 2
(($#)) || { echo "e124_session.sh: no arms given" >&2; exit 2; }
order=("$@")

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e124_session.sh: worktree is dirty; refusing to run over uncommitted work" >&2
  exit 1
fi

session_commit="$(git rev-parse HEAD)"
worker_start="$(
  shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
failures=0
position=0

for arm in "${order[@]}"; do
  position=$((position + 1))
  tag="${label}p${position}${arm}"
  echo "=== ${tag}: arm=${arm} tokens=${tokens} ==="

  research/e124_leg.sh "${tag}" "${arm}" "${tokens}" --sync-head
  status=$?

  worker_now="$(
    shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  {
    echo "e124_session=${label}"
    echo "e124_position=${position}"
    echo "session_commit=${session_commit}"
    echo "worker_sha256_session_start=${worker_start}"
    echo "worker_sha256_after_leg=${worker_now}"
  } >> "research/out/${tag}/meta.txt"

  if [[ "${worker_now}" != "${worker_start}" ]]; then
    echo "e124_session.sh: ${tag} ran a worker that moved during the session" >&2
    status=7
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "e124_session.sh: ${tag} exited ${status}" >&2
    failures=$((failures + 1))
    tail -20 "research/out/${tag}/wrapper.err" 2>/dev/null
  fi

  jq -r --arg tag "${tag}" '.metrics
    | "\($tag) matched=\(.all_tokens_matched) div=\(.residual_divergence_count)"
    + " accept=\(.accepted_draft_rate) eff_d=\(.effective_mean_draft_len)"
    + " mtp_spt=\(.mtp_seconds_per_token) ratio=\(.mtp_decode_speedup)"
    + " head=\(.head_provenance_sha256)"' \
    "research/out/${tag}/score.json" 2>/dev/null \
    || echo "${tag} no score.json"
done

echo "e124_session.sh: ${label} finished with ${failures} failed leg(s)"
exit "${failures}"
