#!/usr/bin/env bash
# E165 Stage 0, deliverable (b): the GPU-idle fraction, measured.
#
#   usage: research/e165_residency_session.sh [TOKENS]
#
# Two pinned-width legs with the hardware residency sampler attached. The two
# widths are the experiment: if the idle fraction is host work that the GPU is
# waiting on, a wider round hides more of it and the idle fraction FALLS with
# width. If the idle fraction is a per-round dependency stall, it holds roughly
# constant. That contrast is the reason to run two widths rather than one.
#
# The legs run `--no-trace`, so the trace writer cannot contribute to either
# the timing or the residency. The census legs already carry the traced
# per-round breakdown; this session answers a different question.
#
# The legs are UNGATED and diagnostic. `gate_qualified_for_timing=false` is
# recorded by the leg runner and must survive into any report.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:-512}"
session_commit="$(git rev-parse HEAD)"
worker="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"

echo "e165 residency session: tokens=${tokens} commit=${session_commit}"
echo "worker_sha256=${worker}"

for depth in 4 7; do
  tag="e165r${depth}"
  echo "=== ${tag}: pinned d=${depth} residency tokens=${tokens} ==="
  MLX_E159_FIXED_DRAFT_DEPTH="${depth}" \
    research/e165_residency_leg.sh "${tag}" "${tokens}" --no-trace
  {
    echo "e165_pinned_depth=${depth}"
    echo "e165_session_commit=${session_commit}"
    echo "e165_session_worker_sha256=${worker}"
  } >> "research/out/${tag}/meta.txt"

  now="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker | awk '{print $1}')"
  if [[ "${now}" != "${worker}" ]]; then
    echo "e165 residency session: worker changed mid-session; legs are not comparable" >&2
    exit 1
  fi
done

python3 research/e165_residency.py research/out/e165r4 research/out/e165r7 \
  --json research/e165-residency.json
