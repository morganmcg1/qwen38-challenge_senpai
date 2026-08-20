#!/usr/bin/env bash
# Count dispatches per command-buffer commit at a list of command-buffer
# geometries, on the throwaway census binary.
#
#   usage: research/e62_census_session.sh TOKENS MB:OPS [MB:OPS ...]
#
# The census takes a lock on every dispatch and every commit, so these legs are
# counted and never timed. A short window is enough: the per-round dispatch and
# commit counts are stable across rounds, and the census writes one record per
# round.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tokens="${1:?usage: e62_census_session.sh TOKENS MB:OPS [MB:OPS ...]}"
shift

for spec in "$@"; do
  IFS=: read -r mb ops <<< "${spec}"
  tag="e62-census-mb${mb}-ops${ops}"
  echo "=== census ${tag}: mb=${mb} ops=${ops}, ${tokens} tokens ==="
  research/e62_run_leg.sh "${tag}" census "${tokens}" \
    --mb "${mb}" --ops "${ops}" --census --label "mb${mb}-ops${ops}" \
    || echo "=== ${tag} FAILED; continuing ===" >&2
done

echo "=== census session complete ==="
exit 0
