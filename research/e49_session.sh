#!/usr/bin/env bash
# Run several E49 legs back to back inside one session.
#
#   research/e49_session.sh ARM:TAG [ARM:TAG ...] [--widths L --reps N --inner N]
#
# Counterbalancing is a property of the ORDER legs are run in, so the caller
# passes the order explicitly and each job's legs are contiguous in time.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

legs=()
while [[ $# -gt 0 && "${1}" != --* ]]; do
  legs+=("${1}")
  shift
done
[[ ${#legs[@]} -gt 0 ]] || { echo "e49_session: no ARM:TAG legs given" >&2; exit 2; }

for spec in "${legs[@]}"; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  date -u "+e49_session: === %Y-%m-%dT%H:%M:%SZ leg ${tag} (${arm}) ==="
  research/e49_run_leg.sh "${arm}" "${tag}" "$@"
done
date -u "+e49_session: === %Y-%m-%dT%H:%M:%SZ session complete ==="
