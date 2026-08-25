#!/usr/bin/env bash
# E219: run every timed pass-anatomy phase in sequence under one tag.
#
#   research/e219_sweeps.sh TAG [PHASE...]
#
# Each phase still runs in its own process through research/e219_session.sh, so
# one phase releases its replica rings before the next allocates. A phase that
# fails does not stop the remaining phases; the exit status is the count of
# failed phases and the per-phase status stays in its own meta file.
#
# NOT GATE-QUALIFIED and never a whole-leg or ranked number (RULE 79).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e219_sweeps.sh TAG [PHASE...]}"
shift
phases=("$@")
if [ "${#phases[@]}" -eq 0 ]; then
    phases=(groups ipg kext next dsplit)
fi

failed=0
for phase in "${phases[@]}"; do
    echo "=== e219 ${tag} ${phase} $(date -u +%H:%M:%SZ) ==="
    if ! research/e219_session.sh "${tag}" "${phase}"; then
        failed=$((failed + 1))
        echo "=== e219 ${tag} ${phase} FAILED ==="
    fi
done
echo "=== e219 ${tag} done, ${failed} failed phase(s) ==="
exit "${failed}"
