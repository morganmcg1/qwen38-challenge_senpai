#!/usr/bin/env bash
# Passive receipt watcher: exit 0 as soon as one ABBA leg has written its score.
#
#   usage: research/e121_wait_leg.sh LEG_TAG
#
# The advisor compressed rung 3 to two ABBA quads, but the driver was already
# running three. This watcher exists only so the controller can wake the
# conversation at the end of quad 2 instead of at the end of quad 3. It reads
# one path, touches no git state, and uses no GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

leg="${1:?usage: research/e121_wait_leg.sh LEG_TAG}"
target="research/out/${leg}/score.json"

while [[ ! -s "${target}" ]]; do
  sleep 20
done

echo "receipt: ${target}"
