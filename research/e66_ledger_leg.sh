#!/usr/bin/env bash
# Adapter so research/e66_whole_leg_session.sh can drive the rung 2 ledger legs.
#
#   research/e66_ledger_leg.sh TAG [--legs N]
#
# The session driver owns the arm patch, the transient commit and the unwind.
# It calls its leg runner as `RUNNER TAG --legs N`. The ledger runner takes only
# an arm name and always runs exactly one untimed pass, so this adapter drops
# the leg count and forwards the tag.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:?usage: research/e66_ledger_leg.sh TAG [--legs N]}"
exec research/e66-ledger-run.sh "${tag}"
