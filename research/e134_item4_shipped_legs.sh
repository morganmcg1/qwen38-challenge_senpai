#!/usr/bin/env bash
# E134 item 4 -- the two authorised shipped-policy trace legs.
#
#   usage: research/e134_item4_shipped_legs.sh
#
# `medicine_hist` and `essays_montaigne` are the two fixtures where rung 1
# found the depth-4 sign inversion, and neither has an archived shipped-policy
# leg. Rung 2 could therefore only score estimator arms on a forced-depth
# population the shipped scheduler never visits. These two legs close that
# population gap for every later scheduler experiment.
#
# NOT A TIMING SESSION. The per-round phase trace writes to a file inside the
# round, so `e128_session.sh` records `timing_valid=false` verbatim.
#
# The shipped policy means `depthPriceArm == .ship`. Run this BEFORE any pb6
# flip, or the archive records the wrong scheduler.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="$(awk '/internal static let depthPriceArm: DepthPriceArm = /{
    sub(/.*= \./, ""); sub(/[^a-z0-9].*/, ""); print; exit }' \
  Sources/MLXFastModel/Qwen36MTPBlockSession.swift)"
if [[ "${arm}" != "ship" ]]; then
  echo "e134_item4: depthPriceArm is '${arm}', not 'ship'" >&2
  exit 2
fi

research/e134_build.sh
E128_FORCE=1 research/e128_session.sh medicine_hist essays_montaigne
