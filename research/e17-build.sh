#!/usr/bin/env bash
# Research-only (qwen38-r1-e17-curve-transfer-and-refit): build the two arms
# whose per-prompt pair E17 measures, from asserted source variants.
#
#   CURVE   this branch's HEAD, i.e. the merged per-depth vector
#   FLAT18  HEAD with the vector set flat to the retired scalar's 0.18
#
# FLAT18 rather than the pre-merge base file: the merged extend test is
# `reach > h[d]*(1+expected)/(1+cumH)`, and with a flat vector `cumH = d*h`,
# so a flat 0.18 vector reproduces the retired scalar rule
# `h*(1+expected)/(1+d*h)` term for term while touching nothing else in the
# file. The counterfactual is then the constants alone, not a partial revert of
# whatever else moved between bases.
#
# usage: research/e17-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
root=.mlxfast-private/e17/bins

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e17-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

materialise() {
  git show "HEAD:${src}" > "${src}" || return 2
  case "$1" in
    CURVE) ;;
    FLAT18)
      # Replace the two value lines of the vector literal, keyed on the shipped
      # values so a base that re-fitted them fails the assert below instead of
      # silently producing a third arm.
      python3 - "${src}" <<'PY' || return 2
import re, sys
path = sys.argv[1]
text = open(path).read()
want = ("        0.0842, 0.0775, 0.2426, 0.3754,\n"
        "        0.2919, 0.3000, 0.2870, 0.3909,\n")
flat = ("        0.1800, 0.1800, 0.1800, 0.1800,\n"
        "        0.1800, 0.1800, 0.1800, 0.1800,\n")
if want not in text:
    sys.exit("e17-build: shipped vector literal not found; refusing to patch")
open(path, "w").write(text.replace(want, flat, 1))
PY
      ;;
    *) echo "e17-build: unknown arm $1" >&2; return 2 ;;
  esac
}

assert_arm() {
  local arm="$1" ok=1
  grep -qE '^ *private static let headStepCostRatioByDepth: \[Double\] = \[' "${src}" \
    || { echo "e17-build: ${arm}: vector declaration missing" >&2; ok=0; }
  grep -qE '^ *private static let headStepCostRatio = ' "${src}" \
    && { echo "e17-build: ${arm}: scalar h also declared" >&2; ok=0; }
  case "${arm}" in
    CURVE)
      grep -qE '^ *0\.0842, 0\.0775, 0\.2426, 0\.3754,$' "${src}" \
        || { echo "e17-build: CURVE: shipped h[0..3] absent" >&2; ok=0; } ;;
    FLAT18)
      [[ "$(grep -cE '^ *0\.1800, 0\.1800, 0\.1800, 0\.1800,$' "${src}")" == 2 ]] \
        || { echo "e17-build: FLAT18: flat h not installed" >&2; ok=0; } ;;
  esac
  # No arm moves the caps or the streak gate; asserting them keeps an
  # unnoticed base change from being read as an h effect.
  grep -qE '^ *private static let sdpaWidthWallDepthCap = 5$' "${src}" \
    || { echo "e17-build: ${arm}: sdpaWidthWallDepthCap != 5" >&2; ok=0; }
  grep -qE '^ *private static let segmentedVerifyDepthCap = 8$' "${src}" \
    || { echo "e17-build: ${arm}: segmentedVerifyDepthCap != 8" >&2; ok=0; }
  grep -qE '^ *private static let segmentedStreakGate = 3$' "${src}" \
    || { echo "e17-build: ${arm}: segmentedStreakGate != 3" >&2; ok=0; }
  ((ok)) || return 3
  echo "e17-build: ${arm}: source asserted"
  grep -A2 -E '^ *private static let headStepCostRatioByDepth' "${src}"
}

status=0
for arm in "$@"; do
  echo "=== e17-build: ${arm} ==="
  materialise "${arm}" || { status=1; break; }
  assert_arm "${arm}" || { status=1; break; }
  research/rebuild.sh || { echo "e17-build: ${arm}: build failed" >&2; status=1; break; }
  dest="${root}/${arm}"
  rm -rf "${dest}"; mkdir -p "${dest}"
  cp .build/release/mlxfast-swift "${dest}/mlxfast-swift"
  cp .build-worker/release/mlxfast-runtime-worker "${dest}/mlxfast-runtime-worker"
  cp "${src}" "${dest}/source.swift"
  (cd "${dest}" && shasum -a 256 mlxfast-swift mlxfast-runtime-worker source.swift \
    > sha256.txt)
  echo "e17-build: ${arm}: installed"
  cat "${dest}/sha256.txt"
  restore
done

echo "=== e17-build: hash table ==="
for d in "${root}"/*/; do
  [[ -f "${d}sha256.txt" ]] || continue
  printf '%s\t%s\n' "$(basename "${d}")" "$(tr '\n' ' ' < "${d}sha256.txt")"
done
exit "${status}"
