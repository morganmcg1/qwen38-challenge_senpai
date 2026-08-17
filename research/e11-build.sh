#!/usr/bin/env bash
# Research-only (qwen38-r1-e11-depth-lever-showdown): build one arm's binary
# pair from an asserted source variant and stash it with hashes.
#
# E11 compares three depth levers that are all COMPILE-TIME constants, so the
# arms cannot share a binary the way an env-var A/B can. Each arm therefore
# gets its own built pair, hashed at build time and re-hashed at install time,
# so "two arms that had to differ but shared a hash" is a detectable failure
# rather than a silently duplicated measurement.
#
# OUTCOME: W5 shipped and H did not, so this harness's polarity inverted after
# it was written. W5 (sdpaWidthWallDepthCap 4 -> 5) is the branch default now,
# and H/H8 are pinned to the reverted curve commit below. Every other recipe
# stays arm scaffolding: built, measured, never committed. Note the other
# polarity too: since PR #2 merged, cap 7 is the SHIPPED default, so the cap
# arms open the cap back to 8 (C8/H8) rather than closing it.
#
# The two caps are the two arms of ONE ternary, selected per round by the
# full-accept streak gate, so an arm moves either the HOT ceiling
# (segmentedVerifyDepthCap: C8/H8) or the COLD floor (sdpaWidthWallDepthCap:
# W5/W6), and F3 pins both to 3.
#
# usage: research/e11-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
base_sha="${E11_BASE_SHA:-8970d775a63a28b610fd418c68873c236ce6b86c}"
root=.mlxfast-private/e11/bins

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e11-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

# Materialise the arm source. C/C8/F3/W5/W6 start from the campaign base (flat
# scalar h); H/H8 start from 7c85b4f, the curve default this branch has since
# reverted, so both recipes stay reproducible from history rather than from a
# HEAD that has moved on.
materialise() {
  case "$1" in
    C | C8 | F3 | W5 | W6) git show "${base_sha}:${src}" > "${src}" ;;
    H | H8) git show "${E11_CURVE_SHA:-7c85b4f}:${src}" > "${src}" ;;
    *) echo "e11-build: unknown arm $1" >&2; return 2 ;;
  esac
  case "$1" in
    C8 | H8)
      sed -i '' 's/^\( *private static let segmentedVerifyDepthCap = \)7$/\18/' \
        "${src}" ;;
    F3)
      sed -i '' 's/^\( *private static let sdpaWidthWallDepthCap = \)4$/\13/' \
        "${src}"
      sed -i '' 's/^\( *private static let segmentedVerifyDepthCap = \)7$/\13/' \
        "${src}" ;;
    # The COLD arm of the widthCap ternary. Cap 7 stays, so these raise the
    # floor without touching the ceiling.
    W5)
      sed -i '' 's/^\( *private static let sdpaWidthWallDepthCap = \)4$/\15/' \
        "${src}" ;;
    W6)
      sed -i '' 's/^\( *private static let sdpaWidthWallDepthCap = \)4$/\16/' \
        "${src}" ;;
  esac
}

# Every constant that defines the arm is asserted in the file that is about to
# be compiled. A recipe that silently no-ops (renamed constant, changed default)
# fails here instead of producing a second copy of the control.
assert_arm() {
  local arm="$1" want ok=1
  case "${arm}" in
    C | C8 | F3 | W5 | W6) want='private static let headStepCostRatio = 0.20' ;;
    H | H8) want='private static let defaultHeadStepCostRatioByDepth: \[Double\] = \[' ;;
  esac
  grep -qE "^ *${want}" "${src}" || { echo "e11-build: ${arm}: missing h form" >&2; ok=0; }
  local width=4 segcap=7
  case "${arm}" in
    C8 | H8) segcap=8 ;;
    F3) width=3; segcap=3 ;;
    W5) width=5 ;;
    W6) width=6 ;;
  esac
  grep -qE "^ *private static let sdpaWidthWallDepthCap = ${width}$" "${src}" \
    || { echo "e11-build: ${arm}: sdpaWidthWallDepthCap != ${width}" >&2; ok=0; }
  grep -qE "^ *private static let segmentedVerifyDepthCap = ${segcap}$" "${src}" \
    || { echo "e11-build: ${arm}: segmentedVerifyDepthCap != ${segcap}" >&2; ok=0; }
  grep -qE '^ *private static let segmentedStreakGate = 3$' "${src}" \
    || { echo "e11-build: ${arm}: segmentedStreakGate != 3" >&2; ok=0; }
  ((ok)) || return 3
  echo "e11-build: ${arm}: source asserted (h=${want:0:44}..., width=${width}, segcap=${segcap})"
}

status=0
for arm in "$@"; do
  echo "=== e11-build: ${arm} ==="
  materialise "${arm}" || { status=1; break; }
  assert_arm "${arm}" || { status=1; break; }
  research/rebuild.sh || { echo "e11-build: ${arm}: build failed" >&2; status=1; break; }
  dest="${root}/${arm}"
  rm -rf "${dest}"; mkdir -p "${dest}"
  cp .build/release/mlxfast-swift "${dest}/mlxfast-swift"
  cp .build-worker/release/mlxfast-runtime-worker "${dest}/mlxfast-runtime-worker"
  cp "${src}" "${dest}/source.swift"
  (cd "${dest}" && shasum -a 256 mlxfast-swift mlxfast-runtime-worker source.swift \
    > sha256.txt)
  echo "e11-build: ${arm}: installed"
  cat "${dest}/sha256.txt"
  restore
done

echo "=== e11-build: hash table ==="
for d in "${root}"/*/; do
  [[ -f "${d}sha256.txt" ]] || continue
  printf '%s\t%s\n' "$(basename "${d}")" "$(tr '\n' ' ' < "${d}sha256.txt")"
done
exit "${status}"
