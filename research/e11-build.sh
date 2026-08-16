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
# Only the H recipe is shippable. The cap edits (K/F3/HK) are arm scaffolding:
# built, measured, and never committed.
#
# usage: research/e11-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
base_sha="${E11_BASE_SHA:-807c80b8dec55d39df395fae57023cd143acf1fa}"
root=.mlxfast-private/e11/bins

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e11-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

# Materialise the arm source. C/K/F3 start from the campaign base (flat scalar
# h); H/HK start from this branch's committed measured-curve default.
materialise() {
  case "$1" in
    C | K | F3) git show "${base_sha}:${src}" > "${src}" ;;
    H | HK) git show "HEAD:${src}" > "${src}" ;;
    *) echo "e11-build: unknown arm $1" >&2; return 2 ;;
  esac
  case "$1" in
    K | HK)
      sed -i '' 's/^\( *private static let segmentedVerifyDepthCap = \)8$/\17/' \
        "${src}" ;;
    F3)
      sed -i '' 's/^\( *private static let sdpaWidthWallDepthCap = \)4$/\13/' \
        "${src}"
      sed -i '' 's/^\( *private static let segmentedVerifyDepthCap = \)8$/\13/' \
        "${src}" ;;
  esac
}

# Every constant that defines the arm is asserted in the file that is about to
# be compiled. A recipe that silently no-ops (renamed constant, changed default)
# fails here instead of producing a second copy of the control.
assert_arm() {
  local arm="$1" want ok=1
  case "${arm}" in
    C | K | F3) want='private static let headStepCostRatio = 0.20' ;;
    H | HK) want='private static let defaultHeadStepCostRatioByDepth: \[Double\] = \[' ;;
  esac
  grep -qE "^ *${want}" "${src}" || { echo "e11-build: ${arm}: missing h form" >&2; ok=0; }
  local width=4 segcap=8
  case "${arm}" in
    K | HK) segcap=7 ;;
    F3) width=3; segcap=3 ;;
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
