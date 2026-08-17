#!/usr/bin/env bash
# Research-only (qwen38-r1-e11-depth-lever-showdown): build one arm's binary
# pair from an asserted source variant and stash it with hashes.
#
# E11 compares depth levers that are all COMPILE-TIME constants, so the arms
# cannot share a binary the way an env-var A/B can. Each arm therefore gets
# its own built pair, hashed at build time and re-hashed at install time, so
# "two arms that had to differ but shared a hash" is a detectable failure
# rather than a silently duplicated measurement.
#
# r3 REWRITE. The r2 arm set (C/C8/F3/W5/W6/H/H8) is retired: its constants no
# longer exist on this base. The promoted frontier moved the scalar to 0.18,
# the cold width wall to 5, and the hot cap to 8, and it deleted every E1
# research hook (no overrideHeadStepCostRatioByDepth, no MLX_QWEN_MTP_H_VECTOR,
# no forcedDepth), so the only h form left is the declaration itself.
#
#   S18  control: campaign base bytes, verbatim (scalar 0.18)
#   HV   candidate: this branch's HEAD (per-depth vector)
#   S20  decomposition-only: base bytes with the scalar moved back to 0.20, so
#        the r2 headline can be split into "the base moved the scalar" and
#        "the curve beats the scalar it is actually shipped against"
#
# usage: research/e11-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
base_sha="${E11_BASE_SHA:-fe38ecc21e4084e4d17dac3aa76264bb5897a614}"
curve_ref="${E11_CURVE_SHA:-HEAD}"
root=.mlxfast-private/e11/bins

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e11-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

materialise() {
  case "$1" in
    S18 | S20) git show "${base_sha}:${src}" > "${src}" ;;
    HV) git show "${curve_ref}:${src}" > "${src}" ;;
    *) echo "e11-build: unknown arm $1" >&2; return 2 ;;
  esac
  case "$1" in
    S20)
      sed -i '' 's/^\( *private static let headStepCostRatio = \)0\.18$/\10.20/' \
        "${src}" ;;
  esac
}

# Every constant that defines the arm is asserted in the file that is about to
# be compiled. A recipe that silently no-ops (renamed constant, changed
# default) fails here instead of producing a second copy of the control.
assert_arm() {
  local arm="$1" want ok=1
  case "${arm}" in
    S18) want='private static let headStepCostRatio = 0\.18' ;;
    S20) want='private static let headStepCostRatio = 0\.20' ;;
    HV) want='private static let headStepCostRatioByDepth: \[Double\] = \[' ;;
  esac
  grep -qE "^ *${want}" "${src}" \
    || { echo "e11-build: ${arm}: missing h form" >&2; ok=0; }
  # The scalar and the vector are mutually exclusive: a merge that left both
  # declarations behind would compile and measure as the control.
  case "${arm}" in
    HV) grep -qE '^ *private static let headStepCostRatio = ' "${src}" \
          && { echo "e11-build: HV: scalar h still declared" >&2; ok=0; } ;;
    *) grep -qE '^ *private static let headStepCostRatioByDepth' "${src}" \
          && { echo "e11-build: ${arm}: vector h still declared" >&2; ok=0; } ;;
  esac
  # Both depth caps are frontier defaults in r3; no arm moves them. Asserting
  # them anyway keeps an unnoticed base change from being read as a curve
  # effect.
  grep -qE '^ *private static let sdpaWidthWallDepthCap = 5$' "${src}" \
    || { echo "e11-build: ${arm}: sdpaWidthWallDepthCap != 5" >&2; ok=0; }
  grep -qE '^ *private static let segmentedVerifyDepthCap = 8$' "${src}" \
    || { echo "e11-build: ${arm}: segmentedVerifyDepthCap != 8" >&2; ok=0; }
  grep -qE '^ *private static let segmentedStreakGate = 3$' "${src}" \
    || { echo "e11-build: ${arm}: segmentedStreakGate != 3" >&2; ok=0; }
  ((ok)) || return 3
  echo "e11-build: ${arm}: source asserted (h=${want:0:48}..., width=5, segcap=8)"
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
