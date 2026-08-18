#!/usr/bin/env bash
# Research-only (qwen38-r1-e21-depth-preserving-row-declination): build the
# arms of the row-declination experiment from asserted source variants.
#
#   S18I   HEAD verbatim: the shipped scalar-0.18 cost-model schedule plus the
#          trace-gated schedule-signal instrumentation.  CONTROL.
#   DECL   S18I plus the whole-round declination pre-gate.
#
# BOTH arms carry the instrumentation, so the branch it adds to the timed path
# is common-mode and cannot show up as an arm effect.
#
# S18I is asserted byte-identical to HEAD rather than merely "unmodified", so a
# dirty checkout cannot be mistaken for the control.
#
# usage: research/e21-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
root=.mlxfast-private/e21/bins

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e21-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

materialise() {
  git show "HEAD:${src}" > "${src}" || return 2
  case "$1" in
    S18I) return 0 ;;
    DECL) : ;;  # patch applied by apply_decl below
    *) echo "e21-build: unknown arm $1" >&2; return 2 ;;
  esac
  apply_decl || return 2
}

# The declination pre-gate. Keyed on the exact shipped lines so a base that
# moved any of them fails here instead of silently producing a different arm.
apply_decl() {
  python3 - "${src}" <<'PY' || return 2
import sys
path = sys.argv[1]
text = open(path).read()
old = (
    "        if Self.traceRounds { snapshotScheduleSignal(widthCap: widthCap) }\n"
    "        guard cap > 0 else { return 0 }\n"
)
new = (
    "        if Self.traceRounds { snapshotScheduleSignal(widthCap: widthCap) }\n"
    "        guard cap > 0 else { return 0 }\n"
    "        if shouldDeclineRound() { return 0 }\n"
)
if text.count(old) != 1:
    sys.exit("e21-build: DECL: schedule entry anchor not unique")
text = text.replace(old, new, 1)
open(path, "w").write(text)
PY
}

# Everything outside the declination gate must be the shipped value on EVERY
# arm, so a stale checkout or unrelated drift cannot be mistaken for its effect.
assert_shared() {
  local arm="$1" ok=1 line
  local -a want=(
    'private static let headStepCostRatio = 0.18'
    'private static let acceptEMAAlpha = 0.15'
    'private static let sdpaWidthWallDepthCap = 5'
    'private static let segmentedVerifyDepthCap = 8'
    'private static let segmentedStreakGate = 2'
    'let conf = 1.0 / (1.0 + exp(-margin / 2.0))'
    'let conf2 = 1.0 / (1.0 + exp(-margin / 3.0))'
    'let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)'
    'if Self.traceRounds { snapshotScheduleSignal(widthCap: widthCap) }'
  )
  for line in "${want[@]}"; do
    grep -qF -- "${line}" "${src}" \
      || { echo "e21-build: ${arm}: shared invariant missing: ${line}" >&2; ok=0; }
  done
  ((ok)) || return 2
}

assert_arm() {
  local arm="$1" ok=1 touched
  touched="$(git diff --numstat -- "${src}" | awk '{print $1+$2}')"
  touched="${touched:-0}"
  case "${arm}" in
    S18I)
      grep -qF -- 'shouldDeclineRound' "${src}" \
        && { echo "e21-build: S18I: declination gate present in the control" >&2; ok=0; }
      ((touched == 0)) \
        || { echo "e21-build: S18I is not byte-identical to HEAD" >&2; ok=0; }
      ;;
    DECL)
      grep -qF -- 'if shouldDeclineRound() { return 0 }' "${src}" \
        || { echo "e21-build: DECL: declination gate missing" >&2; ok=0; }
      ((touched > 0 && touched <= 4)) \
        || { echo "e21-build: DECL: diff touches ${touched} lines, expected 1..4" >&2; ok=0; }
      ;;
  esac
  ((ok)) || return 2
}

((${#@})) || { echo "usage: research/e21-build.sh ARM [ARM ...]" >&2; exit 2; }

for arm in "$@"; do
  echo "=== e21-build: ${arm} ==="
  materialise "${arm}" || { echo "e21-build: ${arm}: materialise failed" >&2; exit 2; }
  assert_shared "${arm}" || exit 2
  assert_arm "${arm}" || exit 2
  git --no-pager diff --stat -- "${src}"
  research/rebuild.sh || { echo "e21-build: ${arm}: build failed" >&2; exit 2; }
  out="${root}/${arm}"
  mkdir -p "${out}"
  install -m 755 .build/release/mlxfast-swift "${out}/mlxfast-swift"
  install -m 755 .build-worker/release/mlxfast-runtime-worker \
    "${out}/mlxfast-runtime-worker"
  cp "${src}" "${out}/source.swift"
  ( cd "${out}" && shasum -a 256 mlxfast-swift mlxfast-runtime-worker source.swift \
      > sha256.txt )
  cat "${out}/sha256.txt"
  git checkout -- "${src}"
done
echo "e21-build: done"
