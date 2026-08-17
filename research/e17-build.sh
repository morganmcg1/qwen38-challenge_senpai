#!/usr/bin/env bash
# Research-only (qwen38-r1-e17-curve-transfer-and-refit, r2): build the arms of
# the r2 depth-price experiment from asserted source variants.
#
# BASE-STATE CORRECTION, established before any arm was built. The r2
# assignment assumes the per-depth vector `headStepCostRatioByDepth` is the
# SHIPPED policy on `af80b0fc`. It is not. `git show af80b0fc:${src}` declares
# the SCALAR `headStepCostRatio = 0.18` and the scalar extend test
# `h*(1+expected)/(1+Double(depth)*h)`; the vector and its `cumH` algebra were
# dropped by the frontier merge `b85e782` and are absent from every commit
# after it. The live shipped default is therefore r1's FLAT18 arm policy (with
# `segmentedStreakGate` now 2, not 3). `S18` below is the real control.
#
#   S18      HEAD unmodified: scalar 0.18, streak gate 2.  CONTROL.
#   CURVE    HEAD with the r1 per-depth vector and its cumH algebra restored.
#   H1LO     flat-0.18 vector, h[1] = 0.0800  (below the measured marginal)
#   H1MEAS   flat-0.18 vector, h[1] = 0.1152  (AT the measured marginal)
#   H1HI     flat-0.18 vector, h[1] = 0.3000  (above it; asymmetric bracket)
#
# INDEX CONVENTION (asserted by reading the loop, not the prose): inside
# `costModelDepth` the extend test for the step out of depth `depth` reads
# `h[depth]`, and taking that step makes the round draft `depth+1` tokens,
# which the target verifies at width M = depth+2. So h[i] prices the step from
# depth i to depth i+1 and buys verify width M = i+2. The element that governs
# "go to depth 2" is therefore h[1] -- shipped 0.18 here, 0.0775 in the r1
# curve, 0.1152 by direct forced-depth measurement.
#
# The H1* arms use the flat-0.18 vector rather than the scalar so that ONLY
# h[1] differs between them and the vector algebra is shared. With a flat
# vector `cumH` equals `Double(depth)*h` BITWISE for depths 0..5 and by 1 ulp
# at depths 6..7 (checked in float64), so the flat vector reproduces the
# shipped scalar rule term for term over the whole reachable range.
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

vector_for() {
  case "$1" in
    CURVE)  echo "0.0842, 0.0775, 0.2426, 0.3754, 0.2919, 0.3000, 0.2870, 0.3909" ;;
    H1LO)   echo "0.1800, 0.0800, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800" ;;
    H1MEAS) echo "0.1800, 0.1152, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800" ;;
    H1HI)   echo "0.1800, 0.3000, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800, 0.1800" ;;
    *) return 2 ;;
  esac
}

vec_line() { # ARM 1|2 -> the exact source line for that half of the literal
  local half
  half="$(vector_for "$1" | cut -d, -f"$([[ $2 == 1 ]] && echo 1-4 || echo 5-8)")"
  echo "        ${half# },"
}

materialise() {
  git show "HEAD:${src}" > "${src}" || return 2
  [[ "$1" == S18 ]] && return 0
  local vec
  vec="$(vector_for "$1")" || { echo "e17-build: unknown arm $1" >&2; return 2; }
  # One patch, keyed on the exact shipped lines, so a base that moved any of
  # them fails here instead of silently producing a different arm.
  VEC="${vec}" python3 - "${src}" <<'PY' || return 2
import os, sys
path = sys.argv[1]
text = open(path).read()
vec = [f"{float(v):.4f}" for v in os.environ["VEC"].split(",")]
assert len(vec) == 8
edits = [
    ("    private static let headStepCostRatio = 0.18\n",
     "    private static let headStepCostRatioByDepth: [Double] = [\n"
     f"        {', '.join(vec[:4])},\n"
     f"        {', '.join(vec[4:])},\n"
     "    ]\n"),
    ("        let h = Self.headStepCostRatio\n"
     "        var reach = 1.0\n"
     "        var expected = 0.0\n"
     "        var depth = 0\n",
     "        let h = Self.headStepCostRatioByDepth\n"
     "        var reach = 1.0\n"
     "        var expected = 0.0\n"
     "        var cumH = 0.0\n"
     "        var depth = 0\n"),
    ("            let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)\n"
     "            guard reach > threshold else { break }\n"
     "            expected += reach\n"
     "            depth += 1\n",
     "            let threshold = h[depth] * (1.0 + expected) / (1.0 + cumH)\n"
     "            guard reach > threshold else { break }\n"
     "            expected += reach\n"
     "            cumH += h[depth]\n"
     "            depth += 1\n"),
]
for old, new in edits:
    if text.count(old) != 1:
        sys.exit(f"e17-build: expected exactly one occurrence of:\n{old}")
    text = text.replace(old, new, 1)
open(path, "w").write(text)
PY
}

# Everything outside the priced-depth constants must be the shipped value on
# EVERY arm, so a stale checkout or an unrelated drift cannot be mistaken for
# an h effect.
assert_shared() {
  local arm="$1" ok=1 line
  local -a want=(
    'private static let acceptEMAAlpha = 0.15'
    'private static let sdpaWidthWallDepthCap = 5'
    'private static let segmentedVerifyDepthCap = 8'
    'private static let segmentedStreakGate = 2'
    'let conf = 1.0 / (1.0 + exp(-margin / 2.0))'
    'let conf2 = 1.0 / (1.0 + exp(-margin / 3.0))'
  )
  for line in "${want[@]}"; do
    grep -qF -- "${line}" "${src}" \
      || { echo "e17-build: ${arm}: shared invariant missing: ${line}" >&2; ok=0; }
  done
  ((ok)) || return 2
}

assert_arm() {
  local arm="$1" ok=1
  if [[ "${arm}" == S18 ]]; then
    grep -qF -- 'private static let headStepCostRatio = 0.18' "${src}" \
      || { echo "e17-build: S18: shipped scalar 0.18 missing" >&2; ok=0; }
    grep -qF -- 'let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)' "${src}" \
      || { echo "e17-build: S18: shipped scalar extend test missing" >&2; ok=0; }
    grep -qF -- 'headStepCostRatioByDepth' "${src}" \
      && { echo "e17-build: S18: vector present in the control arm" >&2; ok=0; }
  else
    local first second
    first="$(vec_line "${arm}" 1)"
    second="$(vec_line "${arm}" 2)"
    grep -qF -- 'private static let headStepCostRatioByDepth: [Double] = [' "${src}" \
      || { echo "e17-build: ${arm}: vector declaration missing" >&2; ok=0; }
    grep -qF -- "${first}" "${src}" \
      || { echo "e17-build: ${arm}: vector head '${first}' missing" >&2; ok=0; }
    grep -qF -- "${second}" "${src}" \
      || { echo "e17-build: ${arm}: vector tail '${second}' missing" >&2; ok=0; }
    grep -qF -- 'let threshold = h[depth] * (1.0 + expected) / (1.0 + cumH)' "${src}" \
      || { echo "e17-build: ${arm}: vector extend test missing" >&2; ok=0; }
    grep -qF -- 'private static let headStepCostRatio = ' "${src}" \
      && { echo "e17-build: ${arm}: scalar h also declared" >&2; ok=0; }
    # The three sweep arms differ from each other in h[1] and nothing else:
    # h[0] and h[2..7] must all still be the shipped 0.1800.
    if [[ "${arm}" == H1* ]]; then
      [[ "${first}" == "        0.1800, "*", 0.1800, 0.1800," \
        && "${second}" == "        0.1800, 0.1800, 0.1800, 0.1800," ]] \
        || { echo "e17-build: ${arm}: sweep moved an element other than h[1]" >&2; ok=0; }
    fi
  fi
  # The diff against the shipped file must touch the priced-depth block only.
  local touched
  touched="$(git diff --numstat -- "${src}" | awk '{print $1+$2}')"
  touched="${touched:-0}"
  if [[ "${arm}" == S18 ]]; then
    ((touched == 0)) || { echo "e17-build: S18 is not byte-identical to HEAD" >&2; ok=0; }
  else
    ((touched > 0 && touched <= 24)) \
      || { echo "e17-build: ${arm}: diff touches ${touched} lines, expected 1..24" >&2; ok=0; }
  fi
  ((ok)) || return 2
}

((${#@})) || { echo "usage: research/e17-build.sh ARM [ARM ...]" >&2; exit 2; }

for arm in "$@"; do
  echo "=== e17-build: ${arm} ==="
  materialise "${arm}" || { echo "e17-build: ${arm}: materialise failed" >&2; exit 2; }
  assert_shared "${arm}" || exit 2
  assert_arm "${arm}" || exit 2
  git --no-pager diff --stat -- "${src}"
  research/rebuild.sh || { echo "e17-build: ${arm}: build failed" >&2; exit 2; }
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
echo "e17-build: done"
