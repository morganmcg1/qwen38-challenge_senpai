#!/usr/bin/env bash
# Research-only (qwen38-r1-e25-per-row-draft-price): build the two arms of the
# per-row draft-price contrast.
#
#   BASE    the promoted campaign base verbatim: scalar h = 0.18, marginal row
#           priced h/(1 + d*h).  CONTROL.
#   PRICE   HEAD verbatim: arm D, marginal row priced
#           max(h/(1 + d*h), measuredRowStepRatio[d]).
#
# NEITHER ARM IS PATCHED. E21 materialised its control from HEAD and produced
# its treatment with a keyed `sed`-style patch; the patch is the part that can
# go wrong. Here both arms are `git show`n straight out of committed objects, so
# each one is byte-identical to a blob a reviewer can name:
#
#   BASE  == ${base_sha}:${src}
#   PRICE == HEAD:${src}
#
# and the ONLY difference between the two binaries is the committed E25 diff.
# That is asserted below by blob digest, not inferred from a build log.
#
# Both arms therefore also carry E21's trace-gated schedule-signal
# instrumentation, which is already in the base: the branch it adds to the timed
# path is common-mode and cannot appear as an arm effect.
#
# usage: research/e25-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Sources/MLXFastModel/Qwen36MTPBlockSession.swift
root=.mlxfast-private/e25/bins
base_sha=0d2eef9cac75d890de06a5eef4fd686c3c34c1ef

# The shipped scalar price, spelled exactly as the base spells it. If the base
# ever moves this line the control assertion fails here rather than quietly
# timing something else.
shipped_line='let threshold = h * (1.0 + expected) / (1.0 + Double(depth) * h)'
price_line='let threshold = Self.rowPriceCoefficient(depth) * (1.0 + expected)'

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e25-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

git merge-base --is-ancestor "${base_sha}" HEAD || {
  echo "e25-build: ${base_sha} is not an ancestor of HEAD; the recorded base moved" >&2
  exit 2
}

# The two arms must differ in this file and NOWHERE else, or the contrast is
# measuring the rest of the branch as well as the price.
changed="$(git diff --name-only "${base_sha}" HEAD -- Sources Package.swift Package.resolved)"
if [[ -n "${changed}" && "${changed}" != "${src}" ]]; then
  echo "e25-build: base..HEAD touches build inputs beyond ${src}:" >&2
  echo "${changed}" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

blob_for() {
  case "$1" in
    BASE) echo "${base_sha}:${src}" ;;
    PRICE) echo "HEAD:${src}" ;;
    *) return 2 ;;
  esac
}

materialise() {
  local ref
  ref="$(blob_for "$1")" || { echo "e25-build: unknown arm $1" >&2; return 2; }
  git show "${ref}" > "${src}" || return 2
  # Byte-identity with the named object, checked on the file the compiler will
  # actually read.
  local want got
  want="$(git rev-parse "${ref}")"
  got="$(git hash-object "${src}")"
  [[ "${want}" == "${got}" ]] || {
    echo "e25-build: $1: materialised file is not ${ref} (${got} != ${want})" >&2
    return 2; }
}

# Everything outside the priced threshold must be the base value on BOTH arms,
# so unrelated drift cannot be mistaken for the price's effect. The threshold
# line itself is deliberately absent from this list -- it is the treatment.
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
    'reach *= p'
    'if Self.traceRounds { snapshotScheduleSignal(widthCap: widthCap) }'
  )
  for line in "${want[@]}"; do
    grep -qF -- "${line}" "${src}" \
      || { echo "e25-build: ${arm}: shared invariant missing: ${line}" >&2; ok=0; }
  done
  ((ok)) || return 2
}

assert_arm() {
  local arm="$1" ok=1
  case "${arm}" in
    BASE)
      grep -qF -- "${shipped_line}" "${src}" \
        || { echo "e25-build: BASE: shipped scalar price missing" >&2; ok=0; }
      grep -qF -- 'rowPriceCoefficient' "${src}" \
        && { echo "e25-build: BASE: per-row price present in the control" >&2; ok=0; }
      ;;
    PRICE)
      grep -qF -- "${price_line}" "${src}" \
        || { echo "e25-build: PRICE: per-row price missing" >&2; ok=0; }
      grep -qF -- "${shipped_line}" "${src}" \
        && { echo "e25-build: PRICE: shipped scalar price still live" >&2; ok=0; }
      grep -qF -- 'private static let measuredRowStepRatio: [Double] = [0.0, 0.095904, 0.152261, 0.442442]' "${src}" \
        || { echo "e25-build: PRICE: measured step-ratio table missing or changed" >&2; ok=0; }
      ;;
  esac
  ((ok)) || return 2
}

((${#@})) || { echo "usage: research/e25-build.sh ARM [ARM ...]" >&2; exit 2; }

for arm in "$@"; do
  echo "=== e25-build: ${arm} ==="
  materialise "${arm}" || { echo "e25-build: ${arm}: materialise failed" >&2; exit 2; }
  assert_shared "${arm}" || exit 2
  assert_arm "${arm}" || exit 2
  git --no-pager diff --stat -- "${src}"
  research/rebuild.sh || { echo "e25-build: ${arm}: build failed" >&2; exit 2; }
  out="${root}/${arm}"
  mkdir -p "${out}"
  install -m 755 .build/release/mlxfast-swift "${out}/mlxfast-swift"
  install -m 755 .build-worker/release/mlxfast-runtime-worker \
    "${out}/mlxfast-runtime-worker"
  cp "${src}" "${out}/source.swift"
  git rev-parse "$(blob_for "${arm}")" > "${out}/source-blob.txt"
  ( cd "${out}" && shasum -a 256 mlxfast-swift mlxfast-runtime-worker source.swift \
      > sha256.txt )
  cat "${out}/sha256.txt"
  git checkout -- "${src}"
done

# A changed worker digest does not prove a semantic change (calibration fact b),
# but an UNCHANGED pair of digests across two arms that must differ does prove
# the experiment is void. Check it whenever both arms are present.
if [[ -s "${root}/BASE/sha256.txt" && -s "${root}/PRICE/sha256.txt" ]]; then
  for f in mlxfast-swift mlxfast-runtime-worker source.swift; do
    a="$(awk -v f="${f}" '$2==f{print $1}' "${root}/BASE/sha256.txt")"
    b="$(awk -v f="${f}" '$2==f{print $1}' "${root}/PRICE/sha256.txt")"
    if [[ "${a}" == "${b}" ]]; then
      echo "e25-build: BASE and PRICE share ${f} digest ${a}; arms are not distinct" >&2
      exit 2
    fi
  done
  echo "e25-build: BASE and PRICE digests differ for cli, worker and source"
fi
echo "e25-build: done"
