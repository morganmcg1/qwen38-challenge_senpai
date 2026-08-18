#!/usr/bin/env bash
# Research-only (qwen38-r1-e24-constant-scalar-dispatch-tax): build the two arms
# of the constant-hoist experiment from named git refs.
#
#   BASE  Qwen35.swift exactly as it stands on the campaign base 55c727e9.
#         Four GDN call sites each rebuild `MLXArray(...).asType(dtype)` per
#         forward, so a verify forward pays 2 x 48 = 96 one-thread
#         `v_copy_float32_bfloat16` launches.
#   MEMO  Qwen35.swift at this branch HEAD: the same two constants served from
#         a per-layer dtype-keyed memo, so those 96 launches happen once.
#
# WHY REFS AND NOT A PATCH. e17-build.sh materialises arms by rewriting source
# lines, which is right when arms are points on a parameter sweep. Here the two
# arms are two committed states of one file, so naming the refs makes the arm
# definition unforgeable: there is no edit step that could drift.
#
# ONLY Qwen35.swift SEPARATES THE ARMS. `git diff 55c727e9..HEAD` touches six
# files; the other five are Tests/ and research/, neither of which links into
# .build/release/mlxfast-swift or .build-worker/release/mlxfast-runtime-worker.
# Materialising this one file therefore yields a true base binary from the
# current checkout, and the recorded per-arm hashes prove the two builds differ.
#
# usage: research/e24-build.sh ARM [ARM ...]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

src=Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift
root=.mlxfast-private/e24/bins
base_sha="${E24_BASE_SHA:-55c727e959e26cf24333d3e8c0896f7d97ab1224}"

if [[ -n "$(git status --porcelain -- "${src}")" ]]; then
  echo "e24-build: ${src} is dirty; commit or restore before building" >&2
  exit 2
fi

restore() { git checkout -- "${src}"; }
trap restore EXIT

ref_for() {
  case "$1" in
    BASE) echo "${base_sha}" ;;
    MEMO) echo "HEAD" ;;
    *) return 2 ;;
  esac
}

materialise() {
  local ref
  ref="$(ref_for "$1")" || { echo "e24-build: unknown arm $1" >&2; return 2; }
  git show "${ref}:${src}" > "${src}" || return 2
}

# Invariants that must hold on BOTH arms. If the base moved any of these, the
# experiment is measuring something other than the constant hoist.
assert_shared() {
  local arm="$1" ok=1 line
  local -a want=(
    'final class Qwen35GatedDeltaNet: Module'
    'MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)'
    'MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)'
    'private var _negExpALog: MLXArray?'
  )
  for line in "${want[@]}"; do
    grep -qF -- "${line}" "${src}" \
      || { echo "e24-build: ${arm}: shared invariant missing: ${line}" >&2; ok=0; }
  done
  ((ok)) || return 2
}

assert_arm() {
  local arm="$1" ok=1 inline memo touched
  inline="$(grep -c 'let invScale = pow(Float(headKDim), -0.5)' "${src}")"
  memo="$(grep -c 'invScalePair' "${src}")"
  touched="$(git diff --numstat -- "${src}" | awk '{print $1+$2}')"
  touched="${touched:-0}"
  if [[ "${arm}" == BASE ]]; then
    # Four independent rebuild sites, no memo anywhere.
    ((inline == 4)) || { echo "e24-build: BASE: expected 4 inline invScale sites, found ${inline}" >&2; ok=0; }
    ((memo == 0)) || { echo "e24-build: BASE: memo present in the control arm" >&2; ok=0; }
    ((touched > 0 && touched <= 80)) \
      || { echo "e24-build: BASE: diff touches ${touched} lines, expected 1..80" >&2; ok=0; }
  else
    # One surviving literal, inside the memo; one declaration plus four calls.
    ((inline == 1)) || { echo "e24-build: MEMO: expected 1 inline invScale site, found ${inline}" >&2; ok=0; }
    ((memo == 5)) || { echo "e24-build: MEMO: expected 5 invScalePair mentions, found ${memo}" >&2; ok=0; }
    ((touched == 0)) || { echo "e24-build: MEMO is not byte-identical to HEAD" >&2; ok=0; }
  fi
  ((ok)) || return 2
}

((${#@})) || { echo "usage: research/e24-build.sh ARM [ARM ...]" >&2; exit 2; }

for arm in "$@"; do
  echo "=== e24-build: ${arm} ($(ref_for "${arm}")) ==="
  materialise "${arm}" || { echo "e24-build: ${arm}: materialise failed" >&2; exit 2; }
  assert_shared "${arm}" || exit 2
  assert_arm "${arm}" || exit 2
  git --no-pager diff --stat -- "${src}"
  research/rebuild.sh || { echo "e24-build: ${arm}: build failed" >&2; exit 2; }
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
echo "e24-build: done"
