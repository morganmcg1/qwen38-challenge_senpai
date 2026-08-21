#!/usr/bin/env bash
# Build and unit-test one E94 arm without leaving it in the tree.
#
#   usage: research/e94_test_arm.sh ARM [FILTER]
#
# Applies `research/e94-artifacts/e94-depth-price-arms.patch`, flips the arm
# selector, runs `swift test --force-resolved-versions`, and unwinds to the
# branch tip on every exit path. `ship` runs the tip bytes with no patch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e94_test_arm.sh ARM [FILTER]}"
filter="${2:-QwenMTPDepthPriceTests}"

readonly SCORED_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
readonly TESTS_FILE="Tests/MLXFastTests/QwenMTPDepthPriceTests.swift"
readonly PATCH="research/e94-artifacts/e94-depth-price-arms.patch"

pre_sha="$(git rev-parse HEAD)"

unwind() {
  git checkout -q "${pre_sha}" -- "${SCORED_FILE}" "${TESTS_FILE}" 2>/dev/null || true
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e94_test_arm: worktree is dirty; refusing to patch over uncommitted work" >&2
  exit 1
fi

if [[ "${arm}" != "ship" ]]; then
  git apply "${PATCH}" || {
    echo "e94_test_arm: ${PATCH} does not apply to ${pre_sha}" >&2
    exit 2
  }
  perl -pi -e "s/depthPriceArm: DepthPriceArm = \\.ship/depthPriceArm: DepthPriceArm = .${arm}/" \
    "${SCORED_FILE}"
  grep -q "depthPriceArm: DepthPriceArm = \.${arm}" "${SCORED_FILE}" || {
    echo "e94_test_arm: arm selector was not flipped to ${arm}" >&2
    exit 2
  }
fi

swift test --force-resolved-versions --filter "${filter}"
