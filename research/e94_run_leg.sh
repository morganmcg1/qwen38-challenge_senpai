#!/usr/bin/env bash
# One E94 arm leg: install an arm -> commit -> build -> witness -> measure ->
# unwind.
#
#   usage: research/e94_run_leg.sh ARM TAG CAP [TOKENS]
#
# ARM is ship | snap4 | amin | amine92.
#
# `ship` runs the BRANCH TIP bytes with no patch at all, so the control stays
# bit-identical to the tip as the assignment requires. Every other arm applies
# `research/e94-artifacts/e94-depth-price-arms.patch` and then flips the single
# arm selector line, so the arms differ from each other by that one literal.
#
# The arm witness is a RUNTIME witness, not a symbol-table one: the selector is
# a static enum value, and `snapshotScheduleSignal` writes `arm=<rawValue>` on
# every traced round. A leg whose trace does not carry the requested label is
# discarded. The worker digest is asserted before and after the leg, and a
# digest that moves between the two reads invalidates the leg.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e94_run_leg.sh ARM TAG CAP [TOKENS]}"
tag="${2:?usage: e94_run_leg.sh ARM TAG CAP [TOKENS]}"
cap="${3:?usage: e94_run_leg.sh ARM TAG CAP [TOKENS]}"
tokens="${4:-512}"

readonly SCORED_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
readonly TESTS_FILE="Tests/MLXFastTests/QwenMTPDepthPriceTests.swift"
readonly PATCH="research/e94-artifacts/e94-depth-price-arms.patch"

case "${arm}" in
  ship|snap4|amin|amine92) ;;
  *) echo "e94_run_leg: unknown arm ${arm}" >&2; exit 2 ;;
esac

pre_patch_sha="$(git rev-parse HEAD)"
transient_sha=""

unwind() {
  if [[ -n "${transient_sha}" && "$(git rev-parse HEAD)" == "${transient_sha}" ]]; then
    git reset -q "${pre_patch_sha}"
  fi
  git checkout -q "${pre_patch_sha}" -- "${SCORED_FILE}" "${TESTS_FILE}" 2>/dev/null || true
}
trap unwind EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e94_run_leg: worktree is dirty; refusing to time over uncommitted work" >&2
  exit 1
fi

if [[ "${arm}" != "ship" ]]; then
  git apply "${PATCH}" || {
    echo "e94_run_leg: ${PATCH} does not apply to ${pre_patch_sha}" >&2
    exit 2
  }
  perl -pi -e "s/depthPriceArm: DepthPriceArm = \\.ship/depthPriceArm: DepthPriceArm = .${arm}/" \
    "${SCORED_FILE}"
  grep -q "depthPriceArm: DepthPriceArm = \.${arm}" "${SCORED_FILE}" || {
    echo "e94_run_leg: arm selector was not flipped to ${arm}" >&2
    exit 2
  }
  git add -- "${SCORED_FILE}" "${TESTS_FILE}"
  git commit -q --allow-empty -m "E94 leg ${tag}: TRANSIENT ${arm} arm bytes under measurement

Unwound to ${pre_patch_sha} when the leg exits, including on a crash, so the
branch's scored surface stays byte-identical between legs."
  transient_sha="$(git rev-parse HEAD)"
fi

out="research/out/${tag}"
mkdir -p "${out}"

senpai/rebuild-and-assert-worker.sh \
  --require qwen35_dual_rms_norm_concat_bf16_v1 \
  --require-symbol snapshotScheduleSignal \
  --forbid qwen35_dual_rms_norm_bf16_v1 \
  > "${out}/worker-assert-pre.txt" 2>&1 || {
  echo "e94_run_leg: worker assert failed before ${tag}" >&2
  tail -20 "${out}/worker-assert-pre.txt" >&2
  exit 5
}
pre_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-pre.txt" | tail -1)"
cp "${out}/worker-assert-pre.txt" /tmp/e94-worker-assert-pre.txt

MLXFAST_QWEN_MTP_DEPTH="${cap}" research/e79_trace_leg.sh "${tag}" "${tokens}"
status=$?

# e79_trace_leg.sh recreates the leg directory, so the pre-assert record is
# copied back after the run rather than before it.
cp /tmp/e94-worker-assert-pre.txt "${out}/worker-assert-pre.txt"

senpai/rebuild-and-assert-worker.sh --no-build \
  --require qwen35_dual_rms_norm_concat_bf16_v1 \
  --require-symbol snapshotScheduleSignal \
  --forbid qwen35_dual_rms_norm_bf16_v1 \
  > "${out}/worker-assert-post.txt" 2>&1
post_rc=$?
post_worker="$(awk '/^worker_sha256 /{print $2}' "${out}/worker-assert-post.txt" | tail -1)"

arm_rounds="$(grep -c "arm=${arm} " "${out}/trace.txt" 2>/dev/null || true)"
other_arm_rounds="$(grep -c 'arm=' "${out}/trace.txt" 2>/dev/null || true)"

{
  echo "e94_arm=${arm}"
  echo "e94_cap=${cap}"
  echo "experiment=e94-rung2"
  echo "branch_commit=${pre_patch_sha}"
  echo "measured_commit_unwound=${transient_sha:-<tip>}"
  echo "worker_sha256_pre=${pre_worker}"
  echo "worker_sha256_post=${post_worker}"
  echo "worker_assert_post_exit=${post_rc}"
  echo "arm_labelled_rounds=${arm_rounds}"
  echo "labelled_rounds_total=${other_arm_rounds}"
} >> "${out}/meta.txt"

if ((post_rc != 0)) || [[ "${post_worker}" != "${pre_worker}" ]]; then
  echo "e94_run_leg: ${tag} worker changed or failed its post-assert; discarding" >&2
  status=7
fi
if ((status == 0)) && [[ "${arm_rounds}" != "${other_arm_rounds}" || "${arm_rounds}" == "0" ]]; then
  echo "e94_run_leg: ${tag} traced ${arm_rounds}/${other_arm_rounds} rounds under arm=${arm}; discarding" >&2
  status=8
fi

echo "status=${status}" >> "${out}/meta.txt"
exit "${status}"
