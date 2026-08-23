#!/usr/bin/env bash
# E151 revision r2: the one runtime gate pass the r2 request asks for.
#
# Three steps, in order, no matrix. The worker rebuild doubles as the proof
# that the built artifact carries the R1 arm and does NOT carry the parked R2
# double buffer, which is the §2 blocker stated as a property of the binary
# rather than of the source.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
OUT="research/e151-artifacts"
mkdir -p "${OUT}"

R1_ON='constexpr bool kE147NaxRetileOn = true;'
R1_OFF='constexpr bool kE147NaxRetileOn = false;'
R2_ARMED='kE151NaxDoubleBufferArmed'

echo "=== 1/3 rebuild-and-assert-worker: R1 armed, R2 absent ==="
senpai/rebuild-and-assert-worker.sh \
  --require "${R1_ON}" \
  --forbid "${R1_OFF}" \
  --forbid "${R2_ARMED}" \
  > "${OUT}/r2-worker-rebuild.log" 2>&1
rc_worker=$?
grep -E '^(worker_sha256|worker_mtime|ok |FAIL|require|forbid)' \
  "${OUT}/r2-worker-rebuild.log" | tail -12
echo "rebuild-and-assert-worker rc=${rc_worker}"
[ "${rc_worker}" -eq 0 ] || { tail -30 "${OUT}/r2-worker-rebuild.log"; exit 1; }

echo "=== 2/3 swift test --force-resolved-versions ==="
swift test --force-resolved-versions > "${OUT}/r2-swift-test.log" 2>&1
rc_test=$?
grep -cE '^.*: (error|warning): ' "${OUT}/r2-swift-test.log" >/dev/null 2>&1
tail -5 "${OUT}/r2-swift-test.log"
echo "swift test rc=${rc_test}"

echo "=== 3/3 512-token --local-submit ==="
MLXFAST_QWEN_MTP_LOCAL_SUBMIT_TOKENS=512 ./benchmark-qwen-mtp.sh --local-submit \
  > "${OUT}/r2-local-submit-512.log" 2>&1
rc_submit=$?
tail -40 "${OUT}/r2-local-submit-512.log"
echo "local-submit rc=${rc_submit}"

echo "=== chain summary ==="
echo "worker=${rc_worker} swift_test=${rc_test} local_submit=${rc_submit}"
[ "${rc_submit}" -eq 0 ] || exit 1
exit 0
