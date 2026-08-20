#!/usr/bin/env bash
# Run the E59 close-out validation chain for one arm.
#
#   research/e59_final_chain.sh base
#   research/e59_final_chain.sh candidate --local-submit
#
# Ledger 202(H): --local-submit can silently time a stale worker, because the
# wrapper refreshes the metallib but not .build-worker. For the quantized family
# the runtime-effective source is the JIT string inside the worker binary, so
# every arm asserts that string by content before and after it is timed.
#
# Ledger 202(I): a bare __TEXT,__text digest is not a content witness. Assert by
# string content.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e59_final_chain.sh base|candidate [--local-submit]}"
shift
run_submit=0
[[ "${1:-}" == "--local-submit" ]] && run_submit=1

BASE_REF="${E59_BASE_REF:-origin/senpai/qwen38-mtp-r1}"
TWINS=(
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
)
OUT_DIR="research/e59-artifacts"
LOG_DIR="${OUT_DIR}/final-chain-logs"
mkdir -p "${LOG_DIR}"

case "${arm}" in
  base)
    git checkout "${BASE_REF}" -- "${TWINS[@]}" || exit 1
    ASSERT=(--require '<T, 5, 3, true>' --require '<T, 6, 6, true>'
            --forbid  '<T, 5, 5, true>')
    ;;
  candidate)
    git checkout HEAD -- "${TWINS[@]}" || exit 1
    ASSERT=(--require '<T, 5, 5, true>' --require '<T, 6, 6, true>'
            --forbid  '<T, 5, 3, true>')
    ;;
  *)
    echo "e59_final_chain: unknown arm '${arm}'" >&2
    exit 2
    ;;
esac

# The base arm deliberately dirties the scored files. Always hand the worktree
# back in its committed state, including on failure.
restore() { git checkout HEAD -- "${TWINS[@]}"; }
trap restore EXIT

twin_sha="$(shasum -a 256 "${TWINS[@]}" | awk '{print $1}' | tr '\n' ' ')"
echo "== arm ${arm}: twin sha256 ${twin_sha}"

echo "== rebuilding the metallib into every build root =="
tools/build-mlx-metallib.sh --all-build-roots > "${LOG_DIR}/${arm}-metallib.log" 2>&1
metallib_rc=$?
tail -n 3 "${LOG_DIR}/${arm}-metallib.log"

echo "== rebuilding the worker and asserting kernel content (before timing) =="
senpai/rebuild-and-assert-worker.sh "${ASSERT[@]}" \
  > "${LOG_DIR}/${arm}-assert-before.log" 2>&1
assert_before_rc=$?
grep -E "^worker_|^(ok|FAIL) |^rebuild-and-assert-worker:" \
  "${LOG_DIR}/${arm}-assert-before.log" || tail -n 12 "${LOG_DIR}/${arm}-assert-before.log"

echo "== full Swift suite, runtime tests enabled =="
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions \
  > "${LOG_DIR}/${arm}-swift-test.log" 2>&1
swift_rc=$?
tail -n 3 "${LOG_DIR}/${arm}-swift-test.log"

submit_rc=""
assert_after_rc=""
if [ "${run_submit}" -eq 1 ]; then
  echo "== --local-submit =="
  MLXFAST_LOCAL_RUN_LOCK_DIR=/tmp/mlxfast-shared \
  DARKBLOOM_STARTUP_MEMORY_PROFILE=full \
  MLX_MAX_MB_PER_BUFFER=512 \
  MLX_MAX_OPS_PER_BUFFER=50 \
  MLXFAST_QWEN_MTP_HEAD_DIR="${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared-run" \
    ./benchmark-qwen-mtp.sh --local-submit \
    > "${LOG_DIR}/${arm}-local-submit.log" 2>&1
  submit_rc=$?
  tail -n 24 "${LOG_DIR}/${arm}-local-submit.log"
  [ -f score.json ] && cp score.json "${OUT_DIR}/e59-local-submit.json"

  echo "== asserting kernel content again (after timing) =="
  senpai/rebuild-and-assert-worker.sh --no-build "${ASSERT[@]}" \
    > "${LOG_DIR}/${arm}-assert-after.log" 2>&1
  assert_after_rc=$?
  grep -E "^worker_" "${LOG_DIR}/${arm}-assert-after.log"
fi

ARM="${arm}" TWIN_SHA="${twin_sha}" METALLIB_RC="${metallib_rc}" \
ASSERT_BEFORE_RC="${assert_before_rc}" SWIFT_RC="${swift_rc}" \
SUBMIT_RC="${submit_rc}" ASSERT_AFTER_RC="${assert_after_rc}" \
python3 - "${OUT_DIR}" "${LOG_DIR}" <<'PY'
import json, os, pathlib, re, sys

out_dir = pathlib.Path(sys.argv[1])
log_dir = pathlib.Path(sys.argv[2])
arm = os.environ["ARM"]


def rc(name):
    raw = os.environ[name]
    return int(raw) if raw != "" else None


def worker_fields(log):
    path = log_dir / log
    fields = {}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^worker_(mtime|sha256)\s+(\S+)", line)
            if m:
                fields[m.group(1)] = m.group(2)
    return fields


test_log = log_dir / f"{arm}-swift-test.log"
failing = sorted(set(re.findall(r"Test (?:case )?'?([A-Za-z0-9_]+)\(\)'? .*failed",
                                test_log.read_text(errors="replace"))))\
    if test_log.exists() else []
summary = ""
if test_log.exists():
    for line in test_log.read_text(errors="replace").splitlines():
        if "Test run with" in line and "issue" in line:
            summary = line.strip()

payload = {
    "arm": arm,
    "twin_sha256": os.environ["TWIN_SHA"].split(),
    "metallib_exit_code": rc("METALLIB_RC"),
    "assert_before": {"exit_code": rc("ASSERT_BEFORE_RC"),
                      **worker_fields(f"{arm}-assert-before.log")},
    "swift_test": {"exit_code": rc("SWIFT_RC"),
                   "summary": summary,
                   "failing_tests": failing,
                   "failing_count": len(failing)},
}
if rc("SUBMIT_RC") is not None:
    payload["local_submit"] = {"exit_code": rc("SUBMIT_RC")}
    score = pathlib.Path("score.json")
    if score.exists():
        payload["local_submit"]["score"] = json.loads(score.read_text())
    payload["assert_after"] = {"exit_code": rc("ASSERT_AFTER_RC"),
                               **worker_fields(f"{arm}-assert-after.log")}
    before = payload["assert_before"].get("sha256")
    after = payload["assert_after"].get("sha256")
    payload["worker_unchanged_across_timing"] = (
        before is not None and before == after)

path = out_dir / f"e59-final-chain-{arm}.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
print("wrote %s" % path)
print("  swift_test exit=%s failing=%d" % (payload["swift_test"]["exit_code"],
                                           payload["swift_test"]["failing_count"]))
if "local_submit" in payload:
    print("  local_submit exit=%s worker_unchanged=%s"
          % (payload["local_submit"]["exit_code"],
             payload["worker_unchanged_across_timing"]))
PY
