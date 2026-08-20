#!/bin/bash
# Rebuild the runtime worker from source and assert kernel content inside the
# built artifact.
#
# WHY THIS EXISTS (ledger 202(H)). `./benchmark-qwen-mtp.sh --local-submit` can
# silently time a STALE worker binary and still report `passed: true`. The
# wrapper's METALLIB-GUARD block at benchmark-qwen-mtp.sh:200-204 extracts only
# `metallib_rebuild_required()`. It does not extract the sibling
# `swift_build_required()` at benchmark.sh:1791-1805, which is the function that
# guards `.build-worker/release/mlxfast-runtime-worker`. For the `quantized`
# kernel family the runtime-effective source is the JIT string compiled INTO the
# worker binary, so the half the wrapper refreshes is exactly the half that does
# not govern. A worker 14 minutes older than the candidate edit passed a full
# --local-submit run.
#
# `mlxfast-swift` does not carry the quantized JIT string, so no mtime or
# existence check on the CLI can ever witness a kernel edit. Only the
# .build-worker twin can.
#
# ALSO (ledger 202(I)): do not use a bare `__TEXT,__text` digest as an arm
# certificate. That digest tracks link-time layout, not kernel source content.
# Two builds of the same tree produced different digests, and two builds of
# different trees produced the same one. Assert by string content instead.
#
# USAGE
#   senpai/rebuild-and-assert-worker.sh \
#       --require '<T, 5, 5, true>' --require '<T, 6, 6, true>' \
#       --forbid  '<T, 5, 3, true>' --forbid  '<T, 6, 3, true>'
#
# Run it BEFORE and AFTER every timed leg and compare the reported mtime and
# sha256. Exit status is 0 only when every --require string is present at least
# once and every --forbid string is absent.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

WORKER=".build-worker/release/mlxfast-runtime-worker"
SKIP_BUILD=0
REQUIRE=()
FORBID=()

while [ $# -gt 0 ]; do
  case "$1" in
    --require) REQUIRE+=("$2"); shift 2 ;;
    --forbid)  FORBID+=("$2");  shift 2 ;;
    --no-build) SKIP_BUILD=1; shift ;;
    --worker)  WORKER="$2"; shift 2 ;;
    *) echo "rebuild-and-assert-worker: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

if [ "${#REQUIRE[@]}" -eq 0 ] && [ "${#FORBID[@]}" -eq 0 ]; then
  echo "rebuild-and-assert-worker: refusing to run with no --require and no --forbid." >&2
  echo "A guard that asserts nothing is not a guard." >&2
  exit 2
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "== building runtime worker (the artifact that carries the JIT string) =="
  CLANG_MODULE_CACHE_PATH="$PWD/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1
  echo "== building CLI (content-addressed no-op for kernel edits) =="
  CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1
fi

if [ ! -f "$WORKER" ]; then
  echo "rebuild-and-assert-worker: missing worker at $WORKER" >&2
  exit 1
fi

echo "worker      $WORKER"
echo "worker_mtime $(date -u -r "$WORKER" +%Y-%m-%dT%H:%M:%SZ)"
echo "worker_sha256 $(shasum -a 256 "$WORKER" | awk '{print $1}')"

STATUS=0

for needle in "${REQUIRE[@]+"${REQUIRE[@]}"}"; do
  n=$(strings -a "$WORKER" | grep -c -- "$needle")
  if [ "$n" -lt 1 ]; then
    echo "FAIL require '$needle': found $n copies, expected at least 1"
    STATUS=1
  else
    echo "ok   require '$needle': $n"
  fi
done

for needle in "${FORBID[@]+"${FORBID[@]}"}"; do
  n=$(strings -a "$WORKER" | grep -c -- "$needle")
  if [ "$n" -ne 0 ]; then
    echo "FAIL forbid  '$needle': found $n copies, expected 0"
    STATUS=1
  else
    echo "ok   forbid  '$needle': 0"
  fi
done

# A guard against a silent failure must not itself be able to fail silently.
# `strings` returning nothing at all means the extraction broke, not that the
# binary is clean.
TOTAL=$(strings -a "$WORKER" | wc -l | tr -d ' ')
if [ "$TOTAL" -lt 1000 ]; then
  echo "FAIL extraction: strings returned only $TOTAL lines; the probe itself is broken"
  STATUS=1
else
  echo "ok   extraction: $TOTAL strings"
fi

if [ "$STATUS" -eq 0 ]; then
  echo "rebuild-and-assert-worker: PASS"
else
  echo "rebuild-and-assert-worker: FAIL"
fi
exit "$STATUS"
