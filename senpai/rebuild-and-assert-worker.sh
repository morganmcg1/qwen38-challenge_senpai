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
# CHOOSE THE WITNESS THAT MATCHES THE LANGUAGE OF YOUR ARM.
#
# `--require` / `--forbid` read the STRING table. That is correct for a Metal
# JIT arm, because the runtime-effective source of the `quantized` family is a
# real string literal compiled into the worker.
#
# `--require-symbol` / `--forbid-symbol` read the SYMBOL table through `nm -a`.
# That is the only correct witness for a SWIFT arm. A Swift function name
# reaches the binary mangled in the symbol table and never appears in the
# string table, so `strings` reports 0 for a function that is certainly
# compiled in. Measured on this campaign's worker (qwen-alphonse, PR #68):
#
#   warmAllDepthShapes                   strings=0   nm -a=22
#   snapshotScheduleSignal               strings=0
#   linearTopTwoRows                     strings=0
#
# Using `--require` on a Swift identifier therefore fails a correct build, and
# using `--forbid` on one passes every build unconditionally. The second error
# is the dangerous one: it is a guard that cannot fail.
#
# NEEDLES ARE LITERAL BY DEFAULT (ledger, HARNESS DEFECT 14).
#
# Every needle used to reach `grep` as a basic regular expression. An arm
# witness such as `sums[m] += xm[0] + xm[1];` is then read as bracket
# expressions: `[m]` matches one `m`, `[0]` matches one `0`, and the pattern as
# a whole matches nothing. That made `--require` fail on a correct build and
# `--forbid` pass on every build, which is the same guard-that-cannot-fail
# failure mode described above for Swift identifiers.
#
# Matching is therefore `grep -F` now. Pass `--regex` to opt back in to basic
# regular expressions for a caller that genuinely wants them. Run
# `--self-test` for the positive control that a bracketed `--forbid` needle
# fires.
#
# USAGE
#   senpai/rebuild-and-assert-worker.sh \
#       --require '<T, 5, 5, true>' --require '<T, 6, 6, true>' \
#       --forbid  '<T, 5, 3, true>' --forbid  '<T, 6, 3, true>'
#
#   senpai/rebuild-and-assert-worker.sh \
#       --require-symbol warmTargetLaterWindowSDPA
#
#   senpai/rebuild-and-assert-worker.sh --self-test
#
# Run it BEFORE and AFTER every timed leg and compare the reported mtime and
# sha256. Exit status is 0 only when every required needle is present at least
# once and every forbidden needle is absent.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

WORKER=".build-worker/release/mlxfast-runtime-worker"
SKIP_BUILD=0
SELF_TEST=0
FIXED=(-F)
REQUIRE=()
FORBID=()
REQUIRE_SYM=()
FORBID_SYM=()

while [ $# -gt 0 ]; do
  case "$1" in
    --require) REQUIRE+=("$2"); shift 2 ;;
    --forbid)  FORBID+=("$2");  shift 2 ;;
    --require-symbol) REQUIRE_SYM+=("$2"); shift 2 ;;
    --forbid-symbol)  FORBID_SYM+=("$2");  shift 2 ;;
    --regex)   FIXED=(); shift ;;
    --self-test) SELF_TEST=1; shift ;;
    --no-build) SKIP_BUILD=1; shift ;;
    --worker)  WORKER="$2"; shift 2 ;;
    *) echo "rebuild-and-assert-worker: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

count_strings() {
  strings -a "$1" | grep -c "${FIXED[@]+"${FIXED[@]}"}" -- "$2"
}

count_symbols() {
  nm -a "$1" 2>/dev/null | grep -c "${FIXED[@]+"${FIXED[@]}"}" -- "$2"
}

if [ "$SELF_TEST" -eq 1 ]; then
  NEEDLE='sums[m] += xm[0] + xm[1] + xm[2] + xm[3];'
  FIXTURE="$(mktemp -t rebuild-assert-selftest)"
  printf 'alpha\n%s\nomega\n' "$NEEDLE" > "$FIXTURE"
  N_FIXED="$(strings -a "$FIXTURE" | grep -c -F -- "$NEEDLE")"
  N_BRE="$(strings -a "$FIXTURE" | grep -c -- "$NEEDLE")"
  N_SHIPPED="$(count_strings "$FIXTURE" "$NEEDLE")"
  rm -f "$FIXTURE"
  echo "self-test needle        $NEEDLE"
  echo "self-test fixed         $N_FIXED (expected 1)"
  echo "self-test basic regex   $N_BRE (expected 0, HARNESS DEFECT 14)"
  echo "self-test default path  $N_SHIPPED (expected 1)"
  STATUS=0
  [ "$N_FIXED" -eq 1 ] || { echo "FAIL self-test: fixed matching missed the bracketed needle"; STATUS=1; }
  [ "$N_BRE" -eq 0 ] || { echo "FAIL self-test: basic regex matched, so this control proves nothing"; STATUS=1; }
  [ "$N_SHIPPED" -eq 1 ] || { echo "FAIL self-test: the default matcher missed the bracketed needle"; STATUS=1; }
  if [ "$STATUS" -eq 0 ]; then
    echo "rebuild-and-assert-worker: self-test PASS"
  else
    echo "rebuild-and-assert-worker: self-test FAIL"
  fi
  exit "$STATUS"
fi

if [ "${#REQUIRE[@]}" -eq 0 ] && [ "${#FORBID[@]}" -eq 0 ] \
  && [ "${#REQUIRE_SYM[@]}" -eq 0 ] && [ "${#FORBID_SYM[@]}" -eq 0 ]
then
  echo "rebuild-and-assert-worker: refusing to run with no assertion." >&2
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
  n=$(count_strings "$WORKER" "$needle")
  if [ "$n" -lt 1 ]; then
    echo "FAIL require '$needle': found $n copies, expected at least 1"
    STATUS=1
  else
    echo "ok   require '$needle': $n"
  fi
done

for needle in "${FORBID[@]+"${FORBID[@]}"}"; do
  n=$(count_strings "$WORKER" "$needle")
  if [ "$n" -ne 0 ]; then
    echo "FAIL forbid  '$needle': found $n copies, expected 0"
    STATUS=1
  else
    echo "ok   forbid  '$needle': 0"
  fi
done

for needle in "${REQUIRE_SYM[@]+"${REQUIRE_SYM[@]}"}"; do
  n=$(count_symbols "$WORKER" "$needle")
  if [ "$n" -lt 1 ]; then
    echo "FAIL require-symbol '$needle': found $n, expected at least 1"
    STATUS=1
  else
    echo "ok   require-symbol '$needle': $n"
  fi
done

for needle in "${FORBID_SYM[@]+"${FORBID_SYM[@]}"}"; do
  n=$(count_symbols "$WORKER" "$needle")
  if [ "$n" -ne 0 ]; then
    echo "FAIL forbid-symbol  '$needle': found $n, expected 0"
    STATUS=1
  else
    echo "ok   forbid-symbol  '$needle': 0"
  fi
done

# A guard against a silent failure must not itself be able to fail silently.
# An empty extraction means the probe broke, not that the binary is clean, so
# self-check each table that this invocation actually relied on.
if [ "${#REQUIRE[@]}" -ne 0 ] || [ "${#FORBID[@]}" -ne 0 ]; then
  TOTAL=$(strings -a "$WORKER" | wc -l | tr -d ' ')
  if [ "$TOTAL" -lt 1000 ]; then
    echo "FAIL extraction: strings returned only $TOTAL lines; the probe itself is broken"
    STATUS=1
  else
    echo "ok   extraction: $TOTAL strings"
  fi
fi

if [ "${#REQUIRE_SYM[@]}" -ne 0 ] || [ "${#FORBID_SYM[@]}" -ne 0 ]; then
  NSYM=$(nm -a "$WORKER" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NSYM" -lt 1000 ]; then
    echo "FAIL extraction: nm -a returned only $NSYM symbols; the probe itself is broken"
    STATUS=1
  else
    echo "ok   extraction: $NSYM symbols"
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "rebuild-and-assert-worker: PASS"
else
  echo "rebuild-and-assert-worker: FAIL"
fi
exit "$STATUS"
