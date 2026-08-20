#!/usr/bin/env bash
# Build one E65 arm from the CURRENT tree and stage it with a content witness.
#
#   usage: research/e65_build_arm.sh ARM [--require SYM] [--forbid SYM]
#
# WHY `nm` AND NOT `strings`
#
# senpai/rebuild-and-assert-worker.sh asserts arm content with
# `strings -a WORKER | grep -c NEEDLE`. That works for a Metal JIT kernel
# source string, which is a real string literal in the binary. It does NOT work
# for a Swift function: measured on this base, `strings` finds 0 copies of
# `warmAllDepthShapes`, `snapshotScheduleSignal` and `clearRecurrentRollback`,
# all of which are certainly compiled in. Swift identifiers reach the binary
# through the MANGLED SYMBOL TABLE, not the string table.
#
# `nm -a` finds 22 copies of `warmAllDepthShapes` in the same binary. A symbol
# name is still a content assertion keyed to the source identifier, so this is
# not the retracted `__TEXT,__text` digest form (ledger 202(I)): it names the
# function under test and fails when that function is absent.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e65_build_arm.sh ARM [--require SYM] [--forbid SYM]}"
shift

require=()
forbid=()
while (($#)); do
  case "$1" in
    --require) require+=("$2"); shift 2 ;;
    --forbid) forbid+=("$2"); shift 2 ;;
    *) echo "e65_build_arm.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

worker=".build-worker/release/mlxfast-runtime-worker"
cli=".build/release/mlxfast-swift"

echo "== e65: building runtime worker for arm ${arm} =="
CLANG_MODULE_CACHE_PATH="$PWD/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1
echo "== e65: building CLI for arm ${arm} =="
CLANG_MODULE_CACHE_PATH="$PWD/.build/clang-module-cache" \
  swift build -c release --force-resolved-versions --product mlxfast-swift || exit 1

status=0
sym_total="$(nm -a "${worker}" 2>/dev/null | wc -l | tr -d ' ')"
if ((sym_total < 1000)); then
  echo "FAIL extraction: nm returned ${sym_total} lines; the probe is broken"
  status=1
else
  echo "ok   extraction: ${sym_total} symbols"
fi

witness=()
for needle in "${require[@]+"${require[@]}"}"; do
  n="$(nm -a "${worker}" 2>/dev/null | grep -c -- "${needle}")"
  if ((n < 1)); then
    echo "FAIL require '${needle}': ${n} symbols, expected >= 1"
    status=1
  else
    echo "ok   require '${needle}': ${n}"
  fi
  witness+=("require:${needle}=${n}")
done
for needle in "${forbid[@]+"${forbid[@]}"}"; do
  n="$(nm -a "${worker}" 2>/dev/null | grep -c -- "${needle}")"
  if ((n != 0)); then
    echo "FAIL forbid  '${needle}': ${n} symbols, expected 0"
    status=1
  else
    echo "ok   forbid  '${needle}': 0"
  fi
  witness+=("forbid:${needle}=${n}")
done

((status == 0)) || { echo "e65_build_arm.sh: FAIL"; exit 1; }

bin_dir="research/out/e65/bin/${arm}"
mkdir -p "${bin_dir}"
cp "${cli}" "${bin_dir}/mlxfast-swift"
cp "${worker}" "${bin_dir}/mlxfast-runtime-worker"
{
  echo "arm=${arm}"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "tree_dirty_sources=$(
    git status --porcelain -- Sources Vendor Package.swift | wc -l | tr -d ' ')"
  echo "session_sha256=$(
    shasum -a 256 Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
      | awk '{print $1}')"
  echo "worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
  echo "cli_sha256=$(shasum -a 256 "${cli}" | awk '{print $1}')"
  echo "nm_symbols=${sym_total}"
  for w in "${witness[@]+"${witness[@]}"}"; do echo "nm_${w}"; done
} > "${bin_dir}/provenance.txt"

echo "e65_build_arm.sh: PASS -> ${bin_dir}"
cat "${bin_dir}/provenance.txt"
