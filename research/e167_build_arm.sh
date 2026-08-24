#!/usr/bin/env bash
# E167: build and stage ONE arm of the counter-strip session.
#
#   usage: research/e167_build_arm.sh {B0|B1|B2|B3}
#
# The four arms separate three things that the maintained base changes
# together, relative to organizer parity, in the 257-cell-per-round routed
# call `Qwen35CustomQMV.matmul`:
#
#   B0  maintained base, unmodified.               writes yes, public yes
#   B1  the two counter WRITES removed.            writes no,  public yes
#   B2  both files at organizer parity.            the full strip that ships
#   B3  the two globals made internal.             writes yes, public no
#
# B1 minus B0 prices the writes. B3 minus B0 prices the public linkage, which
# is the hypothesis that a module-visible side effect defeats the optimiser's
# effects summary at all 257 call sites whether or not the branch is taken.
# B2 minus B0 is the total that actually ships.
#
# B3 also removes two dead cross-module reads, because making the globals
# internal stops the session file from reading them. Those reads sit inside
# `if Self.traceRounds` and never execute in a timed leg, so both halves of B3
# are declaration-side and the contrast still separates declaration from write.
#
# EVERY PATCH IS ASSERTED, NOT ASSUMED. The script counts the increments and
# the declarations before and after, and refuses to stage an arm whose source
# does not match its label. It restores the tree on exit, so a failed build
# never leaves a patched worktree behind.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

arm="${1:?usage: e167_build_arm.sh ARM, one of B0 B1 B2 B3}"

qwen35="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
session="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
worker=".build-worker/release/mlxfast-runtime-worker"
metallib=".build-worker/release/mlx.metallib"
workers_root="${MLXFAST_E167_WORKERS:-$(cd ../.. && pwd)/e167-workers}"

# The tree must be clean before a patch, or the arm is not the arm it claims.
if [[ -n "$(git status --porcelain -- "${qwen35}" "${session}")" ]]; then
  echo "e167_build_arm: ${qwen35} or ${session} is already modified" >&2
  exit 1
fi
restore() { git checkout -- "${qwen35}" "${session}" 2>/dev/null || true; }
trap restore EXIT

increments_of() { grep -c 'qwen35XSums.* &+= 1' "${qwen35}" || true; }
public_decls_of() {
  grep -c '^public nonisolated(unsafe) var qwen35XSums' "${qwen35}" || true
}

before_inc="$(increments_of)"
before_pub="$(public_decls_of)"
if [[ "${before_inc}" != "2" || "${before_pub}" != "2" ]]; then
  echo "e167_build_arm: base is not as expected (inc=${before_inc} pub=${before_pub})" >&2
  exit 1
fi

case "${arm}" in
  B0)
    ;;
  B1)
    python3 - "${qwen35}" <<'PY' || exit 1
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
block = (
    "            if fused == nil {\n"
    "                qwen35XSumsStandaloneFills &+= 1\n"
    "            } else {\n"
    "                qwen35XSumsSidecarHits &+= 1\n"
    "            }\n"
)
if src.count(block) != 1:
    raise SystemExit(f"B1: expected exactly one counter block, found {src.count(block)}")
p.write_text(src.replace(block, ""))
PY
    ;;
  B2)
    git checkout upstream/main -- "${qwen35}" "${session}" || exit 1
    ;;
  B3)
    python3 - "${qwen35}" "${session}" <<'PY' || exit 1
import sys, pathlib
q, s = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
src = q.read_text()
for name in ("qwen35XSumsSidecarHits", "qwen35XSumsStandaloneFills"):
    old = f"public nonisolated(unsafe) var {name}"
    new = f"nonisolated(unsafe) var {name}"
    if src.count(old) != 1:
        raise SystemExit(f"B3: expected one declaration of {name}")
    src = src.replace(old, new)
q.write_text(src)

# The globals are no longer visible outside MLXLLM, so the two trace reads in
# MLXFastModel cannot compile. They never execute in a timed leg.
txt = s.read_text()
for name in ("qwen35XSumsSidecarHits", "qwen35XSumsStandaloneFills"):
    old = f"\\({name})"
    if txt.count(old) != 1:
        raise SystemExit(f"B3: expected one trace read of {name}")
    txt = txt.replace(old, "-1")
s.write_text(txt)
PY
    ;;
  *)
    echo "e167_build_arm: unknown arm '${arm}'" >&2; exit 1 ;;
esac

after_inc="$(increments_of)"
after_pub="$(public_decls_of)"

expect_inc=2; expect_pub=2
case "${arm}" in
  B1) expect_inc=0 ;;
  B2) expect_inc=0; expect_pub=0 ;;
  B3) expect_pub=0 ;;
esac
if [[ "${after_inc}" != "${expect_inc}" || "${after_pub}" != "${expect_pub}" ]]; then
  echo "e167_build_arm: ${arm} patch wrong (inc=${after_inc} want ${expect_inc}," \
       "pub=${after_pub} want ${expect_pub})" >&2
  exit 1
fi

echo "== e167: building worker for ${arm} (inc ${before_inc}->${after_inc}," \
     "public ${before_pub}->${after_pub}) =="
CLANG_MODULE_CACHE_PATH="$PWD/.build-worker/clang-module-cache" \
  swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product mlxfast-runtime-worker || exit 1

# RULE 197 witnesses, read from the binary that will actually run, with an
# anchor that must be present in every arm so an empty probe cannot read as a
# stripped arm.
sym_total="$(nm -a "${worker}" 2>/dev/null | wc -l | tr -d ' ')"
xsums="$(nm -a "${worker}" 2>/dev/null | grep -c 'qwen35XSums' || true)"
addr="$(nm -a "${worker}" 2>/dev/null | grep 'qwen35XSums' | grep -c 'Sivau' || true)"
anchor="$(nm -a "${worker}" 2>/dev/null | grep -c 'Qwen36MTPBlockSession' || true)"
if ((sym_total < 1000)) || ((anchor < 1)); then
  echo "e167_build_arm: witness probe is broken (symbols=${sym_total} anchor=${anchor})" >&2
  exit 1
fi
if [[ "${arm}" == "B2" ]] && ((xsums != 0)); then
  echo "e167_build_arm: B2 still carries ${xsums} qwen35XSums symbols" >&2; exit 1
fi
# The addressor probe needs its own positive control. The mangled name ends in
# `Sivau` with no separator before it, so a pattern like `_Sivau` silently
# matches nothing and reports a stripped arm for every build. B0 and B1 keep
# the public declarations, so they MUST show addressors; if they do not, the
# probe is broken rather than the arm being clean.
case "${arm}" in
  B0|B1)
    if ((addr < 1)); then
      echo "e167_build_arm: addressor probe returned 0 on ${arm}, which keeps the" \
           "public declarations; the probe is broken" >&2
      exit 1
    fi ;;
  B3)
    if ((addr != 0)); then
      echo "e167_build_arm: B3 still carries ${addr} unsafe mutable addressors" >&2
      exit 1
    fi ;;
esac

dest="${workers_root}/${arm}"
mkdir -p "${dest}"
cp "${worker}" "${dest}/mlxfast-runtime-worker"
cp "${metallib}" "${dest}/mlx.metallib"
{
  echo "arm=${arm}"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "increments=${after_inc}"
  echo "public_declarations=${after_pub}"
  echo "qwen35_sha256=$(shasum -a 256 "${qwen35}" | awk '{print $1}')"
  echo "session_sha256=$(shasum -a 256 "${session}" | awk '{print $1}')"
  echo "worker_sha256=$(shasum -a 256 "${dest}/mlxfast-runtime-worker" | awk '{print $1}')"
  echo "metallib_sha256=$(shasum -a 256 "${dest}/mlx.metallib" | awk '{print $1}')"
  echo "nm_symbols=${sym_total}"
  echo "nm_xsums=${xsums}"
  echo "nm_unsafe_addressors=${addr}"
  echo "nm_anchor=${anchor}"
} > "${dest}/provenance.txt"

echo "e167_build_arm: PASS -> ${dest}"
cat "${dest}/provenance.txt"
