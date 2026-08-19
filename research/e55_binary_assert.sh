#!/usr/bin/env bash
# Research-only (qwen38-r1-e55-compose-m9-two-stream-on-shipped-table): prove by
# CONTENT that the scored worker binary embeds THIS arm's runtime-effective JIT
# source, and refuse to time it otherwise.
#
# Why this replaces e42-run.sh's mtime freshness check:
#
#   * llbuild is content-addressed. When a recompiled object is byte-identical
#     it skips the link, so a product's mtime does not move even though the
#     build is genuinely up to date. Any content-neutral mtime bump under
#     Sources/ or Vendor/ therefore makes an mtime guard permanently
#     unsatisfiable. Measured twice on 2026-08-19: an in-place edit-and-restore
#     of quantized.h, then a touch of quantized.cpp.
#   * .build/release/mlxfast-swift embeds NONE of the quantized JIT string
#     (measured: zero matches for every M=9 dispatch literal and for the
#     static_assert text). It does not consume the kernel source, so it can
#     never witness a kernel edit. Watching its mtime was watching the wrong
#     binary.
#
# The scored binary is the .build-worker twin, and the runtime-effective kernel
# source is a C++ string literal compiled into it, so the exact dispatch line is
# directly greppable. That is a strictly stronger freshness proof than any
# timestamp: it reads the built artefact, not the filesystem.
#
# The expected values are DERIVED from the twin, so this script is arm-agnostic
# and needs no edit between base, m9two and base2.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
worker=".build-worker/release/mlxfast-runtime-worker"

fail() { echo "e55-binary-assert: $*" >&2; exit 1; }

[[ -r "${twin}" ]] || fail "cannot read ${twin}"
[[ -x "${worker}" ]] || fail "cannot execute ${worker}"

dispatch_re='qmv_fast_crossrow_affine4_g64_m<T, 9, [0-9]\{1,\}, true>'
hits="$(grep -c "${dispatch_re}" "${twin}")"
[[ "${hits}" == "1" ]] \
  || fail "expected exactly 1 M=9 dispatch in ${twin}, found ${hits}"

na="$(sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, 9, \([0-9]\{1,\}\), true>.*/\1/p" "${twin}")"
bound="$(sed -n 's/.*wide multi-row QMV supports NA in \[2, \([0-9]\{1,\}\)\].*/\1/p' "${twin}")"
[[ -n "${na}" ]] || fail "could not read the M=9 NA from ${twin}"
[[ -n "${bound}" ]] || fail "could not read the static_assert bound from ${twin}"
((na >= 2)) || fail "M=9 NA=${na} is below the wide-helper minimum of 2"
((na <= bound)) \
  || fail "source is self-inconsistent: M=9 dispatches NA=${na} but the wide helper asserts NA <= ${bound}"

# The arm's own literal must be present exactly once, and EVERY other NA the
# wide helper could carry must be absent. Absence of the counterpart is what
# rules out a stale binary from the previous arm.
want="qmv_fast_crossrow_affine4_g64_m<T, 9, ${na}, true>"
got="$(grep -ac "${want}" "${worker}" || true)"
[[ "${got}" == "1" ]] \
  || fail "${worker} contains ${got} copies of '${want}'; expected 1 -- the binary does not hold this arm's source"

for other in 2 3 4 5 6 7 8; do
  ((other == na)) && continue
  stray="qmv_fast_crossrow_affine4_g64_m<T, 9, ${other}, true>"
  n="$(grep -ac "${stray}" "${worker}" || true)"
  [[ "${n}" == "0" ]] \
    || fail "${worker} still contains '${stray}' (${n}x); a previous arm's source is embedded"
done

want_bound="wide multi-row QMV supports NA in [2, ${bound}]"
got_bound="$(grep -acF "${want_bound}" "${worker}" || true)"
[[ "${got_bound}" == "1" ]] \
  || fail "${worker} contains ${got_bound} copies of '${want_bound}'; expected 1"

for other in 3 4 5 6; do
  ((other == bound)) && continue
  stray="wide multi-row QMV supports NA in [2, ${other}]"
  n="$(grep -acF "${stray}" "${worker}" || true)"
  [[ "${n}" == "0" ]] \
    || fail "${worker} still contains '${stray}' (${n}x); a previous arm's static_assert is embedded"
done

echo "e55-binary-assert OK: ${worker} embeds M=9 NA=${na}, wide-helper bound [2, ${bound}]" >&2
echo "e55_binary_assert_m9_na=${na}"
echo "e55_binary_assert_wide_bound=${bound}"
echo "e55_binary_assert_worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
