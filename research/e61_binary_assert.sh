#!/usr/bin/env bash
# Research-only (qwen38-r1-e61): prove by CONTENT that the scored worker binary
# embeds THIS arm's runtime-effective JIT source, and refuse to time it
# otherwise. Same method as research/e55_binary_assert.sh, keyed on the M=6
# dispatch instead of the M=9 one, with the M=9 cell also pinned because E61's
# base already carries E55's `<T,9,5>` and an arm must not silently disturb it.
#
# Why content and not mtime: llbuild is content-addressed, so a byte-identical
# relink is skipped and the product mtime does not move even when the build is
# up to date; and .build/release/mlxfast-swift embeds none of the quantized JIT
# string, so its mtime cannot witness a kernel edit at all. The scored binary is
# the .build-worker twin, and the kernel source is a C++ string literal compiled
# into it, so the exact dispatch line is directly greppable.
#
# The expected values are DERIVED from the twin, so this script is arm-agnostic
# and needs no edit between base, m6 and base2.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
worker=".build-worker/release/mlxfast-runtime-worker"

fail() { echo "e61-binary-assert: $*" >&2; exit 1; }

[[ -r "${twin}" ]] || fail "cannot read ${twin}"
[[ -x "${worker}" ]] || fail "cannot execute ${worker}"

# --- read what the twin claims, per width -------------------------------------
read_na() {  # read_na M -> the NA the twin dispatches at width M
  local m="$1" hits na
  hits="$(grep -c "qmv_fast_crossrow_affine4_g64_m<T, ${m}, [0-9]\{1,\}, true>" "${twin}")"
  [[ "${hits}" == "1" ]] \
    || fail "expected exactly 1 M=${m} dispatch in ${twin}, found ${hits}"
  na="$(sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, ${m}, \([0-9]\{1,\}\), true>.*/\1/p" "${twin}")"
  [[ -n "${na}" ]] || fail "could not read the M=${m} NA from ${twin}"
  echo "${na}"
}

na6="$(read_na 6)" || exit 1
na9="$(read_na 9)" || exit 1
bound="$(sed -n 's/.*wide multi-row QMV supports NA in \[2, \([0-9]\{1,\}\)\].*/\1/p' "${twin}")"
[[ -n "${bound}" ]] || fail "could not read the static_assert bound from ${twin}"

((na6 >= 2)) || fail "M=6 NA=${na6} is below the wide-helper minimum of 2"
((na6 <= bound)) \
  || fail "source is self-inconsistent: M=6 dispatches NA=${na6} but the wide helper asserts NA <= ${bound}"
((na9 <= bound)) \
  || fail "source is self-inconsistent: M=9 dispatches NA=${na9} but the wide helper asserts NA <= ${bound}"
# E61 composes on top of E55's promoted M=9 two-stream cell. No E61 arm touches
# it, so a value other than 5 means an arm patch went somewhere it should not.
[[ "${na9}" == "5" ]] \
  || fail "M=9 dispatches NA=${na9}; the E61 base carries E55's <T,9,5> and no arm may change it"

# --- prove the binary holds exactly this, and none of the alternatives ---------
assert_only() {  # assert_only M NA
  local m="$1" na="$2" want got stray n
  want="qmv_fast_crossrow_affine4_g64_m<T, ${m}, ${na}, true>"
  got="$(grep -ac "${want}" "${worker}" || true)"
  [[ "${got}" == "1" ]] \
    || fail "${worker} contains ${got} copies of '${want}'; expected 1 -- the binary does not hold this arm's source"
  for other in 2 3 4 5 6 7 8; do
    ((other == na)) && continue
    stray="qmv_fast_crossrow_affine4_g64_m<T, ${m}, ${other}, true>"
    n="$(grep -ac "${stray}" "${worker}" || true)"
    [[ "${n}" == "0" ]] \
      || fail "${worker} still contains '${stray}' (${n}x); a previous arm's source is embedded"
  done
}

assert_only 6 "${na6}" || exit 1
assert_only 9 "${na9}" || exit 1

want_bound="wide multi-row QMV supports NA in [2, ${bound}]"
got_bound="$(grep -acF "${want_bound}" "${worker}" || true)"
[[ "${got_bound}" == "1" ]] \
  || fail "${worker} contains ${got_bound} copies of '${want_bound}'; expected 1"

for other in 3 4 5 6 7 8; do
  ((other == bound)) && continue
  stray="wide multi-row QMV supports NA in [2, ${other}]"
  n="$(grep -acF "${stray}" "${worker}" || true)"
  [[ "${n}" == "0" ]] \
    || fail "${worker} still contains '${stray}' (${n}x); a previous arm's static_assert is embedded"
done

echo "e61-binary-assert OK: ${worker} embeds M=6 NA=${na6}, M=9 NA=${na9}, wide-helper bound [2, ${bound}]" >&2
echo "e61_binary_assert_m6_na=${na6}"
echo "e61_binary_assert_m9_na=${na9}"
echo "e61_binary_assert_wide_bound=${bound}"
echo "e61_binary_assert_worker_sha256=$(shasum -a 256 "${worker}" | awk '{print $1}')"
