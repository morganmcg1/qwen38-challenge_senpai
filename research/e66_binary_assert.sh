#!/usr/bin/env bash
# Research-only (qwen38-r1-e66): prove by CONTENT that the scored worker binary
# embeds THIS arm's runtime-effective JIT source, and refuse to time it
# otherwise.
#
# Same method as research/e61_binary_assert.sh, extended to pin M=5 as well as
# M=6, because E66 moves both cells. M=9 is pinned too: it carries E55's
# `<T,9,5>` and no E66 arm may disturb it.
#
# The expected values are DERIVED from the twin, so this script is arm-agnostic
# and needs no edit between arms A, B and C.
#
# It also records `__TEXT,__text` and `__TEXT,__cstring` digests. Neither is a
# gate. `__text` alone is not a content witness (ledger 202(I)); the pair is
# reported so a reader can check the expected asymmetry for a JIT-string-only
# change, where `__cstring` must move between different arms and `__text` need
# not.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
worker=".build-worker/release/mlxfast-runtime-worker"

fail() { echo "e66-binary-assert: $*" >&2; exit 1; }

[[ -r "${twin}" ]] || fail "cannot read ${twin}"
[[ -x "${worker}" ]] || fail "cannot execute ${worker}"

read_na() {  # read_na M -> the NA the twin dispatches at width M
  local m="$1" hits na
  hits="$(grep -c "qmv_fast_crossrow_affine4_g64_m<T, ${m}, [0-9]\{1,\}, true>" "${twin}")"
  [[ "${hits}" == "1" ]] \
    || fail "expected exactly 1 M=${m} dispatch in ${twin}, found ${hits}"
  na="$(sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, ${m}, \([0-9]\{1,\}\), true>.*/\1/p" "${twin}")"
  [[ -n "${na}" ]] || fail "could not read the M=${m} NA from ${twin}"
  echo "${na}"
}

na5="$(read_na 5)" || exit 1
na6="$(read_na 6)" || exit 1
na9="$(read_na 9)" || exit 1
bound="$(sed -n 's/.*wide multi-row QMV supports NA in \[2, \([0-9]\{1,\}\)\].*/\1/p' "${twin}")"
[[ -n "${bound}" ]] || fail "could not read the static_assert bound from ${twin}"

for pair in "5:${na5}" "6:${na6}" "9:${na9}"; do
  m="${pair%%:*}"; na="${pair##*:}"
  ((na >= 2)) || fail "M=${m} NA=${na} is below the wide-helper minimum of 2"
  ((na <= bound)) \
    || fail "source is self-inconsistent: M=${m} dispatches NA=${na} but the wide helper asserts NA <= ${bound}"
done
# E66 composes on top of E55's promoted M=9 two-stream cell. No E66 arm touches
# it, so a value other than 5 means an arm patch went somewhere it should not.
[[ "${na9}" == "5" ]] \
  || fail "M=9 dispatches NA=${na9}; the E66 base carries E55's <T,9,5> and no arm may change it"

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

assert_only 5 "${na5}" || exit 1
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

# The lane-perturbation positive control must never reach a timed leg or an
# arm that is certified for submission, and it is greppable in the same string.
perturb_n="$(grep -ac "const int lane = (NA == 5)" "${worker}" || true)"

echo "e66-binary-assert OK: ${worker} embeds M=5 NA=${na5}, M=6 NA=${na6}, M=9 NA=${na9}, wide-helper bound [2, ${bound}], lane_perturb=${perturb_n}" >&2
echo "e66_binary_assert_m5_na=${na5}"
echo "e66_binary_assert_m6_na=${na6}"
echo "e66_binary_assert_m9_na=${na9}"
echo "e66_binary_assert_wide_bound=${bound}"
echo "e66_binary_assert_lane_perturb_copies=${perturb_n}"

section_sha() {  # section_sha SEGNAME SECTNAME
  python3 - "${worker}" "$1" "$2" <<'PY'
import hashlib
import subprocess
import sys

path, segname, sectname = sys.argv[1], sys.argv[2], sys.argv[3]
lines = subprocess.run(
    ["otool", "-l", path], capture_output=True, text=True, check=True
).stdout.splitlines()

hits = []
for i, line in enumerate(lines):
    if line.strip() != "sectname %s" % sectname:
        continue
    block = lines[i : i + 8]
    if not any(l.strip() == "segname %s" % segname for l in block):
        continue
    off = size = None
    for l in block:
        parts = l.split()
        if len(parts) != 2:
            continue
        if parts[0] == "size":
            size = int(parts[1], 16)
        elif parts[0] == "offset":
            off = int(parts[1])
    if off is None or size is None:
        sys.exit("e66-binary-assert: incomplete %s,%s header" % (segname, sectname))
    hits.append((off, size))

if len(hits) != 1:
    sys.exit("e66-binary-assert: found %d %s,%s sections; expected 1"
             % (len(hits), segname, sectname))

off, size = hits[0]
with open(path, "rb") as fh:
    fh.seek(off)
    data = fh.read(size)
if len(data) != size:
    sys.exit("e66-binary-assert: short read of %s,%s" % (segname, sectname))
print(hashlib.sha256(data).hexdigest())
PY
}

text_sha="$(section_sha __TEXT __text)" || fail "cannot hash __TEXT,__text"
cstring_sha="$(section_sha __TEXT __cstring)" || fail "cannot hash __TEXT,__cstring"

echo "e66_binary_assert_worker_text_sha256=${text_sha}"
echo "e66_binary_assert_worker_cstring_sha256=${cstring_sha}"
