#!/usr/bin/env bash
# Research-only (qwen38-r1-e78): prove by CONTENT that the scored worker binary
# embeds THIS arm's runtime-effective JIT source, and refuse to run it otherwise.
#
# Same method as research/e66_binary_assert.sh, generalised for E78. A hybrid
# arm dispatches TWO inner-group counts at one width, so the expectation is a
# SET per width rather than a single value. The set is derived from the twin,
# which makes the script arm-agnostic.
#
# It also records `__TEXT,__text` and `__TEXT,__cstring` digests. Neither is a
# gate. `__text` alone is not a content witness (ledger 202(I)); the pair is
# reported so a reader can check the expected asymmetry for a JIT-string-only
# change, where `__cstring` must move between different arms.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

twin="Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
worker=".build-worker/release/mlxfast-runtime-worker"

fail() { echo "e78-binary-assert: $*" >&2; exit 1; }

[[ -r "${twin}" ]] || fail "cannot read ${twin}"
[[ -x "${worker}" ]] || fail "cannot execute ${worker}"

read_ipgs() {  # read_ipgs M -> every IPG the twin dispatches at width M, sorted
  sed -n "s/.*qmv_fast_crossrow_affine4_g64_m<T, $1, \([0-9]\{1,\}\), true>.*/\1/p" \
    "${twin}" | sort -u | tr '\n' ' ' | sed 's/ $//'
}

bound="$(sed -n 's/.*wide multi-row QMV supports NA in \[2, \([0-9]\{1,\}\)\].*/\1/p' "${twin}")"
[[ -n "${bound}" ]] || fail "could not read the static_assert bound from ${twin}"

cutoff="$(sed -n 's/.*constexpr int kQmvNarrowOutVecCutoff = \([0-9]\{1,\}\);.*/\1/p' "${twin}")"
cutoff="${cutoff:-none}"

# Width 8 is guarded by senpai/campaign-invariants.txt and no E78 arm may move
# it. Widths 3, 4 and 7 are identical in both tables and are checked the same
# way; only 5, 6 and 9 are expected to carry a set.
declare -a summary=()
for m in 3 4 5 6 7 8 9; do
  ipgs="$(read_ipgs "${m}")"
  [[ -n "${ipgs}" ]] || fail "no M=${m} wide dispatch found in ${twin}"
  for ipg in ${ipgs}; do
    ((ipg >= 2)) || fail "M=${m} IPG=${ipg} is below the wide-helper minimum of 2"
    ((ipg <= bound)) \
      || fail "source is self-inconsistent: M=${m} dispatches IPG=${ipg} but the wide helper asserts NA <= ${bound}"
    want="qmv_fast_crossrow_affine4_g64_m<T, ${m}, ${ipg}, true>"
    got="$(grep -ac "${want}" "${worker}" || true)"
    [[ "${got}" == "1" ]] \
      || fail "${worker} contains ${got} copies of '${want}'; expected 1 -- the binary does not hold this arm's source"
  done
  for other in 2 3 4 5 6 7 8; do
    [[ " ${ipgs} " == *" ${other} "* ]] && continue
    stray="qmv_fast_crossrow_affine4_g64_m<T, ${m}, ${other}, true>"
    n="$(grep -ac "${stray}" "${worker}" || true)"
    [[ "${n}" == "0" ]] \
      || fail "${worker} still contains '${stray}' (${n}x); a previous arm's source is embedded"
  done
  summary+=("m${m}=[${ipgs}]")
  echo "e78_binary_assert_m${m}_ipgs=${ipgs// /,}"
done

# The invariant the campaign pins independently of this experiment.
[[ "$(read_ipgs 8)" == "4" ]] \
  || fail "M=8 dispatches $(read_ipgs 8); senpai/campaign-invariants.txt pins it to 4"

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

# The cutoff constant must be embedded exactly when the twin declares one.
if [[ "${cutoff}" == "none" ]]; then
  n="$(grep -ac "kQmvNarrowOutVecCutoff" "${worker}" || true)"
  [[ "${n}" == "0" ]] \
    || fail "${worker} embeds kQmvNarrowOutVecCutoff (${n}x) but this arm's twin declares none"
else
  want_cutoff="constexpr int kQmvNarrowOutVecCutoff = ${cutoff};"
  n="$(grep -acF "${want_cutoff}" "${worker}" || true)"
  [[ "${n}" == "1" ]] \
    || fail "${worker} contains ${n} copies of '${want_cutoff}'; expected 1"
fi

# The lane-perturbation positive control must never reach a timed leg or an arm
# certified for submission, and it is greppable in the same string.
perturb_n="$(grep -ac "const int lane = (NA == 5)" "${worker}" || true)"

echo "e78_binary_assert_wide_bound=${bound}"
echo "e78_binary_assert_cutoff=${cutoff}"
echo "e78_binary_assert_lane_perturb_copies=${perturb_n}"
echo "e78-binary-assert OK: ${worker} embeds ${summary[*]}, wide-helper bound [2, ${bound}], cutoff=${cutoff}, lane_perturb=${perturb_n}" >&2

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
        sys.exit("e78-binary-assert: incomplete %s,%s header" % (segname, sectname))
    hits.append((off, size))

if len(hits) != 1:
    sys.exit("e78-binary-assert: found %d %s,%s sections; expected 1"
             % (len(hits), segname, sectname))

off, size = hits[0]
with open(path, "rb") as fh:
    fh.seek(off)
    data = fh.read(size)
if len(data) != size:
    sys.exit("e78-binary-assert: short read of %s,%s" % (segname, sectname))
print(hashlib.sha256(data).hexdigest())
PY
}

text_sha="$(section_sha __TEXT __text)" || fail "cannot hash __TEXT,__text"
cstring_sha="$(section_sha __TEXT __cstring)" || fail "cannot hash __TEXT,__cstring"

echo "e78_binary_assert_worker_text_sha256=${text_sha}"
echo "e78_binary_assert_worker_cstring_sha256=${cstring_sha}"
