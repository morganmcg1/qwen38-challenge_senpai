#!/usr/bin/env python3
"""Derive the E72 rung-2 arms FROM the shipped wide crossrow QMV kernel.

E69's AIR census found one structural break in the NA sweep: `plain` holds a
single `[4 x [4 x i16]]` alloca at NA = 2..5, and at NA = 6 a second alloca
`[4 x <6 x float>]` appears while peak live registers jump 130 -> 182. NA = 6
carries the largest single share of ranked verify-width time, so the one scored
width with private-memory traffic is also the most expensive one.

E72 rung 2 asks whether that alloca is removable at bit-identity. The answer is
a control-flow question, not an arithmetic one: the `[4 x <6 x float>]` is the
`acc` array, and it reaches memory only because the compiler stops fully
unrolling the constant-trip `r` and `m` loops once the NA = 6 body crosses its
unroll budget. Forcing those loops open is a pure storage and layout change:
`for (int r = 0; ...)` with a compile-time trip count emits exactly the same
operations on exactly the same values in exactly the same order whether it is
rolled or unrolled, and nothing in this function reduces across `r`, so no arm
below can change an emitted bit.

Arms:

  plain        the shipped body, renamed. Control.
  tailfull     `#pragma clang loop unroll(full)` on the two FINAL loops only,
               where `acc[r][m]` is indexed by a variable `m`. Smallest change
               that can reach the float alloca.
  allfull      the same pragma on every constant-trip loop in the function:
               the `r` loops, the `i` loops and the `m` loop. Reaches the
               `[4 x [4 x i16]]` packed alloca as well.
  xvec         E69 arm X, re-derived here as the mechanism control: one 8-byte
               `vec<T, 4>` load in place of four scalar T loads of x. It won
               -3.56 % at NA = 5 and did nothing (-0.22 %) at NA = 6.
  tailfullxvec tailfull + xvec.
  allfullxvec  allfull + xvec. The direct test of the E69 mechanism note: if
               `xvec` pays only where the cell is off the bandwidth roof AND
               free of scratch traffic, removing the scratch traffic at NA = 6
               should let `xvec` recover something like its NA = 5 behaviour.

  python3 research/e72_wide_gen.py            # write the generated header
  python3 research/e72_wide_gen.py --check    # verify it is still in sync
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

SHIPPED = pathlib.Path(
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
)
GENERATED = pathlib.Path("research/generated/e72_wide_arms.h")

TEMPLATE_LINE = "template <typename T, int NA, bool DIRECT_NIBBLES = false>"
SIGNATURE = "METAL_FUNC void qmv_fast_crossrow_affine4_g64_wide("
NA_ASSERT = (
    '  static_assert(NA >= 2 && NA <= 6, "wide multi-row QMV supports NA in [2, 6]");'
)
PROBE_ASSERT = '  static_assert(NA >= 2 && NA <= 9, "probe-only NA bound");'

FULL = "#pragma clang loop unroll(full)"

# ---------------------------------------------------------------------------
# The final reduction. `acc[r][m]` is indexed by the loop variable `m`, so a
# rolled `m` loop needs the vector in addressable memory. At NA <= 5 the
# compiler unrolls both loops itself and the array stays in registers; at
# NA = 6 it gives up and `[4 x <6 x float>]` appears.
# ---------------------------------------------------------------------------
TAIL = """  for (int r = 0; r < rows_per_simd; r++) {
    for (int m = 0; m < NA; m++) {
      const float reduced = simd_sum(acc[r][m]);"""
TAIL_FULL = f"""  // E72 rung 2: force open the two constant-trip loops that index `acc` by a
  // variable. Both trip counts are compile-time constants, so this changes
  // where the values live and nothing about which values are computed.
  {FULL}
  for (int r = 0; r < rows_per_simd; r++) {{
    {FULL}
    for (int m = 0; m < NA; m++) {{
      const float reduced = simd_sum(acc[r][m]);"""

# ---------------------------------------------------------------------------
# Every other constant-trip loop in the function, in source order. Each entry
# is (anchor, replacement); the anchor must appear exactly once in the shipped
# body or the generator fails rather than silently patching the wrong loop.
# ---------------------------------------------------------------------------
_INNER = [
    ("""  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }""",
     f"""  VF acc[rows_per_simd];
  {FULL}
  for (int r = 0; r < rows_per_simd; r++) {{
    acc[r] = VF(0.0f);
  }}"""),
    ("""    for (int r = 0; r < rows_per_simd; r++) {
      const int row = out_row + r;""",
     f"""    {FULL}
    for (int r = 0; r < rows_per_simd; r++) {{
      const int row = out_row + r;"""),
    ("""      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }""",
     f"""      {FULL}
      for (int i = 0; i < 4; i++) {{
        packed[r][i] = ws[i];
      }}"""),
    ("""    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }""",
     f"""    VF partial[rows_per_simd];
    {FULL}
    for (int r = 0; r < rows_per_simd; r++) {{
      partial[r] = VF(0.0f);
    }}"""),
    ("""    for (int i = 0; i < 4; i++) {
      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {""",
     f"""    {FULL}
    for (int i = 0; i < 4; i++) {{
      VF a0, a1, a2, a3;
      {FULL}
      for (int m = 0; m < NA; m++) {{"""),
    ("""      for (int r = 0; r < rows_per_simd; r++) {
        if (DIRECT_NIBBLES) {""",
     f"""      {FULL}
      for (int r = 0; r < rows_per_simd; r++) {{
        if (DIRECT_NIBBLES) {{"""),
    ("""    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }""",
     f"""    {FULL}
    for (int r = 0; r < rows_per_simd; r++) {{
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }}"""),
]

# ---------------------------------------------------------------------------
# E69 arm X, re-derived so E72 does not depend on the E69 generator.
# ---------------------------------------------------------------------------
X_SCALAR_READ = """        thread float xc[4];
        if (DIRECT_NIBBLES) {
          xc[0] = static_cast<float>(xm[0]);
          xc[1] = static_cast<float>(xm[1]);
          xc[2] = static_cast<float>(xm[2]);
          xc[3] = static_cast<float>(xm[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];
        } else {"""
X_VECTOR_READ = """        thread float xc[4];
        if (DIRECT_NIBBLES) {
          // E69 arm X: one 8-byte load in place of four 2-byte loads. The
          // compiler cannot widen these itself because `in_vec_size` is a
          // runtime value, so it can only assume 2-byte alignment. The host
          // gates this kernel on `K % 512 == 0`, and every other term of this
          // address is a multiple of 4 elements, so the 8-byte alignment does
          // hold. The same T values reach the same expressions in the same
          // order, so the arm is bit-identical by construction.
          const vec<T, 4> xv = *reinterpret_cast<const device vec<T, 4>*>(xm);
          xc[0] = static_cast<float>(xv[0]);
          xc[1] = static_cast<float>(xv[1]);
          xc[2] = static_cast<float>(xv[2]);
          xc[3] = static_cast<float>(xv[3]);
          // Preserve the incumbent BF16 expression tree used for the affine
          // bias correction; only the qdot nibble extraction changes.
          sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
        } else {"""


def shipped_body(text: str) -> str:
    start = text.index(SIGNATURE)
    start = text.rindex(TEMPLATE_LINE, 0, start)
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def substitute(body: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if body.count(old) != 1:
            raise SystemExit(
                f"anchor is not unique ({body.count(old)} matches):\n{old}"
            )
        body = body.replace(old, new)
    return body


def arm(body: str, name: str, pairs: list[tuple[str, str]]) -> str:
    out = substitute(body, pairs)
    out = out.replace(SIGNATURE, f"METAL_FUNC void {name}(")
    out = out.replace(NA_ASSERT, PROBE_ASSERT)
    if f"void {name}(" not in out:
        raise SystemExit(f"rename failed for {name}")
    return out


# ---------------------------------------------------------------------------
# Arm `split`, the advisor's named layout edit: never declare a non-native
# vector width. All accumulator state is held in ceil(NA/4) native `float4`
# chunks, where element m lives at chunk m/4, lane m%4. Every operation stays
# element-wise on the same elements in the same order, so each output element
# is the same sequence of roundings; only the container changes.
# ---------------------------------------------------------------------------
SPLIT = [
    ("  typedef vec<float, NA> VF;",
     """  // E72 arm `split`: VF is ALWAYS a native width. Element m of the old
  // vec<float, NA> lives at chunk m / 4, lane m % 4. Lanes above NA are
  // initialised and computed but never read out.
  typedef vec<float, 4> VF;
  constexpr int NC = (NA + 3) / 4;"""),
    ("""  VF acc[rows_per_simd];
  for (int r = 0; r < rows_per_simd; r++) {
    acc[r] = VF(0.0f);
  }""",
     """  VF acc[rows_per_simd][NC];
  for (int r = 0; r < rows_per_simd; r++) {
    for (int c = 0; c < NC; c++) {
      acc[r][c] = VF(0.0f);
    }
  }"""),
    ("""    VF sums = VF(0.0f);
    VF partial[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
      partial[r] = VF(0.0f);
    }""",
     """    VF sums[NC];
    for (int c = 0; c < NC; c++) {
      sums[c] = VF(0.0f);
    }
    VF partial[rows_per_simd][NC];
    for (int r = 0; r < rows_per_simd; r++) {
      for (int c = 0; c < NC; c++) {
        partial[r][c] = VF(0.0f);
      }
    }"""),
    ("""      VF a0, a1, a2, a3;
      for (int m = 0; m < NA; m++) {""",
     """      VF a0[NC], a1[NC], a2[NC], a3[NC];
      for (int c = 0; c < NC; c++) {
        a0[c] = VF(0.0f);
        a1[c] = VF(0.0f);
        a2[c] = VF(0.0f);
        a3[c] = VF(0.0f);
      }
      for (int m = 0; m < NA; m++) {"""),
    ("          sums[m] += xm[0] + xm[1] + xm[2] + xm[3];",
     "          sums[m / 4][m % 4] += xm[0] + xm[1] + xm[2] + xm[3];"),
    ("          sums[m] += load_vector<T, float, 4, 4>(xm, xc);",
     "          sums[m / 4][m % 4] += load_vector<T, float, 4, 4>(xm, xc);"),
    ("""        a0[m] = xc[0];
        a1[m] = xc[1];
        a2[m] = xc[2];
        a3[m] = xc[3];""",
     """        a0[m / 4][m % 4] = xc[0];
        a1[m / 4][m % 4] = xc[1];
        a2[m / 4][m % 4] = xc[2];
        a3[m / 4][m % 4] = xc[3];"""),
    ("""      for (int r = 0; r < rows_per_simd; r++) {
        if (DIRECT_NIBBLES) {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * ((packed[r][i] >> 4) & 0x000f) +
                         a2 * ((packed[r][i] >> 8) & 0x000f) +
                         a3 * ((packed[r][i] >> 12) & 0x000f));
        } else {
          partial[r] += (a0 * (packed[r][i] & 0x000f) +
                         a1 * (packed[r][i] & 0x00f0) +
                         a2 * (packed[r][i] & 0x0f00) +
                         a3 * (packed[r][i] & 0xf000));
        }
      }""",
     """      for (int r = 0; r < rows_per_simd; r++) {
        for (int c = 0; c < NC; c++) {
          if (DIRECT_NIBBLES) {
            partial[r][c] += (a0[c] * (packed[r][i] & 0x000f) +
                              a1[c] * ((packed[r][i] >> 4) & 0x000f) +
                              a2[c] * ((packed[r][i] >> 8) & 0x000f) +
                              a3[c] * ((packed[r][i] >> 12) & 0x000f));
          } else {
            partial[r][c] += (a0[c] * (packed[r][i] & 0x000f) +
                              a1[c] * (packed[r][i] & 0x00f0) +
                              a2[c] * (packed[r][i] & 0x0f00) +
                              a3[c] * (packed[r][i] & 0xf000));
          }
        }
      }"""),
    ("""    for (int r = 0; r < rows_per_simd; r++) {
      acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
    }""",
     """    for (int r = 0; r < rows_per_simd; r++) {
      for (int c = 0; c < NC; c++) {
        acc[r][c] += scale_local[r] * partial[r][c] + sums[c] * bias_local[r];
      }
    }"""),
    ("      const float reduced = simd_sum(acc[r][m]);",
     "      const float reduced = simd_sum(acc[r][m / 4][m % 4]);"),
]

TAILFULL = [(TAIL, TAIL_FULL)]
# _INNER[4] is the only substitution that opens the `m` loop; every other entry
# opens an `r` loop or the 4-trip `i` loop over the packed weights. Splitting
# them gives a two-way bisect of any codegen fault the parity harness finds.
MFULL = [_INNER[4]]
RFULL = [pair for index, pair in enumerate(_INNER) if index != 4] + TAILFULL
ALLFULL = _INNER + TAILFULL
XVEC = [(X_SCALAR_READ, X_VECTOR_READ)]

BASE_SYMBOL = "qmv_fast_crossrow_affine4_g64_wide"

ARMS = {
    "e72plain": [],
    "e72tailfull": TAILFULL,
    "e72mfull": MFULL,
    "e72rfull": RFULL,
    "e72allfull": ALLFULL,
    "e72split": SPLIT,
    "e72xvec": XVEC,
    "e72tailfullxvec": TAILFULL + XVEC,
}

HEADER = """// GENERATED by research/e72_wide_gen.py. Do not edit.
//
// Every arm is the shipped `qmv_fast_crossrow_affine4_g64_wide` body plus a
// closed list of exact substitutions, so no arm can drift from what ships.
// Research only: this header is never compiled into a submitted path.
//
// shipped quantized.h sha256: {digest}
#pragma once

"""


def strip_pragmas(text: str) -> str:
    """Drop the unroll pragmas and the comment block that introduces them."""
    keep, skip = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == FULL:
            skip = False
            continue
        if stripped.startswith("// E72 rung 2:"):
            skip = True
        if skip:
            if not stripped.startswith("//"):
                skip = False
            else:
                continue
        keep.append(line)
    return "".join(keep)


def assert_pragma_only(body: str, name: str, pairs: list[tuple[str, str]]) -> None:
    """A pragma-only arm must reduce to `plain` once the pragmas are removed.

    This is the bit-identity argument for the unrolled arms, stated as a check
    that can fail: `#pragma clang loop unroll(full)` on a constant-trip loop is
    a control-flow directive, so if the arm differs from the shipped body ONLY
    by such lines, it cannot compute a different value.
    """
    got = strip_pragmas(arm(body, name, pairs))
    want = arm(body, name, [])
    if got != want:
        import difflib

        diff = "\n".join(difflib.unified_diff(
            want.splitlines(), got.splitlines(), "plain", name, lineterm=""))
        raise SystemExit(f"{name} is not pragma-only:\n{diff}")


PRAGMA_ONLY = {"e72tailfull", "e72mfull", "e72rfull", "e72allfull"}


def render(text: str) -> str:
    body = shipped_body(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    parts = [HEADER.format(digest=digest)]
    for name, pairs in ARMS.items():
        full_name = f"{BASE_SYMBOL}_{name}"
        if name in PRAGMA_ONLY:
            assert_pragma_only(body, full_name, pairs)
        parts.append(
            f"// ---------------------------------------------------------------- {name}\n"
        )
        parts.append(arm(body, full_name, pairs))
        parts.append("\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = render(SHIPPED.read_text())
    if args.check:
        if not GENERATED.exists() or GENERATED.read_text() != rendered:
            print(f"{GENERATED} is stale; rerun research/e72_wide_gen.py")
            return 1
        print(f"{GENERATED} is in sync with {SHIPPED}")
        return 0
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(rendered)
    print(f"wrote {GENERATED} ({len(rendered)} bytes, {len(ARMS)} arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
