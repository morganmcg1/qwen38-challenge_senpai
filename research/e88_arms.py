#!/usr/bin/env python3
"""E88 arm definitions: vectorize the packed-weight load in the crossrow QMV.

Two sites fetch eight contiguous, naturally grouped bytes with four separate
2-byte device loads:

  site A  qmv_fast_crossrow_affine4_g64<T, M>        (M = 2, the pair kernel)
  site B  qmv_fast_crossrow_affine4_g64_wide<T, NA>  (M = 3..9, through `_m`)

Every arm replaces those four scalar loads with one 8-byte load. The loads are
INTEGER loads of packed nibbles: the bit pattern delivered to the arithmetic is
identical however many instructions fetch it, and no floating-point operation,
accumulation order, partial-sum layout or `simd_sum` is touched.

The arms differ only in where the loaded vector is KEPT, because that is what
the register allocator sees:

  w         the tile array itself becomes `ushort4 packed[rows_per_simd]`, so
            the vector never leaves vector form. The two `qdot_affine4_loaded*`
            helpers take the vector by value.
  w_unpack  the tile array stays `uint16_t packed[rows_per_simd][4]` and the
            loaded vector is split into its four components immediately. The
            device traffic is identical to `w`; only the storage form differs.

Arms are STATES, not patches: every arm is rendered from the pinned base text
read out of Git, so no arm can stack on another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
SOURCES = (HEADER, TWIN)

BASE_SHA = "217998560606a32f5d05e413a0419e7bb8322dd6"

# --- anchors ------------------------------------------------------------
#
# The two declarations are textually identical, so each is anchored on the
# line that follows it: only site A's is followed by a blank line.

DECL_TAIL_A = """    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];

"""
DECL_TAIL_B = """    thread float scale_local[rows_per_simd];
    thread float bias_local[rows_per_simd];
    for (int r = 0; r < rows_per_simd; r++) {
"""
DECL_SCALAR = "    thread uint16_t packed[rows_per_simd][4];\n"
DECL_VECTOR = "    thread ushort4 packed[rows_per_simd];\n"

LOAD_A_FROM = """      const device uint8_t* wb =
          reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 2 + simd_lid * bytes_per_lane;
      const device uint16_t* ws =
          reinterpret_cast<const device uint16_t*>(wb);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
"""
LOAD_B_FROM = """      const device uint16_t* ws = reinterpret_cast<const device uint16_t*>(
          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane);
      for (int i = 0; i < 4; i++) {
        packed[r][i] = ws[i];
      }
"""

ADDR_A = """          reinterpret_cast<const device uint8_t*>(w) +
          row * in_vec_size_w + k / 2 + simd_lid * bytes_per_lane"""
ADDR_B = """          reinterpret_cast<const device uint8_t*>(w) + row * in_vec_size_w +
          k / 2 + simd_lid * bytes_per_lane"""

# The two consumers of site A's array. They index `ws[i]` with an unrolled
# constant, so taking the vector by value keeps every extracted nibble and
# every product identical while removing the last reason to address the array.
HELPER_FROM = "    const thread uint16_t* ws,\n"
HELPER_TO = "    const ushort4 ws,\n"

ARMS = ("shipped", "w", "w_unpack")


def _replace_once(text: str, src: str, dst: str, tag: str, want: int = 1) -> str:
    got = text.count(src)
    if got != want:
        raise SystemExit(
            "e88_arms: edit %s matched %d sites, expected %d" % (tag, got, want))
    return text.replace(src, dst)


def _vector_load(addr: str) -> str:
    return ("      packed[r] = *reinterpret_cast<const device ushort4*>(\n"
            "%s);\n" % addr)


def _unpacked_load(addr: str) -> str:
    return ("      const ushort4 wv = *reinterpret_cast<const device ushort4*>(\n"
            "%s);\n"
            "      packed[r][0] = wv.x;\n"
            "      packed[r][1] = wv.y;\n"
            "      packed[r][2] = wv.z;\n"
            "      packed[r][3] = wv.w;\n" % addr)


def apply_arm(text: str, name: str) -> str:
    if name == "shipped":
        return text
    if name not in ARMS:
        raise SystemExit("e88_arms: unknown arm %r" % name)

    if name == "w":
        for tag, tail in (("decl_a", DECL_TAIL_A), ("decl_b", DECL_TAIL_B)):
            text = _replace_once(
                text, DECL_SCALAR + tail, DECL_VECTOR + tail, tag)
        text = _replace_once(text, LOAD_A_FROM, _vector_load(ADDR_A), "load_a")
        text = _replace_once(text, LOAD_B_FROM, _vector_load(ADDR_B), "load_b")
        return _replace_once(text, HELPER_FROM, HELPER_TO, "helper", want=2)

    text = _replace_once(text, LOAD_A_FROM, _unpacked_load(ADDR_A), "load_a")
    return _replace_once(text, LOAD_B_FROM, _unpacked_load(ADDR_B), "load_b")


def base_text(path: str, rev: str = BASE_SHA) -> str:
    return subprocess.run(
        ["git", "show", "%s:%s" % (rev, path)],
        cwd=REPO, capture_output=True, text=True, check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="write the named arm into the worktree")
    ap.add_argument("--rev", default=BASE_SHA)
    args = ap.parse_args()

    report = {"base_sha": args.rev, "arms": {}}
    for arm in ARMS:
        digests = {}
        for path in SOURCES:
            text = apply_arm(base_text(path, args.rev), arm)
            digests[path] = {
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes": len(text.encode()),
            }
            if args.write == arm:
                (REPO / path).write_text(text)
        report["arms"][arm] = digests
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
