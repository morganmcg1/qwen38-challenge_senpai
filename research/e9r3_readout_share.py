"""E9 r3 / A3: readout-share arithmetic on the promoted-frontier base.

The compact draft readout is a row slice of the BACKBONE `lmHead`, which is
affine 4-bit group-64 on both the old and the new base, with identical row
constants. So the bytes a bit-width lever would attack are unchanged by the
rebase.

The head the MTP block runs differs by *measurement path*, not by base: the
local wrapper always loads the pinned bf16 head, while the trusted parent
resolves the manifest's declared 4-bit/g64 head. That changes the denominator
of the readout's share of a head step, so it changes how large a readout win
can look. Both are priced here from exact byte counts.
"""

ROWS_PADDED = 98_336  # compactDraftPaddedCount
COLS = 5_120  # hiddenSize
GROUP = 64
GROUPS = COLS // GROUP

# Byte-verified artifact sizes.
PINNED_BF16_HEAD_BYTES = 849_400_347  # 15 BF16 tensors, no draft_lm_head
DECLARED_Q4_HEAD_BYTES = 238_934_129  # 31 tensors = 8 Linear x 3 + 7 norms
# The declared head was ALREADY 4-bit at the old base bc5e15fd's parent 8970d775
# (lowskillcoding/qwen38-mtp-head-4bit-g64, 238_934_093 B). The new base only
# swaps the source repo. So bf16-vs-4bit is LOCAL-vs-RANKED, not old-vs-new:
# benchmark-qwen-mtp.sh always passes the pinned bf16 cache dir, while the
# trusted parent resolves the manifest and attaches the declared 4-bit head.
STREAM_PEAK_BPS = 227_128_791_836.97
KERNEL_SPEEDUP_3BIT = 0.242  # measured qmv sweep, M=1


def affine_bytes(bits: int) -> tuple[int, int, int, int]:
    weight = ROWS_PADDED * COLS * bits // 8
    scales = ROWS_PADDED * GROUPS * 2
    biases = ROWS_PADDED * GROUPS * 2
    return weight, scales, biases, weight + scales + biases


def main() -> None:
    print("compact draft readout (slice of BACKBONE lmHead) -- identical on both bases")
    for bits in (4, 3, 2):
        w, s, z, t = affine_bytes(bits)
        print(
            f"  {bits}-bit: weight {w / 1e6:9.3f} MB  scales {s / 1e6:6.3f} MB"
            f"  biases {z / 1e6:6.3f} MB  total {t / 1e6:8.3f} MB"
        )

    t4 = affine_bytes(4)[3]
    t3 = affine_bytes(3)[3]
    print(
        f"\n  3-bit saves {(t4 - t3) / 1e6:.3f} MB"
        f" = {100 * (t4 - t3) / t4:.2f}% of readout bytes"
    )
    print(
        f"  at STREAM peak {STREAM_PEAK_BPS / 1e9:.2f} GB/s:"
        f" 4-bit {1e3 * t4 / STREAM_PEAK_BPS:.4f} ms,"
        f" 3-bit {1e3 * t3 / STREAM_PEAK_BPS:.4f} ms,"
        f" saving {1e3 * (t4 - t3) / STREAM_PEAK_BPS:.4f} ms"
    )

    print("\nMTP head hidden-forward weights (what the manifest change actually altered)")
    print(f"  pinned bf16 head : {PINNED_BF16_HEAD_BYTES / 1e6:9.3f} MB")
    print(f"  declared 4-bit   : {DECLARED_Q4_HEAD_BYTES / 1e6:9.3f} MB")
    print(f"  ratio            : {PINNED_BF16_HEAD_BYTES / DECLARED_Q4_HEAD_BYTES:.3f}x smaller")

    print("\nreadout share of ONE head step (readout + head hidden forward), bytes model")
    for name, head in (
        ("pinned bf16 head (LOCAL path)", PINNED_BF16_HEAD_BYTES),
        ("declared 4-bit head (RANKED path)", DECLARED_Q4_HEAD_BYTES),
    ):
        share = t4 / (head + t4)
        print(
            f"  {name:22s}: {100 * share:5.2f}%"
            f"  -> a {100 * KERNEL_SPEEDUP_3BIT:.1f}%-faster readout caps"
            f" the head step at -{100 * KERNEL_SPEEDUP_3BIT * share:.2f}%"
        )


if __name__ == "__main__":
    main()
