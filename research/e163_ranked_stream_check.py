#!/usr/bin/env python3
"""E163 F5 item 3: is the ranked first weight stream at 295 GB/s real?

The advisor asked one question: does the `14.418 GB` per-verify weight stream
include the `248,320 x 5,120` vocabulary readout and the group-64 scales, and
what is the number both ways?

This script answers it from the safetensors headers of the pinned target
checkpoint, splits the streamed set by storage kind, and then divides each
variant into the ranked essays round so the advisor can read the implied
ranked rate off a table instead of recomputing it.

It also records the one enforcing fact that changes the F5 section 3 argument:
`.github/workflows/qwen-mtp-ranked-benchmark.yml` names the ranked box as
`m5-max-128gb-3`, and its Metal library cache key is `mlx-metallib-m5-max-v2`.
The ranked host is an M5 **Max**, not a base M5, so `590 GB/s` is not by itself
an implausible DRAM figure for that part. FINDING 375 does not need that
argument: the disproportion test below is stronger and is independent of any
unpublished M5 Max datasheet number.

Every byte count is exact. Every ranked timing input is quoted from the
advisor's `harness=ranked` fits and is labelled as such.

    python3 research/e163_ranked_stream_check.py
"""
from __future__ import annotations

import json
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "research" / "e163-artifacts"
TARGET = ROOT / "reference_weights" / "Qwen3.8-27B-4bit"
HEAD = ROOT / "reference_weights" / "Qwen3.6-27B-MTP-4bit"

DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32, "U64": 64, "I64": 64, "F64": 64,
}

COMPACT_DRAFT_PADDED_ROWS = 98_336
LM_HEAD_ROWS = 248_320

# harness=ranked, quoted from the advisor. FINDING 352 on 5a9f130a is
# R = 16.158 ms + 5.3351 ms x rows; F5 section 2 quotes the essays round as
# 48.93 ms and the essays mean verify width as 6.087 rows.
RANKED_INTERCEPT_MS = 16.158
RANKED_PER_ROW_MS = 5.3351
RANKED_ESSAYS_ROUND_MS = 48.93
RANKED_ESSAYS_WIDTH = 6.087
# F5 section 2: one pass at width 6 is about +0.20 ms/round SLOWER on the
# ranked M5, so removing a stream is worth about -0.20 ms there.
RANKED_STREAM_SAVING_MS = -0.20

# harness=local, this session, gated, 512 tokens, six-leg palindrome.
LOCAL_W6_ROUND_MS = 128.001
LOCAL_STREAM_SAVING_MS = 28.605
LOCAL_MEASURED_PEAK_GB_S = 255.485

RANKED_BOX = "m5-max-128gb-3"


def _header(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n).decode("utf-8"))


def walk(root: Path):
    for name in sorted(os.listdir(root)):
        if not name.endswith(".safetensors"):
            continue
        for key, meta in _header(root / name).items():
            if key == "__metadata__":
                continue
            nelem = 1
            for d in meta["shape"]:
                nelem *= d
            yield key, meta["dtype"], nelem * DTYPE_BITS[meta["dtype"]] // 8


def kind_of(key: str, dtype: str) -> str:
    if key.endswith(".scales") or key.endswith(".biases"):
        return "scales_biases"
    if dtype == "U32":
        return "packed_4bit"
    return "dense_bf16"


def main() -> int:
    split: dict[tuple[str, str], int] = defaultdict(int)
    for key, dtype, nbytes in walk(TARGET):
        role = ("embed_gather" if "embed_tokens" in key
                else "vocab_readout" if "lm_head" in key
                else "backbone" if re.search(r"layers\.\d+\.", key)
                else "top_level")
        split[(role, kind_of(key, dtype))] += nbytes

    def total(roles, kinds) -> int:
        return sum(v for (r, k), v in split.items() if r in roles and k in kinds)

    all_kinds = {"packed_4bit", "scales_biases", "dense_bf16"}
    streamed = {"backbone", "vocab_readout", "top_level"}
    no_vocab = {"backbone", "top_level"}

    variants = [
        ("with vocab readout, with scales/biases",
         True, True, total(streamed, all_kinds)),
        ("with vocab readout, packed 4-bit only",
         True, False, total(streamed, {"packed_4bit"})),
        ("no vocab readout, with scales/biases",
         False, True, total(no_vocab, all_kinds)),
        ("no vocab readout, packed 4-bit only",
         False, False, total(no_vocab, {"packed_4bit"})),
    ]

    head_bytes = sum(n for _, _, n in walk(HEAD))
    compact = round(total({"vocab_readout"}, all_kinds)
                    * COMPACT_DRAFT_PADDED_ROWS / LM_HEAD_ROWS)
    per_draft = head_bytes + compact
    drafts = RANKED_ESSAYS_WIDTH - 1.0

    rows = []
    for label, has_vocab, has_scales, nbytes in variants:
        rows.append({
            "variant": label,
            "includes_vocab_readout": has_vocab,
            "includes_group64_scales": has_scales,
            "bytes": nbytes,
            "gb": nbytes / 1e9,
            "ranked_one_stream_gb_per_s": nbytes / 1e9 / (RANKED_ESSAYS_ROUND_MS / 1e3),
            "local_one_stream_percent_of_measured_peak":
                100 * (nbytes / 1e9) / (LOCAL_W6_ROUND_MS / 1e3)
                / LOCAL_MEASURED_PEAK_GB_S,
        })

    full = rows[0]["bytes"]
    ranked_from_352 = RANKED_INTERCEPT_MS + RANKED_PER_ROW_MS * RANKED_ESSAYS_WIDTH
    shipped_two_stream = 2 * full
    with_heads = shipped_two_stream + drafts * per_draft

    # The disproportion test. If the second stream cost the ranked host the same
    # FRACTION of its round that it costs mine, it would cost this much there.
    local_share = LOCAL_STREAM_SAVING_MS / LOCAL_W6_ROUND_MS
    expected_ranked_ms = local_share * RANKED_ESSAYS_ROUND_MS
    # And on my own host the second stream already outruns my own DRAM.
    local_effective_gb_s = (full / 1e9) / (LOCAL_STREAM_SAVING_MS / 1e3)

    out = {
        "harness": "local bytes, ranked timings quoted",
        "official_or_ranked_score": False,
        "ranked_box": RANKED_BOX,
        "ranked_box_source":
            ".github/workflows/qwen-mtp-ranked-benchmark.yml:196 and :1370",
        "target_stream_variants": rows,
        "answer_includes_vocab_readout": True,
        "answer_includes_group64_scales": True,
        "target_stream_bytes": full,
        "vocab_readout_bytes": total({"vocab_readout"}, all_kinds),
        "scales_biases_bytes_in_stream": total(streamed, {"scales_biases"}),
        "packed_bytes_in_stream": total(streamed, {"packed_4bit"}),
        "dense_bf16_bytes_in_stream": total(streamed, {"dense_bf16"}),
        "head_bytes_per_draft_step": per_draft,
        "ranked_essays_round_ms": RANKED_ESSAYS_ROUND_MS,
        "ranked_essays_round_ms_from_finding_352": ranked_from_352,
        "ranked_essays_width": RANKED_ESSAYS_WIDTH,
        "ranked_one_stream_gb_per_s": full / 1e9 / (RANKED_ESSAYS_ROUND_MS / 1e3),
        "ranked_shipped_two_stream_gb_per_s":
            shipped_two_stream / 1e9 / (RANKED_ESSAYS_ROUND_MS / 1e3),
        "ranked_shipped_two_stream_with_heads_gb_per_s":
            with_heads / 1e9 / (RANKED_ESSAYS_ROUND_MS / 1e3),
        "local_second_stream_effective_gb_per_s": local_effective_gb_s,
        "local_second_stream_over_measured_peak":
            local_effective_gb_s / LOCAL_MEASURED_PEAK_GB_S,
        "local_stream_share_of_round": local_share,
        "ranked_ms_predicted_by_equal_share": expected_ranked_ms,
        "ranked_ms_measured": RANKED_STREAM_SAVING_MS,
        "disproportion_factor":
            RANKED_STREAM_SAVING_MS / expected_ranked_ms,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "e163_ranked_stream.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")

    lines = []
    add = lines.append
    add("=" * 78)
    add("F5 ITEM 3 - DOES 14.418 GB INCLUDE THE VOCABULARY READOUT AND THE")
    add("            GROUP-64 SCALES?   Answer: YES to both.")
    add("=" * 78)
    add("  Source: safetensors headers of the pinned target checkpoint.")
    add("  The streamed set is the whole checkpoint minus embed_tokens, which is")
    add("  GATHERED (M rows of 5,120) and never streamed. lm_head is not tied")
    add("  (tie_word_embeddings=false) and IS streamed, because the verify pass")
    add("  needs exact top-two evidence over all 248,320 vocabulary rows.")
    add("")
    add(f"  {'variant':<40}{'bytes':>16}{'GB':>9}{'ranked GB/s':>13}")
    for r in rows:
        add(f"  {r['variant']:<40}{r['bytes']:>16,}{r['gb']:>9.3f}"
            f"{r['ranked_one_stream_gb_per_s']:>13.1f}")
    add("")
    add("  storage kinds inside the streamed set")
    add(f"    packed 4-bit    {out['packed_bytes_in_stream']:>16,}"
        f"{out['packed_bytes_in_stream']/1e9:>9.3f} GB")
    add(f"    scales+biases   {out['scales_biases_bytes_in_stream']:>16,}"
        f"{out['scales_biases_bytes_in_stream']/1e9:>9.3f} GB")
    add(f"    dense bf16      {out['dense_bf16_bytes_in_stream']:>16,}"
        f"{out['dense_bf16_bytes_in_stream']/1e9:>9.3f} GB")
    add("")
    add(f"  ranked essays round {RANKED_ESSAYS_ROUND_MS:.2f} ms at width "
        f"{RANKED_ESSAYS_WIDTH:.3f}   (FINDING 352 predicts "
        f"{ranked_from_352:.2f} ms)")
    add(f"  ONE stream                        "
        f"{out['ranked_one_stream_gb_per_s']:>7.1f} GB/s   <- the 295 figure")
    add(f"  shipped TWO streams, weights only "
        f"{out['ranked_shipped_two_stream_gb_per_s']:>7.1f} GB/s")
    add(f"  shipped TWO streams, plus {drafts:.3f} head passes "
        f"{out['ranked_shipped_two_stream_with_heads_gb_per_s']:>7.1f} GB/s")
    add("")
    add("=" * 78)
    add("IS 295 GB/s REAL?  Yes, and it is not even stressed on that box.")
    add("=" * 78)
    add(f"  The ranked box is {RANKED_BOX} "
        f"({out['ranked_box_source']}).")
    add("  That is an M5 MAX with 128 GB, not a base M5. Its predecessor part")
    add("  already reaches several hundred GB/s, so neither 295 GB/s nor")
    add("  590 GB/s is self-evidently impossible there. The F5 section 3")
    add("  implausibility argument therefore does not carry, but FINDING 375")
    add("  does not need it. Use the disproportion test instead:")
    add("")
    add(f"  local  second stream = {LOCAL_STREAM_SAVING_MS:.3f} ms of a "
        f"{LOCAL_W6_ROUND_MS:.3f} ms round = {100*local_share:.2f} % of it")
    add(f"  equal-share prediction for the ranked round      "
        f"{expected_ranked_ms:>7.2f} ms")
    add(f"  measured ranked effect                           "
        f"{RANKED_STREAM_SAVING_MS:>7.2f} ms")
    add(f"  disproportion                                    "
        f"{out['disproportion_factor']:>7.3f} x")
    add("")
    add("  No uniform bandwidth scaling can turn 22 % of a round into 0 %.")
    add("  A faster bus shrinks the stream and the rest of the round together.")
    add("  Only a change in WHERE the second stream is served can remove it,")
    add("  so FINDING 375's cache-absorption reading survives on stronger")
    add("  evidence than the 590 GB/s figure.")
    add("")
    add("  The same effect is already visible on my own host, at half strength:")
    add(f"  a second {full/1e9:.3f} GB stream costs {LOCAL_STREAM_SAVING_MS:.3f} ms, "
        f"an effective {local_effective_gb_s:.1f} GB/s,")
    add(f"  which is {out['local_second_stream_over_measured_peak']:.2f} x my "
        f"MEASURED {LOCAL_MEASURED_PEAK_GB_S:.1f} GB/s DRAM peak. Roughly half")
    add("  of the second pass is already cache-served on an M4 Pro. The M5 Max")
    add("  finishes the job.")
    text = "\n".join(lines) + "\n"
    (ARTIFACTS / "e163_ranked_stream.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
