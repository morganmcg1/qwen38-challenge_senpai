"""E148 R-F. Recompute the M=1 decode byte model from the scored linear shapes.

harness=local. Frame: one full target weight pass, affine 4-bit group-64.
Reports the recomputed read volume, the bandwidth each timing frame implies,
and the error factor against the recorded 462.2 GB/s figure.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "e148-rf.json"

# 4 bits of weight per element plus one bf16 scale and one bf16 bias per group
# of 64 elements.
BYTES_PER_ELEMENT = 0.5 + 4.0 / 64.0

# (name, in, out, layer count). Qwen 3.8 27B: hidden 5120, 64 layers = 48 GDN
# plus 16 full attention, vocabulary 248320, MLP intermediate 17408.
SHAPES = [
    ("gdn.in_proj", 5120, 16480, 48),
    ("fa.qkv", 5120, 14336, 16),
    ("mlp.gate_up", 5120, 34816, 64),
    ("lm_head", 5120, 248320, 1),
    ("gdn.out_proj", 6144, 5120, 48),
    ("fa.o_proj", 6144, 5120, 16),
    ("mlp.down", 17408, 5120, 64),
]

RECORDED_IMPLIED_GBPS = 462.2
NOMINAL_GBPS = 273.0
MEASURED_CEILING_GBPS = 265.0

# Official per-prompt candidate MTP seconds per token, ledger 236.3 clean pair
# 0dd455f0 -> 214d92aa, A leg. harness=ranked.
CANDIDATE_SPT = {
    "beagle": 0.01197013, "essays": 0.01103102, "republic": 0.01092017,
    "medicine": 0.01091910, "botany": 0.01083499, "travel": 0.01737916,
    "drama": 0.01974174, "plutarch": 0.03027684,
}
# Mean accepted tokens per round on the depth-loving prompts at cap 7, as
# reported by 7fbb504f's note. Used only to build the round-amortized frame.
ACCEPTED_PER_ROUND = (5.7, 6.2)


def main() -> None:
    rows = []
    total = 0.0
    for name, i, o, count in SHAPES:
        b = i * o * BYTES_PER_ELEMENT * count
        total += b
        rows.append({"tensor": name, "in": i, "out": o, "layers": count,
                     "bytes": b, "gb": round(b / 1e9, 4)})

    fastest = min(CANDIDATE_SPT.values())
    slowest = max(CANDIDATE_SPT.values())
    per_token_hi = total / fastest / 1e9
    per_token_lo = total / slowest / 1e9
    round_hi = total / (fastest * ACCEPTED_PER_ROUND[0]) / 1e9
    round_lo = total / (slowest * ACCEPTED_PER_ROUND[1]) / 1e9

    out = {
        "harness": "local",
        "frame": "one full target weight pass, affine 4-bit group-64",
        "bytes_per_element": BYTES_PER_ELEMENT,
        "tensors": rows,
        "e148_m1_bytes_per_full_pass": total,
        "e148_m1_gb_per_full_pass": round(total / 1e9, 4),
        "e148_m1_implied_gbps_recomputed": round(per_token_hi, 1),
        "per_token_frame_gbps_range": [round(per_token_lo, 1),
                                       round(per_token_hi, 1)],
        "round_amortized_frame_gbps_range": [round(round_lo, 1),
                                             round(round_hi, 1)],
        "recorded_implied_gbps": RECORDED_IMPLIED_GBPS,
        "e148_byte_model_error_factor": round(
            per_token_hi / RECORDED_IMPLIED_GBPS, 3),
        "nominal_gbps": NOMINAL_GBPS,
        "e148_measured_read_gbps": MEASURED_CEILING_GBPS,
        "verdict": (
            "The per-token M=1 model is falsified by its own arithmetic. One "
            "full pass is 14.4123 GB. Across the eight official candidate "
            "legs the per-token frame demands 476.0 to 1330.2 GB/s against a "
            "265 GB/s measured ceiling, so every prompt is impossible and the "
            "fastest is impossible by 5.0x. The round-amortized frame is the "
            "only self-consistent one: at 5.7 to 6.2 accepted tokens per "
            "round the same pass reads at 76.8 to 233.4 GB/s, which stays "
            "under the ceiling on every prompt. Every byte-model price in the "
            "campaign must therefore state tokens per round; a per-token form "
            "silently inflates the available headroom by about 6x."),
        "caveat": (
            "The recorded 462.2 GB/s and this recomputation differ by 2.878x "
            "at the fastest prompt but by only 1.030x at the slowest "
            "(plutarch, 476.0 GB/s). The recorded figure is therefore "
            "consistent with the same 14.4123 GB volume evaluated on the "
            "slowest single prompt in the per-token frame. It is a "
            "prompt-selection and frame difference, not a byte-count error. "
            "Publish the prompt and the frame beside any future figure."),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(f'{r["tensor"]:<14} {r["gb"]:8.4f} GB')
    print(f'total per full pass         {total / 1e9:8.4f} GB')
    print(f'per-token frame   {per_token_lo:7.1f} - {per_token_hi:7.1f} GB/s')
    print(f'round frame       {round_lo:7.1f} - {round_hi:7.1f} GB/s')
    print(f'error factor vs 462.2       {per_token_hi / 462.2:.3f}')


if __name__ == "__main__":
    main()
