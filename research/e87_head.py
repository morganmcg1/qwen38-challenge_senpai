#!/usr/bin/env python3
"""E87: the compact draft-vocabulary heads, loaded exactly as the runtime does.

Two tensors sets matter for the shortlist question:

  exact   the target's own `lm_head`, sliced to the compact draft layout by
          `makeCompactDraftHead` (Qwen35.swift:3494). Affine 4-bit group 64.
          `draftTokenIDWithDeclaredRerank` reranks its 32 shortlisted rows with
          this and nothing else, so `argmax` over ALL its real rows is the
          proposal any lossless shortlist has to return.

  coarse  the declared head's `draft_lm_head.*`. Affine 2-bit group 64 over the
          same 98,336-row layout. It is a retrieval index and never a value.

The compact slicing rule is copied from the Swift, not guessed:
prefix `0 ..< 98_304`, controls `248_044 ..< 248_070`, then
`98_336 - 98_330 = 6` padding rows taken from the front. Only the first 98,330
rows are reachable: both the kernel and the argPartition fallback slice
`0 ..< compactDraftRealCount` before selecting.

Run with the Homebrew interpreter, which is the one that carries MLX:
  /opt/homebrew/bin/python3 research/e87_head.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e82_st import SafeTensors  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WEIGHTS_SHARD = REPO / "weights/model-00003-of-00003.safetensors"
DECLARED_HEAD = Path(
    os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared/model.safetensors")
)

PREFIX_COUNT = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070
REAL_COUNT = PREFIX_COUNT + (CONTROL_END - CONTROL_START)  # 98_330
PADDED_COUNT = 98_336
HIDDEN = 5_120

# Declared-head accounting, from the assignment's byte census.
DECLARED_HEAD_TENSOR_BYTES = 427_738_112
COARSE_ROW_BYTES = 320 * 4 + 80 * 2 + 80 * 2  # 1600 B/row at 2-bit g64
COARSE_STAGE_BYTES = PADDED_COUNT * COARSE_ROW_BYTES  # 157,337,600
# E82 route (c): 1 % of declared-head bytes -> 0.0815 % of candidate s/token.
BYTES_TO_SCORE_PCT = 0.0815
# E82 rung 0: 1 point of pooled acceptance -> 2.21 % of score, and a lost
# argmax costs about 93.5 points of pooled acceptance on the affected draft.
MISS_TO_SCORE_PCT = 206.6


def compact_rows(array: np.ndarray) -> np.ndarray:
    padding = PADDED_COUNT - REAL_COUNT
    return np.concatenate(
        [array[:PREFIX_COUNT], array[CONTROL_START:CONTROL_END], array[:padding]], axis=0
    )


def _bf16(raw_u16: np.ndarray) -> mx.array:
    return mx.array(raw_u16).view(mx.bfloat16)


def load_exact() -> dict[str, mx.array]:
    st = SafeTensors(WEIGHTS_SHARD)
    weight = compact_rows(st.array("language_model.lm_head.weight"))
    scales = compact_rows(st.array("language_model.lm_head.scales"))
    biases = compact_rows(st.array("language_model.lm_head.biases"))
    assert weight.shape == (PADDED_COUNT, 640), weight.shape
    assert scales.shape == (PADDED_COUNT, 80), scales.shape
    return {
        "weight": mx.array(weight),
        "scales": _bf16(scales),
        "biases": _bf16(biases),
        "group_size": 64,
        "bits": 4,
    }


def load_coarse() -> dict[str, mx.array]:
    st = SafeTensors(DECLARED_HEAD)
    weight = st.array("draft_lm_head.weight")
    scales = st.array("draft_lm_head.scales")
    biases = st.array("draft_lm_head.biases")
    assert weight.shape == (PADDED_COUNT, 320), weight.shape
    assert scales.shape == (PADDED_COUNT, 80), scales.shape
    return {
        "weight": mx.array(weight),
        "scales": _bf16(scales),
        "biases": _bf16(biases),
        "group_size": 64,
        "bits": 2,
    }


def scores_all(head: dict[str, mx.array], x: mx.array) -> mx.array:
    """`[B, rows]` logits over every row of a quantized table."""
    return mx.quantized_matmul(
        x,
        head["weight"],
        head["scales"],
        head["biases"],
        transpose=True,
        group_size=head["group_size"],
        bits=head["bits"],
    )


def scores(head: dict[str, mx.array], x: mx.array) -> mx.array:
    """`[B, REAL_COUNT]` logits over the reachable compact rows only."""
    return scores_all(head, x)[:, :REAL_COUNT]


def dequantized(head: dict[str, mx.array]) -> mx.array:
    return mx.dequantize(
        head["weight"],
        head["scales"],
        head["biases"],
        group_size=head["group_size"],
        bits=head["bits"],
    )


def requantize(rows: mx.array, group_size: int, bits: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(rows, group_size=group_size, bits=bits)
    return {
        "weight": weight,
        "scales": scales,
        "biases": biases,
        "group_size": group_size,
        "bits": bits,
    }


def compact_to_vocab(index: mx.array) -> mx.array:
    """The device-side `mapDraftTokenIds` rule, applied offline."""
    return mx.where(index < PREFIX_COUNT, index, index + (CONTROL_START - PREFIX_COUNT))


def vocab_to_compact(token: np.ndarray) -> np.ndarray:
    """Inverse of `compact_to_vocab` for tokens the compact head can emit."""
    return np.where(token < PREFIX_COUNT, token,
                    token - (CONTROL_START - PREFIX_COUNT))


def main() -> None:
    exact = load_exact()
    coarse = load_coarse()
    rows = dequantized(exact)
    mx.eval(rows)
    print(f"exact  weight {exact['weight'].shape} {exact['weight'].dtype}")
    print(f"coarse weight {coarse['weight'].shape} {coarse['weight'].dtype}")
    print(f"dequantized rows {rows.shape} {rows.dtype}")

    # Provenance: is the shipped coarse head exactly MLX's own affine-2 g64
    # quantization of the exact compact rows? If it is, the arm G artifact is
    # the same rule at group 128 and needs no further justification.
    rebuilt = requantize(rows, 64, 2)
    same_w = bool(mx.all(rebuilt["weight"] == coarse["weight"]).item())
    same_s = bool(mx.all(rebuilt["scales"] == coarse["scales"]).item())
    same_b = bool(mx.all(rebuilt["biases"] == coarse["biases"]).item())
    print(f"coarse == mx.quantize(dequant(exact), 64, 2): "
          f"weight={same_w} scales={same_s} biases={same_b}")
    if not same_w:
        agree = float(mx.mean((rebuilt["weight"] == coarse["weight"]).astype(mx.float32)).item())
        print(f"  packed-word agreement {agree:.6f}")


if __name__ == "__main__":
    main()
