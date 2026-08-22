#!/usr/bin/env python3
"""E124 stage 0.4: reprice the four island arms from the measured census.

Every byte here comes from a tensor header this experiment read at the live
pin (`research/e124_head_census.py`), and every operating-point number comes
from a 512-token trace on this host. Nothing is copied from the assignment
table.

Rule 69 forbids mixing transfer classes inside one price, so the byte class and
the dispatch class are kept in separate columns and only summed at the end,
with the coefficient applied to each named explicitly.

  python3 research/e124_price.py --out research/out/e124-price.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# ---------------------------------------------------------------- measured --
# Tensor bytes, read from the declared head at tree digest 559b24eb... .
# affine-4 group-64 row cost = 5120/2 weight + 80*2 scales + 80*2 biases.
AFFINE4_ROW_BYTES = 5120 // 2 + 80 * 2 + 80 * 2  # 2880
BF16_ROW_BYTES = 5120 * 2  # 10240
Q_OUT, K_OUT, V_OUT = 12_288, 1_024, 1_024
ISLAND_ROWS = 1_024
Q_INDEX_BYTES = 1_024 * 4

# Operating point, research/out/e116x512k0 on this host (Apple M4 Pro, 512
# decode tokens, declared head, shipped schedule). 78 rounds.
TRACE = Path("research/out/e116x512k0/trace.txt")

# Transfer classes, Finding 85 table.
#   draft-path head BYTES     ranked = local x 0.24 .. 0.327
#   draft-path DISPATCH       ranked = local x 0.95 in percent
# Finding 85's usage trap: 0.237/0.24 is the DEPTH-CORRECTED coefficient and
# needs a local gain already restated at ranked draft depth. The gains below
# are raw local-depth gains, so 0.327 is the correct point multiplier and 0.24
# is carried as the conservative floor.
BYTE_TRANSFER = (0.24, 0.327)
DISPATCH_TRANSFER = 0.95

# Corrected per-dispatch marginal cost. Advisor error 62, corrected twice: the
# 9.90 us fitted intercept is struck; the real boundary is about 1.8 us and
# askeladd's packed marginal is 1.05 us.
DISPATCH_US = (1.05, 1.80)

# Ledger 268.18: the byte model over-predicted by a measured realisation ratio
# of 0.675 on E87 arm C. Reported as a discount, not applied to the headline.
BYTE_REALISATION = 0.675


def qkv_bytes(install_q: bool, install_kv: bool) -> tuple[int, dict[str, int]]:
    """Weight bytes one proposal-step `Qwen35Attention.qkv` call streams."""
    parts: dict[str, int] = {}
    if install_kv:
        # Fast branch: affine-4 pack is narrowed to Q, K and V come from the
        # dense BF16 island matrix.
        parts["q_pack_affine4"] = Q_OUT * AFFINE4_ROW_BYTES
        parts["kv_island_dense_bf16"] = (K_OUT + V_OUT) * BF16_ROW_BYTES
    else:
        # No dense KV: the full affine-4 Q+K+V pack runs and K/V rows are read.
        parts["qkv_pack_affine4"] = (Q_OUT + K_OUT + V_OUT) * AFFINE4_ROW_BYTES
    if install_q:
        parts["q_island_bf16"] = ISLAND_ROWS * BF16_ROW_BYTES
        parts["q_island_indices"] = Q_INDEX_BYTES
    return sum(parts.values()), parts


def kv_bytes(install_kv: bool) -> tuple[int, dict[str, int]]:
    """Weight bytes one `appendHistoryKV` flush streams through `kv(_:)`.

    The assignment's byte table omits this path. It runs once per round, not
    once per draft step, and it moves in the SAME direction as the draft-step
    saving: without the KV islands the flush reads a 2,048-row affine-4 pack
    instead of a 2,048-row dense BF16 matrix.
    """
    if install_kv:
        return (K_OUT + V_OUT) * BF16_ROW_BYTES, {"kv_island_dense_bf16": (K_OUT + V_OUT) * BF16_ROW_BYTES}
    return (K_OUT + V_OUT) * AFFINE4_ROW_BYTES, {"kv_pack_affine4": (K_OUT + V_OUT) * AFFINE4_ROW_BYTES}


def qkv_ops(install_q: bool, install_kv: bool) -> list[str]:
    """MLX ops one `qkv` call issues. Views (slice, transpose) are free."""
    ops = []
    ops.append("quantizedMM(q_only)" if install_kv else "quantizedMM(qkv)")
    if install_q:
        ops += ["matmul(q_island)", "putAlong(q_scatter)"]
    if install_kv:
        ops.append("matmul(kv_dense)")
    return ops


ARMS = {
    "all": (True, True),
    "kv": (False, True),
    "q": (True, False),
    "none": (False, False),
}


def operating_point() -> dict:
    rounds = []
    for line in TRACE.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        rounds.append({k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", line)})
    if not rounds:
        raise SystemExit(f"no rounds in {TRACE}")
    return {
        "source": str(TRACE),
        "rounds": len(rounds),
        "mean_draft_len": statistics.mean(r["d"] for r in rounds),
        "mean_accepted": statistics.mean(r["acc"] for r in rounds),
        "acceptance_rate": sum(r["acc"] for r in rounds) / sum(r["d"] for r in rounds),
        "mean_round_us": statistics.mean(r["round_us"] for r in rounds),
        "mean_draft_build_us": statistics.mean(r["draft_build_us"] for r in rounds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/out/e124-price.json")
    args = ap.parse_args()

    op = operating_point()
    d = op["mean_draft_len"]
    round_us = op["mean_round_us"]

    base_qkv, base_qkv_parts = qkv_bytes(True, True)
    base_kv, _ = kv_bytes(True)
    base_ops = qkv_ops(True, True)
    # Effective streaming rate implied by the E87 head-byte law on this host:
    # 1 % of 427,738,112 declared-head bytes per draft step -> 0.0815 % of
    # candidate seconds per token. Deriving the rate rather than assuming peak
    # bandwidth keeps the price anchored to a campaign measurement.
    law_bytes = 0.01 * 427_738_112
    law_local_pct = 0.0815
    law_us_per_round = law_local_pct / 100.0 * round_us
    bytes_per_round_law = law_bytes * d
    gbps = bytes_per_round_law / (law_us_per_round * 1e-6) / 1e9

    report = {
        "operating_point": op,
        "effective_stream_gbps_from_e87_law": gbps,
        "baseline_arm_all": {
            "qkv_bytes_per_draft_step": base_qkv,
            "qkv_parts": base_qkv_parts,
            "kv_flush_bytes_per_round": base_kv,
            "qkv_ops_per_draft_step": base_ops,
        },
        "arms": {},
    }

    print(f"operating point   {op['rounds']} rounds, d={d:.4f}, "
          f"acceptance={op['acceptance_rate']:.6f}, round={round_us:,.0f} us")
    print(f"effective stream  {gbps:.1f} GB/s (derived from the E87 head-byte law)")
    print()
    print("arm   qkv B/step    flush B/rnd   dB/round      ops/step  "
          "d_ops/rnd  local %   ranked %")

    for arm, (iq, ikv) in ARMS.items():
        qb, qparts = qkv_bytes(iq, ikv)
        kb, _ = kv_bytes(ikv)
        ops = qkv_ops(iq, ikv)

        d_bytes_round = (base_qkv - qb) * d + (base_kv - kb)
        d_ops_round = (len(base_ops) - len(ops)) * d

        byte_local = d_bytes_round / (gbps * 1e9) / (round_us * 1e-6) * 100.0
        disp_local = tuple(
            d_ops_round * us / round_us * 100.0 for us in DISPATCH_US)

        ranked_lo = byte_local * BYTE_TRANSFER[0] + disp_local[0] * DISPATCH_TRANSFER
        ranked_hi = byte_local * BYTE_TRANSFER[1] + disp_local[1] * DISPATCH_TRANSFER

        report["arms"][arm] = {
            "installs_q": iq,
            "installs_kv": ikv,
            "qkv_bytes_per_draft_step": qb,
            "qkv_parts": qparts,
            "kv_flush_bytes_per_round": kb,
            "qkv_ops_per_draft_step": ops,
            "delta_bytes_per_round_vs_all": d_bytes_round,
            "delta_ops_per_round_vs_all": d_ops_round,
            "byte_class_local_pct": byte_local,
            "byte_class_ranked_pct": [byte_local * c for c in BYTE_TRANSFER],
            "dispatch_class_local_pct": list(disp_local),
            "dispatch_class_ranked_pct": [
                x * DISPATCH_TRANSFER for x in disp_local],
            "total_local_pct": [byte_local + disp_local[0],
                                byte_local + disp_local[1]],
            "total_ranked_pct": [ranked_lo, ranked_hi],
            "total_ranked_pct_after_byte_realisation_discount": [
                byte_local * BYTE_REALISATION * BYTE_TRANSFER[0]
                + disp_local[0] * DISPATCH_TRANSFER,
                byte_local * BYTE_REALISATION * BYTE_TRANSFER[1]
                + disp_local[1] * DISPATCH_TRANSFER,
            ],
        }
        print(f"{arm:<5} {qb:>11,}  {kb:>12,}  {d_bytes_round:>11,.0f}  "
              f"{len(ops):>8}  {d_ops_round:>9.2f}  "
              f"{byte_local + disp_local[1]:>6.3f}  "
              f"{ranked_lo:>5.3f}-{ranked_hi:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
