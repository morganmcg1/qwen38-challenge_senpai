#!/usr/bin/env python3
"""Price one Qwen 3.8 27B decode round from the E120 rung-2 fill measurements.

harness=local. Measured on Apple M4 Pro (applegpu_g16s); the ranked runner is M5
(applegpu_g17s), so absolute microseconds do not transfer. The ranked baseline
lives in a separate prebuilt workspace, so every microsecond removed from the
candidate QMV path lowers the ranked candidate seconds per token directly. Never
subtract a local serial-path share from these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

# Every activation tensor that feeds exactly one wide affine-4/group-64 QMV.
# 64 layers = 48 Gated DeltaNet + 16 full attention; 4 tensors per layer + lm_head.
ROUND = [
    ("mlp.gate_up", 5120, 34816, 64),
    ("mlp.down", 17408, 5120, 64),
    ("gdn.in_proj", 5120, 16480, 48),
    ("gdn.out_proj", 6144, 5120, 48),
    ("fa.qkv", 5120, 14336, 16),
    ("fa.o_proj", 6144, 5120, 16),
    ("lm_head", 5120, 248320, 1),
]

MEASURED = {  # (shape, M) -> (fill_us, gain_us)
    ("mlp.gate_up", 4): (4.945, 30.248),
    ("mlp.gate_up", 8): (5.781, 65.078),
    ("mlp.down", 4): (4.918, 15.892),
    ("mlp.down", 8): (4.775, 33.430),
}

# gain_us ~= c(M) * N * k_blocks * M, fitted on the two large measured shapes.
# c is in microseconds per (output column * k-block * row).
FIT_C = {4: 0.02228e-3, 8: 0.02337e-3}
FILL_US = 4.85  # median measured marginal cost of one in-stream fill dispatch

# Measured replica-arm QMV latency, microseconds, from the same session.
BASELINE = {("mlp.gate_up", 4): 497.4, ("mlp.gate_up", 8): 953.0,
            ("mlp.down", 4): 298.1, ("mlp.down", 8): 597.6}
# Affine 4-bit group-64 costs 32 packed bytes + 2 scale + 2 bias per 64 weights.
BYTES_PER_WEIGHT = 36.0 / 64.0
# Effective weight throughput fitted per width on the two measured shapes. The
# wide QMV does not saturate DRAM at M=8, so throughput roughly halves from M=4.
EFFECTIVE_GBPS = {4: 185.0, 8: 95.0}


def kblocks(k: int) -> int:
    return k // 512


def baseline_us(name: str, k: int, n: int, m: int) -> tuple[float, str]:
    if (name, m) in BASELINE:
        return BASELINE[(name, m)], "measured"
    return k * n * BYTES_PER_WEIGHT / (EFFECTIVE_GBPS[m] * 1e3), "modelled"


def main() -> None:
    out = {"harness": "local", "chip": "Apple M4 Pro", "fill_us": FILL_US, "widths": {}}
    for m, c in FIT_C.items():
        rows = []
        for name, k, n, count in ROUND:
            kb = kblocks(k)
            key = (name, m)
            if key in MEASURED:
                fill, gain = MEASURED[key]
                src = "measured"
            else:
                fill, gain = FILL_US, c * n * kb * m
                src = "modelled"
            base, bsrc = baseline_us(name, k, n, m)
            rows.append(
                {
                    "tensor": name,
                    "k": k,
                    "k_blocks": kb,
                    "n": n,
                    "count": count,
                    "source": src,
                    "baseline_source": bsrc,
                    "baseline_us": base,
                    "fill_us": fill,
                    "gain_us": gain,
                    "net_us": gain - fill,
                    "round_baseline_us": base * count,
                    "round_gain_us": gain * count,
                    "round_fill_us": fill * count,
                    "round_net_us": (gain - fill) * count,
                }
            )
        tg = sum(r["round_gain_us"] for r in rows)
        tf = sum(r["round_fill_us"] for r in rows)
        tb = sum(r["round_baseline_us"] for r in rows)
        out["widths"][m] = {
            "rows": rows,
            "round_qmv_baseline_us": tb,
            "round_gain_us": tg,
            "round_fill_us": tf,
            "round_net_us": tg - tf,
            "route_b_pct_of_qmv": 100.0 * (tg - tf) / tb,
            "route_c_pct_of_qmv": 100.0 * tg / tb,
        }

        print(f"\n=== M = {m} rows per QMV (harness=local, M4 Pro) ===")
        hdr = (f"{'tensor':<14}{'K':>7}{'N':>8}{'kb':>4}{'x':>4}{'src':>10}"
               f"{'base_us':>9}{'gain_us':>9}{'fill_us':>9}{'net_us':>9}{'round_net':>11}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['tensor']:<14}{r['k']:>7}{r['n']:>8}{r['k_blocks']:>4}{r['count']:>4}"
                f"{r['source']:>10}{r['baseline_us']:>9.1f}{r['gain_us']:>9.2f}"
                f"{r['fill_us']:>9.2f}{r['net_us']:>9.2f}{r['round_net_us']:>11.1f}"
            )
        print(f"{'TOTAL':<14}{'':>7}{'':>8}{'':>4}{257:>4}{'':>10}"
              f"{tb:>9.0f}{tg:>9.1f}{tf:>9.1f}{tg - tf:>9.1f}{tg - tf:>11.1f}")
        print(f"  wide-QMV share of round baseline     : {tb / 1000:9.2f} ms")
        print(f"  Route B (standalone fills) round net : {tg - tf:9.1f} us "
              f"= {100.0 * (tg - tf) / tb:5.2f} % of wide-QMV time")
        print(f"  Route C (fused into producer) ceiling: {tg:9.1f} us "
              f"= {100.0 * tg / tb:5.2f} % (+{tf:.1f} us recovered fill overhead)")

    Path("research/out/e120-rung2-fill/round_model.json").write_text(
        json.dumps(out, indent=2))
    print("\nwrote research/out/e120-rung2-fill/round_model.json")


if __name__ == "__main__":
    main()
