#!/usr/bin/env python3
"""Price one Qwen 3.8 27B decode round from measured E120 fill-cost cells.

harness=local. Measured on Apple M4 Pro (applegpu_g16s); the ranked runner is M5
(applegpu_g17s), so absolute microseconds do not transfer. The ranked baseline
lives in a separate prebuilt workspace, so every microsecond removed from the
candidate QMV path lowers the ranked candidate seconds per token directly. Never
subtract a local serial-path share from these numbers.

Every row is labelled `measured` or `modelled`. A modelled row uses the fitted
law `gain ~= c(M) * N * k_blocks * M` and the median measured fill cost, and its
baseline uses a fitted effective weight throughput. Pass one or more
`fill_report.json` files to replace modelled rows with real cells.
"""

from __future__ import annotations

import argparse
import json
import statistics
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

# Fallback law for a cell with no measurement, fitted on the rung-2 large shapes.
# c is in microseconds per (output column * k-block * row).
FIT_C = {3: 0.02228e-3, 4: 0.02228e-3, 8: 0.02337e-3}
FALLBACK_FILL_US = 4.85
# Affine 4-bit group-64 costs 32 packed bytes + 2 scale + 2 bias per 64 weights.
BYTES_PER_WEIGHT = 36.0 / 64.0
# Effective weight throughput fitted per width on the rung-2 measured shapes. The
# wide QMV does not saturate DRAM at M=8, so throughput roughly halves from M=4.
EFFECTIVE_GBPS = {3: 185.0, 4: 185.0, 8: 95.0}


def kblocks(k: int) -> int:
    return k // 512


def load_cells(paths: list[Path]) -> dict[tuple[str, int], dict]:
    """Later files win, so a rerun of one shape can override an earlier session."""
    cells: dict[tuple[str, int], dict] = {}
    for path in paths:
        report = json.loads(path.read_text())
        for cell in report["cells"]:
            cells[(cell["shape"], cell["width"])] = dict(cell, source_file=path.name)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fill_reports", type=Path, nargs="*")
    ap.add_argument("--out", type=Path,
                    default=Path("research/out/e120-rung2-fill/round_model.json"))
    args = ap.parse_args()

    measured = load_cells(args.fill_reports)
    fills = [c["fill_us_per_dispatch"] for c in measured.values()]
    fill_median = statistics.median(fills) if fills else FALLBACK_FILL_US

    widths = sorted({w for _, w in measured} or set(FIT_C))
    out = {
        "harness": "local",
        "chip": "Apple M4 Pro",
        "gate_qualified_for_timing": False,
        "fill_us_median_measured": fill_median,
        "fill_reports": [str(p) for p in args.fill_reports],
        "widths": {},
    }
    for m in widths:
        rows = []
        for name, k, n, count in ROUND:
            kb = kblocks(k)
            cell = measured.get((name, m))
            if cell is not None:
                fill = cell["fill_us_per_dispatch"]
                gain = cell["consumer_gain_us_per_matvec"]
                base = cell["a_replica_us"] / cell["layers"]
                src = bsrc = "measured"
            else:
                fill = fill_median
                gain = FIT_C[m] * n * kb * m
                base = k * n * BYTES_PER_WEIGHT / (EFFECTIVE_GBPS[m] * 1e3)
                src = bsrc = "modelled"
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
        modelled_net = sum(r["round_net_us"] for r in rows if r["source"] == "modelled")
        out["widths"][m] = {
            "rows": rows,
            "round_qmv_baseline_us": tb,
            "round_gain_us": tg,
            "round_fill_us": tf,
            "round_net_us": tg - tf,
            "round_net_us_from_modelled_rows": modelled_net,
            "modelled_share_of_net": modelled_net / (tg - tf) if tg != tf else 0.0,
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
        print(f"  net still carried by modelled rows   : {modelled_net:9.1f} us "
              f"= {100.0 * out['widths'][m]['modelled_share_of_net']:5.1f} % of Route B net")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
