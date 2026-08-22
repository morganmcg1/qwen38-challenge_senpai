#!/usr/bin/env python3
"""Price every candidate chunk-sum-table gate over the measured decode round.

Reads research/out/e120-additivity.json (written by research/e120_additivity.py)
and multiplies each measured per-matvec net by its call count in one 257-call
wide-QMV decode round.
"""

import argparse
import json
from pathlib import Path

# Wide affine-4/group-64 QMV calls in one decode round (64 layers:
# 48 Gated DeltaNet + 16 full attention), from research/e120_census.
ROUND_CALLS = {
    "mlp.gate_up": 64,
    "mlp.down": 64,
    "gdn.in_proj": 48,
    "gdn.out_proj": 48,
    "fa.qkv": 16,
    "fa.o_proj": 16,
    "lm_head": 1,
}
GROUPS = {3: [3], 4: [4], 5: [5], 6: [3, 3], 7: [4, 3], 8: [4, 4], 9: [3, 3, 3]}


def gates(c):
    volume = c["n"] * c["k_blocks"] * c["m"]
    return {
        "accept_all": True,
        "volume_100k": volume > 100_000,
        "m4_and_volume_100k": c["m"] >= 4 and volume > 100_000,
        "measured_table": c["net_us"] > 1.0,
        "oracle": c["net_us"] > 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="research/out/e120-additivity.json")
    ap.add_argument("--out", default="research/out/e120-gate-price.json")
    args = ap.parse_args()

    doc = json.loads(Path(args.cells).read_text())
    by_key = {(c["shape"], c["m"]): c for c in doc["cells"]}
    names = list(gates(next(iter(by_key.values()))).keys())

    print("== per-cell net (us/matvec) and gate decisions ==")
    print(
        f"{'shape':<14}{'M':>2} {'calls':>6}{'net_us':>9}{'alt_net':>9}"
        + "".join(f"{n[:11]:>12}" for n in names)
    )
    rows = []
    for m in range(3, 10):
        for shape, calls in ROUND_CALLS.items():
            c = by_key.get((shape, m))
            if not c:
                print(f"{shape:<14}{m:>2} {calls:>6}   MISSING")
                continue
            g = gates(c)
            flag = " *sign-disagree" if (c["net_us"] > 0) != (c["net_alt_us"] > 0) else ""
            print(
                f"{shape:<14}{m:>2} {calls:>6}{c['net_us']:>9.3f}{c['net_alt_us']:>9.3f}"
                + "".join(f"{'take' if g[n] else 'skip':>12}" for n in names)
                + flag
            )
            rows.append((shape, m, calls, c, g))
        print()

    print("== round totals (us saved per decode round, and % of round base) ==")
    hdr = f"{'M':>2} {'groups':<10}{'base_ms':>9}" + "".join(f"{n[:13]:>15}" for n in names)
    print(hdr)
    summary = {}
    for m in range(3, 10):
        sel = [r for r in rows if r[1] == m]
        base = sum(r[2] * r[3]["base_us"] for r in sel)
        line = f"{m:>2} {str(GROUPS[m]):<10}{base / 1000:>9.2f}"
        summary[m] = {"base_us": base, "gates": {}}
        for n in names:
            net = sum(r[2] * r[3]["net_us"] for r in sel if r[4][n])
            summary[m]["gates"][n] = {"net_us": net, "pct": 100 * net / base}
            line += f"{net:>9.0f}/{100 * net / base:>4.2f}%"
        print(line)

    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
