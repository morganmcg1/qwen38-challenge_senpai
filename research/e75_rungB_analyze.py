#!/usr/bin/env python3
"""E75 rung B: aggregate the eight-leg crown-vs-ours verify-cost curve.

The session is a balanced mirrored palindrome `ours, crown, crown, ours, ours,
crown, crown, ours`, so each arm has four legs and monotone thermal drift
cancels to first order.

The two tables differ at exactly three dispatch cells (verify widths 5, 6, 9).
The other six measured widths are therefore a *negative control*: an honest
instrument must report no arm effect there. That control is what turns the
changed-cell numbers into a measured null instead of an assumed one.

The script also re-prices the rung C 2x2 prediction with the measured crown
curve, replacing the closed-ladder estimate that was committed before the
session ran.
"""

from __future__ import annotations

import json
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGS = REPO / ".mlxfast-private/e75-legs"
CURVE = REPO / ".mlxfast-private/qmv-curve"

TAGS = [f"e75-rB-a{i}" for i in range(1, 9)]
CHANGED = (5, 6, 9)

# Deterministic round histograms recorded per arm in E68 rung 3, and the
# two-layer round model solved exactly from those two arms in rung C.
HIST = {
    "ship": {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34},
    "pbfit": {4: 5, 5: 42, 6: 5, 8: 7, 9: 26},
}
ROWS = {"ship": 567, "pbfit": 550}
A_PER_ROUND_MS = 4.942
B_PER_ROW_MS = 8.697

# Committed before the session ran, from the closed QMV cost ladder.
PREREGISTERED_CELL_MS = {5: 122.314, 6: 129.065, 9: 186.003}
PREREGISTERED_2X2 = {
    "O-ship": 16.106,
    "O-pbfit": 15.538,
    "C-ship": 17.143,
    "C-pbfit": 17.274,
}


def load():
    per_arm: dict[str, list[dict[int, float]]] = {"ours": [], "crown": []}
    meta = []
    for tag in TAGS:
        leg = json.loads((LEGS / f"{tag}-leg.json").read_text())
        summary = json.loads((CURVE / tag / "summary.json").read_text())
        curve = {
            r["verify_width"]: r["gemm_seconds"] * 1000.0
            for r in summary["round_cost_model"]["rows"]
        }
        per_arm[leg["arm"]].append(curve)
        meta.append((tag, leg))
    return per_arm, meta


def mean_sd(xs):
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def decode_ms(hist, rows, cost):
    gemm = sum(n * cost[w] for w, n in hist.items())
    return gemm + A_PER_ROUND_MS * sum(hist.values()) + B_PER_ROW_MS * rows


def main() -> None:
    per_arm, meta = load()
    widths = sorted(per_arm["ours"][0])

    print("=" * 78)
    print("E75 RUNG B  harness=local  8 legs, balanced mirrored palindrome")
    print("=" * 78)
    print("\nleg provenance")
    for tag, leg in meta:
        p = leg["arm_patch"]
        print(
            f"  {tag}  arm={leg['arm']:<5} cell={p['cell']:<22} na_max={p['na_max']}"
            f"  verified={p.get('crown_bytes_verified', 'n/a')}"
        )

    stats = {}
    for arm, curves in per_arm.items():
        stats[arm] = {w: mean_sd([c[w] for c in curves]) for w in widths}

    print("\nverify cost per width, milliseconds (n=4 legs per arm)")
    print(f"  {'M':>2}  {'ours':>18}  {'crown':>18}  {'crown-ours':>11}  {'':>8}  role")
    for w in widths:
        om, osd = stats["ours"][w]
        cm, csd = stats["crown"][w]
        d = cm - om
        role = "CHANGED CELL" if w in CHANGED else "null (unchanged)"
        print(
            f"  {w:>2}  {om:9.3f} +-{osd:5.3f}  {cm:9.3f} +-{csd:5.3f}"
            f"  {d:+10.3f}  {d / om * 100:+7.2f}%  {role}"
        )

    unchanged = [w for w in widths if w not in CHANGED]
    null_abs = [abs(stats["crown"][w][0] - stats["ours"][w][0]) / stats["ours"][w][0] * 100 for w in unchanged]
    print(f"\nMEASURED SESSION NULL from the {len(unchanged)} unchanged cells")
    print(f"  max |crown-ours| on an unchanged cell : {max(null_abs):.3f} %")
    print(f"  mean                                  : {statistics.fmean(null_abs):.3f} %")
    wide_null = [abs(stats["crown"][w][0] - stats["ours"][w][0]) / stats["ours"][w][0] * 100 for w in (7, 8)]
    print(f"  at the wide unchanged cells 7 and 8   : {max(wide_null):.3f} %")

    print("\nmarginal cost of the step INTO each width, milliseconds")
    print(f"  {'M':>2}  {'ours step':>10}  {'crown step':>11}  {'delta':>9}")
    for w in widths[1:]:
        so = stats["ours"][w][0] - stats["ours"][w - 1][0]
        sc = stats["crown"][w][0] - stats["crown"][w - 1][0]
        mark = "  <== headline pair" if w in (5, 6) else ""
        print(f"  {w:>2}  {so:10.3f}  {sc:11.3f}  {sc - so:+9.3f}{mark}")

    print("\nPRE-REGISTERED vs MEASURED, changed cells")
    print(f"  {'M':>2}  {'predicted':>10}  {'measured':>10}  {'error':>9}  {'':>8}")
    for w in CHANGED:
        pred = PREREGISTERED_CELL_MS[w]
        got = stats["crown"][w][0]
        print(f"  {w:>2}  {pred:10.3f}  {got:10.3f}  {got - pred:+9.3f}  {(got - pred) / pred * 100:+7.2f}%")

    cost = {arm: {w: stats[arm][w][0] for w in widths} for arm in stats}
    cells = {
        "O-ship": decode_ms(HIST["ship"], ROWS["ship"], cost["ours"]),
        "O-pbfit": decode_ms(HIST["pbfit"], ROWS["pbfit"], cost["ours"]),
        "C-ship": decode_ms(HIST["ship"], ROWS["ship"], cost["crown"]),
        "C-pbfit": decode_ms(HIST["pbfit"], ROWS["pbfit"], cost["crown"]),
    }

    print("\nRUNG C 2x2 RE-PRICED WITH THE MEASURED CROWN CURVE (decode seconds)")
    print(f"  {'cell':>8}  {'pre-registered':>15}  {'re-priced':>10}  {'shift':>8}")
    for k, v in cells.items():
        pre = PREREGISTERED_2X2[k]
        print(f"  {k:>8}  {pre:15.3f}  {v / 1000:10.3f}  {(v / 1000 - pre) / pre * 100:+7.2f}%")

    o = (cells["O-pbfit"] - cells["O-ship"]) / cells["O-ship"] * 100
    c = (cells["C-pbfit"] - cells["C-ship"]) / cells["C-ship"] * 100
    t = (cells["C-ship"] - cells["O-ship"]) / cells["O-ship"] * 100
    print("\n  effect of pbfit on OUR table   : "
          f"{o:+.3f} %   (pre-registered -3.525 %, E68 measured -3.500 %)")
    print(f"  effect of pbfit on CROWN table : {c:+.3f} %   (pre-registered +0.77 %)")
    print(f"  effect of table, at ship       : {t:+.3f} %   (pre-registered +6.44 %)")
    print(f"  INTERACTION                    : {c - o:+.3f} pp (pre-registered +4.29 pp)")


if __name__ == "__main__":
    main()
