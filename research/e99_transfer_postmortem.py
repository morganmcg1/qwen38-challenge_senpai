#!/usr/bin/env python3
"""E99 post-mortem: why the margin gate lost on the official M5 runner.

Official submission 87b654b2 scored 3.12600524 and was rejected. Our own
promoted row f04b102 scored 3.32824629, so the gate cost 0.20224 of official
score while the local ranked-curve figure predicted a 3.222 percent gain.

The local `ranked_us_per_token` figure is not an M5 measurement. It prices the
locally recorded round sequence through a two-line MODEL of the M5 round cost,
`e99_oracle.ranked_round_us`, which has a step at the M = 4 to M = 5 group
boundary. The margin gate clamps a firing round to width 4 or less, so its
whole value is avoiding that one crossing. This script measures how much of the
predicted gain depends on the assumed size of that step.

Three outputs:

  empirical    the local round cost by width, taken from the round traces,
               against the LOCAL_ROUND_US table the oracle uses.
  sensitivity  the predicted gain when the ranked step at M = 4 to 5 is
               resized, holding the recorded round sequences fixed.
  breakeven    the step size at which the gate stops paying.

usage:
  python3 research/e99_transfer_postmortem.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e99_oracle import (  # noqa: E402
    LOCAL_ROUND_US, RANKED_A1, RANKED_A2, RANKED_C1, RANKED_C2, load_leg)

ART = pathlib.Path(__file__).resolve().parent / "e99-artifacts"
OUT = pathlib.Path(__file__).resolve().parent / "out"

OFF_LEGS = ["e99r6n1o", "e99r6n4o", "e99r6n5o", "e99r6n8o"]
ON_LEGS = ["e99r6n2s", "e99r6n3s", "e99r6n6s", "e99r6n7s"]

OFFICIAL_BASELINE = 3.32824628683457   # f04b102, promoted
OFFICIAL_CANDIDATE = 3.12600524008429  # 87b654b, rejected


def base_ranked(width: int) -> float:
    if width <= 4:
        return RANKED_A1 + RANKED_C1 * width
    return RANKED_A2 + RANKED_C2 * width


def curve_with_step(step_us: float):
    """Ranked curve whose M=4 to M=5 jump is resized to `step_us`.

    The two within-group slopes are held fixed. Only the offset of the upper
    group moves, so widths 5..8 shift together.
    """
    shift = step_us - (base_ranked(5) - base_ranked(4))

    def curve(width: int) -> float:
        value = base_ranked(width)
        return value + shift if width >= 5 else value

    return curve


def price(rounds, curve) -> float:
    cost = sum(curve(r.depth + 1) for r in rounds)
    tokens = sum(r.accepted + 1 for r in rounds)
    return cost / tokens


def main() -> None:
    off = [r for tag in OFF_LEGS for r in load_leg(tag, OUT)[1]]
    on = [r for tag in ON_LEGS for r in load_leg(tag, OUT)[1]]

    print("=" * 78)
    print("EMPIRICAL LOCAL ROUND COST BY WIDTH, FROM THE RECORDED TRACES")
    print("=" * 78)
    by_width: dict[int, list[float]] = {}
    for r in off + on:
        by_width.setdefault(r.depth + 1, []).append(r.round_us)
    print(" M      n   median us   oracle table    diff %   incr median")
    prev = None
    for m in sorted(by_width):
        med = statistics.median(by_width[m])
        table = LOCAL_ROUND_US[m]
        incr = "" if prev is None else f"{med - prev:11.1f}"
        print(f"{m:2d}  {len(by_width[m]):5d}  {med:10.1f}  {table:13.1f}"
              f"  {(med - table) / table * 100:+7.2f}  {incr}")
        prev = med

    print()
    print("=" * 78)
    print("SENSITIVITY OF THE PREDICTED GAIN TO THE ASSUMED M5 STEP AT M=4->5")
    print("=" * 78)
    assumed = base_ranked(5) - base_ranked(4)
    local_step = LOCAL_ROUND_US[5] - LOCAL_ROUND_US[4]
    print(f"assumed M5 step (e99_oracle two-line fit) : {assumed:9.1f} us")
    print(f"measured local M4 Pro step               : {local_step:9.1f} us")
    print()
    print("  step us   ratio to assumed   off us/tok   on us/tok    gain %")
    rows = []
    for step in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7233, 8000,
                 9946.3, 12000, 15000, 20000, 30000, 39865.7]:
        curve = curve_with_step(step)
        a, b = price(off, curve), price(on, curve)
        gain = (a - b) / a * 100
        rows.append((step, gain))
        mark = "  <- oracle assumption" if abs(step - assumed) < 1 else (
            "  <- local measured step" if abs(step - local_step) < 1 else "")
        print(f"{step:9.1f}   {step / assumed:16.2f}   {a:10.1f}  {b:10.1f}"
              f"  {gain:+8.3f}{mark}")

    lo = None
    for (s0, g0), (s1, g1) in zip(rows, rows[1:]):
        if g0 <= 0 < g1 or g0 < 0 <= g1:
            lo = s0 + (s1 - s0) * (0 - g0) / (g1 - g0)
            break
    print()
    if lo is not None:
        print(f"BREAK-EVEN: the gate pays only when the real M5 step exceeds"
              f" {lo:.0f} us.")
        print(f"            The oracle assumed {assumed:.0f} us"
              f" ({assumed / lo:.2f}x the break-even).")
    else:
        print("BREAK-EVEN: not crossed inside the swept range.")

    print()
    print("=" * 78)
    print("GAIN AGAINST FIRING SHARE: THE THRESHOLD IS CALIBRATED TO ONE")
    print("MARGIN DISTRIBUTION, AND COLLAPSES WHEN THE GATE OVERFIRES")
    print("=" * 78)
    tmap = json.loads((ART / "threshold-map.json").read_text())
    cap8 = sorted(
        (c for c in tmap["cells"]
         if c["cap"] == 8 and c["gate_depth"] == 3),
        key=lambda c: c["fired_share"])
    print("  threshold   fired share   ranked us/tok    gain %")
    for c in cap8:
        print(f"{c['threshold']:11g}   {c['fired_share'] * 100:10.1f} %"
              f"   {c['ranked_us']:13.1f}  {c['gain_pct']:+8.3f}")
    always = next(c for c in cap8 if c["fired_share"] >= 0.999)
    official_pct = (OFFICIAL_CANDIDATE - OFFICIAL_BASELINE) \
        / OFFICIAL_BASELINE * 100
    print()
    print(f"always-fire regime (t=1000) local gain : {always['gain_pct']:+.3f} %")
    print(f"official score change against f04b102  : {official_pct:+.3f} %")
    print("The official outcome sits at the local always-fire value. A gate")
    print("calibrated to one prompt's margin distribution overfires when the")
    print("distribution shifts, and then it only removes accepted tokens.")

    print()
    print("=" * 78)
    print("LOCAL MARGIN DISTRIBUTION THAT SET THE THRESHOLD")
    print("=" * 78)
    margins = sorted(r.margin for r in off)
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        idx = min(int(q * len(margins)), len(margins) - 1)
        print(f"  q{q:<5.2f} margin {margins[idx]:8.4f}")
    fired = sum(1 for m in margins if m <= 9.4375) / len(margins)
    print(f"  shipped threshold 9.4375 sits at quantile {fired:.3f}")
    print("  A quantile-relative threshold would hold this firing share when")
    print("  the margin distribution moves. An absolute constant does not.")

    print()
    print("=" * 78)
    print("OFFICIAL OUTCOME")
    print("=" * 78)
    delta = OFFICIAL_CANDIDATE - OFFICIAL_BASELINE
    print(f"promoted baseline f04b102 : {OFFICIAL_BASELINE:.8f}")
    print(f"candidate       87b654b   : {OFFICIAL_CANDIDATE:.8f}")
    print(f"delta                     : {delta:+.8f}"
          f" ({delta / OFFICIAL_BASELINE * 100:+.2f} %)")
    print("verdict: below the advisor's <3.30 refuted band.")

    payload = {
        "empirical_local_curve": {
            str(m): statistics.median(v) for m, v in sorted(by_width.items())
        },
        "oracle_local_table": LOCAL_ROUND_US,
        "assumed_ranked_step_us": assumed,
        "measured_local_step_us": local_step,
        "sensitivity": [{"step_us": s, "gain_pct": g} for s, g in rows],
        "breakeven_step_us": lo,
        "official": {
            "baseline_submission": "f04b102",
            "baseline_score": OFFICIAL_BASELINE,
            "candidate_submission": "87b654b",
            "candidate_score": OFFICIAL_CANDIDATE,
            "delta": delta,
        },
    }
    path = ART / "transfer-postmortem.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
