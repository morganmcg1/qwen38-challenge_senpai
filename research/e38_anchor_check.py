#!/usr/bin/env python3
"""Cross-session anchor: is this session comparable to E33's?

The advisor's requirement -- "E33's cross-session anchor held to 0.017 %
(128.843 vs 128.865); use the same anchor to prove this session is comparable".

E33's base ladder was measured at BASE_SHA 4e5dc2b; E38's at 54248ce.  The
crossrow kernel files are byte-identical between those two commits, so any
difference is session/thermal drift and not a code change.

  python3 research/e38_anchor_check.py [tag]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import e38_prereg as P  # noqa: E402

CURVE = ".mlxfast-private/qmv-curve/%s/vendored.json"


def c_round(tag):
    d = json.load(open(CURVE % tag))
    t = {}
    for sh in d["shapes"]:
        for r in sh["rows"]:
            t[r["m"]] = t.get(r["m"], 0.0) + sh["calls_per_verify"] * r["seconds_per_call"]
    return {m: v * 1e3 for m, v in t.items()}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "e38-base-r1"
    if not os.path.exists(CURVE % tag):
        sys.exit(f"missing {CURVE % tag}")
    mine = c_round(tag)
    e33 = P.BASE_C_ROUND_MS

    print("C_round(M) ms, unchanged base kernel, two sessions")
    print(f"  {'M':>3} {'E33 (4e5dc2b)':>15} {'E38 (54248ce)':>15} {'drift %':>9}")
    drift = {}
    for m in sorted(set(mine) & set(e33)):
        d = 100 * (mine[m] / e33[m] - 1)
        drift[m] = d
        flag = ("  <-- treated width" if m == 6
                else "  <-- warmup-contaminated" if m <= 2 else "")
        print(f"  {m:>3} {e33[m]:15.3f} {mine[m]:15.3f} {d:+9.3f}{flag}")

    # M=1 and M=2 are the first widths timed in each session and carry the
    # session's warmup: their within-run (mean-min)/min spread is 13.8 % and
    # 11.1 % against 2.6-4.6 % everywhere else.  They are not usable anchors,
    # and for the same reason they must not enter a drift/control estimate.
    settled = {m: d for m, d in drift.items() if m >= 3}
    worst_all = max(abs(d) for d in drift.values())
    worst_settled = max(abs(d) for d in settled.values())
    print()
    print(f"  worst |drift| all widths  = {worst_all:.3f} %"
          f"  (driven by M<=2 warmup)")
    print(f"  worst |drift| M>=3        = {worst_settled:.3f} %"
          f"  (E33's own anchor: 0.017 %)")
    print(f"  M=6 anchor                = {drift[6]:+.3f} %")
    print()
    if worst_settled < 0.5:
        print("VERDICT: comparable for M>=3; E33 per-shape ratios may be quoted")
        print("alongside E38's with the drift stated.  Exclude M<=2 from any")
        print("control or drift estimate.")
    else:
        print("VERDICT: drift too large to quote E33 numbers as-is.")


if __name__ == "__main__":
    main()
