#!/usr/bin/env python3
"""Re-weight the E78 rung 3 result onto the ranked round pool, with sign tests.

  python3 research/e78_rung3_reweight.py

Rung 3 measured arm E end to end on one local fixture whose verify-width
histogram is fixed. Arm E only moves `mlp.down` at M = 6, so the effect at
every other width is exactly zero and the local-to-ranked transfer is a single
ratio of M = 6 weights. This is DERIVED re-weighting of one measured number,
not a new measurement.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from math import comb

ART = pathlib.Path("research/e78-artifacts")
CALLS_MLP_DOWN = 64
LOCAL_ROUNDS = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
RANKED_SHARE_PCT = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}
TOKENS = 512


def main() -> int:
    cells = json.loads((ART / "rung2a-cells.json").read_text())
    cell = {(c["shape"], c["m"], c["ipg"]): c for c in cells["cells"]}
    r3 = json.loads((ART / "rung3.json").read_text())

    base = r3["baseline_seconds_per_token"]
    meas = r3["arms"]["e_kdown"]["delta_vs_baseline"]
    null = r3["session_null_seconds_per_token"]
    thr = r3["useful_effect_threshold_seconds_per_token"]

    per_disp = (cell[("mlp.down", 6, 3)]["seconds_per_dispatch"]["mean"]
                - cell[("mlp.down", 6, 6)]["seconds_per_dispatch"]["mean"])
    d6_ms = CALLS_MLP_DOWN * per_disp * 1000.0

    local_total = sum(LOCAL_ROUNDS.values())
    ranked_total = sum(RANKED_SHARE_PCT.values())
    widths = sorted(set(LOCAL_ROUNDS) | set(RANKED_SHARE_PCT))

    print("mlp.down M=6 per-dispatch delta = %+.6f ms" % (per_disp * 1000.0))
    print("arm E delta at M=6 = %+.5f ms/round (64 calls)" % d6_ms)
    print()
    print("| M | local rounds | local weight | ranked share % | ranked weight"
          " | arm E delta ms/round |")
    print("|---:|---:|---:|---:|---:|---:|")
    for m in widths:
        lr = LOCAL_ROUNDS.get(m, 0)
        rs = RANKED_SHARE_PCT.get(m)
        dm = d6_ms if m == 6 else 0.0
        print("| %d | %d | %.4f | %s | %s | %+.5f |"
              % (m, lr, lr / local_total,
                 "%.2f" % rs if rs is not None else "-",
                 "%.4f" % (rs / ranked_total) if rs is not None else "-",
                 dm))

    loc = (LOCAL_ROUNDS[6] / local_total) * d6_ms
    rnk = (RANKED_SHARE_PCT[6] / ranked_total) * d6_ms
    print()
    print("local-mix weighted mean  = %+.5f ms/round" % loc)
    print("ranked-mix weighted mean = %+.5f ms/round" % rnk)
    print("ranked / local           = %.5f" % (rnk / loc))

    pred = loc * local_total / TOKENS / 1000.0
    print()
    print("predicted local delta = %+.6e s/tok (%+.4f %%)"
          % (pred, 100 * pred / base))
    print("measured  local delta = %+.6e s/tok (%+.4f %%)"
          % (meas, 100 * meas / base))
    print("prediction / measurement = %.3f x" % (pred / meas))

    rk = meas * (rnk / loc)
    print("ranked-reweighted measured estimate = %+.6e s/tok (%+.4f %%)"
          % (rk, 100 * rk / base))
    print()
    print("session null = %.6e s/tok (%.4f %%)" % (null, 100 * null / base))
    print("threshold    = %.6e s/tok (%.4f %%)" % (thr, 100 * thr / base))
    print("|measured| / null    = %.3f" % (abs(meas) / null))
    print("|ranked est| / null  = %.3f" % (abs(rk) / null))
    print("passes threshold: local=%s ranked=%s"
          % (abs(meas) >= thr, abs(rk) >= thr))

    a = [leg["metrics"]["mtp_seconds_per_token"]
         for tag in ("a1", "a2") for leg in r3["groups"][tag]["legs"]]
    e = [leg["metrics"]["mtp_seconds_per_token"]
         for tag in ("e1", "e2") for leg in r3["groups"][tag]["legs"]]
    u = sum(1 for x in a for y in e if y < x)
    vals = a + e
    hits = tot = 0
    for combo in itertools.combinations(range(len(vals)), len(e)):
        ei = [vals[i] for i in combo]
        ai = [vals[i] for i in range(len(vals)) if i not in combo]
        tot += 1
        if sum(1 for x in ai for y in ei if y < x) >= u:
            hits += 1
    print()
    print("leg rank test: U = %d of %d, exact one-sided p = %.5f (%d/%d)"
          % (u, len(a) * len(e), hits / tot, hits, tot))

    pairs = ([(r3["groups"]["a1"]["legs"][i]["metrics"]["mtp_seconds_per_token"],
               r3["groups"]["e1"]["legs"][i]["metrics"]["mtp_seconds_per_token"])
              for i in range(3)]
             + [(r3["groups"]["a2"]["legs"][i]["metrics"]["mtp_seconds_per_token"],
                 r3["groups"]["e2"]["legs"][i]["metrics"]["mtp_seconds_per_token"])
                for i in range(3)])
    diffs = [y - x for x, y in pairs]
    neg = sum(1 for x in diffs if x < 0)
    sp = sum(comb(6, k) for k in range(neg, 7)) / 2 ** 6
    print("position-matched pairs: %s"
          % ", ".join("%+.3e" % x for x in diffs))
    print("paired mean = %+.6e s/tok, negatives %d/6, sign-test p = %.4f"
          % (sum(diffs) / 6, neg, sp))

    p24 = sum(comb(24, k) for k in range(23, 25)) / 2 ** 24
    print()
    print("rung 2a cell sign test: 23 of 24 cells favour fewer groups; "
          "one-sided p = %.3e, two-sided p = %.3e" % (p24, 2 * p24))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
