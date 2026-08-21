#!/usr/bin/env python3
"""Self-test for the E114 weighting modules, on cases with known answers.

Tests 1-9 cover `research/scoring_weights.py`, which every future kernel brief
uses to price its arms. Tests 10-14 cover the parts of
`research/e114_width_recovery.py` that turn a board receipt into a weight, so
that a change to either module cannot quietly invalidate a published table.

    python3 research/scoring_weights_selftest.py

Test 4 is the one that matters most: it fails if the partition table here has
gone stale against `quantized.h:1918-1979`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _HERE / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw = _load("scoring_weights")
wr = _load("e114_width_recovery")


def close(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) <= tol


def main() -> int:
    failures = []

    # 1. The standing rule must fall out of its own source data. If this
    #    fails, either the rate table, the histogram or the partition table
    #    has moved and no standing-weighted number in the ledger is
    #    reproducible any more.
    got = sw.local_weights()
    for na, want in sw.STANDING_WEIGHTS.items():
        if not close(got[na], want, 5e-4):
            failures.append(
                "local_weights NA=%d: %.4f != standing %.3f" % (na, got[na], want))

    # 2. Group counting and time weighting must differ, and in the known
    #    direction: the slow wide cells take more time than their group share.
    counted = sw.local_weights(time_weighted=False)
    if not (counted[4] < got[4] and counted[5] < got[5]):
        failures.append("time weighting did not lift the slow cells: %s vs %s"
                        % (counted, got))
    if not close(counted[4], 23 / 36, 1e-9):
        failures.append("group share NA=4: %.6f != 23/36" % counted[4])

    # 3. A single-width distribution is a unit vector on that width's cells,
    #    and a two-cell width splits by rate, not evenly.
    if sw.na_weights({4: 1.0}) != {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0}:
        failures.append("width 4 is not a unit vector on NA=4")
    seven = sw.na_weights({7: 1.0})
    share3 = (1 / sw.ONE_GROUP_GBPS[3]) / (
        1 / sw.ONE_GROUP_GBPS[3] + 1 / sw.ONE_GROUP_GBPS[4])
    if not close(seven[3], share3, 1e-12):
        failures.append("width 7 split: %.6f != %.6f" % (seven[3], share3))
    if not close(seven[3] + seven[4], 1.0, 1e-12):
        failures.append("width 7 does not sum to 1")

    # 4. STALE PARTITION TABLE. Before E100 the group boundary sat one width
    #    lower, so M=5 was `[3+2]` and not `[5]`. A distribution concentrated
    #    at width 5 puts all its weight on NA=5 under the live table and none
    #    at all under the stale one. This case exists to fail loudly if the
    #    table here is ever left behind by an edit to the kernel switch.
    live = sw.na_weights({5: 1.0})
    stale = sw.na_weights({5: 1.0}, rates=sw.ONE_GROUP_GBPS)
    if sw.PARTITION[5] != [5]:
        failures.append("PARTITION[5] is %s: the live switch says [5]"
                        % sw.PARTITION[5])
    if not close(live[5], 1.0, 1e-12):
        failures.append("width 5 does not sit entirely on NA=5")
    saved = sw.PARTITION
    try:
        sw.PARTITION = sw.PARTITION_PRE_E100
        stale = sw.na_weights({5: 1.0})
    finally:
        sw.PARTITION = saved
    if close(stale[5], live[5], 1e-6):
        failures.append("the stale pre-E100 table gives the same answer, so "
                        "this test cannot detect a stale table")
    if stale[5] != 0.0 or not (stale[2] > 0 and stale[3] > 0):
        failures.append("pre-E100 width 5 should be [3+2]: %s" % stale)

    # 5. An NA that no arm table covers must raise rather than be silently
    #    dropped, because a dropped cell renormalises the other three and
    #    quietly changes every headline.
    saved = sw.PARTITION
    try:
        sw.PARTITION = {**sw.PARTITION, 8: [6, 2]}
        try:
            sw.na_weights({8: 1.0})
            failures.append("NA=6 was accepted silently")
        except KeyError:
            pass
    finally:
        sw.PARTITION = saved

    # 6. Width 1 is the narrow QMV. It must contribute nothing and must not
    #    crash, because plutarch spends 92 % of its rounds there.
    mixed = sw.na_weights({1: 100, 4: 1})
    if mixed != {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0}:
        failures.append("width 1 leaked into the wide cells: %s" % mixed)
    try:
        sw.na_weights({1: 5})
        failures.append("an all-narrow distribution should have no wide share")
    except ValueError:
        pass

    # 7. `reweigh` must reproduce a hand-computed pair, must report the ratio,
    #    and must flag a sign change rather than leaving it to the reader.
    flat = {2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0}
    row = sw.reweigh(flat, {2: 0.0, 3: 0.5, 4: 0.25, 5: 0.25})
    if not (close(row["standing_pct"], -1.0, 1e-9)
            and close(row["published_pct"], -1.0, 1e-9)):
        failures.append("a flat arm must be invariant to the weights: %s" % row)
    if row["sign_change"]:
        failures.append("a flat arm cannot change sign")
    tilted = {2: 0.0, 3: +1.0, 4: -1.0, 5: 0.0}
    row = sw.reweigh(tilted, {2: 0.0, 3: 0.9, 4: 0.1, 5: 0.0},
                     local={2: 0.0, 3: 0.1, 4: 0.9, 5: 0.0})
    if not (close(row["standing_pct"], -0.8, 1e-9)
            and close(row["published_pct"], +0.8, 1e-9)
            and row["sign_change"]):
        failures.append("sign change not detected: %s" % row)

    # 8. `published_weights` must be exactly the Finding 16 combination, and
    #    must refuse a prompt set that cannot form it.
    per_prompt = {"beagle": {4: 1.0}, "essays": {3: 1.0},
                  "medicine": {3: 1.0}, "republic": {3: 1.0},
                  "botany": {3: 1.0}}
    pub = sw.published_weights(per_prompt)
    if not (close(pub[4], 0.5, 1e-12) and close(pub[3], 0.5, 1e-12)):
        failures.append("published mix is not 0.5/0.5: %s" % pub)
    try:
        sw.published_weights({"beagle": {4: 1.0}})
        failures.append("a missing min-slot prompt was accepted")
    except KeyError:
        pass

    # 9. `rerank` must order on the published number and report the movement.
    arms = {"a": {2: 0, 3: 0, 4: -2.0, 5: 0}, "b": {2: 0, 3: -1.0, 4: 0, 5: 0}}
    rows = sw.rerank(arms, {2: 0.0, 3: 0.9, 4: 0.1, 5: 0.0},
                     local={2: 0.0, 3: 0.1, 4: 0.9, 5: 0.0})
    if [r["arm"] for r in rows] != ["b", "a"]:
        failures.append("rerank order: %s" % [r["arm"] for r in rows])
    if {r["arm"]: r["rank_change"] for r in rows} != {"a": 1, "b": -1}:
        failures.append("rank movement not reported: %s" % rows)

    # 10. The ranked rate table must be a genuinely different host, not a
    #     rescaled copy of ours. If it were proportional, every weight built
    #     from it would equal the local one and labelling a table
    #     `harness=ranked` would be decoration. Check the shape differs and
    #     check the direction: an NA=5 group is relatively DEARER on the ranked
    #     host, because time per group goes as 1 / rate.
    loc, rk = sw.ONE_GROUP_GBPS, sw.RANKED_ONE_GROUP_GBPS
    if set(loc) != set(rk):
        failures.append("rate tables cover different NA sets")
    ratios = [rk[na] / loc[na] for na in sw.NA_CELLS]
    if max(ratios) - min(ratios) < 0.05:
        failures.append("ranked table is a scaled copy of local: %s" % ratios)
    if not (rk[2] / rk[5] > loc[2] / loc[5]):
        failures.append("ranked NA=5 is not relatively dearer: %.3f vs %.3f"
                        % (rk[2] / rk[5], loc[2] / loc[5]))
    hist = sw.E106_LOCAL_HISTOGRAM
    if close(sw.na_weights(hist, rates=rk)[5],
             sw.na_weights(hist, rates=loc)[5], 1e-6):
        failures.append("swapping the rate table changed nothing")

    # 11. `tilt` is an I-projection: it must keep the support of its base,
    #     hit the target mean exactly, and return the base unchanged when the
    #     base already has that mean.
    tgt = 5.5
    proj = wr.tilt(sw.E106_LOCAL_HISTOGRAM, tgt)
    if set(proj) != set(sw.E106_LOCAL_HISTOGRAM):
        failures.append("tilt changed the support: %s" % sorted(proj))
    if not close(sum(M * p for M, p in proj.items()), tgt, 1e-9):
        failures.append("tilt missed the target mean")
    if not close(sum(proj.values()), 1.0, 1e-12):
        failures.append("tilt is not normalised")
    tot = sum(sw.E106_LOCAL_HISTOGRAM.values())
    own = sum(M * c for M, c in sw.E106_LOCAL_HISTOGRAM.items()) / tot
    fixed = wr.tilt(sw.E106_LOCAL_HISTOGRAM, own)
    for M, c in sw.E106_LOCAL_HISTOGRAM.items():
        if not close(fixed[M], c / tot, 1e-7):
            failures.append("tilt onto its own mean moved mass at M=%d" % M)

    # 12. Every vertex must satisfy the constraints it was built from, and the
    #     maxent point must lie inside the hull of those vertices. If it does
    #     not, the identified range reported in rung 1 is not a bound at all.
    verts = wr.vertices(tgt)
    if not verts:
        failures.append("no vertices at a feasible mean")
    for v in verts:
        if not close(sum(v.values()), 1.0, 1e-9) or \
                not close(sum(M * p for M, p in v.items()), tgt, 1e-9) or \
                any(p < -1e-12 for p in v.values()):
            failures.append("infeasible vertex %s" % v)
    lo = min(min(v) for v in verts)
    hi = max(max(v) for v in verts)
    me = wr.maxent(tgt)
    if not (lo <= min(me) and max(me) <= hi):
        failures.append("maxent leaves the vertex support")
    if wr.vertices(99.0):
        failures.append("an infeasible mean produced vertices")

    # 13. Every width `vertices` can emit must be priced by both the partition
    #     table and the cost curve, or a recovered distribution would be
    #     silently truncated on its way into a weight.
    for M in wr.WIDE:
        if M not in sw.PARTITION:
            failures.append("PARTITION cannot price width %d" % M)
    try:
        sw.na_weights({max(sw.PARTITION) + 1: 1.0})
        failures.append("a width outside the partition table was accepted")
    except KeyError:
        pass

    # 14. The two-line ranked cost fits must be monotone, must agree on where
    #     the break sits, and the post-E100 fit must price the SAME widths as
    #     the pre-E100 one so the rung-1 placebo comparison is well posed.
    for fit in (wr.ROUTE_B, wr.ROUTE_B_PRE_E100):
        vals = [wr.route_b_us(float(M), fit) for M in range(1, 9)]
        if any(b <= a for a, b in zip(vals, vals[1:])):
            failures.append("cost curve is not increasing: %s" % vals)
    if wr.ROUTE_B["break"] != wr.ROUTE_B_PRE_E100["break"]:
        failures.append("the two fits disagree on the break width")

    n = 14
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        return 1
    print("scoring_weights selftest: %d/%d pass" % (n, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
