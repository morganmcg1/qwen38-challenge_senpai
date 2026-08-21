#!/usr/bin/env python3
"""E94 rung 3: what the ranked-shaped depth price does to the greedy walk.

The advisor fitted two per-group affine round-cost models to 50 official
ranked runs on the reference schedule:

    G = 1, M = 1..4 : round_us = 27181.5 + 3995.1 * M
    G = 2, M = 5..8 : round_us = 16943.2 + 7233.0 * M

Normalised by the ranked width-1 round of 31176.6 us that gives per-step
marginals of 0.128 below the group boundary, 2.490 * 0.128 for the step into
verify width 5, and 1.810 * 0.128 above it. This script prints the resulting
price vector, the per-position walk coefficients against the shipped flat
price, and the depth the greedy walk selects at each flat acceptance rate on
both the shipped price and the ranked-shaped one.

Nothing here is measured. It is the arithmetic the Swift arm implements, kept
next to the experiment so the predicted histogram direction is checkable
without a build.
"""

from __future__ import annotations

SHIP_H = 0.18
RANKED_STEP = 0.128
RANKED_GROUP_FACTOR = 2.490
RANKED_TIER_FACTOR = 1.810
MAX_DEPTH = 8
SEGMENTED_CAP = 7

# The ranked fit itself, used to score each depth in ranked microseconds.
RANKED_G1 = (27181.5, 3995.1)
RANKED_G2 = (16943.2, 7233.0)


def ranked_round_us(width: int) -> float:
    intercept, slope = RANKED_G1 if width <= 4 else RANKED_G2
    return intercept + slope * width


def ranked_marginal() -> list[float]:
    """Marginal price of the step into verify width `index + 2`."""
    out = []
    for index in range(MAX_DEPTH):
        width = index + 2
        if width == 5:
            out.append(RANKED_STEP * RANKED_GROUP_FACTOR)
        elif width > 5:
            out.append(RANKED_STEP * RANKED_TIER_FACTOR)
        else:
            out.append(RANKED_STEP)
    return out


def cumulative(marginal: list[float]) -> list[float]:
    out = [1.0]
    running = 1.0
    for value in marginal:
        running += value
        out.append(running)
    return out


def walk(marginal: list[float], cumul: list[float], cap: int, p: float,
         entry_gate: float | None = None) -> int:
    """The shipped greedy walk. `entry_gate` is hard guard 2.

    The ranked first-tier step is cheaper than the shipped one, so without a
    gate the arm would draft at depth 1 on rounds the shipped price leaves
    non-drafting. The gate holds the entry test at the shipped threshold, so
    the arm's non-drafting set is exactly the shipped one.
    """
    if entry_gate is not None and p <= entry_gate:
        return 0
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= p
        threshold = marginal[depth] * (1.0 + expected) / cumul[depth]
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def ranked_us_per_token(depth: int, p: float) -> float:
    expected = sum(p ** i for i in range(1, depth + 1))
    return ranked_round_us(depth + 1) / (1.0 + expected)


def main() -> None:
    ship = [SHIP_H] * MAX_DEPTH
    m5fit = ranked_marginal()
    cship, cm5 = cumulative(ship), cumulative(m5fit)

    print("index | entering width | ship marginal | m5fit marginal | "
          "ship coef | m5fit coef")
    for index in range(MAX_DEPTH):
        print("%5d | %14d | %13.6f | %14.6f | %9.6f | %10.6f" % (
            index, index + 2, ship[index], m5fit[index],
            ship[index] / cship[index], m5fit[index] / cm5[index]))
    print("total: ship %.6f   m5fit %.6f" % (sum(ship), sum(m5fit)))

    print()
    print("hard guard 1 and 2, on a 1/10000 grid of flat q, cap 1..7")
    ungated_opens = 0
    shallower = 0
    for cap in range(1, SEGMENTED_CAP + 1):
        for step in range(1, 10001):
            p = step / 10000.0
            d_ship = walk(ship, cship, cap, p)
            d_open = walk(m5fit, cm5, cap, p)
            d_gate = walk(m5fit, cm5, cap, p, entry_gate=SHIP_H)
            if d_ship == 0 and d_open > 0:
                ungated_opens += 1
            if d_gate in (1, 2) and d_ship > d_gate:
                shallower += 1
            assert (d_gate == 0) == (d_ship == 0), (cap, p, d_ship, d_gate)
    print("rounds the UNGATED arm would open that ship leaves closed: %d"
          % ungated_opens)
    print("gated arm non-drafting set equals ship's: yes")
    print("gated arm choices of depth 1 or 2 that are shallower than ship: %d"
          % shallower)

    print()
    print("flat q | ship depth | m5fit depth | ranked optimum | "
          "ranked us/tok ship | m5fit | optimum | m5fit vs ship")
    for step in range(60, 101):
        p = step / 100.0
        d_ship = walk(ship, cship, SEGMENTED_CAP, p)
        d_m5 = walk(m5fit, cm5, SEGMENTED_CAP, p, entry_gate=SHIP_H)
        costs = [ranked_us_per_token(d, p) for d in range(SEGMENTED_CAP + 1)]
        best = min(range(SEGMENTED_CAP + 1), key=lambda d: costs[d])
        print("%6.2f | %10d | %11d | %14d | %18.1f | %9.1f | %7.1f | %+6.2f %%"
              % (p, d_ship, d_m5, best, costs[d_ship], costs[d_m5],
                 costs[best],
                 100.0 * (costs[d_m5] / costs[d_ship] - 1.0)))

    print()
    print("at the accept rates the advisor measured on the ranked board")
    print("prompt | q | ship depth | m5fit depth | ranked us/tok ship | "
          "m5fit | m5fit vs ship")
    for name, p in [("beagle", 0.834), ("botany", 0.866), ("medicine", 0.892),
                    ("essays", 0.897), ("republic", 0.903)]:
        d_ship = walk(ship, cship, SEGMENTED_CAP, p)
        d_m5 = walk(m5fit, cm5, SEGMENTED_CAP, p, entry_gate=SHIP_H)
        c_ship = ranked_us_per_token(d_ship, p)
        c_m5 = ranked_us_per_token(d_m5, p)
        print("%8s | %.3f | %10d | %11d | %18.1f | %9.1f | %+6.2f %%" % (
            name, p, d_ship, d_m5, c_ship, c_m5,
            100.0 * (c_m5 / c_ship - 1.0)))


if __name__ == "__main__":
    main()
