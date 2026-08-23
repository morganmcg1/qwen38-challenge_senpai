"""F215 / Campaign Rule 126 — price a shaped candidate-leg gain as an EXPECTATION
over the serial lottery.

Why this tool exists
--------------------
Rule 121 says: predict the eight raw ratios, sort them, read positions 3 and 4.
That is correct. But *which* prompt occupies position 4 depends on that row's
serial draw, and the candidate cannot control the serial draw. Pricing a shaped
gain against one row's realised sort order therefore conditions the answer on
that row's lottery ticket.

Finding 215 measured how large that effect is. Over a 29-row board window the
essays serial leg has sd 0.306 % and a range of -0.217 % to +1.409 %. The
published crown 684821ed beat the previous candidate frontier 1760479a while
being SLOWER on 7 of 8 candidate legs, purely because one essays serial leg drew
+1.4322 % slow. De-lucking the board by replacing every serial leg with the
window median collapses the top five rows to within 0.09 % of each other.

Campaign Rule 126
-----------------
Hold the anchor candidate vector fixed. Resample WHOLE serial vectors from the
board window, which preserves the within-run correlation between the eight legs.
Apply the gain shape. Take the expectation. Report the expectation, p05, and the
upper-slot occupancy census.

The headline consequence: a UNIFORM gain has exactly zero variance across the
serial lottery, by construction, because it scales all eight raw ratios by the
same factor and the median follows 1:1. A shaped gain does not. An essays-heavy
shape is worth a lot when essays binds and nearly nothing when it does not, and
essays binds on 89.7 % of board rows but not on the crown's own row.

Usage
-----
    python3 research/f215_lottery_price.py --uniform 0.50
    python3 research/f215_lottery_price.py --gain beagle=1.1482
    python3 research/f215_lottery_price.py \
        --gain beagle=0.4057,essays=0.7781,republic=0.2207,medicine=0.3177,\
botany=0.3862,drama=0.8146,travel=0.6301,plutarch=0.0008
    python3 research/f215_lottery_price.py --ladder
    python3 research/f215_lottery_price.py --anchor 684821ed --uniform 0.30

A gain is expressed as a PERCENTAGE REDUCTION IN CANDIDATE SECONDS PER TOKEN on
that prompt. Positive means faster. Missing prompts default to zero.

Data
----
research/f215-serial-pool.json holds the ranked per-prompt legs from the public
Yukon board, one whole vector per submission row. Regenerate it from a fresh
board dump when the window moves; the pool is deliberately a snapshot so results
are reproducible.
"""
import argparse
import json
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "f215-serial-pool.json")
DEFAULT_BAR = 3.71959722580154
N = 40000
SEED = 20260823


def load():
    with open(POOL) as fh:
        return json.load(fh)


def median_pair(cand, ser):
    """The published score: sort the eight raw ratios, mean of positions 3 and 4."""
    v = sorted(s / c for s, c in zip(ser, cand))
    return (v[3] + v[4]) / 2.0


def upper_owner(cand, ser, order):
    v = sorted((s / c, n) for s, c, n in zip(ser, cand, order))
    return v[4][1]


def parse_gain(spec, uniform, order):
    if uniform is not None:
        return {n: uniform for n in order}
    g = {n: 0.0 for n in order}
    if not spec:
        return g
    for part in spec.replace("\n", "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        k = k.strip()
        if k not in g:
            raise SystemExit("unknown prompt %r; expected one of %s" % (k, order))
        g[k] = float(v)
    return g


def price(data, anchor, gain, bar, label):
    order = data["order"]
    pool = [r["serial"] for r in data["serial_pool"]]
    anchors = data["anchor_candidate_vectors"]
    if anchor not in anchors:
        raise SystemExit("no anchor %r; have %s" % (anchor, sorted(anchors)))
    cand0 = anchors[anchor]
    cand1 = [c / (1.0 + gain[n] / 100.0) for c, n in zip(cand0, order)]

    random.seed(SEED)
    deltas, levels, owner = [], [], {}
    for _ in range(N):
        s = random.choice(pool)
        m0 = median_pair(cand0, s)
        m1 = median_pair(cand1, s)
        deltas.append(100.0 * (m1 / m0 - 1.0))
        levels.append(m1)
        u = upper_owner(cand0, s, order)
        owner[u] = owner.get(u, 0) + 1
    deltas.sort()
    levels.sort()
    beat = sum(1 for m in levels if m > bar) / float(N)

    # the same shape priced against each anchor row's own realised serial draw,
    # which is what Rule 126 tells you NOT to rely on
    cond = {}
    for r in data["serial_pool"]:
        if r["id8"] in anchors:
            s = r["serial"]
            cond[r["id8"]] = 100.0 * (median_pair(cand1, s) / median_pair(cand0, s) - 1.0)

    print("%s   anchor candidate leg %s   pool %d vectors" % (label, anchor, len(pool)))
    print("  expected median gain %+.4f %%   sd %.4f   p05 %+.4f   p50 %+.4f   p95 %+.4f"
          % (statistics.fmean(deltas), statistics.pstdev(deltas),
             deltas[int(0.05 * N)], deltas[N // 2], deltas[int(0.95 * N)]))
    print("  expected published median %.5f   p05 %.5f   p95 %.5f   P(beat %.8f) %.1f %%"
          % (statistics.fmean(levels), levels[int(0.05 * N)],
             levels[int(0.95 * N)], bar, 100.0 * beat))
    if cond:
        print("  conditioned on one row's realised serial draw: "
              + "  ".join("%s %+.4f %%" % (k, v) for k, v in sorted(cond.items())))
    print("  upper-slot owner in the resample: "
          + "  ".join("%s %.1f %%" % (k, 100.0 * v / N)
                      for k, v in sorted(owner.items(), key=lambda t: -t[1])))
    print()


def ladder(data, anchor, bar):
    order = data["order"]
    for u in (0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00):
        price(data, anchor, {n: u for n in order}, bar, "uniform %+.2f %%" % u)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", default="1760479a",
                    help="candidate leg to apply the gain to (default 1760479a, "
                         "the fastest promoted candidate leg; Rule 124)")
    ap.add_argument("--uniform", type=float, default=None,
                    help="uniform gain in percent, applied to all eight prompts")
    ap.add_argument("--gain", default=None,
                    help="shaped gain, e.g. beagle=1.1482,essays=0.5")
    ap.add_argument("--bar", type=float, default=DEFAULT_BAR,
                    help="published median to beat (default: the current bar)")
    ap.add_argument("--ladder", action="store_true",
                    help="print the uniform gain ladder instead of one shape")
    a = ap.parse_args()

    data = load()
    print("F215 serial pool captured %s, window since %s"
          % (data["captured_utc"], data["window_since_utc"]))
    print()
    if a.ladder:
        ladder(data, a.anchor, a.bar)
        return
    gain = parse_gain(a.gain, a.uniform, data["order"])
    label = "uniform %+.4f %%" % a.uniform if a.uniform is not None else "shaped gain"
    price(data, a.anchor, gain, a.bar, label)


if __name__ == "__main__":
    main()
