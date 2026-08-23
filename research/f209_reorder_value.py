#!/usr/bin/env python3
"""FINDING 209 - the published median is a SORTED order statistic, so the
per-prompt marginal weights of Rule 116 are valid only for a move that
preserves the rank order of the eight raw ratios.

This tool prices a per-prompt gain the way the ranked workflow actually
computes it: apply the gain, re-sort the eight raw ratios, average sorted
positions 3 and 4.

Rule 116 gave w_beagle = 0.478 and w_essays = 0.522, which reads as "essays is
worth slightly more per unit than beagle". That is true only infinitesimally.
For any finite move essays saturates almost immediately and beagle does not.

usage
  python3 research/f209_reorder_value.py --anchor 572b2cc4
  python3 research/f209_reorder_value.py --anchor 1760479a --uniform 1.0
  python3 research/f209_reorder_value.py --anchor 572b2cc4 --gain beagle=3.0,essays=1.0
"""
import argparse
import json
import os

BOARD = os.environ.get("YUKON_BOARD", "/tmp/yukon-board/full.json")
NAMES = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}


def load(anchor):
    rows = json.load(open(BOARD))["submissions"]
    for r in rows:
        if (r.get("id") or "")[:8] != anchor:
            continue
        pp = (r.get("officialMetrics") or {}).get("per_prompt")
        if not pp:
            raise SystemExit("anchor %s has no per_prompt evidence" % anchor)
        vec = {NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]):
               e["raw_ratio_of_means"] for e in pp}
        return r, vec
    raise SystemExit("anchor %s not on the board" % anchor)


def median(vec):
    s = sorted(vec.values())
    return 0.5 * (s[3] + s[4])


def apply_gain(base, gain):
    return {k: v * (1.0 + gain.get(k, 0.0) / 100.0) for k, v in base.items()}


def saturation_scan(base, prompt, hi=40.0, step=0.005):
    """Walk the gain up and record every kink: the value of the median, and
    the point past which extra gain on this prompt stops paying."""
    m0 = median(base)
    kinks, last_slope, x = [], None, 0.0
    prev = m0
    ceiling_x, ceiling_m = 0.0, m0
    while x < hi:
        x += step
        m = median(apply_gain(base, {prompt: x}))
        slope = (m - prev) / step
        if m > ceiling_m + 1e-12:
            ceiling_x, ceiling_m = x, m
        if last_slope is not None and abs(slope - last_slope) > 1e-6:
            kinks.append((x, m, last_slope, slope))
        last_slope, prev = slope, m
    return m0, kinks, ceiling_x, ceiling_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--uniform", type=float, default=None,
                    help="apply this percent gain to every prompt")
    ap.add_argument("--gain", default=None,
                    help="comma list, e.g. beagle=3.0,essays=1.0")
    args = ap.parse_args()

    row, base = load(args.anchor)
    m0 = median(base)
    print("anchor %s  %s  officialScore %.8f  recomputed %.8f"
          % (args.anchor, row.get("solverUsername"), row["officialScore"], m0))

    order = sorted(base.items(), key=lambda kv: kv[1])
    print("\nsorted raw ratios")
    for i, (nm, v) in enumerate(order):
        tag = "  <== median pair" if i in (3, 4) else ""
        gap = ("  gap to next %+7.3f %%"
               % ((order[i + 1][1] / v - 1.0) * 100.0)) if i + 1 < 8 else ""
        print("  %d %-9s %9.5f%s%s" % (i, nm, v, gap, tag))

    print("\nbinding gap for Rule 121: rank 4 -> rank 5 is %.3f %%"
          % ((order[5][1] / order[4][1] - 1.0) * 100.0))

    if args.uniform is not None:
        g = {k: args.uniform for k in base}
        m = median(apply_gain(base, g))
        print("\nuniform %+.3f %% on every prompt -> median %.8f  (%+.4f %%)"
              % (args.uniform, m, (m / m0 - 1.0) * 100.0))
        print("  a broad mechanism preserves order exactly and converts 1:1")

    if args.gain:
        g = {}
        for part in args.gain.split(","):
            k, v = part.split("=")
            g[k.strip()] = float(v)
        after = apply_gain(base, g)
        m = median(after)
        print("\ngain %s -> median %.8f  (%+.4f %%)"
              % (args.gain, m, (m / m0 - 1.0) * 100.0))
        for i, (nm, v) in enumerate(sorted(after.items(), key=lambda kv: kv[1])):
            tag = "  <== median pair" if i in (3, 4) else ""
            print("  %d %-9s %9.5f%s" % (i, nm, v, tag))

    print("\nSINGLE-PROMPT SATURATION - how far each prompt can carry the median")
    print("  prompt     dM/dx at 0   first kink   value at kink   CEILING x"
          "   CEILING value")
    for nm in [k for k, _ in order]:
        m0b, kinks, cx, cm = saturation_scan(base, nm)
        slope0 = ((median(apply_gain(base, {nm: 0.01})) - m0b) / 0.01
                  / m0b * 100.0)
        if kinks:
            kx, km, _, _ = kinks[0]
            kink = "%9.3f %%  %+10.4f %%" % (kx, (km / m0b - 1.0) * 100.0)
        else:
            kink = "     none            -    "
        print("  %-9s %10.4f   %s   %8.3f %%   %+9.4f %%"
              % (nm, slope0, kink, cx, (cm / m0b - 1.0) * 100.0))
    print("\n  dM/dx at 0 is percent of median per percent of that prompt.")
    print("  CEILING is the largest median this prompt can reach alone.")


if __name__ == "__main__":
    main()
