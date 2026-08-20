#!/usr/bin/env python3
"""E68 rung 1: turn the measured QMV curve into a marginal verify-cost vector.

Two deliverables, both read straight off the leg artifacts:

1. C(M), the measured cost of one verify pass at each row count M, and its
   first difference. The E68 premise is that the 4->5 and 5->6 steps inverted
   after `t55`. This script measures both steps and prices them against the
   largest same-arm spread, so a claimed inversion has to clear the null bar
   the session itself produced.

2. The NA >= 7 closure. The shipped table routes every width up to 6 to a
   single group, so the `shipped` legs measure the single-stream ladder
   C1(NA) for NA in [2, 6] directly. The `t789` legs route 7/8/9 to their
   lone-NA form, so they extend the same ladder to NA in [7, 9]. The shipped
   mixed widths 7/8/9 then over-determine the concurrency discount that the
   advisor's model fixes at 0.80, which lets the session measure it instead of
   assuming it.

Everything here is arithmetic over artifacts. It runs no GPU work.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CURVE_DIR = REPO / ".mlxfast-private" / "qmv-curve"

# The advisor's modelled ladder, single stream, milliseconds, W = 14.412 GB.
# NA=8 and NA=9 are extrapolation: the measured ladder stops at NA=7.
LADDER_MS = {2: 64.40, 3: 72.17, 4: 82.24, 5: 95.48, 6: 122.34, 7: 147.21}
LADDER_EXTRAPOLATED_MS = {8: 172.21, 9: 199.21}
CONCURRENCY_DISCOUNT = 0.80

# The shipped dispatch table, verified live by `e59_arms.routing_table`.
SHIPPED_GROUPS = {
    2: (2,), 3: (3,), 4: (4,), 5: (5,), 6: (6,),
    7: (4, 3), 8: (4, 4), 9: (5, 4),
}


def modelled_c(width):
    groups = SHIPPED_GROUPS.get(width)
    if groups is None:
        return None
    ladder = dict(LADDER_MS)
    ladder.update(LADDER_EXTRAPOLATED_MS)
    if any(g not in ladder for g in groups):
        return None
    return ladder[groups[0]] + CONCURRENCY_DISCOUNT * sum(
        ladder[g] for g in groups[1:])


def load_legs(tags, curve_dir):
    legs = []
    for arm, tag, position in tags:
        base = curve_dir / tag
        summary = json.loads((base / "summary.json").read_text())
        vendored = json.loads((base / "vendored.json").read_text())
        legs.append({
            "arm": arm,
            "tag": tag,
            "position": position,
            "weighted": {int(k): v for k, v in
                         summary["weighted_verify_seconds"].items()},
            "shapes": vendored["shapes"],
            "base_sha": summary.get("base_sha"),
            "host": summary.get("host"),
        })
    return legs


def per_arm(legs, arm):
    return [leg for leg in legs if leg["arm"] == arm]


def spread(values):
    return max(values) - min(values) if len(values) > 1 else 0.0


def curve_table(legs, widths):
    """C(M) per arm: median, spread, and the same-arm null bar."""
    out = {}
    for arm in sorted({leg["arm"] for leg in legs}):
        rows = {}
        arm_legs = per_arm(legs, arm)
        for w in widths:
            vals = [leg["weighted"][w] for leg in arm_legs
                    if w in leg["weighted"]]
            if not vals:
                continue
            med = statistics.median(vals)
            rows[w] = {
                "n": len(vals),
                "median_s": med,
                "spread_s": spread(vals),
                "spread_frac": spread(vals) / med if med else 0.0,
                "values_s": vals,
            }
        out[arm] = rows
    return out


def marginal_curve(curve, arm, widths):
    """First difference of C(M): the cost of admitting one more verify row."""
    rows = curve.get(arm, {})
    out = {}
    for w in widths[1:]:
        if w not in rows or (w - 1) not in rows:
            continue
        step = rows[w]["median_s"] - rows[w - 1]["median_s"]
        # A difference of two medians carries both null bars.
        bar = rows[w]["spread_s"] + rows[w - 1]["spread_s"]
        out[w] = {
            "step_s": step,
            "step_ms": step * 1e3,
            "null_bar_s": bar,
            "clears_null": abs(step) > bar,
            "frac_of_c1": step / rows[1]["median_s"] if 1 in rows else None,
        }
    return out


def modelled_marginals():
    out = {}
    prev = None
    for w in sorted(SHIPPED_GROUPS):
        c = modelled_c(w)
        if c is not None and prev is not None:
            out[w] = c - prev
        if c is not None:
            prev = c
    return out


def dominant_shapes(legs, top):
    """Rank shapes by their m=1 cost times calls per verify."""
    leg = legs[0]
    scored = []
    for shape in leg["shapes"]:
        row1 = next((r for r in shape["rows"] if r["m"] == 1), None)
        if row1 is None:
            continue
        scored.append((row1["seconds_per_call"] * shape["calls_per_verify"],
                       shape["name"]))
    scored.sort(reverse=True)
    return [name for _, name in scored[:top]]


def shape_row(leg, shape_name, m):
    for shape in leg["shapes"]:
        if shape["name"] != shape_name:
            continue
        for row in shape["rows"]:
            if row["m"] == m:
                return row
    return None


def shape_median(legs, shape_name, m):
    vals = [r["seconds_per_call"] for r in
            (shape_row(leg, shape_name, m) for leg in legs) if r]
    return statistics.median(vals) if vals else None


def ladder_closure(legs, shapes):
    """Measure C1(NA) for NA in [2, 9] and back out the group discount.

    `shipped` widths 2..6 are already lone-NA, so they give C1(2..6).
    `t789` widths 7..9 give C1(7..9). The shipped mixed widths then give three
    independent readings of the discount the model fixes at 0.80:

        shipped C(7) = C1(4) + d * C1(3)
        shipped C(8) = C1(4) + d * C1(4)
        shipped C(9) = C1(5) + d * C1(4)
    """
    ship = per_arm(legs, "shipped")
    t789 = per_arm(legs, "t789")
    out = {}
    for name in shapes:
        c1 = {}
        for na in range(2, 7):
            v = shape_median(ship, name, na)
            if v is not None:
                c1[na] = v
        for na in (7, 8, 9):
            v = shape_median(t789, name, na)
            if v is not None:
                c1[na] = v
        discounts = {}
        for width, (head, tail) in ((7, (4, 3)), (8, (4, 4)), (9, (5, 4))):
            mixed = shape_median(ship, name, width)
            if mixed is None or head not in c1 or tail not in c1 or not c1[tail]:
                continue
            discounts[width] = (mixed - c1[head]) / c1[tail]
        # The rejection the advisor made rests on t789 costing more than the
        # shipped split at every one of 7, 8, 9.
        contrasts = {}
        for width in (7, 8, 9):
            a = shape_median(ship, name, width)
            b = shape_median(t789, name, width)
            if a and b:
                contrasts[width] = {
                    "shipped_s": a, "t789_s": b, "ratio": b / a,
                    "pct": 100.0 * (b / a - 1.0),
                }
        out[name] = {
            "c1_seconds_per_call": c1,
            "measured_discount": discounts,
            "t789_vs_shipped": contrasts,
        }
    return out


def table_contrast(curve, widths):
    """t789 against shipped at the whole-verify level."""
    out = {}
    ship = curve.get("shipped", {})
    cand = curve.get("t789", {})
    for w in widths:
        if w not in ship or w not in cand:
            continue
        a, b = ship[w]["median_s"], cand[w]["median_s"]
        bar = ship[w]["spread_s"] + cand[w]["spread_s"]
        out[w] = {
            "shipped_s": a,
            "t789_s": b,
            "delta_s": b - a,
            "pct": 100.0 * (b / a - 1.0),
            "null_bar_s": bar,
            "clears_null": abs(b - a) > bar,
        }
    return out


def pbfit_marginals(curve, verify_forward_s, head_ratio, max_depth,
                    rank_flatten=1.0):
    """The measured depth price vector, in the scheduler's own units.

    The scheduler prices one more draft as a fraction of the width-1 verify
    forward pass, and the shipped vector is flat at `head_ratio`. The measured
    vector keeps the same total, so this is a pure shape test:

        raw[d]      = head_ratio + (C(d + 2) - C(d + 1)) / verify_forward_s
        marginal[d] = raw[d] * (max_depth * head_ratio) / sum(raw)

    Index d is 0-based and prices the step INTO verify width d + 2.

    `rank_flatten` divides only the measured width term. The ranked host is
    further from its bandwidth roof than this Mac, so its width curve is
    flatter; dividing the width term by that factor estimates the curve the
    ranked scheduler faces. The rescale to a fixed total then divides most of
    a uniform flattening back out, which is itself a reportable result.
    """
    rows = curve.get("shipped", {})
    raw = []
    for d in range(max_depth):
        w = d + 2
        if w not in rows or (w - 1) not in rows:
            raw.append(None)
            continue
        step = (rows[w]["median_s"] - rows[w - 1]["median_s"]) / rank_flatten
        raw.append(head_ratio + step / verify_forward_s)
    if any(v is None for v in raw):
        return {"error": "curve incomplete", "raw": raw}
    total = max_depth * head_ratio
    scale = total / sum(raw)
    return {
        "raw": raw,
        "marginal": [v * scale for v in raw],
        "scale": scale,
        "total": total,
        "verify_forward_s": verify_forward_s,
        "rank_flatten": rank_flatten,
    }


BOUNDARY_TIER_FACTOR = 2.0301


def uniform_marginals(head_ratio, max_depth):
    return [head_ratio] * max_depth


def boundary_marginals(head_ratio, max_depth, entering_verify_width):
    """E56's one-boundary vector, generalised. Holds the total at 8 * h.

    `entering_verify_width` w prices the step INTO verify width w, which is
    0-based index w - 2. That index carries `BOUNDARY_TIER_FACTOR` times the
    within-tier price, and the within-tier price absorbs the difference so the
    vector sums to `max_depth * head_ratio`.
    """
    index = entering_verify_width - 2
    if not 0 <= index < max_depth:
        raise ValueError("verify width %d is outside the vector"
                         % entering_verify_width)
    within = (max_depth * head_ratio /
              (max_depth - 1 + BOUNDARY_TIER_FACTOR))
    out = [within] * max_depth
    out[index] = within * BOUNDARY_TIER_FACTOR
    return out


def walk_depth(marginals, cap, p):
    """Replay `costModelDepth` exactly, under a flat acceptance p.

    The shipped walk is:

        reach = 1; expected = 0; depth = 0
        while depth < cap:
            reach *= p
            threshold = h * (1 + expected) / (1 + depth * h)
            if not reach > threshold: break
            expected += reach; depth += 1

    `expected` accumulates every previous reach, and the denominator is the
    running cost. The per-position generalisation replaces `h` by
    `marginals[depth]` and `1 + depth * h` by `1 + sum(marginals[:depth])`.
    """
    reach = 1.0
    expected = 0.0
    cumulative = 1.0
    depth = 0
    steps = []
    for d in range(min(cap, len(marginals))):
        reach *= p
        threshold = marginals[d] * (1.0 + expected) / cumulative
        take = reach > threshold
        steps.append({
            "depth": d + 1, "verify_width": d + 2,
            "reach": reach, "threshold": threshold,
            "headroom_pct": 100.0 * (reach / threshold - 1.0)
                            if threshold else float("inf"),
            "take": take,
        })
        if not take:
            break
        expected += reach
        cumulative += marginals[d]
        depth = d + 1
    return {"depth": depth, "verify_width": depth + 1, "steps": steps}


def crossing_p(marginals, cap, depth_index, tolerance=1e-9):
    """The flat acceptance at which the walk first reaches `depth_index + 1`.

    `walk_depth` is monotone in p, so bisect it. Returns None when the step is
    unreachable at any p below 1, which happens once the cap truncates first.
    """
    target = depth_index + 1
    if target > cap:
        return None
    if walk_depth(marginals, cap, 1.0 - 1e-12)["depth"] < target:
        return None
    low, high = 0.0, 1.0 - 1e-12
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if walk_depth(marginals, cap, mid)["depth"] >= target:
            high = mid
        else:
            low = mid
    return high


def crossing_table(arms, cap, max_depth):
    return {name: {"depth_%d" % (d + 1): crossing_p(vec, cap, d)
                   for d in range(min(max_depth, cap))}
            for name, vec in sorted(arms.items())}


def self_check():
    """Reproduce every published E56 constant from this module's arithmetic.

    E56 is the only prior art for a non-uniform depth price here, and its
    branch is outside this launch's isolation scope. Reproducing its four
    published numbers from the generalised walk is what licenses reusing the
    construction: if any of them misses, the reconstruction is wrong and the
    rung-3 arms are not the arms the advisor named.
    """
    vec = boundary_marginals(0.18, 8, 5)
    within, boundary = vec[0], vec[3]
    cumulative = 1.0 + sum(vec[:3])
    coefficient = boundary / cumulative
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lhs = mid ** 4
        rhs = coefficient * (1 + mid + mid ** 2 + mid ** 3)
        if lhs > rhs:
            hi = mid
        else:
            lo = mid
    # Each tolerance is half an ulp of the precision the source publishes.
    # The ledger prints six significant figures, so demanding more would test
    # my transcription rather than the reconstruction.
    checks = [
        ("withinTier", within, 0.159467, 5e-7),
        ("boundary marginal", boundary, 0.323733, 5e-6),
        ("cumulative[3]", cumulative, 1.478400, 5e-7),
        ("crossing coefficient", coefficient, 0.218975, 5e-6),
        ("crossing p", 0.5 * (lo + hi), 0.9491, 5e-4),
    ]
    ok = True
    print("E56 reconstruction self-check")
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok = ok and good
        print("  %-22s %.8f  published %.8f  %s"
              % (name, got, want, "ok" if good else "MISMATCH"))
    if not ok:
        raise SystemExit("e68_rung1_analysis: E56 reconstruction failed")
    print()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", action="append", default=[],
                    metavar="ARM:TAG",
                    help="one leg, in session order; repeat")
    ap.add_argument("--curve-dir", default=str(CURVE_DIR))
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--top-shapes", type=int, default=4)
    ap.add_argument("--verify-forward-s", default="0.0657",
                    help="width-1 verify forward pass, seconds; the "
                         "scheduler's unit for `headStepCostRatio`. Accepts a "
                         "comma list: V only sets how much curve shape enters "
                         "the vector, so the depth decision must be reported "
                         "with its sensitivity to V.")
    ap.add_argument("--head-ratio", type=float, default=0.18)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--acceptance", default="0.8351,0.8750,0.9189,0.9625")
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--rank-flatten", type=float, default=1.160,
                    help="the ranked width curve is flatter than the local "
                         "one by this factor, so `pbfit_ranked` divides the "
                         "measured width term by it. One significant figure.")
    ap.add_argument("--out")
    ap.add_argument("--self-check-only", action="store_true")
    args = ap.parse_args()

    self_check()
    if args.self_check_only:
        return 0
    if not args.leg:
        raise SystemExit("e68_rung1_analysis: pass at least one --leg ARM:TAG")
    tags = []
    for position, spec in enumerate(args.leg, start=1):
        arm, _, tag = spec.partition(":")
        if not tag:
            raise SystemExit("e68_rung1_analysis: bad --leg %r" % spec)
        tags.append((arm, tag, position))

    widths = [int(w) for w in args.widths.split(",")]
    legs = load_legs(tags, pathlib.Path(args.curve_dir))
    curve = curve_table(legs, widths)
    shapes = dominant_shapes(legs, args.top_shapes)

    ship_marg = marginal_curve(curve, "shipped", widths)
    verify_forwards = [float(v) for v in args.verify_forward_s.split(",")]
    fits = {"%.6f" % v: pbfit_marginals(curve, v, args.head_ratio,
                                        args.max_depth)
            for v in verify_forwards}
    fit = fits["%.6f" % verify_forwards[0]]
    fits_ranked = {"%.6f" % v: pbfit_marginals(curve, v, args.head_ratio,
                                               args.max_depth,
                                               args.rank_flatten)
                   for v in verify_forwards}

    acceptances = [float(x) for x in args.acceptance.split(",")]
    base_arms = {
        "ship": uniform_marginals(args.head_ratio, args.max_depth),
        "pb5": boundary_marginals(args.head_ratio, args.max_depth, 5),
        "pb7": boundary_marginals(args.head_ratio, args.max_depth, 7),
    }
    walks = {}
    crossings = {}
    for key, f in sorted(fits.items()):
        if "marginal" not in f:
            continue
        arms = dict(base_arms, pbfit=f["marginal"])
        ranked = fits_ranked[key]
        if "marginal" in ranked:
            arms["pbfit_ranked"] = ranked["marginal"]
        walks[key] = {
            "%.4f" % p: {name: walk_depth(vec, args.cap, p)
                         for name, vec in arms.items()}
            for p in acceptances
        }
        # The cap hides every crossing above it, so solve crossings against
        # the streak-opened cap as well as the cap in force.
        crossings[key] = {
            "cap_%d" % args.cap: crossing_table(arms, args.cap,
                                                args.max_depth),
            "cap_%d" % args.max_depth: crossing_table(arms, args.max_depth,
                                                      args.max_depth),
        }

    payload = {
        "legs": [{k: leg[k] for k in
                  ("arm", "tag", "position", "base_sha", "host")}
                 for leg in legs],
        "widths": widths,
        "curve_seconds": curve,
        "shipped_marginal": ship_marg,
        "modelled_marginal_ms": modelled_marginals(),
        "t789_vs_shipped": table_contrast(curve, widths),
        "ladder_closure": ladder_closure(legs, shapes),
        "pbfit": fit,
        "pbfit_by_verify_forward": fits,
        "pbfit_ranked": fits_ranked["%.6f" % verify_forwards[0]],
        "pbfit_ranked_by_verify_forward": fits_ranked,
        "rank_flatten": args.rank_flatten,
        "walks": walks,
        "crossing_p": crossings,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out:
        pathlib.Path(args.out).write_text(text)

    ship = curve.get("shipped", {})
    print("C(M) shipped, milliseconds per verify pass")
    for w in widths:
        if w not in ship:
            continue
        row = ship[w]
        step = ship_marg.get(w)
        print("  M=%-2d  %8.3f  spread %6.3f (%4.2f%%)  step %+8.3f%s"
              % (w, row["median_s"] * 1e3, row["spread_s"] * 1e3,
                 100.0 * row["spread_frac"],
                 step["step_ms"] if step else 0.0,
                 "" if not step else
                 ("  [clears null]" if step["clears_null"] else "  [in null]")))

    print("\nthe E68 premise: is 5->6 now dearer than 4->5?")
    a, b = ship_marg.get(5), ship_marg.get(6)
    if a and b:
        print("  4->5 %+8.3f ms   5->6 %+8.3f ms   inverted=%s"
              % (a["step_ms"], b["step_ms"], b["step_ms"] > a["step_ms"]))
        print("  null bar 4->5 %.3f ms, 5->6 %.3f ms"
              % (a["null_bar_s"] * 1e3, b["null_bar_s"] * 1e3))

    print("\nt789 against shipped")
    for w, row in sorted(payload["t789_vs_shipped"].items()):
        print("  M=%-2d  %+6.2f%%  (null bar %.3f ms, delta %.3f ms) %s"
              % (w, row["pct"], row["null_bar_s"] * 1e3,
                 row["delta_s"] * 1e3,
                 "clears" if row["clears_null"] else "IN NULL"))

    for key in sorted(fits):
        f = fits[key]
        if "marginal" not in f:
            print("\npbfit V=%s: %s" % (key, f.get("error")))
            continue
        print("\npbfit marginal vector, V=%s s (total held at %.4f)"
              % (key, f["total"]))
        print("  " + ", ".join("%.6f" % v for v in f["marginal"]))
        r = fits_ranked[key]
        if "marginal" in r:
            print("  ranked-flattened (width term / %.3f)" % args.rank_flatten)
            print("  " + ", ".join("%.6f" % v for v in r["marginal"]))
        names = [n for n in ("ship", "pb5", "pb7", "pbfit", "pbfit_ranked")
                 if n in walks[key][sorted(walks[key])[0]]]
        for p, w in sorted(walks[key].items()):
            print("  p=%s  " % p + "   ".join(
                "%s d%d/w%d %+.0f%%"
                % (name, w[name]["depth"], w[name]["verify_width"],
                   w[name]["steps"][-1]["headroom_pct"])
                for name in names))
        print("  crossing p, the acceptance that first buys each depth")
        for cap_key, table in sorted(crossings[key].items()):
            for name in names:
                cells = table[name]
                print("    %-6s %-13s " % (cap_key, name) + " ".join(
                    "d%d=%s" % (d + 1,
                                "----" if cells["depth_%d" % (d + 1)] is None
                                else "%.4f" % cells["depth_%d" % (d + 1)])
                    for d in range(args.max_depth)
                    if "depth_%d" % (d + 1) in cells))

    if args.out:
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
