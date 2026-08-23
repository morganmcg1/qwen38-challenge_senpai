#!/usr/bin/env python3
"""E150 R0: reprice every E128 arm on the measured per-width cost curve.

`harness=local instrument`. Zero GPU. Zero Swift.

E128 priced 36 depth-policy arms against a cost curve it inverted from a
ranked receipt pair. E145 R3 then measured a per-width round cost curve from
a live decode with the depth pinned. The two curves disagree, and E145 R7
showed the disagreement changes signs, not only magnitudes: `rankedprice`
moves from -3.1234 to +0.1338 and the decision oracle moves from +8.9390 to
+6.3508.

This rung asks whether any E128 arm survives that reprice. It runs all 36
names in `e128_price.ARMS` through the adapter E145 R7-5 exercised, on both
curves, in one process against one cache, so the two columns differ in the
cost curve and in nothing else.

State the reason this rung may be empty before running it. `argmax_full` on
the shipped information state is, by construction, the optimal policy for a
rule that reads `positionAcceptEMA` as a per-position marginal, and it lands
at exactly the shipped +0.1338. No arm that shares that reading can beat it.
The arms that can are the ones that read the EMA as something else.

Usage:
  python3 e150_r0.py --json e150-artifacts/r0.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
import e150_lib  # noqa: E402
from e150_lib import (  # noqa: E402
    R7_SHIPPED_PCT, assert_price_ordering, build_env, score_arm,
    write_artifact,
)

JENSEN = HERE / "e128-artifacts" / "jensen-and-sign.json"
E128_PUBLISHED = HERE / "e128-artifacts" / "rung2-ours-pricing.json"

# Arms that read the realised capability of the round they are choosing for.
# They are ceilings, not candidates.
NOT_IMPLEMENTABLE = {"oracle"}


def pooled_level():
    rows = {row["prompt_id"]: row
            for row in json.loads(JENSEN.read_text())["hypothesis_j"]}
    return e128_price.pooled_level(
        rows, e128_price.RANKED_PROMPTS["beagle"]["fixture"])


def e128_walker(arm: str, level: dict):
    """The four-line adapter `research/e145-result.md` publishes."""
    policy = e128_price.make_policy(arm, level=level)

    def chooser(ema, margin, offer, adjust, ctx, force, price):
        return policy(ema, margin, offer, ctx["capability"])

    return lambda seed, prompt, entry: chooser


def price_column(env, curve_name: str, curve, price, level) -> dict:
    """Every E128 arm on one curve, with the ordering trap asserted first."""
    assert_price_ordering(curve, price)
    print("  ordering assertion held for the %s curve" % curve_name)
    out = {}
    for arm in e128_price.ARMS:
        row = score_arm(env, curve_name, curve, price, e128_walker(arm, level))
        out[arm] = row
        print("    %-22s %+9.4f  depth %.4f  width1 %.4f"
              % (arm, row["median_pct_mean"], row["weighted_mean_depth"],
                 row["width_histogram"]["1"]))
    return out


def rankings_preserved(left: dict, right: dict, names) -> dict:
    """Fraction of ordered arm pairs whose ranking survives the reprice."""
    pairs = list(itertools.combinations(names, 2))
    kept = 0
    flipped = []
    for a, b in pairs:
        da, db = left[a] - left[b], right[a] - right[b]
        if da == 0.0 or db == 0.0:
            continue
        if (da > 0) == (db > 0):
            kept += 1
        else:
            flipped.append({"a": a, "b": b,
                            "left_delta": da, "right_delta": db})
    flipped.sort(key=lambda row: -abs(row["left_delta"] - row["right_delta"]))
    return {"pairs": len(pairs), "kept": kept,
            "fraction": kept / len(pairs) if pairs else 0.0,
            "largest_flips": flipped[:12]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--json", type=pathlib.Path,
                    default=e150_lib.ARTIFACTS / "r0.json")
    args = ap.parse_args()

    print("harness=local instrument  E150 R0  zero GPU  zero Swift")
    env = build_env(windows=args.windows, seeds=args.seeds)
    print("  attachment gate %s" % env.gate)
    print("  admissible widths %s" % sorted(env.admitted))
    level = pooled_level()

    print("\n## the replayed curve")
    replayed = price_column(env, "replayed", env.replayed, env.replayed_price,
                            level)
    print("\n## the measured curve")
    measured = price_column(env, "measured", env.measured, env.measured_price,
                            level)

    published = json.loads(E128_PUBLISHED.read_text())[
        "median_gain_pct_vs_ship"]

    table = []
    for arm in e128_price.ARMS:
        table.append({
            "arm": arm,
            "implementable": arm not in NOT_IMPLEMENTABLE,
            "e128_published_replayed_pct": published.get(arm),
            "replayed_pct": replayed[arm]["median_pct_mean"],
            "measured_pct": measured[arm]["median_pct_mean"],
            "measured_pct_sd": measured[arm]["median_pct_sd"],
            "delta_measured_minus_replayed_pp": (
                measured[arm]["median_pct_mean"]
                - replayed[arm]["median_pct_mean"]),
            "measured_mean_depth": measured[arm]["weighted_mean_depth"],
            "measured_width1_share": measured[arm]["width_histogram"]["1"],
            "measured_accept_rate": measured[arm]["weighted_accept_rate"],
        })
    table.sort(key=lambda row: -row["measured_pct"])

    names = [a for a in e128_price.ARMS if a in published]
    left = {a: published[a] for a in names}
    mid = {a: replayed[a]["median_pct_mean"] for a in names}
    right = {a: measured[a]["median_pct_mean"] for a in names}
    preserved_published = rankings_preserved(left, right, names)
    preserved_local = rankings_preserved(mid, right, names)

    candidates = [row for row in table if row["implementable"]]
    best = max(candidates, key=lambda row: row["measured_pct"])
    beats = [row["arm"] for row in candidates
             if row["measured_pct"] > R7_SHIPPED_PCT]

    out = dict(env.identity())
    out.update({
        "rung": "E150 R0",
        "what": "every e128_price.ARMS name repriced on the measured "
                "per-width round cost curve",
        "e150_e128_reprice_table": table,
        "e150_e128_best_measured_arm": best["arm"],
        "e150_e128_best_measured_pct": best["measured_pct"],
        "e150_e128_shipped_reference_pct": R7_SHIPPED_PCT,
        "e150_e128_arms_beating_shipped": beats,
        "e150_e128_rankings_preserved_frac": preserved_published["fraction"],
        "e150_e128_rankings_preserved_vs_local_replay_frac":
            preserved_local["fraction"],
        "e150_e128_rankings_published": preserved_published,
        "e150_e128_rankings_local": preserved_local,
        "e150_e128_arms_priced": len(e128_price.ARMS),
        "arms_replayed": replayed,
        "arms_measured": measured,
    })
    path = write_artifact(args.json.name, out)

    print("\n## R0  the reprice, sorted by measured-curve median percent")
    print("  %-22s %10s %10s %10s %8s %8s"
          % ("arm", "e128 pub", "replayed", "measured", "depth", "width1"))
    for row in table:
        pub = ("%+10.4f" % row["e128_published_replayed_pct"]
               if row["e128_published_replayed_pct"] is not None else
               "%10s" % "-")
        print("  %-22s %s %+10.4f %+10.4f %8.4f %8.4f"
              % (row["arm"], pub, row["replayed_pct"], row["measured_pct"],
                 row["measured_mean_depth"], row["measured_width1_share"]))
    print("\n  shipped reference on the measured curve   %+8.4f"
          % R7_SHIPPED_PCT)
    print("  best implementable measured arm           %s %+8.4f"
          % (best["arm"], best["measured_pct"]))
    print("  arms beating the shipped reference        %s" % (beats or "none"))
    print("  rankings preserved vs E128 published      %.4f  (%d/%d pairs)"
          % (preserved_published["fraction"], preserved_published["kept"],
             preserved_published["pairs"]))
    print("  rankings preserved vs this run's replay   %.4f  (%d/%d pairs)"
          % (preserved_local["fraction"], preserved_local["kept"],
             preserved_local["pairs"]))
    print("\nwrote %s" % path.relative_to(HERE.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
