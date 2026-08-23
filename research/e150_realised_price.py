#!/usr/bin/env python3
"""E150 R4: price the schedules the Swift rule actually produced.

`harness=local`. Zero GPU. CAMPAIGN RULE 79 is respected: this module reads
`d=` from the per-round trace and never reads `round_us`, so nothing here is a
timing contrast. It is an offline price, exactly like the R0.5/R1 replay.

Why this exists, and why it is stronger than the replay
-------------------------------------------------------
The R0.5 and R1 replays price a SIMULATED schedule: they drive a Python copy of
the rule with a fitted per-position acceptance model. Two things in that chain
can be wrong -- the model of the rule and the model of acceptance.

This module removes both. The sixteen 512-token exactness legs already ran the
real Swift rule on eight real prompts, and every leg recorded the depth it
actually drafted in every round. Pricing those recorded widths on a per-width
cost curve therefore asks one narrow question with no fitting anywhere in it:

    given the widths the two arms really chose, which arm buys 512 tokens for
    fewer round-cost units?

What it still assumes
---------------------
It assumes round cost depends on verify width and nothing else, which is the
same assumption both campaign curves encode. It cannot see a per-round host
overhead that the width curve does not model, and it is not a ranked score. It
is priced on both campaign curves for the same reason the bracket is: the two
disagree on the steps this rule crosses.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e128_price import ranked_round_us  # noqa: E402
from e128_price import PROMPT_NAMES  # noqa: E402
from e150_lib import write_artifact  # noqa: E402

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+)")
# `arm=` is the legacy DEPTH PRICE arm and reads `ship` in both arms of this
# experiment, which does not vary here. The schedule witness is `rule=`.
RULE_RE = re.compile(r"\brule=(\S+)")
ARM_RE = re.compile(r"\barm=(\S+)")

# Advisor F4 section 4.1. Share of board rows at scores >= 3.5 for which each
# prompt family sits in the median pair. A zero-weight family cannot move the
# published median in this score band whatever its per-prompt gain is.
RULE_148_WEIGHTS = {"beagle": 0.5000, "essays": 0.4474,
                    "republic": 0.0329, "medicine": 0.0197}
FIXTURE_FAMILY = {
    "beagle_a": "beagle",
    "beagle_b": "beagle",
    "essays_montaigne": "essays",
    "essays_bacon": "essays",
    "republic_jowett": "republic",
    "medicine_hippoc": "medicine",
    "medicine_hist": "medicine",
}

# The measured per-width round cost this candidate compiles, widths 1..9, in
# microseconds. Same values the Swift table carries.
MEASURED_US = {
    1: 65778.93576562102,
    2: 70905.344451175013,
    3: 76196.540204972174,
    4: 84374.327044333186,
    5: 95820.260836797606,
    6: 124257.1286225723,
    7: 150803.19359188987,
    8: 153965.40820598602,
    9: 157127.62282008218,
}
REPLAYED_US = {w: ranked_round_us(w) for w in range(1, 10)}
CURVES = {"measured": MEASURED_US, "replayed": REPLAYED_US}


def median_of(values):
    """The ranked aggregation: mean of the two middle values of eight."""
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def rule_148_rollup(gain_by_fixture: dict) -> dict:
    """Average fixtures inside a family, then weight the families."""
    families = collections.defaultdict(list)
    for fixture, gain in gain_by_fixture.items():
        family = FIXTURE_FAMILY.get(fixture)
        if family:
            families[family].append(gain)
    by_family = {name: statistics.fmean(v) for name, v in families.items()}
    total = sum(RULE_148_WEIGHTS[name] for name in by_family)
    weighted = sum(RULE_148_WEIGHTS[name] * gain
                   for name, gain in by_family.items())
    return {
        "family_gain_pp": by_family,
        "weights_used": {name: RULE_148_WEIGHTS[name] for name in by_family},
        "weight_total": total,
        "e150_policy_gain_rule148_weighted_pp": weighted / total if total
        else float("nan"),
        "carrying_families_negative": sorted(
            name for name, gain in by_family.items() if gain < 0.0),
        "unweighted_all_fixtures_mean_pp":
            statistics.fmean(gain_by_fixture.values()),
    }


def read_leg(leg_dir):
    depths = collections.Counter()
    arm = rule = None
    trace = os.path.join(leg_dir, "trace.txt")
    if not os.path.exists(trace):
        return None
    for line in open(trace):
        match = ROUND_RE.match(line)
        if not match:
            continue
        depths[int(match.group(2))] += 1
        hit = ARM_RE.search(line)
        if hit:
            arm = hit.group(1)
        hit = RULE_RE.search(line)
        if hit:
            rule = hit.group(1)
    report = json.load(open(os.path.join(leg_dir, "report.json")))
    return {
        "arm": arm,
        "rule": rule,
        "depths": depths,
        "rounds": sum(depths.values()),
        "decode_tokens": report["decode_token_count"],
        "all_tokens_matched": bool(report["all_tokens_matched"]),
        "report_round_count": report["round_count"],
        "effective_mean_draft_len": report["effective_mean_draft_len"],
        "non_drafting_round_count": report["non_drafting_round_count"],
    }


def read_arm(runs_dir):
    legs = {}
    for prompt_id in sorted(os.listdir(runs_dir)):
        leg_dir = os.path.join(runs_dir, prompt_id)
        if not os.path.isdir(leg_dir):
            continue
        leg = read_leg(leg_dir)
        if leg:
            legs[prompt_id] = leg
    return legs


def price(leg, curve):
    """Round-cost units bought per emitted token, normalised to width 1."""
    total = sum(count * curve[depth + 1] for depth, count in
                leg["depths"].items())
    return (total / leg["decode_tokens"]) / curve[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--linearised", required=True)
    ap.add_argument("--json", default="r4_realised_price.json")
    args = ap.parse_args()

    ship = read_arm(args.shipped)
    lin = read_arm(args.linearised)
    prompts = sorted(set(ship) & set(lin))
    if len(prompts) != 8:
        print("WARNING: %d paired legs, expected 8" % len(prompts))

    # A leg whose trace round count disagrees with its report, or whose arm
    # witness is wrong, cannot be priced. Rule 114: witness the arm from the
    # run, never from the variable it was asked with.
    problems = []
    for name, legs, want in (("shipped", ship, "shipped"),
                             ("linearised", lin, "linearised")):
        for prompt, leg in legs.items():
            if leg["rule"] != want:
                problems.append("%s/%s rule witness %r" % (name, prompt,
                                                           leg["rule"]))
            if leg["rounds"] != leg["report_round_count"]:
                problems.append("%s/%s trace %d vs report %d"
                                % (name, prompt, leg["rounds"],
                                   leg["report_round_count"]))
            if not leg["all_tokens_matched"]:
                problems.append("%s/%s did not match" % (name, prompt))

    print("E150 R4 - the price of the schedules the Swift rule really chose")
    print("  harness=local  gpu_used=False  rule_79=not_engaged  "
          "timing_valid=false")
    print("  paired legs %d  problems %d" % (len(prompts), len(problems)))
    for line in problems:
        print("  PROBLEM %s" % line)

    out = {
        "harness": "local",
        "gpu_used": False,
        "timing_valid": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "rung": "E150 R4 realised-schedule price",
        "frame": "offline price of recorded verify widths, both campaign curves",
        "e150_realised_paired_legs": len(prompts),
        "e150_realised_problems": problems,
        "e150_realised_ok": not problems and len(prompts) == 8,
        "curves_us": {"measured": MEASURED_US, "replayed": REPLAYED_US},
        "by_curve": {},
    }

    for curve_name, curve in CURVES.items():
        rows, pooled_ship, pooled_lin, tokens = {}, 0.0, 0.0, 0
        for prompt in prompts:
            ship_price = price(ship[prompt], curve)
            lin_price = price(lin[prompt], curve)
            rows[PROMPT_NAMES.get(prompt, prompt)] = {
                "shipped_units_per_token": ship_price,
                "linearised_units_per_token": lin_price,
                "gain_pct": (ship_price / lin_price - 1.0) * 100.0,
                "shipped_rounds": ship[prompt]["rounds"],
                "linearised_rounds": lin[prompt]["rounds"],
                "shipped_edl": ship[prompt]["effective_mean_draft_len"],
                "linearised_edl": lin[prompt]["effective_mean_draft_len"],
                "edl_delta": (lin[prompt]["effective_mean_draft_len"]
                              - ship[prompt]["effective_mean_draft_len"]),
                "shipped_non_drafting_rounds":
                    ship[prompt]["non_drafting_round_count"],
                "linearised_non_drafting_rounds":
                    lin[prompt]["non_drafting_round_count"],
            }
            pooled_ship += sum(c * curve[d + 1] for d, c
                               in ship[prompt]["depths"].items())
            pooled_lin += sum(c * curve[d + 1] for d, c
                              in lin[prompt]["depths"].items())
            tokens += ship[prompt]["decode_tokens"]
        gains = [r["gain_pct"] for r in rows.values()]
        rollup = rule_148_rollup({n: r["gain_pct"] for n, r in rows.items()})
        summary = {
            "rows": rows,
            "rule_148": rollup,
            "median_gain_pct": median_of(gains),
            "mean_gain_pct": statistics.fmean(gains),
            "min_gain_pct": min(gains),
            "max_gain_pct": max(gains),
            "prompts_improved": sum(1 for g in gains if g > 0.0),
            "pooled_gain_pct": (pooled_ship / pooled_lin - 1.0) * 100.0,
            "pooled_shipped_units_per_token":
                (pooled_ship / tokens) / curve[1],
            "pooled_linearised_units_per_token":
                (pooled_lin / tokens) / curve[1],
        }
        out["by_curve"][curve_name] = summary

        print("\n## priced on the %s curve" % curve_name)
        print("  %-10s %10s %10s %10s %10s %10s"
              % ("prompt", "ship u/t", "lin u/t", "gain %", "ship edl",
                 "lin edl"))
        for name in sorted(rows, key=lambda n: rows[n]["gain_pct"]):
            r = rows[name]
            print("  %-10s %10.5f %10.5f %+10.4f %10.4f %10.4f"
                  % (name, r["shipped_units_per_token"],
                     r["linearised_units_per_token"], r["gain_pct"],
                     r["shipped_edl"], r["linearised_edl"]))
        print("  %-10s %10.5f %10.5f %+10.4f"
              % ("MEDIAN", summary["pooled_shipped_units_per_token"],
                 summary["pooled_linearised_units_per_token"],
                 summary["median_gain_pct"]))
        print("  pooled %+0.4f %%   improved on %d of %d prompts"
              % (summary["pooled_gain_pct"], summary["prompts_improved"],
                 len(gains)))
        print("  Rule 148 family gains: %s"
              % "  ".join("%s %+0.4f" % (n, g)
                          for n, g in sorted(rollup["family_gain_pp"].items())))
        print("  Rule 148 WEIGHTED roll-up %+0.4f pp   negative carriers: %s"
              % (rollup["e150_policy_gain_rule148_weighted_pp"],
                 ", ".join(rollup["carrying_families_negative"]) or "none"))

    out["e150_realised_median_gain_measured_pct"] = (
        out["by_curve"]["measured"]["median_gain_pct"])
    out["e150_realised_median_gain_replayed_pct"] = (
        out["by_curve"]["replayed"]["median_gain_pct"])
    out["e150_realised_median_gain_min_over_curves_pct"] = min(
        out["e150_realised_median_gain_measured_pct"],
        out["e150_realised_median_gain_replayed_pct"])
    out["e150_realised_rule148_weighted_pp_by_curve"] = {
        c: out["by_curve"][c]["rule_148"][
            "e150_policy_gain_rule148_weighted_pp"] for c in CURVES}
    out["e150_realised_rule148_weighted_min_over_curves_pp"] = min(
        out["e150_realised_rule148_weighted_pp_by_curve"].values())

    # Rule 114 landing witness on the legs that actually ran. The ranked host
    # is a different prompt set, so this proves the arm changes behaviour, not
    # what the ranked draft lengths will be.
    edl = out["by_curve"]["measured"]["rows"]
    moved = sum(1 for r in edl.values() if abs(r["edl_delta"]) >= 5e-4)
    out["e150_r4_local_edl_moved_legs"] = moved
    out["e150_r4_local_edl_moved_all"] = moved == len(edl)

    path = write_artifact(args.json, out)
    print("\n## headline")
    print("  median gain, measured curve   %+9.4f %%"
          % out["e150_realised_median_gain_measured_pct"])
    print("  median gain, replayed curve   %+9.4f %%"
          % out["e150_realised_median_gain_replayed_pct"])
    print("  worst of the two              %+9.4f %%"
          % out["e150_realised_median_gain_min_over_curves_pct"])
    print("  local draft length moved on %d of %d legs" % (moved, len(edl)))
    print("\nwrote %s" % path)
    return 0 if out["e150_realised_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
