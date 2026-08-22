#!/usr/bin/env python3
"""E129 -- price each candidate width table on the RANKED prompt mix.

Two channels move when the table changes, and they move in opposite
directions at the wide widths:

  passes  `ceil(M / IPG)` reads of the whole weight matrix. Weight traffic
          dominates every routed cell, so this is the first-order term.
  regs    the entry point's register ceiling, reported as resident
          simdgroups. F131 measured this channel's coefficient at about
          zero on g16s, so it is reported but never used to price a table.

`harness=ranked`. The mix is the F83 median-sensitivity weighting over the
E114 max-entropy routed width histograms. The local fixture histogram is NOT
used: it puts 0.77 of its mass at M=8, which is where the tables differ most,
and it flips the sign of the comparison.

Zero GPU seconds. Register figures come from `e129_entry_point_census.py`.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e129_entry_point_census as census  # noqa: E402

OUT = pathlib.Path("research/out")
# The ranked verify-width cap is 8, so M=9 never reaches the board.
RANKED_WIDTHS = tuple(range(3, 9))


def passes(plan) -> dict[int, int]:
    return {m: -(-m // ipg) for m, ipg, _ in plan}


def main() -> int:
    hists = census.ranked_histograms()
    weights = {p: w for p, (_, w) in census.RANKED_WIDTH_MIX.items()}
    total = sum(weights.values())
    weights = {p: w / total for p, w in weights.items()}

    base = census.PLANS["shipped"]
    base_passes = passes(base)

    records = []
    print("harness=ranked  price = ranked-weighted passes over the weight matrix")
    print("%-14s %9s %9s %9s   %s" % (
        "table", "passes", "vs base", "traffic", "per-prompt passes"))
    for name, plan in census.PLANS.items():
        pl = passes(plan)
        per_prompt = {}
        for prompt, hist in hists.items():
            per_prompt[prompt] = sum(hist[m] * pl[m] for m in RANKED_WIDTHS)
        mean = sum(weights[p] * per_prompt[p] for p in per_prompt)
        base_mean = sum(
            weights[p] * sum(hists[p][m] * base_passes[m] for m in RANKED_WIDTHS)
            for p in hists)
        cut = mean / base_mean - 1.0
        print("%-14s %9.4f %+8.2f%% %+8.2f%%   %s" % (
            name, mean, cut * 100.0, cut * 100.0,
            "  ".join("%s %.3f" % (p[:4], per_prompt[p]) for p in sorted(per_prompt))))
        records.append({
            "table": name,
            "plan": [list(c) for c in plan],
            "passes_per_width": pl,
            "ranked_weighted_passes": mean,
            "ranked_weighted_passes_base": base_mean,
            "ranked_weight_traffic_delta": cut,
            "per_prompt_passes": per_prompt,
        })

    print()
    print("routed width mass on the ranked mix (max-entropy, renormalised 3..8)")
    print("%-10s %s" % ("prompt", "  ".join("M=%d" % m for m in RANKED_WIDTHS)))
    for prompt in sorted(hists):
        print("%-10s %s" % (
            prompt, "  ".join("%.3f" % hists[prompt][m] for m in RANKED_WIDTHS)))
    pooled = {m: sum(weights[p] * hists[p][m] for p in hists) for m in RANKED_WIDTHS}
    print("%-10s %s" % (
        "POOLED", "  ".join("%.3f" % pooled[m] for m in RANKED_WIDTHS)))

    out = OUT / "e129-pass-price.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "harness": "ranked",
        "official_or_ranked_score": False,
        "instrument": "analytic pass count over the E114 max-entropy routed mix",
        "ranked_widths": list(RANKED_WIDTHS),
        "prompt_weights": weights,
        "routed_histograms": {p: {str(m): v for m, v in h.items()}
                              for p, h in hists.items()},
        "pooled_histogram": {str(m): v for m, v in pooled.items()},
        "tables": records,
    }, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
