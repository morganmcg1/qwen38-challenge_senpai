#!/usr/bin/env python3
"""Print the E62 rung 1 headline table from the three reduced artifacts."""

from __future__ import annotations

import json
import pathlib

ART = pathlib.Path("research/e62-artifacts")


def main() -> int:
    roof = json.loads((ART / "e62-roofline.json").read_text())
    session = json.loads((ART / "e62-r1ops.json").read_text())
    census = json.loads((ART / "e62-census.json").read_text())

    def candidate_commits(mb: int, ops: int) -> float:
        leg = next(l for l in census["legs"] if l["mb"] == mb and l["ops"] == ops)
        phases = leg["phases"]
        return (phases["draft_head"]["commits"]
                + phases["target_verify"]["commits"]) / 10.0

    ceiling = roof["constants"]["ceiling_b_s_per_token_per_commit_per_round"]
    ship_mean = session["arm_means"]["ship"]["mean"]
    terms = session["regression"]["terms"]

    print("=== CEILING, ledger 199(A) ===")
    print(f"  c_max = {roof['constants']['ceiling_per_commit_cost_s']*1e6:.2f} us/commit"
          f"   b_max = {ceiling:.3e} s/token per pooled commit/round")

    print("\n=== MEASURED PER-COMMIT COST ===")
    blocks = [("full ladder", roof["full_ladder"]),
              ("low segment, >= ship commits", roof["low_segment_above_shipped_commits"]),
              ("high segment, <= ship commits", roof["high_segment_below_shipped_commits"])]
    for name, block in blocks:
        cost = block["per_commit_cost_seconds"] * 1e6
        low, high = [x * 1e6 for x in block["per_commit_cost_ci95_seconds"]]
        print(f"  {name:<30} c={cost:>7.2f} us  CI[{low:>7.2f},{high:>7.2f}]"
              f"  t={block['t']:>6.2f}  {100*block['fraction_of_ceiling']:>5.1f}% of ceiling")

    full = roof["full_ladder"]
    print("\n=== POWER, advisor request B ===")
    print(f"  residual sd            {full['residual_sd_percent']:.4f} % of mean")
    print(f"  95 % CI half-width     {full['ci95_halfwidth']:.3e}"
          f" = {100*full['ci95_halfwidth_as_fraction_of_ceiling']:.1f} % of ceiling")
    print(f"  detectable @80 % power {full['detectable_effect_b_80pct_power']:.3e}"
          f" = {100*full['detectable_effect_as_fraction_of_ceiling']:.1f} % of ceiling")

    print("\n=== LADDER ===")
    rows = [("ops6", 4096, 6), ("ops12", 4096, 12), ("ops25", 4096, 25),
            ("ship", 512, 50), ("null", 4096, 50),
            ("ops100", 4096, 100), ("ops200", 4096, 200)]
    print(f"{'arm':>8}{'cand cmt/rnd':>14}{'vs ship %':>11}{'95 % CI':>22}{'t':>8}")
    for arm, mb, ops in rows:
        commits = candidate_commits(mb, ops)
        if arm == "ship":
            print(f"{arm:>8}{commits:>14.1f}{0.0:>11.3f}{'(reference)':>22}{'':>8}")
            continue
        term = terms[f"arm[{arm}]"]
        print(f"{arm:>8}{commits:>14.1f}"
              f"{100*term['estimate']/ship_mean:>11.3f}"
              f"   [{term['percent_ci95_low']:>+7.3f},{term['percent_ci95_high']:>+7.3f}]"
              f"{term['t']:>8.2f}")

    print("\n=== NULL CONTROL and DRIFT ===")
    spreads = session["same_arm_spreads_percent"]
    print(f"  null vs ship {100*terms['arm[null]']['estimate']/ship_mean:+.3f} %"
          f"  t={terms['arm[null]']['t']:+.2f}")
    for sep in sorted(spreads, key=int):
        print(f"  same-arm spread at separation {sep:>2}: "
              f"{spreads[sep]['max_percent']:.4f} %")
    print(f"  entry temperature spread {session['entry_temperature_spread_c']:.3f} C")
    print(f"  leg_position slope {session['regression']['terms']['leg_position']['estimate']:+.3e}"
          f" s/token per leg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
