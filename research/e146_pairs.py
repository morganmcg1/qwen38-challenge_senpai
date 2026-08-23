"""E146 R-A rung 2: the pure-nuisance pair set and what unit the nuisance uses.

Every pair here is a solver-declared zero-delta resample whose T32 schedule
fingerprint is digit-identical to its target. The candidate code is therefore
the same on both sides by declaration and the schedule is the same by
measurement, so the whole pair difference is nuisance.

TWO OBJECTIVE FILTERS on top of the declaration:

  * the eight `head_provenance_sha256` digests must agree, so the proposal head
    is the same tree as well;
  * the pair must be inside one tree generation, defined here as the target and
    the replicate being adjacent draws with no intervening promotion of a
    different schedule for that solver's lineage. The weaker practical form used
    below is a bounded age gap, reported per pair so a reader can re-filter.

The unit question is answered ACROSS lineages, not inside one cluster. A
cluster-internal R2 is circular when the cluster was selected by k. Different
lineages have different candidate speeds, different round counts and different
draft lengths, so the unit that keeps the offset constant across lineages is the
unit the nuisance is really charged in.
"""

import json
import math
import os
import sys

import e146_lib as L
import e146_replicates as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e146-pairs.json")

UNITS = ["drafting_round", "round", "verify_row", "draft_step",
         "emitted_token", "non_drafting_round"]


def pairs(rows):
    built = R.build(rows)
    out = []
    for row in built["declared"]:
        targets = [t for t in R.resolve_targets(row, rows)
                   if t.draft_key() == row.draft_key()]
        if not targets:
            continue
        target = sorted(targets, key=lambda r: r.created)[-1]
        heads_match = all(row.head(p) == target.head(p) for p in L.PROMPT_ORDER)
        fit = L.fit_k(target, row, basis_row=target)
        age_h = (_hours(row.created) - _hours(target.created))
        out.append({
            "replicate": row, "target": target, "fit": fit,
            "heads_match": heads_match, "age_hours": age_h,
        })
    out.sort(key=lambda p: p["fit"]["k_us_per_drafting_round"])
    return out


def _hours(iso):
    from datetime import datetime
    return datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S").timestamp() / 3600.0


def unit_constants(pair):
    """The offset this pair implies under each rival physical unit."""
    target, row = pair["target"], pair["replicate"]
    out = {}
    for unit in UNITS:
        fit = L.fit_on_basis(target, row, unit)
        out[unit] = fit
    out["flat_pct"] = {"k": pair["fit"]["cand_mean8_pct"], "r2": 0.0}
    return out


def main():
    path, rows = L.load()
    ps = pairs(rows)
    print("board %s  scored %d  declared same-schedule pairs %d"
          % (path, len(rows), len(ps)))
    print()
    print("%9s %9s %9s %8s %7s %6s %6s %s" %
          ("k us/dr", "cand8%", "serial8%", "resid", "age_h", "heads", "r2", "pair"))
    for pair in ps:
        fit = pair["fit"]
        serial = L.serial_pct_diff(pair["target"], pair["replicate"])
        s8 = sum(serial.values()) / 8.0
        print("%9.1f %+9.4f %+9.4f %8.4f %7.1f %6s %6.2f %s<-%s %s" %
              (fit["k_us_per_drafting_round"], fit["cand_mean8_pct"], s8,
               fit["residual_sd_pp"], pair["age_hours"],
               "y" if pair["heads_match"] else "N", fit["r2_through_origin"],
               pair["replicate"].id8, pair["target"].id8,
               pair["replicate"].solver[:12]))

    print("\n--- per-prompt shape, pairs above +300 us/dr ---")
    high = [p for p in ps if p["fit"]["k_us_per_drafting_round"] > 300]
    low = [p for p in ps if abs(p["fit"]["k_us_per_drafting_round"]) <= 300]
    header = "%-10s" % "pair" + "".join("%9s" % p[:7] for p in L.PROMPT_ORDER)
    print(header)
    for pair in high:
        obs = pair["fit"]["observed_pct"]
        print("%-10s" % pair["replicate"].id8
              + "".join("%+9.3f" % obs[p] for p in L.PROMPT_ORDER))
    if high:
        print("%-10s" % "mean"
              + "".join("%+9.3f" % (sum(p["fit"]["observed_pct"][q] for p in high)
                                    / len(high)) for q in L.PROMPT_ORDER))
    print("%-10s" % "low mean"
          + "".join("%+9.3f" % (sum(p["fit"]["observed_pct"][q] for p in low)
                                / len(low)) for q in L.PROMPT_ORDER))

    print("\n--- which unit keeps the HIGH offset constant ACROSS lineages ---")
    print("%-20s %12s %12s %10s" % ("unit", "mean", "sd", "cv"))
    unit_rows = {}
    for unit in UNITS + ["flat_pct"]:
        vals = []
        for pair in high:
            vals.append(unit_constants(pair)[unit]["k"])
        m = L.mean(vals)
        s = L.sd(vals)
        unit_rows[unit] = {"mean": m, "sd": s, "cv": abs(s / m) if m else float("nan"),
                           "values": vals}
        print("%-20s %12.4g %12.4g %10.3f" % (unit, m, s, abs(s / m) if m else float("nan")))

    payload = {
        "harness": "ranked",
        "board_path": path,
        "pair_count": len(ps),
        "pairs": [{
            "replicate": p["replicate"].id8,
            "target": p["target"].id8,
            "solver": p["replicate"].solver,
            "created": p["replicate"].created,
            "k_us_per_drafting_round": p["fit"]["k_us_per_drafting_round"],
            "cand_mean8_pct": p["fit"]["cand_mean8_pct"],
            "residual_sd_pp": p["fit"]["residual_sd_pp"],
            "r2_through_origin": p["fit"]["r2_through_origin"],
            "heads_match": p["heads_match"],
            "age_hours": p["age_hours"],
            "observed_pct": p["fit"]["observed_pct"],
            "serial_pct": L.serial_pct_diff(p["target"], p["replicate"]),
        } for p in ps],
        "unit_discrimination_high": {k: {"mean": v["mean"], "sd": v["sd"],
                                         "cv": v["cv"]}
                                     for k, v in unit_rows.items()},
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
