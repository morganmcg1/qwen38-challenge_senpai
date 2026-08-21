#!/usr/bin/env python3
"""NA weights for kernel arms, at the LOCAL and at the PUBLISHED operating point.

Every NA-dependent isolated kernel measurement in this campaign is reduced to
one headline by weighting NA = 2/3/4/5. The standing rule uses

    0.024 / 0.275 / 0.667 / 0.034

and those four numbers are the LOCAL public fixture's realised verify-width
histogram, mapped through the shipped partition table and weighted by the local
one-group streaming rate. `local_weights()` reproduces them exactly, which is
the proof of provenance. The published score is not measured at that operating
point: it is set by beagle and by whichever of essays / medicine / republic /
botany is lowest, and those prompts run 1 to 2 verify rows narrower.

    harness=local      the width distribution comes from our own fixture
    harness=ranked     the width distribution comes from the official board

The `rate(NA)` table is always `harness=local`: it is measured on our M4 Pro,
and it converts a group count into a share of streaming time. A weight vector
built from a ranked width distribution and this rate table therefore re-weights
a LOCAL probe measurement to the RANKED operating point. It is not a ranked
prediction: the Finding 13 / Finding 35 transfer multiplier still applies on
top, exactly as it does to a standing-weighted number.

Usage:

    python3 research/scoring_weights.py                  # the weight vectors
    python3 research/scoring_weights.py --arms           # every arm re-ranked
    python3 research/scoring_weights_selftest.py         # before trusting it
"""

from __future__ import annotations

import argparse
import json

# --- the one place the partition table lives -------------------------------
#
# Verified at `Vendor/mlx-swift/Source/Cmlx/mlx-generated/metal/quantized.h`
# lines 1918-1979 on the current base (Finding 45). `PARTITION[M]` lists the
# rows-per-group of each concurrent group the wide QMV launches for verify
# width M, so `len(PARTITION[M])` is G and each entry is one NA cell.
#
# A future edit to that switch is a one-line edit here. `M = 1` is the narrow
# single-row QMV: it is a different kernel family and carries no NA=2..5 cell,
# so it contributes no weight and is kept explicit rather than implied.
PARTITION_SOURCE = "quantized.h:1918-1979 (Finding 45), current base"
PARTITION: dict[int, list[int]] = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [3, 3],
    7: [4, 3],
    8: [4, 4],
    9: [3, 3, 3],
}

# The partition table as it stood BEFORE E100 moved the group boundary. Kept
# only so the self-test can prove the live table is not the stale one.
PARTITION_PRE_E100: dict[int, list[int]] = {**PARTITION, 5: [3, 2]}

# One-group streaming rate per NA, local M4 Pro, from Edward's E106 per-width
# refit of the streaming law (R^2 >= 0.99948). Time per group is proportional
# to 1 / rate, which is what turns a group count into a share of round time.
ONE_GROUP_GBPS: dict[int, float] = {2: 253.6, 3: 245.6, 4: 211.7, 5: 178.8}

# The same table for the ranked M5 host, from E113's route-A rate table (the
# G=1 entries of `RANKED_RATE`, Finding 31 identity). The ranked host is not a
# scaled copy of ours: an NA=5 group costs 1.51x an NA=2 group there against
# 1.42x locally, so the wide cells carry relatively MORE ranked time than the
# standing local rule gives them. E113 finding 1 also shows the ranked COST
# tier stays at M=5 while the local one moves to M=6. Any table built with this
# rate table is `harness=ranked` on both axes.
RANKED_ONE_GROUP_GBPS: dict[int, float] = {2: 409.8, 3: 368.0, 4: 333.9,
                                           5: 272.2}
RANKED_RATE_SOURCE = "E113 route A, Finding 31 identity, ranked M5 receipts"

NA_CELLS = (2, 3, 4, 5)

# Edward's traced local histogram over verify width M, E106, 19 native MTP
# rounds, W&B run `19kgn6xi`. mean width 6.947, accept 0.9735.
E106_LOCAL_HISTOGRAM: dict[int, float] = {2: 1, 5: 1, 6: 4, 7: 3, 8: 10}

# The standing campaign rule, quoted so the self-test can check we still
# reproduce it rather than trusting that we do.
STANDING_WEIGHTS: dict[int, float] = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# Finding 16, over 81 strong runs: the published median is
# `0.5 * raw_beagle + 0.5 * min(essays, medicine, republic, botany)`.
# beagle holds rank 4 in 100.0 % of them; rank 5 occupancy is the mixture.
MIN_SLOT_OCCUPANCY: dict[str, float] = {
    "essays": 0.667, "medicine": 0.198, "republic": 0.074, "botany": 0.062,
}
BEAGLE_SHARE = 0.5


def na_weights(width_dist: dict[int, float],
               rates: dict[int, float] | None = None,
               time_weighted: bool = True) -> dict[int, float]:
    """Share of wide-QMV streaming time carried by each NA cell.

    `width_dist` maps verify width M to a count or a probability; it is
    normalised here, so either works. With `time_weighted=False` the result is
    the raw share of GROUPS, which is the rate-free sensitivity check.
    """
    rates = rates or ONE_GROUP_GBPS
    acc = {na: 0.0 for na in NA_CELLS}
    for width, mass in width_dist.items():
        if mass == 0:
            continue
        if width not in PARTITION:
            raise KeyError(
                "verify width %r is not in the partition table (%s)"
                % (width, PARTITION_SOURCE))
        for na in PARTITION[width]:
            if na not in acc:
                # NA=1 is the narrow QMV and carries no wide cell. Any other
                # unexpected NA is a stale partition table, not a rounding
                # question, so it must fail rather than be dropped.
                if na == 1:
                    continue
                raise KeyError(
                    "verify width %d maps to NA=%d, which no arm table covers"
                    % (width, na))
            acc[na] += mass / rates[na] if time_weighted else mass
    total = sum(acc.values())
    if total <= 0:
        raise ValueError("width distribution carries no wide-QMV group")
    return {na: acc[na] / total for na in NA_CELLS}


def local_weights(**kwargs) -> dict[int, float]:
    """harness=local. The standing rule, rebuilt from its own source data."""
    return na_weights(E106_LOCAL_HISTOGRAM, **kwargs)


def published_weights(per_prompt: dict[str, dict[int, float]],
                      occupancy: dict[str, float] | None = None,
                      **kwargs) -> dict[int, float]:
    """harness=ranked. `0.5 * beagle + 0.5 * argmin`, per Finding 16.

    `per_prompt` maps a prompt name to its verify-width distribution. The
    argmin slot is a mixture over the four cluster members, weighted by their
    measured rank-5 occupancy, because which one is lowest changes run to run.
    """
    occupancy = occupancy or MIN_SLOT_OCCUPANCY
    missing = [p for p in ["beagle", *occupancy] if p not in per_prompt]
    if missing:
        raise KeyError("no width distribution for %s" % ", ".join(missing))
    beagle = na_weights(per_prompt["beagle"], **kwargs)
    occ_total = sum(occupancy.values())
    mixed = {na: 0.0 for na in NA_CELLS}
    for prompt, share in occupancy.items():
        part = na_weights(per_prompt[prompt], **kwargs)
        for na in NA_CELLS:
            mixed[na] += (share / occ_total) * part[na]
    return {na: BEAGLE_SHARE * beagle[na] + (1 - BEAGLE_SHARE) * mixed[na]
            for na in NA_CELLS}


def weighted(arm_table: dict[int, float], weights: dict[int, float]) -> float:
    """One NA-resolved arm reduced to a single number under `weights`."""
    missing = [na for na in NA_CELLS if na not in arm_table]
    if missing:
        raise KeyError("arm table has no cell for NA=%s"
                       % ", ".join(str(na) for na in missing))
    return sum(weights[na] * arm_table[na] for na in NA_CELLS)


def reweigh(arm_table: dict[int, float],
            published: dict[int, float],
            local: dict[int, float] | None = None) -> dict[str, float]:
    """Both numbers for one arm, plus the movement between them."""
    local = local or local_weights()
    standing = weighted(arm_table, local)
    corrected = weighted(arm_table, published)
    return {
        "standing_pct": standing,
        "published_pct": corrected,
        "delta_pp": corrected - standing,
        "ratio": corrected / standing if standing else float("nan"),
        "sign_change": (standing > 0) != (corrected > 0),
    }


def rerank(arms: dict[str, dict[int, float]],
           published: dict[int, float],
           local: dict[int, float] | None = None) -> list[dict]:
    """Every arm re-ranked, with the rank movement made explicit."""
    local = local or local_weights()
    rows = [dict(arm=name, **reweigh(table, published, local))
            for name, table in arms.items()]
    order_standing = sorted(rows, key=lambda r: r["standing_pct"])
    order_published = sorted(rows, key=lambda r: r["published_pct"])
    for row in rows:
        row["rank_standing"] = order_standing.index(row) + 1
        row["rank_published"] = order_published.index(row) + 1
        row["rank_change"] = row["rank_published"] - row["rank_standing"]
    return sorted(rows, key=lambda r: r["published_pct"])


def _fmt(weights: dict[int, float]) -> str:
    return "  ".join("NA%d %.4f" % (na, weights[na]) for na in NA_CELLS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", help="JSON file of ranked width distributions")
    parser.add_argument("--arms", help="JSON file of NA-resolved arm tables")
    parser.add_argument("--json", help="write the result here")
    args = parser.parse_args()

    out: dict = {
        "partition_source": PARTITION_SOURCE,
        "partition": {str(k): v for k, v in PARTITION.items()},
        "one_group_gbps": {str(k): v for k, v in ONE_GROUP_GBPS.items()},
        "local_weights": local_weights(),
        "local_weights_group_counted": local_weights(time_weighted=False),
        "standing_weights": STANDING_WEIGHTS,
    }
    print("harness=local   %s   (standing rule %s)"
          % (_fmt(out["local_weights"]),
             _fmt({int(k): v for k, v in STANDING_WEIGHTS.items()})))

    if args.widths:
        widths = {p: {int(w): m for w, m in d.items()}
                  for p, d in json.load(open(args.widths)).items()}
        pub = published_weights(widths)
        out["per_prompt_weights"] = {p: na_weights(d) for p, d in widths.items()}
        out["published_weights"] = pub
        print("harness=ranked  %s   (0.5 beagle + 0.5 argmin)" % _fmt(pub))
        if args.arms:
            arms = {a: {int(na): v for na, v in t.items()}
                    for a, t in json.load(open(args.arms)).items()}
            out["arms"] = rerank(arms, pub)
            print("\n%-14s %10s %10s %8s %7s %s"
                  % ("arm", "standing", "published", "delta", "ratio", "rank"))
            for row in out["arms"]:
                print("%-14s %10.3f %10.3f %+8.3f %7.2f  %d -> %d%s"
                      % (row["arm"], row["standing_pct"], row["published_pct"],
                         row["delta_pp"], row["ratio"], row["rank_standing"],
                         row["rank_published"],
                         "  SIGN CHANGE" if row["sign_change"] else ""))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(out, handle, indent=1, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
