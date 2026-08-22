#!/usr/bin/env python3
"""Calibrate the ranked instrument on receipts that share one draft schedule.

    usage: research/e130_schedule_family.py [--out PATH]

`research/board_noise_identification.py` could not identify the candidate leg
from below, because grouping by `effective_mean_draft_len` mixes fast and slow
trees whenever the campaign changes candidate speed at a fixed schedule.

This narrows that. It selects only the receipts whose eight-tuple of
`effective_mean_draft_len` matches the current frontier tree to twelve decimal
places, and reports every pairwise candidate-leg delta inside that family next
to the published score gap. A family member that differs in published score but
not in candidate decode time did not change the tree; it drew a different
serial leg.

harness=ranked.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import statistics as st

BOARD = pathlib.Path(os.environ.get("YUKON_BOARD_JSON",
                                    "/tmp/yukon-board/full.json"))

PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
NAMES = sorted(PROMPT_NAMES.values())

FRONTIER_SIGNATURE = (
    4.381818181818, 6.148148148148, 2.297619047619, 5.086956521739,
    5.255555555556, 0.154004106776, 4.989247311828, 2.655660377358,
)


def per_prompt(row: dict) -> dict[str, dict] | None:
    entries = (row.get("officialMetrics") or {}).get("per_prompt")
    if not entries or len(entries) != 8 or row.get("officialScore") is None:
        return None
    out = {}
    for entry in entries:
        name = PROMPT_NAMES.get(str(entry.get("prompt_sha256", ""))[:8])
        if name:
            out[name] = entry
    return out if len(out) == 8 else None


def signature(pp: dict[str, dict]) -> tuple:
    return tuple(round(pp[n]["effective_mean_draft_len"], 12) for n in NAMES)


def rel_delta(a: dict, b: dict, key: str) -> list[float]:
    return [100.0 * (a[n][key] - b[n][key]) / b[n][key] for n in NAMES]


TIER_TOLERANCE = 0.001


def speed_tiers(family: list[tuple[dict, dict]]) -> list[list]:
    """Group receipts whose mean candidate decode time agrees within tolerance.

    Sorting by mean candidate decode time and cutting wherever the relative
    step exceeds the tolerance separates distinct trees, because the campaign's
    real steps are far larger than the instrument's repeat spread.
    """
    ordered = sorted(
        family,
        key=lambda t: st.mean([t[1][n]["mtp_seconds_per_token_mean"]
                               for n in NAMES]))
    tiers: list[list] = []
    for item in ordered:
        value = st.mean([item[1][n]["mtp_seconds_per_token_mean"]
                         for n in NAMES])
        if tiers:
            last = st.mean([tiers[-1][-1][1][n]["mtp_seconds_per_token_mean"]
                            for n in NAMES])
            if (value - last) / last <= TIER_TOLERANCE:
                tiers[-1].append(item)
                continue
        tiers.append([item])
    return tiers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rows = json.loads(BOARD.read_text())
    family = []
    for row in rows:
        pp = per_prompt(row)
        if pp and signature(pp) == FRONTIER_SIGNATURE:
            family.append((row, pp))
    family.sort(key=lambda t: -float(t[0]["officialScore"]))

    print("harness=ranked")
    print("receipts sharing the frontier draft schedule to 1e-12: %d\n"
          % len(family))
    header = ("%-9s %-15s %12s %12s %12s  %s"
              % ("id", "solver", "score", "cand_mean", "serial_mean",
                 "created"))
    print(header)
    print("-" * len(header))
    members = []
    for row, pp in family:
        cand = st.mean([pp[n]["mtp_seconds_per_token_mean"] for n in NAMES])
        serial = st.mean([pp[n]["serial_seconds_per_token_mean"]
                          for n in NAMES])
        members.append({
            "id": row["id"], "solver": row.get("solverUsername"),
            "score": float(row["officialScore"]),
            "status": row.get("status"),
            "promotion": row.get("promotionStatus"),
            "candidate_mean_spt": cand, "serial_mean_spt": serial,
            "created": row["createdAt"],
        })
        print("%-9s %-15s %12.8f %12.8f %12.8f  %s"
              % (row["id"][:8], row.get("solverUsername"),
                 float(row["officialScore"]), cand, serial,
                 row["createdAt"][:19]))

    print("\npairwise, newer minus older. "
          "positive candidate delta means SLOWER.\n")
    header = ("%-9s %-9s %9s %8s %8s %9s %8s %12s"
              % ("a", "b", "cand %", "sd", "se", "serial %", "sd", "score gap"))
    print(header)
    print("-" * len(header))
    pairs = []
    for (ra, pa), (rb, pb) in itertools.combinations(family, 2):
        cand = rel_delta(pa, pb, "mtp_seconds_per_token_mean")
        serial = rel_delta(pa, pb, "serial_seconds_per_token_mean")
        gap = float(ra["officialScore"]) - float(rb["officialScore"])
        pairs.append({
            "a": ra["id"], "b": rb["id"],
            "candidate_mean_delta_pct": st.mean(cand),
            "candidate_sd_pct": st.stdev(cand),
            "candidate_se_pct": st.stdev(cand) / math.sqrt(8.0),
            "serial_mean_delta_pct": st.mean(serial),
            "serial_sd_pct": st.stdev(serial),
            "score_gap": gap,
        })
        print("%-9s %-9s %+9.4f %8.4f %8.4f %+9.4f %8.4f %+12.8f"
              % (ra["id"][:8], rb["id"][:8], st.mean(cand), st.stdev(cand),
                 st.stdev(cand) / math.sqrt(8.0), st.mean(serial),
                 st.stdev(serial), gap))

    cand_sds = [p["candidate_sd_pct"] for p in pairs]
    cand_ses = [p["candidate_se_pct"] for p in pairs]
    print("\nCANDIDATE-LEG PAIRED NOISE INSIDE ONE SCHEDULE FAMILY")
    print("  per-prompt sd of the paired delta: min %.4f %%  median %.4f %%  "
          "max %.4f %%" % (min(cand_sds), st.median(cand_sds), max(cand_sds)))
    print("  se of the eight-prompt mean:       min %.4f %%  median %.4f %%  "
          "max %.4f %%" % (min(cand_ses), st.median(cand_ses), max(cand_ses)))
    print("  Pooled over the whole family this is still not pure noise: the")
    print("  family holds several different trees that share one schedule.")

    tiers = speed_tiers(family)
    print("\nSPEED TIERS INSIDE THE FAMILY")
    print("  Receipts are grouped when their mean candidate decode time")
    print("  agrees within %.2f %%. A tier is one tree measured repeatedly."
          % (100.0 * TIER_TOLERANCE))
    tier_report = []
    for rank, tier in enumerate(tiers):
        if len(tier) < 2:
            continue
        cand = [st.mean([pp[n]["mtp_seconds_per_token_mean"] for n in NAMES])
                for _, pp in tier]
        scores = [float(row["officialScore"]) for row, _ in tier]
        cand_rel_sd = 100.0 * st.stdev(cand) / st.mean(cand)
        score_rel_sd = 100.0 * st.stdev(scores) / st.mean(scores)
        entry = {
            "rank": rank,
            "members": [row["id"] for row, _ in tier],
            "solvers": sorted({row.get("solverUsername") for row, _ in tier}),
            "n": len(tier),
            "candidate_mean_spt": st.mean(cand),
            "candidate_rel_sd_pct": cand_rel_sd,
            "score_mean": st.mean(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "score_rel_sd_pct": score_rel_sd,
            "published_median_noise_amplification": (
                score_rel_sd / cand_rel_sd if cand_rel_sd else None),
        }
        tier_report.append(entry)
        print("\n  tier %d  n=%d  solvers %s"
              % (rank, len(tier), ", ".join(entry["solvers"])))
        print("    members            %s"
              % " ".join(i[:8] for i in entry["members"]))
        print("    mean candidate spt %.8f   rel sd %.4f %%"
              % (entry["candidate_mean_spt"], cand_rel_sd))
        print("    published score    %.8f to %.8f   rel sd %.4f %%"
              % (min(scores), max(scores), score_rel_sd))
        if cand_rel_sd:
            print("    the published median carries %.1fx the noise of mean"
                  % (score_rel_sd / cand_rel_sd))
            print("    candidate decode time on the same tree.")

    report = {
        "harness": "ranked",
        "frontier_signature": list(FRONTIER_SIGNATURE),
        "members": members,
        "pairs": pairs,
        "candidate_paired_sd_pct": {
            "min": min(cand_sds), "median": st.median(cand_sds),
            "max": max(cand_sds),
        },
        "candidate_paired_se_pct": {
            "min": min(cand_ses), "median": st.median(cand_ses),
            "max": max(cand_ses),
        },
        "speed_tiers": tier_report,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
