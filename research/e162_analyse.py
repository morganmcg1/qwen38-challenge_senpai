#!/usr/bin/env python3
"""E162: reduce the gated ABBA session to the numbers the assignment asks for.

The contrast is priced the way the advisor set in E162 feedback 3:

    price = prefill_delta_pct * 0.104 + decode_delta_pct * 0.896

0.104 is the RANKED prefill share, measured by the advisor from receipt
5a9f130a. It is deliberately not the local share, which is far larger on this
host because gen 16 has no NAX kernel to accelerate the prefill GEMMs.

Significance uses an exact permutation test over all C(8,4) = 70 arm
assignments rather than a t-test, because four observations per arm do not
support a normal approximation and the design is a fixed counterbalanced
schedule.
"""

from __future__ import annotations

import csv
import itertools
import json
import pathlib
import statistics
import sys

LEGS = pathlib.Path("research/out/e162/abba/legs.tsv")
PREFILL_WEIGHT = 0.104
DECODE_WEIGHT = 0.896
# Ranked per-leg figures from receipt 5a9f130a, quoted by the advisor in F4.
RANKED_PREFILL_SECONDS_PER_LEG = 0.527
RANKED_MTP_SPT = 0.01069609
RANKED_TOKENS = 512


def exact_permutation_p(values: list[float], arms: list[str]) -> float:
    """Two-sided p over every way to split 8 legs into two arms of 4."""
    observed = abs(
        statistics.fmean([v for v, a in zip(values, arms) if a == "C"])
        - statistics.fmean([v for v, a in zip(values, arms) if a == "P"])
    )
    n = len(values)
    hits = total = 0
    for combo in itertools.combinations(range(n), n // 2):
        left = [values[i] for i in combo]
        right = [values[i] for i in range(n) if i not in combo]
        total += 1
        if abs(statistics.fmean(left) - statistics.fmean(right)) >= observed - 1e-15:
            hits += 1
    return hits / total


def main() -> int:
    rows = list(csv.DictReader(LEGS.open(), delimiter="\t"))
    if not rows:
        raise SystemExit(f"no legs in {LEGS}")

    for row in rows:
        witness = int(row["witness"])
        if row["arm"] == "P" and witness != 0:
            raise SystemExit(f"leg {row['leg']} labelled P but witness={witness}")
        if row["arm"] == "C" and witness < 1:
            raise SystemExit(f"leg {row['leg']} labelled C but witness={witness}")

    arms = [r["arm"] for r in rows]
    report: dict = {"leg_count": len(rows), "fields": {}}

    for field in ("prefill_s", "decode_only_spt", "spt", "decode_s"):
        values = [float(r[field]) for r in rows]
        base = [v for v, a in zip(values, arms) if a == "P"]
        cand = [v for v, a in zip(values, arms) if a == "C"]
        base_mean, cand_mean = statistics.fmean(base), statistics.fmean(cand)
        report["fields"][field] = {
            "base_mean": base_mean,
            "cand_mean": cand_mean,
            "base_rel_sd_pct": statistics.stdev(base) / base_mean * 100,
            "cand_rel_sd_pct": statistics.stdev(cand) / cand_mean * 100,
            "delta_pct": (cand_mean - base_mean) / base_mean * 100,
            # Complete separation of two arms of four is the strongest
            # non-parametric statement this design can make.
            "arms_fully_separated": max(cand) < min(base) or min(cand) > max(base),
            "exact_permutation_p": exact_permutation_p(values, arms),
        }

    prefill_pct = report["fields"]["prefill_s"]["delta_pct"]
    decode_pct = report["fields"]["decode_only_spt"]["delta_pct"]
    base_prefill = report["fields"]["prefill_s"]["base_mean"]
    base_decode_s = report["fields"]["decode_s"]["base_mean"]

    report["e162_prefill_pct"] = prefill_pct
    report["e162_decode_pct"] = decode_pct
    report["e162_priced_pct"] = prefill_pct * PREFILL_WEIGHT + decode_pct * DECODE_WEIGHT
    report["local_prefill_share_pct"] = base_prefill / base_decode_s * 100
    report["ranked_prefill_share_pct"] = PREFILL_WEIGHT * 100
    report["local_over_ranked_prefill_share"] = (
        base_prefill / base_decode_s / PREFILL_WEIGHT
    )
    report["local_prefill_seconds"] = base_prefill
    report["ranked_prefill_seconds"] = RANKED_PREFILL_SECONDS_PER_LEG
    report["prefill_slower_here_than_ranked_x"] = (
        base_prefill / RANKED_PREFILL_SECONDS_PER_LEG
    )

    # The ceiling this mechanism can reach on the ranked runner even if the
    # same relative prefill improvement transferred perfectly to the NAX
    # kernel: the whole prefill phase is only 0.527 s of a 5.476 s leg.
    ranked_leg = RANKED_MTP_SPT * RANKED_TOKENS
    saved = RANKED_PREFILL_SECONDS_PER_LEG * (-prefill_pct / 100)
    report["ranked_leg_seconds"] = ranked_leg
    report["ranked_best_case_leg_gain_pct"] = saved / ranked_leg * 100
    report["ranked_best_case_published_gain_pct"] = (
        ranked_leg / (ranked_leg - saved) - 1
    ) * 100

    counters = {k: sorted({r[k] for r in rows}) for k in
                ("edl", "accepted", "rounds", "matched")}
    report["counters"] = counters
    report["counters_unchanged"] = all(len(v) == 1 for k, v in counters.items()
                                       if k != "matched")
    report["all_legs_matched"] = counters["matched"] == ["true"]
    entry = [float(r["entry_c"]) for r in rows]
    report["entry_temp_min_c"] = min(entry)
    report["entry_temp_max_c"] = max(entry)
    report["entry_temp_spread_c"] = max(entry) - min(entry)
    report["exit_temp_mean_c"] = statistics.fmean(float(r["exit_c"]) for r in rows)

    print(json.dumps(report, indent=2))
    out = pathlib.Path("research/out/e162/abba/analysis.json")
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
