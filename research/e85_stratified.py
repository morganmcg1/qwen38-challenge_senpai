#!/usr/bin/env python3
"""Host-state-stratified arm contrast for an E85 traced session.

    usage: research/e85_stratified.py SESSION_DIR [--json OUT]

`qwen-alphonse` found that some legs carry several times the host cost of an
interior leg. This session is worse than the extreme case he described: the
legs fall into two well separated host states, and the arms did not draw them
in equal proportion. That imbalance, not the treatment, is the largest term in
the raw arm contrast.

The correction splits the legs on host state and reports the arm contrast
inside each stratum. It reports the same contrast on the depth-0 serial leg,
which is unchanged code, as the null. A stratum whose null is larger than its
arm contrast cannot support a claim about the arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from e85_phases import HOST_FIELDS
from e85_round_pairs import parse_rounds, timed_segment

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def split_states(values: list[float]) -> float:
    """Threshold between two host states, at the widest gap in sorted order."""
    ordered = sorted(values)
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    _, index = max(gaps)
    return 0.5 * (ordered[index] + ordered[index + 1])


def contrast(rows: list[dict], field: str) -> dict:
    base = [r[field] for r in rows if r["arm"] == "base"]
    treat = [r[field] for r in rows if r["arm"] != "base"]
    if len(base) < 1 or len(treat) < 1:
        return {"n_base": len(base), "n_treat": len(treat),
                "delta_us_per_token": math.nan}
    delta = (statistics.fmean(treat) - statistics.fmean(base)) * 1e6
    dof = len(base) + len(treat) - 2
    if dof >= 1:
        pooled = math.sqrt(
            ((len(base) - 1) * (statistics.variance(base) if len(base) > 1 else 0.0)
             + (len(treat) - 1)
             * (statistics.variance(treat) if len(treat) > 1 else 0.0)) / dof)
        se = pooled * math.sqrt(1 / len(base) + 1 / len(treat)) * 1e6
    else:
        se = math.nan
    half = T_CRIT_95.get(dof, 1.96) * se if se == se else math.nan
    return {
        "n_base": len(base),
        "n_treat": len(treat),
        "base_mean": statistics.fmean(base),
        "treat_mean": statistics.fmean(treat),
        "delta_us_per_token": delta,
        "se_us_per_token": se,
        "t": delta / se if se else math.nan,
        "ci95_lo_us_per_token": delta - half,
        "ci95_hi_us_per_token": delta + half,
        "pct_of_base": 100.0 * delta / (statistics.fmean(base) * 1e6),
        "mean_position_base": statistics.fmean(
            [r["position"] for r in rows if r["arm"] == "base"]),
        "mean_position_treat": statistics.fmean(
            [r["position"] for r in rows if r["arm"] != "base"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.session)
    with (root / "legs.tsv").open() as handle:
        legs = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for position, row in enumerate(legs):
        leg = int(row["leg"])
        rounds = timed_segment(
            parse_rounds(root / f"leg{leg:02d}-{row['arm']}" / "rounds.txt"))
        rows.append({
            "leg": leg,
            "position": position,
            "arm": row["arm"],
            "host_sum_us": statistics.median(
                sum(r[f] for f in HOST_FIELDS) for r in rounds),
            "mtp": float(row["mtp_s_per_tok"]),
            "serial": float(row["serial_s_per_tok"]),
        })

    threshold = split_states([r["host_sum_us"] for r in rows])
    for r in rows:
        r["state"] = "clean" if r["host_sum_us"] < threshold else "contaminated"

    print(f"host-state threshold = {threshold:.0f} us/round\n")
    print(f"{'state':<13s} {'legs':>4s} {'base':>5s} {'ab':>4s} "
          f"{'host_sum':>9s} {'arm delta':>10s} {'95% CI':>22s} "
          f"{'% base':>8s} {'null(serial)':>13s}")
    report = {"threshold_us_per_round": threshold, "strata": {}, "legs": rows}
    for state in ("clean", "contaminated"):
        sub = [r for r in rows if r["state"] == state]
        if not sub:
            continue
        arm = contrast(sub, "mtp")
        null = contrast(sub, "serial")
        report["strata"][state] = {"arm": arm, "null_serial": null,
                                   "legs": [r["leg"] for r in sub]}
        print(f"{state:<13s} {len(sub):4d} {arm['n_base']:5d} "
              f"{arm['n_treat']:4d} "
              f"{statistics.fmean(r['host_sum_us'] for r in sub):9.0f} "
              f"{arm['delta_us_per_token']:+10.1f} "
              f"[{arm['ci95_lo_us_per_token']:+9.1f},"
              f"{arm['ci95_hi_us_per_token']:+9.1f}] "
              f"{arm['pct_of_base']:+8.4f} {null['delta_us_per_token']:+13.1f}")

    whole = contrast(rows, "mtp")
    whole_null = contrast(rows, "serial")
    report["all_legs"] = {"arm": whole, "null_serial": whole_null}
    print(f"{'ALL (biased)':<13s} {len(rows):4d} {whole['n_base']:5d} "
          f"{whole['n_treat']:4d} "
          f"{statistics.fmean(r['host_sum_us'] for r in rows):9.0f} "
          f"{whole['delta_us_per_token']:+10.1f} "
          f"[{whole['ci95_lo_us_per_token']:+9.1f},"
          f"{whole['ci95_hi_us_per_token']:+9.1f}] "
          f"{whole['pct_of_base']:+8.4f} "
          f"{whole_null['delta_us_per_token']:+13.1f}")

    clean = report["strata"].get("clean", {}).get("arm", {})
    dirty = report["strata"].get("contaminated", {}).get("arm", {})
    if clean and dirty:
        diff = clean["delta_us_per_token"] - dirty["delta_us_per_token"]
        se = math.hypot(clean["se_us_per_token"], dirty["se_us_per_token"])
        report["heterogeneity"] = {"difference_us_per_token": diff,
                                   "se_us_per_token": se,
                                   "t": diff / se if se else math.nan}
        print(f"\nstratum heterogeneity: {diff:+.1f} us/token, t={diff / se:+.2f}"
              " (a large value means no pooled estimate is valid)")

    print("\nhost-state draw per arm (equal proportions = balanced)")
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        dirty_n = sum(1 for r in sub if r["state"] == "contaminated")
        print(f"  {arm:<5s} {dirty_n}/{len(sub)} contaminated")
    report["contaminated_draw"] = {
        arm: sum(1 for r in rows
                 if r["arm"] == arm and r["state"] == "contaminated")
        for arm in sorted({r["arm"] for r in rows})}

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
