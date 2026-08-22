#!/usr/bin/env python3
"""Pool the ranked instrument's repeat spread over every identified repeat set.

    usage: research/e130_instrument_sd.py [--out PATH]

`research/e130_schedule_family.py` estimated the spread of mean candidate
decode time from the four receipts carrying our parent tree. Four observations
give three degrees of freedom, so the chi-square 95 % interval on that sd spans
a factor of 6.6 and the band on any derived coupling is uncomfortably wide.

This pools the same quantity over every repeat set it can identify, so the
degrees of freedom rise and the interval narrows. Three identifications are
used, from strongest to weakest:

A. `submissionCommitSha` repeats. Two receipts of the same commit submitted the
   same bytes. Any difference is pure instrument noise, with no tree term at
   all. This is the gold standard.
B. Draft-schedule plus speed tiers, as in `e130_schedule_family.py`, restricted
   to small tiers so a tier is unlikely to mix trees.
C. The serial leg across every receipt, which is a pinned prebuilt binary in
   the runner-owned workspace and therefore cannot carry a tree term.

A and B estimate the candidate leg. C bounds how much of the spread is the
host rather than the candidate build.

harness=ranked.
"""

from __future__ import annotations

import argparse
import collections
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

TIER_TOLERANCE = 0.001
MAX_TIER_SIZE_FOR_POOLING = 6

# Chi-square quantiles, two sided at 95 %, indexed by degrees of freedom.
CHI2_LO = {1: 0.000982, 2: 0.0506, 3: 0.216, 4: 0.484, 5: 0.831, 6: 1.237,
           7: 1.690, 8: 2.180, 9: 2.700, 10: 3.247, 12: 4.404, 15: 6.262,
           20: 9.591, 25: 13.120, 30: 16.791, 40: 24.433, 50: 32.357,
           60: 40.482, 80: 57.153, 100: 74.222}
CHI2_HI = {1: 5.024, 2: 7.378, 3: 9.348, 4: 11.143, 5: 12.833, 6: 14.449,
           7: 16.013, 8: 17.535, 9: 19.023, 10: 20.483, 12: 23.337,
           15: 27.488, 20: 34.170, 25: 40.646, 30: 46.979, 40: 59.342,
           50: 71.420, 60: 83.298, 80: 106.629, 100: 129.561}


def nearest(table: dict[int, float], dof: int) -> float:
    key = min(table, key=lambda k: abs(k - dof))
    return table[key]


def sd_interval(sd: float, dof: int) -> tuple[float, float]:
    lo = sd * math.sqrt(dof / nearest(CHI2_HI, dof))
    hi = sd * math.sqrt(dof / nearest(CHI2_LO, dof))
    return lo, hi


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


def mean_spt(pp: dict[str, dict], key: str) -> float:
    return st.mean([pp[n][key] for n in NAMES])


def pooled(groups: list[list[float]]) -> tuple[float, int]:
    """Pooled relative sd across groups, each group expressed as a fraction."""
    num, dof = 0.0, 0
    for values in groups:
        if len(values) < 2:
            continue
        m = st.mean(values)
        num += sum(((v - m) / m) ** 2 for v in values)
        dof += len(values) - 1
    return (100.0 * math.sqrt(num / dof), dof) if dof else (float("nan"), 0)


def speed_tiers(family: list[tuple[dict, dict]]) -> list[list]:
    ordered = sorted(family, key=lambda t: mean_spt(
        t[1], "mtp_seconds_per_token_mean"))
    tiers: list[list] = []
    for item in ordered:
        value = mean_spt(item[1], "mtp_seconds_per_token_mean")
        if tiers:
            last = mean_spt(tiers[-1][-1][1], "mtp_seconds_per_token_mean")
            if (value - last) / last <= TIER_TOLERANCE:
                tiers[-1].append(item)
                continue
        tiers.append([item])
    return tiers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    rows = [(r, per_prompt(r)) for r in json.loads(BOARD.read_text())]
    rows = [(r, pp) for r, pp in rows if pp]
    print("harness=ranked")
    print("receipts with a complete eight-prompt table: %d\n" % len(rows))
    report: dict = {"harness": "ranked", "n_receipts": len(rows)}

    # ---- A. identical submitted commit ----
    by_commit = collections.defaultdict(list)
    for row, pp in rows:
        sha = row.get("submissionCommitSha")
        if sha:
            by_commit[sha].append((row, pp))
    repeats = {k: v for k, v in by_commit.items() if len(v) > 1}
    print("A. IDENTICAL submissionCommitSha")
    print("   commits submitted more than once: %d" % len(repeats))
    groups_cand, groups_score, groups_serial = [], [], []
    for sha, members in sorted(repeats.items(),
                               key=lambda kv: -len(kv[1]))[:12]:
        cand = [mean_spt(pp, "mtp_seconds_per_token_mean")
                for _, pp in members]
        serial = [mean_spt(pp, "serial_seconds_per_token_mean")
                  for _, pp in members]
        score = [float(r["officialScore"]) for r, _ in members]
        groups_cand.append(cand)
        groups_serial.append(serial)
        groups_score.append(score)
        print("   %s n=%d  cand rel sd %.4f %%  score rel sd %.4f %%  "
              "solvers %s"
              % (sha[:8], len(members),
                 100.0 * st.stdev(cand) / st.mean(cand),
                 100.0 * st.stdev(score) / st.mean(score),
                 ",".join(sorted({r.get("solverUsername") for r, _ in members}))
                 ))
    for sha, members in sorted(repeats.items(),
                               key=lambda kv: -len(kv[1]))[12:]:
        cand = [mean_spt(pp, "mtp_seconds_per_token_mean")
                for _, pp in members]
        groups_cand.append(cand)
        groups_serial.append([mean_spt(pp, "serial_seconds_per_token_mean")
                              for _, pp in members])
        groups_score.append([float(r["officialScore"]) for r, _ in members])

    a_cand, a_dof = pooled(groups_cand)
    a_score, _ = pooled(groups_score)
    a_serial, a_sdof = pooled(groups_serial)
    if a_dof:
        lo, hi = sd_interval(a_cand, a_dof)
        print("   POOLED candidate rel sd %.4f %% on %d dof, 95 %% "
              "[%.4f, %.4f]" % (a_cand, a_dof, lo, hi))
        print("   POOLED score     rel sd %.4f %%" % a_score)
        print("   POOLED serial    rel sd %.4f %% on %d dof"
              % (a_serial, a_sdof))
        print("   published median carries %.1fx the candidate-leg noise"
              % (a_score / a_cand))
    report["identical_commit"] = {
        "n_repeat_commits": len(repeats),
        "candidate_rel_sd_pct": a_cand, "dof": a_dof,
        "candidate_rel_sd_ci95": list(sd_interval(a_cand, a_dof)) if a_dof
        else None,
        "score_rel_sd_pct": a_score,
        "serial_rel_sd_pct": a_serial,
        "median_noise_amplification": a_score / a_cand if a_dof else None,
    }

    # ---- B. draft-schedule plus speed tiers ----
    by_sig = collections.defaultdict(list)
    for row, pp in rows:
        sig = tuple(round(pp[n]["effective_mean_draft_len"], 12)
                    for n in NAMES)
        by_sig[sig].append((row, pp))
    tier_cand, tier_score = [], []
    n_tiers = 0
    for family in by_sig.values():
        if len(family) < 2:
            continue
        for tier in speed_tiers(family):
            if 2 <= len(tier) <= MAX_TIER_SIZE_FOR_POOLING:
                tier_cand.append([mean_spt(pp, "mtp_seconds_per_token_mean")
                                  for _, pp in tier])
                tier_score.append([float(r["officialScore"])
                                   for r, _ in tier])
                n_tiers += 1
    b_cand, b_dof = pooled(tier_cand)
    b_score, _ = pooled(tier_score)
    print("\nB. DRAFT SCHEDULE PLUS SPEED TIER, tiers of size 2 to %d"
          % MAX_TIER_SIZE_FOR_POOLING)
    print("   tiers used: %d" % n_tiers)
    if b_dof:
        lo, hi = sd_interval(b_cand, b_dof)
        print("   POOLED candidate rel sd %.4f %% on %d dof, 95 %% "
              "[%.4f, %.4f]" % (b_cand, b_dof, lo, hi))
        print("   POOLED score     rel sd %.4f %%" % b_score)
        print("   published median carries %.1fx the candidate-leg noise"
              % (b_score / b_cand))
    report["schedule_tiers"] = {
        "n_tiers": n_tiers,
        "candidate_rel_sd_pct": b_cand, "dof": b_dof,
        "candidate_rel_sd_ci95": list(sd_interval(b_cand, b_dof)) if b_dof
        else None,
        "score_rel_sd_pct": b_score,
        "median_noise_amplification": b_score / b_cand if b_dof else None,
    }

    # ---- C. the serial leg, which carries no tree term ----
    serial_all = [mean_spt(pp, "serial_seconds_per_token_mean")
                  for _, pp in rows]
    c_sd = 100.0 * st.stdev(serial_all) / st.mean(serial_all)
    print("\nC. SERIAL LEG ACROSS EVERY RECEIPT, n=%d" % len(serial_all))
    print("   rel sd of mean serial decode time %.4f %%" % c_sd)
    print("   The serial leg is a pinned prebuilt binary, so this is the")
    print("   host-and-day component that no candidate edit can move.")
    report["serial_leg_all"] = {"n": len(serial_all), "rel_sd_pct": c_sd}

    print("\nCONCLUSION")
    best = a_cand if a_dof >= 5 else b_cand
    best_dof = a_dof if a_dof >= 5 else b_dof
    label = "identical commit" if a_dof >= 5 else "schedule tier"
    lo, hi = sd_interval(best, best_dof)
    print("  Use the %s estimate: candidate rel sd %.4f %% on %d dof,"
          % (label, best, best_dof))
    print("  95 %% interval [%.4f, %.4f]. One new receipt against a 4-run"
          % (lo, hi))
    print("  reference mean has se %.4f %%, interval [%.4f, %.4f]."
          % (best * math.sqrt(1.25), lo * math.sqrt(1.25),
             hi * math.sqrt(1.25)))
    report["recommended"] = {
        "source": label, "candidate_rel_sd_pct": best, "dof": best_dof,
        "ci95": [lo, hi],
        "se_vs_4run_mean_pct": best * math.sqrt(1.25),
        "se_vs_4run_mean_ci95": [lo * math.sqrt(1.25), hi * math.sqrt(1.25)],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
