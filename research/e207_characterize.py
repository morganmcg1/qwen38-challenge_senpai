#!/usr/bin/env python3
"""E207 characterization deliverable for the "frozen is worse" branch.

The assignment names three items for that branch: per-position EMA-vs-prior
divergence over time, the rounds where the two arms choose different depths,
and which prompts pay. This script produces all three open-loop from the E168
p7 corpus, reusing the desk port that already passed its positive control
(1955 rounds, 0 mismatches).

Everything here is `harness=local` desk replay. It explains a predicted sign;
it does not measure one. The stage-1 legs remain the measurement.
"""

import argparse
import json
import statistics

import e207_desk as D

PROMPTS = ["benchfixture", "dramatic", "english", "medicine", "narrative",
           "natural_history", "philosophy", "technical", "travel"]


def ema_trajectory(pairs, cap):
    """Live EMA vector after every round, plus the depth each arm chose."""
    ema = list(D.PRIOR)
    rows = []
    for pair in pairs:
        live_depth = D.depth_walk(ema, pair["m"], cap)
        frozen_depth = D.depth_walk(D.PRIOR, pair["m"], cap)
        accepted = min(pair["k"], live_depth)
        rows.append({
            "margin": pair["m"],
            "k": pair["k"],
            "live_depth": live_depth,
            "frozen_depth": frozen_depth,
            "ema_pre": list(ema),
        })
        D.update_ema(ema, accepted, live_depth)
        rows[-1]["ema_post"] = list(ema)
    return rows, ema


def divergence_profile(rows, final_ema):
    """Item 1: how far each position's EMA travels from its prior."""
    profile = []
    for pos in range(D.MAX_DEPTH):
        series = [r["ema_post"][pos] for r in rows]
        profile.append({
            "position": pos,
            "prior": D.PRIOR[pos],
            "final": final_ema[pos],
            "delta_from_prior": final_ema[pos] - D.PRIOR[pos],
            "max": max(series),
            "min": min(series),
            "mean": statistics.fmean(series),
            "rounds_above_prior": sum(1 for v in series if v > D.PRIOR[pos]),
        })
    return profile


def settle_round(rows, tol=0.01):
    """Last round after which position 0 stays within `tol` of its final value.

    A small number here means the update channel finishes its work early, so
    the channel behaves like a one-off level shift rather than a continuously
    adaptive controller.
    """
    if not rows:
        return None
    final = rows[-1]["ema_post"][0]
    settled = 0
    for i, row in enumerate(rows):
        if abs(row["ema_post"][0] - final) > tol:
            settled = i + 1
    return settled


def depth_divergence(rows):
    """Item 2: the rounds where the arms disagree, and by how much."""
    diffs = [r["live_depth"] - r["frozen_depth"] for r in rows]
    disagree = [i for i, d in enumerate(diffs) if d != 0]
    histogram = {}
    for d in diffs:
        histogram[str(d)] = histogram.get(str(d), 0) + 1
    first_block = [i for i in disagree if i < 20]
    return {
        "rounds": len(rows),
        "rounds_disagreeing": len(disagree),
        "disagreement_rate": len(disagree) / len(rows) if rows else 0.0,
        "mean_signed_depth_gap": statistics.fmean(diffs) if diffs else 0.0,
        "signed_gap_histogram": histogram,
        "first_disagreement_round": disagree[0] if disagree else None,
        "disagreements_in_first_20_rounds": len(first_block),
    }


def main():
    ap = argparse.ArgumentParser()
    # Cap 7 matches the pinned p7 corpus and the measured width table, which
    # stops at width 8. Cap 8 would price a width-9 round that no leg measured.
    ap.add_argument("--cap", type=int, default=7)
    ap.add_argument("--json",
                    default="research/e207-artifacts/stage0-characterize.json")
    args = ap.parse_args()

    per_prompt, lines = {}, []
    lines.append("E207 characterization (harness=local desk replay, cap %d)"
                 % args.cap)
    lines.append("corpus: E168 p7 pinned-depth-7; walk port control 1955/0")
    lines.append("")

    for prompt in PROMPTS:
        pairs = D.corpus(prompt)
        if not pairs:
            continue
        rows, final_ema = ema_trajectory(pairs, args.cap)
        live = D.run_policy(pairs, "live", args.cap)
        frozen = D.run_policy(pairs, "frozen", args.cap)
        pay_pct = D.pct(frozen["ms_per_token"], live["ms_per_token"])
        per_prompt[prompt] = {
            "divergence_profile": divergence_profile(rows, final_ema),
            "position0_settle_round": settle_round(rows),
            "depth_divergence": depth_divergence(rows),
            "live_ms_per_token": live["ms_per_token"],
            "frozen_ms_per_token": frozen["ms_per_token"],
            "frozen_minus_live_pct": pay_pct,
            "live_mean_depth": live["mean_depth"],
            "frozen_mean_depth": frozen["mean_depth"],
        }

    # Item 3: which prompts pay, ordered worst first.
    payers = sorted(per_prompt.items(),
                    key=lambda kv: -kv[1]["frozen_minus_live_pct"])

    lines.append("item 3 - which prompts pay (frozen minus live, %):")
    for prompt, row in payers:
        lines.append("  %-16s %+7.2f%%  live d=%.2f  frozen d=%.2f  "
                     "disagree %.0f%%"
                     % (prompt, row["frozen_minus_live_pct"],
                        row["live_mean_depth"], row["frozen_mean_depth"],
                        100.0 * row["depth_divergence"]["disagreement_rate"]))
    lines.append("")

    lines.append("item 1 - position 0 EMA travel from prior (%.3f):"
                 % D.PRIOR[0])
    for prompt, row in payers:
        p0 = row["divergence_profile"][0]
        lines.append("  %-16s final %.4f  delta %+.4f  settles by round %s"
                     % (prompt, p0["final"], p0["delta_from_prior"],
                        row["position0_settle_round"]))
    lines.append("")

    lines.append("item 2 - depth disagreement:")
    for prompt, row in payers:
        dd = row["depth_divergence"]
        lines.append("  %-16s %d/%d rounds, mean signed gap %+.2f, "
                     "first at round %s"
                     % (prompt, dd["rounds_disagreeing"], dd["rounds"],
                        dd["mean_signed_depth_gap"],
                        dd["first_disagreement_round"]))

    payload = {
        "harness": "local",
        "note": "desk replay of the assignment's worse-branch deliverable; "
                "explains a predicted sign, does not measure one",
        "cap": args.cap,
        "per_prompt": per_prompt,
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    text = "\n".join(lines)
    print(text)
    with open(args.json.replace(".json", ".txt"), "w") as handle:
        handle.write(text + "\n")
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
