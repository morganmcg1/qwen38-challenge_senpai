#!/usr/bin/env python3
"""E207 section 4: the oracle ceiling for the within-prompt depth-policy family.

Two hindsight policies are priced against the shipped live-EMA rule on the
E168 p7 pinned-depth-7 corpus, using the same validated port (positive control
1955 rounds, 0 mismatches):

  (a) best fixed depth  per prompt, d in {0..7}, chosen in hindsight;
  (b) per-round oracle  depth, chosen with the realized accept length known.

(b) is the strict upper bound on anything this family can reach: no causal
controller can beat a controller that already knows each round's answer.

RULE 392 replay validity: the corpus pins depth 7, so the accept length k is
known for every position <= 7 and any policy choosing d <= 7 replays exactly
as accepted = min(k, d). Cap 8 is NOT evaluated: pricing a width-9 round would
need a cost cell no leg ever measured, and an INFERRED cell cannot carry a
ceiling claim.

Everything here is harness=local desk replay.
"""

import argparse
import json
import statistics
import sys

import e207_desk as D

PROMPTS = ["benchfixture", "dramatic", "english", "medicine", "narrative",
           "natural_history", "philosophy", "technical", "travel"]
PROSE = [p for p in PROMPTS if p != "benchfixture"]
CAP = 7

# Composition bar from the assignment: 0.2 ms/round with 2 sigma clear of zero.
COMPOSITION_BAR_MS_PER_ROUND = 0.2


def policy_value(pairs, depths):
    """ms/token and ms/round for one explicit per-round depth sequence."""
    cost = sum(D.round_ms(d) for d in depths)
    accepted = sum(min(pair["k"], d) for pair, d in zip(pairs, depths))
    rounds = len(depths)
    tokens = rounds + accepted
    return {"ms_per_token": cost / tokens,
            "ms_per_round": cost / rounds,
            "tokens_per_round": tokens / rounds,
            "mean_depth": sum(depths) / rounds,
            "mean_accepted": accepted / rounds,
            "rounds": rounds}


def best_fixed_depth(pairs):
    """(a) hindsight-optimal single depth for the whole prompt."""
    table = {}
    for d in range(CAP + 1):
        table[d] = policy_value(pairs, [d] * len(pairs))["ms_per_token"]
    best = min(table, key=table.get)
    return best, table


def oracle_per_round(pairs, tol=1e-12, max_iter=100):
    """(b) per-round oracle depth minimizing the LEG ratio sum(c)/sum(t).

    Minimizing a sum-ratio is not separable round by round, so a per-round
    argmin of c(d)/(1+min(k,d)) is not the optimum. Dinkelbach's method solves
    it exactly: for a candidate rate lam, minimize sum(c_i - lam * t_i)
    round by round; the lam where that sum reaches zero is the optimal ratio.
    """
    depths = [CAP] * len(pairs)
    lam = policy_value(pairs, depths)["ms_per_token"]
    for _ in range(max_iter):
        depths = []
        for pair in pairs:
            best_d, best_obj = None, None
            for d in range(CAP + 1):
                obj = D.round_ms(d) - lam * (1 + min(pair["k"], d))
                if best_obj is None or obj < best_obj:
                    best_d, best_obj = d, obj
            depths.append(best_d)
        new_lam = policy_value(pairs, depths)["ms_per_token"]
        if abs(new_lam - lam) < tol:
            lam = new_lam
            break
        lam = new_lam
    return depths, lam


def depth_k_correlation(depths, ks):
    """Correlation of the chosen depth with the realized accept length.

    Near zero means the rule's round-level variation carries no information,
    which is the precondition for treating that variation as a pure Jensen
    cost against the convex round-cost table.
    """
    try:
        return statistics.correlation(depths, ks)
    except statistics.StatisticsError:
        return None


def cost_second_differences():
    """Curvature of the measured round-cost table, by width."""
    return {str(m): D.COST_MS[m + 1] - 2 * D.COST_MS[m] + D.COST_MS[m - 1]
            for m in range(2, 8)}


def delta(shipped, policy):
    """Improvement of `policy` over `shipped`, positive means faster.

    The ms/round figure converts the ms/token gain at the shipped
    tokens-per-round, so it is comparable with the 0.2 ms/round bar even
    though the two policies emit different token counts per round.
    """
    gain_per_token = shipped["ms_per_token"] - policy["ms_per_token"]
    return {
        "pct_faster": -D.pct(policy["ms_per_token"], shipped["ms_per_token"]),
        "ms_per_round_equivalent": gain_per_token
        * shipped["tokens_per_round"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",
                    default="research/e207-artifacts/stage0-oracle.json")
    args = ap.parse_args()

    per_prompt, lines = {}, []
    lines.append("E207 oracle ceiling for the depth-policy family "
                 "(harness=local, cap %d)" % CAP)
    lines.append("corpus: E168 p7 pinned-depth-7; port control 1955/0; "
                 "cap 8 NOT evaluated (no measured width-9 cell)")
    lines.append("")

    for prompt in PROMPTS:
        pairs = D.corpus(prompt)
        if not pairs:
            continue
        shipped_run = D.run_policy(pairs, "live", CAP)
        shipped = policy_value(pairs, shipped_run["depths"])
        fixed_d, fixed_table = best_fixed_depth(pairs)
        fixed = policy_value(pairs, [fixed_d] * len(pairs))
        oracle_depths, _ = oracle_per_round(pairs)
        oracle = policy_value(pairs, oracle_depths)
        # Holding the shipped rule's own average level fixed isolates the cost
        # of its round-to-round variation from the cost of its chosen level.
        level_d = min(CAP, max(0, round(shipped["mean_depth"])))
        level = policy_value(pairs, [level_d] * len(pairs))
        per_prompt[prompt] = {
            "shipped": shipped,
            "best_fixed_depth": fixed_d,
            "best_fixed": fixed,
            "best_fixed_delta": delta(shipped, fixed),
            "fixed_depth_table_ms_per_token": fixed_table,
            "shipped_level_depth": level_d,
            "variance_only_delta": delta(shipped, level),
            "shipped_depth_sd": statistics.pstdev(shipped_run["depths"]),
            "shipped_depth_k_correlation": depth_k_correlation(
                shipped_run["depths"], [pair["k"] for pair in pairs]),
            "oracle": oracle,
            "oracle_delta": delta(shipped, oracle),
            "oracle_depth_histogram": {
                str(d): oracle_depths.count(d) for d in sorted(set(oracle_depths))
            },
        }

    lines.append("%-16s %9s | %5s %8s %10s | %8s %10s"
                 % ("prompt", "shipped", "d*", "(a) %", "(a) ms/rd",
                    "(b) %", "(b) ms/rd"))
    for prompt in PROMPTS:
        if prompt not in per_prompt:
            continue
        row = per_prompt[prompt]
        lines.append("%-16s %9.4f | %5d %+8.2f %+10.3f | %+8.2f %+10.3f"
                     % (prompt, row["shipped"]["ms_per_token"],
                        row["best_fixed_depth"],
                        row["best_fixed_delta"]["pct_faster"],
                        row["best_fixed_delta"]["ms_per_round_equivalent"],
                        row["oracle_delta"]["pct_faster"],
                        row["oracle_delta"]["ms_per_round_equivalent"]))
    lines.append("")

    summary = {"harness": "local", "cap": CAP,
               "compositionBarMsPerRound": COMPOSITION_BAR_MS_PER_ROUND}
    for key, field in (("bestFixed", "best_fixed_delta"),
                       ("varianceOnly", "variance_only_delta"),
                       ("oracle", "oracle_delta")):
        prose_pct = sorted(per_prompt[p][field]["pct_faster"] for p in PROSE
                           if p in per_prompt)
        prose_ms = sorted(per_prompt[p][field]["ms_per_round_equivalent"]
                          for p in PROSE if p in per_prompt)
        summary["%s_proseMedianPctFaster" % key] = statistics.median(prose_pct)
        summary["%s_proseMedianMsPerRound" % key] = statistics.median(prose_ms)
        summary["%s_proseMaxPctFaster" % key] = max(prose_pct)
        summary["%s_proseMinPctFaster" % key] = min(prose_pct)
        summary["%s_benchfixturePctFaster" % key] = \
            per_prompt["benchfixture"][field]["pct_faster"]

    ceiling_ms = summary["oracle_proseMedianMsPerRound"]
    summary["oracleCeilingBelowCompositionBar"] = \
        ceiling_ms < COMPOSITION_BAR_MS_PER_ROUND
    summary["oracleCeilingMsPerRoundOverBar"] = \
        ceiling_ms / COMPOSITION_BAR_MS_PER_ROUND

    lines.append("prose-population medians (benchfixture EXCLUDED, "
                 "sign-inverted per FINDING 538):")
    lines.append("  (a) best fixed depth : %+.3f%%  %+.4f ms/round"
                 % (summary["bestFixed_proseMedianPctFaster"],
                    summary["bestFixed_proseMedianMsPerRound"]))
    lines.append("  (b) per-round oracle : %+.3f%%  %+.4f ms/round"
                 % (summary["oracle_proseMedianPctFaster"],
                    summary["oracle_proseMedianMsPerRound"]))
    lines.append("  variance-only (fixed at the shipped rule's own level):")
    lines.append("                       : %+.3f%%  %+.4f ms/round"
                 % (summary["varianceOnly_proseMedianPctFaster"],
                    summary["varianceOnly_proseMedianMsPerRound"]))
    lines.append("")
    lines.append("composition bar        : %.3f ms/round"
                 % COMPOSITION_BAR_MS_PER_ROUND)
    lines.append("oracle ceiling / bar   : %.3fx"
                 % summary["oracleCeilingMsPerRoundOverBar"])
    lines.append("VERDICT: oracle ceiling is %s the composition bar"
                 % ("BELOW" if summary["oracleCeilingBelowCompositionBar"]
                    else "ABOVE"))

    prose_corr = [per_prompt[p]["shipped_depth_k_correlation"] for p in PROSE
                  if p in per_prompt]
    summary["proseMaxAbsDepthKCorrelation"] = max(abs(c) for c in prose_corr)
    summary["proseMedianDepthSd"] = statistics.median(
        [per_prompt[p]["shipped_depth_sd"] for p in PROSE if p in per_prompt])

    lines.append("")
    lines.append("mechanism: uninformative variation against a convex cost")
    lines.append("  max |corr(depth, k)| over prose : %.4f"
                 % summary["proseMaxAbsDepthKCorrelation"])
    lines.append("  median shipped depth sd         : %.3f"
                 % summary["proseMedianDepthSd"])
    lines.append("  cost second differences by width: %s"
                 % {m: round(v, 2)
                    for m, v in cost_second_differences().items()})

    payload = {"harness": "local", "cap": CAP,
               "note": "hindsight ceilings for the within-prompt depth-policy "
                       "family; (b) is a strict upper bound",
               "cost_second_differences": cost_second_differences(),
               "summary": summary, "per_prompt": per_prompt}
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    text = "\n".join(lines)
    print(text)
    with open(args.json.replace(".json", ".txt"), "w") as handle:
        handle.write(text + "\n")
    print("\nwrote %s" % args.json)


if __name__ == "__main__":
    sys.exit(main())
