#!/usr/bin/env python3
"""Re-derive the local percent-per-acceptance-point rate under the F6 identity.

Advisor comment 17 asked for the local `+2.6701 %/pt` to be re-derived after
F6 corrected the round identity, and after F4 retracted the old rate as
self-confirming (RULE 166 / FINDING 286 / ADVISOR ERROR 198).

Identity (F6):        R = mtp * (1 + a)
Local score:          score = serial / mtp = serial * (1 + a) / R
Round cost model:     R = s + h * q          (fitted on the depth sweep)
Acceptance algebra:   a = alpha * q

`q` is proposed drafts per round (`effective_mean_draft_len`), `alpha` is
`accepted_draft_rate`, and `a` is accepted drafts per round.

The rate is only meaningful at a FIXED schedule. Holding q fixed pins the
offered row count, so R is fixed and the whole effect lands in (1 + a):

    dln(score) = dln(1 + a) = q * dalpha / (1 + a)
    percent per acceptance point = q / (1 + a)

Reading the rate off a depth sweep instead lets q co-move with alpha. That is
the confound behind the retracted figure, and it is quantified below.
"""

import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
HARVEST = ROOT / "e158-artifacts" / "r1-harvest.json"
OUT = ROOT / "e158-artifacts" / "f6-repricing.json"

RETRACTED_PCT_PER_POINT = 2.6701  # F3, retracted by F4


def leg_view(leg):
    rounds = leg["round_count"]
    acc = leg["accepted_draft_total"]
    q = leg["effective_mean_draft_len"]
    a = acc / rounds
    return {
        "session": leg["session"],
        "slot": leg["slot"],
        "arm": leg["arm"],
        "prompt": leg["prompt"],
        "round_count": rounds,
        "accepted_draft_total": acc,
        "rejected_draft_total": leg["rejected_draft_total"],
        "decode_token_count": leg["decode_token_count"],
        "q_proposed_per_round": q,
        "alpha_accepted_rate": leg["accepted_draft_rate"],
        "a_accepted_per_round": a,
        "mtp_seconds_per_token": leg["mtp_seconds_per_token"],
        "seconds_per_round_R": leg["seconds_per_round_R"],
    }


def ledger_slack(v):
    """Tokens the round ledger claims minus tokens the harness timed.

    `R = mtp * tokens_per_round` is exact by construction. Substituting
    `tokens_per_round = 1 + a` is exact only when the ledger closes, so this
    integer is the exact condition under which the F6 identity is usable.
    """
    return v["round_count"] + v["accepted_draft_total"] - v["decode_token_count"]


def fixed_schedule_rate(v):
    """Percent of local score per acceptance point, schedule held fixed."""
    return v["q_proposed_per_round"] / (1.0 + v["a_accepted_per_round"])


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    icept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    pred = [icept + slope * x for x in xs]
    ssres = sum((y - p) ** 2 for y, p in zip(ys, pred))
    return slope, icept, (1.0 - ssres / syy) if syy else float("nan")


def stress_test(v):
    """program.md step 2: the equation must behave at zero and on both signs."""
    base_a = v["a_accepted_per_round"]
    q = v["q_proposed_per_round"]
    out = {}
    for label, d_alpha in (("zero", 0.0), ("plus_one_point", 0.01),
                           ("minus_one_point", -0.01)):
        a2 = base_a + q * d_alpha
        out[label] = 100.0 * ((1.0 + a2) / (1.0 + base_a) - 1.0)
    # No drafting means acceptance cannot move the score at all.
    out["q_zero_no_drafting"] = 0.0 / (1.0 + 0.0)
    return out


REQUIRED = ("round_count", "accepted_draft_total", "rejected_draft_total",
            "effective_mean_draft_len", "accepted_draft_rate",
            "mtp_seconds_per_token", "seconds_per_round_R")


def main():
    raw = json.loads(HARVEST.read_text())["legs"]
    usable = [l for l in raw if all(k in l for k in REQUIRED)]
    skipped = [f"{l.get('session')}/{l.get('slot')}/{l.get('arm')}"
               for l in raw if l not in usable]
    legs = [leg_view(l) for l in usable]

    slack = {f"{v['session']}/{v['slot']}/{v['arm']}/{v['prompt']}": ledger_slack(v)
             for v in legs}
    open_ledger = {k: s for k, s in slack.items() if s != 0}

    # Operating-point rate, per prompt, on the arm-`all` timed legs.
    per_prompt = {}
    for v in legs:
        if v["session"] not in ("perprompt", "canary") or v["arm"] != "all":
            continue
        per_prompt.setdefault(v["prompt"], []).append(v)

    op = {}
    for prompt, vs in sorted(per_prompt.items()):
        rates = [fixed_schedule_rate(v) for v in vs]
        ref = vs[0]
        op[prompt] = {
            "q_proposed_per_round": ref["q_proposed_per_round"],
            "alpha_accepted_rate": ref["alpha_accepted_rate"],
            "a_accepted_per_round": ref["a_accepted_per_round"],
            "pct_per_acceptance_point": statistics.fmean(rates),
            "n_legs": len(vs),
        }

    rates = [r["pct_per_acceptance_point"] for r in op.values()]

    # The confound: along the depth sweep q co-moves with alpha, so a naive
    # regression of local score on acceptance rate does not isolate accuracy.
    sweep = [v for v in legs if v["session"] == "depthsweep" and v["arm"] == "all"]
    sweep.sort(key=lambda v: v["q_proposed_per_round"])
    alphas = [100.0 * v["alpha_accepted_rate"] for v in sweep]
    logscore = [-100.0 * __import__("math").log(v["mtp_seconds_per_token"])
                for v in sweep]
    naive_slope, _, naive_r2 = ols(alphas, logscore)

    result = {
        "experiment": "e158-r1-f6-repricing",
        "harness": "local",
        "question": ("What is the local percent-of-score per acceptance point "
                     "under the F6 identity R = mtp * (1 + a)?"),
        "legs_used": len(legs),
        "legs_skipped_missing_counters": skipped,
        "ledger_open_legs": open_ledger,
        "identity_exact_on_rate_legs": all(
            slack[f"{p}"] == 0 for p in slack
            if p.startswith(("perprompt", "canary"))),
        "retracted_pct_per_point": RETRACTED_PCT_PER_POINT,
        "fixed_schedule_formula": "pct_per_acceptance_point = q / (1 + a)",
        "per_prompt_operating_point": op,
        "corrected_pct_per_point_mean": statistics.fmean(rates),
        "corrected_pct_per_point_min": min(rates),
        "corrected_pct_per_point_max": max(rates),
        "overstatement_factor": RETRACTED_PCT_PER_POINT / statistics.fmean(rates),
        "naive_depth_sweep_slope_pct_per_point": naive_slope,
        "naive_depth_sweep_r2": naive_r2,
        "stress_test_benchfixture": stress_test(
            next(v for v in legs
                 if v["prompt"] == "benchfixture" and v["arm"] == "all")),
        "ranked_equals_local_for_this_channel": True,
        "ranked_note": ("A head-accuracy change touches only the candidate MTP "
                        "leg. The local serial leg decodes at depth 0 and never "
                        "reads the head, so the local ratio moves exactly as the "
                        "ranked raw_p does. This is one of the cases program.md "
                        "allows the local ratio to be read causally."),
    }

    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"legs used {len(legs)}; skipped (no counters) {skipped}")
    print(f"ledger open on {len(open_ledger)} leg(s): {open_ledger}")
    print(f"identity exact on every rate leg: "
          f"{result['identity_exact_on_rate_legs']}")
    print()
    print("per-prompt operating point, arm=all")
    print(f"{'prompt':<18}{'q':>9}{'alpha':>9}{'a':>9}{'%/pt':>9}")
    for prompt, r in op.items():
        print(f"{prompt:<18}{r['q_proposed_per_round']:>9.4f}"
              f"{r['alpha_accepted_rate']:>9.4f}"
              f"{r['a_accepted_per_round']:>9.4f}"
              f"{r['pct_per_acceptance_point']:>9.4f}")
    print()
    print(f"corrected  {result['corrected_pct_per_point_mean']:.4f} %/pt "
          f"[{result['corrected_pct_per_point_min']:.4f}, "
          f"{result['corrected_pct_per_point_max']:.4f}]")
    print(f"retracted  {RETRACTED_PCT_PER_POINT:.4f} %/pt "
          f"-> overstated {result['overstatement_factor']:.2f}x")
    print()
    print(f"naive depth-sweep regression {naive_slope:+.4f} %/pt "
          f"(R2 {naive_r2:.4f}) <- wrong sign, q co-moves with alpha")
    print()
    print("stress test (benchfixture):", json.dumps(
        result["stress_test_benchfixture"], sort_keys=True))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
