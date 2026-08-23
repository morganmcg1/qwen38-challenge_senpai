"""E146 F227 check: is published prefill really charged inside the candidate leg?

harness=ranked. The advisor's F227 says `mtp_seconds_per_token_mean` already
contains the 512-token seed prefill and that `prefill_seconds_per_token` reports
that share on the same per-decode-token basis. If that is true then

    decode_only_p = mtp_seconds_per_token_mean_p - prefill_seconds_per_token_p

is the leg with the prefill removed, and the implied absolute prefill seconds
must be the same for every prompt, because every seed is 512 tokens.

That prompt-invariance is the test. It is a strong one: the eight prompts differ
by 3.1x in candidate seconds per token, so eight independent quantities landing
on one value is not something a wrong convention produces.

The same identity is confirmed directly on this host by the R-C legs, where the
worker reports all three numbers:

    parent_measured_seconds_per_token * 512 - seed_prefill_seconds
        == blocks_only_seconds     (exact to the printed digits)
"""

import json
import os
import sys

import e146_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RC_LEGS = os.path.join(
    os.path.dirname(HERE), ".mlxfast-private", "e146", "rc", "legs.jsonl")


def main():
    path, rows = L.load()
    by_id = {r.id8: r for r in rows}
    print("board %s, %d scored rows" % (path, len(rows)))

    print("\n=== the ranked convention, at the bar 684821ed ===")
    bar = by_id["684821ed"]
    print("%-10s %12s %12s %12s %10s" % (
        "prompt", "cand s/tok", "prefill s/tok", "decode s/tok", "prefill s"))
    absolutes = []
    for p in L.PROMPT_ORDER:
        cand = bar.cand(p)
        pre = bar.prompts[p]["prefill_seconds_per_token"]
        absolutes.append(pre * L.DECODE_TOKENS)
        print("%-10s %12.7f %12.7f %12.7f %10.5f"
              % (p, cand, pre, cand - pre, pre * L.DECODE_TOKENS))
    m, s = L.mean(absolutes), L.sd(absolutes)
    print("\nimplied absolute seed prefill: mean %.5f s, sd %.5f s, CV %.3f %%"
          % (m, s, 100.0 * s / m))
    print("advisor F227 measured 0.52643 s per leg; delta %.5f s" % (m - 0.52643))
    share = [100.0 * bar.prompts[p]["prefill_seconds_per_token"] / bar.cand(p)
             for p in L.PROMPT_ORDER]
    print("prefill share of the candidate leg: %.2f %% (beagle %.2f %%)"
          % (L.mean(share), share[0]))

    print("\n=== the same identity, measured locally by the R-C worker ===")
    if os.path.exists(RC_LEGS):
        for line in list(open(RC_LEGS))[:3]:
            leg = json.loads(line)
            met = leg["metrics"]
            lhs = (met["parent_measured_seconds_per_token"] * L.DECODE_TOKENS
                   - met["seed_prefill_seconds"])
            rhs = met["blocks_only_seconds"]
            print("%-10s spt*512 - prefill = %.6f, blocks_only = %.6f, delta %.2e"
                  % (leg["label"], lhs, rhs, lhs - rhs))
    else:
        print("R-C legs not present yet")

    print("\n=== precision of each published field, within-row CV ===")
    print("this is the F227 claim that prefill is the board's most precise field")
    median_cv = {}
    for name, key, get in (
            ("candidate 8-prompt", "candidate",
             lambda r: [r.cand(p) for p in L.PROMPT_ORDER]),
            ("serial 8-prompt", "serial",
             lambda r: [r.serial(p) for p in L.PROMPT_ORDER]),
            ("prefill 8-prompt", "prefill",
             lambda r: [r.prefill(p) for p in L.PROMPT_ORDER])):
        cvs = []
        for r in rows:
            vals = get(r)
            if all(vals):
                cvs.append(100.0 * L.sd(vals) / L.mean(vals))
        cvs.sort()
        median_cv[key] = {"median_within_row_cv_pct": cvs[len(cvs) // 2],
                          "rows": len(cvs)}
        print("%-20s median within-row CV %.4f %% over %d rows"
              % (name, cvs[len(cvs) // 2], len(cvs)))
    print("\nNote: the candidate and serial CVs are dominated by real")
    print("prompt-to-prompt differences, so they are not comparable to the")
    print("prefill CV, which is a pure precision figure because every seed is")
    print("512 tokens. The prefill CV is the useful one.")

    out = {
        "harness": "ranked",
        "board_path": path,
        "board_rows_scored": len(rows),
        "bar": "684821ed",
        "e146_prefill_implied_seconds_mean": m,
        "e146_prefill_implied_seconds_sd": s,
        "e146_prefill_prompt_invariance_cv_pct": 100.0 * s / m,
        "e146_prefill_delta_vs_advisor_seconds": m - 0.52643,
        "e146_prefill_share_of_candidate_leg_pct": L.mean(share),
        "e146_prefill_share_beagle_pct": share[0],
        "within_row_cv": median_cv,
    }
    with open(os.path.join(HERE, "e146-prefill.json"), "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
