#!/usr/bin/env python3
"""Price the E168 depth cap on the RANKED harness, prompt by prompt.

    usage: research/e168_cap_published.py [--cap 4] [--json OUT]

harness=ranked throughout. Nothing here uses a local second, a local ratio, or
a local round cost: the inputs are the ranked receipt `180db842`, FINDING 374's
round-count identity, and FINDING 407's step cost model. The candidate cap
changes only the candidate MTP leg, so every ranked `raw_p` moves with it and
no serial-path share is ever subtracted.

Model, stated so it can be attacked:

  * FINDING 407 fits each prompt as a two-state mixture of rounds: a fraction
    `f` pinned at the shipped cap of 7 drafts, and `1 - f` at a shallow depth
    the walk chose for itself. Every fitted shallow depth is <= 4.
  * A cap at `C` therefore moves ONLY the pinned rounds, from depth 7 to depth
    C, and leaves the shallow rounds untouched. That is what makes a cap
    cheaper than a pin, and it is why the shallow arm needs no model.
  * Round cost is `23.473*groups(M) + 0.96*edl + 6.899` ms with `M = 1 + edl`.
  * Accepted drafts in a depth-`d` round are `sum_{i=1..d} q^i` at the
    prompt's fitted homogeneous conditional acceptance `q` (FINDING 374).

The two assumptions that can break it are named in the output: homogeneous `q`
within a prompt, and acceptance at a position being independent of how many
drafts the round offered. Both are optimistic for the deep arm, so they bias
this estimate AGAINST the cap.
"""

from __future__ import annotations

import argparse
import json
import pathlib

PASS_MS = 23.473
HEAD_MS = 0.96
FIXED_MS = 6.899
PREFILL_MS_PER_TOKEN = 1.031
DECODE_TOKENS = 512
SHIPPED_CAP = 7
GROUPS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

STATE = pathlib.Path("senpai/frontier-state.json")


def groups_for(width: float) -> int:
    # A fractional fitted depth sits between two integer widths; the weight
    # stream is re-read on whole widths only, so round up to the width the
    # round actually runs at.
    import math

    return GROUPS[min(9, max(1, math.ceil(width - 1e-9)))]


def round_cost(depth: float) -> float:
    return PASS_MS * groups_for(1.0 + depth) + HEAD_MS * depth + FIXED_MS


def accepted_drafts(depth: float, q: float) -> float:
    # sum_{i=1..d} q^i, with the fractional part of a fitted depth weighted
    # into the last position rather than dropped.
    import math

    whole = int(math.floor(depth))
    frac = depth - whole
    total = sum(q ** i for i in range(1, whole + 1))
    if frac > 0.0:
        total += frac * q ** (whole + 1)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--json")
    args = parser.parse_args()

    state = json.loads(STATE.read_text())
    step = state["rankedRoundIsAStepFunction"]
    mixture = step["mixtureFit"]
    identity = state["roundCountIdentity"]["perPrompt_N_alpha_q"]
    raws = state["cheapPromptsAreFree"]["rawsAt180db842"]

    rows = []
    for prompt, fit in mixture.items():
        f = fit["fAtCap7"]
        shallow = fit["shallowD"]
        rounds, _alpha, q = identity[prompt]
        raw = raws[prompt]

        def leg_ms_per_token(deep_depth: float) -> tuple[float, float, float]:
            deep_cost = round_cost(deep_depth)
            deep_tok = 1.0 + accepted_drafts(deep_depth, q)
            shallow_cost = round_cost(shallow)
            shallow_tok = 1.0 + accepted_drafts(shallow, q)
            cost = f * deep_cost + (1.0 - f) * shallow_cost
            tok = f * deep_tok + (1.0 - f) * shallow_tok
            return cost / tok, cost, tok

        before_decode, _, _ = leg_ms_per_token(SHIPPED_CAP)
        after_decode, _, _ = leg_ms_per_token(float(args.cap))

        # Anchor the model on the receipt: the observed decode per token comes
        # from the measured round cost and the measured round count, and the
        # model is only used for the RATIO between the two policies.
        observed_decode = fit["obs"] * rounds / DECODE_TOKENS
        ratio = after_decode / before_decode
        new_decode = observed_decode * ratio

        observed_leg = observed_decode + PREFILL_MS_PER_TOKEN
        new_leg = new_decode + PREFILL_MS_PER_TOKEN
        # raw = pinned serial per token / candidate per token, and the pinned
        # serial numerator is untouched by any candidate edit.
        serial = raw * observed_leg
        new_raw = serial / new_leg

        rows.append(
            {
                "prompt": prompt,
                "mean_m": fit["M"],
                "f_at_cap": f,
                "shallow_depth": shallow,
                "q": q,
                "model_decode_ms_per_token": before_decode,
                "observed_decode_ms_per_token": observed_decode,
                "model_check_pct": (before_decode / observed_decode - 1.0) * 100.0,
                "capped_decode_ms_per_token": new_decode,
                "decode_change_pct": (ratio - 1.0) * 100.0,
                "leg_change_pct": (new_leg / observed_leg - 1.0) * 100.0,
                "raw": raw,
                "capped_raw": new_raw,
            }
        )

    def published(key: str) -> float:
        by = {row["prompt"]: row[key] for row in rows}
        return 0.5 * by["beagle"] + 0.5 * min(
            by["essays"], by["medicine"], by["republic"], by["botany"]
        )

    before_pub = published("raw")
    after_pub = published("capped_raw")

    print(f"harness=ranked   cap={args.cap} drafts (verified width M={args.cap + 1})")
    print(
        f"{'prompt':<10}{'meanM':>7}{'f@7':>7}{'shal':>6}{'q':>8}"
        f"{'modelchk':>10}{'decode%':>9}{'leg%':>8}{'raw':>8}{'capped':>9}"
    )
    for row in sorted(rows, key=lambda r: r["mean_m"]):
        print(
            f"{row['prompt']:<10}{row['mean_m']:>7.3f}{row['f_at_cap']:>7.3f}"
            f"{row['shallow_depth']:>6.2f}{row['q']:>8.4f}"
            f"{row['model_check_pct']:>9.2f}%{row['decode_change_pct']:>8.2f}%"
            f"{row['leg_change_pct']:>7.2f}%{row['raw']:>8.4f}"
            f"{row['capped_raw']:>9.4f}"
        )

    print(f"\npublished before  {before_pub:.6f}")
    print(f"published capped  {after_pub:.6f}   {(after_pub / before_pub - 1) * 100:+.2f}%")
    print("published = 0.5*beagle + 0.5*min(essays, medicine, republic, botany)")

    report = {
        "harness": "ranked",
        "cap": args.cap,
        "rows": rows,
        "published_before": before_pub,
        "published_capped": after_pub,
    }
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
