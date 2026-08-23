#!/usr/bin/env python3
"""FINDING 219 - recover absolute ranked round cost from published board metrics.

The ranked board publishes only `effective_mean_draft_len` and
`mtp_seconds_per_token_mean` per prompt.  It publishes neither the round count
nor the accept rate.  The round count is nevertheless recoverable:

    R = 512 / (1 + a * d)

Validated against the F92 pinned ranked round counts on all eight prompts, worst
error 0.33 rounds.  Therefore

    tokens per round = 1 + a * d
    round cost       = seconds/token * (1 + a * d) = seconds/token * 512 / R

Usage
-----
    python3 research/f219_round_cost.py                 # validate + anchor table
    python3 research/f219_round_cost.py --bound beagle 0.0107118 4.4771

The --bound form takes a prompt name and the candidate seconds/token and
effective draft length of a *changed* row, and reports the bounded round-cost
change against the anchor.  A bound is required rather than a point estimate
because seconds/token is a ratio of round cost to tokens per round, and a change
that moves the proposal distribution moves both terms.  See ADVISOR ERROR 154.
"""

import argparse
import sys

DECODE_TOKENS = 512

# F92 pinned ranked regime, measured on the shipped scheduler.
# name: (effective mean draft length, accept rate, pinned round count)
F92 = {
    "beagle":   (4.3818, 0.834, 110),
    "medicine": (5.2556, 0.892, 90),
    "essays":   (5.0870, 0.897, 92),
    "botany":   (6.1481, 0.865, 81),
    "republic": (4.9892, 0.903, 93),
    "drama":    (2.2976, 0.449, 252),
    "travel":   (2.6479, 0.533, 212),
    "plutarch": (0.1557, 0.333, 487),
}

# Candidate seconds/token at the promoted anchor 1760479a (scarletbright,
# 3.70355222).  Draft lengths at this row equal the F92 pinned values.
ANCHOR = "1760479a"
ANCHOR_SPT = {
    "beagle":   0.0106863,
    "botany":   0.0096534,
    "drama":    0.0178217,
    "essays":   0.0098320,
    "medicine": 0.0097197,
    "plutarch": 0.0300828,
    "republic": 0.0097098,
    "travel":   0.0156178,
}

# Edward's rebuilt per-round width cost curve, rows 1..9, microseconds.
# REPLAYED, not measured.  E145 R2 measures it live.
PER_ROUND_US = [
    31173.2, 34619.3, 38065.4, 41511.4, 44957.5,
    61198.8, 62824.9, 70315.4, 75638.9,
]

ORDER = ["beagle", "botany", "drama", "essays",
         "medicine", "plutarch", "republic", "travel"]


def tokens_per_round(accept_rate, draft_len):
    return 1.0 + accept_rate * draft_len


def round_count(accept_rate, draft_len, decode_tokens=DECODE_TOKENS):
    return decode_tokens / tokens_per_round(accept_rate, draft_len)


def round_cost_seconds(spt, accept_rate, draft_len):
    return spt * tokens_per_round(accept_rate, draft_len)


def validate():
    print("FINDING 219 validation: R = %d / (1 + a * d) against F92 pinned R"
          % DECODE_TOKENS)
    print("  %-9s %9s %7s %10s %8s %9s" %
          ("prompt", "dlen", "rate", "R implied", "R F92", "err"))
    worst = 0.0
    for name in ORDER:
        d, a, R = F92[name]
        implied = round_count(a, d)
        err = implied - R
        worst = max(worst, abs(err))
        print("  %-9s %9.4f %7.3f %10.2f %8d %+9.2f" % (name, d, a, implied, R, err))
    print("  worst absolute error: %.2f rounds" % worst)
    if worst > 0.5:
        print("  IDENTITY FAILED - do not use this module", file=sys.stderr)
        return False
    return True


def anchor_table():
    print()
    print("Ranked round cost at anchor %s" % ANCHOR)
    print("  %-9s %11s %9s %8s %11s %11s" %
          ("prompt", "cand s/tok", "dlen", "R", "tok/round", "round us"))
    for name in ORDER:
        d, a, _ = F92[name]
        spt = ANCHOR_SPT[name]
        tpr = tokens_per_round(a, d)
        print("  %-9s %11.7f %9.4f %8.2f %11.4f %11.1f" %
              (name, spt, d, round_count(a, d), tpr,
               round_cost_seconds(spt, a, d) * 1e6))


def bound(name, spt_new, dlen_new):
    """Bound the round-cost change of a row whose draft length moved.

    The accept rate of the changed row is not published.  Two reference cases
    bracket it, and monotonicity of the width cost curve supplies the sign.
    """
    if name not in F92:
        raise SystemExit("unknown prompt %r; expected one of %s"
                         % (name, ", ".join(ORDER)))
    d0, a0, _ = F92[name]
    s0 = ANCHOR_SPT[name]
    tpr0 = tokens_per_round(a0, d0)
    c0 = round_cost_seconds(s0, a0, d0)

    print()
    print("Bounded round-cost change on %s against anchor %s" % (name, ANCHOR))
    print("  seconds/token   %.7f -> %.7f   %+.4f %%"
          % (s0, spt_new, 100.0 * (spt_new - s0) / s0))
    print("  draft length    %9.4f -> %9.4f   %+.4f %% relative"
          % (d0, dlen_new, 100.0 * (dlen_new - d0) / d0))
    print("  tokens/round at anchor = 1 + %.3f * %.4f = %.4f" % (a0, d0, tpr0))
    print("  round cost   at anchor = %.1f us" % (c0 * 1e6))

    c_rate_held = round_cost_seconds(spt_new, a0, dlen_new)
    c_tpr_held = spt_new * tpr0
    rate_needed = (tpr0 - 1.0) / dlen_new

    print()
    print("  accept rate unchanged at %.3f:" % a0)
    print("    tokens/round -> %.4f   round cost -> %.1f us   %+.4f %%  (%+.1f us)"
          % (tokens_per_round(a0, dlen_new), c_rate_held * 1e6,
             100.0 * (c_rate_held - c0) / c0, (c_rate_held - c0) * 1e6))
    print("  tokens/round unchanged (accept rate absorbs the draft-length move):")
    print("    accept rate  -> %.4f   round cost -> %.1f us   %+.4f %%  (%+.1f us)"
          % (rate_needed, c_tpr_held * 1e6,
             100.0 * (c_tpr_held - c0) / c0, (c_tpr_held - c0) * 1e6))

    lo, hi = sorted([c_rate_held, c_tpr_held])
    print()
    print("  BOUND: round cost changes by %+.4f %% .. %+.4f %%   (%+.0f .. %+.0f us)"
          % (100.0 * (lo - c0) / c0, 100.0 * (hi - c0) / c0,
             (lo - c0) * 1e6, (hi - c0) * 1e6))

    step56 = PER_ROUND_US[5] - PER_ROUND_US[4]
    print("  replayed 5->6 width step = %.1f us  (%.1f %% of this prompt's round)"
          % (step56, 100.0 * step56 / (c0 * 1e6)))
    if dlen_new != d0:
        print("  width mass that must cross 5->6 to explain the bound ends: "
              "%.3f %% .. %.3f %%"
              % (100.0 * (lo - c0) * 1e6 / step56,
                 100.0 * (hi - c0) * 1e6 / step56))
        print("  implied boundary mass density: %.3f .. %.3f per unit width"
              % ((lo - c0) * 1e6 / step56 / (dlen_new - d0),
                 (hi - c0) * 1e6 / step56 / (dlen_new - d0)))
    print()
    print("  The width cost curve above is REPLAYED, not measured. Label any")
    print("  number derived from it as provisional until E145 R2 lands.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bound", nargs=3, metavar=("PROMPT", "SPT", "DLEN"),
                    help="bound the round-cost change of a changed row")
    args = ap.parse_args()

    if not validate():
        return 1
    anchor_table()
    if args.bound:
        bound(args.bound[0], float(args.bound[1]), float(args.bound[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
