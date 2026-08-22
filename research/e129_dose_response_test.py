"""Test the F24 section-3 dose-response against a matched null.

F24 reads the per-prompt candidate-leg table as a dose-response: the arm changes
widths 6 and 7 only, so it should act on prompts whose width mass sits there and
not on prompts below or above. That is a real, falsifiable structural claim.

The question this asks is whether the contrast survives a null. The serial leg is
the matched control: same runner, same session, same prompts, same grouping, and
candidate-editable code provably cannot move it (program.md ranked causal
boundary). Any grouped structure the serial leg shows by chance is structure the
candidate leg can also show by chance.

All deltas are reported as `(new/old - 1) * 100`, so NEGATIVE IS FASTER.

harness=ranked. Read-only over the cached Yukon board.
"""

import itertools
import json
import math
import sys

from e129_schedule_invariance import NAMES, per_prompt

BOARD = "/tmp/yukon-board/full.json"

# F24 section 3 grouping, fixed by the mechanism before any data is seen:
# the arm rewrites widths 6 and 7 only.
ACTS = ["beagle", "republic", "essays", "medicine"]
INERT = ["plutarch", "drama", "travel"]
ABOVE = ["botany"]


def load(prefix):
    for r in json.load(open(BOARD)):
        if r["id"].startswith(prefix):
            return r
    raise SystemExit(f"no board row for {prefix}")


def contrast(vals, names, acts, inert):
    a = [vals[names.index(n)] for n in acts]
    i = [vals[names.index(n)] for n in inert]
    return sum(a) / len(a) - sum(i) / len(i)


def perm_p(vals, names, acts, inert, observed):
    """Two-sided exact permutation p over all ways to relabel the prompts."""
    pool = list(range(len(names)))
    na, ni = len(acts), len(inert)
    hits = tot = 0
    for pick_a in itertools.combinations(pool, na):
        rest = [i for i in pool if i not in pick_a]
        for pick_i in itertools.combinations(rest, ni):
            ma = sum(vals[i] for i in pick_a) / na
            mi = sum(vals[i] for i in pick_i) / ni
            tot += 1
            if abs(ma - mi) >= abs(observed) - 1e-12:
                hits += 1
    return hits / tot, tot


def main(new_id="623e77af", old_id="0c6191b7"):
    new, old = load(new_id), load(old_id)
    a, b = per_prompt(new), per_prompt(old)

    names = sorted(NAMES.values(),
                   key=lambda n: a[n]["effective_mean_draft_len"])
    mbar = [a[n]["effective_mean_draft_len"] + 1.0 for n in names]
    cand = [(a[n]["mtp_seconds_per_token_mean"]
             / b[n]["mtp_seconds_per_token_mean"] - 1) * 100 for n in names]
    ser = [(a[n]["serial_seconds_per_token_mean"]
            / b[n]["serial_seconds_per_token_mean"] - 1) * 100 for n in names]

    print(f"harness=ranked   {new['id'][:8]} vs {old['id'][:8]}")
    print("deltas are (new/old - 1)*100, NEGATIVE IS FASTER")
    print()
    print(f"{'prompt':<10}{'M-bar':>7}{'group':>7}{'candidate':>11}{'serial':>10}")
    for n, m, c, s in zip(names, mbar, cand, ser):
        g = "acts" if n in ACTS else ("above" if n in ABOVE else "inert")
        print(f"{n:<10}{m:>7.3f}{g:>7}{c:>11.3f}{s:>10.3f}")
    print()

    print("--- F24 grouped contrast: mean(acts) - mean(inert) ---")
    for label, vals in (("candidate", cand), ("serial  (null)", ser)):
        obs = contrast(vals, names, ACTS, INERT)
        p, tot = perm_p(vals, names, ACTS, INERT, obs)
        print(f"{label:<16} contrast {obs:+.4f} pp   exact permutation "
              f"p = {p:.4f}  over {tot} relabelings")
    print()
    print("The serial row is the calibration. It cannot contain a candidate")
    print("effect, so whatever contrast it shows is what chance alone supplies")
    print("for this grouping, on these prompts, in this session.")
    print()

    sd_c = math.sqrt(sum((x - sum(cand) / 8) ** 2 for x in cand) / 7)
    sd_s = math.sqrt(sum((x - sum(ser) / 8) ** 2 for x in ser) / 7)
    print(f"per-prompt sd   candidate {sd_c:.4f} %   serial {sd_s:.4f} %")
    print(f"ratio serial/candidate {sd_s / sd_c:.3f}")
    print()

    empirical_null(contrast(cand, names, ACTS, INERT))


def empirical_null(observed, sample=400):
    """Distribution of the same grouped contrast on serial legs of many pairs.

    Every serial-leg pair is a null draw by construction, and it carries the
    real per-prompt persistence that an exchangeable permutation null omits.
    """
    import random

    rows = [r for r in json.load(open(BOARD))
            if (r.get("officialMetrics") or {}).get("per_prompt")
            and len(per_prompt(r)) == len(NAMES)]
    random.seed(20260822)
    names = sorted(NAMES.values())

    stats = []
    for _ in range(sample):
        x, y = random.sample(rows, 2)
        px, py = per_prompt(x), per_prompt(y)
        d = [(px[n]["serial_seconds_per_token_mean"]
              / py[n]["serial_seconds_per_token_mean"] - 1) * 100
             for n in names]
        stats.append(contrast(d, names, ACTS, INERT))

    stats.sort()
    n = len(stats)
    tail = sum(1 for s in stats if abs(s) >= abs(observed)) / n
    print(f"--- empirical null: same contrast on {n} random serial-leg pairs ---")
    print(f"  sd of the null contrast   {math.sqrt(sum(s * s for s in stats) / n):.4f} pp")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"  q{int(q * 100):02d}  {stats[int(q * n) - 1]:+.4f} pp")
    print(f"  observed candidate contrast {observed:+.4f} pp")
    print(f"  two-sided empirical p = {tail:.4f}")
    print("  the serial leg is noisier than the candidate leg, so this null is")
    print("  conservative in scale but honest about per-prompt persistence.")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
