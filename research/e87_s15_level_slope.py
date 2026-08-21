#!/usr/bin/env python3
"""Recover the advisor's per-round level `L` and per-row slope `S` from public board fields.

In `e87-f18` section 5 the advisor asked for `L` and `S` next to every receipt
diff, and in comment 47 I reported that both were blocked on the per-prompt
round count `R`, which the board never publishes.  That blocker was wrong.  `R`
cancels.

Token accounting for one round is one committed primary token plus the accepted
drafts, so with `d` drafts proposed per round and accepted draft rate `a`:

    tokens_per_round = 1 + a*d
    512              = R * (1 + a*d)
    round_time       = 512 * mtp_spt / R = mtp_spt * (1 + a*d)

The 512 and the `R` divide out.  Mean round time is therefore a public quantity
up to the single scalar `a`, and `a` enters only through the small correction
`a*d`.  Fit over the drafting prompts

    round_time_p = L + S * width_p ,   width_p = 1 + d_p

gives the fixed per-round level `L` and the per-verify-row slope `S`.

`plutarch` is excluded from the fit and used as the control: it is about 92 %
non-drafting, so its time is a near mode-free probe of machine speed, and
`L / plutarch` is the advisor's mode-normalised level.

    YUKON_API_TOKEN=... python3 research/board_per_prompt.py fetch
    python3 research/e87_s15_level_slope.py <id> [<id> ...]
    python3 research/e87_s15_level_slope.py --calibrate cb8aeefb 1.9208
"""

import json
import sys

CACHE = "/tmp/yukon-board/full.json"
PLUTARCH = "c1ec5866"

# Local declared-head leg counters, used as the positive control for the
# token-accounting identity.  513 tokens rather than 512: the local harness
# counts the seeded primary.
LOCAL_TOKENS = 513
LOCAL_ROUNDS = 78
LOCAL_DRAFT_LEN = 6.358974358974359
LOCAL_ACCEPT = 0.8770161290322581


def load():
    payload = json.load(open(CACHE))
    rows = payload
    for key in ("submissions", "rows", "data", "items"):
        if isinstance(rows, dict) and key in rows:
            rows = rows[key]
            break
    return [r for r in rows if isinstance(r, dict)]


def per_prompt(row):
    metrics = row.get("officialMetrics") or {}
    entries = metrics.get("per_prompt") if isinstance(metrics, dict) else None
    out = {}
    for entry in entries or []:
        sha = entry.get("prompt_sha256") or entry.get("promptSha256")
        if sha:
            out[sha[:8]] = entry
    return out


def schedule_key(table):
    return tuple(
        (e.get("effective_mean_draft_len"), e.get("non_drafting_round_count"))
        for _, e in sorted(table.items())
    )


def control():
    """The token identity must reproduce the local round count exactly."""
    predicted = LOCAL_TOKENS / (1.0 + LOCAL_ACCEPT * LOCAL_DRAFT_LEN)
    print("  positive control on the local declared-head leg")
    print("    tokens %d  drafts/round %.6f  accepted rate %.6f"
          % (LOCAL_TOKENS, LOCAL_DRAFT_LEN, LOCAL_ACCEPT))
    print("    predicted rounds %.4f   recorded rounds %d   error %.4f"
          % (predicted, LOCAL_ROUNDS, predicted - LOCAL_ROUNDS))
    ok = abs(predicted - LOCAL_ROUNDS) < 0.01
    print("    %s" % ("IDENTITY HOLDS" if ok else "IDENTITY FAILS -- do not use this fit"))
    return ok


def fit(table, accept):
    """Return (L, S, plutarch_round_time, plutarch_token_time) in microseconds."""
    xs, ys = [], []
    for key, entry in table.items():
        d = entry.get("effective_mean_draft_len")
        spt = entry.get("mtp_seconds_per_token_mean")
        if d is None or spt is None:
            continue
        round_us = spt * (1.0 + accept * d) * 1e6
        if key == PLUTARCH:
            plut_round, plut_token = round_us, spt * 1e6
            continue
        xs.append(1.0 + d)
        ys.append(round_us)
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    level = my - slope * mx
    return level, slope, plut_round, plut_token


def cohort_of(rows, table):
    want = schedule_key(table)
    out = []
    for r in rows:
        if r.get("officialScore") is None:
            continue
        t = per_prompt(r)
        if len(t) != 8 or PLUTARCH not in t:
            continue
        if schedule_key(t) == want:
            out.append((r["id"][:8], t))
    return out


def calibrate(rows, rid, target):
    """Scan the accepted rate and both denominators for the advisor's value."""
    hit = [r for r in rows if str(r.get("id", "")).startswith(rid)]
    if not hit:
        raise SystemExit("no submission starts with %s" % rid)
    table = per_prompt(hit[0])
    print("  calibrating against the advisor's published L/plutarch = %.4f for %s"
          % (target, rid))
    print()
    print("    %-8s %12s %12s %12s %12s" % ("accept a", "L us", "S us/row",
                                            "L/plut_round", "L/plut_token"))
    best = None
    for i in range(0, 21):
        a = 0.50 + 0.025 * i
        L, S, pr, pt = fit(table, a)
        r1, r2 = L / pr, L / pt
        print("    %-8.3f %12.1f %12.1f %12.4f %12.4f" % (a, L, S, r1, r2))
        for name, val in (("plutarch round time", r1), ("plutarch token time", r2)):
            err = abs(val - target)
            if best is None or err < best[0]:
                best = (err, a, name, val, L, S)
    print()
    err, a, name, val, L, S = best
    print("    closest cell: a = %.3f, denominator = %s" % (a, name))
    print("    L/plutarch %.4f against target %.4f, error %.4f" % (val, target, err))
    print("    L = %.1f us   S = %.1f us/row" % (L, S))
    print()
    print("    cross-check against edward E92 M=1,G=1 measured round busy 64445 us:")
    print("      fitted L is %+.2f %% of that" % (100.0 * (L / 64445.0 - 1.0)))


def report(rows, ids, accept):
    print("  level and slope at assumed accepted draft rate a = %.3f" % accept)
    print("  plutarch excluded from the fit and used as the mode control")
    print()
    print("  %-9s %12s %12s %12s %10s" % ("receipt", "L us", "S us/row",
                                          "L/plutarch", "rank"))
    for rid in ids:
        hit = [r for r in rows if str(r.get("id", "")).startswith(rid)]
        if not hit:
            print("  %-9s  not on the board yet" % rid)
            continue
        table = per_prompt(hit[0])
        L, S, pr, _ = fit(table, accept)
        peers = cohort_of(rows, table)
        vals = []
        for pid, t in peers:
            pl, _s, ppr, _pt = fit(t, accept)
            vals.append((pl / ppr, pid))
        vals.sort()
        mine = L / pr
        pos = [i for i, v in enumerate(vals) if v[1] == hit[0]["id"][:8]]
        rank = "%d/%d" % (pos[0] + 1, len(vals)) if pos else "n/a"
        print("  %-9s %12.1f %12.1f %12.4f %10s"
              % (rid, L, S, mine, rank))
        if vals:
            lo = vals[0][0]
            hi = vals[-1][0]
            print("      cohort %d runs, L/plutarch min %.4f max %.4f"
                  % (len(vals), lo, hi))


def main(argv):
    rows = load()
    print("E87 s15 -- per-round level and per-row slope from public board fields")
    print()
    if not control():
        return 1
    print()
    if argv and argv[0] == "--calibrate":
        calibrate(rows, argv[1], float(argv[2]))
        return 0
    report(rows, argv, LOCAL_ACCEPT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
