#!/usr/bin/env python3
"""Dispatched verify-width (M) census from an MTP round trace.

`depth_histogram.py` reports the depth the *policy chose*. This reports the row
count M the *target forward was actually dispatched with*, which is the thing a
width-specialised kernel would key on. The mapping is established by source
inspection, not assumed:

  d >= 1  drafting round  -> one callWithHiddenAndNormed([primary] + drafts,
                             nConfirmed: 1), declaredRows = d + 1, so M = d + 1
                             (Qwen36MTPBlockSession.swift:1040, :1262)
  d == 0  skip round      -> one single-row forward, declaredRows = 1, M = 1
                             (the :765 early-return branch)

Widths 6..9 are one dispatch like any other; only the SDPA inside them is split
into two <=5-row calls, which does not change M for the 369 + 32 + 96 quantised
projections per forward (E20 s2.8).

Round share answers "how often". Row share (M * rounds) answers "what fraction
of dispatched target rows", and is the right weight for per-row streaming cost.
Neither is a time share: this trace perturbs timing and makes no timing claim.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_histogram import legs, pid_of  # noqa: E402
import glob  # noqa: E402
import re  # noqa: E402

# The round trace already prints the live width cap and full-accept streak
# (`Qwen36MTPBlockSession.swift:767`), so the gate state is read rather than
# reconstructed. Without it, "never saw M > 6" cannot distinguish "the depth-8
# gate never opened" from "it opened and the cost model still declined".
GATE_RE = re.compile(r"round=(\d+) d=(\d+) acc=(\d+).*? streak=(\d+) cap=(\d+)")

# Board-top per-prompt ratios, the operating point our own row reproduces to
# every printed digit of effective_mean_draft_len on 8/8 prompts
# (senpai/advisor-brief-2026-08-19-m6-single-pass.md s2, row 0cd0a6b4).
RANKED = {"plutarch": 1.2560, "drama": 1.9231, "travel": 2.1895,
          "beagle": 3.1433, "medicine": 3.3553, "essays": 3.3907,
          "republic": 3.4144, "botany": 3.4491}
RANKED_N = {"beagle": 4.533, "medicine": 4.768}


def score_of(ratios):
    v = sorted(ratios.values())
    return (v[3] + v[4]) / 2.0


def gate_state(out_dir, arm):
    """Chosen depth cross-tabulated against the width cap that was in force."""
    tab, streaks = Counter(), []
    for p in glob.glob(os.path.join(out_dir, arm, "trace.txt*")):
        with open(p, errors="replace") as fh:
            for line in fh:
                m = GATE_RE.search(line)
                if m:
                    _, d, _, s, cap = (int(g) for g in m.groups())
                    tab[(cap, d)] += 1
                    streaks.append(s)
    if not tab:
        return None
    total = sum(tab.values())
    open_rounds = sum(v for (cap, _), v in tab.items() if cap > 5)
    deep_used = sum(v for (cap, d), v in tab.items() if cap > 5 and d > 5)
    return {"drafting_rounds": total, "caps_seen": sorted({c for c, _ in tab}),
            "gate_open_rounds": open_rounds,
            "gate_open_share": open_rounds / total,
            "deep_rounds_when_open": deep_used,
            "max_streak": max(streaks),
            "crosstab": {f"cap{c}_d{d}": v for (c, d), v in sorted(tab.items())}}


def census(out_dir, arm, warmup=2):
    paths = sorted(glob.glob(os.path.join(out_dir, arm, "trace.txt*")), key=pid_of)
    all_legs = [(pid_of(p), lg) for p in paths for lg in legs(p) if lg]
    if not all_legs:
        return None
    drafting = [(pid, lg) for pid, lg in all_legs if any(d for _, d, _ in lg)]
    if len(drafting) > 1:
        sys.exit(f"{arm}: {len(drafting)} legs carry nonzero depths "
                 f"{[pid for pid, _ in drafting]}; cannot name one MTP leg")
    pid, rounds = drafting[0] if drafting else all_legs[-1]
    rounds = rounds[warmup:]
    # A d==0 round returns before writing its trace line, so it is invisible
    # here while still consuming a round counter. Recover it from counter gaps.
    span = rounds[-1][0] - rounds[0][0] + 1
    implied_d0 = span - len(rounds)
    widths = [(d + 1, a) for _, d, a in rounds] + [(1, 0)] * implied_d0
    n = len(widths)
    hist = Counter(m for m, _ in widths)
    rows = {m: m * c for m, c in hist.items()}
    total_rows = sum(rows.values())
    toks = Counter()
    for m, a in widths:
        toks[m] += a + 1
    total_toks = sum(toks.values())
    return {"arm": arm, "mtp_leg_pid": pid, "rounds": n,
            "implied_d0_rounds": implied_d0,
            "committed_tokens": total_toks, "dispatched_rows": total_rows,
            "max_width": max(hist), "mean_width": sum(m * c for m, c in hist.items()) / n,
            "hist": {str(m): hist[m] for m in sorted(hist)},
            "round_share": {str(m): hist[m] / n for m in sorted(hist)},
            "row_share": {str(m): rows[m] / total_rows for m in sorted(hist)},
            "token_share": {str(m): toks[m] / total_toks for m in sorted(hist)},
            "round_share_ge6": sum(c for m, c in hist.items() if m >= 6) / n,
            "row_share_ge6": sum(v for m, v in rows.items() if m >= 6) / total_rows,
            "token_share_ge6": sum(v for m, v in toks.items() if m >= 6) / total_toks}


def ranked_ge6_bound(mean_depth, max_depth=8):
    """Tightest lower bounds on the ranked M>=6 mass from published telemetry.

    `effective_mean_draft_len` is the mean chosen depth over *all* rounds
    (plutarch's 0.154 with 449 non-drafting rounds fixes that convention), and
    the shipped policy admits depth 0..8. Everything below is therefore an
    exact linear-programming bound over distributions on {0..8} with that mean
    -- no simulation, no acceptance model, no head assumption.

    Two equality constraints (total mass, mean) leave vertices supported on at
    most two depths, so enumerating depth pairs is an exact minimisation rather
    than an appeal to an analytic guess.
    """
    mean_m = mean_depth + 1.0
    lo, hi = {}, {}
    for i in range(max_depth + 1):
        for j in range(i + 1, max_depth + 1):
            if not (i <= mean_depth <= j):
                continue
            w = (mean_depth - i) / (j - i)  # mass at j
            p = {i: 1 - w, j: w}
            vals = {"round_share_ge6": sum(v for d, v in p.items() if d >= 5),
                    "row_share_ge6": sum((d + 1) * v for d, v in p.items()
                                         if d >= 5) / mean_m}
            wit = {str(d): round(v, 6) for d, v in p.items()}
            for key, val in vals.items():
                if key not in lo or val < lo[key][0]:
                    lo[key] = (val, wit)
                if key not in hi or val > hi[key][0]:
                    hi[key] = (val, wit)
    out = {"mean_m": mean_m, "max_depth_assumed": max_depth}
    for key in ("round_share_ge6", "row_share_ge6"):
        out["min_" + key] = lo[key][0]
        out["max_" + key] = hi[key][0]
        out[key + "_min_witness"] = lo[key][1]
    return out


def payoff():
    base = score_of(RANKED)
    print(f"\n=== E33 payoff frame: score = mean(4th, 5th) = {base:.5f} ===")
    print("d(score)/d(raw_p) = 0.5 for beagle and medicine, 0 for the other six.")
    ordered = sorted(RANKED.values())
    for nm in ("beagle", "medicine"):
        r = RANKED[nm]
        # Headroom = the rise at which this prompt stops being one of the two
        # central order statistics, after which its marginal weight drops to 0.
        others = sorted(v for k, v in RANKED.items() if k != nm)
        cap = others[4]  # 5th smallest of the remaining seven
        head_abs, head_pct = cap - r, 100.0 * (cap / r - 1.0)
        capped = dict(RANKED)
        capped[nm] = cap
        b8 = ranked_ge6_bound(RANKED_N[nm], 8)
        b5 = ranked_ge6_bound(RANKED_N[nm], 5)
        print(f"\n{nm}: raw_p={r:.4f}  n={RANKED_N[nm]:.3f}  mean M={b8['mean_m']:.2f}")
        print("  ranked M>=6 mass, exact bracket from n alone (depths 0..8):")
        print(f"    round share in [{b8['min_round_share_ge6']:.4f}, "
              f"{b8['max_round_share_ge6']:.4f}]  floor witness "
              f"{b8['round_share_ge6_min_witness']}")
        print(f"    row   share in [{b8['min_row_share_ge6']:.4f}, "
              f"{b8['max_row_share_ge6']:.4f}]")
        # Local runs never chose depth > 5 in any round, gate open or not. If
        # that behavioural ceiling also held on rank the floor rises sharply --
        # but botany's ranked n = 5.776 > 5 falsifies it as a universal rule,
        # so this is a scenario, not a prediction.
        print(f"    if the observed depth<=5 ceiling held on rank: round share "
              f">= {b5['min_round_share_ge6']:.4f}, row share "
              f">= {b5['min_row_share_ge6']:.4f}")
        print(f"  headroom before marginal weight -> 0: +{head_abs:.4f} "
              f"(+{head_pct:.2f} % of raw_p)")
        print(f"  score if fully realised: {score_of(capped):.5f} "
              f"(+{100*(score_of(capped)/base-1):.3f} %)")
        # A 1 % speedup confined to a cell holding time-share phi scales the
        # whole candidate leg by (1 - 0.01*phi), and raw_p = serial/candidate.
        print("  per 1 % speedup of a cell holding candidate-leg time share phi:")
        for phi in (0.10, 0.25, 0.50, 1.00):
            d_raw = r * (1.0 / (1.0 - 0.01 * phi) - 1.0)
            d_sc = 0.5 * d_raw
            print(f"    phi={phi:>5.0%}  d(raw_p)={d_raw:+.5f}  d(score)={d_sc:+.5f} "
                  f"({100*d_sc/base:+.4f} % of score)")
    print(f"\nnoise floor sigma_score = 0.078 % = {0.00078*base:.5f} score points")
    print(f"our best ranked row 3.23251 is {100*(3.23250848263467/base-1):+.4f} % vs this top")
    assert abs(base - 3.2493) < 1e-3, ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--json-out")
    ap.add_argument("--payoff", action="store_true")
    args = ap.parse_args()

    report = {}
    for arm in args.arms:
        c = census(args.out_dir, arm, args.warmup)
        if c is None:
            print(f"{arm}: no rounds found", file=sys.stderr)
            continue
        report[arm] = c
        print(f"\n=== {arm}  mtp_leg_pid={c['mtp_leg_pid']}  rounds={c['rounds']}  "
              f"tokens={c['committed_tokens']}  rows={c['dispatched_rows']} ===")
        print(f"{'M':>3} {'rounds':>7} {'round_sh':>9} {'rows':>7} {'row_sh':>8} {'tok_sh':>8}")
        for m in sorted(int(k) for k in c["hist"]):
            k = str(m)
            print(f"{m:>3} {c['hist'][k]:>7} {c['round_share'][k]:>9.4f} "
                  f"{m*c['hist'][k]:>7} {c['row_share'][k]:>8.4f} {c['token_share'][k]:>8.4f}")
        print(f"max_width={c['max_width']}  mean_width={c['mean_width']:.4f}  "
              f"implied_d0={c['implied_d0_rounds']}")
        print(f"M>=6:  round_share={c['round_share_ge6']:.4f}  "
              f"row_share={c['row_share_ge6']:.4f}  token_share={c['token_share_ge6']:.4f}")
        g = gate_state(args.out_dir, arm)
        if g:
            c["gate"] = g
            print(f"depth-8 gate: caps seen {g['caps_seen']}, open in "
                  f"{g['gate_open_rounds']}/{g['drafting_rounds']} rounds "
                  f"({g['gate_open_share']:.2%}), max streak {g['max_streak']}; "
                  f"chose depth>5 while open: {g['deep_rounds_when_open']}")

    if args.payoff:
        payoff()
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
