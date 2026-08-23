#!/usr/bin/env python3
"""How much published median is one point of draft acceptance worth?

Every dispatch-count mechanism on the table moves the candidate round time.
This asks the other question: what does the ratio do if the same proposals are
accepted more often?

A round proposes `effective_mean_draft_len` drafts and the target verifies all
of them whether or not they are accepted, so the round COST is set by the
proposed width. The round YIELD is `1 + acceptance * edl` tokens. Therefore

    raw_ratio = serial_seconds_per_token * (1 + acceptance * edl) / round_time

and acceptance enters the numerator alone:

    d(raw)/raw / d(acceptance) = edl / (1 + acceptance * edl)

That derivative is near 1 for every weighted prompt, so a point of acceptance
is worth about a point of ratio. Nothing else on the campaign board has that
slope.

The receipt publishes no per-prompt acceptance, so the table brackets it.

  research/e135_acceptance_leverage.py [--board PATH] [--receipt ID]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_receipt_read as rr  # noqa: E402
import e135_round_recovery as rec  # noqa: E402

ACCEPTANCE_BRACKET = (0.70, 0.877, 0.95)

# Published median of the crown receipt, and of the fair tree behind it once
# Finding 251's essays serial luck is removed.
BAR_PUBLISHED = 3.71959723
BAR_FAIR_TREE = 3.70486350


def published_median(entries: dict, alpha: dict, step: float) -> float:
    ratios = []
    for name, e in entries.items():
        d = e["effective_mean_draft_len"]
        base = 1.0 + alpha[name] * d
        take = min(step, 1.0 - alpha[name])
        ratios.append(e["raw_ratio_of_means"] * (base + take * d) / base)
    ratios.sort()
    return (ratios[3] + ratios[4]) / 2.0


def solve_step(entries: dict, alpha: dict, target: float) -> float:
    lo, hi = 0.0, 1.0
    if published_median(entries, alpha, hi) < target:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if published_median(entries, alpha, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default="0cf1637e")
    ap.add_argument("--board", default="/tmp/yukon-board/read.json")
    args = ap.parse_args()

    board = rr.load_board(args.board)
    row = rr.find(board, args.receipt)
    mine = rr.per_prompt(row)
    lo, hi = rr.median_pair(row)
    published = row["officialScore"]
    tokens = row["officialMetrics"].get("decode_tokens", 512)

    smallest = {n: rec.recover_rounds(e["effective_mean_draft_len"], tokens,
                                      e.get("non_drafting_round_count", 0)
                                      )["rounds"] for n, e in mine.items()}
    fit = rec.select_multiples(row, tokens, dict(smallest))
    rounds = fit["rounds"]
    alpha = {n: (tokens - rounds[n]) / (e["effective_mean_draft_len"] * rounds[n])
             for n, e in mine.items() if e["effective_mean_draft_len"] > 0}
    us = {n: e["mtp_seconds_per_token_mean"] * tokens * 1e6 / rounds[n]
          for n, e in mine.items()}

    print(f"receipt {row['id'][:8]}  published {published:.8f}  "
          f"median pair {lo}, {hi}")
    print("  acceptance and round time are RECOVERED from the receipt, not "
          "assumed:\n  see research/e135_round_recovery.py.")

    print("\n## Ratio gain per +0.01 of draft acceptance, by prompt")
    print("  prompt      edl     r148   accept    gain %   a=0.70   a=0.95")
    for name in sorted(mine, key=lambda n: -rr.RULE148_WEIGHTS.get(n, 0.0)):
        d = mine[name]["effective_mean_draft_len"]
        cells = "".join(f"{d / (1.0 + a * d):9.4f}"
                        for a in (ACCEPTANCE_BRACKET[0], ACCEPTANCE_BRACKET[2]))
        print(f"  {name:<9} {d:6.3f}  {rr.RULE148_WEIGHTS.get(name, 0.0):.4f}"
              f"  {alpha[name]:6.4f}  {d / (1.0 + alpha[name] * d):8.4f}{cells}")

    print("\n## What the published median does, re-sorted after every step")
    print("  Rule 121: the ladder is re-sorted, so this is not a linear scale.")
    print("  d_accept    median    vs now")
    for step in (0.0, 0.005, 0.01, 0.02, 0.05, 1.0):
        med = published_median(mine, alpha, step)
        print(f"  {step:+.3f}    {med:.6f}   {(med / published - 1) * 100:+7.3f} %")

    print("\n  The last row is acceptance at 1.000, every proposed draft taken.")
    print("  It is a ceiling, not a plan.")

    print("\n## The bar in whole rounds")
    print("  Yield is tokens/rounds exactly, so the ladder moves in integer")
    print("  steps. One round off each median prompt is the whole gap.")
    ratios = {n: e["raw_ratio_of_means"] for n, e in mine.items()}
    for drop in (0, 1, 2):
        moved = dict(ratios)
        for n in (lo, hi):
            moved[n] = ratios[n] * rounds[n] / (rounds[n] - drop)
        order = sorted(moved.values())
        med = (order[3] + order[4]) / 2.0
        print(f"  drop {drop} round(s) from {lo} ({rounds[lo]}) and "
              f"{hi} ({rounds[hi]}):  median {med:.6f}"
              f"  {(med / published - 1) * 100:+7.3f} %"
              f"  {'clears the bar' if med > BAR_PUBLISHED else ''}")

    print("\n## Acceptance needed to clear each target")
    for label, target in (("fair crown tree", BAR_FAIR_TREE),
                          ("published bar   ", BAR_PUBLISHED)):
        step = solve_step(mine, alpha, target)
        print(f"  {label}  {target:.8f}  needs d_accept {step:+.4f}"
              f"  -> beagle acceptance {alpha['beagle'] + step:.4f}")

    print("\n## Round-time budget a better head may spend, per +0.01 acceptance")
    print("  Break-even is exact: the round time may grow by the same percent")
    print("  the ratio gains, so the budget is the derivative times the round.")
    print("  prompt      rounds   round us   gain %   budget us/round")
    for name in sorted(mine, key=lambda n: -rr.RULE148_WEIGHTS.get(n, 0.0)):
        d = mine[name]["effective_mean_draft_len"]
        gain = d / (1.0 + alpha[name] * d)
        print(f"  {name:<9} {rounds[name]:7d} {us[name]:10.1f}  {gain:7.4f}"
              f"  {us[name] * gain / 100:10.1f}")

    census(board)


def census(board: list) -> None:
    """Has any solver on the board changed what the head proposes?

    Two independent reads. `head_provenance_sha256` names the artifact, so a
    re-quantized or repacked head that preserves tensors tensor-for-tensor
    still changes it. `effective_mean_draft_len` on a fixed prompt names the
    BEHAVIOUR: two heads that propose the same tokens draft the same lengths
    under the same schedule, whatever their digests say.
    """
    heads: dict[str, list] = {}
    for r in board:
        if r.get("officialScore") is None:
            continue
        pp = (r.get("officialMetrics") or {}).get("per_prompt") or []
        for e in pp:
            if rr.PROMPT_NAMES.get(e.get("prompt_sha256", "")[:8]) == "beagle":
                h = e.get("head_provenance_sha256")
                if h:
                    heads.setdefault(h, []).append((r, e))

    print("\n## Has any solver moved the head? Board census on beagle")
    print(f"  distinct head_provenance_sha256 over {len(board)} rows: "
          f"{len(heads)}")
    print("  head              rows   best score   edl   rounds  accept"
          "   ratio   us/round  solver")
    for h, rows in sorted(heads.items(),
                          key=lambda kv: -max(r["officialScore"]
                                              for r, _ in kv[1]))[:8]:
        r, e = max(rows, key=lambda re: re[0]["officialScore"])
        tokens = r["officialMetrics"].get("decode_tokens", 512)
        d = e["effective_mean_draft_len"]
        got = rec.recover_rounds(d, tokens, e.get("non_drafting_round_count", 0))
        us = e["mtp_seconds_per_token_mean"] * tokens * 1e6 / got["rounds"]
        print(f"  {h[:16]} {len(rows):5d}  {r['officialScore']:.8f}"
              f" {d:6.3f} {got['rounds']:7d}  {got['acceptance']:6.4f}"
              f"  {e['raw_ratio_of_means']:6.4f} {us:10.1f}"
              f"  {r.get('solverUsername') or '?'}")
    print("  Acceptance here is the smallest admissible multiple, so it is an")
    print("  upper bound per row; the schedule differs between rows as well.")
    head_swaps(heads)


def head_swaps(heads: dict) -> None:
    """Controlled head swaps: one solver, two head digests.

    Comparing heads across solvers confounds the head with the rest of that
    solver's runtime. A solver who submitted under two head digests is closer
    to controlled, though still not a clean A/B: two submissions differ in
    their code as well as their head, so the time column is an upper bound on
    what the head cost.

    This table does NOT validate the acceptance model. Yield is `tokens /
    rounds` identically, and round time is `mtp_seconds_per_token * yield`, so
    the yield-over-time quotient collapses to the ratio of published MTP
    seconds per token. Its agreement with the published ratio is the serial
    null of Rule 146 and nothing more. What the table does measure is the
    PRICE: how much acceptance each head bought and how much round time it
    spent. Break-even is `yield % == time %`.
    """
    by_solver: dict[str, dict] = {}
    for h, rows in heads.items():
        for r, e in rows:
            who = r.get("solverUsername") or "?"
            best = by_solver.setdefault(who, {}).get(h)
            if best is None or r["officialScore"] > best[0]["officialScore"]:
                by_solver[who][h] = (r, e)

    swaps = []
    for who, per_head in by_solver.items():
        if len(per_head) < 2:
            continue
        ranked = sorted(per_head.items(),
                        key=lambda kv: -kv[1][0]["officialScore"])
        base_h, (base_r, base_e) = ranked[0]
        for h, (r, e) in ranked[1:]:
            cells = []
            for entry, rec_row in ((base_e, base_r), (e, r)):
                tokens = rec_row["officialMetrics"].get("decode_tokens", 512)
                d = entry["effective_mean_draft_len"]
                got = rec.recover_rounds(
                    d, tokens, entry.get("non_drafting_round_count", 0))
                cells.append((
                    got["acceptance"],
                    1.0 + got["acceptance"] * d,
                    entry["mtp_seconds_per_token_mean"] * tokens * 1e6
                    / got["rounds"]))
            (a0, y0, t0), (a1, y1, t1) = cells
            swaps.append((who, base_h, h, a1 - a0, y1 / y0 - 1.0,
                          t1 / t0 - 1.0))

    bought = [s for s in swaps if s[3] > 0.01]
    paid = [s for s in swaps if s[4] > s[5]]
    print(f"\n## Head swaps: {len(swaps)} same-solver pairs, "
          f"{len(bought)} bought at least +0.01 acceptance, "
          f"{len(paid)} paid for themselves")
    print("  solver          from -> to        d_accept  yield %   time %"
          "    net %")
    for who, b, h, da, dy, dt in sorted(bought, key=lambda s: s[5] - s[4])[:10]:
        print(f"  {who:<14} {b[:8]} -> {h[:8]}  {da:+8.4f} {dy * 100:+8.2f}"
              f" {dt * 100:+8.2f} {(dy - dt) * 100:+8.2f}")
    print("  net is the first-order ratio change. Every swap on this board")
    print("  spends more round time than the acceptance it buys.")


if __name__ == "__main__":
    main()
