#!/usr/bin/env python3
"""E100: convert the diluted leg time into a per-round decode cost.

Every `--local-iterate` timed leg carries the same 512-token seed prefill INSIDE
the timed window (`QwenRuntimeBenchmark.swift` starts `decodePhaseStart` before
the seed forward). `seconds_per_token` is therefore

    seconds_per_token = (prefill + rounds * round_cost) / tokens

and at the wrapper default of 64 decode tokens the prefill is most of the leg.
A dispatch change that can only touch `round_cost` is diluted by that ratio
before it reaches the reported number, which makes the leg a blunt instrument.

Two serial controls at different token counts pin the prefill exactly, because
the serial leg runs one target forward per token with no drafting:

    prefill + 64  * serial_round = 64  * serial_seconds_per_token
    prefill + 512 * serial_round = 512 * serial_seconds_per_token

With `prefill` known, every MTP leg yields its own round cost, and the round
cost is the quantity the change is allowed to move. Round counts come from the
`mtp-timed:` line in each leg log, not from a reconstruction.

  python3 research/e100_round_model.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

OUT = pathlib.Path("research/out")
ROUNDS = re.compile(r"^mtp-timed: tokens=(\d+) depth=(\d+) rounds=(\d+)",
                    re.MULTILINE)

# tag infix -> (offered depth, decode tokens)
SESSIONS = {"d8": (8, 64), "w512": (8, 512),
            "w512d4": (4, 512)}
SLOT_ARM = {"a1": "collapse", "a2": "collapse", "b1": "base", "b2": "base"}


def legs() -> list[dict]:
    found = []
    for directory in sorted(OUT.glob("e100-e2e-*")):
        parts = directory.name.split("-")
        infix, slot = parts[-2], parts[-1]
        if infix not in SESSIONS or slot not in SLOT_ARM:
            continue
        score_path = directory / "score.json"
        log_path = directory / "run.log"
        if not score_path.exists() or not log_path.exists():
            continue
        score = json.loads(score_path.read_text())["metrics"]
        rounds = {int(d): int(r)
                  for _, d, r in ROUNDS.findall(log_path.read_text())}
        depth, tokens = SESSIONS[infix]
        if depth not in rounds:
            continue
        found.append(dict(
            tag=directory.name, session=infix, slot=slot, arm=SLOT_ARM[slot],
            depth=depth, tokens=tokens, rounds=rounds[depth],
            mtp_spt=score["mtp_seconds_per_token"],
            serial_spt=score["serial_seconds_per_token"],
            mean_draft=score["effective_mean_draft_len"],
            accept=score["accepted_draft_rate"],
        ))
    return found


def solve_prefill(rows: list[dict]) -> tuple[float, float] | None:
    """Return (prefill_seconds, serial_round_seconds) from the serial controls."""
    by_tokens: dict[int, list[float]] = {}
    for row in rows:
        by_tokens.setdefault(row["tokens"], []).append(row["serial_spt"])
    if len(by_tokens) < 2:
        return None
    points = sorted((t, sum(v) / len(v)) for t, v in by_tokens.items())
    (t0, s0), (t1, s1) = points[0], points[-1]
    serial_round = (t1 * s1 - t0 * s0) / (t1 - t0)
    return t0 * s0 - t0 * serial_round, serial_round


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="out")
    args = parser.parse_args()

    rows = legs()
    if not rows:
        print("no legs found under research/out")
        return 2

    print("=" * 100)
    print("SERIAL CONTROLS -- one target forward per token, no drafting")
    print("=" * 100)
    prefill = {}
    for arm in ("base", "collapse"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        for tokens in sorted({r["tokens"] for r in arm_rows}):
            v = [r["serial_spt"] for r in arm_rows if r["tokens"] == tokens]
            print("  %-9s %3d tok  serial s/tok %.6f  (n=%d, spread %.3f %%)"
                  % (arm, tokens, sum(v) / len(v), len(v),
                     100.0 * (max(v) - min(v)) / (sum(v) / len(v))))
        solved = solve_prefill(arm_rows)
        if solved is None:
            continue
        prefill[arm] = solved
        print("  %-9s prefill %.3f s   serial round %.2f ms"
              % (arm, solved[0], 1000.0 * solved[1]))

    if not prefill:
        print("\nneed serial controls at two token counts to pin the prefill")
        return 2
    pooled = sum(p for p, _ in prefill.values()) / len(prefill)
    print("  pooled prefill across arms: %.3f s "
          "(the seed forward is identical in both arms)" % pooled)

    print()
    print("=" * 100)
    print("PER-ROUND DECODE COST")
    print("=" * 100)
    print("  %-20s %-9s %6s %7s %7s %9s %11s"
          % ("leg", "arm", "M_mean", "rounds", "tok", "leg_s", "round_ms"))
    for row in sorted(rows, key=lambda r: (r["session"], r["slot"])):
        leg_s = row["tokens"] * row["mtp_spt"]
        row["round_ms"] = 1000.0 * (leg_s - pooled) / row["rounds"]
        row["m_mean"] = row["mean_draft"] + 1.0
        print("  %-20s %-9s %6.3f %7d %7d %9.3f %11.2f"
              % (row["tag"], row["arm"], row["m_mean"], row["rounds"],
                 row["tokens"], leg_s, row["round_ms"]))

    print()
    print("=" * 100)
    print("ARM COMPARISON AT MATCHED WIDTH")
    print("=" * 100)
    summary = {}
    for session in SESSIONS:
        arms = {}
        for arm in ("base", "collapse"):
            v = [r["round_ms"] for r in rows
                 if r["session"] == session and r["arm"] == arm]
            if v:
                arms[arm] = (sum(v) / len(v), max(v) - min(v), len(v))
        if len(arms) < 2:
            continue
        b, c = arms["base"][0], arms["collapse"][0]
        widths = {r["m_mean"] for r in rows if r["session"] == session}
        delta = 100.0 * (c / b - 1.0)
        summary[session] = dict(base_round_ms=b, collapse_round_ms=c,
                                delta_pct=delta,
                                base_spread_ms=arms["base"][1],
                                collapse_spread_ms=arms["collapse"][1],
                                mean_verify_width=max(widths))
        print("  %-6s M_mean %.3f  base %7.2f ms  collapse %7.2f ms  "
              "delta %+7.3f %%  (within-arm spread base %.2f ms, "
              "collapse %.2f ms)"
              % (session, max(widths), b, c, delta,
                 arms["base"][1], arms["collapse"][1]))

    serial_round = sum(s for _, s in prefill.values()) / len(prefill)
    print()
    print("=" * 100)
    print("WIDTH LAW -- what the extra verify rows actually cost")
    print("=" * 100)
    print("  serial round (M = 1): %.2f ms" % (1000.0 * serial_round))
    pts = sorted((r["m_mean"], r["round_ms"]) for r in rows
                 if r["arm"] == "base")
    if len(pts) >= 2:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        var = sum((p[0] - mx) ** 2 for p in pts)
        slope = (sum((p[0] - mx) * (p[1] - my) for p in pts) / var
                 if var else float("nan"))
        intercept = my - slope * mx
        print("  base fit: round_ms = %.2f + %.2f * M   (n=%d)"
              % (intercept, slope, n))
        print("  cost of the target forward, extrapolated to M = 1: %.2f ms"
              % (intercept + slope))
        print("  each extra verify row costs %.2f ms, against a %.2f ms "
              "serial round" % (slope, 1000.0 * serial_round))

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(dict(
            prefill_seconds=pooled,
            serial_round_seconds=serial_round,
            per_arm_prefill={k: dict(prefill_s=v[0], serial_round_s=v[1])
                             for k, v in prefill.items()},
            sessions=summary,
            legs=rows,
        ), indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
