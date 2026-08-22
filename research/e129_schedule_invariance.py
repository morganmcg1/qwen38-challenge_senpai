#!/usr/bin/env python3
"""How exact is the schedule fingerprint of a ranked receipt?

The E129 validity condition asks whether two receipts ran the same schedule.
The advisor stated it as `within noise`. This measures whether that is the
right form of the test, and it is not: `effective_mean_draft_len` and
`non_drafting_round_count` are bit-identical across archives that share a
scheduler, so the correct test is exact equality and its false-positive rate
is measurable rather than assumed.

    python3 research/e129_schedule_invariance.py [--solver NAME] [--all]

harness=ranked. Reads the board cache written by
`research/board_per_prompt.py fetch`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def per_prompt(row: dict) -> dict:
    return {NAMES[e["prompt_sha256"][:8]]: e
            for e in row["officialMetrics"]["per_prompt"]}


def signature(pp: dict) -> tuple:
    return tuple((round(pp[n]["effective_mean_draft_len"], 9),
                  pp[n]["non_drafting_round_count"])
                 for n in NAMES.values() if n in pp)


def candidate_mean(row: dict) -> float:
    return row["officialMetrics"]["candidate_mtp_seconds_per_token_mean"]


def report(rows: list[dict], who: str) -> None:
    rows = sorted(rows, key=lambda r: r["createdAt"])
    groups: dict[tuple, list[dict]] = collections.OrderedDict()
    for r in rows:
        groups.setdefault(signature(per_prompt(r)), []).append(r)

    print(f"{who}: {len(rows)} receipts with per-prompt data, "
          f"{len(groups)} distinct schedule signatures")
    for i, (sig, rs) in enumerate(groups.items()):
        span_h = 0.0
        if len(rs) > 1:
            import datetime as dt
            a = dt.datetime.fromisoformat(rs[0]["createdAt"].replace("Z", "+00:00"))
            b = dt.datetime.fromisoformat(rs[-1]["createdAt"].replace("Z", "+00:00"))
            span_h = (b - a).total_seconds() / 3600
        means = [candidate_mean(r) for r in rs]
        spread = (max(means) / min(means) - 1) * 100 if len(means) > 1 else 0.0
        print(f"  sig {i}: n={len(rs):3d}  span {span_h:6.1f} h  "
              f"beagle draftlen {sig[0][0]:.9f}  "
              f"candidate-leg spread inside the class {spread:6.3f} %")
        if len(rs) <= 16:
            print("         " + " ".join(r["id"][:8] for r in rs))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", default="morganmcg1")
    ap.add_argument("--all", action="store_true",
                    help="also survey every other solver on the board")
    args = ap.parse_args()

    rows = json.loads(BOARD.read_text())
    scored = [r for r in rows if (r.get("officialMetrics") or {}).get("per_prompt")]

    ours = [r for r in scored if r["solverUsername"] == args.solver]
    report(ours, args.solver)

    if args.all:
        by_solver: dict[str, list[dict]] = collections.defaultdict(list)
        for r in scored:
            by_solver[r["solverUsername"]].append(r)
        print("every solver with at least four scored receipts")
        print(f"{'solver':18s} {'receipts':>8s} {'signatures':>10s} "
              f"{'largest class':>13s} {'spread in it':>12s}")
        survey = []
        for name, rs in sorted(by_solver.items()):
            if len(rs) < 4:
                continue
            groups: dict[tuple, list[dict]] = collections.defaultdict(list)
            for r in rs:
                groups[signature(per_prompt(r))].append(r)
            big = max(groups.values(), key=len)
            means = [candidate_mean(r) for r in big]
            spread = (max(means) / min(means) - 1) * 100 if len(means) > 1 else 0.0
            survey.append((len(rs), name, len(groups), len(big), spread))
        for n, name, sigs, big, spread in sorted(survey, reverse=True):
            print(f"{name:18s} {n:8d} {sigs:10d} {big:13d} {spread:11.3f} %")
        print()

        pooled = 0
        widest = (0.0, "", 0)
        for name, rs in by_solver.items():
            groups: dict[tuple, list[dict]] = collections.defaultdict(list)
            for r in rs:
                groups[signature(per_prompt(r))].append(r)
            for g in groups.values():
                if len(g) < 2:
                    continue
                pooled += len(g)
                means = [candidate_mean(r) for r in g]
                spread = (max(means) / min(means) - 1) * 100
                if spread > widest[0]:
                    widest = (spread, name, len(g))
        print(f"{pooled} receipts sit in a class of two or more, so their "
              f"schedule fields matched another receipt to the last digit.")
        print(f"The widest candidate-leg spread inside any one class is "
              f"{widest[0]:.1f} % ({widest[1]}, n={widest[2]}).")
        print()
        print("A class holds receipts whose two schedule fields are equal to "
              "the last digit. The candidate-leg spread inside a class shows "
              "how much the timed work changed while the schedule did not, "
              "which is what makes exact equality a usable validity test "
              "rather than a tautology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
