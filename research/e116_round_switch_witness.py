#!/usr/bin/env python3
"""E116 rung 0b: prove the arm switch is read once per round, at the round's
realised verify width.

    usage: research/e116_round_switch_witness.py TAG [TAG ...] [--json OUT]

WHY THIS EXISTS. askeladd's E109 v2 within-leg alternating estimator resolves
0.170 % of a round from ONE leg, which is 45x cheaper than any ABBA this
campaign runs. It was unusable for wide-verify arms because his dose flag lived
inside the model forward: at 512 tokens the instrumented boundary saw 380
width-1 forwards and only 12 of 77 timed rounds, and
`round_alignment_verified` came back false. The forward is the wrong boundary
because one round runs the target forward many times, once per verify row and
again for the head.

E116 rung 0b moved the switch into `generateRound`, where it is evaluated once
per round after the round's decisions are fixed. This reducer reads the
`mtp-trace: e116 dose` witness line that the switch writes and reports what the
instrumented boundary actually saw:

  - one line per round, and the count against the round count;
  - the realised verify-width histogram at that boundary, which must be the
    schedule's own histogram and not a column of width-1 forwards;
  - the dosed/undosed assignment, which must be exact round parity.

Reading a trace is free. This check costs no GPU beyond a leg that was going to
run anyway.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def witness(tag: str) -> list[dict[str, int]]:
    path = pathlib.Path("research/out") / tag / "trace.txt"
    if not path.exists():
        sys.exit(f"e116_round_switch_witness: no trace at {path}")
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-trace: e116 dose "):
            continue
        fields = dict(token.partition("=")[::2] for token in line.split()[3:])
        rows.append({key: int(value) for key, value in fields.items()})
    return rows


def sessions(rows: list[dict[str, int]]) -> list[list[dict[str, int]]]:
    """Split the witness at every `round=1`.

    A `--local-iterate` wrapper leg holds TWO block sessions in one process:
    the serial control leg at depth 0, whose rounds are all width 1, and the
    MTP leg at the offered depth. Each session restarts `roundCount` at 1. A
    single histogram over the whole file therefore reads as though the switch
    saw hundreds of width-1 rounds, which is exactly the artefact E109 v1 was
    accused of. Splitting the sessions shows what each leg really saw. A
    `mtp-timed` leg holds one session and is unaffected.
    """
    out: list[list[dict[str, int]]] = []
    for row in rows:
        if row["round"] == 1 or not out:
            out.append([])
        out[-1].append(row)
    return out


def round_lines(tag: str) -> list[dict[str, str]]:
    path = pathlib.Path("research/out") / tag / "trace.txt"
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        fields = dict(token.partition("=")[::2]
                      for token in line.split()[1:] if "=" in token)
        rows.append(fields)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()

    out = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": "0b",
        "boundary": "Qwen36MTPBlockSession.generateRound, read once per round "
                    "after the round's token and row decisions are fixed",
        "e109_v1_boundary": "inside the model forward: 380 width-1 forwards at "
                            "512 tokens, 12 of 77 rounds instrumented",
        "legs": [],
    }

    failures = 0
    for tag in args.tags:
        rows = witness(tag)
        if not rows:
            print(f"{tag}: NO witness lines; the switch did not run")
            failures += 1
            continue
        widths = collections.Counter(row["width"] for row in rows)
        alternate = all(row["alternate"] for row in rows)
        # With alternation off the switch is armed on every round by design, so
        # parity is only the claim when the leg asked for alternation.
        parity_exact = all(
            bool(row["dosed"]) == (row["round"] % 2 == 0 if alternate else True)
            for row in rows)
        traced = round_lines(tag)
        leg = {
            "tag": tag,
            "witness_lines": len(rows),
            "width_histogram": dict(sorted(widths.items())),
            "width_one_lines": widths.get(1, 0),
            "distinct_widths": len(widths),
            "alternating": alternate,
            "dosed_lines": sum(row["dosed"] for row in rows),
            "units_when_dosed": sorted({row["units"] for row in rows
                                        if row["dosed"]}),
            "assignment_is_exact_round_parity": parity_exact,
            "mtp_trace_round_lines": len(traced),
            "one_line_per_round": len(rows) == len(traced) or not traced,
            "sessions": [
                {"rounds": len(part),
                 "width_histogram": dict(sorted(collections.Counter(
                     row["width"] for row in part).items())),
                 "dosed": sum(row["dosed"] for row in part)}
                for part in sessions(rows)],
        }
        out["legs"].append(leg)
        if not parity_exact:
            failures += 1
        print(f"{tag}: {len(rows)} witness lines,"
              f" dosed {leg['dosed_lines']},"
              f" units {leg['units_when_dosed']},"
              f" parity exact {parity_exact}")
        for index, part in enumerate(leg["sessions"]):
            kind = ("serial control leg, depth 0"
                    if set(part["width_histogram"]) == {1} else
                    "MTP leg at the offered depth")
            print(f"    session {index}: {part['rounds']:>4} rounds,"
                  f" {kind}, widths {part['width_histogram']}")

    out["failures"] = failures
    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
