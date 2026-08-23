#!/usr/bin/env python3
"""E150 R3 phase 1. Prove the arm fired and prove it did not move the schedule.

This is the Rule 79 clearance gate, and it is a gate rather than a report: if
it fails, the timed legs are a schedule-policy contrast and must not be
published in any direction.

Three claims, each read from the run's own trace rather than from the variable
the driver exported (CAMPAIGN RULE 114):

1.  The arm fired the expected number of times. rb= per round must be 0 under
    `off`, exactly 1 under `firstOnly`, and exactly d= under `perStep`.
2.  The schedule is untouched. The d= sequence must be identical across all
    three arms, round for round, not merely equal in mean.
3.  The decode is untouched. round_count, effective_mean_draft_len,
    accepted_draft_rate and non_drafting_round_count must all agree.

Claim 2 is the one that matters and the one a mean-only check would miss: two
schedules can have the same mean depth and differ everywhere.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ARMS = ("off", "firstOnly", "perStep")
ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) rb=(\d+)")
REPORT_FIELDS = (
    "round_count",
    "effective_mean_draft_len",
    "accepted_draft_rate",
    "non_drafting_round_count",
    "decode_token_count",
)


def read_rounds(trace: pathlib.Path) -> list[tuple[int, int, int, int]]:
    rows = []
    for line in trace.read_text(errors="replace").splitlines():
        m = ROUND_RE.match(line)
        if m:
            rows.append(tuple(int(g) for g in m.groups()))
    return rows


def expected_rb(arm: str, d: int) -> int:
    if arm == "off":
        return 0
    if arm == "firstOnly":
        return 1
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.runs)
    out: dict = {
        "e150_r3_witness_prompt": args.prompt,
        "e150_r3_witness_label": args.label,
        "arms": {},
        "problems": [],
    }

    rounds: dict[str, list] = {}
    reports: dict[str, dict] = {}
    for arm in ARMS:
        leg = root / f"{args.label}w{arm}" / args.prompt
        trace = leg / "trace.txt"
        report = leg / "report.json"
        if not trace.exists():
            out["problems"].append(f"{arm}: no trace at {trace}")
            continue
        rounds[arm] = read_rounds(trace)
        reports[arm] = json.loads(report.read_text()) if report.exists() else {}

    if len(rounds) != len(ARMS):
        out["e150_r3_witness_ok"] = False
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
        for p in out["problems"]:
            print("PROBLEM", p)
        return 1

    # claim 1: the arm fired the right number of times
    for arm in ARMS:
        bad = [
            {"round": r, "d": d, "rb": rb, "want": expected_rb(arm, d)}
            for (r, d, _acc, rb) in rounds[arm]
            if rb != expected_rb(arm, d)
        ]
        rb_total = sum(rb for (_r, _d, _a, rb) in rounds[arm])
        d_total = sum(d for (_r, d, _a, _rb) in rounds[arm])
        out["arms"][arm] = {
            "rounds": len(rounds[arm]),
            "rb_total": rb_total,
            "d_total": d_total,
            "rb_per_round_mean": rb_total / max(1, len(rounds[arm])),
            "rb_mismatched_rounds": len(bad),
            "rb_first_mismatches": bad[:5],
            "report": {
                k: reports[arm].get(k) for k in REPORT_FIELDS
            },
        }
        if bad:
            out["problems"].append(
                f"{arm}: {len(bad)} rounds have rb= that the arm cannot"
                f" explain, first {bad[0]}"
            )

    # the arm must be distinguishable from the control, or the gate is vacuous
    if out["arms"]["off"]["rb_total"] != 0:
        out["problems"].append("off: the control read the device")
    for arm in ("firstOnly", "perStep"):
        if out["arms"][arm]["rb_total"] == 0:
            out["problems"].append(
                f"{arm}: rb_total is 0, so the arm never fired and the"
                " contrast would be off-vs-off"
            )
    if out["arms"]["perStep"]["rb_total"] <= out["arms"]["firstOnly"]["rb_total"]:
        out["problems"].append(
            "perStep did not read more than firstOnly; the two arms are not"
            " separated"
        )

    # claim 2: the schedule is identical round for round
    ref = [d for (_r, d, _a, _rb) in rounds["off"]]
    for arm in ("firstOnly", "perStep"):
        got = [d for (_r, d, _a, _rb) in rounds[arm]]
        if len(got) != len(ref):
            out["problems"].append(
                f"{arm}: {len(got)} rounds vs {len(ref)} on the control"
            )
            continue
        diffs = [i for i, (a, b) in enumerate(zip(ref, got)) if a != b]
        out["arms"][arm]["depth_sequence_diffs"] = len(diffs)
        out["arms"][arm]["depth_sequence_first_diffs"] = diffs[:5]
        if diffs:
            out["problems"].append(
                f"{arm}: the depth sequence differs from the control in"
                f" {len(diffs)} rounds; RULE 79 CLEARANCE FAILS"
            )
    out["arms"]["off"]["depth_sequence_diffs"] = 0

    # claim 3: the decode is identical
    for field in REPORT_FIELDS:
        vals = {arm: reports[arm].get(field) for arm in ARMS}
        if len(set(json.dumps(v) for v in vals.values())) != 1:
            out["problems"].append(f"{field} differs across arms: {vals}")

    out["e150_r3_arm_fired"] = all(
        out["arms"][a]["rb_mismatched_rounds"] == 0 for a in ARMS
    )
    out["e150_r3_schedule_identical"] = all(
        out["arms"][a].get("depth_sequence_diffs") == 0 for a in ARMS
    )
    out["e150_r3_witness_ok"] = not out["problems"]

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("-- e150 r3 witness --")
    for arm in ARMS:
        a = out["arms"][arm]
        print(
            f"  {arm:10s} rounds {a['rounds']:4d}  rb_total {a['rb_total']:5d}"
            f"  rb/round {a['rb_per_round_mean']:6.3f}"
            f"  depth_seq_diffs {a.get('depth_sequence_diffs')}"
            f"  edl {a['report'].get('effective_mean_draft_len')}"
        )
    for p in out["problems"]:
        print("  PROBLEM", p)
    print(f"  witness_ok {out['e150_r3_witness_ok']}")
    return 0 if out["e150_r3_witness_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
