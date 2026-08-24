#!/usr/bin/env python3
"""E203 shared trace reader: per-round state from a RULE 386 phase trace.

`e128_replay.read_rounds` cannot read these legs. Its `SIGNAL_RE` expects
`streak=N cap=N`, and the e168 traces print `streak=0 offer=8 wcap=7 cap=7`, so
the fixed regex fails and returns zero rounds. This reader parses the trailing
`key=value` fields generically instead, which also survives future field
additions.

The e168 corpus is the only multi-prompt per-round trace set in the repository:

  research/out/e168/p7/<prompt>     fixed_draft_depth=7, 512 decode tokens
  research/out/e168/adapt/<prompt>  shipped adaptive rule, 512 decode tokens

The `p7` arm is the one that answers a question about acceptance dynamics. Its
depth is FIXED, so the accepted-count sequence carries no feedback from the
scheduler's own depth choice; in the `adapt` arm the same sequence is a
closed-loop observation and its serial structure is partly induced by the rule.
"""

from __future__ import annotations

import pathlib
import re

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([^\s]*)")

E168 = pathlib.Path(__file__).resolve().parent / "out" / "e168"


def _scalar(text: str):
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_leg(run_dir: pathlib.Path) -> list[dict]:
    rounds = []
    for line in (run_dir / "trace.txt").read_text(errors="replace").splitlines():
        head = ROUND_RE.match(line)
        if not head:
            continue
        record = {"round": int(head.group(1)), "depth": int(head.group(2)),
                  "accepted": int(head.group(3))}
        for key, value in KV_RE.findall(head.group(4)):
            if key == "ema":
                record["ema"] = [float(v) for v in value.split(",") if v]
            else:
                record[key] = _scalar(value)
        rounds.append(record)
    return rounds


def read_meta(run_dir: pathlib.Path) -> dict:
    out = {}
    for line in (run_dir / "meta.txt").read_text(errors="replace").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def corpus(arm: str) -> dict[str, dict]:
    """{prompt: {"rounds": [...], "meta": {...}}} for one e168 arm."""
    out = {}
    root = E168 / arm
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "trace.txt").exists():
            continue
        rounds = read_leg(path)
        if not rounds:
            continue
        out[path.name] = {"rounds": rounds, "meta": read_meta(path)}
    return out
