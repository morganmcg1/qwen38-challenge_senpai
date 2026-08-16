"""Explain the final-round replay mismatch: the decode window truncates offeredDepth.

Also restates the FB6 clip statistics with that budget-truncated terminal round
excluded, so the clip rate only counts rounds where the cost model was free.
"""

import json
import re
import sys

TRACES = [
    ("C", "research/trace-runC-base-256.log"),
    ("D", "research/trace-runD-cap7-256.log"),
    ("F", "research/trace-runF-cap7-gate1-256.log"),
]

FB6 = {
    "C": "research/fb6-runC.json",
    "D": "research/fb6-runD.json",
    "F": "research/fb6-runF.json",
}


def restate_clip(label: str) -> None:
    data = json.load(open(FB6[label]))
    rounds = data["per_round"][:-1]
    clipped = [r for r in rounds if r["clipped_by_wall"]]
    total = sum(r["d_wall_open"] - r["observed_d"] for r in clipped)
    matches = sum(1 for r in rounds if r["replay_matches"])
    print(
        f"    free rounds={len(rounds)} replay_match={matches}/{len(rounds)} "
        f"clip_rate={len(clipped) / len(rounds):.4f} clipped={len(clipped)} "
        f"clipped_depth_total={total}"
    )


def main() -> int:
    for label, path in TRACES:
        try:
            text = open(path, errors="replace").read()
        except FileNotFoundError:
            print(f"{label}: MISSING {path}")
            continue
        rounds = re.findall(r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+)", text)
        emitted = 0
        tail = []
        for rnum, depth, acc in rounds:
            before = emitted
            emitted += int(acc) + 1
            tail.append((int(rnum), int(depth), int(acc), before, emitted))
        print(f"=== {label}: rounds={len(rounds)} total_emitted={emitted}")
        for rnum, depth, acc, before, after in tail[-2:]:
            print(
                f"    round={rnum} d={depth} acc={acc} "
                f"emitted_before={before} remaining_before={256 - before} after={after}"
            )
        restate_clip(label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
