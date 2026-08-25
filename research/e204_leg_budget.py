"""Split a traced E204 leg into seed processing, round-anchored time, and the
time that falls between the round anchors.

RULE 397 asks whether a round-anchored gain reaches the whole leg. That is only
answerable once the leg's seconds per token are decomposed into the part the
round anchors cover and the part they do not. The uncovered part is where a
relocated wait can hide.
"""

import json
import re
import sys

BEGIN = re.compile(r"mtp-trace: begin seed=(\d+) .*?wall_us=(\d+)")
ROUND = re.compile(r"mtp-trace: round=(\d+) d=(\d+) .*?round_us=(\d+)")
SERIAL = re.compile(r"serial_body=(\d)")


def decompose(trace_path, score_path):
    metrics = json.load(open(score_path))["metrics"]
    begins, rounds = [], []
    for line in open(trace_path, errors="replace"):
        b = BEGIN.search(line)
        if b:
            begins.append({"seed": int(b.group(1)),
                           "wall_us": int(b.group(2)),
                           "rounds": []})
            continue
        r = ROUND.search(line)
        if r and begins:
            s = SERIAL.search(line)
            begins[-1]["rounds"].append(
                {"round": int(r.group(1)), "depth": int(r.group(2)),
                 "round_us": int(r.group(3)),
                 "serial_body": int(s.group(1)) if s else 0})
    legs = []
    for leg in begins:
        depths = [r["depth"] for r in leg["rounds"]]
        mtp = max(depths) > 0
        tokens = metrics["decode_tokens"]
        spt = (metrics["mtp_seconds_per_token"] if mtp
               else metrics["serial_seconds_per_token"])
        leg_us = spt * tokens * 1e6
        anchored_us = sum(r["round_us"] for r in leg["rounds"])
        legs.append({
            "kind": "mtp" if mtp else "serial",
            "rounds": len(leg["rounds"]),
            "seed_wall_us": leg["wall_us"],
            "round_anchored_us": anchored_us,
            "leg_us": leg_us,
            "outside_anchors_us": leg_us - anchored_us - leg["wall_us"],
            "anchored_share": anchored_us / leg_us,
            "seed_share": leg["wall_us"] / leg_us,
            "outside_share": (leg_us - anchored_us - leg["wall_us"]) / leg_us,
            "outside_us_per_round": (
                (leg_us - anchored_us - leg["wall_us"]) / len(leg["rounds"])),
        })
    return legs


if __name__ == "__main__":
    out = {}
    for tag in sys.argv[1:]:
        out[tag] = decompose(f"research/out/{tag}/trace.txt",
                             f"research/out/{tag}/score.json")
    print(json.dumps(out, indent=1))
