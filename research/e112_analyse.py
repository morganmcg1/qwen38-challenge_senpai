#!/usr/bin/env python3
"""Analyse an E112 rung-1 ABBA session.

    usage: research/e112_analyse.py [--label r1] [--json OUT.json]

Reads every `research/out/e112<label>p*` leg and reports two things.

1. The arm contrast on absolute candidate MTP seconds per token, which is the
   quantity the ranked numerator responds to. The local serial-to-MTP ratio is
   also printed, but this arm changes only untimed warm work, so the ratio is
   the weaker readout here.

2. The mechanism itself. The kL=1025 warm exists to compile the
   `sdpa_vector_2pass` pipeline that Metal function constant 26 (`blocks`)
   selects when the key length passes 1024. This script rebuilds the key length
   of every timed round from the trace, names the rounds that cross, and prints
   their `round_us` against the leg's own median. A cold pipeline inside the
   scored window has to appear there or it does not exist.

Key length of round r is the context before the round plus the verify width:

    keys_r = ctx_{r-1} + d_r + 1        ctx_r = ctx_{r-1} + acc_r + 1

with ctx_0 the seed length, because the pending primary is not yet in the
cache at the top of a round.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND_RE = re.compile(
    r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?round_us=(\d+)")
BEGIN_RE = re.compile(r"^mtp-trace: begin seed=(\d+)")
WITNESS_RE = re.compile(r"^mtp-trace: e112 skip_1025_warm=([01]) kL=(\d+)")
BLOCKS_BOUNDARY = 1024


def read_meta(path):
    meta = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.strip().partition("=")
                meta[key] = value
    return meta


def read_trace(path):
    """Split the append-only trace into per-session round blocks."""
    blocks = []
    current = None
    witnesses = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            witness = WITNESS_RE.match(line)
            if witness:
                witnesses.append((int(witness.group(1)), int(witness.group(2))))
                continue
            begin = BEGIN_RE.match(line)
            if begin:
                current = {"seed": int(begin.group(1)), "rounds": []}
                blocks.append(current)
                continue
            match = ROUND_RE.match(line)
            if match and current is not None:
                current["rounds"].append({
                    "round": int(match.group(1)),
                    "d": int(match.group(2)),
                    "acc": int(match.group(3)),
                    "round_us": int(match.group(4)),
                })
    return blocks, witnesses


def annotate(block):
    ctx = block["seed"]
    crossings = []
    for entry in block["rounds"]:
        entry["keys"] = ctx + entry["d"] + 1
        ctx += entry["acc"] + 1
        entry["ctx_after"] = ctx
        if entry["keys"] > BLOCKS_BOUNDARY:
            crossings.append(entry)
    block["final_ctx"] = ctx
    block["crossings"] = crossings
    return block


def pick_blocks(blocks):
    """The last serial (all d==0) and last speculative (any d>0) block."""
    serial = None
    mtp = None
    for block in blocks:
        if not block["rounds"]:
            continue
        if all(entry["d"] == 0 for entry in block["rounds"]):
            serial = block
        else:
            mtp = block
    return serial, mtp


def load_leg(directory):
    meta_path = os.path.join(directory, "meta.txt")
    score_path = os.path.join(directory, "score.json")
    if not os.path.exists(meta_path) or not os.path.exists(score_path):
        return None
    meta = read_meta(meta_path)
    with open(score_path, encoding="utf-8") as handle:
        score = json.load(handle)
    metrics = score.get("metrics", {})
    leg = {
        "tag": os.path.basename(directory),
        "arm": meta.get("e112_arm"),
        "position": int(meta.get("e112_position", 0)),
        "flag": meta.get("MLX_E112_SKIP_1025_WARM"),
        "witness_flag": meta.get("e112_warm_witness_flag"),
        "status": meta.get("status"),
        "tokens": int(meta.get("tokens", 0)),
        "entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
        "exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
        "worker_sha256": meta.get("worker_sha256"),
        "session_commit": meta.get("session_commit"),
        "mtp_s_per_tok": metrics.get("mtp_seconds_per_token"),
        "serial_s_per_tok": metrics.get("serial_seconds_per_token"),
        "speedup": metrics.get("mtp_decode_speedup"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "mean_draft_len": metrics.get("effective_mean_draft_len"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "decode_tokens": metrics.get("decode_tokens"),
    }
    trace_path = os.path.join(directory, "trace.txt")
    if os.path.exists(trace_path):
        blocks, witnesses = read_trace(trace_path)
        blocks = [annotate(block) for block in blocks]
        serial, mtp = pick_blocks(blocks)
        leg["witnesses"] = witnesses
        for name, block in (("serial", serial), ("mtp", mtp)):
            if block is None:
                continue
            times = [entry["round_us"] for entry in block["rounds"]]
            leg[f"{name}_rounds"] = len(times)
            leg[f"{name}_median_round_us"] = statistics.median(times)
            leg[f"{name}_max_key_len"] = max(
                entry["keys"] for entry in block["rounds"])
            leg[f"{name}_crossings"] = [
                {
                    "round": entry["round"],
                    "keys": entry["keys"],
                    "round_us": entry["round_us"],
                    "excess_us": entry["round_us"]
                    - statistics.median(times),
                }
                for entry in block["crossings"]
            ]
    return leg


def summarise(legs, key):
    out = {}
    for arm in ("off", "on"):
        values = [leg[key] for leg in legs
                  if leg["arm"] == arm and leg.get(key) is not None]
        if not values:
            continue
        out[arm] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    if "off" in out and "on" in out:
        base = out["off"]["mean"]
        cand = out["on"]["mean"]
        out["delta"] = cand - base
        out["delta_pct"] = 100.0 * (cand - base) / base if base else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="r1")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    pattern = os.path.join(ROOT, "research", "out", f"e112{args.label}p*")
    legs = [leg for leg in (load_leg(d) for d in sorted(glob.glob(pattern)))
            if leg]
    if not legs:
        print(f"no legs matched {pattern}", file=sys.stderr)
        return 1
    legs.sort(key=lambda leg: leg["position"])

    print(f"{'tag':<14}{'arm':<5}{'pos':<5}{'wit':<6}{'mtp s/tok':>12}"
          f"{'serial s/tok':>14}{'ratio':>8}{'in C':>7}{'out C':>7}"
          f"{'maxkeys':>9}{'cross':>7}")
    for leg in legs:
        cross = len(leg.get("mtp_crossings", []))
        print(f"{leg['tag']:<14}{str(leg['arm']):<5}{leg['position']:<5}"
              f"{str(leg['witness_flag']):<6}"
              f"{leg['mtp_s_per_tok']:>12.6f}{leg['serial_s_per_tok']:>14.6f}"
              f"{leg['speedup']:>8.4f}{leg['entry_c']:>7.1f}"
              f"{leg['exit_c']:>7.1f}"
              f"{leg.get('mtp_max_key_len', 0):>9}{cross:>7}")

    report = {
        "legs": legs,
        "mtp_s_per_tok": summarise(legs, "mtp_s_per_tok"),
        "serial_s_per_tok": summarise(legs, "serial_s_per_tok"),
        "speedup": summarise(legs, "speedup"),
        "mtp_median_round_us": summarise(legs, "mtp_median_round_us"),
        "entry_c": summarise(legs, "entry_c"),
    }

    print("\n-- arm contrast (candidate arm `on` deletes the kL=1025 warm) --")
    for key in ("mtp_s_per_tok", "serial_s_per_tok", "speedup",
                "mtp_median_round_us", "entry_c"):
        block = report[key]
        if "delta_pct" not in block:
            continue
        print(f"{key:<22} off={block['off']['mean']:.6f} "
              f"(sd {block['off']['sd']:.6f}, n {block['off']['n']})  "
              f"on={block['on']['mean']:.6f} "
              f"(sd {block['on']['sd']:.6f}, n {block['on']['n']})  "
              f"delta={block['delta_pct']:+.4f} %")

    print("\n-- key-length crossings inside the timed MTP leg --")
    for leg in legs:
        crossings = leg.get("mtp_crossings", [])
        detail = ", ".join(
            f"r{entry['round']} keys={entry['keys']} "
            f"round_us={entry['round_us']} excess={entry['excess_us']:+.0f}"
            for entry in crossings) or "none"
        print(f"{leg['tag']:<14}{str(leg['arm']):<5}"
              f"median={leg.get('mtp_median_round_us', 0):>9.0f}  {detail}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
