#!/usr/bin/env python3
"""Analyse an E136 rung-2 ABBA session.

    usage: research/e136_analyse.py [--label r2] [--json OUT.json]

Reads every `research/out/e136<label>p*` leg and reports:

1. The headline. Absolute candidate MTP seconds per token, which is the only
   quantity the ranked numerator responds to. Arm C1 changes work that runs
   ONLY inside the candidate MTP leg, so the local serial-to-MTP ratio is a
   valid second readout here rather than a cancelling one. Both are printed
   and the absolute number leads.

2. The mechanism. Draft depth, accepted tokens, realised acceptance and the
   per-round latency split, so a change in absolute time can be attributed to
   the shortlist rather than to a different round mix.

3. The exactness fields the arm must not move: `all_tokens_matched` and
   `residual_divergence_count` on the accepted stream.

The arm witness comes from meta.txt, which `research/e136_abba.sh` fills from
the trace: a C1 leg must show a positive `e136_c1_draft_steps` with
`e136_shipped_selection_draft_steps` at zero, and a base leg the reverse.
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
    r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?"
    r"draft_build_us=(\d+) .*?round_us=(\d+)")
BEGIN_RE = re.compile(r"^mtp-trace: begin seed=(\d+)")

# One added microsecond per draft step is this many ranked percent, from the
# E136 rung-0 conversion: 174.1 us per draft step per ranked percent.
US_PER_DRAFT_STEP_PER_PCT = 174.1


def read_meta(path):
    meta = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.strip().partition("=")
                meta[key] = value
    return meta


def read_trace(path):
    blocks = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
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
                    "draft_build_us": int(match.group(4)),
                    "round_us": int(match.group(5)),
                })
    return blocks


def pick_mtp_block(blocks):
    """The last block that actually drafted."""
    chosen = None
    for block in blocks:
        if block["rounds"] and any(e["d"] > 0 for e in block["rounds"]):
            chosen = block
    return chosen


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
        "arm": meta.get("e136_arm"),
        "position": int(meta.get("e136_position", 0)),
        "flag": meta.get("MLX_E136_C1_SKETCH"),
        "c1_draft_steps": int(meta.get("e136_c1_draft_steps", 0)),
        "shipped_selection_draft_steps":
            int(meta.get("e136_shipped_selection_draft_steps", 0)),
        "status": meta.get("status"),
        "tokens": int(meta.get("tokens", 0)),
        "entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
        "exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
        "worker_sha256": meta.get("worker_sha256"),
        "session_commit": meta.get("session_commit"),
        "cool_gate_passed_real_gate":
            meta.get("cool_gate_passed_real_gate") == "true",
        "gate_qualified_for_timing":
            meta.get("gate_qualified_for_timing") == "true",
        "mtp_s_per_tok": metrics.get("mtp_seconds_per_token"),
        "serial_s_per_tok": metrics.get("serial_seconds_per_token"),
        "speedup": metrics.get("mtp_decode_speedup"),
        "accepted_draft_rate": metrics.get("accepted_draft_rate"),
        "mean_draft_len": metrics.get("effective_mean_draft_len"),
        "all_tokens_matched": metrics.get("all_tokens_matched"),
        "residual_divergence_count": metrics.get("residual_divergence_count"),
        "decode_tokens": metrics.get("decode_tokens"),
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
    }
    leg["arm_witnessed"] = (
        (leg["c1_draft_steps"] > 0 and leg["shipped_selection_draft_steps"] == 0)
        if leg["flag"] == "1"
        else (leg["c1_draft_steps"] == 0
              and leg["shipped_selection_draft_steps"] > 0))

    trace_path = os.path.join(directory, "trace.txt")
    if os.path.exists(trace_path):
        block = pick_mtp_block(read_trace(trace_path))
        if block is not None:
            rounds = block["rounds"]
            leg["rounds"] = len(rounds)
            leg["median_round_us"] = statistics.median(
                e["round_us"] for e in rounds)
            leg["median_draft_build_us"] = statistics.median(
                e["draft_build_us"] for e in rounds)
            leg["mean_d"] = statistics.mean(e["d"] for e in rounds)
            leg["mean_acc"] = statistics.mean(e["acc"] for e in rounds)
            # Realised acceptance: accepted drafts over drafts proposed. This
            # is the rule-107 quantity, not one minus a shortlist miss rate.
            proposed = sum(e["d"] for e in rounds)
            leg["realised_acceptance"] = (
                sum(e["acc"] for e in rounds) / proposed if proposed else None)
            leg["draft_steps"] = proposed + len(rounds)
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
    parser.add_argument("--label", default="r2")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    pattern = os.path.join(ROOT, "research", "out", f"e136{args.label}p*")
    legs = [leg for leg in (load_leg(d) for d in sorted(glob.glob(pattern)))
            if leg]
    if not legs:
        print(f"no legs matched {pattern}", file=sys.stderr)
        return 1
    legs.sort(key=lambda leg: leg["position"])

    print(f"{'tag':<14}{'arm':<5}{'pos':<4}{'wit':<5}{'mtp s/tok':>12}"
          f"{'serial s/tok':>14}{'ratio':>8}{'d':>6}{'acc':>6}"
          f"{'in C':>7}{'out C':>7}{'match':>7}{'div':>5}")
    for leg in legs:
        print(f"{leg['tag']:<14}{str(leg['arm']):<5}{leg['position']:<4}"
              f"{'y' if leg['arm_witnessed'] else 'NO':<5}"
              f"{leg['mtp_s_per_tok']:>12.6f}{leg['serial_s_per_tok']:>14.6f}"
              f"{leg['speedup']:>8.4f}{leg.get('mean_d', 0):>6.2f}"
              f"{leg.get('mean_acc', 0):>6.2f}"
              f"{leg['entry_c']:>7.1f}{leg['exit_c']:>7.1f}"
              f"{str(leg['all_tokens_matched']):>7}"
              f"{str(leg['residual_divergence_count']):>5}")

    keys = ("mtp_s_per_tok", "serial_s_per_tok", "speedup", "mean_draft_len",
            "realised_acceptance", "median_round_us", "median_draft_build_us",
            "entry_c")
    report = {
        "legs": legs,
        "label": args.label,
        "all_arms_witnessed": all(leg["arm_witnessed"] for leg in legs),
        "all_tokens_matched": all(leg["all_tokens_matched"] for leg in legs),
        "accepted_stream_divergences": sum(
            leg["residual_divergence_count"] or 0 for leg in legs),
        "entry_temp_spread_c": (
            max(leg["entry_c"] for leg in legs)
            - min(leg["entry_c"] for leg in legs)),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }
    for key in keys:
        report[key] = summarise(legs, key)

    print("\n-- arm contrast (`on` is the C1 sketch shortlist) --")
    for key in keys:
        block = report[key]
        if "delta_pct" not in block:
            continue
        print(f"{key:<26} off={block['off']['mean']:.6f} "
              f"(sd {block['off']['sd']:.6f}, n {block['off']['n']})  "
              f"on={block['on']['mean']:.6f} "
              f"(sd {block['on']['sd']:.6f}, n {block['on']['n']})  "
              f"delta={block['delta_pct']:+.4f} %")

    # The headline is stated on the absolute candidate leg, sign flipped so a
    # faster candidate reads positive.
    mtp = report["mtp_s_per_tok"]
    if "delta_pct" in mtp:
        report["e136_c1_candidate_leg_pct"] = -mtp["delta_pct"]
        acc = report["realised_acceptance"]
        if "delta" in acc:
            report["e136_realised_acceptance_delta_pp"] = 100.0 * acc["delta"]
        print(f"\ne136_c1_candidate_leg_pct = "
              f"{report['e136_c1_candidate_leg_pct']:+.4f} "
              f"(positive means the C1 candidate leg is faster)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
