#!/usr/bin/env python3
"""Parse MLX_QWEN_MTP_TRACE round records into schedule statistics.

Emitted by Qwen36MTPBlockSession when Self.traceRounds is set:

    mtp-trace: round=<n> d=<drafts> acc=<accepted> draft_build_us=... \
        verify_build_us=... eval_wall_us=... readout_us=... commit_us=... \
        upkeep_us=... round_us=... m=<margin> streak=<n> cap=<n> \
        ema=<csv> sched=<depth>:<p>/<reach>/<threshold>;...

`d` is the depth actually proposed, so the verified row width is d+1: the
pending primary token plus one row per draft.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) ")
KV_RE = re.compile(r"(\w+)=([^\s]+)")
SCHED_RE = re.compile(r"(\d+):([-\d.eE]+)/([-\d.eE]+)/([-\d.eE]+);")


def parse_round(line):
    kv = dict(KV_RE.findall(line))
    sched_at = line.find("sched=")
    walk = []
    if sched_at >= 0:
        walk = [
            {
                "depth": int(d),
                "p": float(p),
                "reach": float(r),
                "threshold": float(t),
            }
            for d, p, r, t in SCHED_RE.findall(line[sched_at:])
        ]
    rec = {"walk": walk}
    for key in ("round", "d", "acc", "streak", "cap"):
        if key in kv:
            rec[key] = int(kv[key])
    for key in ("m", "round_us", "eval_wall_us", "draft_build_us"):
        if key in kv:
            rec[key] = float(kv[key])
    if "ema" in kv:
        rec["ema"] = [float(x) for x in kv["ema"].split(",") if x]
    return rec


def parse_trace(path):
    rounds = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if ROUND_RE.match(line):
            rounds.append(parse_round(line))
    return rounds


def summarize(rounds):
    depths = [r["d"] for r in rounds if "d" in r]
    accs = [r["acc"] for r in rounds if "acc" in r]
    depth_hist = Counter(depths)
    # Verified row width: the committed primary token plus one row per draft.
    width_hist = Counter(d + 1 for d in depths)
    proposed = sum(depths)
    accepted = sum(accs)
    full = sum(1 for r in rounds if r.get("acc") == r.get("d"))
    declined = sum(1 for d in depths if d == 0)
    caps = Counter(r["cap"] for r in rounds if "cap" in r)
    out = {
        "round_count": len(rounds),
        "depth_histogram": dict(sorted(depth_hist.items())),
        "width_histogram_M": dict(sorted(width_hist.items())),
        "effective_mean_draft_len": round(statistics.fmean(depths), 4) if depths else 0.0,
        "effective_max_draft_len": max(depths) if depths else 0,
        "effective_min_draft_len": min(depths) if depths else 0,
        "proposed_draft_rows": proposed,
        "accepted_draft_tokens": accepted,
        "draft_acceptance_rate_pct": round(100.0 * accepted / proposed, 3) if proposed else None,
        "verified_rows_total": proposed + len(rounds),
        "fully_accepted_round_count": full,
        "rejecting_round_count": len(rounds) - full,
        "non_drafting_round_count": declined,
        "cap_histogram": dict(sorted(caps.items())),
    }
    if depths:
        out["depth_stdev"] = round(statistics.pstdev(depths), 4)
    rej = [r for r in rounds if r.get("acc") != r.get("d")]
    if rej:
        out["rejecting_round_mean_depth"] = round(
            statistics.fmean(r["d"] for r in rej), 4)
        out["accepted_inside_rejecting_rounds"] = sum(r["acc"] for r in rej)
    ful = [r for r in rounds if r.get("acc") == r.get("d")]
    if ful:
        out["fully_accepted_round_mean_depth"] = round(
            statistics.fmean(r["d"] for r in ful), 4)
    margins = [r["m"] for r in rounds if "m" in r]
    if margins:
        out["margin_mean"] = round(statistics.fmean(margins), 4)
        out["margin_median"] = round(statistics.median(margins), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+",
                    help="trace.txt paths, optionally as label=path")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--dump-rounds", help="write per-round records here (JSONL)")
    args = ap.parse_args()

    per_prompt, pooled = {}, []
    for spec in args.traces:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        rounds = parse_trace(path)
        if not rounds:
            print(f"warning: no round records in {path}", file=sys.stderr)
            continue
        per_prompt[label] = summarize(rounds)
        pooled.extend(rounds)
        if args.dump_rounds:
            with open(args.dump_rounds, "a") as fh:
                for r in rounds:
                    fh.write(json.dumps({"prompt": label, **r}) + "\n")

    report = {"per_prompt": per_prompt, "pooled": summarize(pooled) if pooled else {}}
    if len(per_prompt) > 1:
        keys = ("effective_mean_draft_len", "effective_max_draft_len",
                "round_count", "proposed_draft_rows", "draft_acceptance_rate_pct")
        report["spread"] = {
            k: {
                "min": min(v[k] for v in per_prompt.values()),
                "max": max(v[k] for v in per_prompt.values()),
                "range": round(max(v[k] for v in per_prompt.values())
                               - min(v[k] for v in per_prompt.values()), 4),
            }
            for k in keys
        }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
