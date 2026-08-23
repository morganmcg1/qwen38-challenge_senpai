#!/usr/bin/env python3
"""E160 F1: realized proposed-depth histogram and chunk-sum fill census.

Reads one `MLX_QWEN_MTP_TRACE_PATH` file and reports, per traced leg:

  * the proposed-depth histogram from `mtp-trace: round=N d=D acc=A`,
    where `d` is the proposed draft count the round actually built;
  * `P(d >= 3)`, the fraction of rounds whose verify width `S = 1 + d`
    reaches the `minimumTableWidth = 4` gate and therefore pays chunk-sum
    fills at all;
  * the accepted-draft total and `a = accepted / rounds`, so `R`, the
    seconds per decode round, can be formed exactly;
  * `xs_hit` / `xs_fill` per round, differenced between consecutive rounds
    because the counters are cumulative over the process;
  * the derived-index geometry the leg's own binary built.

`--score` cross-checks the trace against the trusted parent's own scalars.
Only the parent's numbers are authoritative; the trace is candidate-side
telemetry and is used here only because the wrapper deletes the parent's
`mtp-decode.json` with its scratch directory.

    python3 research/e160_depth_census.py --trace research/out/e160f1/trace.txt \
        --score research/out/e160f1/score.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+) .*?"
    r"round_us=(\d+) .*?"
    r"sel_env=(\S+) sel_fused=(\d+) sel_argpart=(\d+) "
    r"leaf=(\d+) leaves=(\d+) probes=(\d+) "
    r"xs_hit=(\d+) xs_fill=(\d+)"
)
BEGIN_RE = re.compile(r"mtp-trace: begin seed=(\d+)")


class Leg:
    def __init__(self, index: int, seed: int) -> None:
        self.index = index
        self.seed = seed
        self.rounds: list[dict] = []

    @property
    def depths(self) -> list[int]:
        return [r["d"] for r in self.rounds]

    @property
    def accepted_total(self) -> int:
        return sum(r["acc"] for r in self.rounds)

    @property
    def proposed_total(self) -> int:
        return sum(r["d"] for r in self.rounds)


def parse(path: pathlib.Path) -> list[Leg]:
    legs: list[Leg] = []
    for line in path.read_text(errors="replace").splitlines():
        begin = BEGIN_RE.search(line)
        if begin:
            legs.append(Leg(len(legs), int(begin.group(1))))
            continue
        m = ROUND_RE.search(line)
        if not m:
            continue
        if not legs:
            legs.append(Leg(0, -1))
        legs[-1].rounds.append({
            "round": int(m.group(1)),
            "d": int(m.group(2)),
            "acc": int(m.group(3)),
            "round_us": int(m.group(4)),
            "sel_env": m.group(5),
            "sel_fused": int(m.group(6)),
            "sel_argpart": int(m.group(7)),
            "leaf": int(m.group(8)),
            "leaves": int(m.group(9)),
            "probes": int(m.group(10)),
            "xs_hit": int(m.group(11)),
            "xs_fill": int(m.group(12)),
        })
    return legs


def per_round_counter(rounds: list[dict], key: str) -> list[int]:
    """Difference a cumulative process counter into per-round increments."""
    return [b[key] - a[key] for a, b in zip(rounds, rounds[1:])]


def describe(leg: Leg) -> dict:
    rounds = leg.rounds
    n = len(rounds)
    hist = collections.Counter(leg.depths)
    deep = sum(c for d, c in hist.items() if d >= 3)
    out = {
        "leg": leg.index,
        "seed": leg.seed,
        "round_count": n,
        "histogram": {str(d): hist.get(d, 0) for d in range(0, 9)},
        "p_d_ge_3": deep / n if n else 0.0,
        "mean_proposed_d": leg.proposed_total / n if n else 0.0,
        "non_drafting_round_count": hist.get(0, 0),
        "accepted_draft_total": leg.accepted_total,
        "proposed_draft_total": leg.proposed_total,
        "a_accepted_per_round": leg.accepted_total / n if n else 0.0,
        "accepted_draft_rate": (leg.accepted_total / leg.proposed_total
                                if leg.proposed_total else 0.0),
        "tokens_implied": n + leg.accepted_total,
    }
    if n >= 2:
        hits = per_round_counter(rounds, "xs_hit")
        fills = per_round_counter(rounds, "xs_fill")
        out["xs_hit_per_round"] = collections.Counter(hits).most_common()
        out["xs_fill_per_round"] = collections.Counter(fills).most_common()
        out["xs_hit_mean"] = sum(hits) / len(hits)
        out["xs_fill_mean"] = sum(fills) / len(fills)
    geometry = {(r["leaf"], r["leaves"], r["probes"], r["sel_env"],
                 r["sel_fused"] > 0) for r in rounds}
    out["geometry"] = sorted(geometry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--score", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    legs = [leg for leg in parse(pathlib.Path(args.trace)) if leg.rounds]
    if not legs:
        print("no traced rounds found", file=sys.stderr)
        return 1

    summaries = [describe(leg) for leg in legs]
    for s in summaries:
        print(f"=== leg {s['leg']}  seed={s['seed']}  "
              f"rounds={s['round_count']} ===")
        print("d      " + "".join(f"{d:>7}" for d in range(0, 9)))
        print("count  " + "".join(f"{s['histogram'][str(d)]:>7}"
                                  for d in range(0, 9)))
        print("frac   " + "".join(
            f"{s['histogram'][str(d)] / s['round_count']:>7.3f}"
            for d in range(0, 9)))
        print(f"P(d >= 3)              = {s['p_d_ge_3']:.4f}")
        print(f"mean proposed d        = {s['mean_proposed_d']:.4f}")
        print(f"non_drafting_rounds    = {s['non_drafting_round_count']}")
        print(f"accepted_draft_total   = {s['accepted_draft_total']}")
        print(f"a (accepted per round) = {s['a_accepted_per_round']:.4f}")
        print(f"accepted_draft_rate    = {s['accepted_draft_rate']:.6f}")
        print(f"tokens implied         = {s['tokens_implied']}"
              "   (rounds + accepted)")
        if "xs_hit_mean" in s:
            print(f"xs_hit  per round      = {s['xs_hit_mean']:.3f}  "
                  f"{s['xs_hit_per_round']}")
            print(f"xs_fill per round      = {s['xs_fill_mean']:.3f}  "
                  f"{s['xs_fill_per_round']}")
        print(f"geometry (leaf, leaves, probes, sel_env, sel_fused>0)"
              f" = {s['geometry']}")
        print()

    if args.score:
        score = json.loads(pathlib.Path(args.score).read_text())
        m = score["metrics"]
        tokens = m["decode_tokens"]
        mtp_spt = float(m["mtp_seconds_per_token"])
        # The MTP leg is the traced leg whose implied token count matches the
        # parent's decode window; the serial control drafts nothing.
        drafting = [s for s in summaries if s["tokens_implied"] == tokens
                    and s["proposed_draft_total"] > 0]
        print("=== parent cross-check (score.json) ===")
        print(f"decode_tokens            = {tokens}")
        print(f"effective_mean_draft_len = {m['effective_mean_draft_len']}")
        print(f"accepted_draft_rate      = {m['accepted_draft_rate']}")
        print(f"mtp_seconds_per_token    = {mtp_spt}")
        print(f"serial_seconds_per_token = {m['serial_seconds_per_token']}")
        if not drafting:
            print("NO traced leg reproduces the parent's token window; the "
                  "trace and the parent disagree and the histogram is void")
            return 2
        s = drafting[-1]
        d_mean_ok = abs(s["mean_proposed_d"]
                        - float(m["effective_mean_draft_len"])) < 5e-4
        rate_ok = abs(s["accepted_draft_rate"]
                      - float(m["accepted_draft_rate"])) < 5e-6
        rounds = tokens - s["accepted_draft_total"]
        r_per_round = mtp_spt * tokens / rounds
        print(f"trace mean proposed d    = {s['mean_proposed_d']:.6f}"
              f"   match={d_mean_ok}")
        print(f"trace accepted rate      = {s['accepted_draft_rate']:.6f}"
              f"   match={rate_ok}")
        print(f"rounds = 512-form        = {rounds}"
              f"   (trace rounds {s['round_count']})")
        print(f"R (seconds per round)    = {r_per_round:.8f}")
        print(f"a from parent window     = "
              f"{s['accepted_draft_total'] / rounds:.6f}")
        if not (d_mean_ok and rate_ok):
            print("TRACE AND PARENT DISAGREE: report the disagreement, not "
                  "the histogram")
            return 2
        for s in summaries:
            s["parent_cross_check"] = {
                "decode_tokens": tokens,
                "effective_mean_draft_len": m["effective_mean_draft_len"],
                "accepted_draft_rate": m["accepted_draft_rate"],
                "mtp_seconds_per_token": mtp_spt,
                "serial_seconds_per_token": m["serial_seconds_per_token"],
                "rounds_from_token_arithmetic": rounds,
                "R_seconds_per_round": r_per_round,
            }

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
