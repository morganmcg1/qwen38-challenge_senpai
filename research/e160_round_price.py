#!/usr/bin/env python3
"""E160: price the arm contrast per decode round and per removed fill.

`research/e160_fuse.py report` prices the arms against the whole timed leg,
which on this host is 27.05 % seed prefill. The prefill runs one forward over
the seed; the fill count this arm removes is per forward, so the arm can only
act on the decode part. This script restates the same measured means against
the decode-only round and converts the contrast into microseconds per removed
fill.

The round count is not assumed. `effective_mean_draft_len` and
`accepted_draft_rate` are exact rationals over the leg's own integers:

    edl = proposed / rounds        acc = accepted / proposed

so a small-denominator reconstruction recovers `rounds`, `proposed` and
`accepted` from the parent's own scalars, for untraced legs too.

    python3 research/e160_round_price.py --label fuse --prefill 4.0042
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
from fractions import Fraction

ARMS = ("off", "replica", "fuse")
REMOVED_FILLS_PER_ROUND = 64


def reconstruct_rounds(edl: float, acc: float, tokens: int) -> tuple[int, int, int]:
    """Recover (rounds, proposed, accepted) from the parent's two ratios."""
    for rounds in range(1, tokens + 1):
        proposed = edl * rounds
        if abs(proposed - round(proposed)) > 1e-6:
            continue
        proposed = round(proposed)
        if Fraction(proposed, rounds) != Fraction(edl).limit_denominator(tokens):
            continue
        accepted = acc * proposed
        if abs(accepted - round(accepted)) > 1e-6:
            continue
        accepted = round(accepted)
        if Fraction(accepted, proposed) != Fraction(acc).limit_denominator(tokens):
            continue
        return rounds, proposed, accepted
    raise SystemExit(f"no small-denominator round count fits edl={edl} acc={acc}")


def load(label: str) -> list[dict]:
    out = []
    for score_path in sorted(glob.glob(f"research/out/e160{label}k*/score.json")):
        metrics = json.load(open(score_path))["metrics"]
        meta_path = pathlib.Path(score_path).with_name("meta.txt")
        meta = dict(
            line.strip().split("=", 1)
            for line in open(meta_path)
            if "=" in line
        )
        rounds, proposed, accepted = reconstruct_rounds(
            metrics["effective_mean_draft_len"],
            metrics["accepted_draft_rate"],
            metrics["decode_tokens"],
        )
        out.append(
            {
                "tag": meta["tag"],
                "arm": meta["e160_arm"],
                "rep": int(meta.get("e160_rep", 1)),
                "position": int(meta.get("e160_position", 0)),
                "mtp": metrics["mtp_seconds_per_token"],
                "tokens": metrics["decode_tokens"],
                "matched": metrics["all_tokens_matched"],
                "rounds": rounds,
                "proposed": proposed,
                "accepted": accepted,
                "entry_c": float(meta["gpu_temp_entry_c"]),
                "exit_c": float(meta["gpu_temp_exit_c"]),
                "real_gate": meta["cool_gate_passed_real_gate"],
                "qualified": meta["gate_qualified_for_timing"],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="fuse")
    ap.add_argument("--prefill", type=float, default=4.0042,
                    help="seconds of seed prefill inside the timed leg")
    ap.add_argument("--two-se", type=float, default=None,
                    help="2se of the fuse-off contrast, per cent of the leg")
    args = ap.parse_args()

    legs = load(args.label)
    if not legs:
        raise SystemExit(f"no legs under research/out/e160{args.label}k*")

    print("## Legs: round ledger reconstructed from the parent's own ratios")
    for d in legs:
        print(
            f"  k{d['rep']}p{d['position']} {d['arm']:<8} rounds {d['rounds']} "
            f"proposed {d['proposed']} accepted {d['accepted']} "
            f"tokens {d['tokens']} matched {d['matched']} "
            f"entry {d['entry_c']:.2f}C exit {d['exit_c']:.2f}C "
            f"real_gate {d['real_gate']} qualified {d['qualified']}"
        )
    ledgers = {(d["rounds"], d["proposed"], d["accepted"]) for d in legs}
    if len(ledgers) != 1:
        print(f"\nVOID: the arms consumed different accept ledgers: {ledgers}")
        return 1
    rounds, proposed, accepted = ledgers.pop()
    print(f"\n  every leg: {rounds} rounds, {proposed} proposed, {accepted} accepted "
          f"-> the accept ledger is identical and the timing contrast is valid")

    tokens = legs[0]["tokens"]
    mean = {
        arm: statistics.fmean(d["mtp"] for d in legs if d["arm"] == arm)
        for arm in ARMS
        if any(d["arm"] == arm for d in legs)
    }

    print(f"\n## Per round, prefill {args.prefill:.4f} s removed, {rounds} rounds")
    for arm, mtp in mean.items():
        leg_s = mtp * tokens
        decode_s = leg_s - args.prefill
        print(
            f"  {arm:<8} mtp {mtp:.9f}  leg {leg_s:.4f} s  decode {decode_s:.4f} s  "
            f"R_leg {leg_s / rounds * 1e3:.3f} ms  R_decode {decode_s / rounds * 1e3:.3f} ms"
        )

    print("\n## Contrasts")
    for better, base, name in (
        ("fuse", "off", "fuse - off      shippable"),
        ("replica", "off", "replica - off   kernel swap"),
        ("fuse", "replica", "fuse - replica  fill removal"),
    ):
        if better not in mean or base not in mean:
            continue
        saved_s = (mean[base] - mean[better]) * tokens
        us_round = saved_s / rounds * 1e6
        decode_base = mean[base] * tokens - args.prefill
        print(
            f"  {name:<30} {saved_s / (mean[base] * tokens) * 100:+.4f} % of the leg   "
            f"{saved_s / decode_base * 100:+.4f} % of decode   "
            f"{us_round:+8.1f} us/round   {us_round / REMOVED_FILLS_PER_ROUND:+.3f} us per removed fill"
        )

    if args.two_se is not None:
        band_us = args.two_se / 100 * mean["off"] * tokens / rounds * 1e6
        print(
            f"\n  2se {args.two_se:.4f} pp of the leg = {band_us:.1f} us/round = "
            f"{band_us / REMOVED_FILLS_PER_ROUND:.2f} us per removed fill"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
