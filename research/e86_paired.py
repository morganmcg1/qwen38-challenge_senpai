#!/usr/bin/env python3
"""E86: paired per-round comparison of decode-ladder arms.

usage: research/e86_paired.py PREFIX [--ref default]

The ladder is a pure enqueue-timing control, so every arm replays the identical
round sequence: round i has the same draft width and the same accepted count in
every leg. That makes the arms PAIRABLE round by round, which removes all
work-mix variation and is far more powerful than comparing leg totals.

It also matters because the leg totals are contaminated. Host phases outside
the verify window (`d_submit1_us`, `d_chain_us`, `commit_us`) show occasional
multi-millisecond spikes that are OS scheduling jitter, not ladder behaviour. A
mean over rounds inherits those spikes; a median of paired differences does
not. The script reports both so the contamination is visible instead of
assumed.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")


def read_meta(path: Path) -> dict:
    return dict(line.partition("=")[::2] for line in path.read_text().splitlines()
                if "=" in line)


def rounds(tag: str) -> list[dict]:
    out = []
    for line in (OUT / tag / "trace.txt").read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1)), "d": int(m.group(2)), "acc": int(m.group(3))}
        rec.update({k: float(v) for k, v in KV_RE.findall(m.group(4))})
        out.append(rec)
    return out


def bootstrap_ci(values: list[float], stat, n: int = 4000, seed: int = 0):
    import random
    rng = random.Random(seed)
    k = len(values)
    draws = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ref", default="default")
    args = ap.parse_args()

    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(args.prefix + "-") and (p / "trace.txt").exists())
    legs = {}
    for t in tags:
        arm = t[len(args.prefix) + 1:].rpartition("-")[0]
        legs.setdefault(arm, []).append(rounds(t))

    seqs = {tuple((r["d"], r["acc"]) for r in leg) for arm in legs.values() for leg in arm}
    print(f"distinct (d, acc) round sequences over {sum(len(v) for v in legs.values())} legs: "
          f"{len(seqs)}")
    print(f"every arm replayed the identical work sequence: {len(seqs) == 1}\n")
    if len(seqs) != 1:
        print("STOP: the ladder moved real work. It is not enqueue-timing only.")
        return

    ref = args.ref
    print(f"paired per-round deltas vs `{ref}` (us/round; negative is faster)")
    print(f"{'arm':<28}{'median Δ round':>15}{'95% CI':>22}{'mean Δ round':>14}"
          f"{'median Δ vpipe':>16}{'pairs':>7}")
    results = {}
    for arm in sorted(legs):
        if arm == ref:
            continue
        dr, dv = [], []
        for la in legs[arm]:
            for lr in legs[ref]:
                dr += [a["round_us"] - b["round_us"] for a, b in zip(la, lr)]
                dv += [(a["verify_build_us"] + a["eval_wall_us"])
                       - (b["verify_build_us"] + b["eval_wall_us"]) for a, b in zip(la, lr)]
        lo, hi = bootstrap_ci(dr, st.median)
        results[arm] = {"median_round": st.median(dr), "ci": (lo, hi),
                        "mean_round": st.mean(dr), "median_vpipe": st.median(dv)}
        print(f"{arm:<28}{st.median(dr):>+15.0f}  [{lo:>+8.0f},{hi:>+8.0f}]"
              f"{st.mean(dr):>+14.0f}{st.median(dv):>+16.0f}{len(dr):>7}")

    # Within-arm null: split the reference legs into two groups with the same
    # MEAN session position and pair them. With four reference legs the split
    # is outer {first, last} against inner, which is the same position
    # structure every compared arm has. Any arm effect smaller than this null
    # is not a result.
    if len(legs[ref]) >= 2:
        n = len(legs[ref])
        ga, gb = ([legs[ref][0], legs[ref][-1]], legs[ref][1:-1]) if n >= 4 \
            else ([legs[ref][0]], [legs[ref][1]])
        null = [a["round_us"] - b["round_us"]
                for la in ga for lb in gb for a, b in zip(la, lb)]
        lo, hi = bootstrap_ci(null, st.median)
        print(f"\n{'NULL (' + ref + ' vs itself)':<28}{st.median(null):>+15.0f}"
              f"  [{lo:>+8.0f},{hi:>+8.0f}]{st.mean(null):>+14.0f}"
              f"{'':>16}{len(null):>7}")

    base = st.median([r["round_us"] for leg in legs[ref] for r in leg])
    print(f"\nreference median round = {base:.0f} us")
    for arm, r in sorted(results.items(), key=lambda kv: kv[1]["median_round"]):
        print(f"  {arm:<28} {r['median_round'] / base * 100:>+7.3f} % of round "
              f"(CI {r['ci'][0] / base * 100:+.3f} .. {r['ci'][1] / base * 100:+.3f})")


if __name__ == "__main__":
    main()
