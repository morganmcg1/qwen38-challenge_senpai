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


def load_legs(prefix: str) -> dict[str, list[list[dict]]]:
    """Arm name -> list of legs, each a list of round records, chronological.

    A leg tag carries its repeat index, and the session script hands out those
    indices in run order, so tag order inside one arm is also session order.
    """
    tags = sorted(p.name for p in OUT.iterdir()
                  if p.name.startswith(prefix + "-") and (p / "trace.txt").exists())
    legs: dict[str, list[list[dict]]] = {}
    for t in tags:
        legs.setdefault(t[len(prefix) + 1:].rpartition("-")[0], []).append(rounds(t))
    return legs


def paired_summary(prefix: str, ref: str = "default") -> dict:
    """Paired per-round comparison of every arm against `ref`.

    Pairing key: the round index. Valid only because the change is bit-exact,
    so round i carries the same draft width and the same accepted count in
    every leg. `bit_exact_work` records that check and the caller must stop
    when it is false.

    Statistic: the median of the paired differences, over the full cross
    product of arm legs and reference legs. Confidence interval: the 2.5th and
    97.5th percentile of 4000 bootstrap resamples of that median. The mean is
    reported next to it because the score integrates total time; a large
    median-mean gap means host jitter, not a ladder effect.

    Null: the reference arm against itself, split into two groups with equal
    MEAN session position. Position is a real confound, so a null built from
    two reference legs at the two ends of a palindrome measures drift and
    rejects true effects.
    """
    legs = load_legs(prefix)
    seqs = {tuple((r["d"], r["acc"]) for r in leg) for arm in legs.values() for leg in arm}
    out = {
        "prefix": prefix,
        "reference_arm": ref,
        "n_legs": sum(len(v) for v in legs.values()),
        "distinct_work_sequences": len(seqs),
        "bit_exact_work": len(seqs) == 1,
        "pairing_key": "round index",
        "statistic": "median of paired per-round differences",
        "ci": "2.5/97.5 percentile of 4000 bootstrap resamples of the median",
        "arms": {},
        "null": None,
    }
    if not out["bit_exact_work"] or ref not in legs:
        return out

    base = st.median([r["round_us"] for leg in legs[ref] for r in leg])
    out["reference_median_round_us"] = base
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
        out["arms"][arm] = {
            "median_round_us": st.median(dr), "ci_lo_us": lo, "ci_hi_us": hi,
            "mean_round_us": st.mean(dr), "median_vpipe_us": st.median(dv),
            "pairs": len(dr), "pct_of_round": st.median(dr) / base * 100.0,
            "pct_ci_lo": lo / base * 100.0, "pct_ci_hi": hi / base * 100.0,
        }

    if len(legs[ref]) >= 2:
        n = len(legs[ref])
        ga, gb = ([legs[ref][0], legs[ref][-1]], legs[ref][1:-1]) if n >= 4 \
            else ([legs[ref][0]], [legs[ref][1]])
        null = [a["round_us"] - b["round_us"]
                for la in ga for lb in gb for a, b in zip(la, lb)]
        lo, hi = bootstrap_ci(null, st.median)
        out["null"] = {
            "median_round_us": st.median(null), "ci_lo_us": lo, "ci_hi_us": hi,
            "mean_round_us": st.mean(null), "pairs": len(null),
            "position_balanced": n >= 4,
            "pct_of_round": st.median(null) / base * 100.0,
        }
    return out


def render(res: dict) -> None:
    print(f"distinct (d, acc) round sequences over {res['n_legs']} legs: "
          f"{res['distinct_work_sequences']}")
    print(f"every arm replayed the identical work sequence: {res['bit_exact_work']}\n")
    if not res["bit_exact_work"]:
        print("STOP: the ladder moved real work. It is not enqueue-timing only.")
        return

    ref = res["reference_arm"]
    print(f"paired per-round deltas vs `{ref}` (us/round; negative is faster)")
    print(f"{'arm':<28}{'median Δ round':>15}{'95% CI':>22}{'mean Δ round':>14}"
          f"{'median Δ vpipe':>16}{'pairs':>7}")
    for arm, r in res["arms"].items():
        print(f"{arm:<28}{r['median_round_us']:>+15.0f}"
              f"  [{r['ci_lo_us']:>+8.0f},{r['ci_hi_us']:>+8.0f}]"
              f"{r['mean_round_us']:>+14.0f}{r['median_vpipe_us']:>+16.0f}{r['pairs']:>7}")

    if res["null"]:
        n = res["null"]
        tag = "NULL (" + ref + " vs itself)"
        print(f"\n{tag:<28}{n['median_round_us']:>+15.0f}"
              f"  [{n['ci_lo_us']:>+8.0f},{n['ci_hi_us']:>+8.0f}]"
              f"{n['mean_round_us']:>+14.0f}{'':>16}{n['pairs']:>7}"
              f"   position_balanced={n['position_balanced']}")

    print(f"\nreference median round = {res['reference_median_round_us']:.0f} us")
    for arm, r in sorted(res["arms"].items(), key=lambda kv: kv[1]["median_round_us"]):
        print(f"  {arm:<28} {r['pct_of_round']:>+7.3f} % of round "
              f"(CI {r['pct_ci_lo']:+.3f} .. {r['pct_ci_hi']:+.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ref", default="default")
    args = ap.parse_args()
    render(paired_summary(args.prefix, args.ref))


if __name__ == "__main__":
    main()
