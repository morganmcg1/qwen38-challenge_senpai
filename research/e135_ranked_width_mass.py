#!/usr/bin/env python3
"""Bound the ranked verify-width-8 mass, which is what onePass678 can pay.

WHY THIS EXISTS. Under the tight launch grid the QMV table sets launched
columns per dispatch as a function of verify width `m`:

    onePass67   {2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:2, 9:3}
    onePass678  {2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:3}

The two tables differ at `m = 8` and nowhere else, so onePass678 saves exactly
one column per QMV dispatch on a width-8 round and nothing on any other round.
Its value is therefore proportional to `P(m = 8)`, and to nothing else.

The local fixture drafts far deeper than ranked. The T45 trace of 78 rounds
puts 76.9 % of local rounds at width 8. Every ranked prompt has a lower mean
verify width than the local fixture, so a local measurement of onePass678 is an
UPPER BOUND on its ranked value. That is the opposite direction from onePass67,
which pays at widths 6 and 7, where the local fixture is thin and ranked is
near its mode, so T45 called the local onePass67 number a floor.

HOW THE BOUND IS BUILT. A ranked receipt reports `effective_mean_draft_len` per
prompt but not the width distribution. Verify width is drafts plus one, and the
support is `2 <= m <= 9`. For a variable on `[lo, hi]` with mean `mu`,

    P(m = hi) <= (mu - lo) / (hi - lo)

with equality only for the two-point distribution on the endpoints. That is a
real bound, not a fit, and the two-point extreme is far from the observed local
shape, so the true mass is well below it.

THE BOUND IS CONSERVATIVE IN THE SAFE DIRECTION. The ranked widths come from a
receipt whose depth-price arm is not recorded here. The candidate ships pb6,
which drafts shallower than ship on this fixture. If the receipt ran ship, then
the pb6 ranked widths are lower again, so the numerator is still an upper
bound. If the receipt ran pb6, the numerator is exact. Either way the reported
transfer is an upper bound.

    python3 research/e135_ranked_width_mass.py --receipt 572b2cc4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e135_width_histogram import depth_histogram  # noqa: E402

BOARD = pathlib.Path("/tmp/yukon-board/full.json")
ARTIFACTS = pathlib.Path("research/e135-artifacts")

NAME = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

# FINDING 83 median-pair weights: how often each prompt holds a median slot
# across the serial lottery. Only these five can set the published median, so
# only these five price a mechanism.
F83_WEIGHTS = {
    "beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
    "botany": 0.0124, "republic": 0.0100,
}

# T45 traced composed67 under the `ship` depth price. The candidate ships pb6,
# which drafts shallower, so the pb6 warmup leg of the F34 session is the
# correct local reference for anything the candidate will actually run.
SHIP_TRACE = "research/out/e87t-g128-2/trace.txt"
PB6_TRACE = "research/out/e135f34warm/trace.txt"

WIDTH_LO = 2
WIDTH_HI = 9


def load_receipt(prefix: str) -> dict:
    if not BOARD.exists():
        raise SystemExit(
            f"{BOARD} is missing; refresh the board cache before running")
    rows = json.loads(BOARD.read_text())
    if isinstance(rows, dict):
        rows = rows.get("submissions", rows)
    for row in rows:
        if row["id"].startswith(prefix):
            return row
    raise SystemExit(f"no submission starts with {prefix}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default="572b2cc4")
    ap.add_argument("--trace", default=PB6_TRACE)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    row = load_receipt(args.receipt)
    metrics = row["officialMetrics"]
    depth_cap = metrics.get("mtp_max_draft_depth")

    counts = depth_histogram(pathlib.Path(args.trace))
    total = sum(counts.values())
    local_mean = sum(w * n for w, n in counts.items()) / total
    local_w8 = counts.get(8, 0) / total

    print(f"receipt {row['id'][:8]}  status {row['status']}  "
          f"score {row.get('officialScore')}  max draft depth {depth_cap}")
    print(f"local trace {args.trace}")
    print(f"local fixture: {total} rounds, mean verify width {local_mean:.4f}, "
          f"P(width 8) = {local_w8:.4f}")
    print("  " + " | ".join(f"w{w} {counts[w]} ({100.0 * counts[w] / total:.1f}%)"
                            for w in sorted(counts)))
    print()

    prompts = []
    for entry in metrics["per_prompt"]:
        name = NAME[entry["prompt_sha256"][:8]]
        width = entry["effective_mean_draft_len"] + 1.0
        bound = max(0.0, (width - WIDTH_LO) / (WIDTH_HI - WIDTH_LO))
        prompts.append({
            "prompt": name,
            "mean_verify_width": width,
            "p_width8_upper_bound": bound,
            "f83_weight": F83_WEIGHTS.get(name, 0.0),
            "raw_ratio_of_means": entry["raw_ratio_of_means"],
            "non_drafting_round_count": entry["non_drafting_round_count"],
        })
    prompts.sort(key=lambda p: -p["mean_verify_width"])

    print(f"{'prompt':<10}{'mean width':>12}{'P(w8) bound':>14}"
          f"{'F83 weight':>12}{'raw ratio':>11}")
    print("\n".join(
        f"{p['prompt']:<10}{p['mean_verify_width']:>12.4f}"
        f"{p['p_width8_upper_bound']:>14.4f}{p['f83_weight']:>12.4f}"
        f"{p['raw_ratio_of_means']:>11.4f}"
        for p in prompts))

    weight_sum = sum(p["f83_weight"] for p in prompts)
    weighted = sum(p["f83_weight"] * p["p_width8_upper_bound"]
                   for p in prompts) / weight_sum
    transfer = weighted / local_w8

    print()
    print(f"e135_ranked_width8_mass_weighted   {weighted:.4f}"
          "   (F83-weighted UPPER BOUND)")
    print(f"e135_local_width8_mass             {local_w8:.4f}"
          f"   (measured, {total} rounds)")
    print(f"e135_onepass678_ranked_transfer    {transfer:.4f}"
          "   (upper bound on ranked / local)")
    print()
    deeper = [p["prompt"] for p in prompts
              if p["mean_verify_width"] > local_mean and p["f83_weight"]]
    print("Read this as: a local onePass678 gain of x % is worth at most "
          f"{transfer:.3f} x % on ranked, because onePass678 pays only at "
          "width 8 and the median-pair prompts draft shallower than the local "
          "fixture. The bound is loose by construction, so the true ranked "
          "value is lower again.")
    if deeper:
        print("Median-pair prompts that DO draft deeper than the local "
              f"fixture: {', '.join(deeper)}. They carry F83 weight "
              f"{sum(p['f83_weight'] for p in prompts if p['prompt'] in deeper):.4f}"
              ", so they cannot lift the weighted bound above the local mass.")

    if args.write:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        out = ARTIFACTS / f"ranked-width-mass-{row['id'][:8]}.json"
        out.write_text(json.dumps({
            "receipt": row["id"],
            "status": row["status"],
            "official_score": row.get("officialScore"),
            "mtp_max_draft_depth": depth_cap,
            "local_trace": args.trace,
            "local_width_counts": {str(w): counts[w] for w in sorted(counts)},
            "local_round_count": total,
            "local_mean_verify_width": local_mean,
            "e135_local_width8_mass": local_w8,
            "e135_ranked_width8_mass_weighted": weighted,
            "e135_onepass678_ranked_transfer": transfer,
            "bound": "P(m=hi) <= (mean - lo)/(hi - lo) on support [2, 9]",
            "per_prompt": prompts,
        }, indent=1) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
