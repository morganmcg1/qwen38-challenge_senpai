#!/usr/bin/env python3
"""Price a board step in ranked candidate-leg terms.

Rule 63/118: a mechanism is priced from candidate MTP seconds per token, never
from a published-median ratio. The published median is printed only as context
because it also carries the serial draw, which candidate code cannot move.

    python3 research/e135_board_contrast.py 3ba6ee9d:684821ed --label crown_step

`~A:B` inverts the pair, which prices removing what `A:B` added.
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e135_tripwire_readout as tw

F83 = {"beagle": 0.4862, "medicine": 0.2508, "essays": 0.1598,
       "botany": 0.0124, "republic": 0.0100,
       "plutarch": 0.0, "drama": 0.0, "travel": 0.0}
MEDPAIR = ("beagle", "essays")


def load(path):
    if path and os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)["submissions"]
    return tw.fetch()


def provenance(row):
    metrics = row.get("officialMetrics") or {}
    return (f"    {row['id'][:8]}  {row.get('solverUsername'):>14}"
            f"  score {row.get('officialScore')}"
            f"  {row.get('promotionStatus')}"
            f"  ref {row.get('promotedSourceRef')}"
            f"  {row.get('createdAt')}")


def leaderboard(rows, reference, top):
    """Rank receipts by candidate-leg speed, which is what our code moves.

    `mtp_seconds_per_token_mean` is a candidate-only measurement on fixed
    prompts and a thermally gated runner, so it compares directly across
    receipts. The published median additionally carries the serial draw.
    """
    ref = tw.per_prompt(tw.pick(rows, reference))
    scored = []
    for row in rows:
        prompts = tw.per_prompt(row)
        pct = {n: (ref[n]["mtp_seconds_per_token_mean"]
                   / prompts[n]["mtp_seconds_per_token_mean"] - 1.0) * 100.0
               for n in tw.NAMES.values()}
        # A slow serial leg raises `raw = serial / mtp` for free. The candidate
        # cannot move it, so it is a draw, not a mechanism.
        serial = {n: (prompts[n]["serial_seconds_per_token_mean"]
                      / ref[n]["serial_seconds_per_token_mean"] - 1.0) * 100.0
                  for n in tw.NAMES.values()}
        scored.append((
            sum(pct[n] * F83[n] for n in F83) / sum(F83.values()),
            statistics.fmean(pct.values()),
            statistics.fmean(pct[n] for n in MEDPAIR),
            statistics.fmean(serial[n] for n in MEDPAIR),
            row))
    scored.sort(key=lambda t: -t[0])
    print(f"=== candidate-leg leaderboard, % faster than {reference}")
    print("    the first three columns are what candidate code moves;"
          " serial_medpair is the draw")
    print(f"    {'receipt':<9} {'F83':>8} {'mean8':>8} {'medpair':>8}"
          f" {'serialmp':>9} {'published':>10} {'pubrank':>8}"
          "  solver / promotion")
    published = sorted(rows, key=lambda r: -r["officialScore"])
    rank = {r["id"]: i + 1 for i, r in enumerate(published)}
    for i, (f83, mean8, medpair, serialmp, row) in enumerate(scored[:top]):
        print(f"{i + 1:>3} {row['id'][:8]:<9} {f83:+8.4f} {mean8:+8.4f}"
              f" {medpair:+8.4f} {serialmp:+9.4f}"
              f" {row.get('officialScore'):>10.5f} {rank[row['id']]:>8}"
              f"  {row.get('solverUsername')} / {row.get('promotionStatus')}")

    draws = [t[3] for t in scored]
    print(f"\n=== the serial draw, {len(draws)} complete receipts")
    print(f"    medpair serial vs {reference}:"
          f" mean {statistics.fmean(draws):+.4f} %"
          f"  sd {statistics.stdev(draws):.4f} pp"
          f"  range {min(draws):+.4f} .. {max(draws):+.4f}")
    inversions = sum(
        1 for i, a in enumerate(scored) for b in scored[i + 1:]
        if rank[a[4]["id"]] > rank[b[4]["id"]])
    pairs = len(scored) * (len(scored) - 1) // 2
    print(f"    candidate-leg order and published order disagree on"
          f" {inversions}/{pairs} pairs ({100.0 * inversions / pairs:.1f} %)")


def draw_trend(rows, reference, since):
    """Is the serial draw i.i.d. noise or a drift in the runner?

    The answer sets the bar. Under i.i.d. noise our next submission expects the
    pool mean, so a rival that drew high is beatable only with extra candidate
    speed. Under drift the recent level is what we will also be given.
    """
    ref = tw.per_prompt(tw.pick(rows, reference))
    points = []
    for row in rows:
        if row["createdAt"] < since:
            continue
        prompts = tw.per_prompt(row)
        points.append((row["createdAt"], statistics.fmean(
            (prompts[n]["serial_seconds_per_token_mean"]
             / ref[n]["serial_seconds_per_token_mean"] - 1.0) * 100.0
            for n in MEDPAIR)))
    points.sort()
    print(f"=== serial draw by hour since {since}, {len(points)} receipts")
    buckets = {}
    for stamp, value in points:
        buckets.setdefault(stamp[:13], []).append(value)
    for hour in sorted(buckets):
        vals = buckets[hour]
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        print(f"    {hour}Z  n {len(vals):>3}  mean {statistics.fmean(vals):+.4f} %"
              f"  sd {sd:.4f}  range {min(vals):+.4f} .. {max(vals):+.4f}")


def bar(rows, reference, target, since):
    """Uniform candidate-leg speedup we need, in expectation, to pass `target`.

    The published median is the mean of the two middle `serial / mtp` ratios.
    A uniform candidate speedup scales every ratio by the same factor, so it
    cannot move which pair is in the middle. That makes the required factor
    exact rather than an `(1 + s)(1 + c)` approximation, and it avoids assuming
    the rival's median pair is also ours: the crown is scored on
    beagle+medicine while every other top row is scored on beagle+essays.
    """
    ref = tw.per_prompt(tw.pick(rows, reference))
    pool = [tw.per_prompt(r) for r in rows if r["createdAt"] >= since]
    target_row = tw.pick(rows, target)
    goal = target_row["officialScore"]

    def median_with(serial):
        ratios = sorted(serial[n] / ref[n]["mtp_seconds_per_token_mean"]
                        for n in tw.NAMES.values())
        return (ratios[3] + ratios[4]) / 2

    draws = [median_with({n: p[n]["serial_seconds_per_token_mean"]
                          for n in tw.NAMES.values()}) for p in pool]
    mean_m, sd_m = statistics.fmean(draws), statistics.stdev(draws)
    need = goal / mean_m
    own = tw.pick(rows, reference)["officialScore"]

    tp = tw.per_prompt(target_row)
    ordered = tw.sorted_raw(tp)
    t_pair = f"{ordered[3][0]}+{ordered[4][0]}"
    t_draw = median_with({n: tp[n]["serial_seconds_per_token_mean"]
                          for n in tw.NAMES.values()})

    print(f"=== the bar to pass {target} ({goal:.6f}), from {reference}"
          f" ({own:.6f})")
    print(f"    serial draw pool since {since}: n {len(pool)}")
    print(f"    our published median if we only re-drew the serial legs:"
          f" {mean_m:.6f}  sd {sd_m:.6f}")
    print(f"    {target} is scored on {t_pair}; its own serial draw would give"
          f" us {t_draw:.6f}")
    print(f"    required uniform candidate-leg speedup"
          f" {(need - 1) * 100:+.4f} % vs {reference}")
    print(f"    with its lucky draw the same target needs only"
          f" {(goal / t_draw - 1) * 100:+.4f} %"
          f"  -> the draw alone is worth {(t_draw / mean_m - 1) * 100:+.4f} pp")
    z = (t_draw - mean_m) / sd_m
    print(f"    that draw sits at {z:+.2f} sigma, probability"
          f" ~{100 * 0.5 * math.erfc(z / math.sqrt(2)):.3f} %"
          "  so the draw is not a plan")


def headroom(rows, reference):
    """Where a candidate-leg gain still reaches the published median.

    The score reads only ranks 3 and 4 of the sorted raw vector. A gain on a
    prompt in the pair raises the median until that prompt overtakes its
    neighbour, after which the pair changes and the gain stops counting. A
    uniform gain is order preserving and never saturates; a concentrated one
    can promote its own prompt out of the pair and pull in a slower one.
    """
    prompts = tw.per_prompt(tw.pick(rows, reference))
    ordered = tw.sorted_raw(prompts)
    median = (ordered[3][1] + ordered[4][1]) / 2
    print(f"=== median-pair headroom at {reference}, median {median:.6f}")
    print(f"    {'rank':>4} {'prompt':<9} {'raw':>9} {'marginal':>9}"
          f" {'gain until promotion':>21}  note")
    for i, (name, raw) in enumerate(ordered):
        marginal = 0.5 * raw / median if i in (3, 4) else 0.0
        if i == len(ordered) - 1:
            room = float("inf")
        else:
            room = (ordered[i + 1][1] / raw - 1.0) * 100.0
        room_text = "unbounded" if room == float("inf") else f"{room:+.2f} %"
        note = ("in the pair" if i in (3, 4)
                else f"needs {(ordered[3][1] / raw - 1.0) * 100:+.2f} % to reach rank 3"
                if i < 3 else "above the pair, worth nothing")
        print(f"    {i:>4} {name:<9} {raw:>9.4f} {marginal:>9.4f}"
              f" {room_text:>21}  {note}")
    print("    marginal is published-median % per 1 % candidate gain on that"
          " prompt alone; it sums to 1.0 over the pair, which is why a uniform"
          " gain passes through exactly once")


SHAPES = {
    "uniform": {n: 1.0 for n in F83},
    "beagle_only": {"beagle": 1.0},
    "cluster_only": {n: 1.0 for n in ("essays", "republic", "medicine", "botany")},
    "medpair_only": {"beagle": 1.0, "essays": 1.0},
    "drafting_only": {n: 1.0 for n in F83 if n != "plutarch"},
    # F22 measured gain shape, normalised to beagle = 1.000.
    "f22": {"beagle": 1.000, "essays": 1.918, "drama": 2.008, "travel": 1.553,
            "botany": 0.952, "medicine": 0.783, "republic": 0.544,
            "plutarch": 0.002},
}


def shapes(rows, reference, target, since):
    """How much candidate gain each per-prompt shape needs to reach `target`.

    Solved on the exact rule, which re-sorts the raw vector and reads ranks 3
    and 4, rather than on a linearisation. A shape concentrated below the pair
    can be cheap; a shape concentrated above it can be unreachable.
    """
    ref = tw.per_prompt(tw.pick(rows, reference))
    pool = [tw.per_prompt(r) for r in rows if r["createdAt"] >= since]
    serial = {n: statistics.fmean(p[n]["serial_seconds_per_token_mean"]
                                  for p in pool) for n in tw.NAMES.values()}
    goal = tw.pick(rows, target)["officialScore"]

    # A mechanism that saves a fixed time per drafting round shows up as a
    # fraction of that prompt's round time, so its shape is 1 / seconds per
    # round. A prompt whose rounds do not draft cannot gain at all.
    round_seconds = {
        n: ref[n]["mtp_seconds_per_token_mean"]
        * (1.0 + ref[n]["effective_mean_draft_len"])
        for n in tw.NAMES.values()}
    all_shapes = dict(SHAPES)
    all_shapes["per_round"] = {
        n: 0.0 if ref[n]["non_drafting_round_count"] else
        round_seconds["beagle"] / round_seconds[n]
        for n in tw.NAMES.values()}

    def median_at(shape, k):
        ratios = sorted(
            serial[n] * (1.0 + k * shape.get(n, 0.0))
            / ref[n]["mtp_seconds_per_token_mean"] for n in tw.NAMES.values())
        return (ratios[3] + ratios[4]) / 2

    print(f"=== gain needed to reach {target} ({goal:.6f}) from {reference}")
    print(f"    baseline with a pool-average draw: {median_at({}, 0.0):.6f}")
    print(f"    {'shape':<14} {'scale':>9} {'prompts':>8}  per-prompt gain at the solution")
    for name, shape in all_shapes.items():
        lo, hi = 0.0, 1.0
        while median_at(shape, hi) < goal and hi < 64.0:
            hi *= 2.0
        if median_at(shape, hi) < goal:
            print(f"    {name:<14} {'UNREACHABLE':>9}")
            continue
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if median_at(shape, mid) < goal:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2.0
        moved = [f"{n} {k * shape[n] * 100:+.2f}"
                 for n in sorted(shape, key=lambda x: -shape[x])
                 if shape[n] > 0.01]
        print(f"    {name:<14} {k * 100:>8.4f}% {len(moved):>8}  "
              + "  ".join(moved))
    print("    scale is the candidate-leg gain applied at shape weight 1.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contrast", nargs="*", help="board pair A:B or ~A:B")
    ap.add_argument("--leaderboard", metavar="REFERENCE")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--draw-trend", metavar="REFERENCE")
    ap.add_argument("--since", default="2026-08-22T12:00:00Z")
    ap.add_argument("--bar", nargs=2, metavar=("REFERENCE", "TARGET"))
    ap.add_argument("--headroom", metavar="REFERENCE")
    ap.add_argument("--shapes", nargs=2, metavar=("REFERENCE", "TARGET"))
    ap.add_argument("--label", action="append")
    ap.add_argument("--board", default=os.environ.get(
        "YUKON_BOARD", "/tmp/yukon-board/full.json"))
    args = ap.parse_args()

    rows = [r for r in load(args.board) if tw.complete(r)]
    if args.leaderboard:
        leaderboard(rows, args.leaderboard, args.top)
        print()
    if args.draw_trend:
        draw_trend(rows, args.draw_trend, args.since)
        print()
    if args.bar:
        bar(rows, args.bar[0], args.bar[1], args.since)
        print()
    if args.headroom:
        headroom(rows, args.headroom)
        print()
    if args.shapes:
        shapes(rows, args.shapes[0], args.shapes[1], args.since)
        print()
    labels = args.label or []
    labels += [f"c{i}" for i in range(len(labels), len(args.contrast))]

    for label, spec in zip(labels, args.contrast):
        reverse = spec.startswith("~")
        left, right = spec.lstrip("~").split(":")
        a, b = tw.pick(rows, left), tw.pick(rows, right)
        print(f"=== {label}   {spec}")
        print("  provenance (A then B; the factor is A_time / B_time)")
        print(provenance(a))
        print(provenance(b))

        factors = tw.contrast_factors(rows, spec)
        pcts = {n: (f - 1.0) * 100.0 for n, f in factors.items()}
        print("  per-prompt candidate-leg gain, % faster")
        for name, pct in sorted(pcts.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<9} {pct:+8.4f}   F83 weight {F83[name]:.4f}")

        values = list(pcts.values())
        mean8 = statistics.fmean(values)
        medpair = statistics.fmean(pcts[n] for n in MEDPAIR)
        f83 = sum(pcts[n] * F83[n] for n in F83) / sum(F83.values())
        positive = sum(1 for v in values if v > 0)
        print(f"  8-prompt mean        {mean8:+8.4f} %")
        print(f"  medpair mean         {medpair:+8.4f} %  ({'+'.join(MEDPAIR)})")
        print(f"  F83-weighted mean    {f83:+8.4f} %")
        print(f"  sd across prompts    {statistics.stdev(values):8.4f} pp"
              f"   sign test {positive}/8 faster")

        pa, pb = tw.per_prompt(a), tw.per_prompt(b)
        edl_a = [pa[n]["effective_mean_draft_len"] for n in tw.NAMES.values()]
        edl_b = [pb[n]["effective_mean_draft_len"] for n in tw.NAMES.values()]
        same = all(x == y for x, y in zip(edl_a, edl_b))
        print(f"  draft-length tuple   {'IDENTICAL' if same else 'MOVED'}"
              "   (identical means the step is pure runtime, not schedule)")
        if not same:
            for name in tw.NAMES.values():
                if pa[name]["effective_mean_draft_len"] != pb[name]["effective_mean_draft_len"]:
                    print(f"      {name:<9} {pa[name]['effective_mean_draft_len']}"
                          f" -> {pb[name]['effective_mean_draft_len']}")

        sa, sb = a.get("officialScore"), b.get("officialScore")
        if sa and sb:
            step = (sb / sa - 1.0) * 100.0
            step = -step if reverse else step
            print(f"  published median     {step:+8.4f} %  CONTEXT ONLY;"
                  " it carries the serial draw")
        print()


if __name__ == "__main__":
    main()
