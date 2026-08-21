#!/usr/bin/env python3
"""Host-state diagnostics for an E85 traced ABBA session.

    usage: research/e85_hoststate.py SESSION_DIR [--json OUT] [--threshold US]

`research/e85_stratified.py` showed that the twelve legs of the traced session
fall into two host states that differ by 5.4x. This script tests what causes
that split, using only data already on disk:

(a) within-leg dispersion   p25/p50/p75 of per-round `host_sum` per leg.
    A bursty external neighbour smears the distribution inside a leg. A
    process-lifetime state keeps each leg tight and separates the legs.
(b) round-index series      decile medians of `host_sum` inside each leg.
    A process-lifetime state never switches mid-flight. Contention does.
(c) leg wall-clock times    start and end of each timed segment.
    Slow legs clustered in time point at the host. Interleaved legs point at
    the process.
(d) round_us and eval_wall  per-leg medians beside `host_sum`.
    `eval_wall_us` is the GPU verify window. If it is invariant across the two
    states while `round_us` moves, the state is host-side only.

It also reports whether leg N's state predicts leg N+1's state.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
from pathlib import Path

from e85_round_pairs import parse_rounds, timed_segment

HOST_FIELDS = ["d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
               "d_chain_us", "readout_us", "commit_us", "upkeep_us"]
DEFAULT_THRESHOLD_US = 1947.0
DECILES = 10


def quantiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    cuts = statistics.quantiles(ordered, n=4, method="inclusive")
    return cuts[0], cuts[1], cuts[2]


def decile_medians(values: list[float]) -> list[float]:
    n = len(values)
    out = []
    for i in range(DECILES):
        chunk = values[n * i // DECILES: n * (i + 1) // DECILES]
        out.append(statistics.median(chunk) if chunk else float("nan"))
    return out


def load_legs(root: Path) -> list[dict]:
    with (root / "legs.tsv").open() as handle:
        meta = list(csv.DictReader(handle, delimiter="\t"))

    legs = []
    for position, row in enumerate(meta):
        leg = int(row["leg"])
        path = root / f"leg{leg:02d}-{row['arm']}" / "rounds.txt"
        rounds = timed_segment(parse_rounds(path))
        host = [sum(r[f] for f in HOST_FIELDS) for r in rounds]
        round_us = [r["round_us"] for r in rounds]
        eval_us = [r["eval_wall_us"] for r in rounds]
        end = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        span = sum(round_us) / 1e6
        legs.append({
            "leg": leg,
            "position": position,
            "arm": row["arm"],
            "rounds": len(rounds),
            "host_series": host,
            "host_p25": quantiles(host)[0],
            "host_p50": quantiles(host)[1],
            "host_p75": quantiles(host)[2],
            "host_deciles": decile_medians(host),
            "round_p50": statistics.median(round_us),
            "eval_p50": statistics.median(eval_us),
            "non_eval_p50": statistics.median(
                r["round_us"] - r["eval_wall_us"] for r in rounds),
            "mtp_s_per_tok": float(row["mtp_s_per_tok"]),
            "serial_s_per_tok": float(row["serial_s_per_tok"]),
            "local_ratio": float(row["ratio"]),
            "timed_end": end,
            "timed_start": end - dt.timedelta(seconds=span),
            "timed_span_s": span,
            "all_fields": {k: statistics.median(r[k] for r in rounds)
                           for k in rounds[0]},
        })
    return legs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--json", default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_US)
    args = ap.parse_args()

    root = Path(args.session)
    legs = load_legs(root)
    for leg in legs:
        leg["state"] = "slow" if leg["host_p50"] > args.threshold else "fast"

    slow = [l for l in legs if l["state"] == "slow"]
    fast = [l for l in legs if l["state"] == "fast"]

    # Every leg replays one identical round sequence, so a round-index profile
    # separates the deterministic schedule shape from the host state.
    n_rounds = min(l["rounds"] for l in legs)
    profile = {
        name: [statistics.median(l["host_series"][i] for l in group)
               for i in range(n_rounds)]
        for name, group in (("slow", slow), ("fast", fast), ("all", legs))
    }
    for l in legs:
        series = l["host_series"][:n_rounds]
        l["normalised"] = [h / profile["all"][i] for i, h in enumerate(series)]
        l["round_state"] = [
            1 if abs(math.log(h / profile["slow"][i]))
            < abs(math.log(h / profile["fast"][i])) else 0
            for i, h in enumerate(series)
        ]

    print("(a) within-leg dispersion of per-round host_sum, us/round\n")
    print(f"{'pos':>3s} {'leg':>3s} {'arm':<5s} {'state':<5s} {'n':>4s} "
          f"{'p25':>8s} {'p50':>8s} {'p75':>8s} {'IQR':>8s} {'IQR/p50':>8s}")
    for l in legs:
        iqr = l["host_p75"] - l["host_p25"]
        print(f"{l['position']:3d} {l['leg']:3d} {l['arm']:<5s} {l['state']:<5s} "
              f"{l['rounds']:4d} {l['host_p25']:8.0f} {l['host_p50']:8.0f} "
              f"{l['host_p75']:8.0f} {iqr:8.0f} {iqr / l['host_p50']:8.3f}")

    gap = statistics.fmean(l["host_p50"] for l in slow) \
        - statistics.fmean(l["host_p50"] for l in fast)
    worst_iqr = max(l["host_p75"] - l["host_p25"] for l in legs)
    overlap = max(l["host_p75"] for l in fast) >= min(l["host_p25"] for l in slow)
    print(f"\nbetween-state gap  {gap:8.0f} us/round")
    print(f"largest within-leg IQR {worst_iqr:8.0f} us/round "
          f"= {worst_iqr / gap:.3f} of the gap")
    print(f"any fast p75 >= any slow p25 (distribution overlap): {overlap}")

    print("\n(a2) the same dispersion after dividing out the shared "
          "round-index profile\n")
    print(f"{'pos':>3s} {'leg':>3s} {'state':<5s} {'p25':>7s} {'p50':>7s} "
          f"{'p75':>7s} {'IQR':>7s}")
    for l in legs:
        p25, p50, p75 = quantiles(l["normalised"])
        l["norm_p25"], l["norm_p50"], l["norm_p75"] = p25, p50, p75
        print(f"{l['position']:3d} {l['leg']:3d} {l['state']:<5s} "
              f"{p25:7.3f} {p50:7.3f} {p75:7.3f} {p75 - p25:7.3f}")
    worst_norm = max(l["norm_p75"] - l["norm_p25"] for l in legs)
    print(f"\nlargest normalised IQR {worst_norm:.3f}; the two state medians "
          f"differ by {statistics.fmean(l['norm_p50'] for l in slow) - statistics.fmean(l['norm_p50'] for l in fast):.3f}")

    print("\n(b) decile medians of host_sum inside each leg, us/round\n")
    print(f"{'pos':>3s} {'leg':>3s} {'state':<5s} "
          + " ".join(f"{'d' + str(i + 1):>7s}" for i in range(DECILES))
          + f" {'max/min':>8s} {'switch':>7s}")
    for l in legs:
        d = l["host_deciles"]
        crossed = sum(1 for v in d
                      if (v > args.threshold) != (l["state"] == "slow"))
        l["deciles_crossed"] = crossed
        print(f"{l['position']:3d} {l['leg']:3d} {l['state']:<5s} "
              + " ".join(f"{v:7.0f}" for v in d)
              + f" {max(d) / min(d):8.2f} {crossed:7d}")
    switched = sum(l["deciles_crossed"] for l in legs)
    print(f"\ndeciles on the other side of the {args.threshold:.0f} us "
          f"threshold, all legs: {switched}")

    print("\n(b2) per-round state, classified against the two round-index "
          "profiles\n")
    print(f"{'pos':>3s} {'leg':>3s} {'state':<5s} {'slow_rounds':>11s} "
          f"{'runs>=3':>8s} {'switch rounds':<28s}")
    total_switches = 0
    for l in legs:
        smooth = []
        for i in range(n_rounds):
            window = l["round_state"][max(0, i - 2): i + 3]
            smooth.append(1 if sum(window) * 2 > len(window) else 0)
        runs = []
        for i, value in enumerate(smooth):
            if not runs or runs[-1][0] != value:
                runs.append([value, i, i])
            else:
                runs[-1][2] = i
        kept = [r for r in runs if r[2] - r[1] + 1 >= 3]
        boundaries = [r[1] for r in kept[1:]]
        total_switches += len(boundaries)
        l["round_slow_fraction"] = statistics.fmean(l["round_state"])
        l["mid_flight_switches"] = len(boundaries)
        l["switch_rounds"] = boundaries
        print(f"{l['position']:3d} {l['leg']:3d} {l['state']:<5s} "
              f"{l['round_slow_fraction']:11.3f} {len(kept):8d} "
              + str(boundaries))
    print(f"\nlegs that switch state mid-flight: "
          f"{sum(1 for l in legs if l['mid_flight_switches'])} of {len(legs)}; "
          f"{total_switches} switches in total")

    print("\n(c) wall-clock time of each timed segment, UTC\n")
    print(f"{'pos':>3s} {'leg':>3s} {'state':<5s} {'start':>10s} {'end':>10s} "
          f"{'span_s':>7s} {'gap_s':>7s}")
    previous = None
    for l in legs:
        gap_s = (l["timed_start"] - previous).total_seconds() if previous else 0.0
        previous = l["timed_end"]
        print(f"{l['position']:3d} {l['leg']:3d} {l['state']:<5s} "
              f"{l['timed_start']:%H:%M:%S} {l['timed_end']:%H:%M:%S} "
              f"{l['timed_span_s']:7.1f} {gap_s:7.1f}")

    print("\n(d) round_us and eval_wall_us beside host_sum, per-leg medians\n")
    print(f"{'pos':>3s} {'leg':>3s} {'arm':<5s} {'state':<5s} {'host_sum':>9s} "
          f"{'round_us':>9s} {'eval_wall':>9s} {'round-eval':>10s}")
    for l in legs:
        print(f"{l['position']:3d} {l['leg']:3d} {l['arm']:<5s} {l['state']:<5s} "
              f"{l['host_p50']:9.0f} {l['round_p50']:9.0f} "
              f"{l['eval_p50']:9.0f} {l['non_eval_p50']:10.0f}")

    contrasts = {}
    for key, label in (("host_p50", "host_sum"), ("round_p50", "round_us"),
                       ("eval_p50", "eval_wall_us"),
                       ("non_eval_p50", "round_us - eval_wall_us"),
                       ("mtp_s_per_tok", "mtp_s_per_tok"),
                       ("serial_s_per_tok", "serial_s_per_tok"),
                       ("local_ratio", "local serial/mtp ratio")):
        s = statistics.fmean(l[key] for l in slow)
        f = statistics.fmean(l[key] for l in fast)
        contrasts[key] = {"slow": s, "fast": f, "delta": s - f,
                          "pct": 100.0 * (s - f) / f}
        print(f"  {label:<24s} slow {s:12.6g}  fast {f:12.6g}  "
              f"{100.0 * (s - f) / f:+7.3f} %")

    print("\n(e) every traced field by state, per-leg medians averaged\n")
    fields = sorted(set(legs[0]["all_fields"]) - {"round", "d", "acc"})
    print(f"  {'field':<18s} {'slow':>10s} {'fast':>10s} {'delta':>10s} "
          f"{'ratio':>7s}")
    field_contrast = {}
    for field in fields:
        s = statistics.fmean(l["all_fields"][field] for l in slow)
        f = statistics.fmean(l["all_fields"][field] for l in fast)
        field_contrast[field] = {"slow": s, "fast": f, "delta": s - f,
                                 "ratio": s / f if f else float("nan")}
        print(f"  {field:<18s} {s:10.0f} {f:10.0f} {s - f:10.0f} "
              f"{s / f if f else float('nan'):7.2f}")

    states = [1 if l["state"] == "slow" else 0 for l in legs]
    pairs = list(zip(states, states[1:]))
    after_slow = [b for a, b in pairs if a == 1]
    after_fast = [b for a, b in pairs if a == 0]
    mean = statistics.fmean(states)
    num = sum((a - mean) * (b - mean) for a, b in pairs)
    den = sum((s - mean) ** 2 for s in states)
    print(f"\nstate sequence (1 = slow): {states}")
    print(f"P(slow | previous slow) = {statistics.fmean(after_slow):.3f} "
          f"({sum(after_slow)}/{len(after_slow)})")
    print(f"P(slow | previous fast) = {statistics.fmean(after_fast):.3f} "
          f"({sum(after_fast)}/{len(after_fast)})")
    lag1 = num / den
    rng = random.Random(20260821)
    shuffled = list(states)
    draws = []
    for _ in range(20000):
        rng.shuffle(shuffled)
        draws.append(sum((a - mean) * (b - mean)
                         for a, b in zip(shuffled, shuffled[1:])) / den)
    p_two_sided = 2 * min(
        sum(1 for d in draws if d <= lag1), sum(1 for d in draws if d >= lag1)
    ) / len(draws)
    print(f"lag-1 autocorrelation   = {lag1:+.3f}  "
          f"(permutation mean {statistics.fmean(draws):+.3f}, "
          f"sd {statistics.stdev(draws):.3f}, two-sided p = {p_two_sided:.3f})")

    if args.json:
        payload = {
            "threshold_us": args.threshold,
            "lag1_permutation_p": p_two_sided,
            "mid_flight_switch_legs": sum(
                1 for l in legs if l["mid_flight_switches"]),
            "mid_flight_switches": total_switches,
            "largest_normalised_iqr": worst_norm,
            "between_state_gap_us": gap,
            "largest_within_leg_iqr_us": worst_iqr,
            "distribution_overlap": overlap,
            "deciles_crossing_threshold": switched,
            "contrasts": contrasts,
            "state_sequence": states,
            "p_slow_given_slow": statistics.fmean(after_slow),
            "p_slow_given_fast": statistics.fmean(after_fast),
            "lag1_autocorrelation": num / den,
            "field_contrast": field_contrast,
            "legs": [
                {k: (v.isoformat() if isinstance(v, dt.datetime) else v)
                 for k, v in l.items()
                 if k not in ("host_series", "normalised", "round_state")}
                for l in legs
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
