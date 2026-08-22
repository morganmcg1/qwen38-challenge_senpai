#!/usr/bin/env python3
"""E137 item 1: re-key the E130 rung 11 round traces by realised verify width.

    usage: python3 research/e137_width_table.py --out research/e137-artifacts/item1-width-table.json

SOURCE. The twelve traced E130 rung 11 ladder legs under `research/out/`, 924
rounds at 512 decode tokens on one Apple M4 Pro. Each round emits two lines
(`Qwen36MTPBlockSession.swift:1617-1670`, read this session):

  `mtp-trace: round=` carries the twelve named host segments and `round_us`;
  `mtp-anchor: round=` carries the absolute nanosecond anchors, including
  `t_snapshot_done`, which the named segments do not expose.

`verify_build_us` spans the recurrent snapshot, as the emit comment at
`:1652-1655` says. Joining the two lines splits it into `snapshot_us` and
`verify_graph_us`, so this table carries thirteen additive segments.

REALISED VERIFY WIDTH. `M = d + 1`: the verify call evaluates the pending
primary token plus `d` drafts (`verifyTokens` at
`Qwen36MTPBlockSession.swift:1428-1431`). The resulting histogram is
{3: 12, 4: 24, 5: 72, 6: 60, 7: 84, 8: 672}, which is the histogram in the
assignment, so `M` here means rows evaluated and matches the ranked `rows`
axis of the FINDING 167.2 boundary table.

RULE 106. Decoding is deterministic, so all twelve legs realise the SAME width
at the SAME round position. Width and position are therefore confounded by
construction and a raw per-width mean is NOT a width contrast. Three designs
are reported and they are not interchangeable:

  all        every round except position 1, which carries the cold excess;
  late       positions 50 to 63 only, where M=4,5,6,7,8 interleave, so a
             monotone within-leg drift largely cancels;
  local      each M=6 round against the nearest M=5 round(s) of the same leg,
             which is the tightest position control the fixed sequence allows.

Every interval is a cluster bootstrap over the twelve legs, because rounds
inside one leg are not independent.

RULE 111. This is a part table. It finds where the time went. It decides no
gate and it promotes nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import re
import statistics

FIELD = re.compile(r"(\w+)=(-?\d+)")
ROOT = pathlib.Path(__file__).resolve().parent.parent
LEG_GLOB = "research/out/e130-r11lad-*/trace.txt"

# The named host segments, in round order. They sum to `round_us` by
# construction, so the residual line measures only integer truncation unless a
# join is wrong.
SEGMENTS = ("d_pre_us", "d_flush_us", "d_head1_us", "d_submit1_us",
            "d_chain_us", "d_submit2_us", "snapshot_us", "verify_graph_us",
            "eval_wall_us", "readout_us", "commit_us", "upkeep_us")
REPORTED = ("draft_build_us",) + SEGMENTS + ("verify_build_us", "round_us",
                                             "host_thread_cpu_us")
LATE_WINDOW = (50, 63)
BOOTSTRAP = 10000
SEED = 20260822


def read_legs() -> list[dict]:
    """Every round of every traced leg, joined across both trace lines."""
    rounds = []
    for path in sorted(glob.glob(str(ROOT / LEG_GLOB))):
        leg = pathlib.Path(path).parent.name
        named: dict[int, dict] = {}
        anchors: dict[int, dict] = {}
        for line in open(path, errors="replace"):
            if line.startswith("mtp-trace: round="):
                f = {k: int(v) for k, v in FIELD.findall(line)}
                named[f["round"]] = f
            elif line.startswith("mtp-anchor: round="):
                f = {k: int(v) for k, v in FIELD.findall(line)}
                anchors[f["round"]] = f
        for index, row in sorted(named.items()):
            anchor = anchors.get(index)
            if anchor is None or anchor["d"] != row["d"]:
                raise SystemExit(
                    "%s round %d: the named line and the anchor line disagree"
                    % (leg, index))
            entry = {
                "leg": leg, "position": index, "m": row["d"] + 1,
                "d": row["d"], "acc": row["acc"], "pid": anchor["pid"],
                "host_thread_cpu_us": row["host_thread_cpu_ns"] / 1000.0,
            }
            for name in ("d_pre_us", "d_flush_us", "d_head1_us",
                         "d_submit1_us", "d_chain_us", "d_submit2_us",
                         "eval_wall_us", "readout_us", "commit_us",
                         "upkeep_us", "draft_build_us", "verify_build_us",
                         "round_us"):
                entry[name] = float(row[name])
            # `verify_build_us` spans the snapshot; the anchors split it.
            entry["snapshot_us"] = (
                anchor["t_snapshot_done"] - anchor["t_draft_built"]) / 1000.0
            entry["verify_graph_us"] = (
                anchor["t_verify_built"] - anchor["t_snapshot_done"]) / 1000.0
            rounds.append(entry)
    if not rounds:
        raise SystemExit("no traced rounds found under %s" % LEG_GLOB)
    return rounds


def summarise(values: list[float]) -> dict:
    n = len(values)
    ordered = sorted(values)
    return {
        "n": n,
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "sd": round(statistics.stdev(values), 1) if n > 1 else None,
        "min": round(ordered[0], 1),
        "max": round(ordered[-1], 1),
        "p25": round(ordered[max(0, int(0.25 * (n - 1)))], 1),
        "p75": round(ordered[min(n - 1, int(0.75 * (n - 1)))], 1),
    }


def cluster_bootstrap(by_leg: dict[str, list[float]], statistic,
                      seed: int = SEED) -> dict:
    """Percentile interval of `statistic` under a bootstrap over legs."""
    legs = sorted(by_leg)
    rng = random.Random(seed)
    point = statistic({leg: by_leg[leg] for leg in legs})
    if point is None:
        return {"point": None}
    draws = []
    for _ in range(BOOTSTRAP):
        pick = [legs[rng.randrange(len(legs))] for _ in legs]
        sample: dict[str, list[float]] = {}
        for index, leg in enumerate(pick):
            sample["%s#%d" % (leg, index)] = by_leg[leg]
        value = statistic(sample)
        if value is not None:
            draws.append(value)
    draws.sort()
    return {
        "point": round(point, 1),
        "ci_lo": round(draws[int(0.025 * len(draws))], 1),
        "ci_hi": round(draws[int(0.975 * len(draws)) - 1], 1),
        "bootstrap_draws": len(draws),
    }


def mean_of_all(sample: dict[str, list[float]]) -> float | None:
    pooled = [v for values in sample.values() for v in values]
    return statistics.fmean(pooled) if pooled else None


def width_table(rounds: list[dict]) -> dict:
    """Mean, median and n for every segment at every realised width."""
    table: dict[str, dict] = {}
    for m in sorted({r["m"] for r in rounds}):
        stratum = [r for r in rounds if r["m"] == m]
        cell = {
            "n": len(stratum),
            "legs": len({r["leg"] for r in stratum}),
            "positions": sorted({r["position"] for r in stratum}),
            "accepted_mean": round(
                statistics.fmean([r["acc"] for r in stratum]), 3),
            "segments": {},
        }
        for name in REPORTED:
            values = [r[name] for r in stratum]
            by_leg: dict[str, list[float]] = {}
            for r in stratum:
                by_leg.setdefault(r["leg"], []).append(r[name])
            cell["segments"][name] = summarise(values) | {
                "mean_ci": cluster_bootstrap(by_leg, mean_of_all)}
        table[str(m)] = cell
    return table


def contrast(rounds: list[dict], low: int, high: int) -> dict:
    """`mean(high) - mean(low)` for every segment, with a leg bootstrap."""
    out: dict[str, dict] = {}
    for name in REPORTED:
        by_leg: dict[str, list[tuple[float, int]]] = {}
        for r in rounds:
            if r["m"] in (low, high):
                by_leg.setdefault(r["leg"], []).append((r[name], r["m"]))

        def statistic(sample, name=name, low=low, high=high):
            lows = [v for values in sample.values() for v, m in values
                    if m == low]
            highs = [v for values in sample.values() for v, m in values
                     if m == high]
            if not lows or not highs:
                return None
            return statistics.fmean(highs) - statistics.fmean(lows)

        out[name] = cluster_bootstrap(by_leg, statistic)
        out[name]["n_low"] = sum(
            1 for r in rounds if r["m"] == low)
        out[name]["n_high"] = sum(
            1 for r in rounds if r["m"] == high)
    return out


def local_control_pairs(rounds: list[dict], low: int, high: int) -> dict:
    """Each `high` round against the nearest `low` round(s) of its own leg.

    The width sequence is fixed, so this is the tightest position control the
    data allows: the control rounds bracket the treated round in time inside
    the same leg and the same thermal state.
    """
    positions = sorted({r["position"] for r in rounds})
    low_pos = sorted({r["position"] for r in rounds if r["m"] == low})
    high_pos = sorted({r["position"] for r in rounds if r["m"] == high})
    pairing = {}
    for p in high_pos:
        before = [q for q in low_pos if q < p]
        after = [q for q in low_pos if q > p]
        controls = ([before[-1]] if before else []) + ([after[0]] if after
                                                       else [])
        if controls:
            pairing[p] = controls
    if not pairing:
        return {"error": "no %d round has a %d control" % (high, low)}

    index = {(r["leg"], r["position"]): r for r in rounds}
    legs = sorted({r["leg"] for r in rounds})
    out: dict[str, dict] = {}
    for name in REPORTED:
        by_leg: dict[str, list[float]] = {}
        for leg in legs:
            for treated, controls in pairing.items():
                t = index.get((leg, treated))
                cs = [index[(leg, c)] for c in controls if (leg, c) in index]
                if t is None or not cs:
                    continue
                by_leg.setdefault(leg, []).append(
                    t[name] - statistics.fmean([c[name] for c in cs]))
        out[name] = cluster_bootstrap(by_leg, mean_of_all)
    return {"pairing": {str(k): v for k, v in pairing.items()},
            "positions_present": positions, "difference": out}


def drift_probe(rounds: list[dict], window: tuple[int, int]) -> dict:
    """Within-leg position slope of `round_us` at the dominant width.

    M=8 carries 72.7 % of the rounds and appears throughout the window, so its
    own position trend measures the drift that a width contrast inside the
    window must survive.
    """
    lo, hi = window
    stratum = [r for r in rounds
               if r["m"] == 8 and lo <= r["position"] <= hi]
    if len(stratum) < 4:
        return {"error": "too few M=8 rounds in the window"}
    slopes = []
    for leg in sorted({r["leg"] for r in stratum}):
        xs = [float(r["position"]) for r in stratum if r["leg"] == leg]
        ys = [r["round_us"] for r in stratum if r["leg"] == leg]
        if len(set(xs)) < 2:
            continue
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den:
            slopes.append(num / den)
    if not slopes:
        return {"error": "no identified slope"}
    return {
        "window": list(window),
        "n_rounds": len(stratum),
        "legs": len(slopes),
        "slope_us_per_position_mean": round(statistics.fmean(slopes), 1),
        "slope_us_per_position_median": round(statistics.median(slopes), 1),
        "slope_sd": round(statistics.stdev(slopes), 1) if len(slopes) > 1
        else None,
    }


def residual(step: dict) -> dict:
    """Do the segments sum to the round increment?"""
    total = sum(step[name]["point"] for name in SEGMENTS
                if step[name]["point"] is not None)
    round_step = step["round_us"]["point"]
    return {
        "segment_sum_us": round(total, 1),
        "round_us_step": round_step,
        "residual_us": round(round_step - total, 1),
        "residual_pct_of_step": round(
            100.0 * (round_step - total) / round_step, 3) if round_step else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="research/e137-artifacts/item1-width-table.json")
    ap.add_argument("--low", type=int, default=5)
    ap.add_argument("--high", type=int, default=6)
    args = ap.parse_args()

    every = read_legs()
    steady = [r for r in every if r["position"] > 1]
    late = [r for r in steady
            if LATE_WINDOW[0] <= r["position"] <= LATE_WINDOW[1]]

    sequence = {}
    for r in every:
        sequence.setdefault(r["position"], set()).add(r["m"])
    if any(len(v) != 1 for v in sequence.values()):
        raise SystemExit("the legs do not share one width sequence")

    designs = {
        "all": {"rounds": steady,
                "note": "every round except position 1, the cold round"},
        "late": {"rounds": late,
                 "note": "positions %d to %d, where the widths interleave"
                         % LATE_WINDOW},
    }
    report: dict = {
        "experiment": "e137-item1-width-keyed-round-parts",
        "harness": "local",
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "note": "an offline re-key of already-paid-for E130 rung 11 traces",
        "source_legs": sorted({r["leg"] for r in every}),
        "rounds": len(every),
        "chip": "Apple M4 Pro",
        "tokens_per_leg": 512,
        "width_definition": "M = d + 1, rows the verify call evaluates",
        "width_histogram": {
            str(m): sum(1 for r in every if r["m"] == m)
            for m in sorted({r["m"] for r in every})},
        "width_sequence_identical_across_legs": True,
        "width_by_position": {str(p): sorted(v)[0]
                              for p, v in sorted(sequence.items())},
        "segments_sum_to_round_by_construction": True,
        "rule_111": "a part table never decides a gate",
        "rule_106": "width and position are confounded by a fixed decode "
                    "sequence; three designs are reported",
        "cold_round": width_table([r for r in every if r["position"] == 1]),
        "drift_probe_m8_late": drift_probe(steady, LATE_WINDOW),
        "designs": {},
    }
    for name, spec in designs.items():
        step = contrast(spec["rounds"], args.low, args.high)
        report["designs"][name] = {
            "note": spec["note"],
            "rounds": len(spec["rounds"]),
            "width_table": width_table(spec["rounds"]),
            "step_%d_to_%d" % (args.low, args.high): step,
            "residual": residual(step),
        }
    report["designs"]["local"] = {
        "note": "each M=%d round against the nearest M=%d round(s) of its "
                "own leg" % (args.high, args.low),
        "pairs": local_control_pairs(steady, args.low, args.high),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("E137 item 1  %d legs  %d rounds  widths %s"
          % (len(report["source_legs"]), report["rounds"],
             report["width_histogram"]))
    print("drift probe (M=8, positions %d-%d): %s us per position"
          % (LATE_WINDOW[0], LATE_WINDOW[1],
             report["drift_probe_m8_late"].get(
                 "slope_us_per_position_mean")))
    for design in ("all", "late"):
        block = report["designs"][design]
        step = block["step_%d_to_%d" % (args.low, args.high)]
        print()
        print("=== design %s: %s ===" % (design, block["note"]))
        print("%-20s %12s %12s %10s %10s"
              % ("segment", "M=%d mean" % args.low, "M=%d mean" % args.high,
                 "step", "95% CI"))
        for name in REPORTED:
            low = block["width_table"].get(str(args.low), {}).get(
                "segments", {}).get(name, {})
            high = block["width_table"].get(str(args.high), {}).get(
                "segments", {}).get(name, {})
            cell = step[name]
            print("%-20s %12s %12s %10s   [%s, %s]"
                  % (name, low.get("mean"), high.get("mean"),
                     cell.get("point"), cell.get("ci_lo"), cell.get("ci_hi")))
        print("residual: %s" % block["residual"])
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
