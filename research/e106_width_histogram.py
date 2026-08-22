#!/usr/bin/env python3
"""E106 -- what verify width does the shipped policy actually run at?

    usage: research/e106_width_histogram.py TAG [TAG ...] [--json OUT]

Effect B is non-monotone in the verify row count: it peaks near 4 rows and
decays through 9. Every price for it therefore depends on which widths the
shipped draft policy really reaches. A leg run with `forced_drafts=none` lets
the shipped policy choose, and the census `round` events record the realised
`width` for every round.

`--local-iterate` runs two decode passes in two worker processes: a serial
control at depth 0, then the native-MTP pass. The `pid` and `depth` fields
separate them, so the histogram is taken over the MTP pass alone.

The census serialises every command buffer, so the round totals here are
census GPU-busy sums, not timing-leg round times. Numerator and denominator
both come from that same serialised frame, which is what makes the ratio
usable. A census leg is never a timing leg.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

from e106_phase_sweep import PRED_TO_TENSOR, SHAPE_RE, NARROW_GY, short

GDN_OUT_PROJ_PER_ROUND = 48


def legs(records):
    """Split the census round events into decode passes, keyed by pid."""
    passes = collections.defaultdict(list)
    for rec in records:
        if rec.get("event") == "round":
            passes[rec["pid"]].append(rec)
    out = {}
    for pid, rounds in passes.items():
        widths = [r["width"] for r in rounds]
        out[pid] = {
            "pid": pid,
            "rounds": len(rounds),
            "max_depth": max(r["depth"] for r in rounds),
            "mean_width": statistics.fmean(widths),
            "histogram": collections.Counter(widths),
            "accepted": sum(r["accepted"] for r in rounds),
        }
    return out


def round_gpu_us(records, pid):
    """Census GPU-busy microseconds per round, and that round's width."""
    per_round = collections.defaultdict(float)
    width_of = {}
    for rec in records:
        if rec.get("event") != "gputime" or rec.get("pid") != pid:
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        for rnd, _ordinal, _w, shape_id, gpu_ns in rec.get("trace", []):
            match = parsed[shape_id]
            if match is None or match.group("phase") == "outside":
                continue
            per_round[rnd] += gpu_ns / 1e3
    for rec in records:
        if rec.get("event") == "round" and rec.get("pid") == pid:
            width_of[rec["round"]] = rec["width"]
    by_width = collections.defaultdict(list)
    for rnd, us in per_round.items():
        if rnd in width_of:
            by_width[width_of[rnd]].append(us)
    means = {w: statistics.fmean(v) for w, v in by_width.items()}
    return means, {w: len(v) for w, v in by_width.items()}


def effect_b(records):
    """gdn.out_proj minus fa.o_proj per (phase, rows), pooled over the leg."""
    cells = collections.defaultdict(list)
    for rec in records:
        if rec.get("event") != "gputime" or not rec.get("trace"):
            continue
        parsed = [SHAPE_RE.match(s) for s in rec.get("trace_shapes", [])]
        rounds = collections.defaultdict(list)
        for rnd, ordinal, _w, shape_id, gpu_ns in rec["trace"]:
            rounds[rnd].append((ordinal, shape_id, gpu_ns))
        for rows in rounds.values():
            prev = None
            for _ordinal, shape_id, gpu_ns in sorted(rows):
                match = parsed[shape_id]
                if match is None:
                    prev = None
                    continue
                kernel = match.group("kernel")
                if (kernel.startswith("affine_qmv_fast")
                        and int(match.group("gy")) == NARROW_GY):
                    tensor = PRED_TO_TENSOR.get(prev)
                    if tensor in ("gdn.out_proj", "fa.o_proj"):
                        cells[(match.group("phase"),
                               int(match.group("gx")), tensor)].append(
                                   gpu_ns / 1e3)
                prev = short(kernel)
    out = {}
    for phase, rows in {(p, r) for p, r, _t in cells}:
        gdn = cells.get((phase, rows, "gdn.out_proj"))
        fa = cells.get((phase, rows, "fa.o_proj"))
        if not gdn or not fa or len(gdn) < 2 or len(fa) < 2:
            continue
        sem = (statistics.pstdev(gdn) ** 2 / len(gdn)
               + statistics.pstdev(fa) ** 2 / len(fa)) ** 0.5
        diff = statistics.fmean(gdn) - statistics.fmean(fa)
        out[(phase, rows)] = {
            "phase": phase, "rows": rows, "effect_b_us": diff,
            "sem_us": sem, "sigma": diff / sem if sem else float("nan"),
            "n_gdn": len(gdn), "n_fa": len(fa),
        }
    return out


def price(hist, effects, rounds_us, prefer=("target_verify", "outside")):
    """Weight the per-width prize by the realised width histogram."""
    total = sum(hist.values())
    rows, num, den = [], 0.0, 0.0
    for width, count in sorted(hist.items()):
        cell = next((effects[(p, width)] for p in prefer
                     if (p, width) in effects), None)
        round_us = rounds_us.get(width)
        share = count / total
        if cell is None or round_us is None:
            rows.append({"width": width, "count": count, "share": share,
                         "effect_b_us": None, "round_us": round_us,
                         "pct_of_round": None, "source": None})
            continue
        removable = GDN_OUT_PROJ_PER_ROUND * cell["effect_b_us"]
        rows.append({
            "width": width, "count": count, "share": share,
            "effect_b_us": cell["effect_b_us"], "sigma": cell["sigma"],
            "source": cell["phase"], "round_us": round_us,
            "removable_us": removable,
            "pct_of_round": 100.0 * removable / round_us,
        })
        num += count * removable
        den += count * round_us
    return rows, (100.0 * num / den if den else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    payload = {}

    for tag in args.tags:
        path = pathlib.Path("research/out") / tag / "census.jsonl"
        if not path.exists():
            sys.exit(f"e106_width_histogram: no census at {path}")
        records = [json.loads(line) for line in path.open() if line.strip()]

        print(f"=== {tag}")
        passes = legs(records)
        print(f"  {'pid':>7} {'rounds':>7} {'maxdepth':>9} {'accepted':>9} "
              f"{'mean width':>11}  histogram")
        for pid, leg in sorted(passes.items()):
            hist = " ".join(f"w{w}x{n}" for w, n
                            in sorted(leg["histogram"].items()))
            print(f"  {pid:7d} {leg['rounds']:7d} {leg['max_depth']:9d} "
                  f"{leg['accepted']:9d} {leg['mean_width']:11.3f}  {hist}")

        mtp = max(passes.values(), key=lambda leg: leg["max_depth"])
        if mtp["max_depth"] == 0:
            sys.exit(f"e106_width_histogram: {tag} has no MTP pass")
        print(f"\n  MTP pass = pid {mtp['pid']}, {mtp['rounds']} rounds, "
              f"mean verify width {mtp['mean_width']:.3f}")

        rounds_us, n_rounds = round_gpu_us(records, mtp["pid"])
        effects = effect_b(records)
        rows, weighted = price(mtp["histogram"], effects, rounds_us)

        serial = min(passes.values(), key=lambda leg: leg["max_depth"])
        serial_pct = float("nan")
        if serial["pid"] != mtp["pid"]:
            s_us, _ = round_gpu_us(records, serial["pid"])
            _, serial_pct = price(serial["histogram"], effects, s_us,
                                  prefer=("target_forward", "outside"))

        print(f"\n  realised width, effect B and the prize it implies")
        print(f"  {'width':>5} {'rounds':>6} {'share':>7} {'effect B':>9} "
              f"{'src':>14} {'x48 us':>9} {'round us':>10} {'% round':>8}")
        for row in rows:
            if row["effect_b_us"] is None:
                print(f"  {row['width']:5d} {row['count']:6d} "
                      f"{row['share']:7.1%} {'--':>9} {'--':>14} "
                      f"{'--':>9} {'--':>10} {'--':>8}")
                continue
            print(f"  {row['width']:5d} {row['count']:6d} {row['share']:7.1%} "
                  f"{row['effect_b_us']:+9.2f} {row['source']:>14} "
                  f"{row['removable_us']:9.1f} {row['round_us']:10.0f} "
                  f"{row['pct_of_round']:8.3f}")
        print(f"\n  histogram-weighted prize: {weighted:.3f} % of the "
              f"census MTP round")
        print(f"  same removal on the serial control: {serial_pct:.3f} % "
              f"of the census serial round")
        print(f"  implied ratio gain: {weighted - serial_pct:+.3f} % "
              f"(serial-free realisation)")

        payload[tag] = {
            "passes": {str(pid): {**leg,
                                  "histogram": dict(leg["histogram"])}
                       for pid, leg in passes.items()},
            "mtp_pid": mtp["pid"],
            "mtp_rounds": mtp["rounds"],
            "mean_verify_width": mtp["mean_width"],
            "width_histogram": dict(mtp["histogram"]),
            "rounds_sampled_per_width": n_rounds,
            "per_width": rows,
            "weighted_pct_of_round": weighted,
            "serial_pct_of_round": serial_pct,
            "ratio_gain_pct": weighted - serial_pct,
        }

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
