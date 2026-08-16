#!/usr/bin/env python3
"""fb11: label every measured verify block by weight-stream count, not width.

A verify block of draft depth d evaluates width M = d + 1 rows (one primary
plus d drafts). The scored qmv+crossrow family consumes ceil(M/4) weight
streams, so the dispatch bands are M in 1..4 (1 stream), 5..8 (2 streams) and
9 (3 streams). Marginal row cost is expected to be cheap inside a band and to
step at a band boundary."""
import argparse
import glob
import json
import math
import re
import statistics

TRACE_ROUND = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*?round_us=(\d+)"
)


def streams(width: int) -> int:
    return math.ceil(width / 4)


def collect(paths):
    by_depth: dict[int, list[float]] = {}
    per_trace: dict[str, dict[int, int]] = {}
    for path in paths:
        counts: dict[int, int] = {}
        with open(path) as fh:
            for line in fh:
                m = TRACE_ROUND.search(line)
                if not m:
                    continue
                d, us = int(m.group(2)), int(m.group(4))
                by_depth.setdefault(d, []).append(us / 1000.0)
                counts[d] = counts.get(d, 0) + 1
        if counts:
            per_trace[path] = dict(sorted(counts.items()))
    return by_depth, per_trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="research/trace-run*.log")
    ap.add_argument("--head-delta-ms", type=float, default=2.689271766519824)
    ap.add_argument("--out", default="research/fb11-stream-cost.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    by_depth, per_trace = collect(paths)

    rows = []
    for d in sorted(by_depth):
        vals = by_depth[d]
        width = d + 1
        med = statistics.median(vals)
        rows.append(
            {
                "depth": d,
                "width_rows": width,
                "weight_streams": streams(width),
                "rounds_observed": len(vals),
                "median_round_ms_local": round(med, 3),
                "mean_round_ms_local": round(statistics.fmean(vals), 3),
                "median_round_ms_head_rebased": round(med - args.head_delta_ms * d, 3),
            }
        )

    # Marginal cost of the row added when depth goes d -> d+1, attributed to the
    # width band the new row lands in.
    marginals = []
    index = {r["depth"]: r for r in rows}
    for d in sorted(index):
        nxt = index.get(d + 1)
        if not nxt:
            continue
        cur = index[d]
        dm_local = nxt["median_round_ms_local"] - cur["median_round_ms_local"]
        dm_ranked = (
            nxt["median_round_ms_head_rebased"] - cur["median_round_ms_head_rebased"]
        )
        marginals.append(
            {
                "added_row_width": nxt["width_rows"],
                "from_depth": d,
                "to_depth": d + 1,
                "crosses_stream_boundary": cur["weight_streams"]
                != nxt["weight_streams"],
                "stream_transition": f"{cur['weight_streams']}->{nxt['weight_streams']}",
                "marginal_ms_local": round(dm_local, 3),
                "marginal_ms_head_rebased": round(dm_ranked, 3),
                "rounds_backing": min(cur["rounds_observed"], nxt["rounds_observed"]),
            }
        )

    within = [m for m in marginals if not m["crosses_stream_boundary"]]
    across = [m for m in marginals if m["crosses_stream_boundary"]]
    out = {
        "traces": per_trace,
        "stream_rule": "ceil(width/4); width = depth + 1",
        "per_depth": rows,
        "marginal_rows": marginals,
        "mean_marginal_ms_within_band": (
            round(statistics.fmean([m["marginal_ms_local"] for m in within]), 3)
            if within
            else None
        ),
        "mean_marginal_ms_crossing_band": (
            round(statistics.fmean([m["marginal_ms_local"] for m in across]), 3)
            if across
            else None
        ),
        "observed_widths": sorted({r["width_rows"] for r in rows}),
        "missing_widths_1_to_9": [
            w for w in range(1, 10) if w not in {r["width_rows"] for r in rows}
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(
        f"{'depth':>5} {'width':>5} {'streams':>7} {'rounds':>6} "
        f"{'med_ms':>8} {'med_ranked':>10}"
    )
    for r in rows:
        print(
            f"{r['depth']:>5} {r['width_rows']:>5} {r['weight_streams']:>7} "
            f"{r['rounds_observed']:>6} {r['median_round_ms_local']:>8.2f} "
            f"{r['median_round_ms_head_rebased']:>10.2f}"
        )
    print()
    for m in marginals:
        flag = "BOUNDARY" if m["crosses_stream_boundary"] else "within  "
        print(
            f"  +row width {m['added_row_width']}  {flag} {m['stream_transition']}  "
            f"local {m['marginal_ms_local']:>7.2f} ms   "
            f"ranked {m['marginal_ms_head_rebased']:>7.2f} ms"
        )
    print()
    print("within-band mean marginal ms:", out["mean_marginal_ms_within_band"])
    print("boundary  mean marginal ms:", out["mean_marginal_ms_crossing_band"])
    print("missing widths:", out["missing_widths_1_to_9"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
