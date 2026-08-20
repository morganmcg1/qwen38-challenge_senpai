#!/usr/bin/env python3
"""Predict the crown dispatch table, and price the 2x2, BEFORE measuring it.

    research/e75_crown_predict.py --out research/e75-artifacts/e75-predictions.json

This file is the pre-registration for E75 rungs B, C and D. It is committed
before the crown worker is built, so every number below is a prediction and
none of it can be tuned to the answer.

What the two tables are
-----------------------
`qmv_fast_crossrow_affine4_g64_m<T, M, IPG, true>` is dispatched per verify
width M. The frozen host launches M x-groups; the group at `tid.x` claims rows
[tid.x*IPG, tid.x*IPG + IPG) and any group past the end returns without
reading weights. So the width-M round streams the whole weight tile
ceil(M / IPG) times, once per active group.

  width  ours (NA <= 6)     crown (NA <= 4)
      5  [5]   1 stream      [3,2] 2 streams
      6  [6]   1 stream      [3,3] 2 streams
      9  [5,4] 2 streams     [3,3,3] 3 streams

Every other width is byte-identical between the tables. The `ours` entries are
three Senpai campaign commits that are not in Layr-Labs upstream main:
b757237 (M=5), aa8ce50 (M=6) and 2267a84 (M=9). The crown table is therefore
not a rival optimisation; it is this campaign's own base before those three
wins. Rung B replays them on the E68 rung-1 instrument, and rungs C and D ask
whether the E68 depth price still pays once they are removed.

The cost model
--------------
S[w] is the E68 rung-1 curve: `quantized_matmul` at the scored Qwen 3.8 27B
shapes, measured in isolation by research/run-qmv-curve.sh. It is NOT a whole
round. Two layers are needed.

Layer 1, the QMV cell, which is the only thing either table can move:

    S[w]  = L + sum over active groups of c(size of group)
    c(k)  = S[k] - L                      for a width whose partition is [k]

L is the part of the isolated QMV measurement that does not scale with the
group structure. It is overdetermined: every multi-group width on the `ours`
table gives one estimate, and their spread is carried through to the
prediction band rather than hidden.

Layer 2, the round, which turns a QMV curve into decode seconds:

    decode = sum_w n(w) S[w]  +  a * rounds  +  b * rows

`a` is the fixed per-round cost and `b` the per-row cost of everything that is
not this QMV: SDPA, the recurrent state, the proposal head, sampling. Both are
solved exactly from the two measured E68 cells, so the fit has no freedom left
over. It is then checked at width 1, which appears in NEITHER arm's histogram:
S[1] + a + b must reproduce the measured serial seconds per token.

The two layers separate cleanly. `a` and `b` are table-independent, so the
predicted table effect is exactly sum_w n(w) (S_crown[w] - S_ours[w]) and does
not depend on the round fit at all. The round fit only converts that into an
absolute decode second figure.

The width histogram is treated as table-invariant. That is not an assumption
about timing; the emitted stream, the acceptance pattern and the depth walk
are all functions of tokens and the price vector, and the dispatch table
changes none of them. Rung D checks it directly by comparing the realised
histograms of the four cells.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

TABLE_H = REPO / ("Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal"
                  "/kernels/quantized.h")
RUNG1 = REPO / "research/e68-artifacts/e68-rung1.json"
E68_RUNS = REPO / ".mlxfast-private/e68-e2e/runs"
PBFIT_CROWN_OUT = REPO / "research/e75_pbfit_crown_vector.json"

# The crown table, held here as the exact upstream dispatch rather than as a
# formula, so a prediction cannot silently follow a later upstream edit.
# Layr-Labs/qwen-3.8-mtp-challenge main at bfab0de.
CROWN_DISPATCH = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}
CROWN_H_SHA256 = \
    "75d45143959eb3bd7223875da4dbe15ce5be3d1cf45871e010817b1e5249f281"

# The E68 depth-price construction, reproduced so the crown table can be
# refitted with it. See research/e68_swift_arm.py.
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
VERIFY_FORWARD_S = 0.0603
SDPA_WIDTH_WALL_DEPTH_CAP = 5
RANKED_ACCEPTANCE = {"beagle": 0.8351, "medicine": 0.8750, "local": 0.9189}

DISPATCH_RE = re.compile(
    r"qmv_fast_crossrow_affine4_g64_m<T,\s*(\d+),\s*(\d+),\s*true>")


def read_dispatch(path):
    """The (verify width -> IPG) map actually present in a kernel source."""
    text = pathlib.Path(path).read_text()
    return {int(m): int(ipg) for m, ipg in DISPATCH_RE.findall(text)}


def partition(width, ipg):
    """Group sizes the kernel runs, following its own first_m arithmetic."""
    tail = width % ipg
    if tail == 1:
        raise SystemExit(
            "e75_crown_predict: <T,%d,%d> has a one-row tail and is not "
            "instantiable" % (width, ipg))
    groups = []
    first_m = 0
    while first_m < width:
        if tail == 0 or width - first_m >= ipg:
            groups.append(ipg)
        else:
            groups.append(max(tail, 2))
        first_m += ipg
    return groups


def ladder(rung1_path):
    """Measured in-situ round latency at fixed verify width, seconds."""
    payload = json.loads(pathlib.Path(rung1_path).read_text())
    curve = payload["curve_seconds"]["shipped"]
    return {int(w): curve[w]["median_s"] for w in curve}, payload


def fit_launch_floor(s, ours):
    """Every multi-group `ours` width gives one estimate of L."""
    out = {}
    for width, groups in sorted(ours.items()):
        if len(groups) < 2 or any(g not in s for g in groups):
            continue
        out[width] = (sum(s[g] for g in groups) - s[width]) / (len(groups) - 1)
    return out


def predict_cells(s, launch_floor, ours, crown):
    """crown(w) for every width, with a band from the spread in L."""
    floors = sorted(launch_floor.values())
    out = {}
    # Widths outside [3, 9] never reach this kernel, so neither table can move
    # them. They still carry round cost, so the histogram needs them.
    for width in sorted(set(s) | set(ours) | set(crown)):
        if ours.get(width) == crown.get(width):
            out[width] = {"ours_s": s[width], "crown_s": s[width],
                          "crown_low_s": s[width], "crown_high_s": s[width],
                          "changed": False,
                          "ours_groups": ours.get(width),
                          "crown_groups": crown.get(width)}
            continue
        groups = crown[width]
        if any(g not in s for g in groups):
            continue
        values = [sum(s[g] - L for g in groups) + L for L in floors]
        out[width] = {
            "ours_s": s[width],
            "crown_s": sum(s[g] - statistics.fmean(floors)
                           for g in groups) + statistics.fmean(floors),
            "crown_low_s": min(values),
            "crown_high_s": max(values),
            "changed": True,
            "ours_groups": ours.get(width),
            "crown_groups": groups,
            "extra_weight_streams": len(groups) - len(ours[width]),
        }
    return out


def arm_histograms(runs_dir):
    """Realised verify-width histogram per depth-price arm, from E68 rung 3.

    The MTP report only. The session-wide scoop the W&B logger uses also picks
    up the serial leg's 512 zero-length rounds, which are not verify calls.
    """
    out = {}
    for leg in sorted(pathlib.Path(runs_dir).iterdir()):
        timed = leg / "reports/04-mtp-timed.json"
        meta = leg / "meta.txt"
        if not timed.exists() or not meta.exists():
            continue
        fields = dict(line.split("=", 1) for line in
                      meta.read_text().splitlines() if "=" in line)
        if fields.get("status") != "0" or fields.get("warmup_discarded") == "1":
            continue
        payload = json.loads(timed.read_text())
        lengths = payload.get("effective_draft_lengths")
        if not lengths:
            continue
        hist = collections.Counter(int(v) + 1 for v in lengths)
        arm = fields["arm"]
        record = out.setdefault(arm, {"histogram": dict(sorted(hist.items())),
                                      "legs": [], "decode_seconds": [],
                                      "serial_seconds_per_token": []})
        if record["histogram"] != dict(sorted(hist.items())):
            raise SystemExit(
                "e75_crown_predict: arm %s produced two different histograms; "
                "the schedule is not deterministic and the predictor is void"
                % arm)
        record["legs"].append(leg.name)
        record["decode_seconds"].append(payload["decode_seconds"])
        score = leg / "score.json"
        if score.exists():
            record["serial_seconds_per_token"].append(
                json.loads(score.read_text())["metrics"]
                ["serial_seconds_per_token"])
    for arm, record in out.items():
        record["rounds"] = sum(record["histogram"].values())
        record["rows"] = sum(w * n for w, n in record["histogram"].items())
        record["decode_seconds_median"] = statistics.median(
            record["decode_seconds"])
        record["decode_seconds_spread"] = (max(record["decode_seconds"])
                                           - min(record["decode_seconds"]))
        record["serial_seconds_per_token_median"] = statistics.median(
            record["serial_seconds_per_token"])
    return out


def fit_round_model(s, hists):
    """Solve the per-round and per-row cost outside this QMV, then check it.

    Two unknowns, two measured cells, so the fit is exact and has no residual
    to admire. The evidence is the out-of-sample width-1 point: no arm ever
    runs a width-1 verify, so `S[1] + a + b` against the measured serial
    seconds per token is a real test the model can fail.
    """
    def cell(arm):
        h = hists[arm]
        return (h["rounds"], h["rows"],
                sum(n * s[w] for w, n in h["histogram"].items()),
                h["decode_seconds_median"])

    (r1, w1, q1, d1), (r2, w2, q2, d2) = cell("ship"), cell("pbfit")
    det = r1 * w2 - r2 * w1
    y1, y2 = d1 - q1, d2 - q2
    a = (y1 * w2 - y2 * w1) / det
    b = (r1 * y2 - r2 * y1) / det
    measured_serial = statistics.median(
        record["serial_seconds_per_token_median"] for record in hists.values())
    return {
        "per_round_s": a,
        "per_row_s": b,
        "fitted_on": ["ship", "pbfit"],
        "width1_model_s": s[1] + a + b,
        "width1_measured_serial_seconds_per_token": measured_serial,
        "width1_error_pct": 100 * (s[1] + a + b - measured_serial)
                            / measured_serial,
    }


def round_total(histogram, cost, table):
    key = "ours_s" if table == "ours" else "crown_s"
    return sum(n * cost[w][key] for w, n in histogram.items())


def walk_depth(marginal, cap, p):
    reach, expected, cumulative, depth = 1.0, 0.0, 1.0, 0
    for d in range(min(cap, len(marginal))):
        reach *= p
        threshold = marginal[d] * (1.0 + expected) / cumulative
        if not reach > threshold:
            return depth
        expected += reach
        cumulative += marginal[d]
        depth = d + 1
    return depth


def refit_pbfit(s):
    """The E68 pbfit construction, run on an arbitrary width ladder.

    The rescale associates as `v * (total / sum)`, matching Swift's
    `makeMeasuredDepthPrice`. Rung A found that the other association,
    `v * total / sum`, differs by one ulp at three of the eight positions, so
    the vector written here is the one Swift would actually execute.
    """
    raw = [HEAD_STEP_COST_RATIO
           + (s[d + 2] - s[d + 1]) / VERIFY_FORWARD_S
           for d in range(MAX_DEPTH)]
    total = MAX_DEPTH * HEAD_STEP_COST_RATIO
    scale = total / sum(raw)
    marginal = [v * scale for v in raw]
    return {
        "raw": raw,
        "marginal": marginal,
        "marginal_total": sum(marginal),
        "scale": total / sum(raw),
        "verify_forward_s": VERIFY_FORWARD_S,
        "head_step_cost_ratio": HEAD_STEP_COST_RATIO,
        "predicted_depth": {
            name: {"p": p,
                   "default_cap": walk_depth(marginal, MAX_DEPTH, p),
                   "streak_cap": walk_depth(
                       marginal, SDPA_WIDTH_WALL_DEPTH_CAP, p)}
            for name, p in RANKED_ACCEPTANCE.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    args = ap.parse_args()

    ours_dispatch = read_dispatch(TABLE_H)
    ours = {w: partition(w, ipg) for w, ipg in sorted(ours_dispatch.items())}
    crown = {w: partition(w, ipg) for w, ipg in sorted(CROWN_DISPATCH.items())}

    s, _ = ladder(RUNG1)
    launch_floor = fit_launch_floor(s, ours)
    cost = predict_cells(s, launch_floor, ours, crown)
    hists = arm_histograms(E68_RUNS)

    round_model = fit_round_model(s, hists)
    a, b = round_model["per_round_s"], round_model["per_row_s"]

    cells = {}
    for table in ("ours", "crown"):
        for arm in sorted(hists):
            record = hists[arm]
            qmv = round_total(record["histogram"], cost, table)
            cells["%s-%s" % (table, arm)] = {
                "qmv_total_s": qmv,
                "predicted_decode_seconds":
                    qmv + a * record["rounds"] + b * record["rows"],
                "measured_decode_seconds":
                    record["decode_seconds_median"] if table == "ours"
                    else None,
            }

    factorial = {}

    if "ship" in hists and "pbfit" in hists:
        o_ship = cells["ours-ship"]["predicted_decode_seconds"]
        o_pbfit = cells["ours-pbfit"]["predicted_decode_seconds"]
        c_ship = cells["crown-ship"]["predicted_decode_seconds"]
        c_pbfit = cells["crown-pbfit"]["predicted_decode_seconds"]
        factorial = {
            "o_ship_s": o_ship, "o_pbfit_s": o_pbfit,
            "c_ship_s": c_ship, "c_pbfit_s": c_pbfit,
            "pbfit_effect_on_ours_s": o_pbfit - o_ship,
            "pbfit_effect_on_ours_pct": 100 * (o_pbfit - o_ship) / o_ship,
            "pbfit_effect_on_crown_s": c_pbfit - c_ship,
            "pbfit_effect_on_crown_pct": 100 * (c_pbfit - c_ship) / c_ship,
            "table_effect_on_ship_s": c_ship - o_ship,
            "table_effect_on_ship_pct": 100 * (c_ship - o_ship) / o_ship,
            "table_effect_on_pbfit_s": c_pbfit - o_pbfit,
            "table_effect_on_pbfit_pct": 100 * (c_pbfit - o_pbfit) / o_pbfit,
            "interaction_s": (c_pbfit - c_ship) - (o_pbfit - o_ship),
            "interaction_pp": (100 * (c_pbfit - c_ship) / c_ship
                               - 100 * (o_pbfit - o_ship) / o_ship),
        }

    crown_ladder = {w: cost[w]["crown_s"] for w in cost}
    ours_refit, crown_refit = refit_pbfit(s), refit_pbfit(crown_ladder)
    payload = {
        "harness": "local",
        "instrument": "E68 rung-1 fixed-width cell curve, unchanged",
        "ours_table_sha256_h": hashlib.sha256(
            TABLE_H.read_bytes()).hexdigest(),
        "crown_table_sha256_h": CROWN_H_SHA256,
        "ours_dispatch": ours_dispatch,
        "crown_dispatch": CROWN_DISPATCH,
        "ours_partitions": {str(w): g for w, g in ours.items()},
        "crown_partitions": {str(w): g for w, g in crown.items()},
        "measured_ladder_ours_s": s,
        "launch_floor_estimates_s": launch_floor,
        "launch_floor_mean_s": statistics.fmean(launch_floor.values()),
        "launch_floor_spread_s": (max(launch_floor.values())
                                  - min(launch_floor.values())),
        "predicted_cost_s": cost,
        "predicted_crown_step_into_5_s": (crown_ladder[5] - crown_ladder[4]),
        "predicted_crown_step_into_6_s": (crown_ladder[6] - crown_ladder[5]),
        "arm_histograms": hists,
        "round_model": round_model,
        "cells": cells,
        "factorial": factorial,
        "pbfit_ours_refit": ours_refit,
        "pbfit_crown_refit": crown_refit,
    }

    text = json.dumps(payload, indent=1, sort_keys=True, default=str)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print("e75_crown_predict: wrote %s" % out)
    else:
        print(text)

    PBFIT_CROWN_OUT.write_text(json.dumps({
        "harness": "local",
        "arm": "pbfit_crown",
        "provenance": "E75 rung C refit of the E68 pbfit construction onto the"
                      " PREDICTED crown width ladder. Not measured. Replace"
                      " from the rung B curve before shipping it anywhere.",
        "crown_ladder_predicted_s": crown_ladder,
        "raw": crown_refit["raw"],
        "marginal": crown_refit["marginal"],
        "marginal_total": crown_refit["marginal_total"],
        "holds_total": abs(crown_refit["marginal_total"]
                           - MAX_DEPTH * HEAD_STEP_COST_RATIO) < 1e-12,
        "predicted_depth": crown_refit["predicted_depth"],
        "pbfit_ours_for_comparison": {
            "marginal": ours_refit["marginal"],
            "predicted_depth": ours_refit["predicted_depth"],
        },
    }, indent=1, sort_keys=True) + "\n")
    print("e75_crown_predict: wrote %s" % PBFIT_CROWN_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
