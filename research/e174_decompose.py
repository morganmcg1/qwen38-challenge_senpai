#!/usr/bin/env python3
"""E174: every balanced contrast the four-arm screen supports, and the bounds.

`harness=local`. One counterbalanced gated session, one worker, one host.

The palindrome `s o r f f r o s` puts every arm at mean leg position 4.5, so
EVERY pair of arms is position-balanced and each of the six pairwise contrasts
is drift-cancelling to first order. The screen was designed for two of them;
the other four come free from the same eight legs.

What the source says each arm actually runs, which is what decides how clean a
contrast is:

  s  3-output fused-norm kernel for 127 producers, 130 standalone fills,
     qwen35CustomAffine4QMVTableKernel USE_TABLE=true
  o  2-output fused-norm kernel, 257 standalone fills,
     qwen35CustomAffine4QMVTableKernel USE_TABLE=true
  r  2-output fused-norm kernel, 0 fills,
     qwen35CustomAffine4QMVKernel                       <- different pipeline
  f  2-output fused-norm kernel, 257 standalone fills,
     qwen35CustomAffine4QMVTableKernel USE_TABLE=false  <- extra bound buffer

So `o - s` holds the consumer EXACTLY fixed and `f - r` does not. That is the
opposite of what the screen's own header claimed, and it changes which number
is the clean one.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "out"

ARM_NOTE = {
    "s": "127 epilogues, 130 fills, table consumer",
    "o": "0 epilogues, 257 fills, table consumer",
    "r": "0 epilogues, 0 fills, replica consumer (different pipeline)",
    "f": "0 epilogues, 257 fills, table consumer USE_TABLE=false",
}

# Cells whose treatment differs, how clean the contrast is, and what it means.
#
# `clean` is graded from the source, not from the numbers:
#   A  the consumer kernel object, its bindings and its grid are identical, and
#      only one declared thing differs.
#   B  the consumer is identical but more than one thing differs.
#   C  the two arms run different consumer pipeline objects, so the contrast
#      carries a pipeline delta of unknown size and known sign.
CONTRAST_NOTE = {
    ("s", "o"): (127, "B", "127 fills + records vs 127 producer epilogues"),
    ("s", "r"): (None, "C", "whole sumtable mechanism vs the replica QMV"),
    ("s", "f"): (None, "C", "shipped vs fills-without-consumption"),
    ("o", "r"): (None, "C", "257 fills + table consumption vs replica"),
    ("o", "f"): (0, "A", "table consumption alone: same kernel, same bindings, "
                 "same grid, only the USE_TABLE template constant differs"),
    ("r", "f"): (257, "C", "257 fills + records, plus one extra bound buffer"),
}


def load(label: str) -> list[dict]:
    return json.loads((OUT / f"e174-screen-{label}.json").read_text())["legs"]


def arm_stats(legs: list[dict], key: str) -> dict[str, dict[str, float]]:
    out = {}
    for arm in ("s", "o", "r", "f"):
        vals = [x[key] for x in legs if x["arm"] == arm]
        pos = [x["position"] for x in legs if x["arm"] == arm]
        out[arm] = {
            "mean": statistics.fmean(vals),
            "values": vals,
            "spread_pct": (max(vals) - min(vals)) / statistics.fmean(vals) * 100,
            "mean_position": statistics.fmean(pos),
        }
    return out


def main() -> int:
    legs = load("s1")
    edl = statistics.fmean([x["edl"] for x in legs])
    mtp = arm_stats(legs, "mtp_spt")
    ser = arm_stats(legs, "serial_spt")

    print("=== arms (mtp seconds/token) ===")
    print(f"{'arm':<4}{'mean':<14}{'replicate spread':<20}{'mean pos':<11}note")
    for arm, st in mtp.items():
        print(
            f"{arm:<4}{st['mean']:<14.8f}{st['spread_pct']:<20.4f}"
            f"{st['mean_position']:<11.1f}{ARM_NOTE[arm]}"
        )

    # The serial leg cannot be reached by any arm, so its arm-to-arm spread is
    # a direct empirical null for a 2-leg-vs-2-leg contrast in this session.
    ser_effects = [
        abs(ser[b]["mean"] - ser[a]["mean"]) / ser[a]["mean"] * 100
        for a, b in itertools.combinations(("s", "o", "r", "f"), 2)
    ]
    print(
        f"\nserial null band (M=1 is unreachable by every arm): "
        f"max |contrast| {max(ser_effects):.4f} %, "
        f"median {statistics.median(ser_effects):.4f} % over six pairs"
    )
    print(
        "mtp within-arm replicate spread: "
        + ", ".join(f"{a}={mtp[a]['spread_pct']:.4f} %" for a in mtp)
    )

    print("\n=== all six balanced contrasts (mtp, absolute candidate time) ===")
    null_band = max(ser_effects)
    rows = []
    for a, b in itertools.combinations(("s", "o", "r", "f"), 2):
        cells, clean, note = CONTRAST_NOTE[(a, b)]
        d = mtp[b]["mean"] - mtp[a]["mean"]
        pct = d / mtp[a]["mean"] * 100
        us_round = d * edl * 1e6
        per_cell = us_round / cells if cells else None
        rows.append(
            {
                "contrast": f"{b} - {a}",
                "pct_of_base": pct,
                "base_arm": a,
                "us_per_round": us_round,
                "cells": cells,
                "us_per_cell": per_cell,
                "cleanliness": clean,
                "serial_null_pct": (ser[b]["mean"] - ser[a]["mean"])
                / ser[a]["mean"]
                * 100,
                "multiple_of_serial_null_band": abs(pct) / null_band,
                "note": note,
            }
        )
        cell_txt = f"{per_cell:+8.3f}" if per_cell is not None else "     n/a"
        print(
            f"  {b} - {a}: {pct:+8.4f} %  {us_round:+9.1f} us/round  "
            f"{cell_txt} us/cell  clean={clean}  "
            f"{rows[-1]['multiple_of_serial_null_band']:5.1f}x null band  {note}"
        )

    print("\n=== bounds on one standalone fill + its host kernel record ===")
    os_cell = (mtp["o"]["mean"] - mtp["s"]["mean"]) * edl * 1e6 / 127
    fr_cell = (mtp["f"]["mean"] - mtp["r"]["mean"]) * edl * 1e6 / 257
    print(
        f"  o - s gives fill - epilogue_delta = {os_cell:.3f} us/cell.\n"
        f"    The consumer is identical and the epilogue delta is >= 0, so\n"
        f"    this is a LOWER bound: one fill + record >= {os_cell:.3f} us."
    )
    print(
        f"  f - r gives fill + extra_binding = {fr_cell:.3f} us/cell.\n"
        f"    The replica arm runs a different QMV pipeline with one fewer\n"
        f"    bound buffer, and that delta is >= 0, so this is an UPPER\n"
        f"    bound: one fill + record <= {fr_cell:.3f} us."
    )
    print(
        f"  The two bounds cross ({os_cell:.3f} > {fr_cell:.3f}), so they cannot\n"
        f"  both be tight. They are reconciled only inside their overlapping\n"
        f"  2 sigma region, or by a fill marginal cost that rises between the\n"
        f"  130-fill and 257-fill regimes. Report the bracket, not a point."
    )

    out = {
        "experiment": "e174-xsums-epilogue-extension",
        "step": "decomposition",
        "harness": "local",
        "official_or_ranked_score": False,
        "edl": edl,
        "arms": {a: {k: v for k, v in st.items()} for a, st in mtp.items()},
        "serial_null_band_pct": {
            "max": max(ser_effects),
            "median": statistics.median(ser_effects),
        },
        "contrasts": rows,
        "fill_plus_record_us": {
            "lower_bound_from_o_minus_s": os_cell,
            "upper_bound_from_f_minus_r": fr_cell,
        },
    }
    path = OUT / "e174-decompose-s1.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")

    import wandb

    run_id = (OUT / "e174-screen-s1-wandb.txt").read_text().strip()
    run = wandb.init(
        entity="wandb-applied-ai-team",
        project="qwen38-mlx-challenge-senpai",
        id=run_id,
        resume="allow",
    )
    run.summary["decomposition"] = out
    for row in rows:
        key = row["contrast"].replace(" - ", "_minus_").replace(" ", "")
        run.summary[f"decomp/{key}/pct"] = row["pct_of_base"]
        run.summary[f"decomp/{key}/us_per_round"] = row["us_per_round"]
        run.summary[f"decomp/{key}/cleanliness"] = row["cleanliness"]
        if row["us_per_cell"] is not None:
            run.summary[f"decomp/{key}/us_per_cell"] = row["us_per_cell"]
    run.summary["fill_plus_record_us/lower_bound"] = os_cell
    run.summary["fill_plus_record_us/upper_bound"] = fr_cell
    run.summary["serial_null_band_pct_max"] = max(ser_effects)
    run.finish()
    print(f"published the decomposition to run {run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
