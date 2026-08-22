#!/usr/bin/env python3
"""E128-F4 3a - publish the per-prompt ranked width histograms.

harness=ranked. Zero GPU.

Thorfinn needs the realised per-round width distribution for each ranked
prompt so he can weight a per-width dispatch change by how often each width
actually occurs. This script writes that distribution as a standalone,
self-describing artifact instead of leaving it buried inside the section 5
curve fit.

How each histogram is built, and what it is not
-----------------------------------------------

The ranked receipts publish only `effective_mean_draft_len` per prompt. They
do not publish the distribution. So the distribution here is CONSTRUCTED, not
measured on the ranked host, in three steps:

1. Take the realised per-round depth histogram of the local E124/E122 prose
   fixtures that were written to imitate that ranked prompt. This is a real
   measurement, on our hardware, at the current base.
2. Pin plutarch's `non_drafting_round_count = 449` of 487 rounds at width
   M = 1 first, because that count IS published and is exact.
3. Tilt the remaining shape onto the ranked prompt's published mean with a
   maximum-entropy exponential tilt. Among all distributions on the same
   support with the required mean, the tilt is the one that adds the least
   information to the measured shape.

Two consequences a consumer must respect:

- The MEAN of every histogram is exact, because it is pinned to the published
  `effective_mean_draft_len + 1`. Any quantity that depends only on the mean
  is as good as the receipt.
- The SHAPE is inferred from a local proxy fixture. Any quantity that depends
  on the spread, on the tails, or on the mass at one specific width carries
  the proxy's error. Weighting a per-width kernel change is exactly such a
  quantity, so treat the per-width mass as an estimate with a stated
  sensitivity, not as a measurement.

The sensitivity is reported here as the spread across the four pre-registered
R vectors and across the alternative fixture assignment for the two prompts
that have two proxy fixtures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e128_ourcurve import (
    F83_WEIGHT,
    MAX_ROWS,
    PROMPT_FIXTURES,
    build_points,
    fixture_histograms,
    load_receipt,
    prompt_width_histogram,
    r_scenarios,
)

ROWS = np.arange(0, MAX_ROWS, dtype=float) + 1.0


def summarise(probs: np.ndarray) -> dict:
    mean = float((probs * ROWS).sum())
    var = float((probs * (ROWS - mean) ** 2).sum())
    cdf = np.cumsum(probs)
    return {
        "mean_M": mean,
        "sd_M": float(np.sqrt(max(var, 0.0))),
        "p_M_ge_6": float(probs[ROWS >= 6].sum()),
        "p_M_ge_7": float(probs[ROWS >= 7].sum()),
        "p_M_eq_1": float(probs[0]),
        "median_M": float(ROWS[int(np.searchsorted(cdf, 0.5))]),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path,
                        default=Path("/tmp/yukon-board/full.json"))
    parser.add_argument("--identity", type=Path,
                        default=here / "e128-artifacts/rung0-identity.json")
    parser.add_argument("--shipped", type=Path,
                        default=here / "e128-artifacts/rung1-shipped.json")
    parser.add_argument("--receipt", default="d3c491b5")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    hists = fixture_histograms(args.shipped)
    scenarios = r_scenarios(args.identity)
    receipt = load_receipt(args.board, args.receipt)
    points = {p["prompt"]: p
              for p in build_points(receipt, scenarios["assumed"], hists)}

    print("harness=ranked  E128-F4 3a - ranked per-prompt width histograms")
    print("receipt %s  score %.8f  R vector assumed  M = draft depth + 1\n"
          % (receipt["id"][:8], receipt["score"]))

    header = "%-10s %7s %8s " % ("prompt", "F83 wt", "mean M")
    header += " ".join("%7s" % ("M=%d" % m) for m in range(1, MAX_ROWS + 1))
    print(header)

    out = {}
    weighted = np.zeros(MAX_ROWS)
    for prompt in sorted(points, key=lambda p: -F83_WEIGHT[p]):
        point = points[prompt]
        probs = np.array(point["hist"]["probs"])
        weighted += F83_WEIGHT[prompt] * probs
        print("%-10s %7.4f %8.3f " % (prompt, F83_WEIGHT[prompt],
                                      point["mbar"])
              + " ".join("%7.4f" % v for v in probs))
        out[prompt] = {
            "f83_weight": F83_WEIGHT[prompt],
            "published_effective_mean_draft_len": point["mbar"] - 1.0,
            "pinned_mean_M": point["mbar"],
            "ranked_rounds_R_assumed": point["R"],
            "proxy_fixtures": PROMPT_FIXTURES[prompt],
            "non_drafting_share_pinned_at_M1":
                point["hist"]["non_drafting_share"],
            "support_M": [int(m) for m in ROWS],
            "probs": [float(v) for v in probs],
            **summarise(probs),
        }

    total = weighted.sum()
    weighted = weighted / total
    print("%-10s %7.4f %8.3f " % ("F83-WTD", total, (weighted * ROWS).sum())
          + " ".join("%7.4f" % v for v in weighted))

    sens = {}
    for scenario in ("predicted", "band_lo", "band_hi"):
        alt = {p["prompt"]: np.array(p["hist"]["probs"])
               for p in build_points(receipt, scenarios[scenario], hists)}
        sens[scenario] = {
            prompt: float(np.abs(alt[prompt]
                                 - np.array(out[prompt]["probs"])).sum() / 2.0)
            for prompt in out}
    print("\ntotal-variation distance from the assumed-R histogram:")
    print("%-10s " % "prompt"
          + " ".join("%10s" % s for s in sens))
    for prompt in sorted(out, key=lambda p: -F83_WEIGHT[p]):
        print("%-10s " % prompt
              + " ".join("%10.4f" % sens[s][prompt] for s in sens))

    single = {}
    for prompt, fixtures in PROMPT_FIXTURES.items():
        if len(fixtures) < 2:
            continue
        for fixture in fixtures:
            # `prompt_width_histogram` pools every fixture the prompt names,
            # so pointing all of them at one counter isolates that fixture.
            solo = {name: hists[fixture] for name in fixtures}
            probs = prompt_width_histogram(
                prompt, points[prompt]["mbar"], 0,
                points[prompt]["R"], solo)["probs"]
            single.setdefault(prompt, {})[fixture] = [float(v) for v in probs]
    print("\nsingle-proxy-fixture spread, total-variation from the pooled "
          "histogram:")
    for prompt, byfix in single.items():
        base = np.array(out[prompt]["probs"])
        for fixture, probs in byfix.items():
            print("%-10s %-20s %8.4f"
                  % (prompt, fixture,
                     float(np.abs(np.array(probs) - base).sum() / 2.0)))

    payload = {
        "harness": "ranked",
        "source": "e128_width_histograms.py",
        "receipt": {"prefix": args.receipt, "id": receipt["id"],
                    "score": receipt["score"]},
        "r_vector": "assumed",
        "definition_of_M": "draft depth + 1, the target verify row count",
        "construction": [
            "measured local proxy-fixture depth histogram at the current base",
            "plutarch's published 449 non-drafting rounds pinned at M=1",
            "maximum-entropy exponential tilt onto the published mean",
        ],
        "mean_is_exact": True,
        "shape_is_inferred_from_local_proxy_fixtures": True,
        "prompts": out,
        "f83_weighted": {
            "weight_total": float(total),
            "probs": [float(v) for v in weighted],
            **summarise(weighted),
        },
        "sensitivity_total_variation_by_r_vector": sens,
        "sensitivity_single_proxy_fixture": single,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
