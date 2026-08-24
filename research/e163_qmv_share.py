#!/usr/bin/env python3
"""E163: the routed-matvec share of a verify round, and what it predicts.

    usage: research/e163_qmv_share.py ARM.json [ARM.json ...]
                                      [--session research/e163-artifacts/e163_pinned_w5.json]

WHY THIS EXISTS. The advisor's published-point chain is

    published %  ~  0.5892 * X * (share of that width) * (matvec share of the
                                                          per-row term)

and the last factor has never been measured.  The campaign instrument that
would have measured it in situ, the E58/E80 dispatch census behind
`research/e116_qmv_share.py`, is void on this base: no `MLX_E58_DISPATCH_CENSUS`
and no `MLX_E80_GPU_TIME` exists in the tree or in the built worker.  So the
numerator is measured directly instead, from the routed entry point at the
scored shapes, and divided by the round cost the pinned session measured.

FRAMES, NAMED.  Never mix them.

  isolated      One routed dispatch family at a time, nothing else resident.
                What `E163QMVIsolatedTimingTests` measures.
  in_situ       The same work inside a real round, where it overlaps and
                competes.  Estimated by the campaign's 0.7858 isolated-to-in-situ
                factor and labelled a projection, not a measurement.
  round         `R` from the pinned session: milliseconds per decode round with
                the seed prefill removed, harness=local.

Nothing here is an official or ranked score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Campaign chain, E138: an isolated kernel figure converts to in situ by this
# factor. Applied only to a projection and never to a measurement.
ISOLATED_TO_IN_SITU = 0.7858
# FINDING 352: a 1 % cut in the ranked per-verified-row term is worth this much
# published score. harness=ranked. Never applied to a local ratio.
RANKED_PUBLISHED_PER_ROW_PCT = 0.5892
# F1 dilution: share of natural-schedule rounds at each verify width.
NATURAL_WIDTH_SHARE = {4: 0.051, 5: 0.064, 6: 0.064, 7: 0.038, 8: 0.769, 2: 0.013}


def per_round_us(payload: dict, width: int, statistic: str = "per_call_us_min") -> float:
    total = 0.0
    for cell in payload["cells"]:
        if cell["m"] == width:
            total += cell[statistic] * cell["calls_per_verify"]
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--session", default="research/e163-artifacts/e163_pinned_w5.json")
    ap.add_argument("--json", default="research/e163-artifacts/e163_qmv_share.json")
    args = ap.parse_args()

    payloads = {}
    for path in args.arms:
        data = json.loads(pathlib.Path(path).read_text())
        payloads[data["arm"]] = data
    widths = sorted(set(payloads[next(iter(payloads))]["widths"]))

    session = None
    session_path = ROOT / args.session
    if session_path.exists():
        session = json.loads(session_path.read_text())

    round_ms: dict[int, float] = {}
    verify_pipeline_ms: dict[int, float] = {}
    if session:
        for share in session["round_cost_shares"]:
            if share["arm_key"].startswith("shipped@"):
                width = share["verify_width"]
                round_ms[width] = share["R_ms"]
                verify_pipeline_ms[width] = share["R_ms"] * share["verify_pipeline_share_of_round"]

    out: dict[str, object] = {
        "experiment": "e163-routed-matvec-share-of-a-verify-round",
        "harness": "local",
        "official_or_ranked_score": False,
        "timing_valid_as_leg": False,
        "isolated_to_in_situ_factor": ISOLATED_TO_IN_SITU,
        "ranked_published_pct_per_1pct_row_term": RANKED_PUBLISHED_PER_ROW_PCT,
        "session_artifact": args.session if session else None,
        "widths": widths,
        "per_width": {},
    }

    print("=== E163 routed matvec cost of one verify round (harness=local) ===")
    print("isolated microbenchmark, not a leg, not a score")
    for width in widths:
        entry: dict[str, object] = {}
        for arm, payload in sorted(payloads.items()):
            entry[f"{arm}_isolated_us_per_round"] = per_round_us(payload, width)
        shipped = entry.get("shipped_isolated_us_per_round")
        print(f"\nwidth {width}")
        for arm in sorted(payloads):
            value = entry[f"{arm}_isolated_us_per_round"]
            delta = ""
            if shipped and arm != "shipped":
                entry[f"{arm}_delta_us_per_round"] = value - shipped
                entry[f"{arm}_delta_pct_of_matvec"] = (value - shipped) / shipped * 100.0
                delta = (
                    f"   delta {value - shipped:+.1f} us "
                    f"({(value - shipped) / shipped * 100.0:+.3f} % of the matvec)"
                )
            print(f"  {arm:>10} isolated routed matvec {value:9.1f} us/round{delta}")

        if width in round_ms:
            r_us = round_ms[width] * 1e3
            entry["round_us_measured"] = r_us
            entry["verify_pipeline_us_measured"] = verify_pipeline_ms[width] * 1e3
            entry["matvec_share_of_round_isolated"] = shipped / r_us
            entry["matvec_share_of_round_in_situ_projection"] = (
                shipped * ISOLATED_TO_IN_SITU / r_us
            )
            entry["matvec_share_of_verify_pipeline_isolated"] = shipped / (verify_pipeline_ms[width] * 1e3)
            print(
                f"  measured round R {round_ms[width]:.2f} ms, GPU eval "
                f"{verify_pipeline_ms[width]:.2f} ms"
            )
            print(
                f"  matvec share of the round: isolated "
                f"{entry['matvec_share_of_round_isolated']:.3f}, in-situ projection "
                f"{entry['matvec_share_of_round_in_situ_projection']:.3f}"
            )
            print(
                f"  matvec share of the GPU eval, isolated: "
                f"{entry['matvec_share_of_verify_pipeline_isolated']:.3f}"
            )
            for arm in sorted(payloads):
                if arm == "shipped":
                    continue
                delta_us = entry.get(f"{arm}_delta_us_per_round")
                if delta_us is None:
                    continue
                predicted = -delta_us / r_us * 100.0
                entry[f"{arm}_predicted_round_pct_isolated"] = predicted
                entry[f"{arm}_predicted_round_pct_in_situ"] = (
                    predicted * ISOLATED_TO_IN_SITU
                )
                share = NATURAL_WIDTH_SHARE.get(width)
                if share is not None:
                    entry[f"{arm}_predicted_natural_schedule_pct"] = predicted * share
                print(
                    f"  {arm}: predicted pinned round effect {predicted:+.3f} % "
                    f"(in-situ projection {predicted * ISOLATED_TO_IN_SITU:+.3f} %), "
                    f"natural-schedule share x{share}"
                )
        out["per_width"][str(width)] = entry

    if session:
        measured = next(
            (
                c
                for c in session["contrasts"]
                if c["contrast"] == "arm" and c["metric"] == "mtp_seconds_per_token"
            ),
            None,
        )
        if measured:
            out["measured_arm_gain_percent"] = measured["gain_percent"]
            out["measured_arm_half_range_percent"] = measured["half_range_percent"]
            print(
                f"\nmeasured pinned arm gain {measured['gain_percent']:+.4f} % "
                f"+/- {measured['half_range_percent']:.4f} on absolute candidate "
                f"mtp_seconds_per_token"
            )
    else:
        print(
            f"\nno session artifact at {args.session}; shares against a measured "
            "round are omitted",
            file=sys.stderr,
        )

    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nartifact {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
