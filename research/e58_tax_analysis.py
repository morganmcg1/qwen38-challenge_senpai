#!/usr/bin/env python3
"""Turn the E58 counterbalanced tax session into a per-dispatch price, then into
a share of the ranked leg.

The tax adds a known number of trivial dispatches per decode round, so the slope
of matched absolute seconds per token against that number prices one dispatch.
Two taxed arms are reported separately, because they price different things:
a pipelined tax prices one more dispatch in the stream, and a serialised tax
prices encode plus submit plus wait. Their slopes bracket the marginal price.

usage:
  research/e58_tax_analysis.py [SESSION_TAG] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics

# Ranked constants supplied by the assignment. Every one of them is
# harness=ranked; every measurement in this file is harness=local.
RANKED = {
    "beagle": {"leg_ms": 6233.1, "rounds": 107, "ms_per_round": 53.33},
    "medicine": {"leg_ms": 5820.7, "rounds": 99, "ms_per_round": 53.48},
}
# RETRACTED by ledger 193(E): this is 2 sd of the SERIAL leg's jitter applied to the
# score, and the median over eight prompts does not average the candidate-leg common
# mode away. The measured single-pair ranked MDE is 2.10 %, 7.4x larger. The value
# below is kept so this module's published arithmetic stays reproducible; import
# research/ranked_noise.py for any NEW ranked pricing.
RANKED_MDE_PERCENT = 0.283
# RETRACTED by ledger 198(G), confirmed by 202. 0.0629 is ONE adjacent-leg
# same-arm spread, not a null floor. The local null is not monotone in leg
# separation and it is host- and session-specific; measured same-arm spreads run
# to 0.2835 %. Kept so this module's published arithmetic stays reproducible.
# For a NEW decision, take the largest same-arm spread inside your own session.
LOCAL_NULL_FLOOR_PERCENT = 0.0629
RANKED_DEFICIT_PERCENT = 0.5367

# From the matched 512-token census on the same base.
CANDIDATE_ROUNDS = 76
CANDIDATE_DISPATCHES_PER_ROUND = 1048.62
CANDIDATE_COMMITS_PER_ROUND = 7419 / 76
SERIAL_DISPATCHES_PER_ROUND = 1705.41
DECODE_TOKENS = 512

# Independent per-dispatch prices, each measuring a different thing.
STORM_FLOOR_NS = 940.0  # trivial dispatch, packed >=32 per buffer, own queue
STORM_BUFFER_NS = 7760.0  # marginal cost of one more command buffer
CENSUS_ENCODE_NS = 282.2  # host time inside Metal's dispatch call, real path
CENSUS_COMMIT_NS = 1568.2  # host time inside Metal's commit call, real path
E57_CONTAMINATED_NS = 22500.0  # real composed-SDPA dispatches: arithmetic too


def arm(session: str, name: str) -> dict:
    with open(f"research/out/{session}-{name}/score.json", encoding="utf-8") as f:
        return json.load(f)["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", default="e58-tax")
    parser.add_argument("--tax", type=int, default=4096)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    arms = {n: arm(args.session, n) for n in ["a1", "a2", "b1", "b2", "c1", "c2"]}
    tokens_per_round = DECODE_TOKENS / CANDIDATE_ROUNDS

    def pair(names, key):
        values = [arms[n][key] for n in names]
        return statistics.fmean(values), max(values) - min(values)

    groups = [
        ("A  tax=0", ["a1", "a2"]),
        ("B  tax pipelined (wait=0)", ["b1", "b2"]),
        ("C  tax serialised (wait=1)", ["c1", "c2"]),
    ]
    report = {
        "session": args.session,
        "harness": "local",
        "tax_per_round": args.tax,
        "candidate_tokens_per_round": round(tokens_per_round, 4),
        "arms": {},
        "slopes": {},
        "projection": {},
    }

    print("=== arm means, harness=local, ungated (MLXFAST_LOCAL_COOL_GATE=0) ===")
    print(f"candidate tokens per round = {tokens_per_round:.4f}")
    means = {}
    for label, names in groups:
        mtp, mtp_spread = pair(names, "mtp_seconds_per_token")
        ser, ser_spread = pair(names, "serial_seconds_per_token")
        means[label] = (mtp, ser)
        report["arms"][label] = {
            "runs": names,
            "mtp_seconds_per_token": mtp,
            "mtp_within_pair_spread_percent": 100.0 * mtp_spread / mtp,
            "serial_seconds_per_token": ser,
            "serial_within_pair_spread_percent": 100.0 * ser_spread / ser,
        }
        print(
            f"{label:28s} candidate={mtp * 1e3:.6f} ms/tok "
            f"(pair spread {100.0 * mtp_spread / mtp:.4f}%)   "
            f"serial={ser * 1e3:.6f} ms/tok "
            f"(pair spread {100.0 * ser_spread / ser:.4f}%)"
        )

    base_mtp, base_ser = means["A  tax=0"]
    print()
    print("=== slope: price of one taxed dispatch ===")
    for label, _ in groups[1:]:
        mtp, ser = means[label]
        d_mtp_tok = mtp - base_mtp
        d_ser_tok = ser - base_ser
        d_mtp_round = d_mtp_tok * tokens_per_round
        d_ser_round = d_ser_tok  # the serial leg emits one token per round
        cand_ns = d_mtp_round / args.tax * 1e9
        ser_ns = d_ser_round / args.tax * 1e9
        report["slopes"][label] = {
            "candidate_delta_ms_per_token": d_mtp_tok * 1e3,
            "candidate_delta_percent": 100.0 * d_mtp_tok / base_mtp,
            "candidate_delta_ms_per_round": d_mtp_round * 1e3,
            "candidate_ns_per_taxed_dispatch": cand_ns,
            "serial_delta_ms_per_token": d_ser_tok * 1e3,
            "serial_delta_percent": 100.0 * d_ser_tok / base_ser,
            "serial_ns_per_taxed_dispatch": ser_ns,
        }
        print(f"{label}")
        print(
            f"   candidate {d_mtp_tok * 1e3:+.6f} ms/tok "
            f"({100.0 * d_mtp_tok / base_mtp:+.4f}%) = "
            f"{d_mtp_round * 1e3:+.4f} ms/round -> {cand_ns:+.1f} ns/dispatch"
        )
        print(
            f"   serial    {d_ser_tok * 1e3:+.6f} ms/tok "
            f"({100.0 * d_ser_tok / base_ser:+.4f}%) = "
            f"{d_ser_round * 1e3:+.4f} ms/round -> {ser_ns:+.1f} ns/dispatch"
        )

    print()
    print("=== projection onto the ranked legs ===")
    print(
        "Every price below measures a different thing, so they are reported as a "
        "range and never averaged."
    )
    pipelined_ns = report["slopes"]["B  tax pipelined (wait=0)"][
        "candidate_ns_per_taxed_dispatch"
    ]
    methods = {
        "in_situ_pipelined_tax": pipelined_ns,
        "census_host_encode_and_submit": (
            CENSUS_ENCODE_NS
            + CENSUS_COMMIT_NS * CANDIDATE_COMMITS_PER_ROUND
            / CANDIDATE_DISPATCHES_PER_ROUND
        ),
        "storm_serialised_floor": STORM_FLOOR_NS,
        "e57_real_dispatch_regression_contaminated": E57_CONTAMINATED_NS,
    }
    for name, ns in methods.items():
        ms_per_round = CANDIDATE_DISPATCHES_PER_ROUND * ns / 1e6
        row = {"ns_per_dispatch": ns, "candidate_ms_per_round": ms_per_round}
        for leg, spec in RANKED.items():
            share = 100.0 * ms_per_round / spec["ms_per_round"]
            row[f"{leg}_percent_of_ranked_round"] = share
        report["projection"][name] = row
        shares = "  ".join(
            f"{leg}={row[f'{leg}_percent_of_ranked_round']:.3f}%" for leg in RANKED
        )
        print(f"{name:44s} {ns:9.1f} ns  {ms_per_round:7.3f} ms/round   {shares}")

    print()
    print(
        f"ranked MDE at 2 sd = {RANKED_MDE_PERCENT}% | local end-to-end null floor = "
        f"{LOCAL_NULL_FLOOR_PERCENT}% | ranked deficit to close = "
        f"{RANKED_DEFICIT_PERCENT}%"
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
