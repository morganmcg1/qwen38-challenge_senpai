"""E144: what a relL2 improvement factor is worth, and what factor +0.30 pt needs.

EVERY NUMBER THIS SCRIPT PRODUCES IS INFERRED. It interpolates the only two
(reconstruction error, acceptance) points the campaign holds, both from ledger
section J:

    declared   relL2 9.18e-2 .. 9.97e-2   92.31 %   damage 0.82 pt
    qat-q4     relL2 2.89e-2 .. 3.52e-2   93.02 %   damage 0.11 pt
    master     relL2 0                    93.13 %   damage 0

Two points fit exactly one free exponent in `damage = 0.82 * factor ** -k`, so
the fit has no residual and no error bar of its own. The uncertainty reported
here comes only from the ledger quoting relL2 as a per-tensor RANGE, which makes
qat-q4's improvement factor a range too, which makes `k` a range. The linear and
square-law rows bracket that range and are shown to prove the conclusion does
not depend on the fit.

The model is used for one purpose only: to decide whether the R-B stop rule
threshold is calibrated, and whether the measured affine ceiling can reach the
+0.30 pt the assignment needs. It is not a substitute for the R-C replay.
"""

import argparse
import json

DECLARED_DAMAGE_PT = 0.82
QAT_DAMAGE_PT = 0.11
DECLARED_REL_L2 = (9.18e-2, 9.97e-2)
QAT_REL_L2 = (2.89e-2, 3.52e-2)
TARGET_RECOVERY_PT = 0.30

DECLARED_HEAD_BYTES = 427742600
# E79's measured head-bytes law: percent of published score per percent of head bytes.
SCORE_PCT_PER_BYTE_PCT = 0.0844


def exponent(factor):
    """The `k` in `damage = 0.82 * factor ** -k` that reproduces qat-q4."""
    from math import log

    return log(DECLARED_DAMAGE_PT / QAT_DAMAGE_PT) / log(factor)


def recovery(factor, k):
    return DECLARED_DAMAGE_PT * (1.0 - factor ** (-k))


def required_factor(k, target=TARGET_RECOVERY_PT):
    from math import exp, log

    return exp(-log(1.0 - target / DECLARED_DAMAGE_PT) / k)


def group_lever(path, exponents):
    """Price a trunk group-size change: byte saving against reconstruction damage.

    Ledger section J prices the g64 -> g128 trunk step from its byte saving
    alone. Group size is a rate-distortion knob, so it also moves relL2, and the
    acceptance side of that move has to be paid. This function reports both.
    """
    import e144_f219

    with open(path) as handle:
        sweep = json.load(handle)["pooled"]

    # median percent per acceptance point, from the F219 chain with the
    # scheduler held neutral. Neutral is the best case for a gain and therefore
    # the WORST case for a loss, so it understates the cost of g128.
    probe = e144_f219.run_regime(0.10, "neutral", e144_f219.BAR)
    median_pct_per_pt = probe["value"]["e144_expected_median_pct"] / 0.10

    rows = []
    for size, entry in sorted(sweep.items(), key=lambda item: int(item[0])):
        byte_pct = 100.0 * entry["trunk_metadata_bytes_delta_vs_g64"] / DECLARED_HEAD_BYTES
        byte_score_pct = -byte_pct * SCORE_PCT_PER_BYTE_PCT
        factor = entry["rel_l2_factor_vs_g64"]
        recoveries = sorted(recovery(factor, k) for k in exponents.values())
        rows.append(
            {
                "group_size": int(size),
                "rel_l2": entry["rel_l2"],
                "rel_l2_factor_vs_g64": factor,
                "trunk_metadata_bytes_delta": entry["trunk_metadata_bytes_delta_vs_g64"],
                "head_bytes_pct": byte_pct,
                "score_pct_from_bytes": byte_score_pct,
                "acceptance_pt_range": [recoveries[0], recoveries[-1]],
                "score_pct_from_acceptance": [
                    recoveries[0] * median_pct_per_pt,
                    recoveries[-1] * median_pct_per_pt,
                ],
                "net_score_pct_range": sorted(
                    [
                        byte_score_pct + recoveries[0] * median_pct_per_pt,
                        byte_score_pct + recoveries[-1] * median_pct_per_pt,
                    ]
                ),
            }
        )
    return {
        "median_pct_per_acceptance_pt": median_pct_per_pt,
        "median_pct_per_acceptance_pt_source": "e144_f219 neutral regime, anchor 1760479a",
        "score_pct_per_head_byte_pct": SCORE_PCT_PER_BYTE_PCT,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", type=float, required=True, help="pooled relL2 factor achieved")
    parser.add_argument("--ceiling", type=float, required=True, help="affine-4 g64 relL2 ceiling")
    parser.add_argument(
        "--pct-per-pt",
        default="0.85,2.03",
        help="candidate-leg percent per acceptance point; low,high. "
        "2.03 is the advisor F214 figure (+0.406 %% per 0.20 pt), 0.85 is the "
        "rival hadakang reading. Both are contested; neither is established.",
    )
    parser.add_argument("--group-lever", help="e144-group-sweep.json to price as a byte lever")
    parser.add_argument("--out", default="e144-price.json")
    arguments = parser.parse_args()

    # The ledger's relL2 ranges bracket qat-q4's improvement factor.
    factors = sorted(
        (
            DECLARED_REL_L2[0] / QAT_REL_L2[0],
            DECLARED_REL_L2[1] / QAT_REL_L2[1],
        )
    )
    exponents = {
        "linear (k=1, damage proportional to relL2)": 1.0,
        f"fitted low (qat factor {factors[1]:.4f})": exponent(factors[1]),
        f"fitted high (qat factor {factors[0]:.4f})": exponent(factors[0]),
        "square law (k=2, damage proportional to MSE)": 2.0,
    }

    low_pct, high_pct = (float(v) for v in arguments.pct_per_pt.split(","))
    report = {
        "experiment": "e144",
        "harness": "local",
        "inferred": True,
        "note": __doc__.strip().splitlines()[0],
        "qat_improvement_factor_range": factors,
        "rows": [],
    }

    for label, k in exponents.items():
        measured = recovery(arguments.measured, k)
        ceiling = recovery(arguments.ceiling, k)
        report["rows"].append(
            {
                "model": label,
                "k": k,
                "recovery_pt_at_measured": measured,
                "recovery_pt_at_affine_ceiling": ceiling,
                "factor_required_for_0.30pt": required_factor(k),
                "candidate_leg_pct_at_ceiling": [ceiling * low_pct, ceiling * high_pct],
            }
        )
        print(
            f"{label:46s} k={k:.4f}  "
            f"measured {measured:+.4f} pt  ceiling {ceiling:+.4f} pt  "
            f"need x{required_factor(k):.4f}  "
            f"ceiling worth {ceiling * low_pct:+.3f}%..{ceiling * high_pct:+.3f}%"
        )

    report["verdict"] = {
        "measured_factor": arguments.measured,
        "affine_ceiling_factor": arguments.ceiling,
        "min_required_factor": min(r["factor_required_for_0.30pt"] for r in report["rows"]),
        "max_required_factor": max(r["factor_required_for_0.30pt"] for r in report["rows"]),
        "ceiling_reaches_target": arguments.ceiling
        >= min(r["factor_required_for_0.30pt"] for r in report["rows"]),
        "recovery_pt_at_ceiling_range": [
            min(r["recovery_pt_at_affine_ceiling"] for r in report["rows"]),
            max(r["recovery_pt_at_affine_ceiling"] for r in report["rows"]),
        ],
        "recovery_pt_at_measured_range": [
            min(r["recovery_pt_at_measured"] for r in report["rows"]),
            max(r["recovery_pt_at_measured"] for r in report["rows"]),
        ],
    }

    if arguments.group_lever:
        lever = group_lever(arguments.group_lever, exponents)
        report["group_lever"] = lever
        print()
        print(
            "trunk group-size lever   %.4f %% of median per acceptance point"
            % lever["median_pct_per_acceptance_pt"]
        )
        for row in lever["rows"]:
            print(
                "  g%-4d relL2 %.6f  x%.5f  bytes %+.4f %%  score from bytes %+.4f %%  "
                "from acceptance %+.4f..%+.4f %%  net %+.4f..%+.4f %%"
                % (
                    row["group_size"],
                    row["rel_l2"],
                    row["rel_l2_factor_vs_g64"],
                    row["head_bytes_pct"],
                    row["score_pct_from_bytes"],
                    row["score_pct_from_acceptance"][0],
                    row["score_pct_from_acceptance"][1],
                    row["net_score_pct_range"][0],
                    row["net_score_pct_range"][1],
                )
            )

    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print()
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
