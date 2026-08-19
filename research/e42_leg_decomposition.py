#!/usr/bin/env python3
"""Decompose both legs into QMV and non-QMV using only measured shares.

psi(MTP) comes from the p2 arms and psi(serial) from the m1 arm, so both terms
are causal measurements rather than curve predictions. The interesting output is
not either share on its own but how differently the two components scale with
row count: QMV streams one weight matrix for a whole group of rows, so it grows
sublinearly in M, while the per-row work does not.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "research/e42-artifacts/analysis.json"


def main() -> int:
    payload = json.loads(ANALYSIS.read_text())
    by_arm = {r["arm"]: r for r in payload["results"]}
    m1 = by_arm["m1L1"]
    base = payload["arms"]["base"]

    psi_mtp = payload["slope_psi"]["p2"]["psi_from_slope"]
    psi_ser = m1["psi_eff_serial"]
    mtp_s, mtp_r = base["mtp_decode_seconds"], base["round_count"]
    ser_s, ser_r = base["serial_decode_seconds"], m1["serial_rounds"]

    ser_round = 1000.0 * ser_s / ser_r
    mtp_round = 1000.0 * mtp_s / mtp_r
    ser_q, mtp_q = psi_ser * ser_round, psi_mtp * mtp_round
    ser_n, mtp_n = ser_round - ser_q, mtp_round - mtp_q

    out = {
        "psi_serial_measured_by_m1": psi_ser,
        "psi_mtp_measured_by_p2_slope": psi_mtp,
        "serial": {
            "rounds": ser_r,
            "ms_per_round": ser_round,
            "qmv_ms_per_round": ser_q,
            "non_qmv_ms_per_round": ser_n,
            "non_qmv_share": ser_n / ser_round,
        },
        "mtp": {
            "rounds": mtp_r,
            "mean_width": sum(base["widths"]) / len(base["widths"]),
            "ms_per_round": mtp_round,
            "qmv_ms_per_round": mtp_q,
            "non_qmv_ms_per_round": mtp_n,
            "non_qmv_share": mtp_n / mtp_round,
            "non_qmv_seconds_of_leg": (mtp_n / 1000.0) * mtp_r,
        },
        "growth_serial_to_mtp": {
            "qmv": mtp_q / ser_q,
            "non_qmv": mtp_n / ser_n,
            "non_qmv_over_qmv": (mtp_n / ser_n) / (mtp_q / ser_q),
        },
        # The p2 ladder intercept is the SAME number as mtp.non_qmv_ms_per_round
        # by construction, since psi_from_slope is q_mean/t0. Recorded so nobody
        # mistakes the identity for a cross-check.
        "mtp_non_qmv_ms_per_round_from_p2_intercept": payload["slope_psi"]["p2"][
            "non_qmv_ms_per_round"
        ],
        "mtp_non_qmv_intercept_is_identical_by_construction": True,
        # This one IS independent: the isolated --shapes-only curve at width 1
        # never sees the leg, and the m1 injection never sees the curve.
        "serial_qmv_ms_per_round_from_curve": 1000.0
        * m1["q_serial_predicted_seconds"]
        / ser_r,
    }
    out["serial_qmv_curve_vs_injection_pct"] = 100.0 * (
        out["serial_qmv_ms_per_round_from_curve"] / ser_q - 1.0
    )
    print(json.dumps(out, indent=2))
    dest = ROOT / "research/e42-artifacts/leg-decomposition.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
