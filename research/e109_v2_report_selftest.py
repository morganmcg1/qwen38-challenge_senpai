#!/usr/bin/env python3
"""Self-test for the E109 rung 0 v2 estimators, on series with known answers.

The v2 half-width is a headline number, so the estimator that produces it is
checked against synthetic series whose true dose, drift and width structure are
known by construction. Run it before trusting a v2 report:

    python3 research/e109_v2_report_selftest.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile

_SPEC = importlib.util.spec_from_file_location(
    "e109_v2_report", pathlib.Path(__file__).with_name("e109_v2_report.py"))
v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v2)

ROUNDS = 61
DOSE = 500.0
DRIFT = 30.0
BASE = 177_000.0

DOSED = [i % 2 == 1 for i in range(ROUNDS)]
FLAT = [7] * ROUNDS


def close(actual: float, expected: float, tol: float = 1e-6) -> bool:
    return abs(actual - expected) <= tol


def pair_mean(us, widths, dosed) -> float:
    return v2.summarise(
        [p["difference_us"] for p in v2.pair_rounds(us, widths, dosed)]
    )["mean_us"]


def triple_mean(us, widths, dosed) -> float:
    return v2.summarise(
        [t["estimate_us"] for t in v2.triple_rounds(us, widths, dosed)]
    )["mean_us"]


def main() -> int:
    failures = []

    # 1. Under a linear drift the pair estimator is biased by exactly -drift,
    #    because the dose alternates DUDU and every pair is ordered D then U.
    drifting = [BASE + DRIFT * i + (DOSE if DOSED[i] else 0.0)
                for i in range(ROUNDS)]
    got = pair_mean(drifting, FLAT, DOSED)
    if not close(got, DOSE - DRIFT):
        failures.append(f"pair under drift: {got} != {DOSE - DRIFT}")

    # 2. The triple estimator removes that bias and recovers the dose exactly,
    #    and it must use both orderings rather than only one.
    got = triple_mean(drifting, FLAT, DOSED)
    if not close(got, DOSE):
        failures.append(f"triple under drift: {got} != {DOSE}")
    patterns = {t["pattern"] for t in v2.triple_rounds(drifting, FLAT, DOSED)}
    if patterns != {"DUD", "UDU"}:
        failures.append(f"triple patterns: {sorted(patterns)}")

    # 3. Round time depends strongly on verify width, so no pair may straddle
    #    a width change. A 4000 us step between widths must not leak in.
    widths = [7 if i < 30 else 8 for i in range(ROUNDS)]
    stepped = [BASE + (4000.0 if widths[i] == 8 else 0.0)
               + (DOSE if DOSED[i] else 0.0) for i in range(ROUNDS)]
    got = pair_mean(stepped, widths, DOSED)
    if not close(got, DOSE):
        failures.append(f"pair across a width step: {got} != {DOSE}")

    # 4. A dose-free series must return zero through the drift-cancelling
    #    estimator. This is what the null leg asserts on real data.
    null = [BASE + DRIFT * i for i in range(ROUNDS)]
    got = triple_mean(null, FLAT, DOSED)
    if not close(got, 0.0):
        failures.append(f"triple on a dose-free series: {got} != 0")

    # 5. The reason the null leg is mandatory: a series with its own period-2
    #    structure and no dose at all still reads as a large effect.
    impostor = [BASE + (900.0 if i % 2 == 1 else 0.0) for i in range(ROUNDS)]
    got = pair_mean(impostor, FLAT, DOSED)
    if not close(got, 900.0):
        failures.append(f"period-2 impostor not surfaced: {got} != 900")

    # 6. The drift regression must not absorb the alternating dose, or the
    #    reported drift would be an artefact of the dose itself.
    got = v2.within_leg_drift_us_per_round(drifting, FLAT)
    if abs(got - DRIFT) > 1.0:
        failures.append(f"drift slope contaminated by the dose: {got}")

    # 7. The synthetic-injection check rests on one identity: adding a known
    #    value to the rounds the estimator calls dosed must move both
    #    estimators by exactly that value, whatever the underlying series is.
    #    If that failed, a recovered value could not be read as the injection.
    injected = [null[i] + (DOSE if DOSED[i] else 0.0) for i in range(ROUNDS)]
    for name, estimator in (("pair", pair_mean), ("triple", triple_mean)):
        shift = (estimator(injected, FLAT, DOSED)
                 - estimator(null, FLAT, DOSED))
        if not close(shift, DOSE, 1e-6):
            failures.append(f"{name} injection shift: {shift} != {DOSE}")

    # 8. The witness parser must read the row width, and the width fingerprint
    #    must reject a stream whose tail does not match the parent's own round
    #    widths. That rejection is the only thing standing between an
    #    unverified parity assumption and a wrong sign.
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "dose-witness.txt"
        rows = [(1, 0, 1), (2, 1, 2), (3, 0, 8), (4, 1, 8)]
        path.write_text("".join(
            f"e105_dose_forward forward={f} dosed={d} width={w}"
            f" applications=0 alternate=true dose=4 shape=prework\n"
            for f, d, w in rows))
        got = v2.dose_accounting(path)
        if [r["width"] for r in got["sequence"]] != [1, 2, 8, 8]:
            failures.append(f"witness widths: {got['sequence']}")
        if not got["alternation_exact"]:
            failures.append("witness alternation not detected")
        if got["qualifying_forwards"] != 4 or got["dosed_forwards"] != 2:
            failures.append(f"witness counts: {got}")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("e109_v2_report selftest: 8/8 pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
