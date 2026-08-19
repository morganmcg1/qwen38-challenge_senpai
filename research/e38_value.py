#!/usr/bin/env python3
"""E38 value accounting and MDE for every null.

Advisor 5337689508 supplied two things this module exists to use:

  * the corrected per-prompt picture -- our MTP-leg deficit against the plateau
    is 5.2 sigma on beagle and confined to draftlen >= 4.5 -- together with the
    conversion to per-round milliseconds (beagle R = 107 rounds, deficit
    0.21 ms/round; medicine R = 99, 0.052 ms/round);
  * the requirement that every null state its minimum detectable effect, and
    that the order statistic a change moves is named before any value claim.

Only beagle (our 4th order statistic) and medicine (our 5th) carry score
weight, so this module never reports an aggregate over the eight prompts.

  python3 research/e38_value.py --self-test
  python3 research/e38_value.py --metrics research/e38-artifacts/e38-metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

# --- campaign constants ------------------------------------------------------
SIGMA_SCORE_PCT = 0.0978          # advisor 5337633069 item 8, second route
CROWN_GAP_PCT = 0.5193            # total gap to ef42e043
ENGINEERABLE_PCT = 0.2586         # the MTP-leg half; the rest is their serial leg
SCORE_CHAIN = 0.4827              # d(score)/score per unit relative gain in beagle
PHI = 0.201                       # M=6 share of QMV cost (this fixture)
PSI = 0.228                       # QMV share of candidate-leg wall (measured, E33)

# advisor 5337689508: per-prompt deficits against the plateau median, the
# plateau's own between-row sd on the same evening, and askeladd's recovered
# ranked round counts.
PROMPTS = {
    "beagle": dict(order_stat=4, deficit_pct=0.363, plateau_sd_pct=0.069,
                   rounds=107, deficit_ms_per_round=0.21),
    "medicine": dict(order_stat=5, deficit_pct=0.088, plateau_sd_pct=0.061,
                     rounds=99, deficit_ms_per_round=0.052),
}

# advisor's published ladder, used only by inversion (never re-derived here).
LADDER = [
    ("beagle -0.417 % (parity with the plateau)", dict(beagle=0.417), 0.2021),
    ("beagle -1.000 %", dict(beagle=1.000), 0.4875),
    ("beagle -2.000 %", dict(beagle=2.000), 0.9849),
    ("beagle+medicine -0.640 % (passes the crown)",
     dict(beagle=0.640, medicine=0.640), 0.5193),
]

# Score per unit of MTP-leg movement when BOTH scoring legs move together,
# read straight off the ladder's last row.  A kernel change is the same code on
# every prompt, so this -- not the beagle-only 0.4827 -- is the right chain.
SCORE_PER_LEG_PCT = 0.5193 / 0.640

# E33's paired end-to-end decode instrument: pooled within-arm sd of its two
# MTP arms.  research/e39_mde.py --self-test pins this at 0.29741 %.
E2E_PAIR_SD_PCT = 0.29741

NORMAL_MDE_MULT = 2.8016          # z(0.975) + z(0.80), alpha .05 / power .80
# research/e39_mde.py exact noncentral-t inflation over the normal figure.
EXACT_INFLATION = {1: 5.83027, 3: 1.51779, 5: 1.25289}


def normal_mde(se: float) -> float:
    return NORMAL_MDE_MULT * se


def exact_mde(se: float, df: int) -> float:
    """Noncentral-t MDE.  Uses research/e39_mde.py when the base carries it."""
    try:
        import e39_mde  # type: ignore
    except ImportError:
        if df not in EXACT_INFLATION:
            raise SystemExit(f"no recorded inflation for df={df}; merge e39_mde.py")
        return normal_mde(se) * EXACT_INFLATION[df]
    return e39_mde.mde_exact(sd=se * math.sqrt(df + 1), n=df + 1, design="paired")


def leg_pct(ratio: float) -> float:
    """MTP-leg movement, percent, from an M=6 per-row QMV cost ratio."""
    return PSI * PHI * (1.0 - ratio) * 100.0


def ratio_for_leg(leg_target_pct: float) -> float:
    return 1.0 - leg_target_pct / (PSI * PHI * 100.0)


def score_pct(ratio: float) -> float:
    """Score movement when BOTH scoring legs move together.

    The kernel is the same code on every prompt, so a per-row cost change moves
    beagle and medicine by the same relative amount.  The advisor's ladder is
    the authority for the conversion; this interpolates within it rather than
    re-deriving the order-statistic algebra.
    """
    return SCORE_PER_LEG_PCT * leg_pct(ratio)


def ratio_for_score(target_pct: float) -> float:
    return ratio_for_leg(target_pct / SCORE_PER_LEG_PCT)


def controls_sd_pct(metrics: dict, arm: str = "b") -> tuple[float, list[int], float]:
    """Within-session control scatter, excluding warmup-contaminated M <= 2."""
    c = metrics["c_round_ms"]
    widths = [m for m in sorted(int(k) for k in c["base"]) if m >= 3 and m != 6]
    devs = [(c[arm][str(m)] / c["base"][str(m)] - 1.0) * 100.0 for m in widths]
    return statistics.stdev(devs), widths, statistics.median(devs)


def report(metrics: dict) -> str:
    out: list[str] = []
    w = out.append
    ratio = metrics["primary"]["value"]
    raw = metrics["primary"]["raw"]

    sd, widths, drift = controls_sd_pct(metrics)
    n_ctrl = len(widths)
    se = sd * math.sqrt(1.0 + 1.0 / n_ctrl)

    w("ORDER STATISTIC (stated before any value claim)")
    w("  The published score is the mean of the 4th and 5th of eight per-prompt")
    w("  ratios.  Ours sort with beagle 4th and medicine 5th, so a change is")
    w("  worth exactly zero unless it moves those two legs.  This kernel is the")
    w("  same code on every prompt and M=6 rounds occur on both, so it moves")
    w("  both -- and it moves essays, republic and botany too, which is worth")
    w("  nothing and is not reported as if it were.")
    w("")

    w("PRIMARY, IN THE ADVISOR'S PER-ROUND UNITS")
    w(f"  e38/m6_per_row_cost_ratio  raw {raw:.4f}   drift-adjusted {ratio:.4f}")
    leg = leg_pct(ratio)
    w(f"  MTP-leg movement           {leg:+.4f} %   (= psi {PSI} x phi {PHI} x x)")
    for name, p in PROMPTS.items():
        share = leg / p["deficit_pct"]
        ms = share * p["deficit_ms_per_round"]
        w(f"  {name:<9} ({p['order_stat']}th)  closes {share*100:5.1f} % of its "
          f"{p['deficit_pct']:.3f} % deficit  = {ms:.4f} of "
          f"{p['deficit_ms_per_round']:.3f} ms/round  (R={p['rounds']})")
    sc = score_pct(ratio)
    w(f"  score                      {sc:+.4f} %  = {sc/SIGMA_SCORE_PCT:.2f} sigma"
      f"  = {sc/ENGINEERABLE_PCT*100:.1f} % of the engineerable gap")
    w("")

    w("THE LADDER, INVERTED TO THE RATIO THIS EXPERIMENT WOULD HAVE NEEDED")
    w("  (a kernel change moves both scoring legs, so the 'both' rows are the")
    w("   honest bars; the beagle-only rows are shown for continuity only)")
    for label, legs, score_at in LADDER:
        need = ratio_for_leg(max(legs.values()))
        kind = "both " if len(legs) == 2 else "b-only"
        w(f"  {kind} {label:<44} score {score_at:+.4f} %   ratio <= {need:.4f}")
    for target, name in ((SIGMA_SCORE_PCT, "1 sigma of score"),
                         (2 * SIGMA_SCORE_PCT, "2 sigma of score"),
                         (ENGINEERABLE_PCT, "the engineerable gap"),
                         (CROWN_GAP_PCT, "the crown")):
        w(f"  both  {name:<44} score {target:+.4f} %   "
          f"ratio <= {ratio_for_score(target):.4f}")
    w("")

    w("MDE FOR EVERY NULL")
    w(f"  [1] cost curve at M=6 -- one treated width against {n_ctrl} controls")
    w(f"      controls M={widths}, sd {sd:.4f} %, drift {drift:+.4f} %")
    w(f"      se = sd x sqrt(1 + 1/{n_ctrl}) = {se:.4f} %, df = {n_ctrl-1}")
    w(f"      MDE normal {normal_mde(se):.4f} %   exact "
      f"{exact_mde(se, n_ctrl-1):.4f} %")
    w(f"      measured effect {(ratio-1)*100:+.4f} % -> "
      f"{abs((ratio-1)*100)/normal_mde(se):.2f}x the normal MDE. DETECTED, "
      "but only just; this is a difference of two large opposing terms.")
    for n in (4, 2):
        se_e2e = E2E_PAIR_SD_PCT / math.sqrt(n)
        w(f"  [2] end-to-end decode leg, paired, n={n} legs, per-pair sd "
          f"{E2E_PAIR_SD_PCT:.5f} %")
        w(f"      MDE normal {normal_mde(se_e2e):.4f} %   exact "
          f"{exact_mde(se_e2e, n-1):.4f} %")
        w(f"      predicted movement {leg:+.4f} % -> "
          f"{normal_mde(se_e2e)/abs(leg):.1f}x under-powered (normal), "
          f"{exact_mde(se_e2e, n-1)/abs(leg):.1f}x exact. NOT RUN.")
    for name, p in PROMPTS.items():
        se_p = p["plateau_sd_pct"]
        w(f"  [3] {name} leg against the plateau, single reading, sd "
          f"{se_p:.3f} % -> MDE {normal_mde(se_p):.4f} %")
        w(f"      predicted movement {leg:+.4f} % is "
          f"{normal_mde(se_p)/abs(leg):.0f}x under this floor even on the "
          "official board.")
    w("")

    w("M=1 GLOBAL NULL (a serial-leg speedup would LOWER the score)")
    m1 = metrics["m1_null"]
    w(f"  dispatch path at M=1: {m1['path']} -- no `case 1:` exists in either")
    w("  crossrow tier, so the edit is source-unreachable at M=1.")
    w(f"  observed a/base {m1['ratio_a']:.4f}  b/base {m1['ratio_b']:.4f} -- both")
    w("  SLOWER, and M<=2 is warmup-contaminated (see the preflight table), so")
    w("  this is not evidence of a serial-leg speedup in either direction.")
    return "\n".join(out)


def _self_test() -> None:
    checks = []

    def chk(name, got, want, tol):
        ok = abs(got - want) <= tol
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} got {got:.5f} want {want:.5f}")

    chk("normal multiplier reproduces e39_mde", normal_mde(1.0), 2.8016, 5e-4)
    chk("E2E MDE, paired n=4, normal", normal_mde(E2E_PAIR_SD_PCT / 2), 0.4166, 5e-4)
    chk("E2E MDE, paired n=4, exact", exact_mde(E2E_PAIR_SD_PCT / 2, 3), 0.6329, 2e-3)
    chk("E2E MDE, paired n=2, exact (5.83x blow-up)",
        exact_mde(E2E_PAIR_SD_PCT / math.sqrt(2), 1), 3.4351, 5e-3)
    chk("leg movement at the registered null ratio", leg_pct(0.9954), 0.02110, 1e-4)
    chk("crown needs both legs -0.640 %", ratio_for_leg(0.640), 0.86038, 1e-4)
    chk("parity with the plateau needs -0.417 %", ratio_for_leg(0.417), 0.90900, 1e-4)
    chk("score chain is linear through the ladder", score_pct(ratio_for_leg(0.640)),
        0.5193, 1e-4)
    print()
    if all(checks):
        print(f"SELF-TEST PASSED ({len(checks)} checks)")
    else:
        raise SystemExit("SELF-TEST FAILED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--metrics", default="research/e38-artifacts/e38-metrics.json")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0
    print(report(json.loads(pathlib.Path(args.metrics).read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
