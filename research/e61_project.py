#!/usr/bin/env python3
"""Project the measured M=6 cell delta onto the whole decode leg.

Rung 1 measured what the single-stream M=6 cell costs. The promotion rule is
written on whole-leg candidate MTP seconds per token, so the cell delta has to
be carried through two factors:

  f6       the share of local QMV verify time that width 6 accounts for,
           recomputed here from THIS session's measured `shipped` per-width
           costs rather than carried over from E54; and
  psi_mtp  E55's preregistered transfer constant from QMV time to MTP-leg time,
           0.693391, with E55's measured realisation factor 0.946 applied
           afterwards.

The ranked projection is computed from `research/e53-width-mixture.json`, which
does carry a per-width mixture under prompts[p].fits[variant].mixture. Those
keys are already verify widths M, not draft lengths: research/e53_width_
mixture.py stores widths[depth + 1] at :182. Only `mean_draft_len` (:192) is a
draft count, so the ranked mean verify width is 1 + mean_draft_len. The ranked
f6 combines that round mixture with THIS session's measured per-width cell cost,
so it is "E53 ranked width mixture x E61 per-width cell cost measured on this
host". The assignment's 30.9-34.7 % hand band is kept only as a comparison.

  python3 research/e61_project.py --out research/e61-artifacts/e61-projection.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Deterministic across all six E55 legs at 512 tokens (research/e61-prereg.md).
LOCAL_ROUNDS = {2: 1, 4: 5, 5: 5, 6: 23, 7: 4, 8: 6, 9: 34}
# E55 preregistered transfer constant and its measured realisation factor.
PSI_MTP = 0.693391
REALISATION = 0.946
# Assignment: M=6 share of ranked QMV time, kept only for comparison.
RANKED_F6_BAND = (0.309, 0.347)
# E53's base fit; the other five variants are reported as a sensitivity range.
E53_HEADLINE_VARIANT = "A_flat"
# research/e61-prereg.md.
LOCAL_NULL_FLOOR_PCT = 0.0629
PROMOTE_PCT = -0.30
REPORT_ONLY_PCT = 0.10
PREREG_F6 = 0.2671
PREREG_CELL_DELTA_PCT = -9.95
PREREG_WHOLE_LEG_PCT = -1.84


def time_shares(rounds: dict[int, float], cost: dict[str, dict]) -> dict[int, float]:
    """Share of QMV verify time per width, given a round mixture and cell costs."""
    weighted = {m: n * cost[str(m)]["weighted_seconds_per_verify"]
                for m, n in rounds.items() if str(m) in cost}
    total = sum(weighted.values())
    return {m: w / total for m, w in weighted.items()}


def ranked_mixtures(e53: dict) -> dict[str, dict[int, float]]:
    """Prompt-weighted ranked round mixture per E53 fit variant.

    The mixture keys are verify widths already, so they are used unshifted.
    """
    weights = e53["weights"]
    out = {}
    for variant in e53["variants"]:
        blended: dict[int, float] = {}
        for prompt, w in weights.items():
            mixture = e53["prompts"][prompt]["fits"][variant]["mixture"]
            for width, p in mixture.items():
                blended[int(width)] = blended.get(int(width), 0.0) + w * p
        total = sum(blended.values())
        out[variant] = {m: p / total for m, p in blended.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth", default="research/e61-artifacts/e61-bandwidth.json")
    ap.add_argument("--e53", default="research/e53-width-mixture.json")
    ap.add_argument("--out", default="research/e61-artifacts/e61-projection.json")
    args = ap.parse_args()

    bw = json.loads((REPO / args.bandwidth).read_text())
    shipped = bw["per_arm_per_width"]["shipped"]
    cell = bw["cell_deltas"]["t6"]["cells"]

    # Time-weighted share of local QMV verify time per width, on the shipped
    # table, using this session's own per-width costs.
    shares = time_shares(LOCAL_ROUNDS, shipped)
    f6 = shares[6]

    m6_delta_pct = cell["6"]["seconds_delta_pct"]

    def carry(f: float) -> dict:
        q = f * m6_delta_pct
        return {
            "f6": f,
            "qmv_delta_pct": q,
            "leg_delta_pct": PSI_MTP * q,
            "leg_delta_realised_pct": PSI_MTP * q * REALISATION,
        }

    qmv_delta_pct = f6 * m6_delta_pct
    leg_delta_pct = PSI_MTP * qmv_delta_pct
    leg_delta_realised_pct = leg_delta_pct * REALISATION

    e53 = json.loads((REPO / args.e53).read_text())
    ranked_rounds = ranked_mixtures(e53)
    ranked_e53 = {}
    for variant, mixture in ranked_rounds.items():
        s = time_shares(mixture, shipped)
        ranked_e53[variant] = {
            "round_mixture": mixture,
            "time_shares": s,
            "mean_verify_width_rounds": sum(m * p for m, p in mixture.items()),
            **carry(s[6]),
        }
    headline = ranked_e53[E53_HEADLINE_VARIANT]
    f6_values = [v["f6"] for v in ranked_e53.values()]

    # The published score is a median over eight hidden prompts. E53's two
    # weighted prompts both draft deeply; its held-out `plutarch` record barely
    # drafts at all, so it bounds what this change buys on a hard prompt.
    plutarch_mixture = {int(m): p for m, p in e53["plutarch"]["mixture"].items()}
    plutarch = {
        "published_mean_verify_width": 1.0 + e53["plutarch"]["published_mean_draft_len"],
        "time_shares": time_shares(plutarch_mixture, shipped),
        **carry(time_shares(plutarch_mixture, shipped)[6]),
    }

    ranked_band = {name: carry(f)
                   for name, f in (("low", RANKED_F6_BAND[0]),
                                   ("high", RANKED_F6_BAND[1]))}

    verdict = ("promote" if leg_delta_realised_pct <= PROMOTE_PCT
               else "report_only" if leg_delta_realised_pct < REPORT_ONLY_PCT
               else "stop")

    out = {
        "inputs": {
            "local_rounds": LOCAL_ROUNDS,
            "psi_mtp": PSI_MTP,
            "realisation_factor": REALISATION,
            "measured_m6_cell_delta_pct": m6_delta_pct,
            "shipped_seconds_per_verify": {
                m: shipped[str(m)]["weighted_seconds_per_verify"] for m in LOCAL_ROUNDS
                if str(m) in shipped},
        },
        "local_time_weighted_shares": shares,
        "local": {
            "f6": f6,
            "f6_prereg": PREREG_F6,
            "qmv_delta_pct": qmv_delta_pct,
            "leg_delta_pct": leg_delta_pct,
            "leg_delta_realised_pct": leg_delta_realised_pct,
            "multiple_of_null_floor": abs(leg_delta_realised_pct) / LOCAL_NULL_FLOOR_PCT,
        },
        "ranked_e53": {
            "source": "research/e53-width-mixture.json",
            "label": ("E53 ranked width mixture x E61 per-width cell cost "
                      "measured on this host"),
            "prompt_weights": e53["weights"],
            "headline_variant": E53_HEADLINE_VARIANT,
            "headline": headline,
            "variants": ranked_e53,
            "f6_min": min(f6_values),
            "f6_max": max(f6_values),
            "plutarch_low_drafting_case": plutarch,
            "published_mean_verify_width": {
                p: 1.0 + e53["prompts"][p]["published"]["mean_draft_len"]
                for p in e53["prompts"]},
        },
        "ranked_assignment_band": ranked_band,
        "prereg_comparison": {
            "prereg_cell_delta_pct": PREREG_CELL_DELTA_PCT,
            "measured_cell_delta_pct": m6_delta_pct,
            "cell_realisation_ratio": m6_delta_pct / PREREG_CELL_DELTA_PCT,
            "prereg_whole_leg_pct": PREREG_WHOLE_LEG_PCT,
            "revised_whole_leg_pct": leg_delta_realised_pct,
        },
        "decision_bands": {
            "promote_at_or_below_pct": PROMOTE_PCT,
            "report_only_below_pct": REPORT_ONLY_PCT,
            "local_null_floor_pct": LOCAL_NULL_FLOOR_PCT,
        },
        "predicted_verdict_if_realised": verdict,
        "caveat": ("This is a projection from a microbenchmark cell, not a "
                   "measurement. Rung 3 measures the whole leg directly and "
                   "that measurement decides."),
    }

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("local time-weighted QMV shares, this session's shipped costs:")
    for m, s in sorted(shares.items()):
        mark = "  <- treated" if m == 6 else ""
        print("   M=%d  %6.2f %%%s" % (m, 100 * s, mark))
    print("\nf6 = %.4f (prereg %.4f, from E54 costs)" % (f6, PREREG_F6))
    print("measured M=6 cell delta = %+.2f %% (prereg model %+.2f %%, realised %.2fx)"
          % (m6_delta_pct, PREREG_CELL_DELTA_PCT, m6_delta_pct / PREREG_CELL_DELTA_PCT))
    print("\nlocal whole-leg projection")
    print("   QMV time      %+.3f %%" % qmv_delta_pct)
    print("   MTP leg       %+.3f %%   (x psi_mtp %.6f)" % (leg_delta_pct, PSI_MTP))
    print("   after E55 realisation %.3f: %+.3f %%   (%.1fx the %.4f %% null floor)"
          % (REALISATION, leg_delta_realised_pct,
             abs(leg_delta_realised_pct) / LOCAL_NULL_FLOOR_PCT, LOCAL_NULL_FLOOR_PCT))
    print("\nranked projection: E53 ranked width mixture x this host's cell costs")
    print("   published mean VERIFY width = 1 + mean_draft_len: " + ", ".join(
        "%s %.4f" % (p, 1.0 + e53["prompts"][p]["published"]["mean_draft_len"])
        for p in sorted(e53["prompts"])))
    print("   %s round mixture mean verify width %.4f"
          % (E53_HEADLINE_VARIANT, headline["mean_verify_width_rounds"]))
    print("   headline f6=%.4f -> QMV %+.3f %%, MTP leg %+.3f %% realised"
          % (headline["f6"], headline["qmv_delta_pct"],
             headline["leg_delta_realised_pct"]))
    for variant in sorted(ranked_e53):
        if variant == E53_HEADLINE_VARIANT:
            continue
        v = ranked_e53[variant]
        print("     %-14s f6=%.4f -> MTP leg %+.3f %%"
              % (variant, v["f6"], v["leg_delta_realised_pct"]))
    print("   f6 across all six fits: %.4f - %.4f" % (min(f6_values), max(f6_values)))
    print("   E53 held-out low-drafting prompt (plutarch, mean verify width %.3f):"
          % plutarch["published_mean_verify_width"])
    print("     f6=%.4f -> MTP leg %+.3f %%   (the change buys almost nothing here)"
          % (plutarch["f6"], plutarch["leg_delta_realised_pct"]))
    print("   the published score is a MEDIAN over eight hidden prompts, so the")
    print("   realised median depends on how many of them draft deeply.")
    print("\n   assignment hand band %.1f-%.1f %% for comparison only"
          % (100 * RANKED_F6_BAND[0], 100 * RANKED_F6_BAND[1]))
    for name in ("low", "high"):
        print("     f6=%.3f -> MTP leg %+.3f %% realised"
              % (ranked_band[name]["f6"], ranked_band[name]["leg_delta_realised_pct"]))
    print("\npromote at <= %.2f %%, report-only below %+.2f %% -> projection says %s"
          % (PROMOTE_PCT, REPORT_ONLY_PCT, verdict))
    print("wrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
