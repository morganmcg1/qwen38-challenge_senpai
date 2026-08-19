#!/usr/bin/env python3
"""E53: re-price the width-share-dominated mechanisms at the SCORED mixture.

Part 3 (`research/e53_repricing.md`) re-priced the campaign's verdicts at the
new `psi_mtp = 0.693391`, but it kept askeladd's corpus-wide dispatch histogram
`HIST` (78 dispatches, E42, tree 04ad6bf1) as the width weighting, and flagged
that as caveat C2. This script closes C2 for the two mechanisms whose price is
dominated by a width share, using the per-prompt width mixtures identified in
Part 1 (`research/e53-width-mixture.json`).

Two changes versus Part 3, both of which matter:

  1. The width weighting becomes per-prompt and scored-prompt-only.
  2. The two scored prompts are priced SEPARATELY and combined through
     `score_pct_from_leg_gains()`, so the order statistics and the substitution
     kink are applied by re-sorting rather than by any composite weight. The
     0.483694 / 0.516306 marginal weights are therefore never multiplied in;
     they are recovered as a check.

Reproduce:  python3 research/e53_scored_repricing.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qmv_score_leverage as Q

PSI_MTP = 0.693391
PSI_MTP_LO = 0.692292
PSI_MTP_HI = 0.694490

MIXTURE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "e53-width-mixture.json")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "e53-scored-repricing.json")

# E44 r1 measured the same cell body per (shape, width); bit-exact 20/20 lines.
E44_R2_WINS = {("attn_out", 7): 10.46, ("attn_out", 8): 16.65,
               ("mlp_down", 7): 4.46, ("mlp_down", 8): 13.05}


def per_width_cost_shares(mixture):
    """{M: share of this prompt's QMV cost} from a round-frequency mixture."""
    priced = {int(m): p for m, p in mixture.items() if int(m) in Q.T_BY_WIDTH}
    total = sum(p * Q.T_BY_WIDTH[m] for m, p in priced.items())
    return {m: p * Q.T_BY_WIDTH[m] / total for m, p in priced.items()}


def leg_gain_pct(shares, wins, psi):
    """Candidate-leg % speedup from a per-width map of % QMV-cost wins.

    Weight first, then sum -- the module's `mechanism_value_per_width` rule.
    Every width here is >= 2, so the change is gated and the ranked price is
    `psi` with no serial-leg coupling term.
    """
    if 1 in wins:
        raise ValueError("width 1 is not gated; this pricing path assumes M>=2")
    return psi * sum(shares[m] * w for m, w in wins.items() if m in shares)


def score_pct(gains, psi_note):
    """Score % from per-prompt leg gains, kink handled by re-sorting."""
    del psi_note
    return Q.score_pct_from_leg_gains(gains)


def check_module_linearity():
    """Positive control: the module's price is linear in win_pct and in share.

    The re-weighting below rescales a module price by a share ratio. That step
    is only valid if `mechanism_value` is exactly linear in the width share, so
    prove it here instead of assuming it.
    """
    fails = []
    a = Q.mechanism_value((7, 8), 1.0, psi=PSI_MTP)
    b = Q.mechanism_value((7, 8), 3.0, psi=PSI_MTP)
    if abs(b - 3.0 * a) > 1e-12:
        fails.append("mechanism_value is not linear in win_pct")
    c = Q.mechanism_value_per_width({7: 1.0, 8: 1.0}, psi=PSI_MTP)
    expect = PSI_MTP * (Q.qmv_share(7) + Q.qmv_share(8))
    if abs(c - expect) > 1e-12:
        fails.append("mechanism_value_per_width does not equal psi * share")
    # Negative control: the comparison must be able to fail.
    if abs(Q.mechanism_value((7, 8), 1.0, psi=PSI_MTP)
           - Q.mechanism_value((7, 8), 1.0, psi=PSI_MTP * 2)) < 1e-12:
        fails.append("positive control did not fire: psi has no effect")
    return fails


def main():
    fails = check_module_linearity()
    if fails:
        for line in fails:
            print("FAIL:", line)
        return 1
    print("linearity + positive control: PASS")

    mix = json.load(open(MIXTURE_JSON))
    feasible = {p: [f for f in mix["burst"][p] if f["feasible"]]
                for p in ("beagle", "medicine")}
    for p, fits in feasible.items():
        if not fits:
            print("FAIL: no feasible burst fit for", p)
            return 1
        print("%s: %d feasible burst fits" % (p, len(fits)))

    shares = {p: [per_width_cost_shares(f["mixture"]) for f in fits]
              for p, fits in feasible.items()}

    report = {
        "psi_mtp": {"point": PSI_MTP, "low": PSI_MTP_LO, "high": PSI_MTP_HI},
        "board_floor_pct": 0.7678,
        "kink_pct": Q.kink_pct(),
        "saturation_cap_pct": Q.saturation_cap_pct(),
        "marginal_weights": Q.marginal_weights(),
        "corpus_shares": {"f78": Q.width_set_share((7, 8)),
                          "f9": Q.qmv_share(9)},
        "scored_shares": {},
        "mechanisms": {},
    }
    for p in ("beagle", "medicine"):
        f78 = [s.get(7, 0.0) + s.get(8, 0.0) for s in shares[p]]
        f9 = [s.get(9, 0.0) for s in shares[p]]
        report["scored_shares"][p] = {
            "f78": {"low": min(f78), "high": max(f78)},
            "f9": {"low": min(f9), "high": max(f9)},
        }

    # ---- Mechanism 1: the M=9 two-stream prize, <T,9,5> held at <=108 regs.
    # `<T,9,5>` runs TWO streams where the shipped `<T,9,3>` runs three
    # (ceil(9/3)=3 -> ceil(9/5)=2), so it removes exactly ONE stream.
    two_stream_win = Q.STREAM_MS / Q.T_BY_WIDTH[9] * 100.0
    corpus_linear = Q.mechanism_value((9,), two_stream_win, psi=PSI_MTP)
    corpus_order = score_pct({p: corpus_linear for p in Q.SCORED_PROMPTS}, None)
    rows = []
    for bi, sb in enumerate(shares["beagle"]):
        for mi, sm in enumerate(shares["medicine"]):
            gains = {}
            for p, s in (("beagle", sb), ("medicine", sm)):
                gains[p] = leg_gain_pct(s, {9: two_stream_win}, PSI_MTP)
            rows.append({"beagle_fit": bi, "medicine_fit": mi,
                         "leg_gain": dict(gains),
                         "score_pct": score_pct(gains, None)})
    values = [r["score_pct"] for r in rows]
    report["mechanisms"]["m9_two_stream"] = {
        "win_pct_at_m9": two_stream_win,
        "corpus_linear_pct": corpus_linear,
        "corpus_order_stat_pct": corpus_order,
        "scored_low_pct": min(values), "scored_high_pct": max(values),
        "pairs": len(rows),
        "clears_floor_at_worst": min(values) >= report["board_floor_pct"],
        "clears_floor_at_best": max(values) >= report["board_floor_pct"],
    }

    # ---- Mechanism 2: E44 r2 narrow simdgroup-matrix QMV at M in {7,8}.
    shape_mixes = {
        "attn_out only": {7: E44_R2_WINS[("attn_out", 7)],
                          8: E44_R2_WINS[("attn_out", 8)]},
        "mlp_down only": {7: E44_R2_WINS[("mlp_down", 7)],
                          8: E44_R2_WINS[("mlp_down", 8)]},
        "equal shape mix": {
            7: 0.5 * (E44_R2_WINS[("attn_out", 7)] + E44_R2_WINS[("mlp_down", 7)]),
            8: 0.5 * (E44_R2_WINS[("attn_out", 8)] + E44_R2_WINS[("mlp_down", 8)]),
        },
    }
    e44 = {}
    for label, wins in shape_mixes.items():
        corpus_lin = Q.mechanism_value_per_width(wins, psi=PSI_MTP)
        corpus_os = score_pct({p: corpus_lin for p in Q.SCORED_PROMPTS}, None)
        vals = []
        for sb in shares["beagle"]:
            for sm in shares["medicine"]:
                gains = {"beagle": leg_gain_pct(sb, wins, PSI_MTP),
                         "medicine": leg_gain_pct(sm, wins, PSI_MTP)}
                vals.append(score_pct(gains, None))
        e44[label] = {
            "corpus_linear_pct": corpus_lin,
            "corpus_order_stat_pct": corpus_os,
            "scored_low_pct": min(vals), "scored_high_pct": max(vals),
            "clears_floor_at_worst": min(vals) >= report["board_floor_pct"],
        }
    report["mechanisms"]["e44_r2_narrow_78"] = e44

    # ---- psi interval, carried on the widest mechanism only.
    interval = {}
    for name, psi in (("low", PSI_MTP_LO), ("high", PSI_MTP_HI)):
        vals = []
        for sb in shares["beagle"]:
            for sm in shares["medicine"]:
                gains = {"beagle": leg_gain_pct(sb, {9: two_stream_win}, psi),
                         "medicine": leg_gain_pct(sm, {9: two_stream_win}, psi)}
                vals.append(score_pct(gains, None))
        interval[name] = {"low": min(vals), "high": max(vals)}
    report["mechanisms"]["m9_two_stream"]["psi_interval"] = interval

    json.dump(report, open(OUT_JSON, "w"), indent=1, sort_keys=True)

    print("\n=== corpus vs scored width shares")
    print("  corpus  f{7,8}=%.4f  f9=%.4f  (HIST, 78 dispatches, E42)"
          % (report["corpus_shares"]["f78"], report["corpus_shares"]["f9"]))
    for p in ("beagle", "medicine"):
        s = report["scored_shares"][p]
        print("  %-8s f{7,8}=%.4f..%.4f  f9=%.4f..%.4f"
              % (p, s["f78"]["low"], s["f78"]["high"],
                 s["f9"]["low"], s["f9"]["high"]))

    m9 = report["mechanisms"]["m9_two_stream"]
    print("\n=== M=9 two-stream prize (<T,9,5>), win %.3f %% at M=9"
          % m9["win_pct_at_m9"])
    print("  corpus HIST : linear %+.4f %%  order-stat %+.4f %%"
          % (m9["corpus_linear_pct"], m9["corpus_order_stat_pct"]))
    print("  scored mix  : %+.4f .. %+.4f %% over %d fit pairs"
          % (m9["scored_low_pct"], m9["scored_high_pct"], m9["pairs"]))
    print("  psi interval: low %+.4f..%+.4f  high %+.4f..%+.4f"
          % (m9["psi_interval"]["low"]["low"], m9["psi_interval"]["low"]["high"],
             m9["psi_interval"]["high"]["low"], m9["psi_interval"]["high"]["high"]))
    print("  board floor %.4f %%: clears at worst=%s at best=%s"
          % (report["board_floor_pct"], m9["clears_floor_at_worst"],
             m9["clears_floor_at_best"]))

    print("\n=== E44 r2 narrow M in {7,8}")
    for label, row in report["mechanisms"]["e44_r2_narrow_78"].items():
        print("  %-16s corpus %+.4f (os %+.4f) -> scored %+.4f .. %+.4f  "
              "clears floor at worst=%s"
              % (label, row["corpus_linear_pct"], row["corpus_order_stat_pct"],
                 row["scored_low_pct"], row["scored_high_pct"],
                 row["clears_floor_at_worst"]))

    print("\n=== discipline checks")
    print("  kink %+.4f %%   saturation cap %+.4f %%"
          % (report["kink_pct"], report["saturation_cap_pct"]))
    print("  marginal weights recovered from the module: %s"
          % {k: round(v, 6) for k, v in report["marginal_weights"].items()})
    print("\nwrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
