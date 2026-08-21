#!/usr/bin/env python3
"""E114 rung 1. Re-rank every NA-resolved arm at the published operating point.

Rung 0 proves the ranked NA weight vector is NOT point identified by anything
Yukon publishes. Re-ranking an arm is therefore not a matter of swapping one
weight vector for another: the honest object is the RANGE of the arm's
published-weighted value over every width distribution consistent with the
exact board facts.

That range is a linear-fractional program over a product of simplices, one per
prompt, and it is solved exactly here by Dinkelbach iteration over the vertex
sets from `e114_width_recovery`. Three consequences are decision-grade even
though the weight vector is not:

    1. an arm whose range excludes zero movement has a PROVEN reweighting;
    2. an arm whose range lies wholly inside +-0.05 pp is PROVEN immaterial;
    3. an arm whose range straddles both needs a traced ranked width sequence.

Point estimates from the three candidate shapes are reported beside the range
and are explicitly untrusted: every one of them failed the pre-registered rung-0
validation gate.

    python3 research/e114_rerank.py
"""

from __future__ import annotations

import argparse
import json

import e114_width_recovery as wr
import scoring_weights as sw

# --- arm tables, harness=local probe percentages, NA-resolved ---------------
#
# Sign convention as recorded: positive is SLOWER than `a_base`.
# Only arms with all four NA cells recorded can be re-weighted at all. Arms
# whose per-NA cells were never written down are listed in UNRESOLVED.

ARMS: dict[str, dict] = {
    # E110 session A, alphonse, PR #112 r2, `git_head=051091ed`, g16s, ungated,
    # 5 shapes x NA 2..5 x 8 palindromic pairs. Ledger 258.8.
    "l_loadonly": {"na": {2: -3.63, 3: -6.32, 4: -17.63, 5: -29.21},
                   "src": "E110 A", "role": "diagnostic"},
    "b_constw_e110": {"na": {2: -91.31, 3: -87.78, 4: -86.37, 5: -86.17},
                      "src": "E110 A", "role": "diagnostic"},
    "b_barrier": {"na": {2: -0.63, 3: +0.12, 4: +0.41, 5: +0.40},
                  "src": "E110 A", "role": "arm"},
    "xs_stage": {"na": {2: +0.22, 3: -0.81, 4: -1.44, 5: -2.79},
                 "src": "E110 A", "role": "arm"},
    "mo_stage": {"na": {2: +60.66, 3: +127.10, 4: +163.21, 5: +176.63},
                 "src": "E110 A", "role": "arm"},
    "mo_swap": {"na": {2: +60.50, 3: +127.55, 4: +165.93, 5: +180.61},
                "src": "E110 A", "role": "arm"},
    # E111, thorfinn. Only `b_constw` has its per-NA cells recorded, in the
    # withdrawal note at ledger 258.3 item 2.
    "b_constw_e111": {"na": {2: +12.77, 3: +1.85, 4: -2.15, 5: +0.50},
                      "src": "E111", "role": "withdrawn"},
    # FINDING 44, alphonse's measured roofline pair. The gap is
    # (a_base - max(load_only, alu_only)) / max(...), per NA.
    "f44_roofline_gap": {"na": {2: +3.5, 3: +6.0, 4: +21.2, 5: +41.1},
                         "src": "Finding 44", "role": "quantity"},
}

# Arms the advisor listed that CANNOT be re-weighted from anything in this
# checkout, with the exact artefact that would unblock each one.
UNRESOLVED = {
    "xv4": "E110 session B per-NA cells; alphonse PR #112 was still wip at the "
           "recorded base and `research/out/<tag>/arms.json` is host-local",
    "xv4_stage": "same as xv4",
    "xv8": "same as xv4",
    "c_loadonly": "E111 reduced to `weighted %` only; per-NA cells live in "
                  "thorfinn's host-local `research/out/<tag>/arms.json`",
    "n_nobias": "same as c_loadonly",
    "n_nosums": "same as c_loadonly, except the single cell NA=5 = +9.93 %",
    "g_pack32": "same as c_loadonly",
    "d_bias1": "same as c_loadonly",
    "e_bias6": "same as c_loadonly",
    "h_prunenarrow": "E108 reported one pooled percentage, never per NA",
    "i_pruneall": "same as h_prunenarrow",
}

# Finding 16 rank-5 occupancy over 81 strong runs.
MIN_SLOT = dict(sw.MIN_SLOT_OCCUPANCY)


def prompt_mix(rec: dict) -> dict[str, float]:
    """Score sensitivity of the published median to each prompt's candidate time.

    `published = 0.5 * raw_beagle + 0.5 * raw_5th`, and `raw_p` is inversely
    proportional to that prompt's candidate seconds per token, so a relative
    change `dp` in prompt p's candidate time moves the published median by
    `-0.5 * raw_p * dp / published`. Dividing by the prompt's candidate time
    converts that into a weight on ABSOLUTE microseconds, which is the unit the
    per-NA arm work is measured in.
    """
    pub = rec["published"]
    out = {}
    for name, share in [("beagle", 1.0)] + list(MIN_SLOT.items()):
        p = rec["prompts"][name]
        slot = 0.5 if name == "beagle" else 0.5 * share / sum(MIN_SLOT.values())
        out[name] = slot * p["raw"] / (pub * p["mtp_us_per_token"])
    return out


def cell_us(vertex: dict[int, float], p: dict, rates: dict) -> dict[int, float]:
    """Absolute wide-QMV group microseconds per token, per NA cell."""
    scale = (p["rounds"] / wr.T) * (1.0 - p["p_width1"])
    acc = {na: 0.0 for na in sw.NA_CELLS}
    for M, mass in vertex.items():
        for na in sw.PARTITION[M]:
            if na in acc:
                acc[na] += scale * mass / rates[na]
    return acc


def arm_range(delta: dict[int, float], vsets: dict, mix: dict, rec: dict,
              rates: dict, hi: bool) -> float:
    """Exact extremum of the published-weighted arm value (Dinkelbach)."""
    lo_t, hi_t = min(delta.values()) - 1.0, max(delta.values()) + 1.0
    for _ in range(200):
        t = 0.5 * (lo_t + hi_t)
        total = 0.0
        for name, w in mix.items():
            vals = []
            for v in vsets[name]:
                u = cell_us(v, rec["prompts"][name], rates)
                vals.append(sum(u[na] * (delta[na] - t) for na in sw.NA_CELLS))
            total += w * (max(vals) if hi else min(vals))
        # `total` is g(t) = extremum of N(P) - t*D(P) and is strictly
        # decreasing in t, so its root is the extremum of N/D for both signs.
        if total > 0:
            lo_t = t
        else:
            hi_t = t
    return 0.5 * (lo_t + hi_t)


def point_weights(shape: str, vsets_pt: dict, mix: dict, rec: dict,
                  rates: dict) -> dict[int, float]:
    """Published NA weight vector under one named candidate width shape."""
    acc = {na: 0.0 for na in sw.NA_CELLS}
    for name, w in mix.items():
        u = cell_us(vsets_pt[shape][name], rec["prompts"][name], rates)
        for na in sw.NA_CELLS:
            acc[na] += w * u[na]
    tot = sum(acc.values())
    return {na: acc[na] / tot for na in sw.NA_CELLS}


def build(rec: dict, rates: dict, use_cost: bool):
    tol = wr.ROUTE_B["max_resid_pct"] / 100.0
    vsets, vsets_pt = {}, {s: {} for s in ("maxent", "gt1", "gt2")}
    for name in ["beagle", *MIN_SLOT]:
        p = rec["prompts"][name]
        p1 = p["p_width1"]
        mean_wide = (p["mean_width"] - p1) / (1.0 - p1)
        extra = wr.cost_band(p["round_us"], p1, tol) if use_cost else None
        vsets[name] = wr.vertices(mean_wide, extra=extra)
        vsets_pt["maxent"][name] = wr.maxent(mean_wide)
        vsets_pt["gt1"][name] = wr.tilt(wr.GROUND_TRUTH["GT1"]["hist"],
                                        mean_wide)
        vsets_pt["gt2"][name] = wr.tilt(wr.GROUND_TRUTH["GT2"]["hist"],
                                        mean_wide)
    return vsets, vsets_pt


# --- item 5: is E100 worth its register tax at the published operating point?

# The register tax of keeping the `<T,5,5,true>` instantiation. E108's
# `i_minus_case5` moved g17s from 98 to 91 registers, which moves
# `S_ranked = floor(496 KiB / (128 R))` from 40 to 43. Two independent prices:
REGISTER_TAX = {
    "e77_law": 0.0974,      # Omega(S) = (32/S)^0.01346, predicted
    "e102_measured": 0.1068,  # measured, but only on the G=1 prompts
}
# Three recorded prices for the collapse itself, none reconcilable with the
# others. Positive means the collapse makes an M=5 round faster.
COLLAPSE_PRICES = {
    "e104_f33_ranked": {"pct": +1.9, "band": None,
                        "note": "E104 Finding 33, measured ranked"},
    "f32_route1": {"pct": -2.0, "band": None,
                   "note": "Finding 32 ranked route 1"},
    "f32_route2": {"pct": -0.3, "band": 1.5,
                   "note": "Finding 32 ranked route 2"},
}


def collapse_arithmetic(rec: dict, mix: dict) -> dict:
    """What share of published-weighted candidate time sits at exactly M=5?

    The collapse can only act on rounds at verify width 5. The register tax
    acts on every round. Both are expressed as a percentage of candidate time,
    so the net is `gain * share - tax`.
    """
    tol = wr.ROUTE_B["max_resid_pct"] / 100.0
    shares, per_prompt = {}, {}
    for name in ["beagle", *MIN_SLOT]:
        p = rec["prompts"][name]
        p1 = p["p_width1"]
        mean_wide = (p["mean_width"] - p1) / (1.0 - p1)
        extra = wr.cost_band(p["round_us"], p1, tol)
        verts = wr.vertices(mean_wide, extra=extra)
        c5 = wr.route_b_us(5.0)

        def time_share(v):
            tot = sum(m * wr.route_b_us(float(M)) for M, m in v.items())
            return (1.0 - p1) * v.get(5, 0.0) * c5 / (
                (1.0 - p1) * tot + p1 * wr.route_b_us(1.0))

        pts = {s: time_share(d) for s, d in (
            ("maxent", wr.maxent(mean_wide)),
            ("gt1", wr.tilt(wr.GROUND_TRUTH["GT1"]["hist"], mean_wide)),
            ("gt2", wr.tilt(wr.GROUND_TRUTH["GT2"]["hist"], mean_wide)))}
        per_prompt[name] = {
            "lo": min(time_share(v) for v in verts),
            "hi": max(time_share(v) for v in verts), "points": pts}
        shares[name] = pts

    tot = sum(mix.values())
    agg = {s: sum(mix[n] / tot * shares[n][s] for n in shares)
           for s in ("maxent", "gt1", "gt2")}
    agg_lo = sum(mix[n] / tot * per_prompt[n]["lo"] for n in per_prompt)
    agg_hi = sum(mix[n] / tot * per_prompt[n]["hi"] for n in per_prompt)

    # A fourth candidate price: the pre-E100 and post-E100 ranked two-line
    # refits evaluated at M=5. Both fits come from official receipts, so the
    # difference looks like an end-to-end price for E100.
    #
    # It is not one. E100 changed the partition only at M=5; at M=6, 7 and 8
    # the partition is identical across the two fits, so the same difference
    # taken there is a PLACEBO and measures nothing but drift between the two
    # receipt populations. Report the placebo beside the treatment and refuse
    # the price when the placebo is the same size or larger.
    def _diff(M: float) -> dict:
        pre = wr.route_b_us(M, wr.ROUTE_B_PRE_E100)
        post = wr.route_b_us(M)
        return {"pre_us": pre, "post_us": post,
                "pct": 100.0 * (pre - post) / pre}

    treat = _diff(5.0)
    placebo = {int(M): _diff(float(M)) for M in (6, 7, 8)}
    worst_placebo = max(abs(v["pct"]) for v in placebo.values())
    curve = dict(treat)
    curve.update({
        "resid_pct": wr.ROUTE_B["max_resid_pct"],
        "placebo_unchanged_partition": placebo,
        "worst_placebo_pct": worst_placebo,
        "exceeds_fit_residual": abs(treat["pct"]) > wr.ROUTE_B["max_resid_pct"],
        "exceeds_placebo": abs(treat["pct"]) > worst_placebo,
        "usable_as_price": (abs(treat["pct"]) > wr.ROUTE_B["max_resid_pct"]
                            and abs(treat["pct"]) > worst_placebo),
        "note": "M=6,7,8 share one partition across both fits, so any "
                "difference there is drift. The register tax should make the "
                "post-E100 fit SLOWER at M>=6 by about 0.10 %; it does not.",
    })

    nets = {}
    for pname, price in COLLAPSE_PRICES.items():
        for tname, tax in REGISTER_TAX.items():
            g = price["pct"]
            band = price["band"] or 0.0
            nets["%s/%s" % (pname, tname)] = {
                "gain_pct": g, "tax_pct": tax,
                "net_point": g * agg["maxent"] - tax,
                "net_lo": (g - band) * (agg_lo if g - band > 0 else agg_hi)
                - tax,
                "net_hi": (g + band) * (agg_hi if g + band > 0 else agg_lo)
                - tax}
    return {"per_prompt_m5_time_share": per_prompt,
            "published_weighted_share": {"points": agg, "lo": agg_lo,
                                         "hi": agg_hi},
            "register_tax": REGISTER_TAX, "collapse_prices": COLLAPSE_PRICES,
            "curve_difference_price": curve, "nets": nets}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--receipt", default="b8b8b860")
    ap.add_argument("--json", default="research/e114-artifacts/rung1.json")
    args = ap.parse_args()

    rec = wr.load_receipt(args.board, args.receipt)
    mix = prompt_mix(rec)
    standing = {int(k): v for k, v in sw.STANDING_WEIGHTS.items()}
    out = {"receipt": rec["id"], "prompt_mix": mix, "unresolved": UNRESOLVED,
           "standing_weights": sw.STANDING_WEIGHTS, "frames": {}}

    print("=" * 78)
    print("E114 rung 1 - arm re-rank at the published operating point")
    print("=" * 78)
    print("\nscore-sensitivity mix over prompts (Finding 16, receipt %s)"
          % rec["id"][:8])
    tot = sum(mix.values())
    for name, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        print("  %-9s %.6f   share %.3f" % (name, w, w / tot))

    for frame, rates in (("local", sw.ONE_GROUP_GBPS),
                         ("ranked", sw.RANKED_ONE_GROUP_GBPS)):
        vsets, vsets_pt = build(rec, rates, use_cost=True)
        pts = {s: point_weights(s, vsets_pt, mix, rec, rates)
               for s in vsets_pt}
        print("\n-- rate table harness=%s, widths harness=ranked --" % frame)
        print("   standing weights   %s" % wr._fmt(standing))
        for s, w in pts.items():
            print("   published (%-6s) %s" % (s, wr._fmt(w)))
        # positive control: a flat arm must price identically under every
        # distribution, so its range has to collapse onto the flat value.
        flat = {na: 7.0 for na in sw.NA_CELLS}
        f_lo = arm_range(flat, vsets, mix, rec, rates, hi=False)
        f_hi = arm_range(flat, vsets, mix, rec, rates, hi=True)
        assert abs(f_lo - 7.0) < 1e-6 and abs(f_hi - 7.0) < 1e-6, (f_lo, f_hi)
        print("   control: a flat arm prices [%.6f, %.6f], expected 7"
              % (f_lo, f_hi))
        rows = []
        for arm, spec in ARMS.items():
            d = spec["na"]
            base = sw.weighted(d, standing)
            lo = arm_range(d, vsets, mix, rec, rates, hi=False)
            hi = arm_range(d, vsets, mix, rec, rates, hi=True)
            pv = {s: sw.weighted(d, w) for s, w in pts.items()}
            move_lo, move_hi = lo - base, hi - base
            guaranteed = 0.0 if move_lo <= 0 <= move_hi \
                else (move_lo if move_lo > 0 else -move_hi)
            rows.append({
                "arm": arm, "src": spec["src"], "role": spec["role"],
                "na": d, "standing_pct": base,
                "published_lo": lo, "published_hi": hi,
                "move_lo_pp": move_lo, "move_hi_pp": move_hi,
                "guaranteed_move_pp": guaranteed,
                "max_move_pp": max(abs(move_lo), abs(move_hi)),
                "sign_identified": (lo > 0) == (hi > 0),
                "immaterial_proven": max(abs(move_lo), abs(move_hi)) < 0.05,
                "points": pv})
        out["frames"][frame] = {"points": pts, "arms": rows}
        print("\n  %-16s %9s %19s %10s %10s %s"
              % ("arm", "standing", "published range", "move pp", "guaranteed",
                 "point est (maxent/gt1/gt2)"))
        for r in sorted(rows, key=lambda r: -r["max_move_pp"]):
            print("  %-16s %+9.3f [%+8.3f,%+8.3f] %+9.3f/%+.3f %8.3f  "
                  "%+.3f/%+.3f/%+.3f%s"
                  % (r["arm"], r["standing_pct"], r["published_lo"],
                     r["published_hi"], r["move_lo_pp"], r["move_hi_pp"],
                     r["guaranteed_move_pp"], r["points"]["maxent"],
                     r["points"]["gt1"], r["points"]["gt2"],
                     "  SIGN FLIPS" if not r["sign_identified"] else ""))

    ranked = out["frames"]["ranked"]["arms"]
    e110 = [r for r in ranked if r["src"] == "E110 A"]
    out["primary_metric"] = {
        "name": "e114_max_abs_arm_reweight_pp",
        "definition": "max over the E110 arm table of |published-weighted - "
                      "standing-weighted|, at the maxent width shape, rate "
                      "table harness=ranked",
        "value": max(abs(r["points"]["maxent"] - r["standing_pct"])
                     for r in e110),
        "shape_ensemble_lo": min(min(r["points"][s] for s in r["points"])
                                 - r["standing_pct"] for r in e110),
        "shape_ensemble_hi": max(max(r["points"][s] for s in r["points"])
                                 - r["standing_pct"] for r in e110),
        "identified_set_upper_bound": max(r["max_move_pp"] for r in e110),
        "guaranteed_lower_bound": max(r["guaranteed_move_pp"] for r in e110),
        "kill_rule_pp": 0.05,
    }
    pm = out["primary_metric"]
    print("\ne114_max_abs_arm_reweight_pp = %.4f pp   (kill rule 0.05, "
          "KILL RULE %s)" % (pm["value"],
                             "FIRES" if pm["value"] < 0.05 else "DOES NOT FIRE"))
    print("  shape ensemble spread %+.4f to %+.4f pp;  identified-set upper "
          "bound %.4f pp;  guaranteed lower bound %.4f pp"
          % (pm["shape_ensemble_lo"], pm["shape_ensemble_hi"],
             pm["identified_set_upper_bound"], pm["guaranteed_lower_bound"]))

    print("\n-- the reweighting vector any arm owner can apply themselves --")
    for s in ("maxent", "gt1", "gt2"):
        dw = {na: out["frames"]["ranked"]["points"][s][na] - standing[na]
              for na in sw.NA_CELLS}
        print("  dW (%-6s) %s   move_pp = sum_NA dW[NA] * arm_pct[NA]"
              % (s, "  ".join("NA%d %+0.4f" % (na, dw[na])
                              for na in sw.NA_CELLS)))
        out.setdefault("delta_weights", {})[s] = dw

    it5 = out["item5"] = collapse_arithmetic(rec, mix)
    print("\n" + "=" * 78)
    print("ITEM 5 - is E100's M=5 collapse worth its register tax?")
    print("=" * 78)
    print("  share of candidate TIME at exactly verify width 5, harness=ranked")
    print("  %-9s %8s %8s %8s | %s" % ("prompt", "maxent", "gt1", "gt2",
                                       "identified set"))
    for name, r in sorted(it5["per_prompt_m5_time_share"].items()):
        print("  %-9s %8.4f %8.4f %8.4f | [%.4f, %.4f]"
              % (name, r["points"]["maxent"], r["points"]["gt1"],
                 r["points"]["gt2"], r["lo"], r["hi"]))
    ag = it5["published_weighted_share"]
    print("  %-9s %8.4f %8.4f %8.4f | [%.4f, %.4f]   <- published weighting"
          % ("COMBINED", ag["points"]["maxent"], ag["points"]["gt1"],
             ag["points"]["gt2"], ag["lo"], ag["hi"]))
    cd = it5["curve_difference_price"]
    print("\n  fourth candidate price, from our own receipts: pre-E100 M=5 "
          "round %.0f us, post-E100 %.0f us, %+.3f %% (fit residual %.2f %%)"
          % (cd["pre_us"], cd["post_us"], cd["pct"], cd["resid_pct"]))
    print("  PLACEBO at widths whose partition E100 did NOT change:")
    for M in sorted(cd["placebo_unchanged_partition"]):
        pl = cd["placebo_unchanged_partition"][M]
        print("    M=%d  pre %.0f us, post %.0f us, %+.3f %%"
              % (M, pl["pre_us"], pl["post_us"], pl["pct"]))
    print("  treatment %+.3f %% vs worst placebo %.3f %% -> usable_as_price=%s"
          % (cd["pct"], cd["worst_placebo_pct"], cd["usable_as_price"]))
    print("  %s" % cd["note"])
    print("\n  net = gain x share - tax, published weighting, %% of candidate "
          "time")
    print("  %-32s %8s %8s %10s %s" % ("collapse price / tax", "gain", "tax",
                                       "net point", "net band"))
    for k, v in it5["nets"].items():
        print("  %-32s %+8.2f %8.4f %+10.4f [%+.4f, %+.4f]"
              % (k, v["gain_pct"], v["tax_pct"], v["net_point"],
                 v["net_lo"], v["net_hi"]))

    import os
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as h:
        json.dump(out, h, indent=1, sort_keys=True, default=str)
    print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
