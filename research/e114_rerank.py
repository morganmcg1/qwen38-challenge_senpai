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
    # E110 r2, alphonse, sessions A+B+C on one protocol: 5 scored shapes x
    # NA 2..5 x 8 palindromic pairs, first block discarded, paired per-block
    # ratio, `applegpu_g16s`, ungated. Ledger 259.8, the authoritative table.
    # `recorded` is the ledger's own reduction and is reproduced, not reused.
    "l_loadonly": {"na": {2: -3.63, 3: -6.32, 4: -17.63, 5: -29.21},
                   "src": "E110", "role": "diagnostic",
                   "recorded": {"weighted": -14.578, "round": -11.354,
                                "ranked": -10.786}},
    "b_constw_e110": {"na": {2: -91.31, 3: -87.78, 4: -86.37, 5: -86.17},
                      "src": "E110", "role": "diagnostic",
                      "recorded": {"weighted": -86.868, "round": -67.655,
                                   "ranked": -64.272}},
    "b_barrier": {"na": {2: -0.76, 3: +0.11, 4: +0.38, 5: +0.31},
                  "src": "E110", "role": "arm",
                  "recorded": {"weighted": +0.274, "round": +0.213,
                               "ranked": +0.202}},
    "xs_stage": {"na": {2: +0.35, 3: -0.83, 4: -1.71, 5: -2.75},
                 "src": "E110", "role": "arm",
                 "recorded": {"weighted": -1.458, "round": -1.135,
                              "ranked": -1.078}},
    "xv4": {"na": {2: -0.00, 3: -0.39, 4: -0.74, 5: -2.02},
            "src": "E110", "role": "promotion candidate",
            "recorded": {"weighted": -0.673, "round": -0.524,
                         "ranked": -0.498}},
    "xv4_stage": {"na": {2: +0.33, 3: -0.88, 4: -2.31, 5: -2.45},
                  "src": "E110", "role": "arm",
                  "recorded": {"weighted": -1.859, "round": -1.448,
                               "ranked": -1.375}},
    "xv8": {"na": {2: +0.52, 3: +10.32, 4: +5.51, 5: +42.59},
            "src": "E110", "role": "arm",
            "recorded": {"weighted": +7.975, "round": +6.211,
                         "ranked": +5.900}},
    "mo_swap": {"na": {2: +60.50, 3: +127.55, 4: +165.93, 5: +180.61},
                "src": "E110", "role": "arm",
                "recorded": {"weighted": +153.341, "round": +119.426,
                             "ranked": +113.454}},
    "mo_stage": {"na": {2: +60.66, 3: +127.10, 4: +163.21, 5: +176.63},
                 "src": "E110", "role": "arm",
                 "recorded": {"weighted": +151.277, "round": +117.818,
                              "ranked": +111.927}},
    # Ledger 259.8 records these two to one significant figure only ("~ +80",
    # "~ +21"), so their `chk` cannot be scored, but their cells are exact.
    "mu_swap": {"na": {2: +1.15, 3: +71.61, 4: +83.26, 5: +98.96},
                "src": "E110", "role": "arm", "recorded": None},
    "mo_hoist": {"na": {2: +2.07, 3: +11.21, 4: +18.47, 5: +185.29},
                 "src": "E110", "role": "arm", "recorded": None},
    # E111, thorfinn. Only `b_constw` has its per-NA cells recorded, in the
    # withdrawal note at ledger 258.3 item 2.
    "b_constw_e111": {"na": {2: +12.77, 3: +1.85, 4: -2.15, 5: +0.50},
                      "src": "E111", "role": "withdrawn", "recorded": None},
    # FINDING 44, alphonse's measured roofline pair. The gap is
    # (a_base - max(load_only, alu_only)) / max(...), per NA.
    "f44_roofline_gap": {"na": {2: +3.5, 3: +6.0, 4: +21.2, 5: +41.1},
                         "src": "Finding 44", "role": "quantity",
                         "recorded": None},
}

# The advisor's own reweighting of the same table, ledger 260.9 item 2, applied
# with the dW formula this experiment published. Reproduced here so any
# disagreement is visible rather than silent.
ADVISOR_F2 = {
    "b_barrier": {"chk": 0.276, "standing": +0.274, "maxent": +0.231,
                  "gt1": +0.149, "gt2": +0.298, "max_move": 0.125},
    "xs_stage": {"chk": -1.454, "standing": -1.458, "maxent": -1.431,
                 "gt1": -1.240, "gt2": -1.686, "max_move": 0.228},
    "xv4": {"chk": -0.670, "standing": -0.673, "maxent": -0.721,
            "gt1": -0.618, "gt2": -0.890, "max_move": 0.217},
    "xv4_stage": {"chk": -1.858, "standing": -1.859, "maxent": -1.722,
                  "gt1": -1.524, "gt2": -1.982, "max_move": 0.335},
    "xv8": {"chk": 7.974, "standing": +7.975, "maxent": +10.736,
            "gt1": +8.907, "gt2": +13.474, "max_move": 5.499},
    "l_loadonly": {"chk": -14.577, "standing": -14.578, "maxent": -14.303,
                   "gt1": -12.866, "gt2": -16.942, "max_move": 2.364},
    "mo_swap": {"chk": 153.344, "standing": +153.341, "maxent": +149.830,
                "gt1": +141.493, "gt2": +159.097, "max_move": 11.848},
}

# Arms the advisor listed that CANNOT be re-weighted from anything in this
# checkout, with the exact artefact that would unblock each one.
UNRESOLVED = {
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

SHAPES = ("maxent", "gt1", "gt2", "policy")

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


def load_policy_shapes(path: str) -> tuple[dict[str, dict[int, float]], dict]:
    """The `costModelDepth` generator's point, from `e114_policy_sim`.

    Model B is the only one that reproduces the board's accepted-draft rate,
    so it is the shape this module prices. Model A stays in the rung-1b
    artefact as the falsified alternative. The generator's own held-out gate
    is returned with it, because model B fails that gate and the column must
    be read as a diagnostic rather than as a fourth credible weight vector.
    """
    doc = json.load(open(path))
    return ({p: {int(M): m for M, m in d.items()}
             for p, d in doc["shapes"]["B"].items()},
            doc["promotion_gate"]["B"])


def build(rec: dict, rates: dict, use_cost: bool, policy: dict):
    tol = wr.ROUTE_B["max_resid_pct"] / 100.0
    vsets, vsets_pt = {}, {s: {} for s in SHAPES}
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
        vsets_pt["policy"][name] = policy[name]
    return vsets, vsets_pt


# --- the `weighted % -> round %` factor, which also moves -------------------

def factor_rows(rec: dict, vsets: dict, vsets_pt: dict, mix: dict) -> dict:
    """The campaign's 0.7788 scalar, rebuilt as a function of the widths.

    harness=local on both the rate table and the round-time table: the arms
    were timed on g16s and the `round %` column is a local round. Only the
    WIDTHS come from the board. Widths above 5 use an extrapolated round time,
    and the mass that depends on it is carried through.
    """
    def full(dist: dict[int, float], p1: float) -> dict[int, float]:
        out = {M: (1.0 - p1) * m for M, m in dist.items()}
        if p1 > 0:
            out[1] = out.get(1, 0.0) + p1
        return out

    tot = sum(mix.values())
    per_prompt, points = {}, {}
    for s in vsets_pt:
        num = den = ex = 0.0
        for name in vsets_pt[s]:
            p = rec["prompts"][name]
            f = sw.qmv_share_of_round(full(vsets_pt[s][name], p["p_width1"]))
            w = mix[name] / tot
            num += w * f["qmv_us"]
            den += w * f["round_us"]
            ex += w * f["extrapolated_mass"]
            per_prompt.setdefault(name, {})[s] = f["factor"]
        points[s] = {"factor": num / den, "extrapolated_mass": ex}
    lo, hi = [], []
    for name in vsets:
        p = rec["prompts"][name]
        vals = [sw.qmv_share_of_round(full(v, p["p_width1"]))["factor"]
                for v in vsets[name]]
        lo.append((mix[name] / tot, min(vals)))
        hi.append((mix[name] / tot, max(vals)))
    m5 = sw.qmv_share_of_round({5: 1.0})
    e106 = sw.qmv_share_of_round(sw.E106_LOCAL_HISTOGRAM)
    return {
        "ledger_scalar": sw.LEDGER_WEIGHTED_TO_ROUND,
        "m5_point": m5["factor"],
        "m5_point_vs_ledger_pct":
            100.0 * (m5["factor"] - sw.LEDGER_WEIGHTED_TO_ROUND)
            / sw.LEDGER_WEIGHTED_TO_ROUND,
        "e106_realised": e106["factor"],
        "e106_extrapolated_mass": e106["extrapolated_mass"],
        "published": points, "per_prompt": per_prompt,
        "published_lo": sum(w * v for w, v in lo),
        "published_hi": sum(w * v for w, v in hi),
        "local_to_ranked": sw.LOCAL_TO_RANKED,
        "note": "the ledger scalar is the M=5 POINT of this same function, so "
                "the standing conversion is the local operating point on the "
                "width axis as well as on the NA axis",
    }


# --- item 5: is E100 worth its register tax at the published operating point?

# The register tax of keeping the `<T,5,5,true>` instantiation. E108's
# `i_minus_case5` moved g17s from 98 to 91 registers, which moves
# `S_ranked = floor(496 KiB / (128 R))` from 40 to 43. Two independent prices:
REGISTER_TAX = {
    "e77_law": 0.0974,      # Omega(S) = (32/S)^0.01346, predicted
    "e102_measured": 0.1068,  # measured, but only on the G=1 prompts
}

# The three recorded collapse prices are NOT three independent measurements.
# `research/group_scaling.py` derives every one of them from a single local
# group-scaling coefficient `A_local` through one fixed ranked conversion:
#
#     A_ranked    = A_local * adv,  adv = sc_ranked / sc_local
#     ranked gain = 1 - A_ranked / 2
#
# so the prices differ only in the value of `A_local` fed in. Recompute `adv`
# from the same round times that script uses, rather than quoting it.
GS_LOCAL_US = {3: 74778, 5: 126103}
GS_RANKED_US = {3: 39167, 5: 53108}


def ranked_advantage() -> float:
    """`adv` from Finding 32, recomputed from its own round times.

    Both `[3] -> [3+2]` aggregate scalings use `Gof[5] = 2`, the PRE-E100
    partition map, on both hosts. The ratio is what transfers, so the stale map
    cancels; the absolute rates keyed to M=5 in those tables do not.
    """
    sc_loc = 2 * GS_LOCAL_US[3] / GS_LOCAL_US[5]
    sc_rnk = 2 * GS_RANKED_US[3] / GS_RANKED_US[5]
    return sc_rnk / sc_loc


def gain_from_a_local(a_local: float) -> float:
    """Ranked per-M=5-round collapse gain, in percent. Positive means faster."""
    return 100.0 * (1.0 - a_local * ranked_advantage() / 2.0)


# Positive means the collapse makes an M=5 round faster. `kind` records what
# the number actually is: a PER-ROUND price, which must be multiplied by our
# own M=5 share, or an already-measured LEG effect, which must not.
COLLAPSE_PRICES = {
    "e104_f33_ranked": {
        "pct": +1.9, "band": None, "kind": "per_round",
        "a_local": 1.577, "a_local_range": (1.552, 1.641),
        "a_local_provenance": "measured directly, isolated one-group against "
                              "the shipped split, 5 shapes, palindrome order",
        "note": "E104 Finding 33 row M=5"},
    "f32_route1": {
        "pct": -2.0, "band": None, "kind": "per_round",
        "a_local": 1.640, "a_local_range": None,
        "a_local_provenance": "ALGEBRA on COLLAPSE_MEASURED=0.180, not an "
                              "independent measurement of concurrency",
        "note": "Finding 32 ranked route 1"},
    "f32_route2": {
        "pct": -0.3, "band": 1.5, "kind": "leg_effect",
        "leg_effect_pct": -0.070, "leg_effect_band": 0.360,
        "assumed_share": 0.24,
        "a_local": None, "a_local_range": None,
        "a_local_provenance": "backed out of two rival receipts; the MEASURED "
                              "quantity is the leg effect, and the per-round "
                              "price only exists after dividing by an assumed "
                              "M=5 share",
        "note": "Finding 32 ranked route 2"},
}


def collapse_arithmetic(rec: dict, mix: dict, policy: dict) -> dict:
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

        # Finding 32 route 2 divides by a different quantity: the share of
        # PRE-E100 two-group rounds that sat at exactly M=5, i.e. rounds, not
        # time, and conditioned on M >= 5. Recover it so route 2's assumed 0.24
        # can be checked rather than inherited.
        def g2_round_share(v):
            ge5 = sum(m for M, m in v.items() if M >= 5)
            return v.get(5, 0.0) / ge5 if ge5 > 0 else 0.0

        named = (("maxent", wr.maxent(mean_wide)),
                 ("gt1", wr.tilt(wr.GROUND_TRUTH["GT1"]["hist"], mean_wide)),
                 ("gt2", wr.tilt(wr.GROUND_TRUTH["GT2"]["hist"], mean_wide)),
                 ("policy", policy[name]))
        pts = {s: time_share(d) for s, d in named}
        per_prompt[name] = {
            "lo": min(time_share(v) for v in verts),
            "hi": max(time_share(v) for v in verts), "points": pts,
            "g2_round_share": {s: g2_round_share(d) for s, d in named},
            "g2_round_share_lo": min(g2_round_share(v) for v in verts),
            "g2_round_share_hi": max(g2_round_share(v) for v in verts)}
        shares[name] = pts

    tot = sum(mix.values())
    agg = {s: sum(mix[n] / tot * shares[n][s] for n in shares)
           for s in SHAPES}
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
            g, band = price["pct"], price["band"] or 0.0
            if price["kind"] == "leg_effect":
                # The measured quantity is already `gain * share`, so our own
                # share recovery cannot move it and must not be applied again.
                eff = price["leg_effect_pct"]
                eff_band = price["leg_effect_band"]
                lo, hi = eff - eff_band, eff + eff_band
                point = eff
            else:
                point = g * agg["maxent"]
                lo = (g - band) * (agg_lo if g - band > 0 else agg_hi)
                hi = (g + band) * (agg_hi if g + band > 0 else agg_lo)
            nets["%s/%s" % (pname, tname)] = {
                "kind": price["kind"], "gain_pct": g, "tax_pct": tax,
                "share_applied": price["kind"] == "per_round",
                "net_point": point - tax, "net_lo": lo - tax,
                "net_hi": hi - tax}

    # The prices are not independent. Every per-round price is
    # `1 - A_local * adv / 2`, so they differ only in `A_local`. Sweep it.
    adv = ranked_advantage()
    breakeven = 2.0 / adv
    e104 = COLLAPSE_PRICES["e104_f33_ranked"]
    lo_a, hi_a = e104["a_local_range"]
    recon = {
        "ranked_advantage_adv": adv,
        "gain_is": "100 * (1 - A_local * adv / 2)",
        "breakeven_a_local": breakeven,
        "e104_measured_a_local": e104["a_local"],
        "e104_measured_a_local_range": [lo_a, hi_a],
        "e104_range_contains_breakeven": lo_a <= breakeven <= hi_a,
        "sweep": {("%.3f" % a): gain_from_a_local(a)
                  for a in (lo_a, e104["a_local"], breakeven,
                            COLLAPSE_PRICES["f32_route1"]["a_local"], hi_a)},
        "note": "E104 F33 and F32 route 1 pass different values of ONE "
                "coefficient through one transform. The sign of the collapse "
                "gain is undetermined because E104's own measured range for "
                "A_local straddles the break-even point.",
    }

    # Route 2's assumed M=5 share of pre-E100 two-group rounds, checked against
    # the recovered ranked distributions.
    r2 = COLLAPSE_PRICES["f32_route2"]
    r2_share = {
        "assumed": r2["assumed_share"],
        "recovered_points": {
            s: sum(mix[n] / tot * per_prompt[n]["g2_round_share"][s]
                   for n in per_prompt) for s in SHAPES},
        "recovered_lo": sum(mix[n] / tot * per_prompt[n]["g2_round_share_lo"]
                            for n in per_prompt),
        "recovered_hi": sum(mix[n] / tot * per_prompt[n]["g2_round_share_hi"]
                            for n in per_prompt),
    }
    r2_share["assumption_inside_identified_set"] = (
        r2_share["recovered_lo"] <= r2["assumed_share"]
        <= r2_share["recovered_hi"])

    return {"per_prompt_m5_time_share": per_prompt,
            "published_weighted_share": {"points": agg, "lo": agg_lo,
                                         "hi": agg_hi},
            "register_tax": REGISTER_TAX, "collapse_prices": COLLAPSE_PRICES,
            "curve_difference_price": curve, "nets": nets,
            "price_reconciliation": recon, "route2_share_check": r2_share}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--receipt", default="b8b8b860")
    ap.add_argument("--policy", default="research/e114-artifacts/rung1b.json")
    ap.add_argument("--json", default="research/e114-artifacts/rung1.json")
    args = ap.parse_args()

    rec = wr.load_receipt(args.board, args.receipt)
    mix = prompt_mix(rec)
    policy, policy_gate = load_policy_shapes(args.policy)
    standing = {int(k): v for k, v in sw.STANDING_WEIGHTS.items()}
    out = {"receipt": rec["id"], "prompt_mix": mix, "unresolved": UNRESOLVED,
           "standing_weights": sw.STANDING_WEIGHTS, "frames": {},
           "policy_shape_source": args.policy, "policy_gate": policy_gate}

    print("=" * 78)
    print("E114 rung 1 - arm re-rank at the published operating point")
    print("=" * 78)
    print("\nthe `policy` column is a DIAGNOSTIC. Its generator FAILED its own")
    print("held-out gate: worst out-of-sample TVD %.4f against a %.2f bar on "
          "%s," % (policy_gate["worst_tvd"], policy_gate["bar"],
                   policy_gate["worst_tvd_target"]))
    print("and scored prompts out of band: %s. It is shown to demonstrate that"
          % (", ".join(policy_gate["out_of_band"]) or "none"))
    print("even a mechanism-derived shape lands inside the identified set, not")
    print("to narrow that set.")
    print("\nscore-sensitivity mix over prompts (Finding 16, receipt %s)"
          % rec["id"][:8])
    tot = sum(mix.values())
    for name, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        print("  %-9s %.6f   share %.3f" % (name, w, w / tot))

    # The ledger's own reduction of every arm must come back out of this
    # instrument before any reweighting of it means anything.
    print("\n-- provenance check: reproduce the ledger 259.8 `weighted` column")
    chk_rows = []
    for arm, spec in ARMS.items():
        if not spec["recorded"]:
            continue
        got = sw.weighted(spec["na"], standing)
        chk_rows.append({"arm": arm, "recorded": spec["recorded"]["weighted"],
                         "chk": got,
                         "resid_pp": got - spec["recorded"]["weighted"]})
    worst = max(abs(r["resid_pp"]) for r in chk_rows)
    for r in sorted(chk_rows, key=lambda r: -abs(r["resid_pp"])):
        print("   %-16s recorded %+9.3f   chk %+9.3f   resid %+.4f pp"
              % (r["arm"], r["recorded"], r["chk"], r["resid_pp"]))
    print("   worst residual %.4f pp against a 0.005 pp bar -> %s"
          % (worst, "PASS" if worst <= 0.005 else "FAIL"))
    out["chk"] = {"rows": chk_rows, "worst_resid_pp": worst,
                  "bar_pp": 0.005, "pass": bool(worst <= 0.005)}

    for frame, rates in (("local", sw.ONE_GROUP_GBPS),
                         ("ranked", sw.RANKED_ONE_GROUP_GBPS)):
        vsets, vsets_pt = build(rec, rates, use_cost=True, policy=policy)
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
        fac = factor_rows(rec, vsets, vsets_pt, mix)
        out["frames"][frame] = {"points": pts, "arms": rows,
                                "weighted_to_round": fac}
        print("\n  %-16s %9s %19s %10s %10s %s"
              % ("arm", "standing", "published range", "move pp", "guaranteed",
                 "point est (maxent/gt1/gt2/policy)"))
        for r in sorted(rows, key=lambda r: -r["max_move_pp"]):
            print("  %-16s %+9.3f [%+8.3f,%+8.3f] %+9.3f/%+.3f %8.3f  "
                  "%+.3f/%+.3f/%+.3f/%+.3f%s%s"
                  % (r["arm"], r["standing_pct"], r["published_lo"],
                     r["published_hi"], r["move_lo_pp"], r["move_hi_pp"],
                     r["guaranteed_move_pp"], r["points"]["maxent"],
                     r["points"]["gt1"], r["points"]["gt2"],
                     r["points"]["policy"],
                     "  SIGN FLIPS" if not r["sign_identified"] else "",
                     "  IMMATERIAL" if r["immaterial_proven"] else ""))

        print("\n  the `weighted %% -> round %%` factor moves too, "
              "harness=local rates and round times, harness=ranked widths")
        print("    ledger scalar          %.4f   (the M=5 POINT of this same "
              "function)" % fac["ledger_scalar"])
        print("    M=5 point, rebuilt     %.4f   (%+.2f %% against the ledger "
              "scalar)" % (fac["m5_point"], fac["m5_point_vs_ledger_pct"]))
        print("    E106 realised widths   %.4f   (%.0f %% of that mass uses an "
              "extrapolated round time)"
              % (fac["e106_realised"], 100 * fac["e106_extrapolated_mass"]))
        for s in SHAPES:
            print("    published (%-6s)     %.4f   (%.0f %% extrapolated mass)"
                  % (s, fac["published"][s]["factor"],
                     100 * fac["published"][s]["extrapolated_mass"]))
        print("    identified set         [%.4f, %.4f]"
              % (fac["published_lo"], fac["published_hi"]))

    ranked = out["frames"]["ranked"]["arms"]
    e110 = [r for r in ranked if r["src"] == "E110"]
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
        "amended_kill_rule_pp": 0.05,
    }
    pm = out["primary_metric"]
    print("\ne114_max_abs_arm_reweight_pp = %.4f pp" % pm["value"])
    print("  shape ensemble spread %+.4f to %+.4f pp;  identified-set upper "
          "bound %.4f pp;  guaranteed lower bound %.4f pp"
          % (pm["shape_ensemble_lo"], pm["shape_ensemble_hi"],
             pm["identified_set_upper_bound"], pm["guaranteed_lower_bound"]))
    print("  the 0.05 pp kill rule is amended, not applied: the deliverable is "
          "the bound and the sign-invariant list, not one scalar")

    print("\n-- proven results per arm, harness=ranked rates ----------------")
    print("  %-16s %-18s %s" % ("arm", "verdict", "identified move band pp"))
    inv, imm, und = [], [], []
    for r in sorted(ranked, key=lambda r: r["arm"]):
        if r["immaterial_proven"]:
            verdict, bucket = "PROVEN immaterial", imm
        elif r["sign_identified"]:
            verdict, bucket = "SIGN INVARIANT", inv
        else:
            verdict, bucket = "sign undetermined", und
        bucket.append(r["arm"])
        print("  %-16s %-18s [%+.4f, %+.4f]"
              % (r["arm"], verdict, r["move_lo_pp"], r["move_hi_pp"]))
    out["verdicts"] = {"sign_invariant": inv, "proven_immaterial": imm,
                       "sign_undetermined": und}

    print("\n-- reconciliation against the advisor's own reweighting --------")
    print("  three named shapes only; `spread` is the advisor's max over those")
    print("  three, mine is the max over the whole identified set, so those two")
    print("  columns measure different things and are reported side by side.")
    print("  %-12s %8s %8s %9s %9s %9s %9s %9s"
          % ("arm", "chk", "d chk", "d maxent", "d gt1", "d gt2",
             "adv sprd", "my set"))
    recon_rows, worst_recon = [], 0.0
    for arm, adv in ADVISOR_F2.items():
        r = next(x for x in ranked if x["arm"] == arm)
        d = {"chk": r["standing_pct"] - adv["chk"],
             "maxent": r["points"]["maxent"] - adv["maxent"],
             "gt1": r["points"]["gt1"] - adv["gt1"],
             "gt2": r["points"]["gt2"] - adv["gt2"]}
        worst_recon = max(worst_recon, max(abs(v) for v in d.values()))
        recon_rows.append({"arm": arm, "advisor": adv, "mine": {
            "chk": r["standing_pct"], "maxent": r["points"]["maxent"],
            "gt1": r["points"]["gt1"], "gt2": r["points"]["gt2"],
            "max_move": r["max_move_pp"]}, "delta": d,
            "advisor_three_shape_spread_pp": adv["max_move"],
            "my_identified_set_max_move_pp": r["max_move_pp"]})
        print("  %-12s %8.3f %+8.3f %+9.3f %+9.3f %+9.3f %9.3f %9.3f"
              % (arm, r["standing_pct"], d["chk"], d["maxent"], d["gt1"],
                 d["gt2"], adv["max_move"], r["max_move_pp"]))
    print("  worst point-estimate disagreement %.4f pp against a 0.05 pp bar"
          " -> %s" % (worst_recon, "PASS" if worst_recon <= 0.05 else "FAIL"))
    widen = max(r["my_identified_set_max_move_pp"]
                / max(r["advisor_three_shape_spread_pp"], 1e-9)
                for r in recon_rows)
    print("  the identified set is up to %.1fx wider than the three-shape"
          " spread, which is why the set, not the spread, is the deliverable"
          % widen)
    out["advisor_reconciliation"] = {"rows": recon_rows,
                                     "worst_abs_pp": worst_recon,
                                     "worst_set_vs_spread_ratio": widen}

    print("\n-- the reweighting vector any arm owner can apply themselves --")
    for s in SHAPES:
        dw = {na: out["frames"]["ranked"]["points"][s][na] - standing[na]
              for na in sw.NA_CELLS}
        print("  dW (%-6s) %s   move_pp = sum_NA dW[NA] * arm_pct[NA]"
              % (s, "  ".join("NA%d %+0.4f" % (na, dw[na])
                              for na in sw.NA_CELLS)))
        out.setdefault("delta_weights", {})[s] = dw

    it5 = out["item5"] = collapse_arithmetic(rec, mix, policy)
    print("\n" + "=" * 78)
    print("ITEM 5 - is E100's M=5 collapse worth its register tax?")
    print("=" * 78)
    print("  share of candidate TIME at exactly verify width 5, harness=ranked")
    print("  %-9s %8s %8s %8s %8s | %s"
          % ("prompt", *SHAPES, "identified set"))
    for name, r in sorted(it5["per_prompt_m5_time_share"].items()):
        print("  %-9s %8.4f %8.4f %8.4f %8.4f | [%.4f, %.4f]"
              % (name, *[r["points"][s] for s in SHAPES], r["lo"], r["hi"]))
    ag = it5["published_weighted_share"]
    print("  %-9s %8.4f %8.4f %8.4f %8.4f | [%.4f, %.4f]  <- published weight"
          % ("COMBINED", *[ag["points"][s] for s in SHAPES], ag["lo"],
             ag["hi"]))
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
    rc = it5["price_reconciliation"]
    print("\n  the three prices are ONE coefficient: gain = %s, adv = %.5f"
          % (rc["gain_is"], rc["ranked_advantage_adv"]))
    for a, g in sorted(rc["sweep"].items()):
        tag = ""
        if abs(float(a) - rc["breakeven_a_local"]) < 5e-4:
            tag = "  <- BREAK-EVEN"
        elif float(a) == rc["e104_measured_a_local"]:
            tag = "  <- E104 F33 point (+1.9 %)"
        elif float(a) == COLLAPSE_PRICES["f32_route1"]["a_local"]:
            tag = "  <- F32 route 1 (-2.0 %), which is ALGEBRA not a measurement"
        print("    A_local %s -> ranked gain %+6.2f %%%s" % (a, g, tag))
    print("  E104's own measured range for A_local is [%.3f, %.3f] and it "
          "%s the break-even %.4f"
          % (*rc["e104_measured_a_local_range"],
             "CONTAINS" if rc["e104_range_contains_breakeven"] else "excludes",
             rc["breakeven_a_local"]))

    r2c = it5["route2_share_check"]
    print("\n  route 2 assumed an M=5 share of pre-E100 two-group ROUNDS of "
          "%.2f" % r2c["assumed"])
    print("    recovered here: %.4f/%.4f/%.4f/%.4f (%s), identified "
          "[%.4f, %.4f] -> assumption inside the set: %s"
          % (*[r2c["recovered_points"][s] for s in SHAPES], "/".join(SHAPES),
             r2c["recovered_lo"], r2c["recovered_hi"],
             r2c["assumption_inside_identified_set"]))

    print("\n  net = gain x share - tax, published weighting, %% of candidate "
          "time")
    print("  %-32s %6s %8s %8s %10s %s"
          % ("collapse price / tax", "kind", "gain", "tax", "net point",
             "net band"))
    for k, v in it5["nets"].items():
        print("  %-32s %6s %+8.2f %8.4f %+10.4f [%+.4f, %+.4f]"
              % (k, "round" if v["share_applied"] else "LEG", v["gain_pct"],
                 v["tax_pct"], v["net_point"], v["net_lo"], v["net_hi"]))
    print("  `LEG` rows already contain the share: route 2 MEASURED a leg "
          "effect and only inferred a per-round price by dividing by 0.24, so "
          "re-applying a share would double count it.")

    import os
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as h:
        json.dump(out, h, indent=1, sort_keys=True, default=str)
    print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
