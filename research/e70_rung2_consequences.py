#!/usr/bin/env python3
"""E70 rung 2 -- price each divergent, reachable dispatch site.

harness=ranked for every score projection here. No GPU runs. Every input is a
measured campaign number with its source quoted beside it, and both conversion
routes are printed so a disagreement between them is visible, not hidden.

Route A, the median pair. Convert the local saving with that term's own
transfer rate `tau`, then move it through the two prompts that set the
published median. Their legs are measured ranked values from 186(B), so this
route needs no leg ratio at all. A self-check reproduces the published
3.23250848 of submission ca9251b8 from the same table.

Route B, the adopted direct form. Convert with the width-dependent round ratio
`R(M)`, then

    delta_score_pct = 100 * delta_ranked_ms * rounds_at_M / candidate_leg_ms

`R(M)` and `M` are reported beside every converted number, because an
unlabelled conversion is invalid. The table comes from
`research/e70-transfer-constant.json`, which derives it from the same measured
reconstruction that calibrates the candidate depth-0 round.

The flat `R = 2.1383` of campaign-ledger.md:11045 is NOT refuted. An earlier
revision of this file said it was; that claim is retracted in
`research/e70_transfer_constant_provenance.py`. `R = 2.1383` is the correct
value at depth 0. It is superseded only because `R` rises with verify width.

usage:
  python3 research/e70_rung2_consequences.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

# --- measured inputs -------------------------------------------------------

TRANSFER = pathlib.Path("research/e70-transfer-constant.json")

# E65 (research/e65-results.md:114-124). Timed candidate leg, --local-iterate.
LOCAL_LEG_S = 17.349          # mean of the two reported legs 17.344 / 17.354
LOCAL_PREFILL_S = 3.9938      # ledger 186(C):10452, the `begin` segment

# E65 Q3 (research/e65-results.md:59-66). The round-1 `d_submit2` excess, which
# is the 511-row MTP head history flush.
HEAD_PRIME_MS = (29.52 + 28.91) / 2

# Ledger 186(C) (campaign-ledger.md:10452-10453). The measured local->ranked
# transfer rate of the prefill section, which is 84 % nax GEMM at rank.
TAU_PREFILL = 3.9938 / 0.5269           # 7.5798, compute-bound, nax-accelerated

# Ledger 186(B) (campaign-ledger.md:10425-10434). The per-prompt ranked receipt
# of submission ca9251b8: candidate leg and the three score factors. The
# published score is the median of build x spec x dilution over these rows.
RANKED_PROMPTS = {
    #            leg_ms   build    spec     dilution
    "plutarch": (15517, 1.2489, 1.0384, 0.96606),
    "drama": (10126, 1.2468, 1.6216, 0.94799),
    "travel": (8903, 1.2468, 1.8583, 0.94085),
    "beagle": (6233, 1.2494, 2.7277, 0.91552),
    "medicine": (5821, 1.2508, 2.9402, 0.90953),
    "republic": (5726, 1.2485, 2.9937, 0.90803),
    "essays": (5764, 1.2464, 2.9723, 0.90863),
    "botany": (5673, 1.2484, 3.0245, 0.90724),
}
PUBLISHED_SCORE = 3.23250848  # submission ca9251b8

# Assignment baseline block, PR #73.
SCORE_SD_PCT = 0.756
LEG_SD_PCT = 1.092
DEFICIT_PCT = 0.61

# Modelled ranked verify-width shares, item 200(D) (campaign-ledger.md:14830).
# These come from the e53 mixture FIT (research/e53_width_mixture.py), and item
# 184(D):10219 proved the ranked histogram is unidentifiable from public data.
# They are a model output, not a measurement.
WIDTH_SHARES_PCT = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}

# Scored geometry, weights/config.json.
FULL_ATTENTION_LAYERS = 16
HEADS = 24
HEAD_DIM = 256
SEED_TOKENS = 512

# E65 prefill roofline (research/e65-results.md:130-134).
PREFILL_TFLOP_TOTAL = 24.99
PREFILL_TFLOP_PER_S = 6.25


def load_width_table() -> dict[str, dict]:
    if not TRANSFER.exists():
        raise SystemExit(
            f"{TRANSFER} missing. Run "
            "python3 research/e70_transfer_constant_provenance.py first.")
    rows = json.loads(TRANSFER.read_text())["R_of_M"]["table"]
    return {row["prompt"]: row for row in rows}


def sdpa_fallback_flop() -> float:
    """FLOPs of the dense GEMM pair MLX runs instead of fused attention.

    `use_fallback` is true at qL = 512 because head_dim 256 is outside
    sdpa_full's {64, 80, 128} and 512 * gqa exceeds sdpa_vector's cap, so
    mlx/fast.cpp composes `matmul(q, swapaxes(k, -1, -2))` and
    `matmul(scores, v)`. The composed form does NOT exploit causality: it
    builds the full qL x kL score matrix and then masks it.
    """
    qk = 2 * SEED_TOKENS * SEED_TOKENS * HEAD_DIM * HEADS
    pv = 2 * SEED_TOKENS * HEAD_DIM * SEED_TOKENS * HEADS
    return (qk + pv) * FULL_ATTENTION_LAYERS


def raw_ratios() -> dict[str, float]:
    return {
        name: build * spec * dilution
        for name, (_, build, spec, dilution) in RANKED_PROMPTS.items()
    }


def median_pair() -> tuple[list[str], float]:
    """The two prompts whose raw ratios the published median averages."""
    ordered = sorted(raw_ratios().items(), key=lambda item: item[1])
    pair = [ordered[3][0], ordered[4][0]]
    return pair, (ordered[3][1] + ordered[4][1]) / 2.0


def score_sensitivity_per_ms() -> float:
    """Score change for one millisecond removed from every candidate leg.

    `raw_p = ranked_serial_leg / candidate_leg_p`, so a saving of `d` on
    candidate leg `p` lifts `raw_p` by `raw_p * d / leg_p`. The published value
    is the mean of the two middle raw ratios, so only those two prompts carry
    the saving into the score.
    """
    pair, _ = median_pair()
    raws = raw_ratios()
    return 0.5 * sum(raws[p] / RANKED_PROMPTS[p][0] for p in pair)


def route_a(delta_local_ms: float, tau: float) -> dict:
    """Median-pair route: term-specific tau, measured ranked median-pair legs."""
    delta_ranked_ms = delta_local_ms / tau
    _, published = median_pair()
    return {
        "delta_local_ms": delta_local_ms,
        "tau": tau,
        "delta_ranked_ms": delta_ranked_ms,
        "score_pct": 100.0 * delta_ranked_ms * score_sensitivity_per_ms()
                     / published,
    }


def route_b(delta_local_ms: float, rounds_at_m: int,
            widths: dict[str, dict], tau_override: float | None = None) -> dict:
    """Adopted direct form, evaluated on each median-pair prompt.

    `tau_override` replaces R(M) with a stated transfer rate. It exists so the
    two routes can be compared with the transfer constant held equal, which
    isolates how much of any gap is the leg arithmetic and how much is the
    conversion.
    """
    pair, _ = median_pair()
    per_prompt = {}
    for name in pair:
        w = widths[name]
        divisor = tau_override if tau_override is not None else w["R_of_M"]
        delta_ranked_ms = delta_local_ms / divisor
        per_prompt[name] = {
            "verify_width_M": w["verify_width_M"],
            "R_of_M": w["R_of_M"],
            "divisor_used": divisor,
            "rounds_at_M": rounds_at_m,
            "candidate_leg_ms": w["ranked_candidate_leg_ms"],
            "delta_ranked_ms": delta_ranked_ms,
            "score_pct": 100.0 * delta_ranked_ms * rounds_at_m
                         / w["ranked_candidate_leg_ms"],
        }
    return {
        "per_prompt": per_prompt,
        "score_pct": statistics.fmean(
            p["score_pct"] for p in per_prompt.values()),
    }


def price(delta_local_ms: float, tau: float, rounds_at_m: int,
          widths: dict[str, dict], round_resident: bool) -> dict:
    """Both routes for one saving, plus the naive and noise comparisons."""
    a = route_a(delta_local_ms, tau)
    pair_pct = a["score_pct"]

    out = {
        "delta_local_ms": delta_local_ms,
        "tau": tau,
        "rounds_at_M": rounds_at_m,
        "delta_ranked_ms": a["delta_ranked_ms"],
        "route_a_median_pair_score_pct": pair_pct,
        "score_sensitivity_pct_per_ranked_ms":
            100.0 * score_sensitivity_per_ms() / median_pair()[1],
    }

    # Hold the transfer constant equal so the two leg models can be compared.
    matched = route_b(delta_local_ms, rounds_at_m, widths, tau_override=tau)
    out["route_b_direct_form_at_same_tau"] = matched
    out["leg_model_agreement_pct"] = (
        100.0 * (matched["score_pct"] - pair_pct) / pair_pct)

    if round_resident:
        b = route_b(delta_local_ms, rounds_at_m, widths)
        out["route_b_direct_form_at_R_of_M"] = b
        out["route_b_over_route_a"] = b["score_pct"] / pair_pct
    else:
        out["route_b_direct_form_at_R_of_M"] = None
        out["route_b_not_applicable"] = (
            "the cost is inside the seed prefill, not inside a decode round, "
            "so R(M) is the wrong category of constant. Prefill has its own "
            "measured transfer rate, tau_prefill = 7.5798 (186(C)).")

    # What the same saving looks like if you forget that the section is
    # nax-accelerated at rank and transfers at tau instead of 1.
    out["naive_no_tau_score_pct"] = (
        delta_local_ms / (LOCAL_LEG_S * 1000.0) * 100.0)
    out["overstatement_factor"] = (
        out["naive_no_tau_score_pct"] / pair_pct if pair_pct else float("nan"))
    out["sd_of_published_score"] = pair_pct / SCORE_SD_PCT
    out["fraction_of_deficit"] = pair_pct / DEFICIT_PCT
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args()

    widths = load_width_table()
    fallback_flop = sdpa_fallback_flop()
    fallback_ms = fallback_flop / (PREFILL_TFLOP_PER_S * 1e12) * 1000.0
    pair, reconstructed_score = median_pair()
    sens_pct = 100.0 * score_sensitivity_per_ms() / reconstructed_score

    r_values = {n: widths[n]["R_of_M"] for n in RANKED_PROMPTS}
    r_lo_name = min(r_values, key=r_values.get)
    r_hi_name = max(r_values, key=r_values.get)

    report: dict = {
        "harness": "ranked",
        "inputs": {
            "local_leg_s": LOCAL_LEG_S,
            "local_prefill_s": LOCAL_PREFILL_S,
            "head_prime_ms": HEAD_PRIME_MS,
            "tau_prefill": TAU_PREFILL,
            "R_of_M_source": str(TRANSFER),
            "R_of_M_median_pair": {
                n: {"M": widths[n]["verify_width_M"],
                    "R_of_M": widths[n]["R_of_M"],
                    "rounds": widths[n]["rounds_at_M"],
                    "candidate_leg_ms": widths[n]["ranked_candidate_leg_ms"]}
                for n in pair},
            "score_sd_pct": SCORE_SD_PCT,
            "leg_sd_pct": LEG_SD_PCT,
            "deficit_pct": DEFICIT_PCT,
        },
        "pricing_rule": {
            "delta_ranked_ms": "delta_local_ms / R(M)",
            "delta_score_pct":
                "100 * delta_ranked_ms * rounds_at_M / candidate_leg_ms",
            "reporting_requirement":
                "report R(M) and M beside every converted number",
            "rounds_at_M_for_these_sites": 1,
            "why_one": (
                "both divergent reachable sites are once-per-leg costs. The "
                "511-row head prime fires on the first drafting round only, "
                "and the SDPA fallback fires inside the single seed prefill. "
                "Neither is a per-round cost, so rounds_at_M = 1."),
        },
        "median_pair_model": {
            "raw_ratios": raw_ratios(),
            "median_pair_prompts": pair,
            "median_pair_legs_ms": [RANKED_PROMPTS[p][0] for p in pair],
            "reconstructed_published_score": reconstructed_score,
            "actual_published_score": PUBLISHED_SCORE,
            "self_check_relative_error": abs(
                reconstructed_score - PUBLISHED_SCORE) / PUBLISHED_SCORE,
            "score_sensitivity_pct_per_ranked_ms": sens_pct,
            "why": (
                "The two middle raw ratios set the published median, so a "
                "fixed per-leg saving reaches the score only through those "
                "two prompts. Their measured ranked legs replace any leg "
                "ratio."),
        },
        "sdpa_fallback": {
            "gflop_per_seed": fallback_flop / 1e9,
            "share_of_prefill_flop_pct": (
                fallback_flop / 1e12 / PREFILL_TFLOP_TOTAL * 100.0),
            "upper_bound_ms_at_measured_prefill_rate": fallback_ms,
            "e65_roofline_attention_tflop": 0.052,
            "note": (
                "E65's roofline recorded 0.052 TFLOP of prefill attention, "
                "which is half of the dispatched work: the composed fallback "
                "runs both matmuls at full qL x kL and masks afterwards."),
        },
        "sites": {},
    }

    head_prime = price(
        HEAD_PRIME_MS, TAU_PREFILL, 1, widths, round_resident=True)
    report["sites"]["S4_decode_head_prime"] = {
        "site": "quantized.cpp:697 qmm nax gate, plus matmul.cpp:915 for the "
                "bf16 island patch on the same 511 rows",
        "what_it_costs_locally": "the E65 round-1 d_submit2 excess",
        "steerable": False,
        "steerable_reason": (
            "quantized.cpp and matmul.cpp are not in benchmark.json "
            "editablePaths. Only the row count is editable, in "
            "Qwen36MTPBlockSession, and that is the queued E65 follow-up (a)."),
        "arithmetic": head_prime,
        # The head prime's transfer rate is the whole question, so bound it
        # from both ends instead of asserting one value.
        "transfer_band": {
            "adopted_tau_prefill": {
                "divisor": TAU_PREFILL,
                "delta_ranked_ms": HEAD_PRIME_MS / TAU_PREFILL,
                "score_pct": route_a(HEAD_PRIME_MS, TAU_PREFILL)["score_pct"],
            },
            "floor_R_of_M_smallest": {
                "prompt": r_lo_name,
                "M": widths[r_lo_name]["verify_width_M"],
                "divisor": r_values[r_lo_name],
                "delta_ranked_ms": HEAD_PRIME_MS / r_values[r_lo_name],
                "score_pct": route_a(
                    HEAD_PRIME_MS, r_values[r_lo_name])["score_pct"],
            },
            "floor_R_of_M_largest": {
                "prompt": r_hi_name,
                "M": widths[r_hi_name]["verify_width_M"],
                "divisor": r_values[r_hi_name],
                "delta_ranked_ms": HEAD_PRIME_MS / r_values[r_hi_name],
                "score_pct": route_a(
                    HEAD_PRIME_MS, r_values[r_hi_name])["score_pct"],
            },
            "floor_R_depth0": {
                "M": 1.0,
                "divisor": widths["(depth-0 control)"]["R_of_M"],
                "delta_ranked_ms":
                    HEAD_PRIME_MS / widths["(depth-0 control)"]["R_of_M"],
                "score_pct": route_a(
                    HEAD_PRIME_MS,
                    widths["(depth-0 control)"]["R_of_M"])["score_pct"],
            },
            "which_applies": (
                "tau_prefill. This audit proves the 511-row prime runs "
                "affine_qmm_t_nax through quantized.cpp:697 and "
                "steel_gemm_fused_nax through matmul.cpp:915 at rank, which "
                "are exactly the families that give prefill its 7.58x. The "
                "prime is ~100 % GEMM at M = 511 against prefill's 84 % at "
                "M = 512, so if anything it transfers better than 7.58x."),
            "why_it_matters": (
                "At an R(M) decode-round rate the prime would be worth about "
                "0.20-0.23 % of score and E65 follow-up (a) would be a live "
                "target. Rung 1 is what excludes that branch."),
        },
    }
    report["sites"]["S9_prefill_sdpa_fallback"] = {
        "site": "matmul.cpp:915 family selector, with matmul.cpp:176 vs :373 "
                "tile parameters",
        "what_it_costs_locally": (
            "the 32 dense bf16 GEMMs MLX composes because head_dim 256 has no "
            "fused attention kernel"),
        "steerable": False,
        "steerable_reason": (
            "mlx/fast.cpp and matmul.cpp are not editable. AttentionUtils.swift "
            "IS editable, but no chunking of a 512-row prefill query reaches a "
            "fused kernel: sdpa_full excludes head_dim 256 at every width."),
        "arithmetic": price(
            fallback_ms, TAU_PREFILL, 1, widths, round_resident=False),
    }

    report["width_shares"] = {
        "shares_pct": WIDTH_SHARES_PCT,
        "provenance": "MODELLED (e53 mixture fit, ledger 200(D):14830); item "
                      "184(D):10219 proved the ranked histogram is "
                      "unidentifiable from public data",
        "applies_to_any_divergent_site": False,
        "why": (
            "Every divergent site fires either once per leg (the 511-row head "
            "prime) or inside the seed prefill. Neither is inside the "
            "per-round, width-dependent verify work, so the width mixture "
            "cannot change any number above. The one place the mixture would "
            "matter is the M = 10 qmv -> qmm cliff, which no scored width "
            "reaches today."),
        "distance_to_the_cliff": (
            "segmentedVerifyDepthCap 8 bounds M at 9. The widest scored width "
            f"M = 9 carries {WIDTH_SHARES_PCT[9]} % of modelled ranked verify "
            "mass and sits one row below the cliff."),
    }

    print("E70 rung 2 -- score consequences of the divergent sites   harness=ranked")
    print()
    mp = report["median_pair_model"]
    print(f"median pair: {mp['median_pair_prompts']} legs "
          f"{mp['median_pair_legs_ms']} ms")
    print(f"  self-check: reconstructed {mp['reconstructed_published_score']:.8f}"
          f" vs published {PUBLISHED_SCORE:.8f}"
          f"  (relative error {mp['self_check_relative_error']:.2e})")
    print(f"  score sensitivity "
          f"{mp['score_sensitivity_pct_per_ranked_ms']:.6f} % per ranked ms")
    print()
    print("R(M) at the median pair (report M and R(M) with every conversion):")
    for name in pair:
        w = widths[name]
        print(f"  {name:<9} M = {w['verify_width_M']:.4f}"
              f"  R(M) = {w['R_of_M']:.4f}"
              f"  rounds = {w['rounds_at_M']}"
              f"  leg = {w['ranked_candidate_leg_ms']:.1f} ms")
    print()
    fb = report["sdpa_fallback"]
    print(f"prefill SDPA fallback: {fb['gflop_per_seed']:.1f} GFLOP per seed"
          f" = {fb['share_of_prefill_flop_pct']:.3f} % of prefill FLOPs"
          f" <= {fb['upper_bound_ms_at_measured_prefill_rate']:.1f} ms local")
    print()
    for name, site in report["sites"].items():
        a = site["arithmetic"]
        print(f"{name}")
        print(f"  local saving if removed entirely : {a['delta_local_ms']:.2f} ms")
        print(f"  adopted transfer rate            : {a['tau']:.4f}"
              f"  (rounds_at_M = {a['rounds_at_M']})")
        print(f"  ranked saving                    : {a['delta_ranked_ms']:.3f} ms")
        print(f"  route A, median pair             : {a['route_a_median_pair_score_pct']:.4f} % of score")
        matched = a["route_b_direct_form_at_same_tau"]
        print(f"  route B, direct form, same tau   : {matched['score_pct']:.4f} % of score"
              f"  ({a['leg_model_agreement_pct']:+.2f} % vs route A)")
        b = a["route_b_direct_form_at_R_of_M"]
        if b is not None:
            print(f"  route B, direct form, R(M)       : {b['score_pct']:.4f} % of score"
                  f"  ({a['route_b_over_route_a']:.2f}x route A)")
            for pname, pv in b["per_prompt"].items():
                print(f"      {pname:<9} M = {pv['verify_width_M']:.4f}"
                      f"  R(M) = {pv['R_of_M']:.4f}"
                      f"  -> {pv['delta_ranked_ms']:.3f} ms"
                      f"  = {pv['score_pct']:.4f} %")
        else:
            print(f"  route B, direct form, R(M)       : not applicable"
                  f"  -- {a['route_b_not_applicable'].split('.')[0]}")
        if "transfer_band" in site:
            tb = site["transfer_band"]
            print(f"  transfer band                    : "
                  f"{tb['adopted_tau_prefill']['score_pct']:.4f} % at "
                  f"tau_prefill {tb['adopted_tau_prefill']['divisor']:.4f}"
                  f"  to {tb['floor_R_of_M_smallest']['score_pct']:.4f} % at "
                  f"R(M) {tb['floor_R_of_M_smallest']['divisor']:.4f}"
                  f" (M = {tb['floor_R_of_M_smallest']['M']:.2f},"
                  f" {tb['floor_R_of_M_smallest']['prompt']})")
            print(f"                                     "
                  f"narrowest floor {tb['floor_R_of_M_largest']['score_pct']:.4f} % at "
                  f"R(M) {tb['floor_R_of_M_largest']['divisor']:.4f}"
                  f" (M = {tb['floor_R_of_M_largest']['M']:.2f},"
                  f" {tb['floor_R_of_M_largest']['prompt']});"
                  f" depth-0 floor {tb['floor_R_depth0']['score_pct']:.4f} % at "
                  f"R(1) {tb['floor_R_depth0']['divisor']:.4f}")
        print(f"  naive, ignoring the transfer     : {a['naive_no_tau_score_pct']:.4f} %"
              f"  ({a['overstatement_factor']:.2f}x too high)")
        print(f"  vs published-score sd {SCORE_SD_PCT} %      : {a['sd_of_published_score']:.3f} sd")
        print(f"  vs our {DEFICIT_PCT} % deficit             : {a['fraction_of_deficit']*100:.1f} % of it")
        print(f"  steerable by editable code       : {site['steerable']}")
        print()

    total = sum(s["arithmetic"]["route_a_median_pair_score_pct"]
                for s in report["sites"].values())
    report["total_if_every_divergent_site_cost_went_to_zero_pct"] = total
    agreement = [s["arithmetic"]["leg_model_agreement_pct"]
                 for s in report["sites"].values()]
    report["route_agreement"] = {
        "leg_model_max_abs_disagreement_pct": max(abs(v) for v in agreement),
        "verdict": (
            "the two routes agree on the leg arithmetic to within "
            f"{max(abs(v) for v in agreement):.2f} % when the transfer "
            "constant is held equal. The whole remaining gap is the choice of "
            "transfer constant, and rung 1 is the evidence that decides it."),
    }
    print(f"UPPER BOUND: if BOTH divergent reachable sites cost zero at rank, "
          f"the score moves {total:.4f} %.")
    print(f"That is {total / SCORE_SD_PCT:.3f} sd of one published score and "
          f"{total / DEFICIT_PCT * 100:.1f} % of the deficit.")
    print()
    print("ROUTE AGREEMENT: " + report["route_agreement"]["verdict"])
    print()
    print("The modelled width shares do not enter any line above; see "
          "report['width_shares'] for why.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
