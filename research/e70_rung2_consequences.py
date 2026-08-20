#!/usr/bin/env python3
"""E70 rung 2 -- price each divergent, reachable dispatch site.

harness=ranked for every score projection here. No GPU runs. Every input is a
measured campaign number with its source quoted beside it, and the two
independent conversion routes are both printed so a disagreement between them
is visible instead of hidden.

Route A, direct: reconstruct the ranked leg from the two measured transfer
rates and take `delta_ranked / ranked_leg`.

Route B, the ledger's standing rule (campaign-ledger.md:10519-10520): express
the saving as a fraction of ROUND cost, multiply by `R / tau`, then multiply by
the median-pair dilution 0.9125. The rule takes a round fraction, not a leg
fraction; feeding it a leg fraction charges the dilution twice.

usage:
  python3 research/e70_rung2_consequences.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json

# --- measured inputs -------------------------------------------------------

# E65 (research/e65-results.md:114-124). Timed candidate leg, --local-iterate.
LOCAL_LEG_S = 17.349          # mean of the two reported legs 17.344 / 17.354
LOCAL_PREFILL_S = 3.9938      # ledger 186(C):10452, the `begin` segment

# E65 Q3 (research/e65-results.md:59-66). The round-1 `d_submit2` excess, which
# is the 511-row MTP head history flush.
HEAD_PRIME_MS = (29.52 + 28.91) / 2

# Ledger 186(C) (campaign-ledger.md:10452-10453). The two measured local->ranked
# transfer rates.
TAU_PREFILL = 3.9938 / 0.5269           # 7.58, compute-bound, nax-accelerated
R_ROUND = 65.009 / 30.402               # 2.1383, the depth-0 decode round

# Ledger 186(B/F) (campaign-ledger.md:10442, :10519-10520).
DILUTION = 0.9125

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


def ranked_leg_s() -> float:
    local_rounds = LOCAL_LEG_S - LOCAL_PREFILL_S
    return local_rounds / R_ROUND + LOCAL_PREFILL_S / TAU_PREFILL


def price(delta_local_ms: float, tau: float, in_rounds: bool) -> dict:
    """Both conversion routes for one saving."""
    leg = ranked_leg_s()
    delta_ranked_ms = delta_local_ms / tau
    direct_pct = delta_ranked_ms / (leg * 1000.0) * 100.0

    out = {
        "delta_local_ms": delta_local_ms,
        "tau": tau,
        "delta_ranked_ms": delta_ranked_ms,
        "route_a_direct_score_pct": direct_pct,
        "ranked_leg_s": leg,
    }
    if in_rounds:
        local_rounds_ms = (LOCAL_LEG_S - LOCAL_PREFILL_S) * 1000.0
        round_fraction_pct = delta_local_ms / local_rounds_ms * 100.0
        out["route_b_ledger_score_pct"] = (
            round_fraction_pct * (R_ROUND / tau) * DILUTION)
        out["local_round_fraction_pct"] = round_fraction_pct
    # What the same saving looks like if you forget that the section is
    # nax-accelerated at rank and transfers at tau instead of 1.
    out["naive_no_tau_score_pct"] = (
        delta_local_ms / (LOCAL_LEG_S * 1000.0) * 100.0)
    out["overstatement_factor"] = (
        out["naive_no_tau_score_pct"] / direct_pct if direct_pct else float("nan"))
    out["sd_of_published_score"] = direct_pct / SCORE_SD_PCT
    out["fraction_of_deficit"] = direct_pct / DEFICIT_PCT
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args()

    leg = ranked_leg_s()
    fallback_flop = sdpa_fallback_flop()
    fallback_ms = fallback_flop / (PREFILL_TFLOP_PER_S * 1e12) * 1000.0

    report: dict = {
        "harness": "ranked",
        "inputs": {
            "local_leg_s": LOCAL_LEG_S,
            "local_prefill_s": LOCAL_PREFILL_S,
            "head_prime_ms": HEAD_PRIME_MS,
            "tau_prefill": TAU_PREFILL,
            "r_round": R_ROUND,
            "dilution": DILUTION,
            "score_sd_pct": SCORE_SD_PCT,
            "leg_sd_pct": LEG_SD_PCT,
            "deficit_pct": DEFICIT_PCT,
        },
        "reconstructed_ranked_leg": {
            "ranked_prefill_s": LOCAL_PREFILL_S / TAU_PREFILL,
            "ranked_rounds_s": (LOCAL_LEG_S - LOCAL_PREFILL_S) / R_ROUND,
            "ranked_leg_s": leg,
            "ranked_round_share": (
                (LOCAL_LEG_S - LOCAL_PREFILL_S) / R_ROUND / leg),
            "ledger_round_share": DILUTION,
        },
        "sdpa_fallback": {
            "gflop_per_seed": fallback_flop / 1e9,
            "share_of_prefill_flop_pct": (
                fallback_flop / 1e12 / PREFILL_TFLOP_TOTAL * 100.0),
            "upper_bound_ms_at_measured_prefill_rate": fallback_ms,
            "e65_roofline_attention_tflop": 0.052,
            "note": (
                "E65's roofline recorded 0.052 TFLOP of prefill attention, which "
                "is half of the dispatched work: the composed fallback runs both "
                "matmuls at full qL x kL and masks afterwards."),
        },
        "sites": {},
    }

    report["sites"]["S4_decode_head_prime"] = {
        "site": "quantized.cpp:697 qmm nax gate, plus matmul.cpp:915 for the "
                "bf16 island patch on the same 511 rows",
        "what_it_costs_locally": "the E65 round-1 d_submit2 excess",
        "steerable": False,
        "steerable_reason": (
            "quantized.cpp and matmul.cpp are not in benchmark.json "
            "editablePaths. Only the row count is editable, in "
            "Qwen36MTPBlockSession, and that is the queued E65 follow-up (a)."),
        "arithmetic": price(HEAD_PRIME_MS, TAU_PREFILL, in_rounds=True),
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
        "arithmetic": price(fallback_ms, TAU_PREFILL, in_rounds=False),
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
    rl = report["reconstructed_ranked_leg"]
    print(f"reconstructed ranked leg: prefill {rl['ranked_prefill_s']*1000:.1f} ms"
          f" + rounds {rl['ranked_rounds_s']*1000:.1f} ms"
          f" = {rl['ranked_leg_s']*1000:.1f} ms")
    print(f"  ranked round share {rl['ranked_round_share']:.4f}"
          f"  vs the ledger's measured {DILUTION}")
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
        print(f"  transfer rate tau                : {a['tau']:.4f}")
        print(f"  ranked saving                    : {a['delta_ranked_ms']:.3f} ms")
        print(f"  route A, direct                  : {a['route_a_direct_score_pct']:.4f} % of score")
        if "route_b_ledger_score_pct" in a:
            print(f"  route B, ledger standing rule    : {a['route_b_ledger_score_pct']:.4f} % of score")
        print(f"  naive, ignoring tau              : {a['naive_no_tau_score_pct']:.4f} %"
              f"  ({a['overstatement_factor']:.2f}x too high)")
        print(f"  vs published-score sd {SCORE_SD_PCT} %      : {a['sd_of_published_score']:.3f} sd")
        print(f"  vs our {DEFICIT_PCT} % deficit             : {a['fraction_of_deficit']*100:.1f} % of it")
        print(f"  steerable by editable code       : {site['steerable']}")
        print()

    total = sum(s["arithmetic"]["route_a_direct_score_pct"]
                for s in report["sites"].values())
    report["total_if_every_divergent_site_cost_went_to_zero_pct"] = total
    print(f"UPPER BOUND: if BOTH divergent reachable sites cost zero at rank, "
          f"the score moves {total:.4f} %.")
    print(f"That is {total / SCORE_SD_PCT:.3f} sd of one published score and "
          f"{total / DEFICIT_PCT * 100:.1f} % of the deficit.")
    print()
    print("The modelled width shares do not enter any line above; see "
          "report['width_shares'] for why.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
