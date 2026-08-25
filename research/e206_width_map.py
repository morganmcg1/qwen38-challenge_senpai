#!/usr/bin/env python3
"""E206: cap-7 / cap-8 served-width censuses, m=9 work map, G=3 exposure.

Counts and desk arithmetic only. No timing contrast between caps or surfaces
(RULE 79 / FINDING 521). Every number is tagged harness=local.

Inputs:
  - E202 raw per-round traces (cap-7 surface), untracked, this host:
        research/out/e202/session-submit-abba1/<leg>/trace.txt
  - E199 cap-8 per-round W&B tables, vendored under research/e206-inputs/.

Output: research/e206-width-work-map.json
"""

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent
E202_SESSION = ROOT / "out" / "e202" / "session-submit-abba1"
INPUTS = ROOT / "e206-inputs"
OUT = ROOT / "e206-width-work-map.json"

# Fused routed QMV cells: (name, k, n, invocations per round).
CELLS = [
    ("mlp.gate_up", 5120, 34816, 64),
    ("mlp.down", 17408, 5120, 64),
    ("gdn.in_proj", 5120, 16480, 48),
    ("gdn.out_proj", 6144, 5120, 48),
    ("fa.qkv", 5120, 14336, 16),
    ("fa.o_proj", 6144, 5120, 16),
    ("lm_head", 5120, 248320, 1),
]
FAMILY = {
    "mlp.gate_up": "mlp",
    "mlp.down": "mlp",
    "gdn.in_proj": "gdn",
    "gdn.out_proj": "gdn",
    "fa.qkv": "fa",
    "fa.o_proj": "fa",
    "lm_head": "lm_head",
}
# affine 4-bit, group 64: 0.5 B/element payload + one bf16 scale and one bf16
# bias per 64 elements.
BYTES_PER_ELEMENT = 0.5 + 4.0 / 64.0

# Shipped staged QMV width plan, Qwen35.swift:1582.
STAGED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
# E195 selective plan, Qwen35.swift:1641: single pass at m == 6 only,
# and not for mlp.down / unlisted.
SINGLEPASS_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3}

# FINDING 484 local step law, harness=local, M4 Pro, E182 (research/e182-report.json).
STEP_LAW = {"intercept_ms": 18.92, "per_row_ms": 6.51, "per_group_ms": 36.03}

# E182 unperturbed legs, harness=local, pinned depth, full acceptance.
E182_ROUND_MS = {1: 65.297, 4: 77.311, 5: 90.486, 6: 126.081, 7: 137.362,
                 8: 145.774, 9: 185.826}

# E182 band-instrumented legs: the m=8 -> m=9 step, i.e. the G=2 -> G=3 pass.
E182_G3_STEP_BY_BAND_MS = {
    "gdn_mlp": 19.305,
    "gdn_mixer": 8.359,
    "fa_mlp": 6.431,
    "fa_mixer": 2.648,
    "residual_lm_head_and_host": 3.637,
}

# E186 isolated probes (W&B fm5fnfmd, table e186/fits): step_m9_us per cell,
# scaled by invocations per round. The isolated per-cell sum over-explains the
# layer-probe reconstruction by 8.7%, so this ranks cells, it does not price
# them.
E186_STEP_M9_US = {
    "mlp.gate_up": 274.695,
    "mlp.down": 142.635,
    "gdn.in_proj": 180.254,
    "gdn.out_proj": 45.933,
    "fa.qkv": 189.796,
    "fa.o_proj": 47.198,
    "lm_head": 1943.319,
}
E186_LAYER_PROBE_G3_STEP_MS = 39.827
E186_CELL_SUM_OVEREXPLAIN = 0.087

# Local -> ranked transfer of one weight-pass step. FINDING 484's "two thirds
# hidden" came from the FINDING 456 quadratic that FINDING 504 falsified at
# 5.60 sigma; FINDING 505 measured the ranked step directly.
TRANSFER = {
    # E186 local round shape, the denominator E197 uses for transfer_ratio.
    "local_dR6_e186_ms": 37.6726,
    "local_dR6_e182_ms": 35.596,
    "ranked_dR6_measured_ms": 8.940,
    "ranked_dR6_measured_1sigma_ms": [8.345, 9.544],
    "ranked_dR6_e197_refit_ms": [7.93, 8.26],
    "ranked_dR9_e197_smooth_step_total_ms": 1.962,
    "ranked_row_term_ms": 1.339,
    "local_R1_ms": 65.297,
    "ranked_R1_ms": 30.2519,
    "falsified_quadratic_d56_ms": 5.394,
    "falsified_quadratic_d89_ms": 8.848,
}

# E201 / FINDING 523 linearized score engine on the cap-8 chain.
SCORE_ENGINE = {
    "points_per_ms_of_transferred_dR9_excess": 0.019745,
    "flat_cap8_chain_score": 3.829386,
    "paid_cap7_receipt_a": 3.70784519415395,
    "crown": 3.7291100105909,
}


def groups(m: int, ipg: int) -> int:
    return -(-m // ipg)


def shipped_groups(m: int, cell: str) -> int:
    if m == 6 and cell != "mlp.down":
        return groups(m, SINGLEPASS_IPG[m])
    return groups(m, STAGED_IPG[m])


def parse_e202_traces():
    """Cap-7 all-rounds served-width census from the raw E202 traces."""
    if not E202_SESSION.is_dir():
        return None
    field = re.compile(r"(\w+)=([^\s]+)")
    legs = {}
    for leg_dir in sorted(E202_SESSION.iterdir()):
        trace = leg_dir / "trace.txt"
        if not trace.is_file():
            continue
        census, accepted, proposed, rounds = {}, {}, {}, 0
        for line in trace.read_text().splitlines():
            if not line.startswith("mtp-trace: round="):
                continue
            rec = dict(field.findall(line))
            if rec.get("serial_body") == "1":
                continue
            d, acc = int(rec["d"]), int(rec["acc"])
            width = d + 1
            census[width] = census.get(width, 0) + 1
            proposed[width] = proposed.get(width, 0) + d
            accepted[width] = accepted.get(width, 0) + acc
            rounds += 1
        legs[leg_dir.name] = {
            "rounds": rounds,
            "census_by_served_width": census,
            "drafts_proposed_by_width": proposed,
            "drafts_accepted_by_width": accepted,
        }
    return legs


def load_cap8_tables():
    """Cap-8 per-round tables (round, drafts, rows, block_request_seconds)."""
    runs = {}
    for path in sorted(INPUTS.glob("cap8-per-round-*.json")):
        run_id = path.stem.rsplit("-", 1)[-1]
        table = json.loads(path.read_text())
        cols = table["columns"]
        rows = [dict(zip(cols, r)) for r in table["data"]]
        census, latency, drafts = {}, {}, {}
        for r in rows:
            w = int(r["rows"])
            census[w] = census.get(w, 0) + 1
            latency.setdefault(w, []).append(float(r["block_request_seconds"]))
            drafts[w] = drafts.get(w, 0) + int(r["drafts"])
        runs[run_id] = {
            "rounds": len(rows),
            "census_by_served_width": census,
            "drafts_proposed_by_width": drafts,
            "block_request_ms_mean_by_width": {
                w: 1000.0 * statistics.fmean(v) for w, v in latency.items()
            },
            "block_request_ms_total": 1000.0 * sum(sum(v) for v in latency.values()),
        }
    return runs


def weight_byte_table():
    per_cell, total = {}, 0.0
    for name, k, n, inv in CELLS:
        bytes_one = k * n * BYTES_PER_ELEMENT
        bytes_round = bytes_one * inv
        per_cell[name] = {
            "k": k,
            "n": n,
            "invocations_per_round": inv,
            "weight_bytes_per_invocation": bytes_one,
            "weight_bytes_per_round_per_pass": bytes_round,
            "family": FAMILY[name],
        }
        total += bytes_round
    for name, rec in per_cell.items():
        rec["share_of_pass"] = rec["weight_bytes_per_round_per_pass"] / total
    by_family = {}
    for name, rec in per_cell.items():
        fam = rec["family"]
        by_family[fam] = by_family.get(fam, 0.0) + rec["share_of_pass"]
    return per_cell, total, by_family


def m9_plan_map():
    """What each family runs at m=9 under the shipped plan, and the options."""
    cells = {}
    for name, k, n, inv in CELLS:
        cells[name] = {
            "cell_key_k_n": [k, n],
            "invocations_per_round": inv,
            "shipped_variant_at_m9": "staged",
            "shipped_ipg_at_m9": STAGED_IPG[9],
            "shipped_groups_at_m9": groups(9, STAGED_IPG[9]),
            "covered_by_e195_selective_plan_at_m9": False,
            "groups_at_m6_shipped": shipped_groups(6, name),
            "groups_at_m8_shipped": shipped_groups(8, name),
        }
    options = []
    for ipg in range(2, 10):
        tail = 9 % ipg
        legal = tail != 1
        options.append(
            {
                "ipg": ipg,
                "groups": groups(9, ipg),
                "tail_rows": tail,
                "legal_under_static_assert": legal,
                "max_ipg_proven_in_shipped_plan": ipg <= 6,
            }
        )
    return cells, options


def g3_exposure_map(cap8_census):
    """Where the third weight pass lands, and what it is worth to remove it."""
    local_step = E182_ROUND_MS[9] - E182_ROUND_MS[8]
    cell_sum_us = sum(
        E186_STEP_M9_US[name] * inv for name, _, _, inv in CELLS
    )
    per_cell = {}
    for name, _, _, inv in CELLS:
        ms = E186_STEP_M9_US[name] * inv / 1000.0
        per_cell[name] = {
            "invocations_per_round": inv,
            "e186_step_m9_us_per_invocation": E186_STEP_M9_US[name],
            "third_pass_ms_per_m9_round": ms,
            "share_of_isolated_cell_sum": ms / (cell_sum_us / 1000.0),
        }
    ranked_cells = sorted(
        per_cell.items(), key=lambda kv: -kv[1]["third_pass_ms_per_m9_round"]
    )

    rounds = sum(cap8_census.values())
    m9_rounds = cap8_census.get(9, 0)
    m9_share = m9_rounds / rounds if rounds else 0.0

    # Local -> ranked transfer of one weight pass. Denominator is E186's local
    # dR6, which is what E197 stores as transfer_ratio.
    den = TRANSFER["local_dR6_e186_ms"]
    ratio_measured = TRANSFER["ranked_dR6_measured_ms"] / den
    ratio_lo = TRANSFER["ranked_dR6_e197_refit_ms"][0] / den
    ratio_hi = TRANSFER["ranked_dR6_measured_1sigma_ms"][1] / den
    naive_scale = TRANSFER["local_R1_ms"] / TRANSFER["ranked_R1_ms"]

    pass_term = STEP_LAW["per_group_ms"]
    # Scenario (c): the E197 smooth-step law's own dR9 leaves almost no pass
    # component to remove.
    pessimistic_pass_ms = (
        TRANSFER["ranked_dR9_e197_smooth_step_total_ms"]
        - TRANSFER["ranked_row_term_ms"]
    )
    scenarios = {
        "a_measured_f505_transfer": pass_term * ratio_measured,
        "b_e197_refit_transfer": [pass_term * ratio_lo, pass_term * ratio_hi],
        "c_smooth_step_continuation": pessimistic_pass_ms,
        "ceiling_whole_measured_step": local_step
        * TRANSFER["ranked_dR6_measured_1sigma_ms"][1]
        / den,
    }
    k = SCORE_ENGINE["points_per_ms_of_transferred_dR9_excess"]
    flat = SCORE_ENGINE["flat_cap8_chain_score"]
    row = TRANSFER["ranked_row_term_ms"]
    dr9_model = TRANSFER["ranked_dR9_e197_smooth_step_total_ms"]

    def scored(pass_ms):
        base = flat - k * (row + pass_ms - dr9_model)
        no_pass = flat + k * (dr9_model - row)
        return {
            "cap8_with_third_pass": base,
            "cap8_without_third_pass": no_pass,
            "published_delta": no_pass - base,
        }

    return {
        "local_third_pass_ms_per_m9_round": local_step,
        "local_second_pass_ms_per_round": TRANSFER["local_dR6_e182_ms"],
        "third_pass_costs_more_than_second_by": local_step
        / TRANSFER["local_dR6_e182_ms"]
        - 1.0,
        "band_split_e182_in_path_ms": E182_G3_STEP_BY_BAND_MS,
        "cell_split_e186_isolated": per_cell,
        "cell_rank_by_third_pass_cost": [name for name, _ in ranked_cells],
        "top3_cell_share_of_isolated_sum": sum(
            rec["share_of_isolated_cell_sum"] for _, rec in ranked_cells[:3]
        ),
        "isolated_cell_sum_overexplains_layer_probe_by": E186_CELL_SUM_OVEREXPLAIN,
        "cap8_m9_round_share": m9_share,
        "local_round_weighted_third_pass_ms": m9_share * local_step,
        "transfer": {
            "naive_R1_scale_factor": naive_scale,
            "ratio_measured_f505": ratio_measured,
            "ratio_band": [ratio_lo, ratio_hi],
            "realized_fraction_of_naive_scaled_step": TRANSFER[
                "ranked_dR6_measured_ms"
            ]
            / (TRANSFER["local_dR6_e186_ms"] / naive_scale),
            "stale_finding_484_claim": "two thirds of the local step hidden on "
            "M5; derived from the FINDING 456 quadratic that FINDING 504 "
            "falsified at 5.60 sigma",
        },
        "ranked_extrapolated_third_pass_ms_per_m9_round": dict(
            scenarios,
            status="EXTRAPOLATED: ranked dR9 has never been measured; a cap-7 "
            "round verifies at most 8 rows",
        ),
        "ranked_extrapolated_round_weighted_ms": {
            "a_measured_f505_transfer": m9_share
            * scenarios["a_measured_f505_transfer"],
            "b_e197_refit_transfer": [
                m9_share * v for v in scenarios["b_e197_refit_transfer"]
            ],
            "c_smooth_step_continuation": m9_share
            * scenarios["c_smooth_step_continuation"],
        },
        "ranked_published_score_projection": {
            "a_measured_f505_transfer": scored(
                scenarios["a_measured_f505_transfer"]
            ),
            "b_e197_refit_transfer_low": scored(
                scenarios["b_e197_refit_transfer"][0]
            ),
            "c_smooth_step_continuation": scored(
                scenarios["c_smooth_step_continuation"]
            ),
            "engine": SCORE_ENGINE,
            "status": "EXTRAPOLATED, harness=ranked, desk model only. The "
            "in-flight cap-8 receipt measures dR9 directly and supersedes this.",
        },
    }


def step_law_prediction(m, g):
    return (
        STEP_LAW["intercept_ms"]
        + STEP_LAW["per_row_ms"] * m
        + STEP_LAW["per_group_ms"] * g
    )


def main():
    cap7 = parse_e202_traces()
    cap8 = load_cap8_tables()
    per_cell, pass_bytes, family_share = weight_byte_table()
    m9_cells, m9_options = m9_plan_map()

    cap7_total = {}
    if cap7:
        for leg in cap7.values():
            for w, c in leg["census_by_served_width"].items():
                cap7_total[w] = cap7_total.get(w, 0) + c

    # Step-law consistency check against the cap-8 per-round latencies.
    # Within one run only; this is NOT a cap or policy contrast.
    law_check = []
    ref = cap8.get("7y6ap5l8")
    if ref:
        for w in sorted(ref["census_by_served_width"]):
            if w < 2:
                continue
            g_uniform = groups(w, STAGED_IPG[w])
            law_check.append(
                {
                    "served_width": w,
                    "rounds": ref["census_by_served_width"][w],
                    "measured_block_request_ms_mean": ref[
                        "block_request_ms_mean_by_width"
                    ][w],
                    "step_law_ms_uniform_staged_G": step_law_prediction(w, g_uniform),
                    "uniform_staged_G": g_uniform,
                }
            )

    doc = {
        "experiment": "e206-cap8-width-work-map",
        "harness": "local",
        "campaign_base": "40be5063a136725f0ce73ed81b2f8deac5ce184d",
        "notes": [
            "Counts and desk arithmetic only. No cap-8 vs cap-7 timing contrast.",
            "Cap-7 census surface: 7f10e147 (E202). Cap-8 census surface: "
            "c47c7284 = b9ee228c + segmentedVerifyDepthCap=8; b9ee228c has a "
            "zero-byte scored diff against the campaign base 40be5063.",
        ],
        "cap7_census": {
            "source": "E202 raw traces, W&B lpwbno36, 14 legs x 512 tokens",
            "per_leg": cap7,
            "total_by_served_width": cap7_total,
        },
        "cap8_census": {
            "source": "E199 W&B per_round tables (7y6ap5l8, exdb3vyt), "
            "512 tokens, gate-qualified",
            "runs": cap8,
        },
        "weight_byte_model": {
            "bytes_per_element_affine4_g64": BYTES_PER_ELEMENT,
            "per_cell": per_cell,
            "weight_bytes_per_round_per_pass": pass_bytes,
            "family_share_of_pass": family_share,
        },
        "m9_dispatch_map": {
            "shipped_staged_ipg_table": STAGED_IPG,
            "shipped_singlepass_ipg_table": SINGLEPASS_IPG,
            "e195_selective_predicate": "m == 6 && cell != .mlpDown "
            "(Qwen35.swift:1641)",
            "cells": m9_cells,
            "ipg_options_at_m9": m9_options,
        },
        "step_law": STEP_LAW,
        "step_law_consistency_check_within_7y6ap5l8": law_check,
        "g3_exposure_map": g3_exposure_map(
            {int(k): v for k, v in ref["census_by_served_width"].items()}
            if ref
            else {}
        ),
    }
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")

    if cap7_total:
        print("\ncap-7 all-round served-width census (14 legs):")
        n7 = sum(cap7_total.values())
        for w in sorted(cap7_total):
            print(f"  qL={w}: {cap7_total[w]:5d}  {100*cap7_total[w]/n7:6.2f}%")
    print("\ncap-8 served-width census:")
    for run_id, rec in cap8.items():
        n8 = rec["rounds"]
        row = "  ".join(
            f"qL{w}={rec['census_by_served_width'][w]}"
            for w in sorted(rec["census_by_served_width"])
        )
        print(f"  {run_id} ({n8} rounds): {row}")
    print(f"\nweight bytes per round per pass: {pass_bytes/1e9:.3f} GB")
    for fam, share in sorted(family_share.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:8s} {100*share:5.1f}%")
    print("\nstep-law check inside 7y6ap5l8 (harness=local, within one leg):")
    for rec in law_check:
        print(
            f"  qL={rec['served_width']} n={rec['rounds']:3d} "
            f"measured={rec['measured_block_request_ms_mean']:7.2f} ms  "
            f"law(G={rec['uniform_staged_G']})="
            f"{rec['step_law_ms_uniform_staged_G']:7.2f} ms"
        )
    exp = doc["g3_exposure_map"]
    print("\nG=3 exposure (harness=local unless marked):")
    print(
        f"  third pass at m=9: {exp['local_third_pass_ms_per_m9_round']:.3f} ms "
        f"({100*exp['third_pass_costs_more_than_second_by']:.1f}% dearer than "
        f"the second pass)"
    )
    print(
        f"  cap-8 m=9 share {100*exp['cap8_m9_round_share']:.1f}% -> "
        f"round-weighted {exp['local_round_weighted_third_pass_ms']:.2f} ms/round"
    )
    r = exp["ranked_extrapolated_round_weighted_ms"]
    print(
        f"  ranked EXTRAPOLATED round-weighted: "
        f"(a) {r['a_measured_f505_transfer']:.3f}  "
        f"(b) {r['b_e197_refit_transfer'][0]:.3f}-"
        f"{r['b_e197_refit_transfer'][1]:.3f}  "
        f"(c) {r['c_smooth_step_continuation']:.3f} ms/round"
    )
    for name, rec in exp["ranked_published_score_projection"].items():
        if not isinstance(rec, dict) or "published_delta" not in rec:
            continue
        print(
            f"    {name:32s} {rec['cap8_with_third_pass']:.4f} -> "
            f"{rec['cap8_without_third_pass']:.4f}  "
            f"(+{rec['published_delta']:.4f})"
        )
    print("  cell rank by third-pass cost:")
    for name in exp["cell_rank_by_third_pass_cost"]:
        rec = exp["cell_split_e186_isolated"][name]
        print(
            f"    {name:14s} {rec['third_pass_ms_per_m9_round']:6.3f} ms  "
            f"{100*rec['share_of_isolated_cell_sum']:5.1f}%"
        )
    print(
        f"  top-3 cells carry "
        f"{100*exp['top3_cell_share_of_isolated_sum']:.1f}% of the third pass"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
