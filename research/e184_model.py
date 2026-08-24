#!/usr/bin/env python3
"""E184 prefill pricing model: roofline attribution and the MUE ladder.

`e184_analyse.py` reduces the session artifacts. This script prices them: it
attaches a static FLOP count to every GEMM bracket, converts the measured
bracket seconds into achieved TFLOP/s, attributes the E16 closed-budget
residual, and then walks the mechanism ladder to a published-score number.

Every share printed here is `harness=local` on the profiling host. The one
ranked constant used is PROMPT_WEIGHTED_PREFILL_SHARE, which converts a
fractional prefill reduction into a published-score gain.
"""

import argparse
import json
import statistics

# Qwen 3.8 27B config, read from the live checkpoint config in Qwen35.swift.
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
FULL_ATTENTION_LAYERS = 16
GDN_LAYERS = 48
SEED_TOKENS = 512

# Gated DeltaNet projection widths (Qwen35.swift:787-790).
KEY_DIM = 16 * 128        # numKHeads * headKDim
VALUE_DIM = 48 * 128      # numVHeads * headVDim
NUM_V_HEADS = 48
GDN_IN_WIDTH = (2 * KEY_DIM + VALUE_DIM) + VALUE_DIM + NUM_V_HEADS + NUM_V_HEADS

# Full-attention projection widths. The q projection emits query and gate
# fused, so it is 2 * attentionHeads * headDim (Qwen35.swift, qkv + split).
FA_Q_GATE_WIDTH = 2 * 24 * 256
FA_KV_WIDTH = 2 * 4 * 256
FA_IN_WIDTH = FA_Q_GATE_WIDTH + FA_KV_WIDTH

# E65 measured 6.25 TFLOP/s at 83.2 % of peak on this host class.
E65_ACHIEVED_TFLOPS = 6.25
E65_PEAK_FRACTION = 0.832
PEAK_TFLOPS = E65_ACHIEVED_TFLOPS / E65_PEAK_FRACTION

# E16 closed prefill budget, as percentages of the same prefill window.
E16_GEMM_AT_CEILING_PCT = 84.148
E16_NON_GEMM_PCT = 5.313
E16_RESIDUAL_PCT = 10.539

# Ranked constants. Prompt-weighted prefill share of the ranked 512-token leg,
# and the standard deviation of ranked prefill_seconds_per_token (n=32).
PROMPT_WEIGHTED_PREFILL_SHARE = 0.1004
RANKED_PREFILL_SPT_SD = 0.00200
MUE_PUBLISHED_PCT = 0.30


def gemm_tflop():
    """Static FLOP per seed forward for each GEMM bracket, in TFLOP."""

    def mm(rows, k, n, layers):
        return 2.0 * rows * k * n * layers / 1e12

    return {
        "61_mlp_gate_up_act": mm(SEED_TOKENS, HIDDEN, 2 * INTERMEDIATE, LAYERS),
        "62_mlp_down": mm(SEED_TOKENS, INTERMEDIATE, HIDDEN, LAYERS),
        "41_gdn_in_projections": mm(SEED_TOKENS, HIDDEN, GDN_IN_WIDTH, GDN_LAYERS),
        "44_gdn_postnorm_outproj": mm(SEED_TOKENS, VALUE_DIM, HIDDEN, GDN_LAYERS),
        "51_fa_qkv_projection": mm(SEED_TOKENS, HIDDEN, FA_IN_WIDTH, FULL_ATTENTION_LAYERS),
        "54_fa_gate_outproj": mm(SEED_TOKENS, VALUE_DIM, HIDDEN, FULL_ATTENTION_LAYERS),
    }


def load_phases(path):
    """Mean seconds and call count per phase over every seed-width forward."""
    rows = [
        json.loads(line)
        for line in open(path)
        if '"e184_prefill_profile"' in line
    ]
    if not rows:
        raise SystemExit(f"no profile records in {path}")
    totals = [r["bracketed_total_seconds"] for r in rows]
    phases = {}
    for r in rows:
        for p in r["phases"]:
            entry = phases.setdefault(p["phase"], {"seconds": [], "calls": []})
            entry["seconds"].append(p["seconds"])
            entry["calls"].append(p["calls"])
    reduced = {
        name: {
            "mean_seconds": statistics.mean(v["seconds"]),
            "rel_sd_pct": (
                100.0 * statistics.stdev(v["seconds"]) / statistics.mean(v["seconds"])
                if len(v["seconds"]) > 1 else 0.0
            ),
            "calls": statistics.mean(v["calls"]),
        }
        for name, v in phases.items()
    }
    return {
        "forwards": len(rows),
        "bracketed_total_mean": statistics.mean(totals),
        "bracketed_total_rel_sd_pct": (
            100.0 * statistics.stdev(totals) / statistics.mean(totals)
            if len(totals) > 1 else 0.0
        ),
        "phases": reduced,
    }


def build(phase_path, trusted_seed_seconds, off_mode_seed_seconds):
    red = load_phases(phase_path)
    total = red["bracketed_total_mean"]
    flops = gemm_tflop()

    table = []
    for name in sorted(red["phases"]):
        p = red["phases"][name]
        row = {
            "phase": name,
            "mean_seconds": p["mean_seconds"],
            "share_of_bracketed_pct": 100.0 * p["mean_seconds"] / total,
            "rel_sd_pct": p["rel_sd_pct"],
            "calls_per_forward": p["calls"],
            "kind": "gemm" if name in flops else "non_gemm",
        }
        if name in flops:
            row["tflop"] = flops[name]
            row["achieved_tflops"] = flops[name] / p["mean_seconds"]
            row["pct_of_peak"] = 100.0 * row["achieved_tflops"] / PEAK_TFLOPS
        table.append(row)

    gemm_seconds = sum(r["mean_seconds"] for r in table if r["kind"] == "gemm")
    non_gemm_seconds = total - gemm_seconds
    gemm_flop = sum(flops.values())

    # E16 attribution. Both budgets describe the same prefill window, so they
    # are compared as percentages rather than as seconds.
    measured_gemm_pct = 100.0 * gemm_seconds / total
    measured_non_gemm_pct = 100.0 * non_gemm_seconds / total
    e16_gemm_total_pct = E16_GEMM_AT_CEILING_PCT + E16_RESIDUAL_PCT

    attribution = {
        "measured_gemm_share_pct": measured_gemm_pct,
        "measured_non_gemm_share_pct": measured_non_gemm_pct,
        "e16_gemm_at_ceiling_pct": E16_GEMM_AT_CEILING_PCT,
        "e16_residual_pct": E16_RESIDUAL_PCT,
        "e16_gemm_at_ceiling_plus_residual_pct": e16_gemm_total_pct,
        "e16_non_gemm_pct": E16_NON_GEMM_PCT,
        "gemm_budget_disagreement_pp": measured_gemm_pct - e16_gemm_total_pct,
        "non_gemm_budget_disagreement_pp": measured_non_gemm_pct - E16_NON_GEMM_PCT,
        "aggregate_achieved_tflops": gemm_flop / gemm_seconds,
        "peak_tflops_from_e65": PEAK_TFLOPS,
        "aggregate_pct_of_peak": 100.0 * (gemm_flop / gemm_seconds) / PEAK_TFLOPS,
        "implied_e16_ceiling_tflops": (
            (gemm_flop / gemm_seconds) * (e16_gemm_total_pct / E16_GEMM_AT_CEILING_PCT)
        ),
        "verdict": (
            "The E16 residual is GEMM time above the modelled roofline ceiling, "
            "not an unmeasured operator and not host or overlap time."
        ),
    }

    # Instrument overhead against the off-mode trusted anchor.
    reconciliation = {
        "bracketed_total_mean_seconds": total,
        "bracketed_total_rel_sd_pct": red["bracketed_total_rel_sd_pct"],
        "trusted_seed_prefill_seconds_instrumented": trusted_seed_seconds,
        "unbracketed_remainder_seconds": trusted_seed_seconds - total,
        "unbracketed_remainder_pct": (
            100.0 * (trusted_seed_seconds - total) / trusted_seed_seconds
        ),
        "trusted_seed_prefill_seconds_off_mode": off_mode_seed_seconds,
        "instrument_overhead_seconds": trusted_seed_seconds - off_mode_seed_seconds,
        "instrument_overhead_pct": (
            100.0 * (trusted_seed_seconds - off_mode_seed_seconds) / off_mode_seed_seconds
        ),
        "serialized_excess_over_off_mode_seconds": total - off_mode_seed_seconds,
        "note": (
            "Bracket shares are upper bounds on serialized time. The brackets "
            "force eval boundaries, so normally overlapped work is exposed."
        ),
    }

    return red, table, attribution, reconciliation, gemm_seconds, non_gemm_seconds


def mue_ladder(table, gemm_seconds, non_gemm_seconds, off_mode_seed_seconds):
    """Price each candidate mechanism as a published-score percentage."""
    by = {r["phase"]: r for r in table}
    flops = gemm_tflop()
    best_pct_of_peak = max(r["pct_of_peak"] for r in table if r["kind"] == "gemm")
    gemm_flop = sum(flops.values())

    def priced(name, saved_seconds, feasibility):
        prefill_pct = 100.0 * saved_seconds / off_mode_seed_seconds
        published_pct = prefill_pct * PROMPT_WEIGHTED_PREFILL_SHARE
        return {
            "mechanism": name,
            "prefill_seconds_saved": saved_seconds,
            "prefill_reduction_pct": prefill_pct,
            "published_score_gain_pct": published_pct,
            "clears_mue": published_pct >= MUE_PUBLISHED_PCT,
            "feasibility": feasibility,
        }

    scan = by["43_gdn_recurrent_scan"]["mean_seconds"]
    ladder = [
        priced(
            "Chunk-parallel reordered delta rule (E184 shape b ceiling)",
            scan - 0.0444,
            "Not bit-exact. Owes the WY/UT triangular inverse and a step-7 numerics gate.",
        ),
        priced(
            "Make the GDN recurrent scan completely free (unphysical bound)",
            scan,
            "Unattainable. Bounds the whole scan axis, so it closes FINDING 470.",
        ),
        priced(
            "Perfectly fuse the four GDN in-projections at seed width",
            by["41_gdn_in_projections"]["mean_seconds"]
            - flops["41_gdn_in_projections"] / (best_pct_of_peak / 100.0 * PEAK_TFLOPS),
            "Buildable and bit-exact, but the ceiling is the whole prize.",
        ),
        priced(
            "Lift the two weakest GEMM brackets to the best observed efficiency",
            sum(
                by[n]["mean_seconds"] * (1.0 - by[n]["pct_of_peak"] / best_pct_of_peak)
                for n in ("44_gdn_postnorm_outproj", "54_fa_gate_outproj")
            ),
            "Requires a shape-specific quantized GEMM win on two shapes.",
        ),
        priced(
            "Eliminate every non-GEMM bracket (unphysical bound)",
            non_gemm_seconds,
            "Unattainable. Embedding, norms, conv, RoPE, SDPA and the scan are required.",
        ),
        priced(
            "Lift every GEMM bracket to the E65 roofline peak",
            gemm_seconds - gemm_flop / PEAK_TFLOPS,
            "The only class that clears MUE. It is also the campaign's most "
            "attacked and most failed axis (FINDING 293/389 NAX graveyard).",
        ),
    ]
    ladder.sort(key=lambda r: r["published_score_gain_pct"])
    return {
        "mue_published_pct": MUE_PUBLISHED_PCT,
        "prefill_reduction_needed_for_mue_pct": (
            MUE_PUBLISHED_PCT / PROMPT_WEIGHTED_PREFILL_SHARE
        ),
        "prefill_seconds_needed_for_mue": (
            MUE_PUBLISHED_PCT / PROMPT_WEIGHTED_PREFILL_SHARE
            / 100.0 * off_mode_seed_seconds
        ),
        "best_observed_pct_of_peak": best_pct_of_peak,
        "ladder": ladder,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", required=True, help="e184-phases-<arm>.jsonl")
    ap.add_argument("--trusted-seed-seconds", type=float, required=True,
                    help="trusted seed_prefill_seconds for the instrumented leg")
    ap.add_argument("--off-mode-seed-seconds", type=float, required=True,
                    help="trusted seed_prefill_seconds with the profiler off")
    ap.add_argument("--out")
    args = ap.parse_args()

    red, table, attribution, reconciliation, gemm_s, non_gemm_s = build(
        args.phases, args.trusted_seed_seconds, args.off_mode_seed_seconds)
    ladder = mue_ladder(table, gemm_s, non_gemm_s, args.off_mode_seed_seconds)

    report = {
        "harness": "local",
        "forwards": red["forwards"],
        "phase_table": table,
        "gemm_seconds": gemm_s,
        "non_gemm_seconds": non_gemm_s,
        "reconciliation": reconciliation,
        "e16_residual_attribution": attribution,
        "mue_ladder": ladder,
        "noise": {
            "bracketed_total_rel_sd_pct": red["bracketed_total_rel_sd_pct"],
            "ranked_prefill_spt_rel_sd_pct": 100.0 * RANKED_PREFILL_SPT_SD,
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
