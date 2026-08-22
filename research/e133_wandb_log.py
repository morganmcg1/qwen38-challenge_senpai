#!/usr/bin/env python3
"""Publish the E133 offline screen to W&B.

    usage: research/e133_wandb_log.py --rung 1|2|3 [--dry]

E133 is an offline screen. It replays a captured hidden-state corpus through
numpy/MLX arrays and never times a decode leg, so no run here is a timing
measurement, a gated measurement, or a score. Every run therefore logs
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e133-sketch-first-draft-readout-offline-screen"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "197e0550ab46842b639a4ff4fe3f4889ca3b01ec"

# T0 kills a cell whose worst GATING-STRATUM absolute net miss exceeds this.
# T0b kills a cell that loses the exact affine-2 top-1 more often than 0.3 %
# of the time. F1.2 makes `beagle` and `min_carriers` the gating strata;
# `zero_weight` is reported and never gates.
T0_NET_MISS = 3.0e-3
T0B_RECALL = 0.997
GATING_STRATA = ("beagle", "min_carriers")
# F2.3. A watch line repeats rows a real stratum already counted.
WATCH_STRATA = ("essays_bacon", "essays_bacon_holdout")
SAMPLE_FLOOR = 4000
# F2.1. `hybridA` is a second candidate, not a control, so each stage_a is
# gated and selected on its own.
STAGE_A_LABEL = {"sketch": "full_c1", "affine2": "hybrid_a",
                 "nocentroid": "full_c1_no_centroid"}

RUNGS = {
    "1": {
        "run_name": "e133-rung1-corpus-capture",
        "file": "research/e133-corpus.json",
        "question":
            "how many draft-row hidden states did the instrumented worker "
            "capture over the reweighted E133 corpus, does each gating "
            "stratum clear 4,000 samples, and does every seed keep MTP "
            "parity with its serial reference",
        "command":
            "research/e133_job.sh research/e133_capture.sh all "
            "--steps 512 --depth 8",
    },
    "2": {
        "run_name": "e133-rung2-screen-validation",
        "file": "research/e133-validate.json",
        "question":
            "does the offline model of the shipped chain agree with the "
            "runtime proposal token, and can a deliberately damaged 8-bit "
            "SimHash make the miss column fail",
        "command": "python3 research/e133_screen.py validate",
    },
    "3": {
        "run_name": "e133-rung3-sketch-cell-sweep",
        "file": "research/e133-screen.json",
        "question":
            "does any compact sketch keep the shipped draft argmax while it "
            "removes enough of the 59.09 MB two-pass readout to pay for "
            "itself on the ranked score",
        "command": "python3 research/e133_screen.py screen",
    },
    "4": {
        "run_name": "e133-rung4-base-miss-attribution",
        "file": "research/e133-attrib.json",
        "question":
            "which stage of the shipped readout chain causes the base miss, "
            "does that reconcile the E133 replay with the E87 arm C ledger "
            "number, and does widening the affine-4 shortlist buy more "
            "ranked value per byte than any sketch",
        "command": "python3 research/e133_screen.py attrib",
    },
}

CELL_COLUMNS = [
    "arm", "family", "size", "stage_a", "bytes_per_row", "proj_bytes",
    "survivors", "probe_fraction", "cross_fit", "n", "n_gating",
    "net_miss_worst_gating", "net_miss_worst_gating_hi",
    "m_absolute_worst_gating", "m_absolute_worst_gating_hi",
    "m_incremental_worst_gating", "recall_worst_gating",
    "acceptance_loss_worst_gating",
    "net_miss_essays_bacon", "m_absolute_essays_bacon", "recall_essays_bacon",
    "net_miss_essays_bacon_holdout", "net_miss_essays_bacon_holdout_hi",
    "m_absolute_essays_bacon_holdout", "recall_essays_bacon_holdout",
    "arm_stage_bytes", "shipped_stage_bytes", "removed_bytes",
    "removed_step_fraction", "pct_byte_rate", "pct_head_share_7",
    "pct_head_share_9", "predicted_pct_absolute", "predicted_pct_absolute_9",
    "predicted_pct_gating", "predicted_pct_pooled",
    "predicted_pct_raw_miss", "acceptance_loss_pooled_worst_gating",
    "substitutions_live_gating", "passes_t0", "passes_t0b",
]

ATTRIB_STRATUM_COLUMNS = [
    "stratum", "gating", "watch", "n", "base_misses", "m_absolute",
    "m_absolute_lo", "m_absolute_hi", "cause_P", "cause_C", "cause_R",
    "cause_P_rate", "cause_C_rate", "cause_R_rate",
    "causes_sum_to_base_miss", "cause_R_rank", "cause_R_tie",
    "cause_R_rerank", "cause_R_rank_rate", "cause_R_tie_rate",
    "cause_R_rerank_rate", "cause_R_splits_exactly",
    "e87_strict_rank_miss", "e87_strict_rank_miss_rate",
    "probe_hit_rate_affine2", "probe_hit_rate_exact_centroid",
    "m_absolute_exact_centroid_chain", "base_miss_live", "live_rate",
    "perfect_readout_acceptance_gain", "perfect_readout_pct_realised",
    "perfect_readout_pct_full_rate",
    "miss_exact_score_tie", "miss_gap_below_1e6_relative",
    "miss_gap_below_1e4_relative", "mean_tied_rows_at_argmax_on_miss",
    "rows_with_a_tied_argmax", "rows_with_a_tied_argmax_rate",
    "sketch_net_miss", "sketch_net_miss_rate", "sketch_net_miss_exact_tie",
    "sketch_net_miss_gap_below_1e6", "sketch_net_miss_live",
    "sketch_net_miss_acceptance_delta",
]

ATTRIB_K_COLUMNS = [
    "chain", "shortlist", "rerank_bytes", "extra_bytes",
    "cost_pct_head_share_7", "cost_pct_head_share_9", "cost_pct_byte_rate",
    "step_fraction", "recovered_worst_gating", "net_pct_full_rate",
    "net_pct_realised", "stratum", "m_absolute", "m_absolute_lo",
    "m_absolute_hi", "misses", "rank_predicted_misses",
    "ge_predicted_misses", "m_absolute_strict_rank", "m_absolute_non_strict",
    "recovered_vs_k32", "swapped_live", "acceptance_gain_realised",
    "pct_full_rate", "pct_realised",
]

ATTRIB_PROBE_COLUMNS = [
    "probe_fraction", "clusters", "rows_scored", "stratum", "m_absolute",
    "m_absolute_lo", "m_absolute_hi", "misses", "probe_hit_rate",
]

# `qlowrank` and `wlowrank` fit their basis on captured hidden states. F3.1
# rules that legal under four conditions, but a basis-free column is still
# published so the two answers can be compared side by side.
BASIS_FREE_FAMILIES = ("simhash", "lowrank", "sign")

SPECTRUM_COLUMNS = [
    "stage_a", "rank", "captured_beagle", "captured_min_carriers",
    "captured_essays_bacon", "cells", "best_arm", "best_net_miss",
    "best_recall", "best_bytes_per_row",
]

LADDER_COLUMNS = [
    "stage_a", "bytes_per_row", "cells", "cells_passing_both",
    "max_byte_rate_gain_pct", "min_net_miss_worst_gating", "best_arm",
    "best_family", "best_predicted_pct", "best_predicted_pct_incremental",
    "best_predicted_pct_pooled",
    "best_predicted_pct_raw_miss", "best_net_miss", "best_recall",
]

# F4.4. The probe fraction moves `P` and `C` without moving a cell's bytes per
# row, so the byte ladder cannot show it.
PROBE_LADDER_COLUMNS = [
    "stage_a", "probe_fraction", "cells", "cells_passing_both",
    "max_byte_rate_gain_pct", "min_net_miss_worst_gating", "best_arm",
    "best_bytes_per_row", "best_gross_pct", "best_predicted_pct",
    "best_predicted_pct_incremental", "best_net_miss", "best_m_absolute",
    "best_m_incremental", "best_recall",
]

WHITENING_COLUMNS = [
    "field", "better", "whitened_wins", "ties", "whitened_losses",
    "mean_delta_whitened_minus_plain", "sign_test_p",
]

STRATUM_COLUMNS = [
    "arm", "stage_a", "stratum", "gating", "watch", "n",
    "misses_absolute", "m_absolute", "m_absolute_lo", "m_absolute_hi",
    "misses_incremental", "m_incremental", "m_incremental_lo",
    "m_incremental_hi",
    "net_miss", "net_miss_lo", "net_miss_hi", "discordant",
    "recall", "probe_hit_rate", "survivor_hit_rate",
    "p_head_step_accuracy", "p_shipped_is_target_on_live_substituted",
    "q_substitute_is_target", "substitutions", "substitutions_live",
    "acceptance_loss", "acceptance_loss_pooled_p",
    "tail_fit_usable", "tail_fit_p", "tail_fit_lo", "tail_fit_hi",
]

SURVIVAL_COLUMNS = ["arm", "stratum", "n", "width", "count", "survival"]

SEED_COLUMNS = [
    "seed", "domain", "stratum", "samples", "dumped_rows",
    "warmup_rows_skipped", "shards", "accepted_draft_rate",
    "accepted_draft_total", "effective_mean_draft_len", "round_count",
    "parity_all_ok", "all_tokens_matched", "head_sha256",
]


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def gates(cell: dict) -> tuple[bool, bool]:
    """Re-derive T0 and T0b here instead of trusting the screen's own flags.

    The publisher and the screen must agree, and an independent recomputation
    is what makes a disagreement visible.
    """
    return (cell["net_miss_worst_gating"] <= T0_NET_MISS,
            cell["recall_worst_gating"] >= T0B_RECALL)


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def corpus_summary(payload: dict) -> tuple[dict, dict]:
    seeds = payload["seeds"]
    parity = [s for s in seeds if s.get("parity_all_ok") is not True]
    by_stratum = payload["by_stratum"]
    summary = {
        "corpus_samples": payload["samples"],
        "corpus_seeds": len(seeds),
        "corpus_seeds_without_parity": len(parity),
    }
    flatten("by_stratum", by_stratum, summary)
    for stratum in GATING_STRATA:
        summary["gating_floor_met/%s" % stratum] = (
            by_stratum.get(stratum, 0) >= SAMPLE_FLOOR)
    summary["all_gating_floors_met"] = all(
        by_stratum.get(s, 0) >= SAMPLE_FLOOR for s in GATING_STRATA)
    return summary, {"corpus_seeds": table(SEED_COLUMNS, seeds)}


def validate_summary(payload: dict) -> tuple[dict, dict]:
    summary: dict = {"validate_samples": payload["samples"]}
    for key in ("offline_argmax_matches_runtime_proposal",
                "ledger_verdict_factors_as_live_and_match",
                "p_row_accepted", "prefix_live_rate", "p_row_matches_reference",
                "offline_shipped_chain_reproduces_runtime",
                ):
        summary[key] = payload.get(key)
    for section in ("proposal_mismatch", "m_shipped_live_chain",
                    "m_damaged_simhash8_control"):
        flatten(section, payload.get(section), summary)
    # The control only proves anything if it actually fails.
    summary["control_can_fail"] = (
        payload["m_damaged_simhash8_control"]["p"]
        > 10 * max(payload["m_shipped_live_chain"]["p"], 1e-6))
    return summary, {}


def screen_summary(payload: dict) -> tuple[dict, dict]:
    cells = []
    strata = []
    survival = []
    disagreements = 0
    for cell in payload["cells"]:
        t0, t0b = gates(cell)
        disagreements += int(t0 != cell["passes_t0"] or t0b != cell["passes_t0b"])
        row = dict(cell)
        row["passes_t0"] = t0
        row["passes_t0b"] = t0b
        cells.append(row)
        for stratum, stats in cell.get("by_stratum", {}).items():
            fit = stats.get("tail_fit_at_survivors") or {}
            strata.append({
                "arm": cell["arm"], "stage_a": cell["stage_a"],
                "stratum": stratum,
                "gating": stratum in GATING_STRATA,
                "watch": stratum in WATCH_STRATA,
                "tail_fit_usable": fit.get("usable", False),
                "tail_fit_p": fit.get("p"), "tail_fit_lo": fit.get("lo"),
                "tail_fit_hi": fit.get("hi"),
                **{k: v for k, v in stats.items()
                   if k not in ("survival_curve", "tail_fit_at_survivors")},
            })
            for width, count in stats.get("survival_curve", {}).items():
                survival.append({
                    "arm": cell["arm"], "stratum": stratum, "n": stats["n"],
                    "width": int(width), "count": count,
                    "survival": count / stats["n"] if stats["n"] else None,
                })

    survivors = [c for c in cells if c["passes_t0"] and c["passes_t0b"]]
    best = max(survivors, key=lambda c: c["predicted_pct_absolute"], default=None)
    # The primary metric is what a compliant cell is worth on the ranked
    # score. No compliant cell means the mechanism is worth exactly zero.
    # Advisor error 125 makes the absolute-miss price the headline, so the
    # incremental price is published beside it rather than as the metric.
    summary = {
        "screen_samples": payload["samples"],
        "screen_cells": len(cells),
        "cells_passing_t0": sum(1 for c in cells if c["passes_t0"]),
        "cells_passing_t0b": sum(1 for c in cells if c["passes_t0b"]),
        "cells_passing_both": len(survivors),
        "gate_recomputation_disagreements": disagreements,
        "e133_best_cell_ranked_pct":
            best["predicted_pct_absolute"] if best else 0.0,
        "e133_best_cell_ranked_pct_incremental":
            best["predicted_pct_gating"] if best else 0.0,
        # The selected cell's own worst gating stratum when a compliant cell
        # exists. With no compliant cell there is nothing to select, so the
        # closest any cell came to T0 is reported instead.
        "e133_worst_domain_net_miss_rate":
            best["net_miss_worst_gating"] if best else
            min((c["net_miss_worst_gating"] for c in cells),
                default=float("nan")),
        "e133_best_cell_ranked_pct_pooled":
            best["predicted_pct_pooled"] if best else 0.0,
        "e133_best_cell_ranked_pct_raw_miss":
            best["predicted_pct_raw_miss"] if best else 0.0,
        "best_cell_arm": best["arm"] if best else None,
        "p_head_step_accuracy": payload.get("p_head_step_accuracy"),
        "p_row_accepted": payload.get("p_row_accepted"),
        "offline_shipped_chain_reproduces_runtime":
            payload.get("offline_shipped_chain_reproduces_runtime"),
        "t0_threshold": T0_NET_MISS,
        "t0b_threshold": T0B_RECALL,
        "miss_to_score_pct": 203.0,
    }
    flatten("p_by_stratum", payload.get("p_head_step_accuracy_by_stratum", {}),
            summary)
    keys = ("family", "size", "stage_a", "survivors", "probe_fraction",
            "bytes_per_row", "proj_bytes", "removed_bytes", "pct_byte_rate",
            "pct_head_share_7", "pct_head_share_9", "net_miss_worst_gating",
            "m_absolute_worst_gating", "m_incremental_worst_gating",
            "acceptance_loss_worst_gating", "recall_worst_gating",
            "net_miss_essays_bacon", "recall_essays_bacon",
            "net_miss_essays_bacon_holdout", "recall_essays_bacon_holdout",
            "acceptance_loss_pooled_worst_gating", "substitutions_live_gating",
            "predicted_pct_absolute", "predicted_pct_absolute_9",
            "predicted_pct_gating", "predicted_pct_pooled",
            "predicted_pct_raw_miss")
    if best:
        for key in keys:
            summary["best_cell/%s" % key] = best[key]
    basis_free = [c for c in survivors if c["family"] in BASIS_FREE_FAMILIES]
    chosen = max(basis_free, key=lambda c: c["predicted_pct_absolute"],
                 default=None)
    summary["basis_free/cells_passing_both"] = len(basis_free)
    summary["basis_free/best_arm"] = chosen["arm"] if chosen else None
    summary["basis_free/best_predicted_pct"] = (
        chosen["predicted_pct_absolute"] if chosen else 0.0)
    if chosen:
        for key in keys:
            summary["basis_free/best_cell/%s" % key] = chosen[key]
    # F2.1. Full C1 and hybridA are gated and selected independently, so a
    # stage-A kill cannot silently take hybridA down with it.
    for stage_a, block in payload.get("by_stage_a", {}).items():
        tag = STAGE_A_LABEL.get(stage_a, stage_a)
        for field in ("cells", "cells_passing_t0", "cells_passing_t0b",
                      "cells_passing_both", "best_arm", "best_predicted_pct",
                      "best_predicted_pct_pooled",
                      "best_predicted_pct_raw_miss", "byte_ceiling_searched",
                      "cheapest_arm", "cheapest_bytes_per_row",
                      "cheapest_predicted_pct", "cheapest_predicted_pct_pooled",
                      "cheapest_predicted_pct_raw_miss"):
            summary["%s/%s" % (tag, field)] = block.get(field)
        for label, picked in (("best_cell", block.get("best_cell")),
                              ("cheapest_cell", block.get("cheapest_cell"))):
            if picked:
                for key in keys:
                    summary["%s/%s/%s" % (tag, label, key)] = picked[key]
    flatten("shipped", {k: v for k, v in payload["shipped"].items()
                        if k != "by_stratum"}, summary)
    flatten("shipped_by_stratum",
            {s: {k: v for k, v in stats.items()
                 if k in ("n", "misses_absolute", "m_absolute",
                          "m_absolute_lo", "m_absolute_hi",
                          "p_head_step_accuracy")}
             for s, stats in payload["shipped"]["by_stratum"].items()}, summary)
    flatten("shipped_structural_proxy",
            {k: v for k, v in payload["shipped_structural_proxy"].items()
             if k != "by_stratum"}, summary)
    spectrum = [{"stage_a": stage_a, "rank": int(rank), **row}
                for stage_a, rows in payload.get("spectrum_vs_miss", {}).items()
                for rank, row in rows.items()]
    flatten("query_energy",
            payload.get("query_basis", {}).get("energy_kept", {}), summary)
    ladder = [{"stage_a": stage_a, "bytes_per_row": int(size), **row}
              for stage_a, rows in payload.get("byte_ladder", {}).items()
              for size, row in rows.items()]
    # F4 error 123 withdrew the cheapest-clearing-cell rule: selection is on
    # predicted ranked value, and bytes are an input to that price rather than
    # a selection statistic. The cheapest rung stays published only to show
    # what the withdrawn rule would have cost.
    for stage_a, rows in payload.get("byte_ladder", {}).items():
        tag = STAGE_A_LABEL.get(stage_a, stage_a)
        clearing = [(int(s), r) for s, r in rows.items()
                    if r["cells_passing_both"]]
        summary["%s/ladder/rungs" % tag] = len(rows)
        summary["%s/ladder/rungs_with_a_clearing_cell" % tag] = len(clearing)
        if clearing:
            cheap_size, cheap = min(clearing, key=lambda kv: kv[0])
            best_size, top = max(clearing,
                                 key=lambda kv: kv[1]["best_predicted_pct"])
            summary["%s/ladder/cheapest_bytes_per_row" % tag] = cheap_size
            summary["%s/ladder/cheapest_arm" % tag] = cheap["best_arm"]
            summary["%s/ladder/cheapest_predicted_pct" % tag] = \
                cheap["best_predicted_pct"]
            summary["%s/ladder/best_priced_bytes_per_row" % tag] = best_size
            summary["%s/ladder/best_priced_arm" % tag] = top["best_arm"]
            summary["%s/ladder/best_priced_predicted_pct" % tag] = \
                top["best_predicted_pct"]
            summary["%s/ladder/withdrawn_cheapest_rule_costs_pct" % tag] = \
                top["best_predicted_pct"] - cheap["best_predicted_pct"]
    probe_rows = [{"stage_a": stage_a, "probe_fraction": float(p), **row}
                  for stage_a, rows in payload.get("probe_ladder", {}).items()
                  for p, row in rows.items()]
    for stage_a, rows in payload.get("probe_ladder", {}).items():
        tag = STAGE_A_LABEL.get(stage_a, stage_a)
        clearing = [(float(p), r) for p, r in rows.items()
                    if r["cells_passing_both"]]
        summary["%s/probe_ladder/fractions" % tag] = len(rows)
        summary["%s/probe_ladder/fractions_with_a_clearing_cell" % tag] = \
            len(clearing)
        if clearing:
            best_p, top = max(clearing,
                              key=lambda kv: kv[1]["best_predicted_pct"])
            summary["%s/probe_ladder/best_probe_fraction" % tag] = best_p
            summary["%s/probe_ladder/best_arm" % tag] = top["best_arm"]
            summary["%s/probe_ladder/best_predicted_pct" % tag] = \
                top["best_predicted_pct"]
    whitening = payload.get("whitening_paired", {})
    white_rows = [{"field": field, **row}
                  for field, row in whitening.get("fields", {}).items()]
    for field in ("paired_cells", "t0_verdict_flips", "t0b_verdict_flips",
                  "best_clearing_plain_arm",
                  "best_clearing_plain_predicted_pct",
                  "best_clearing_whitened_arm",
                  "best_clearing_whitened_predicted_pct"):
        summary["whitening/%s" % field] = whitening.get(field)
    for field, row in whitening.get("fields", {}).items():
        for stat in ("whitened_wins", "ties", "whitened_losses",
                     "mean_delta_whitened_minus_plain", "sign_test_p"):
            summary["whitening/%s/%s" % (field, stat)] = row.get(stat)
    return summary, {"screen_cells": table(CELL_COLUMNS, cells),
                     "screen_by_stratum": table(STRATUM_COLUMNS, strata),
                     "screen_survival": table(SURVIVAL_COLUMNS, survival),
                     "screen_spectrum": table(SPECTRUM_COLUMNS, spectrum),
                     "screen_byte_ladder": table(LADDER_COLUMNS, ladder),
                     "screen_probe_ladder": table(PROBE_LADDER_COLUMNS,
                                                  probe_rows),
                     "screen_whitening": table(WHITENING_COLUMNS, white_rows)}


def attrib_summary(payload: dict) -> tuple[dict, dict]:
    """F4.1-F4.3 and F5.1-F5.4: where the base miss comes from, and what the
    one-integer shortlist lever buys against it."""
    summary: dict = {
        "attrib_samples": payload["samples"],
        "attrib_sketch_cell": payload.get("sketch_cell"),
        "shortlist_shipped": payload["shortlist_shipped"],
        "probe_fraction_shipped": payload["probe_fraction_shipped"],
    }
    strata = []
    for name, row in payload["by_stratum"].items():
        strata.append({"stratum": name, **row})
        flatten("attrib/%s" % name, row, summary)
    summary["all_causes_sum_to_base_miss"] = all(
        r["causes_sum_to_base_miss"] for r in payload["by_stratum"].values())
    summary["all_cause_R_splits_exactly"] = all(
        r["cause_R_splits_exactly"] for r in payload["by_stratum"].values())
    gating = [r for n, r in payload["by_stratum"].items() if r["gating"]]
    summary["worst_gating_m_absolute"] = max(
        r["m_absolute"] for r in gating)
    summary["worst_gating_e87_strict_rank_miss_rate"] = max(
        r["e87_strict_rank_miss_rate"] for r in gating)

    k_rows = []
    for chain, key in (("shipped", "k_curve"), ("sketch", "k_curve_sketch")):
        for k, row in payload.get(key, {}).items():
            head = {c: row.get(c) for c in ATTRIB_K_COLUMNS if c in row}
            for name, sub in row["by_stratum"].items():
                k_rows.append({"chain": chain, **head, "stratum": name, **sub})
            if chain == "shipped":
                flatten("k_curve/%s" % k,
                        {c: row.get(c) for c in
                         ("extra_bytes", "cost_pct_head_share_7",
                          "recovered_worst_gating", "net_pct_full_rate",
                          "net_pct_realised")}, summary)
    shipped_k = payload.get("k_curve", {})
    if shipped_k:
        best_k, best = max(shipped_k.items(),
                           key=lambda kv: kv[1]["net_pct_full_rate"])
        summary["k_curve/argmax_shortlist"] = int(best_k)
        summary["k_curve/argmax_net_pct_full_rate"] = best["net_pct_full_rate"]
        summary["k_curve/argmax_net_pct_realised"] = best["net_pct_realised"]
        summary["k_curve/argmax_recovered_worst_gating"] = \
            best["recovered_worst_gating"]

    probe_rows = []
    for p, row in payload.get("probe_curve", {}).items():
        head = {"probe_fraction": float(p), "clusters": row["clusters"],
                "rows_scored": row["rows_scored"]}
        for name, sub in row["by_stratum"].items():
            probe_rows.append({**head, "stratum": name, **sub})
        flatten("probe_curve/%s" % p,
                {n: s["m_absolute"] for n, s in row["by_stratum"].items()},
                summary)
    return summary, {
        "attrib_by_stratum": table(ATTRIB_STRATUM_COLUMNS, strata),
        "attrib_k_curve": table(ATTRIB_K_COLUMNS, k_rows),
        "attrib_probe_curve": table(ATTRIB_PROBE_COLUMNS, probe_rows),
    }


BUILDERS = {"1": corpus_summary, "2": validate_summary, "3": screen_summary,
            "4": attrib_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    payload = json.loads(pathlib.Path(spec["file"]).read_text())
    summary, tables = BUILDERS[args.rung](payload)
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "offline",
        "timing_valid": False,
        "host": HOST,
        "base_sha": BASE_SHA,
        "rung": args.rung,
        "question": spec["question"],
        "command": spec["command"],
    })

    if args.dry:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name=spec["run_name"], job_type="offline-screen",
                     config={"experiment": "E133", "rung": args.rung,
                             "base_sha": BASE_SHA, "host": HOST,
                             "corpus": "e133-corpus-manifest.json",
                             "command": spec["command"],
                             "question": spec["question"]})
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e133-rung%s" % args.rung, type="offline-screen")
    artifact.add_file(spec["file"])
    run.log_artifact(artifact)
    print(run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
