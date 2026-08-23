#!/usr/bin/env python3
"""Publish E156 evidence to Weights and Biases.

One resumable run for the whole assignment, so the run id quoted in the first
interim comment is the same id that carries the terminal report. Re-running
this script adds the sections whose JSON inputs now exist and leaves the rest
untouched.

E156 ships ONE blind composed candidate: the 128x32 NAX seed-prefill retile
together with the double-buffered k-loop, built on the crown NAX file pair.

FRAME AND SIGN CONVENTION, RULE 144. Every percentage below names its frame in
words and carries an explicit sign convention. A NEGATIVE prefill percentage
means the candidate spends LESS time than its base, which is an improvement.
Every number is tagged `harness=local`, `harness=ranked` or `harness=offline`.
Nothing here is priced from an AIR delta: on this family AIR overstates the ISA
change by 181.3 times, so AIR is reported as a build fact only.

THE HOST CANNOT EXECUTE THE ARM. `is_nax_available()` requires Apple GPU
architecture generation 17 or newer. This Mac is an M4 Pro, `applegpu_g16s`,
generation 16, so `affine_qmm_t_nax` runs zero times locally. Every E156 claim
is therefore a build-time and source-level claim, and the prefill effect on the
crown base is reported as `not_locally_measurable` rather than estimated.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
RESUME_ID = "e156alphonser1"

PR_NUMBER = 156
BRANCH = "qwen-alphonse/e156-prefill-double-buffer-and-minimal-surface"
BASE_SHA = "d0422d1d4c3c8901cbc2382dc9ee79666e786d94"
CROWN_SHA = "0863b06ac16e26e48fc06e97444095b00feb66d4"
BUDGET_BASE = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
PARKED_DBUF_COMMIT = "a6fccd2b"
RETILE_ONLY_COMMIT = "8b515fd8"

# The published prior for the retile arm's prefill effect. It was measured on
# another base, on hardware that can execute NAX. It is NOT an E156
# measurement and must never be reported as one.
PREFILL_PRIOR_PCT = -4.12
PREFILL_PRIOR_LABEL = "prior, measured off this base, NOT an E156 measurement"

# Minimum useful effect for the prefill share, set by the advisor.
MIN_USEFUL_EFFECT_PCT = -1.5403

# Published-score conversion for a prefill-share percentage.
PUBLISHED_PCT_PER_PREFILL_PCT = 0.100438

HOST = "Apple M4 Pro, Mac mini Mac16,11, applegpu_g16s, GPU arch generation 16"


def load(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def compose_section(audit: dict) -> tuple[dict, dict]:
    census = audit["e156_instantiation_census"]
    crossings = audit["e156_tgp_limit_crossings"]
    metrics = {
        "e156_float_instantiation_emitted":
            float(audit["e156_float_instantiation_emitted"]),
        "e156_float_bn64_instantiations": census["float_BN64_instantiations"],
        "e156_tgp_limit_crossing_pairs": len(crossings),
        "e156_tgp_limit_crossing_instantiations":
            census["e156_tgp_limit_crossing_instantiations"],
        "e156_armed_kernel_instantiations_total": census["instantiations_total"],
        "e156_tgp_bytes_dbuf_64x64_float":
            audit["e156_tgp_bytes_dbuf_64x64"]["float"],
        "e156_tgp_bytes_dbuf_64x64_bfloat16_t":
            audit["e156_tgp_bytes_dbuf_64x64"]["bfloat16_t"],
        "e156_tgp_bytes_dbuf_128x32_float":
            audit["e156_tgp_bytes_dbuf_128x32"]["float"],
        "e156_tgp_bytes_dbuf_128x32_bfloat16_t":
            audit["e156_tgp_bytes_dbuf_128x32"]["bfloat16_t"],
        "e156_composed_fits_every_shipped_cell":
            float(audit["e156_composed_fits_every_shipped_cell"]),
        # FALSE, and reported as false. The composition is free on the SCORED
        # cell but not on the group-32 cells, where the retile's loader guard
        # keeps the 64-wide host tile and the double buffer stages a second
        # 64-wide half.
        "e156_composed_costs_zero_extra_tgp_bytes":
            float(audit["e156_composed_costs_zero_extra_tgp_bytes"]),
        "e156_composed_scored_cell_bytes": audit["e156_composed_scored_cell_bytes"],
        "e156_composed_scored_cell_armed":
            float(audit["e156_composed_scored_cell_armed"]),
        "e156_composed_arms_every_shipped_cell":
            float(audit["e156_composed_arms_every_shipped_cell"]),
        "e156_composed_cells_disarmed_by_capacity":
            len(audit["e156_composed_cells_disarmed_by_capacity"]),
        "e156_compose_shared_crown_lines":
            audit["e156_compose_shared_crown_lines"],
        "e156_compose_overlap_is_retile_inserted_plumbing_only":
            float(audit["e156_compose_overlap_is_retile_inserted_plumbing_only"]),
        "e156_crown_nax_pair_equals_frontier_pair":
            float(audit["e156_crown_nax_pair_equals_frontier_pair"]),
    }
    summary = {
        "e156_compose_verdict": audit["e156_compose_verdict"],
        "e156_compose_region_result": audit["e156_compose_region_result"],
        "e156_compose_source_evidence": audit["e156_compose_source_evidence"],
        "e156_tgp_limit_crossing_evidence": audit["e156_tgp_limit_crossing_evidence"],
        "e156_tgp_limit_crossings": json.dumps(audit["e156_tgp_limit_crossings"]),
        "e156_dbuf_base_assumption": (
            "The parked double buffer a6fccd2b ASSUMES the retile; it is not "
            "standalone. Its parent is 8b515fd8, the E151 R1 retile tip, and "
            "the compile gate confirms the dependency rather than inferring "
            "it from history: with the retile off and the capacity predicate "
            "defeated, the compiler refuses the float cell by name. The crown "
            "pair alone therefore cannot carry the double buffer at BN 64. "
            "harness=offline"
        ),
    }
    return metrics, summary


def budget_section(gate: dict) -> tuple[dict, dict]:
    """Byte-budget and scaffold fields from the gate-chain record."""
    ba = gate["byte_accounting"]
    sizes = ba["quantized_nax_h_plus_cpp_bytes"]
    enforced = next(p for p in gate["phases"] if p["n"] == 5)
    attributable = next(p for p in gate["phases"] if p["n"] == "5b")
    metrics = {
        "e156_growth_enforced": enforced["growth_enforced_bytes"],
        "e156_growth_enforced_limit": enforced["growth_limit_bytes"],
        "e156_growth_enforced_headroom": enforced["growth_headroom_bytes"],
        "e156_growth_attributable": attributable["growth_attributable_bytes"],
        "e156_scaffold_closed_by_inheritance_bytes":
            ba["e156_scaffold_closed_by_inheritance_bytes"],
        "e156_nax_pair_bytes_crown": sizes["crown_0863b06a"],
        "e156_nax_pair_bytes_pr_base": sizes["pr_base_d0422d1d"],
        "e156_nax_pair_bytes_candidate": sizes["candidate_head"],
        "e156_net_growth_crown_to_candidate_bytes":
            ba["net_growth_crown_to_candidate_bytes"],
    }
    summary = {
        "e156_scaffold_closed_by_inheritance": (
            "The 23,298 B frontier-to-base scaffold delta reproduces exactly "
            "across 14 hunks. Its UN-PRICED character is closed for this "
            "candidate, because the retile arm is ON and those bytes are now "
            "the shipped mechanism rather than dead research plumbing. Its "
            "BYTE cost is NOT reclaimed: the retile arm IS that scaffold, so "
            "the candidate keeps it and the double buffer adds more. Two-file "
            "totals: crown 99,218 B, PR base 108,920 B, candidate 129,532 B. "
            "growth_attributable is 20,612 B of source, not emitted code; the "
            "retile off-branch text remains present and compile-time dead. "
            "harness=offline"
        ),
        "e156_growth_frame": (
            "growth_enforced is measured against campaign frontier "
            "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf and growth_attributable "
            "against this experiment's own PR base "
            "d0422d1d4c3c8901cbc2382dc9ee79666e786d94. Both are source bytes "
            "against the 262,144 B growth limit. harness=offline"
        ),
    }
    return metrics, summary


def barrier_section(proof: dict) -> tuple[dict, dict]:
    metrics = {
        "e156_barrier_proof_exact": float(proof["e156_barrier_proof_exact"]),
        "e156_barrier_configurations_enumerated":
            proof["configurations_enumerated"],
        "e156_barrier_orderings_checked": proof["orderings_checked"],
        "e156_barrier_failing_controls_caught":
            proof["e156_barrier_failing_controls_caught"],
        "e156_barrier_failing_controls_expected":
            proof["failing_controls_expected"],
        "e156_k_order_preserved": float(proof["e156_k_order_preserved"]),
        "e156_barrier_source_binding_ok":
            float(proof["source_binding_ok"]),
    }
    summary = {
        "e156_barrier_proof_evidence": proof["evidence"],
        "e156_k_order_preserved_evidence": proof["k_order_evidence"],
    }
    return metrics, summary


def ulp_section(u: dict) -> tuple[dict, dict]:
    """F3 section 5: the k accumulation order is unchanged, with a control."""
    src = u["e156_k_order_source_evidence"]
    num = u["e156_k_order_numerical"]
    req = num["required_controls_that_must_fail"]
    diag = num["diagnostic_controls"]
    trials = num["trials"]
    metrics = {
        "e156_k_accumulation_order_unchanged":
            float(u["e156_k_accumulation_order_unchanged"]),
        "e156_kord_source_identical_after_normalisation":
            float(src["identical_after_normalisation"]),
        "e156_kord_header_and_twin_agree":
            float(src["header_and_twin_agree"]),
        "e156_kord_arithmetic_statements_removed":
            len(src["arithmetic_statements_removed"]),
        "e156_kord_event_lists_equal": float(num["event_lists_equal"]),
        "e156_kord_events_per_output_element":
            num["events_per_output_element"],
        "e156_kord_arms_agree_bit_for_bit": num["arms_agree_bit_for_bit"],
        "e156_kord_trials": trials,
        "e156_kord_required_controls_all_caught":
            float(num["required_controls_all_caught"]),
        "e156_kord_control_one_ulp_output_caught": req["one_ulp_output"],
        "e156_kord_control_reverse_all_events_caught":
            req["reverse_all_events"],
        "e156_kord_diag_one_ulp_weight_caught": diag["one_ulp_weight"],
        "e156_kord_diag_one_ulp_activation_caught":
            diag["one_ulp_activation"],
        "e156_kord_diag_swap_two_k_tiles_caught": diag["swap_two_k_tiles"],
        "e156_kord_diag_reverse_sk_substeps_caught":
            diag["reverse_sk_substeps"],
        "e156_kord_controls_caught": u["e156_k_order_rule92_controls_caught"],
        "e156_kord_controls_expected":
            u["e156_k_order_rule92_controls_expected"],
    }
    summary = {
        "e156_k_accumulation_order_evidence": (
            "Rule 92 positive control, harness=offline. The double-buffered "
            f"k-loop (header line {src['header_dbuf_loop_line']}) and the "
            f"single-buffered k-loop (line {src['header_single_loop_line']}) "
            "differ only in WHERE the weight tile is read from: Wk = Ws + "
            "cur * Ws_tile instead of Ws. The inner kk1 accumulation is "
            "otherwise statement-identical, "
            f"{len(src['arithmetic_statements_removed'])} arithmetic "
            "statements were removed, and the emitted fused-multiply-add "
            "event lists are equal at "
            f"{num['events_per_output_element']} events per output "
            f"element. Over {trials} randomised trials the two arms agree "
            f"bit for bit {num['arms_agree_bit_for_bit']}/{trials}. The "
            "comparison is proven able to fail: perturbing the accumulator "
            f"by one ulp is caught {req['one_ulp_output']}/{trials} and "
            "reversing the whole event order is caught "
            f"{req['reverse_all_events']}/{trials}. This is double buffering "
            "in the strict sense of F3 section 5: data moves earlier in "
            "time, no addition moves."
        ),
        "e156_k_accumulation_order_caveat": (
            "Honest limit on the controls. Perturbing an INPUT by one ulp is "
            f"caught {diag['one_ulp_weight']}/{trials} for a weight and "
            f"{diag['one_ulp_activation']}/{trials} for an activation, "
            "because a one-ulp input change is usually absorbed inside this "
            "GEMM's own accumulation. Those two are reported as diagnostics, "
            "not as pass criteria; only the output-side and full-reversal "
            "controls gate the claim. Read alongside F3 section 5, this "
            "TEMPERS the expectation that a one-ulp input difference "
            "necessarily propagates: within a single GEMM it usually does "
            "not. It does not weaken the order claim itself, which rests on "
            "equal event lists rather than on a sampling argument."
        ),
        "e156_k_loop_shape": json.dumps(
            {"K": num["K"], "BK": num["BK"], "SK": num["SK"]}),
        "e156_k_order_accumulation_nest": src["accumulation_nest"],
    }
    return metrics, summary


def prefill_share_section(p: dict) -> tuple[dict, dict]:
    """F3 section 4: how much prefill actually reaches affine_qmm_t_nax."""
    pm = p["e156_prize_model"]
    cc = p["checkpoint_cross_check"]
    sens = p["e156_prefill_qmm_share_sensitivity"]
    metrics = {
        "e156_prefill_qmm_share_estimate":
            p["e156_prefill_qmm_share_estimate"],
        "e156_prefill_all_gemm_share": p["e156_prefill_all_gemm_share"],
        "e156_prefill_qmm_share_if_lm_head_scores_all_positions":
            sens["lm_head_every_seed_position"],
        "e156_prefill_published_pct_per_kernel_pct":
            pm["published_pct_per_kernel_pct"],
        "e156_prefill_kernel_pct_to_close_the_gap":
            pm["kernel_speedup_needed_to_close_the_1_2578_pct_gap"],
        "e156_prefill_checkpoint_cross_check_ok":
            float(cc["every_projection_agrees_with_the_checkpoint"]),
        "e156_prefill_projections_enumerated": len(p["projections"]),
        "e156_prefill_projections_routed_to_nax": sum(
            1 for r in p["projections"] if r["routes_to_affine_qmm_t_nax"]),
        "e156_prefill_layers_full_attention": p["layers_full_attention"],
        "e156_prefill_layers_gdn": p["layers_gdn"],
    }
    routed = [r["family"] for r in p["projections"]
              if r["routes_to_affine_qmm_t_nax"]]
    missed = [(r["family"], r["routing_reason"]) for r in p["projections"]
              if not r["routes_to_affine_qmm_t_nax"]]
    summary = {
        "e156_prefill_qmm_share_evidence": (
            "harness=offline, FLOP share and NOT a time share. Of one "
            "512-token seed prefill forward pass, "
            f"{p['e156_prefill_qmm_share_estimate'] * 100:.2f} percent of the "
            "multiply-accumulate work sits in projections that reach "
            f"affine_qmm_t_nax. {len(routed)} of {len(p['projections'])} "
            "projection families route there. The misses are small and "
            "structural: "
            + "; ".join(f"{f} ({why})" for f, why in missed)
            + ". Every derived width, depth and layer count was cross-checked "
            "against the packed safetensors headers and all agree "
            f"({cc['every_projection_agrees_with_the_checkpoint']}), so a "
            "misread of the Swift source could not survive into the share."
        ),
        "e156_prefill_qmm_share_routing_rule": (
            "The easy trap is that a transposed non-batched matmul with "
            "M >= vector_limit enters qmm_splitk FIRST "
            "(quantized.cpp:1418-1424) and only falls through to qmm, and so "
            "to NAX, when split_k collapses to 1 (quantized.cpp:805-810). At "
            "M=512 that means N must exceed about 512. It is why the two "
            "48-wide GDN b and a projections never reach the kernel this "
            "candidate changes, even though they are quantized and "
            "transposed."
        ),
        "e156_prefill_prize_equation": pm["equation"],
        "e156_prefill_flop_vs_time": p["flop_share_is_not_time_share"],
        "e156_prefill_share_sensitivity_verdict": sens["verdict"],
        "e156_prefill_source_provenance":
            json.dumps(p["source_provenance"], indent=1),
    }
    return metrics, summary


def compile_section(cg: dict) -> tuple[dict, dict]:
    metrics = {
        f"e156_cg_{k[len('e156_'):]}": float(bool(cg[k]))
        for k in cg["e156_compile_gate_required_fields"]
    }
    metrics["e156_compile_gate_pass"] = float(cg["e156_compile_gate_pass"])
    metrics["e156_compile_gate_required_checks"] = len(
        cg["e156_compile_gate_required_fields"])
    for label, dg in cg["e156_arm_combination_digests"].items():
        metrics[f"e156_air_bytes_{label}"] = cg["e156_air_bytes"][
            {"both_off": "tree_both_off", "retile_alone": "retile_alone_tree",
             "dbuf_alone": "dbuf_alone", "composed": "composed"}[label]]
    summary = {
        "e156_arm_combination_digests":
            json.dumps(cg["e156_arm_combination_digests"], indent=1),
        "e156_compile_gate_evidence": (
            "All four arm cells build and are distinct. The verdict "
            "compose_forced_by_tgp_limit is compiler-confirmed, not just "
            "arithmetic: the unguarded standalone double buffer is REFUSED at "
            "the float cell by the named capacity assert "
            f"({cg['e156_standalone_dbuf_refusal_names_capacity']}) while the "
            "same source still builds the scored bfloat16 cell "
            f"({cg['e156_standalone_dbuf_scored_cell_still_compiles']}), so the "
            "refusal is that cell's byte count and not a blanket breakage. "
            "Defeating the capacity predicate is refused by name "
            f"({cg['e156_capacity_assert_named_in_refusal']}) and shrinking the "
            "limit is refused by the arm-liveness assert "
            f"({cg['e156_arm_liveness_named_in_refusal']}), so neither guard is "
            "decoration. At group_size 32 the retile is inert "
            f"({cg['e156_g32_retile_is_inert']}) and the composed tree is "
            "digest-identical to the double-buffer-alone build "
            f"({cg['e156_g32_composed_equals_dbuf_alone']}). Untouched call "
            f"sites are byte-identical ({cg['e156_untouched_sites_identical']}). "
            "AIR is identity evidence only; nothing is priced from it."
        ),
    }
    return metrics, summary


def main() -> None:
    audit = load(HERE / "e156-compose-audit.json")
    proof = load(HERE / "e156-barrier-proof.json")
    cgate = load(HERE / "e156-compile-gate.json")
    ulp = load(HERE / "e156-k-order-ulp-control.json")
    pshare = load(HERE / "e156-prefill-qmm-share.json")
    gate = load(OUT / "e156-gate-chain.json")
    submit = load(OUT / "e156-local-submit.json")

    metrics: dict[str, float | int] = {
        # The prefill effect on the crown base is NOT measurable on this host.
        # It is published as a string in the summary, never as a number.
        "e156_prefill_prior_pct": PREFILL_PRIOR_PCT,
        "e156_minimum_useful_effect_pct": MIN_USEFUL_EFFECT_PCT,
        "e156_published_pct_from_prior":
            PREFILL_PRIOR_PCT * PUBLISHED_PCT_PER_PREFILL_PCT,
        "e156_published_pct_at_minimum_useful_effect":
            MIN_USEFUL_EFFECT_PCT * PUBLISHED_PCT_PER_PREFILL_PCT,
        "e156_nax_executions_on_this_host": 0,
        "e156_host_gpu_arch_generation": 16,
        "e156_nax_required_arch_generation": 17,
    }
    summary: dict[str, str] = {
        "e156_prefill_pct_on_crown_base": "not_locally_measurable",
        "e156_prefill_pct_frame":
            "share of the seed-prefill phase, candidate minus base, sign "
            "convention NEGATIVE IS FASTER, harness=local",
        "e156_prefill_prior_pct_label": PREFILL_PRIOR_LABEL,
        "e156_published_pct_frame":
            "published median score points, prefill percentage multiplied by "
            f"{PUBLISHED_PCT_PER_PREFILL_PCT}, sign convention NEGATIVE "
            "PREFILL PERCENTAGE IS A GAIN, harness=ranked, this is a MODEL",
        "e156_air_pricing_policy":
            "no E156 number is priced from an AIR delta; on this kernel "
            "family AIR overstates the ISA change by 181.3 times",
        "e156_published_conversion_constant_discrepancy":
            f"This run uses {PUBLISHED_PCT_PER_PREFILL_PCT} published points "
            "per prefill percentage point. Advisor F3 uses 0.100436. The "
            "0.000002 difference is flagged rather than silently adopted; it "
            "moves no E156 conclusion, because the largest quantity it "
            "multiplies is the 4.12 prior and the resulting published-point "
            "change is about 8e-6.",
    }

    if audit is not None:
        m, s = compose_section(audit)
        metrics.update(m)
        summary.update(s)
    if proof is not None:
        m, s = barrier_section(proof)
        metrics.update(m)
        summary.update(s)
    if cgate is not None:
        m, s = compile_section(cgate)
        metrics.update(m)
        summary.update(s)
    if ulp is not None:
        m, s = ulp_section(ulp)
        metrics.update(m)
        summary.update(s)
    if pshare is not None:
        m, s = prefill_share_section(pshare)
        metrics.update(m)
        summary.update(s)
    if gate is not None:
        metrics["e156_gate_chain_all_green"] = float(gate["all_green"])
        summary["e156_gate_chain"] = json.dumps(gate["phases"], indent=1)
        m, s = budget_section(gate)
        metrics.update(m)
        summary.update(s)
    if submit is not None:
        metrics.update({
            f"e156_local_submit_{k}": v
            for k, v in submit.get("metrics", {}).items()
        })
        summary["e156_local_submit_note"] = submit.get("note", "")

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RESUME_ID,
        resume="allow",
        name="e156-composed-prefill-retile-and-double-buffer",
        job_type="analysis",
        tags=["e156", "alphonse", "nax", "prefill-retile", "double-buffer"],
        config={
            "experiment": "E156",
            "rung": "R1",
            "pr_number": PR_NUMBER,
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "crown_sha": CROWN_SHA,
            "budget_base_sha": BUDGET_BASE,
            "parked_dbuf_commit": PARKED_DBUF_COMMIT,
            "retile_only_commit": RETILE_ONLY_COMMIT,
            "worktree_head": git("rev-parse", "HEAD"),
            "host": HOST,
            "ranked_host": "m5-qwen38-27b-mtp (not this machine)",
            "harness": "local",
            "decode_tokens_local_submit": 512,
            "submitted_files": [
                "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
                "quantized_nax.h",
                "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp",
            ],
        },
    )
    run.log(metrics)
    run.summary.update(summary)

    if audit is not None:
        census = audit["e156_instantiation_census"]
        table = wandb.Table(columns=[
            "T", "BN", "instantiations", "dbuf_alone_bytes",
            "composed_alloc_bytes_max", "limit_bytes", "dbuf_alone_crosses",
            "composed_crosses", "composed_instantiations_disarmed_by_capacity"])
        for row in census["per_pair_table"]:
            table.add_data(
                row["T"], row["BN"], row["instantiations"],
                row["dbuf_alone_bytes"], row["composed_alloc_bytes_max"],
                row["limit_bytes"], row["dbuf_alone_crosses"],
                row["composed_crosses"],
                row["composed_instantiations_disarmed_by_capacity"])
        run.log({"e156_tgp_limit_crossing_table": table})

        budget = wandb.Table(columns=[
            "T", "retile", "dbuf", "group_size", "staged_BN",
            "retile_disarmed_by_loader_guard", "dbuf_armed",
            "dbuf_disarmed_by_capacity", "want_bytes", "alloc_bytes",
            "limit_bytes", "fits", "delta_vs_crown_bytes", "scored_cell"])
        for row in audit["e156_compose_tgp_budget_table"]:
            budget.add_data(
                row["T"], row["retile"], row["dbuf"], row["group_size"],
                row["staged_BN"], row["retile_disarmed_by_loader_guard"],
                row["dbuf_armed"], row["dbuf_disarmed_by_capacity"],
                row["want_bytes"], row["alloc_bytes"], row["limit_bytes"],
                row["fits"], row["delta_vs_crown_bytes"], row["scored_cell"])
        run.log({"e156_compose_tgp_budget_table": budget})

    art = wandb.Artifact("e156-evidence", type="analysis")
    for name in (
        "e156_compose_audit.py", "e156-compose-audit.json",
        "e156_barrier_proof.py", "e156-barrier-proof.json",
        "e156_compile_gate.py", "e156-compile-gate.json",
        "e156_wandb_log.py",
    ):
        path = HERE / name
        if path.exists():
            art.add_file(str(path))
    for name in ("e156-gate-chain.json", "e156-local-submit.json"):
        path = OUT / name
        if path.exists():
            art.add_file(str(path))
    run.log_artifact(art)

    print(run.id)
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
