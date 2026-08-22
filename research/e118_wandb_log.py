#!/usr/bin/env python3
"""Publish the E118 wide-qmv metadata-load instruction screen to W&B.

    usage: research/e118_wandb_log.py [--only RUN]

  `e118-arms`           the isolated Metal arms at NA = 2, 3, 4 and 5 over five
                        scored shapes, the primary metric and the discriminator.
  `e118-static-budget`  AIR device loads, registers, spill bytes and ISA text
                        for the local `applegpu_g16s` and the ranked
                        `applegpu_g17s`.
  `e118-spill-defect`   the NA=5 exactness failure and the `z_ballast` control
                        that attributes it to spilling.

Every timed leg here is a standalone Metal microbenchmark. It holds no model
and runs no benchmark wrapper, so it passes no thermal gate. Each run logs
`cool_gate_passed_real_gate` and `gate_qualified_for_timing` verbatim as
false, and no leg here is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e118-wide-qmv-metadata-load-instruction-screen"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
ART = pathlib.Path("research/e118-artifacts")

PR = 120
ASSIGNMENT_BASE_SHA = "1d2320bece29cddc94b95e5f99f00331b05a5025"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
KILL_RULE_PCT = 0.5

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e118-arms": "e118arms1",
    "e118-static-budget": "e118stat1",
    "e118-spill-defect": "e118spil1",
    "e118-cost-model": "e118cost1",
    "e118-rung2-finding53": "e118rng21",
    "e118-sumshoist-ceiling": "e118hst01",
}


def summary() -> dict:
    return json.loads((ART / "summary.json").read_text())


def meta() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ART / "meta.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def start(job_type: str, name: str, config: dict[str, object]):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name, id=RUN_IDS.get(name), resume="allow", config=config,
        reinit=True)


def gate_flags(instrument: str, timing_valid: bool) -> dict[str, object]:
    return {
        "timing_valid": timing_valid,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "instrument": instrument,
    }


def identity() -> dict[str, object]:
    m = meta()
    return {
        "experiment": GROUP,
        "pr": PR,
        "assignment_base_sha": ASSIGNMENT_BASE_SHA,
        "leg_commit": m.get("git_head"),
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": "Apple M4 Pro",
        "metal_version": m.get("metal_version"),
        "swift_version": m.get("swift_version"),
        "entry_points": "e118_iso_na2..na5",
        "transcribed_from": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h qmv_fast_crossrow_affine4_g64_wide"
        ),
        "candidate_files_changed": 0,
        "fast_math": False,
    }


def log_arms() -> None:
    doc = summary()
    run = start(
        job_type="isolated-metal-arms", name="e118-arms",
        config={
            "question": (
                "does removing or repacking the per-group metadata loads in "
                "the wide qmv inner loop make the kernel faster"
            ),
            "leg_command": (
                "research/e118_probe.sh e118-full2 --shapes 0,1,2,3,4 "
                "--widths 2,3,4,5 --pairs 8 --samples 24"
            ),
            "shapes": list(doc["per_shape"].get("g_pack32", {}).keys()),
            "headline_shape": doc["shape"],
            "widths": doc["widths"],
            "pairs": doc["pairs"],
            "warmup_blocks": doc["warmup_blocks"],
            "standing_weights": doc["standing_weights"],
            "kill_rule_pct": KILL_RULE_PCT,
            "sign_convention": doc["sign_convention"],
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", True),
        },
    )

    weighted = doc["weighted"]
    table = wandb.Table(columns=[
        "arm", "role", "round_weighted_pct_faster", "na2_pct", "na3_pct",
        "na4_pct", "na5_pct", "identified_local_lo", "identified_local_hi",
        "identified_ranked_lo", "identified_ranked_hi"])
    for arm, row in sorted(weighted.items(),
                           key=lambda kv: -kv[1]["standing_pct"]):
        na = row["na"]
        loc = row.get("identified_local") or [None, None]
        rnk = row.get("identified_ranked") or [None, None]
        table.add_data(arm, row["role"], row["standing_pct"],
                       na.get("2"), na.get("3"), na.get("4"), na.get("5"),
                       loc[0], loc[1], rnk[0], rnk[1])
    run.log({"arms_round_weighted": table})

    abs_us = wandb.Table(columns=["cell", "median_us"])
    for cell, value in sorted(doc["absolute_us"].items()):
        abs_us.add_data(cell, value)
    run.log({"absolute_microseconds": abs_us})

    shape_tbl = wandb.Table(columns=["arm", "shape", "round_weighted_pct"])
    for arm, per in doc["per_shape"].items():
        for shape, row in per.items():
            shape_tbl.add_data(arm, shape, row["weighted_pct"])
    run.log({"per_shape_round_weighted": shape_tbl})

    gaps = wandb.Table(columns=["arm", "na", "forward_reverse_gap_pct"])
    for key, value in sorted(doc["forward_reverse_gap_pct"].items()):
        arm, _, na = key.partition("|")
        gaps.add_data(arm, na, value)
    run.log({"defect16_forward_reverse_gap": gaps})

    bias = doc.get("bias_axis")
    if bias:
        bias_tbl = wandb.Table(columns=[
            "quantity", "na2_pct", "na3_pct", "na4_pct", "na5_pct",
            "round_weighted_pct"])
        for key in ("whole_bias_axis", "bias_arithmetic", "bias_load",
                    "bias6_ceiling", "bias6_real", "bias6_reconstruction"):
            row = bias[key]
            na = row["na"]
            bias_tbl.add_data(key, na.get("2"), na.get("3"), na.get("4"),
                              na.get("5"), row["weighted"])
        run.log({"e111_bias_axis_by_width": bias_tbl})

    pm = doc["primary_metric"]
    disc = doc["discriminator"]
    best = weighted[pm["arm"]]
    ident_local = best.get("identified_local") or [None, None]
    ident_ranked = best.get("identified_ranked") or [None, None]
    run.summary.update({
        "primary_metric_name": pm["name"],
        "e118_best_bit_exact_arm_round_weighted_pct_faster_vs_a_base":
            pm["value"],
        "primary_metric_arm": pm["arm"],
        "kill_rule_pct": pm["kill_rule_pct"],
        "kill_rule_cleared": pm["cleared"],
        "primary_metric_identified_local_lo": ident_local[0],
        "primary_metric_identified_local_hi": ident_local[1],
        "primary_metric_identified_ranked_lo": ident_ranked[0],
        "primary_metric_identified_ranked_hi": ident_ranked[1],
        "kill_rule_cleared_anywhere_in_identified_set":
            ident_local[1] is not None and ident_local[1] >= KILL_RULE_PCT,
        "discriminator_verdict": disc["verdict"],
        "discriminator_s_bcast_pct": disc["s_bcast"],
        "discriminator_s_bcast_all_pct": disc["s_bcast_all"],
        "discriminator_p_split_meta_pct": disc["p_split_meta"],
        "discriminator_n_nosums_pct": disc["n_nosums"],
        "finding44_round_weighted_gap_pct":
            doc["finding44"]["round_weighted_gap_pct"],
        "exact_failure_count": len(doc["fidelity"]["exact_failures"]),
        "positive_control_failure_count":
            len(doc["fidelity"]["control_failures"]),
        "screen_weakened_cells":
            len(doc["fidelity"].get("screen_weakened_by_nonfinite", [])),
        "conclusion": (
            "The shipped wide qmv inner loop already issues only 7 device "
            "loads per entry point, so the 8 scalar metadata reads are "
            "already coalesced by the front end. Broadcasting them costs "
            "registers and shuffles and loses. Removing one load with a "
            "packed scale-bias word gains %+.3f %%, below the %+.2f %% kill "
            "rule. The load-issue port is not the binding resource."
            % (pm["value"], KILL_RULE_PCT)),
    })
    run.finish()


def log_static() -> None:
    doc = summary()
    census = doc["census"]
    run = start(
        job_type="static-budget", name="e118-static-budget",
        config={
            "question": (
                "how many device loads, registers, spill bytes and ISA text "
                "bytes does each arm cost on the local and the ranked "
                "architecture"
            ),
            "leg_command": (
                "research/e118_arms.py --emit DIR && "
                "research/e118_arms.py --census DIR --out census.json"
            ),
            "widths": census["widths"],
            **identity(),
            **gate_flags("offline metal-arch translation on CPU, no GPU",
                         False),
        },
    )

    air = wandb.Table(columns=["arm", "na", "device_loads", "shuffles"])
    static = wandb.Table(columns=[
        "arm", "arch", "na", "registers", "spill_bytes", "text_bytes"])
    for arm, row in census["arms"].items():
        for na in census["widths"]:
            cell = row["air"].get(str(na), {})
            air.add_data(arm, na, cell.get("device_loads"),
                         cell.get("shuffles", 0))
            for arch in (census["local_arch"], census["ranked_arch"]):
                v = (row.get(arch) or {}).get(str(na))
                if v is None:
                    continue
                static.add_data(arm, arch, na, v["registers"],
                                v["spill_bytes"] or 0, v["text_bytes"])
    run.log({"air_device_loads": air, "registers_spill_text": static})

    base_loads = census["arms"]["a_base"]["air"]["5"]["device_loads"]
    run.summary.update({
        "shipped_device_loads_per_entry_point": base_loads,
        "conclusion": (
            "a_base already issues %d device loads. The eight scalar metadata "
            "reads are coalesced before any arm touches them, so the "
            "broadcast arms remove zero loads and only add registers and "
            "shuffles." % base_loads),
    })
    run.finish()


def log_spill_defect() -> None:
    doc = summary()
    se = doc.get("spill_exactness")
    fid = doc["fidelity"]
    run = start(
        job_type="exactness-defect", name="e118-spill-defect",
        config={
            "question": (
                "is the NA=5 bit-exactness failure caused by an arm's "
                "mechanism or by the compiler spilling"
            ),
            "control": (
                "z_ballast adds 12 dead loop-carried floats that are consumed "
                "only inside a branch that never runs, so no value it "
                "computes reaches y. It changes the spill budget and nothing "
                "else."
            ),
            "reference": (
                "every arm is scored against an exact affine-4 reference "
                "evaluated in double on the CPU"
            ),
            **identity(),
            **gate_flags("standalone Metal microbenchmark, one GPU", False),
        },
    )

    failures = wandb.Table(columns=[
        "arm", "shape", "na", "differing", "total", "max_ulp", "max_rel",
        "vs_double_max_rel"])
    for row in fid["exact_failures"]:
        failures.add_data(row["arm"], row["shape"], row["m"], row["differing"],
                          row["total"], row["max_ulp"], row["max_rel"],
                          row.get("vs_double_max_rel"))
    run.log({"exactness_failures": failures})

    base_ref = wandb.Table(columns=["shape", "na", "base_vs_double_max_rel",
                                    "base_vs_double_rms_over_signal"])
    for row in fid["base_vs_double"]:
        base_ref.add_data(row["shape"], row["m"], row["max_rel"], row["rms"])
    run.log({"base_against_double_reference": base_ref})

    if se:
        join = wandb.Table(columns=[
            "arm", "na", "spill_bytes", "registers", "exact",
            "vs_double_max_rel"])
        for arm, by_na in sorted(se["arms"].items()):
            for na, cell in sorted(by_na.items()):
                join.add_data(arm, int(na), cell["spill_bytes"],
                              cell["registers"], cell["exact"],
                              cell["vs_double_max_rel"])
        run.log({"spill_against_exactness": join})
        zb = se["arms"].get("z_ballast", {})
        zwrong = sorted(int(na) for na, c in zb.items() if not c["exact"])
        run.summary.update({
            "arch": se["arch"],
            "max_spill_while_exact_bytes": se["max_spill_while_exact"],
            "min_spill_while_wrong_bytes": se["min_spill_while_wrong"],
            "spill_separates_exact_from_wrong": se["separates"],
            "z_ballast_wrong_at_na": zwrong,
            "conclusion": (
                "z_ballast changes no arithmetic that reaches y and is still "
                "wrong at NA %s, so the NA=5 divergence is caused by the "
                "compiler spilling, not by any arm's mechanism. On this host "
                "and toolchain, keep the wide qmv NA=5 entry point at or "
                "below %s B of spill and check exactness whenever it spills."
                % (zwrong, se["max_spill_while_exact"])),
        })
    run.finish()


def log_cost_model() -> None:
    """The instruction price ladder, which is the transferable result here.

    The screen answers one question about one mechanism. This run answers
    "what does one instruction of this class cost in this kernel", which any
    later experiment can apply to an arm before spending GPU time on it.
    """
    doc = summary()
    model = doc.get("cost_model")
    if not model:
        print("no cost_model in summary.json, skipping")
        return
    run = start(
        job_type="instruction-cost-model", name="e118-cost-model",
        config={
            "question": (
                "what does one device load, one ALU instruction and one "
                "simd_shuffle cost in the wide affine-4 qmv inner loop"
            ),
            "method": (
                "arms that inject an exactly known count of one instruction "
                "class per k-block iteration while holding the register "
                "footprint fixed, so the slope is measured and not fitted"
            ),
            "unit": model["unit"],
            **identity(),
            **gate_flags("standalone Metal probe, ungated local timing", False),
        },
    )

    prices = wandb.Table(columns=[
        "class", "pct_per_instruction", "sem", "r2_median", "cells",
        "us_per_instruction", "scaffold_pct_cancelled",
        "biased_three_point_slope"])
    for klass, c in model["classes"].items():
        prices.add_data(klass, c["pct_per_instruction_median"], c["sem"],
                        c["r2_median"], c["cells"],
                        c.get("us_per_instruction_median"),
                        c.get("scaffold_pct_median"),
                        c.get("slope_three_point_median"))

    # The single biggest finding of the ladder is that the price is not
    # constant in NA, so it gets its own table rather than a summary scalar.
    per_width = wandb.Table(columns=[
        "class", "na", "pct_per_instruction", "us_per_instruction",
        "r2_median", "cells"])
    for klass, c in model["classes"].items():
        for na, row in sorted(c.get("per_na", {}).items()):
            per_width.add_data(klass, int(na),
                               row.get("pct_per_instruction_median"),
                               row.get("us_per_instruction_median"),
                               row.get("r2_median"), row.get("cells"))

    percell = wandb.Table(columns=["class", "shape", "na", "slope", "r2"])
    for klass, fits in model.get("per_cell", {}).items():
        for f in fits:
            percell.add_data(klass, f["shape"], f["m"], f["slope"], f["r2"])

    # The prediction is made per width, because the price is per width. A
    # width with no measured price for a class the arm moves is dropped
    # rather than predicted from the pooled price, and `widths_dropped`
    # plus `standing_weight_covered` say exactly how much of the standing
    # weight the prediction is allowed to speak for.
    pred = wandb.Table(columns=[
        "arm", "na", "delta_device_loads", "delta_shuffles",
        "predicted_pct_faster", "measured_pct_faster", "residual_pp",
        "shapes"])
    coverage = wandb.Table(columns=[
        "arm", "delta_device_loads", "delta_shuffles",
        "weighted_predicted_over_covered", "weighted_measured_over_covered",
        "weighted_residual_pp_over_covered", "standing_weight_covered",
        "widths_ok", "widths_dropped", "complete_over_standing_weights"])
    for arm, p in sorted(model.get("screen_prediction", {}).items()):
        for na, row in sorted(p.get("per_na", {}).items()):
            if row.get("status") != "ok":
                continue
            pred.add_data(arm, int(na), p["delta_device_loads"],
                          p["delta_shuffles"], row["predicted_pct_faster"],
                          row["measured_pct_faster"], row["residual_pct"],
                          row["shapes"])
        wm = p.get("weighted_measured_over_covered")
        wp = p.get("weighted_predicted_over_covered")
        coverage.add_data(
            arm, p["delta_device_loads"], p["delta_shuffles"], wp, wm,
            (wm - wp) if (wm is not None and wp is not None) else None,
            p.get("standing_weight_covered"),
            ",".join(str(w) for w in p.get("widths_ok", [])),
            ",".join(str(w) for w in p.get("widths_dropped", [])),
            p.get("complete_over_standing_weights"))

    run.log({"instruction_price": prices, "price_per_width": per_width,
             "price_per_cell": percell, "screen_prediction": pred,
             "screen_prediction_coverage": coverage})

    classes = model["classes"]
    updates: dict[str, object] = {}
    for klass in ("ld", "alu", "shuf"):
        if klass in classes:
            c = classes[klass]
            updates["pct_per_instruction_" + klass] = \
                c["pct_per_instruction_median"]
            updates["sem_" + klass] = c["sem"]
            updates["r2_" + klass] = c["r2_median"]
            updates["us_per_instruction_" + klass] = \
                c.get("us_per_instruction_median")
            # The three-point slope uses the scaffolded low anchor, so the
            # gap between it and the headline slope is the scaffolding bias
            # that the scaffold-cancelled fit removes.
            updates["biased_three_point_slope_" + klass] = \
                c.get("slope_three_point_median")
    ilp = model.get("ilp_control")
    if ilp:
        updates["ilp_two_chain_pct_per_instruction"] = \
            ilp["two_chain_pct_per_instruction"]
        updates["ilp_four_chain_pct_per_instruction"] = \
            ilp["four_chain_pct_per_instruction"]
        # Four independent chains cost slightly MORE than two, so the ALU
        # price is issue throughput and not dependency latency.
        updates["ilp_four_minus_two_pct"] = ilp["four_minus_two_pct"]
        updates["ilp_reads"] = ilp["reads"]
    if "shuf" in classes and "ld" in classes:
        updates["shuffle_over_load_price_ratio"] = (
            classes["shuf"]["pct_per_instruction_median"] /
            classes["ld"]["pct_per_instruction_median"])
    run.summary.update(updates)
    run.finish()


def log_rung2() -> None:
    """Finding 53, decomposed into mechanism, scaffolding and exchange.

    The registered ceiling arm wraps a uniform branch around a duplicated `i`
    loop, so it prices the mechanism together with the scaffolding the
    mechanism needs. `n_halfsums_free` drops the same half of the add tree at
    compile time in both simdgroups, which is the mechanism with no branch and
    no duplication, so the two differences below are identified with no fitted
    parameter.
    """
    doc = summary()
    d = doc.get("rung2_finding53")
    if not d:
        print("no rung2_finding53 in summary.json, skipping")
        return
    verdict = doc.get("rung2_exact_verdict", {})
    run = start(
        job_type="rung2-reduction-sharing", name="e118-rung2-finding53",
        config={
            "question": (
                "both simdgroups of a threadgroup compute the identical "
                "`sums` reduction; is sharing it across the two simdgroups "
                "worth the exchange it costs"
            ),
            "method": (
                "a scaffolding-free ceiling arm, a full-duplication ceiling "
                "arm, and three bit-exact exchange arms, all in one "
                "counterbalanced session with the instruction-price ladder"
            ),
            "rule": (
                "the registered rule was: stop unless the registered ceiling "
                "n_halfsums reaches +2.5 %% at NA=4. It reached %+.3f %% and "
                "the rule fired. Rung 2b is an UNREGISTERED follow-up that "
                "first de-confounds the ceiling arm from its scaffolding."
                % next((r["registered_ceiling_pct"] for r in d["rows"]
                        if r["na"] == 4), float("nan"))
            ),
            "shape": d["shape"],
            "note": d["note"],
            **identity(),
            **gate_flags("standalone Metal probe, ungated local timing", False),
        },
    )

    decomp = wandb.Table(columns=[
        "na", "mechanism_ceiling_pct", "registered_ceiling_pct",
        "body_duplication_cost_pp", "min_exchange_pct", "exchange_cost_pp",
        "split_pct", "owner_pct", "captured_fraction"])
    for r in d["rows"]:
        decomp.add_data(r["na"], r["mechanism_ceiling_pct"],
                        r["registered_ceiling_pct"],
                        r["body_duplication_cost_pp"], r["min_exchange_pct"],
                        r["exchange_cost_pp"], r["split_pct"], r["owner_pct"],
                        r["captured_fraction"])

    occ = wandb.Table(columns=[
        "arm", "na", "threadgroup_bytes", "threadgroup_budget_bytes",
        "threadgroups_allowed_by_shared_memory", "air_threadgroup_ops",
        "air_total_instructions", "g16s_registers", "g16s_spill_bytes",
        "g16s_text_bytes", "g17s_registers", "g17s_spill_bytes",
        "g17s_text_bytes"])
    for c in d["occupancy"]:
        g16 = c.get("applegpu_g16s", {})
        g17 = c.get("applegpu_g17s", {})
        occ.add_data(c["arm"], c["na"], c["threadgroup_bytes"],
                     c["threadgroup_budget_bytes"],
                     c["threadgroups_allowed_by_shared_memory"],
                     c["air_threadgroup_ops"], c["air_total_instructions"],
                     g16.get("registers"), g16.get("spill_bytes"),
                     g16.get("text_bytes"), g17.get("registers"),
                     g17.get("spill_bytes"), g17.get("text_bytes"))

    run.log({"finding53_decomposition": decomp, "occupancy_budget": occ})

    # The best exchange arm spills 160 B at NA=5 and loses 31 % there, which
    # drags the standing average below the bar on its own. Both readings are
    # published: the honest all-width one that decides the bar, and the
    # spill-excluded one with the weight it actually covers, so a later
    # experiment can see the mechanism is worth about +1.1 % where it fits.
    updates: dict[str, object] = {
        "rung2_best_exact_arm": verdict.get("best_arm"),
        "rung2_best_exact_round_weighted_pct": verdict.get("standing_pct"),
        "rung2_bar_pct": verdict.get("bar_pct"),
        "rung2_bar_cleared": verdict.get("cleared"),
        "rung2_counts_in_primary_metric": verdict.get("in_primary_metric"),
        "rung2_standing_pct_excluding_local_spill":
            verdict.get("standing_pct_excluding_local_spill"),
        "rung2_excluding_local_spill_coverage":
            verdict.get("excluding_local_spill_coverage"),
        "rung2_dropped_widths": json.dumps(verdict.get("dropped_widths", {})),
    }
    for r in d["rows"]:
        for key in ("mechanism_ceiling_pct", "registered_ceiling_pct",
                    "body_duplication_cost_pp", "min_exchange_pct",
                    "exchange_cost_pp", "captured_fraction"):
            if r.get(key) is not None:
                updates["na%d_%s" % (r["na"], key)] = r[key]
    run.summary.update(updates)
    run.finish()


def log_hoist() -> None:
    """Feedback 3's whole-table hoist, and why it is a ceiling and not an arm.

    `sums` carries no `out_row`, no `simd_gid` and no `tid.y`, so the whole
    table is recomputed once per simdgroup per threadgroup in y. Replacing the
    computation with a load from a precomputed table is bit exact and prices
    the entire redundancy. It is logged separately from the primary metric
    because the delivery route is not editable from `research/`.
    """
    doc = summary()
    hoist = doc.get("hoist_verdict") or {}
    d = hoist.get("x_sumshoist")
    if not d:
        print("no hoist_verdict in summary.json, skipping")
        return
    run = start(
        job_type="sums-table-hoist", name="e118-sumshoist-ceiling",
        config={
            "question": (
                "the `sums` reduction is recomputed once per simdgroup per "
                "threadgroup in y -- 8704 times at mlp.gate_up N=34816. What "
                "is a bit-exact, load-paying removal of the whole redundancy "
                "worth, and how much of the free `n_nosums` ceiling does it "
                "capture"
            ),
            "method": (
                "a tenth device buffer holding a [k_block][lane][m] table with "
                "the per-lane stride padded to 8 floats, filled by a Metal "
                "kernel that writes the identical expression tree in the "
                "identical two precisions, read as one vec<float,4> at NA<=4 "
                "and vec<float,4> plus one scalar at NA=5"
            ),
            "not_shippable_reason": (
                "the host-side buffer binding lives in quantized.cpp, which "
                "is not in editablePaths. This is a ceiling measurement for a "
                "delivery decision, not an arm."
            ),
            "excludes_table_production": True,
            **identity(),
            **gate_flags("standalone Metal probe, ungated local timing", False),
        },
    )

    per_na = wandb.Table(columns=[
        "na", "sumshoist_pct_faster", "n_nosums_ceiling_pct",
        "capture_of_ceiling", "load_price_pp"])
    for na in sorted(d["na"], key=int):
        h = d["na"][na]
        c = d["nosums_na"].get(na)
        per_na.add_data(int(na), h, c,
                        d["capture_of_ceiling_na"].get(na)
                        if "capture_of_ceiling_na" in d
                        else d["capture_of_nosums_na"].get(na),
                        (h - c) if c is not None else None)

    # The per-width numbers above deliberately exclude the cost of building
    # the table. Charging the whole fill to a single dispatch is the harshest
    # possible accounting and it is what decides the axis, so it is logged
    # next to the gross number rather than in prose.
    absu = doc.get("absolute_us", {})
    weights = {int(k): v for k, v in doc.get("standing_weights", {}).items()}
    fill = wandb.Table(columns=[
        "shape", "table_bytes", "fill_us", "max_m", "na",
        "a_base_us", "sumshoist_us", "gross_pct_faster",
        "net_pct_faster_including_fill"])
    net_weighted: dict[str, float] = {}
    for row in doc.get("sums_table", []):
        shape = row["shape"]
        acc, wsum = 0.0, 0.0
        for na in sorted(weights):
            cell = absu.get("%s|NA%d" % (shape, na))
            if not cell or "a_base" not in cell or "x_sumshoist" not in cell:
                continue
            base, hoisted = cell["a_base"], cell["x_sumshoist"]
            gross = (base - hoisted) / base * 100.0
            net = (base - (hoisted + row["fill_us"])) / base * 100.0
            fill.add_data(shape, row["table_bytes"], row["fill_us"],
                          row["max_m"], na, base, hoisted, gross, net)
            acc += weights[na] * net
            wsum += weights[na]
        if wsum:
            net_weighted[shape] = acc / wsum

    run.log({"sumshoist_per_width": per_na, "table_production_cost": fill})
    updates: dict[str, object] = {
        "sumshoist_round_weighted_pct": d["standing_pct"],
        "n_nosums_ceiling_round_weighted_pct": d["nosums_standing_pct"],
        "capture_of_ceiling_weighted": d["capture_of_nosums_weighted"],
        "shippable_from_research": d["shippable_from_research"],
        "excludes_table_production": d["excludes_table_production"],
        "counts_in_primary_metric": False,
    }
    for shape, val in net_weighted.items():
        updates["net_round_weighted_pct_including_fill|" + shape] = val
    # One fill is one full command-buffer round trip, so these net numbers
    # are an upper bound on the true production cost, and they are also the
    # cost of building the table once for one dispatch. A table shared over
    # many matvecs with the same x and K amortises it.
    updates["fill_accounting"] = (
        "whole fill charged to one dispatch; upper bound on true cost")
    run.summary.update(updates)
    run.finish()


RUNS = {
    "e118-arms": log_arms,
    "e118-static-budget": log_static,
    "e118-spill-defect": log_spill_defect,
    "e118-cost-model": log_cost_model,
    "e118-rung2-finding53": log_rung2,
    "e118-sumshoist-ceiling": log_hoist,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only in (None, name):
            print(f"== {name}")
            fn()


if __name__ == "__main__":
    main()
