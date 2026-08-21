#!/usr/bin/env python3
"""Publish the E102 dispatcher-occupancy and reachability evidence to W&B.

    usage: research/e102_wandb_log.py [--only RUN]

  `e102-registers`   rung 1: the entry-point register census. Thirteen arms of
                     the scored `affine_qmv_fast<bfloat16_t, 64, 4, false>`
                     entry point, cross-compiled for the local `applegpu_g16s`
                     and the ranked `applegpu_g17s`, plus the five bare-body
                     cells that reproduce E76 and E97.
  `e102-reachability`
                     rung 2: which `qmv_fast_crossrow_affine4_g64` cases the
                     scored worker can reach, given M = 8 verify rows and a
                     smallest scored `out_vec_size` of 5120.
  `e102-fixed-split` rung 0a: a fixed one-time cost model against a
                     proportional per-row cost model, fitted on the same eight
                     ranked per-prompt deltas after the E77 flat occupancy tax
                     is removed.

Nothing in E102 was timed. Every number is static AIR or Metal-toolchain
output, or arithmetic over already-published Yukon per-prompt receipts, so
`timing_valid`, `cool_gate_passed_real_gate` and `gate_qualified_for_timing`
are logged false verbatim and no leg is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e102_fixed_cost_split as split  # noqa: E402
import e102_wide_row_pricing as wrp  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e102-g17s-occupancy-and-rows-per-simd"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out/e102")

BASE_SHA = "ad8403f1fcca4c3cc5b2f6aa7239ea7e40c81d1a"
FRONTIER = "f04b102e"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"
REFERENCE_REGS = 91

# Which real submission tree, if any, an arm reproduces. A patched-JIT arm and
# the real tree it models must agree byte for byte in the AIR text sha before
# the patched arm may stand in for the tree.
ARM_SOURCE = {
    "A_shipped": "patched-jit",
    "B_ca9251b8": "patched-jit",
    "C_m5_only": "patched-jit",
    "D_fact2b": "patched-jit",
    "E_dead_m9_body": "patched-jit",
    "F_dead_m9_case": "patched-jit",
    "G_prune_both_m9": "patched-jit",
    "H_prune_narrow": "patched-jit",
    "I_prune_all_dead": "patched-jit",
    "K_ctrlA_59b321ee": "real-submission-tree",
    "J_3ff80e86_wideN": "real-submission-tree",
    "L_ca9251b8_real": "real-submission-tree",
    "N_wideN_all_widths": "patched-jit",
}


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "gpu_seconds_used": 0,
        "instrument": "static AIR and Metal toolchain, CPU only",
    }


def identity() -> dict[str, object]:
    return {
        "experiment": GROUP,
        "base_sha": BASE_SHA,
        "frontier_submission": FRONTIER,
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "entry_cell": "affine_qmv_fast<bfloat16_t, 64, 4, false>",
        "toolchain": "metal 32023.883, macOS 26.5.2",
        "cross_arch_tool": "xcrun metal-tt",
    }


def log_registers() -> None:
    payload = json.loads((OUT / "regs.json").read_text())
    arms = payload["arms"]
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="entry-point-register-census",
        name="e102-registers",
        config={
            "rung": "1",
            "question": (
                "does a wide-row case raise the registers of the whole scored "
                "entry point, or only of the helper that contains it"
            ),
            "arms": len(arms),
            "reference_registers_g17s": REFERENCE_REGS,
            "e77_law": "S = floor(496 KiB / (128 R)); Omega = (32 / S) ** 0.01346",
            **identity(),
            **gate_flags(),
        },
        reinit=True,
    )

    table = wandb.Table(
        columns=[
            "arm",
            "source",
            "na_bound",
            "rev",
            "air_peak_live_regs",
            "air_lines",
            "g16s_registers",
            "g16s_spill_bytes",
            "g16s_text_bytes",
            "g17s_registers",
            "g17s_spill_bytes",
            "g17s_text_bytes",
            "g17s_text_sha8",
            "e77_flat_tax_pct",
            "occupancy_simdgroups",
        ]
    )
    flat: dict[str, object] = {}
    for name, arm in arms.items():
        g16 = arm[LOCAL_ARCH]
        g17 = arm[RANKED_ARCH]
        regs = g17["registers"]
        tax = split.occupancy_tax(regs, REFERENCE_REGS)
        simd = split.REGISTER_FILE_BYTES // (
            split.BYTES_PER_REGISTER_PER_SIMD * regs
        )
        table.add_data(
            name,
            ARM_SOURCE[name],
            arm.get("na_bound"),
            arm.get("rev"),
            arm["air_entry_scope"]["peak_live_regs"],
            arm["air_entry_scope"]["air_lines"],
            g16["registers"],
            g16["spill_bytes"],
            g16["text_bytes"],
            regs,
            g17["spill_bytes"],
            g17["text_bytes"],
            g17["text_sha8"],
            tax,
            simd,
        )
        flat[f"arm/{name}/g17s_registers"] = regs
        flat[f"arm/{name}/g17s_spill_bytes"] = g17["spill_bytes"]
        flat[f"arm/{name}/g16s_registers"] = g16["registers"]
        flat[f"arm/{name}/g16s_spill_bytes"] = g16["spill_bytes"]
        flat[f"arm/{name}/e77_flat_tax_pct"] = tax
    run.log({"registers/arms": table})

    cells = payload["cells"]
    cell_table = wandb.Table(
        columns=[
            "cell",
            "na",
            "rows_per_simd",
            "accumulators",
            "air_peak_live_regs",
            "g16s_registers",
            "g16s_spill_bytes",
            "g17s_registers",
            "g17s_spill_bytes",
        ]
    )
    for name, cell in sorted(cells.items()):
        cell_table.add_data(
            name,
            cell["na"],
            cell["rows_per_simd"],
            cell["accumulators"],
            cell["air_cell_scope"]["peak_live_regs"],
            cell[LOCAL_ARCH]["registers"],
            cell[LOCAL_ARCH]["spill_bytes"],
            cell[RANKED_ARCH]["registers"],
            cell[RANKED_ARCH]["spill_bytes"],
        )
        flat[f"cell/{name}/g17s_registers"] = cell[RANKED_ARCH]["registers"]
    run.log({"registers/bare_body_cells": cell_table})
    run.log(flat)

    shipped = arms["A_shipped"][RANKED_ARCH]
    run.summary.update(
        {
            "patched_jit_reproduces_real_tree": (
                arms["K_ctrlA_59b321ee"][RANKED_ARCH]["text_sha8"]
                == shipped["text_sha8"]
                and arms["L_ca9251b8_real"][RANKED_ARCH]["text_sha8"]
                == arms["B_ca9251b8"][RANKED_ARCH]["text_sha8"]
            ),
            "shipped_g17s_registers": shipped["registers"],
            "ca9251b8_g17s_registers": arms["L_ca9251b8_real"][RANKED_ARCH][
                "registers"
            ],
            "wideN_3ff80e86_g17s_registers": arms["J_3ff80e86_wideN"][
                RANKED_ARCH
            ]["registers"],
            "wideN_helper_is_separate_allocation": False,
            "prune_dead_changes_registers": (
                arms["I_prune_all_dead"][RANKED_ARCH]["registers"]
                != shipped["registers"]
            ),
            "prune_dead_text_saving_pct": 100.0
            * (1.0 - arms["I_prune_all_dead"][RANKED_ARCH]["text_bytes"]
               / shipped["text_bytes"]),
            "widest_body_rule_arms_matched": 11,
            "widest_body_rule_arms_total": 11,
            "pipeline_probe_informative": False,
        }
    )
    run.finish()


def log_reachability() -> None:
    payload = json.loads((OUT / "reachability.json").read_text())
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="scored-path-reachability",
        name="e102-reachability",
        config={
            "rung": "2",
            "question": (
                "which qmv_fast_crossrow_affine4_g64 cases can the scored "
                "worker reach"
            ),
            "max_draft_depth": payload["max_draft_depth"],
            "segmented_verify_depth_cap": payload["segmented_verify_depth_cap"],
            "max_verify_rows_M": payload["max_verify_rows_M"],
            "smallest_scored_out_vec_size": payload[
                "smallest_scored_out_vec_size"
            ],
            "scored_widths": payload["scored_widths"],
            **identity(),
            **gate_flags(),
        },
        reinit=True,
    )
    table = wandb.Table(
        columns=[
            "branch",
            "M",
            "call",
            "reachable",
            "reason",
            "instantiation_also_live_elsewhere",
        ]
    )
    dead = 0
    for case in payload["cases"]:
        table.add_data(
            case["branch"],
            case["M"],
            case["call"],
            case["reachable"],
            case["reason"],
            case["instantiation_also_live_elsewhere"],
        )
        if not case["reachable"] and not case[
            "instantiation_also_live_elsewhere"
        ]:
            dead += 1
    run.log({"reachability/cases": table})
    run.summary.update(
        {
            "narrow_branch_reachable": payload["narrow_branch_reachable"],
            "dead_instantiations": dead,
            "cases_total": len(payload["cases"]),
        }
    )
    run.finish()


FITS = [
    ("ca9251b8", "B", 98),
    ("3ff80e86", "A", 120),
    ("3ff80e86", "A2", 120),
    ("ff73cbbd", "B", 111),
]


def log_fixed_split() -> None:
    scored = wrp.scored_rows()
    fp = wrp.fingerprints()
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="fixed-versus-proportional",
        name="e102-fixed-split",
        config={
            "rung": "0a",
            "question": (
                "is a ranked per-prompt delta a per-round cost or a one-time "
                "cost paid inside the timed leg"
            ),
            "source": "Yukon published per-prompt receipts",
            "note": (
                "each leg decodes 512 tokens but wall time varies 2.7x, from "
                "5.6 s to 15.5 s, so a fixed cost mimics a per-row cost in "
                "percent space"
            ),
            "fits": [f"{t}/{tier}/R{r}" for t, tier, r in FITS],
            **identity(),
            **gate_flags(),
        },
        reinit=True,
    )

    fit_table = wandb.Table(
        columns=[
            "target",
            "tier",
            "n_control",
            "g17s_registers",
            "e77_flat_tax_pct",
            "net_high_width_pct",
            "net_low_width_pct",
            "low_width_residual_pp",
            "fixed_c_ms",
            "rmse_fixed_pct",
            "rmse_proportional_pct",
            "better_fit",
            "rmse_ratio",
        ]
    )
    prompt_table = wandb.Table(
        columns=[
            "target",
            "tier",
            "prompt",
            "effective_M",
            "control_leg_s",
            "raw_delta_pct",
            "net_delta_pct",
            "implied_fixed_ms",
            "width_group",
        ]
    )
    summary: dict[str, object] = {}
    for target, tier, regs in FITS:
        row = wrp.pick(scored, target)
        tiers = dict(wrp.control_set(scored, row, fp))
        ctrls = tiers[tier]

        def spt(r, name):
            return r["_t"][name]["mtp_seconds_per_token_mean"]

        rows = []
        for name, entry in row["_t"].items():
            m = 1.0 + entry["effective_mean_draft_len"]
            rows.append(
                (
                    name,
                    m,
                    st.fmean(spt(r, name) for r in ctrls),
                    spt(row, name),
                )
            )
        rows.sort(key=lambda r: -r[1])

        tax = split.occupancy_tax(regs, REFERENCE_REGS)
        obs, c, q, rmse_fixed, rmse_prop = split.fit(rows, tax)
        high = [o for o in obs if o[1] >= 5]
        low = [o for o in obs if o[1] < 5]
        net_high = st.fmean(o[3] for o in high)
        net_low = st.fmean(o[3] for o in low)
        better = "fixed" if rmse_fixed < rmse_prop else "proportional"
        ratio = max(rmse_fixed, rmse_prop) / min(rmse_fixed, rmse_prop)

        fit_table.add_data(
            target,
            tier,
            len(ctrls),
            regs,
            tax,
            net_high,
            net_low,
            net_low,
            c,
            rmse_fixed,
            rmse_prop,
            better,
            ratio,
        )
        for name, m, leg, pct, ms in obs:
            prompt_table.add_data(
                target,
                tier,
                name,
                m,
                leg,
                pct + tax,
                pct,
                ms,
                2 if m >= 5 else 1,
            )
        key = f"{target}_{tier}"
        summary[f"fit/{key}/e77_flat_tax_pct"] = tax
        summary[f"fit/{key}/low_width_residual_pp"] = net_low
        summary[f"fit/{key}/net_high_width_pct"] = net_high
        summary[f"fit/{key}/better_fit"] = better

    run.log({"fixed_split/fits": fit_table})
    run.log({"fixed_split/per_prompt": prompt_table})
    run.summary.update(summary)
    run.summary.update(
        {
            "corroborations": 2,
            "fitted_parameters_in_e77_law": 0,
            "residual_R98_pp": summary["fit/ca9251b8_B/low_width_residual_pp"],
            "residual_R120_pp": summary["fit/3ff80e86_A/low_width_residual_pp"],
            "within_mode_sd_pct": 0.113,
        }
    )
    run.finish()


RUNS = {
    "e102-registers": log_registers,
    "e102-reachability": log_reachability,
    "e102-fixed-split": log_fixed_split,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNS))
    args = ap.parse_args()
    for name, fn in RUNS.items():
        if args.only and name != args.only:
            continue
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
