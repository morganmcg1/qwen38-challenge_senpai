#!/usr/bin/env python3
"""Publish the E111 one-byte affine-4 bias recoding evidence to W&B.

    usage: research/e111_wandb_log.py [--only RUN]

  `e111-rung0-recode`   the zero-GPU lossless-recoding census over every
                        scored group of all seven scored tensors.
  `e111-rung1-arms`     the isolated Metal arms at NA = 2, 3, 4 and 5, with
                        the round-weighted decision.
  `e111-static-budget`  registers, spill bytes and ISA text for the local
                        `applegpu_g16s` and the ranked `applegpu_g17s`.

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

import e111_analyze as an

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e111-lossless-onebyte-affine4-bias-recode"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

PR = 113
ASSIGNMENT_BASE_SHA = "05321b0f5867e82d73c479638a5f9a1cf503ec2f"
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"

LOCAL_ROUND_US = 102864.0
PROMOTION_BAR_PCT = 0.20
DRAM_PEAK_GBPS = 273.0

# The census was run with --include-unscored so the embedding table could act
# as an out-of-family check. It is not on the scored decode path, so it is
# excluded from every scored total reported here.
UNSCORED_FAMILIES = {"embed_tokens"}

# Republishing must correct a run in place. A second run of the same evidence
# would leave two disagreeing records of one experiment.
RUN_IDS = {
    "e111-rung0-recode": "rpsoy9ih",
    "e111-rung1-arms": "25o3d9v0",
    "e111-static-budget": "5odxjqmn",
}

# Sessions, in the order they ran. The NA=4 replicate and the control rerun
# are separate sessions on purpose: rule 35 forbids deciding a sub-0.5 %
# arm from one pair, and the first pack control was too weak to fire.
SESSIONS = [
    ("e111-r1-gateup-na2", "mlp.gate_up", 2, "sweep"),
    ("e111-r1-gateup-na3", "mlp.gate_up", 3, "sweep"),
    ("e111-r1-gateup-na4", "mlp.gate_up", 4, "sweep"),
    ("e111-r1-gateup-na5", "mlp.gate_up", 5, "sweep"),
    ("e111-r1-gateup-na4-rep", "mlp.gate_up", 4, "replicate"),
    ("e111-r1-gateup-na4-ctl", "mlp.gate_up", 4, "control rerun"),
    ("e111-r1-down-na4", "mlp.down", 4, "cross-shape"),
    ("e111-r1-down", "mlp.down", 5, "pre-f1"),
    ("e111-r1-lmhead", "lm_head", 5, "pre-f1"),
    ("e111-r1-gateup", "mlp.gate_up", 5, "pre-f1"),
]
WEIGHTED_TAGS = ["e111-r1-gateup-na2", "e111-r1-gateup-na3",
                 "e111-r1-gateup-na4", "e111-r1-gateup-na5"]

ARM_ROLE = {
    "a_shipped": "line-aligned transcription of quantized.h:969-1063",
    "n_nobias": "bias load and its accumulation both deleted, 34 B",
    "n_nosums": "bias load kept, the sum accumulation deleted, 36 B",
    "d_bias1": "1-byte code replaces the bias, no reconstruction, 35 B",
    "e_bias6": "1-byte code replaces the bias, exact reconstruction, 35 B",
    "b_constw": "the 32 B weight load deleted, all arithmetic kept, 4 B",
    "c_loadonly": "every load kept, extract and fma deleted, 36 B",
    "g_pack32": "shipped values from one interleaved 32-bit record, 36 B",
}


def start(job_type: str, name: str, config: dict[str, object]):
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name, id=RUN_IDS.get(name), resume="allow", config=config,
        reinit=True)


def gate_flags(instrument: str, gpu_seconds: float,
               timing_valid: bool) -> dict[str, object]:
    return {
        "timing_valid": timing_valid,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
        "gpu_seconds_used": gpu_seconds,
        "instrument": instrument,
    }


def identity(base_sha: str) -> dict[str, object]:
    return {
        "experiment": GROUP,
        "pr": PR,
        "assignment_base_sha": ASSIGNMENT_BASE_SHA,
        "leg_base_sha": base_sha,
        "host": HOST,
        "local_arch": LOCAL_ARCH,
        "ranked_arch": RANKED_ARCH,
        "chip": "Apple M4 Pro",
        "os_build": "25F84",
        "metal_version": "Apple metal version 32023.883 (metalfe-32023.883)",
        "transcribed_from": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/"
            "quantized.h:969-1063 qmv_fast_crossrow_affine4_g64_wide"
        ),
        "candidate_files_changed": 0,
        "local_round_us": LOCAL_ROUND_US,
        "promotion_bar_pct_of_round": PROMOTION_BAR_PCT,
        "qmv_share_of_round": an.QMV_SHARE_OF_ROUND,
    }


def read_meta(tag: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in (OUT / tag / "meta.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        if _:
            meta[key] = value
    return meta


def log_rung0() -> None:
    doc = json.loads(
        pathlib.Path("research/e111-rung0-bias-recode.json").read_text())
    families = {name: row for name, row in doc["families"].items()
                if name not in UNSCORED_FAMILIES}

    run = start(
        job_type="lossless-recoding-census", name="e111-rung0-recode",
        config={
            "rung": "0",
            "question": (
                "is every stored affine-4 bias exactly bf16(-z * scale) for a "
                "unique 4-bit z, up to one bf16 ordinal of correction"
            ),
            "leg_command": "research/e111_rung0_bias_recode.py",
            "search_range_z": "0..63, wider than the 4 bits the code allows",
            **identity(ASSIGNMENT_BASE_SHA),
            **gate_flags("numpy over the checkpoint metadata, no GPU", 0.0,
                         False),
        },
    )

    table = wandb.Table(columns=[
        "family", "k", "n", "tensor_count", "groups", "share_with_match",
        "share_unique_match", "correction_nonzero_pct", "z_max", "z_over_15",
        "max_delta_ordinals", "bytes_per_pass_shipped", "bytes_saved_bias6",
        "bytes_saved_nobias"])
    total = nomatch = nonuniq = corrected = 0
    shipped_bytes = saved_bytes = 0
    for name, row in families.items():
        groups = row["groups"]
        table.add_data(
            name, row["K"], row["N"], row["tensor_count"], groups,
            row["share_with_match"], row["share_unique_match"],
            100.0 * row["share_needing_correction"], row["z_max"],
            row["z_over_15"], row["max_abs_delta_ordinals"],
            row["bytes_per_pass_shipped"], row["bytes_saved_bias6"],
            row["bytes_saved_nobias"])
        total += groups
        nomatch += round(groups * (1.0 - row["share_with_match"]))
        nonuniq += round(groups * (row["share_with_match"]
                                   - row["share_unique_match"]))
        corrected += round(groups * row["share_needing_correction"])
        shipped_bytes += row["bytes_per_pass_shipped"]
        saved_bytes += row["bytes_saved_bias6"]
    run.log({"per_family": table})

    # The census keeps the first z it finds. This table replays a sample
    # exhaustively over the whole search range and adds the control that
    # proves the search can fail: a correction of +-2 or +-3 ordinals must
    # not be reachable, or the uniqueness claim would be vacuous.
    checks = wandb.Table(columns=[
        "family", "tensor", "sampled_groups", "exhaustive_unique",
        "exhaustive_none", "exhaustive_multi", "control_plus2_matches",
        "control_plus3_matches"])
    for name, row in doc["verification"].items():
        checks.add_data(
            name, row["tensor"], row["sampled_groups"],
            row["exhaustive_unique"], row["exhaustive_none"],
            row["exhaustive_multi"], row["control_plus2_matches"],
            row["control_plus3_matches"])
    run.log({"exhaustive_verification": checks})

    run.summary.update({
        "scored_families": len(families),
        "scored_groups": total,
        "groups_with_no_match": nomatch,
        "groups_with_non_unique_z": nonuniq,
        "groups_needing_correction": corrected,
        "correction_nonzero_pct": 100.0 * corrected / total,
        "max_delta_ordinals": max(r["max_abs_delta_ordinals"]
                                  for r in families.values()),
        "z_over_15_groups": sum(r["z_over_15"] for r in families.values()),
        "recoding_is_lossless": nomatch == 0 and nonuniq == 0,
        "fallback_tensors": 0,
        "weight_side_bytes_per_pass": shipped_bytes,
        "bias6_bytes_saved_per_pass": saved_bytes,
        "honest_byte_saving_pct": 100.0 * saved_bytes / shipped_bytes,
        "code_bits_used": 6,
        "code_bits_spare": 2,
        "rung0_verdict": "pass",
        "root_cause": (
            "quantized.h:2976-2980 sets q0 = round(edge/scale), then "
            "scale = edge/q0 and bias = edge, so bias == q0*scale in fp32 "
            "before both are rounded to bf16 independently. z == -q0."
        ),
    })
    run.finish()


def log_rung1() -> None:
    sessions = {tag: an.reduce_session(tag) for tag, *_ in SESSIONS
                if (OUT / tag / "arms.json").exists()}
    meta = read_meta("e111-r1-gateup-na4-ctl")

    run = start(
        job_type="isolated-metal-arms", name="e111-rung1-arms",
        config={
            "rung": "1",
            "question": (
                "does replacing the 2-byte bf16 bias load with a 1-byte code "
                "make the wide verify QMV faster"
            ),
            "leg_command": (
                "research/e111_iso_leg.sh TAG mlp.gate_up --na N "
                "--blocks 8 --inner 4 --reps 2 --warm 3"
            ),
            "order": "palindrome_within_block",
            "blocks": 8, "blocks_used": 8, "inner": 4, "reps": 2,
            "warm_passes": 3,
            "discarded_untimed_palindromes": 1,
            "na_swept": sorted({s["na"] for s in sessions.values()}),
            "na_weights": an.NA_WEIGHT,
            "na_weight_source": "Edward realised width histogram, wandb 19kgn6xi",
            "kill1_pct": an.KILL1_PCT,
            "kill2_pct": an.KILL2_PCT,
            "advance_pct": an.ADVANCE_PCT,
            "dram_peak_gbps": DRAM_PEAK_GBPS,
            "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"]),
            "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"]),
            "harness_sha256": meta.get("harness_sha256"),
            "kernel_src_sha256": meta.get("kernel_src_sha256"),
            **identity(meta["base_sha"]),
            **gate_flags("standalone Metal microbenchmark, GPU", 60.0, True),
        },
    )

    arms = wandb.Table(columns=[
        "session", "role", "shape", "na", "arm", "arm_role", "group_bytes",
        "gpu_us_mean", "gpu_us_sd", "blocks", "gbps", "pct_of_peak",
        "paired_pct", "paired_sd", "saving_pct", "round_pct"])
    for tag, shape, na, role in SESSIONS:
        s = sessions.get(tag)
        if s is None:
            continue
        for r in s["arms"]:
            arms.add_data(
                tag, role, shape, na, r["arm"], ARM_ROLE[r["arm"]],
                r["group_bytes"], r["us_mean"], r["us_sd"], r["blocks"],
                r["gbps"], 100.0 * r["gbps"] / DRAM_PEAK_GBPS,
                r["paired_pct"], r["paired_sd"], r["saving_pct"],
                r["saving_pct"] * an.QMV_SHARE_OF_ROUND)
    run.log({"arms": arms})

    shipped = wandb.Table(columns=[
        "session", "shape", "na", "gbps", "pct_of_peak",
        "bias_load_cost_pct", "bias_load_cost_per_byte_pct",
        "b_constw_implied_byte_price", "b_constw_usable"])
    for tag, shape, na, _ in SESSIONS:
        s = sessions.get(tag)
        if s is None:
            continue
        shipped.add_data(
            tag, shape, na, s["shipped_gbps"], s["shipped_pct_of_peak"],
            s["bias_load_cost_pct"], s["bias_load_cost_per_byte_pct"],
            s["b_constw_implied_byte_price"],
            s["b_constw_usable_as_byte_price"])
    run.log({"shipped_rate_and_byte_price": shipped})

    controls = wandb.Table(columns=[
        "session", "control", "detail", "fires"])
    for tag, *_ in SESSIONS:
        s = sessions.get(tag)
        if s is None:
            continue
        w, c = s["weight_control"], s["code_control"]
        controls.add_data(
            tag, "weight",
            f"perturbed {w['perturbed_rel']:.3e} restored "
            f"{w['restored_rel']:.3e}", w["detected"])
        controls.add_data(
            tag, "code",
            f"damaged {c['damaged_differing']} restored "
            f"{c['restored_differing']}", c["detected"])
        p = s["pack_control"]
        if p:
            controls.add_data(
                tag, "pack",
                f"damaged {p['damaged_differing']} restored "
                f"{p['restored_differing']}", p["detected"])
        f = s["fidelity"]
        controls.add_data(
            tag, "fidelity vs cpu",
            f"worst_rel {f['worst_rel']:.3e} tol {f['tolerance']}", f["pass"])
        for e in s["arm_exactness"]:
            if e["expect_bit_exact"]:
                controls.add_data(
                    tag, f"bit-exact {e['arm']}",
                    f"{e['differing']}/{e['total']} differing", e["pass"])
    run.log({"controls": controls})

    weighted = [sessions[t] for t in WEIGHTED_TAGS if t in sessions]
    out = {}
    for arm in an.ARMS:
        val, _ = an.weighted(weighted, arm)
        if val == val:
            out[arm] = val
    bias_load = out["n_nobias"] - out["n_nosums"]

    wt = wandb.Table(columns=["arm", "arm_role", "weighted_saving_pct",
                              "round_pct", "clears_promotion_bar"])
    for arm, val in out.items():
        wt.add_data(arm, ARM_ROLE[arm], val, val * an.QMV_SHARE_OF_ROUND,
                    val * an.QMV_SHARE_OF_ROUND >= PROMOTION_BAR_PCT)
    run.log({"round_weighted": wt})

    run.summary.update({
        **{f"weighted_{a}_pct": v for a, v in out.items()},
        "weighted_d_bias1_pct": out["d_bias1"],
        "weighted_e_bias6_pct": out["e_bias6"],
        "e_bias6_round_pct": out["e_bias6"] * an.QMV_SHARE_OF_ROUND,
        "kill1_n_nobias_fires": out["n_nobias"] < an.KILL1_PCT,
        "kill2_d_bias1_fires": out["d_bias1"] < an.KILL2_PCT,
        "advance_e_bias6_fires": out["e_bias6"] >= an.ADVANCE_PCT,
        "byte_accounting_weight_side_bytes": 1152,
        "byte_accounting_bias_bytes": 64,
        "byte_accounting_matches_advisor": True,
        "bias6_traffic_cut_pct": 100.0 / 36,
        "verdict": "killed",
        "reconstruction_cost_pp": out["e_bias6"] - out["d_bias1"],
        "whole_bias_load_cost_pct": bias_load,
        "e_bias6_absolute_ceiling_pct": bias_load
        + (out["e_bias6"] - out["d_bias1"]),
        "verdict_reason": (
            "d_bias1 is the exact ceiling of Bias6: it takes the whole 1-byte "
            "traffic cut and pays no reconstruction. Round weighted over the "
            "realised widths it is negative, so the traffic the mechanism "
            "removes is not on the critical path at the widths that score."
        ),
        "kill_is_generation_robust": True,
        "kill_argument": (
            "n_nobias minus n_nosums prices the whole 2-byte bias load, its "
            "bytes, its issue slot and its registers together, at +2.26 % "
            "round weighted. e_bias6 minus d_bias1 prices the exact "
            "reconstruction at -2.97 pp. The reconstruction therefore costs "
            "more than deleting the entire bias would ever save, so the "
            "absolute ceiling of any lossless recoding of this field is "
            "-0.71 %. That bound uses only differences measured on this "
            "host and does not depend on how much of the load cost is "
            "bytes, so it holds on the ranked generation too."
        ),
        "causal_reading": (
            "The cost of the stored bias is its load instruction, not its "
            "bytes. d_bias1 keeps the load and narrows it from 2 bytes to 1, "
            "and that measures as nothing. A mechanism that removes metadata "
            "bytes but keeps the load cannot pay in this kernel."
        ),
        "surviving_lead": "g_pack32",
        "surviving_lead_note": (
            "one interleaved 32-bit metadata record for the same bytes and "
            "the same values is +0.33 % on mlp.gate_up at NA=4 over two "
            "independent sessions, but +0.04 +- 0.91 % on mlp.down at NA=4, "
            "so it does not generalise across shapes and is not promotable "
            "on this evidence"
        ),
    })
    run.finish()


def log_static() -> None:
    regs: dict[tuple[int, str], dict] = {}
    path = OUT / "e111" / "regs.txt"
    if not path.exists():
        print("no static census recorded, skipping")
        return
    for line in path.read_text().splitlines():
        head, arch, payload = line.split(" ", 2)
        regs[(int(head.split("=")[1]), arch)] = json.loads(payload)
    nas = sorted({na for na, _ in regs})

    sessions = {s["na"]: s for s in
                (an.reduce_session(t) for t in WEIGHTED_TAGS
                 if (OUT / t / "arms.json").exists())}
    local, ranked = regs[(4, LOCAL_ARCH)], regs[(4, RANKED_ARCH)]

    run = start(
        job_type="static-budget", name="e111-static-budget",
        config={
            "rung": "1",
            "question": (
                "do register pressure, spill or ISA text explain the "
                "measured ranking of the bias arms, and does that "
                "explanation transfer to the ranked generation"
            ),
            "tool": "xcrun metal-tt via research/agx_crossarch.py census",
            "metallib": (
                "xcrun metal -std=metal3.1 -DE111_NA=N "
                "research/e111_bias6_arms.metal, N in 2 3 4 5"
            ),
            "na_swept": nas,
            "selftest": "PASS, both arches, permutation guard included",
            **identity(ASSIGNMENT_BASE_SHA),
            **gate_flags("toolchain static analysis, no GPU", 0.0, False),
        },
    )
    table = wandb.Table(columns=[
        "arm", "arm_role", "arch", "na", "registers", "spill_bytes",
        "text_bytes", "delta_spill_vs_shipped", "delta_text_vs_shipped",
        "measured_saving_pct", "na_weight", "text_sha8"])
    for (na, arch), kernels in sorted(regs.items()):
        base = kernels["a_shipped"]
        measured = sessions.get(na)
        for arm, value in sorted(kernels.items()):
            hit = next((r for r in measured["arms"] if r["arm"] == arm),
                       None) if measured else None
            table.add_data(
                arm, ARM_ROLE.get(arm, "bandwidth reference"), arch, na,
                value["registers"], value["spill_bytes"], value["text_bytes"],
                value["spill_bytes"] - base["spill_bytes"],
                value["text_bytes"] - base["text_bytes"],
                hit["saving_pct"] if hit else None, an.NA_WEIGHT.get(na),
                value["text_sha8"])
    run.log({"registers_and_text": table})

    # NA=4 carries 66.7 % of streaming time, so it is the width that decides
    # whether the local measurement transfers.
    run.summary.update({
        "dominant_na": 4,
        "local_spilling_widths": [
            na for na in nas
            if any(v["spill_bytes"] > 0
                   for v in regs[(na, LOCAL_ARCH)].values())],
        "ranked_spilling_widths": [
            na for na in nas
            if any(v["spill_bytes"] > 0
                   for v in regs[(na, RANKED_ARCH)].values())],
        "shipped_spill_bytes_local_na4": local["a_shipped"]["spill_bytes"],
        "shipped_spill_bytes_ranked_na4": ranked["a_shipped"]["spill_bytes"],
        "d_bias1_extra_spill_vs_shipped_local_na4":
            local["d_bias1"]["spill_bytes"] - local["a_shipped"]["spill_bytes"],
        "d_bias1_extra_spill_vs_shipped_ranked_na4":
            ranked["d_bias1"]["spill_bytes"]
            - ranked["a_shipped"]["spill_bytes"],
        "e_bias6_extra_text_vs_d_bias1_ranked_na4":
            ranked["e_bias6"]["text_bytes"] - ranked["d_bias1"]["text_bytes"],
        "d_bias1_extra_text_vs_shipped_ranked_na4":
            ranked["d_bias1"]["text_bytes"] - ranked["a_shipped"]["text_bytes"],
        "prereg_census_was_na5_only": True,
        "spill_ranks_these_arms": False,
        "static_note": (
            "The static budget depends on NA, and the census in my "
            "pre-registration was NA=5 only. Correcting it: the local "
            "generation spills at NA=3, 4 and 5 and pins the full arms at 96 "
            "registers, while the ranked generation spills at NA=5 only. At "
            "NA=4, the width that carries 66.7 % of streaming time, every "
            "local arm spills 16 to 48 B and no ranked arm spills at all, so "
            "the dominant local measurement sits in a spill regime the "
            "ranked runner does not enter. d_bias1 is the only arm that "
            "spills more than the shipped kernel locally at NA=4, 48 B "
            "against 32 B, which is a local-only confound on the arm whose "
            "result kills the traffic hypothesis."
        ),
        "transfer_note": (
            "The kill still transfers. At NA=4 on the ranked generation "
            "d_bias1 and e_bias6 are both spill free, which is the regime "
            "campaign rule 36 was fitted on, and e_bias6 carries 416 B more "
            "ISA text than d_bias1 there, about +6 %. Rule 36 therefore "
            "predicts the same ordering on the ranked runner that the local "
            "timing measured. Locally the reconstruction penalty is also not "
            "a spill artefact: at NA=4 e_bias6 spills 32 B less than d_bias1 "
            "and is still 2.83 pp slower."
        ),
    })
    run.finish()


RUNS = {
    "e111-rung0-recode": log_rung0,
    "e111-rung1-arms": log_rung1,
    "e111-static-budget": log_static,
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
