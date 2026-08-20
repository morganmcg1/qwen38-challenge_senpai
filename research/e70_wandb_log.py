#!/usr/bin/env python3
"""Log the E70 dispatch divergence audit to W&B.

E70 produces no timing. The record it leaves is the audit itself: which
kernel-selection sites diverge between our gen-16 host and the ranked gen-17
M5, the kernel names measured on each architecture arm, and the score
arithmetic for each divergent site. All three go into one run so a later agent
can reproduce the verdicts without rerunning the probe.

  research/e70_wandb_log.py \
      --rung0 research/e70-rung0.json \
      --rung1 research/out/e70-rung1 \
      --rung2 research/e70-rung2.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
ARTIFACTS = pathlib.Path("research/e70-artifacts")
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"

LIVE_PROMOTED_FRONTIER_SCORE = 3.25238228
OUR_BEST_OFFICIAL_SCORE = 3.23250848
RANKED_RUN_PUBLISHED_SCORE_SD_PERCENT = 0.756
RANKED_CANDIDATE_LEG_SD_PERCENT = 1.092

# Written by an earlier revision that wrongly refuted the 30.402 ms candidate
# depth-0 round. The W&B backend merges summary metrics instead of replacing
# them, so these keys cannot be deleted; they are overwritten with their own
# retraction. See research/e70_transfer_constant_provenance.py.
RETRACTED_SUMMARY_KEYS = {
    "rung2/head_prime_score_pct_at_decode_tau":
        "RETRACTED: built on tau_depth0_round = 1.7590, which divides the "
        "local CANDIDATE round by the ranked SERIAL round. See "
        "rung2/head_prime_score_pct_at_R_of_M.",
    "transfer/R_change_pct":
        "RETRACTED: R(depth-0) = 2.1383 stands. See transfer/R_depth0.",
    "transfer/R_serial_leg_corrected":
        "RETRACTED: derived from a build-mixing transfer constant. A leg "
        "ratio is not R; see transfer/R_of_M.",
    "transfer/board_samples_compatible_with_ledger_value":
        "RETRACTED: wrong population. 30.402 ms is a CANDIDATE-build depth-0 "
        "round and was tested against PINNED SERIAL legs.",
    "transfer/ranked_depth0_round_ms_confirmed":
        "SUPERSEDED by transfer/pinned_serial_depth0_round_ms. The value was "
        "right; the name did not say which build it measures.",
    "transfer/ranked_depth0_round_ms_ledger_refuted":
        "RETRACTED: 30.402 ms is not refuted. It is the candidate build's "
        "depth-0 round; see transfer/candidate_depth0_round_ms.",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e70-local-ranked-dispatch-divergence-audit",
        "revision_id": "r1",
        "pr_number": 73,
        "assignment_base_sha": "bdfbc4e92c93d216503980fb46258ff0b314145a",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_gpu_architecture": "applegpu_g16s",
        "host_mlx_devc": "s",
        "host_mlx_arch_gen": 16,
        "host_physical_memory_gib": 48,
        "host_startup_memory_profile": "low (48 GiB < the 64 GiB full-profile minimum)",
        "ranked_gpu_architecture_pro_max": "applegpu_g17s",
        "ranked_gpu_architecture_base": "applegpu_g17g",
        # E70 measures no time at all, so no thermal gate applies. These stay
        # false so the run can never be mistaken for a timed arm.
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "produces_timing_evidence": False,
        "live_promoted_frontier_score": LIVE_PROMOTED_FRONTIER_SCORE,
        "our_best_official_score": OUR_BEST_OFFICIAL_SCORE,
        "ranked_run_published_score_sd_percent":
            RANKED_RUN_PUBLISHED_SCORE_SD_PERCENT,
        "ranked_candidate_leg_sd_percent": RANKED_CANDIDATE_LEG_SD_PERCENT,
    }


def load_rung1(root: pathlib.Path) -> dict:
    """Collect every per-cell capture, keyed by architecture arm and cell."""
    arms: dict[str, dict] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        exits = {}
        manifest = arm_dir / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    exits[entry["cell"]] = entry["exit"]
        cells: dict[str, dict] = {}
        for path in sorted(arm_dir.glob("*.json")):
            report = json.loads(path.read_text())
            for cell in report.get("cells", []):
                cells[cell["cell"]] = cell
        arms[arm_dir.name] = {"exit_codes": exits, "cells": cells}
    return arms


def signal_kernels(names: list[str]) -> list[str]:
    """The audited families, with the RNG and elementwise noise dropped."""
    keep = ("affine_", "steel_", "sdpa_", "gemv", "block_softmax")
    return [n for n in names if n.startswith(keep)]


FAILED_TEST = re.compile(r"^✘ Test (\S+\(\)) failed after", re.M)
TEST_RUN_TOTAL = re.compile(
    r"Test run with (\d+) tests in (\d+) suites failed .* with (\d+) issues")


def swift_test_arm(path: pathlib.Path) -> dict | None:
    """Summarise one `swift test` log so the two arms can be compared."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    failed = sorted({m.group(1) for m in FAILED_TEST.finditer(text)})
    total = TEST_RUN_TOTAL.search(text)
    return {
        "failing_tests": failed,
        "failing_test_count": len(failed),
        "tests": int(total.group(1)) if total else None,
        "suites": int(total.group(2)) if total else None,
        "issues": int(total.group(3)) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung0", required=True)
    parser.add_argument("--rung1", required=True)
    parser.add_argument("--rung2", required=True)
    args = parser.parse_args()

    rung0 = json.loads(pathlib.Path(args.rung0).read_text())
    rung2 = json.loads(pathlib.Path(args.rung2).read_text())
    rung1 = load_rung1(pathlib.Path(args.rung1))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    run_id = RUN_ID_FILE.read_text().strip() if RUN_ID_FILE.exists() else None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id or None,
        resume="allow",
        name="e70-local-ranked-dispatch-divergence-audit",
        job_type="dispatch-divergence-audit",
        config=identity(),
        tags=["e70", "qwen-alphonse", "arch-probe", "source-audit",
              "kernel-selection", "nax", "transfer-risk"],
    )
    RUN_ID_FILE.write_text(run.id + "\n")

    # --- rung 0: the site table --------------------------------------------
    site_table = wandb.Table(columns=[
        "id", "file", "line", "predicate", "reached_in_decode",
        "reached_in_prefill", "local_g16s", "ranked_g17s", "base_m5_g17g",
        "verdict", "kernels",
    ])
    diverging = []
    unreachable = []
    for site in rung0["sites"]:
        site_table.add_data(
            site["id"], site["file"], site["line"], site["predicate"],
            site["decode"], site["prefill"], site["local"], site["ranked"],
            site["base_m5"], site["verdict"], site["kernels"])
        if site["verdict"].startswith("DIVERGES"):
            diverging.append(site["id"])
        if "unreachable" in site["verdict"]:
            unreachable.append(site["id"])

    check_table = wandb.Table(columns=["check", "passed"])
    for check in rung0["checks"]:
        check_table.add_data(check["name"], check["pass"])

    mutation_table = wandb.Table(columns=["mutated_check", "outcome"])
    for mutation in rung0["mutations"]:
        mutation_table.add_data(mutation["label"], mutation["verdict"])

    # --- rung 1: the measured kernel names ----------------------------------
    kernel_table = wandb.Table(columns=[
        "cell", "site", "shape", "arch_arm", "exit_code", "dispatches",
        "selected_kernels", "all_kernels", "rung0_prediction",
    ])
    cell_ids: list[str] = []
    for arm in rung1.values():
        for cell_id in arm["cells"]:
            if cell_id not in cell_ids:
                cell_ids.append(cell_id)
    for arm_name, arm in sorted(rung1.items()):
        forced = arm_name != "real"
        for cell_id in cell_ids:
            cell = arm["cells"].get(cell_id)
            if cell is None:
                continue
            exit_code = arm["exit_codes"].get(
                cell_id, arm["exit_codes"].get("all"))
            kernel_table.add_data(
                cell_id, cell["site"], cell["shape"], arm_name, exit_code,
                cell["dispatches"],
                ", ".join(signal_kernels(cell["kernel_names"])),
                ", ".join(cell["kernel_names"]),
                cell["rung0_forced_prediction"] if forced
                else cell["rung0_local_prediction"])

    # A side-by-side diff is the deliverable the assignment named, so build it
    # explicitly rather than making the reader pivot the table above.
    arms_present = sorted(rung1)
    forced_arms = [a for a in arms_present if a != "real"]
    diff_table = wandb.Table(columns=[
        "cell", "site", "real_applegpu_g16s", "forced_arch",
        "forced_kernels", "forced_exit", "kernel_changed",
    ])
    changed_cells = []
    for cell_id in cell_ids:
        real_cell = rung1.get("real", {}).get("cells", {}).get(cell_id)
        real_names = signal_kernels(real_cell["kernel_names"]) if real_cell else []
        for arm_name in forced_arms:
            forced_cell = rung1[arm_name]["cells"].get(cell_id)
            forced_names = (
                signal_kernels(forced_cell["kernel_names"]) if forced_cell else [])
            changed = bool(real_cell and forced_cell) and (
                real_names != forced_names)
            if changed:
                changed_cells.append(cell_id)
            diff_table.add_data(
                cell_id,
                (real_cell or forced_cell or {}).get("site", ""),
                ", ".join(real_names) or "<not captured>",
                arm_name,
                ", ".join(forced_names) or "<not captured>",
                rung1[arm_name]["exit_codes"].get(cell_id),
                changed)

    # --- rung 2: the score arithmetic ---------------------------------------
    consequence_table = wandb.Table(columns=[
        "site", "delta_local_ms", "tau", "rounds_at_M", "delta_ranked_ms",
        "score_pct_route_a_median_pair", "score_pct_route_b_same_tau",
        "score_pct_route_b_R_of_M", "leg_model_agreement_pct",
        "naive_score_pct", "overstatement_factor", "sd_of_published_score",
        "fraction_of_deficit", "steerable", "steerable_reason",
    ])
    for name, site in rung2["sites"].items():
        a = site["arithmetic"]
        route_b_r = a.get("route_b_direct_form_at_R_of_M")
        consequence_table.add_data(
            name, a["delta_local_ms"], a["tau"], a["rounds_at_M"],
            a["delta_ranked_ms"],
            a["route_a_median_pair_score_pct"],
            a["route_b_direct_form_at_same_tau"]["score_pct"],
            route_b_r["score_pct"] if route_b_r else None,
            a["leg_model_agreement_pct"],
            a["naive_no_tau_score_pct"], a["overstatement_factor"],
            a["sd_of_published_score"], a["fraction_of_deficit"],
            site["steerable"], site["steerable_reason"])

    # --- the transfer-constant retraction and the R(M) table ----------------
    transfer = json.loads(
        pathlib.Path("research/e70-transfer-constant.json").read_text())
    r_of_m = transfer["R_of_M"]
    two_builds = transfer["two_builds"]
    ceiling = transfer["model_free_ceiling"]

    width_table = wandb.Table(columns=[
        "prompt", "verify_width_M", "local_round_ms", "ranked_round_ms",
        "R_of_M", "rounds_at_M", "tokens_per_round", "rounds_plus_accepted",
        "ranked_candidate_leg_ms",
    ])
    for row in r_of_m["table"]:
        width_table.add_data(
            row["prompt"], row["verify_width_M"], row["local_round_ms"],
            row["ranked_round_ms"], row["R_of_M"], row["rounds_at_M"],
            row["tokens_per_round"], row["rounds_plus_accepted"],
            row["ranked_candidate_leg_ms"])

    constant_table = wandb.Table(columns=["quantity", "value", "population"])
    for quantity, value, population in [
        ("pinned serial depth-0 round ms",
         two_builds["pinned_serial_depth0_round_ms"],
         "3768 board serial legs, 3 routes agreeing to "
         f"{two_builds['pinned_serial_routes']['route_spread_pct']:.3f} %"),
        ("candidate depth-0 round ms",
         two_builds["candidate_depth0_round_ms"],
         "reconstruction of ca9251b8 from row['mtp_spt']"),
        ("build factor serial/candidate",
         two_builds["serial_over_candidate"], "quotient of the two above"),
        ("model-free ceiling on c1 ms", ceiling["ceiling_ms"],
         f"{ceiling['anchor_prompt']} mean round over "
         f"{ceiling['anchor_rounds']} rounds"),
        ("R(depth-0)", r_of_m["table"][0]["R_of_M"], "65.009 / 30.402"),
        ("R(M) low shelf mean", r_of_m["low_width_group_mean"], "M <= 3.66"),
        ("R(M) high shelf mean", r_of_m["high_width_group_mean"], "M >= 5.53"),
        ("tau prefill", transfer["pricing_rule"]["tau_prefill"],
         "ledger 186(C), unchanged"),
    ]:
        constant_table.add_data(quantity, value, population)

    run.log({
        "rung0/site_table": site_table,
        "transfer/constants": constant_table,
        "transfer/R_of_M": width_table,
        "rung0/structural_checks": check_table,
        "rung0/mutation_controls": mutation_table,
        "rung1/kernel_capture": kernel_table,
        "rung1/arch_arm_diff": diff_table,
        "rung2/score_consequences": consequence_table,
    })

    checks_passed = sum(1 for c in rung0["checks"] if c["pass"])
    mutations_flipped = sum(
        1 for m in rung0["mutations"] if m["verdict"] == "flipped")
    forced_failures = {
        arm: [c for c, code in data["exit_codes"].items() if code != 0]
        for arm, data in rung1.items()
    }

    run.summary.update({
        "rung0/sites_audited": len(rung0["sites"]),
        "rung0/sites_diverging_on_m5_pro_max": len(diverging),
        "rung0/diverging_site_ids": diverging,
        "rung0/sites_unreachable": len(unreachable),
        "rung0/unreachable_site_ids": unreachable,
        "rung0/structural_checks_passed": checks_passed,
        "rung0/structural_checks_total": len(rung0["checks"]),
        "rung0/mutation_controls_flipped": mutations_flipped,
        "rung0/mutation_controls_total": len(rung0["mutations"]),
        "rung0/sdpa_full_nax_gate_reachable": False,
        "rung0/qmv_decode_path_identical": True,
        "rung0/arch_probe_moves_arch_gen_and_devc": True,
        "rung1/arch_arms": arms_present,
        "rung1/cells_probed": len(cell_ids),
        "rung1/cells_whose_kernel_changed_under_forced_arch": sorted(
            set(changed_cells)),
        "rung1/forced_arch_cell_failures": forced_failures,
        "rung2/score_pct_if_both_divergent_sites_cost_zero":
            rung2["total_if_every_divergent_site_cost_went_to_zero_pct"],
        "rung2/head_prime_score_pct":
            rung2["sites"]["S4_decode_head_prime"]["arithmetic"][
                "route_a_median_pair_score_pct"],
        "rung2/prefill_sdpa_fallback_score_pct":
            rung2["sites"]["S9_prefill_sdpa_fallback"]["arithmetic"][
                "route_a_median_pair_score_pct"],
        "rung2/width_shares_apply_to_any_divergent_site": False,
        "rung2/median_pair_prompts": rung2["median_pair_model"][
            "median_pair_prompts"],
        "rung2/median_pair_self_check_relative_error": rung2[
            "median_pair_model"]["self_check_relative_error"],
        "rung2/head_prime_score_pct_at_R_of_M": rung2["sites"][
            "S4_decode_head_prime"]["arithmetic"][
                "route_b_direct_form_at_R_of_M"]["score_pct"],
        "rung2/route_leg_model_max_disagreement_pct": rung2[
            "route_agreement"]["leg_model_max_abs_disagreement_pct"],
        "transfer/refutation_of_30_402_status":
            transfer["retraction"]["status"],
        "transfer/pinned_serial_depth0_round_ms":
            two_builds["pinned_serial_depth0_round_ms"],
        "transfer/candidate_depth0_round_ms":
            two_builds["candidate_depth0_round_ms"],
        "transfer/build_factor_serial_over_candidate":
            two_builds["serial_over_candidate"],
        "transfer/model_free_ceiling_on_c1_ms": ceiling["ceiling_ms"],
        "transfer/candidate_margin_under_ceiling_pct":
            ceiling["candidate_margin_under_ceiling_pct"],
        "transfer/pinned_serial_excess_over_ceiling_pct":
            ceiling["pinned_serial_excess_over_ceiling_pct"],
        "transfer/board_serial_leg_samples":
            transfer["board_evidence"]["serial_spt_samples"],
        "transfer/R_depth0": r_of_m["table"][0]["R_of_M"],
        "transfer/R_of_M_low_shelf_mean": r_of_m["low_width_group_mean"],
        "transfer/R_of_M_high_shelf_mean": r_of_m["high_width_group_mean"],
        "transfer/R_of_M_step_pct_group_means":
            r_of_m["step_pct_group_means"],
        "transfer/R_of_M_step_pct_shelf_edges":
            r_of_m["step_pct_shelf_edges"],
        "transfer/surviving_defect_188A":
            transfer["surviving_defect"]["defect"],
    })

    tests_root = pathlib.Path("research/out/e70-tests")
    head_arm = swift_test_arm(tests_root / "head.log")
    base_arm = swift_test_arm(tests_root / "base.log")
    if head_arm and base_arm:
        run.summary.update({
            "tests/head_failing": head_arm["failing_tests"],
            "tests/head_failing_count": head_arm["failing_test_count"],
            "tests/head_tests": head_arm["tests"],
            "tests/head_issues": head_arm["issues"],
            "tests/base_failing": base_arm["failing_tests"],
            "tests/base_failing_count": base_arm["failing_test_count"],
            "tests/base_tests": base_arm["tests"],
            "tests/base_issues": base_arm["issues"],
            "tests/failing_set_identical_to_base":
                head_arm["failing_tests"] == base_arm["failing_tests"],
            "tests/new_failures_introduced_by_branch": sorted(
                set(head_arm["failing_tests"]) - set(base_arm["failing_tests"])),
        })

    artifact = wandb.Artifact("e70-dispatch-divergence-audit", type="audit")
    artifact.add_file(args.rung0)
    artifact.add_file(args.rung2)
    for extra in ("research/e70-rung1-diff.json",
                  "research/e70-transfer-constant.json",
                  "research/e70-results.md"):
        path = pathlib.Path(extra)
        if path.exists():
            artifact.add_file(str(path))
    for arm_dir in sorted(p for p in pathlib.Path(args.rung1).iterdir() if p.is_dir()):
        for path in sorted(arm_dir.glob("*.json")):
            artifact.add_file(str(path), name=f"rung1/{arm_dir.name}/{path.name}")
        manifest = arm_dir / "manifest.jsonl"
        if manifest.exists():
            artifact.add_file(
                str(manifest), name=f"rung1/{arm_dir.name}/manifest.jsonl")
    run.log_artifact(artifact)

    run.summary.update(RETRACTED_SUMMARY_KEYS)

    print(f"run  {run.id}")
    print(f"url  {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
