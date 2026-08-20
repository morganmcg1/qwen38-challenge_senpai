#!/usr/bin/env python3
"""Log E59 to W&B, one rung at a time and one timed leg at a time.

The advisor's standing instruction is to log while measuring, never once at
session end, so this script resumes one run id and is called after every rung
and after every timed leg.

  research/e59_wandb_log.py --stage rung1
  research/e59_wandb_log.py --stage rung2
  research/e59_wandb_log.py --stage rung3
  research/e59_wandb_log.py --stage rung4-parity
  research/e59_wandb_log.py --stage rung4-leg --leg .mlxfast-private/e59-e2e/runs/TAG
  research/e59_wandb_log.py --stage rung4
  research/e59_wandb_log.py --stage gates

The run id lives in research/e59-artifacts/wandb-run-id.txt.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
REPO = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "research/e59-artifacts"
RUN_ID_FILE = ARTIFACTS / "wandb-run-id.txt"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False, cwd=REPO).stdout.strip()


def identity() -> dict:
    return {
        "assignment_id": "qwen38-r1-e59-m5-rowblock-r2-route",
        "revision_id": "r1",
        "pr_number": 62,
        "base_sha": "989596895b7c8f889443dac0c87e024a428e6e9e",
        "head_sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host_chip": "Apple M4 Pro",
        "host_gpu_architecture": "applegpu_g16s",
        "nax_available": False,
        "harness": "local",
        "local_mode": "--local-iterate",
        "scored_files": (
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h,"
            "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
        ),
        "register_legality_floor": 108,
        "shipped_kernel_maximum": 108,
        "null_floor_percent": 0.0629,
        "ranked_mde_2sd_percent": 0.283,
        "deficit_to_close_percent": 0.5367,
        "live_promoted_frontier": 3.24985583421771,
        "our_best_official_score": 3.23250848263467,
    }


def start_run(resume: str | None):
    run_id = resume
    if run_id is None and RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip() or None
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="allow" if run_id else None,
        name="e59-m5-rowblock-r2-route",
        group="e59-m5-rowblock-r2",
        job_type="register-ceiling-route",
        tags=["e59", "qmv", "crossrow", "rows-per-simd", "register-ceiling",
              "m5", "qwen-thorfinn"],
        config=identity(),
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUN_ID_FILE.write_text(run.id + "\n")
    return run


# --- rung 1: register census and coverage proof --------------------------------

def log_rung1(run) -> None:
    census = json.loads((ARTIFACTS / "e59-reg-census.json").read_text())
    coverage = json.loads((ARTIFACTS / "e59-coverage.json").read_text())
    ceiling = census["ceiling"]

    arm_columns = ["arm", "family", "cell", "kernel_wide_reg_max",
                   "reachable_reg_max", "entry_point_reg_max",
                   "exceeds_ceiling", "argmax_width", "per_width_regs",
                   "routing"]
    arm_table = wandb.Table(columns=arm_columns)
    cell_columns = ["arm", "width", "wrapper", "ipg", "rows_per_simd",
                    "uniform", "na_cells", "peak_live_regs",
                    "peak_live_values", "allocas", "alloca_types",
                    "acc_alloca_types", "device_loads", "air_lines",
                    "predicted"]
    cell_table = wandb.Table(columns=cell_columns)
    summary: dict = {"rung1/register_ceiling": ceiling,
                     "rung1/any_arm_exceeds_ceiling": census["any_exceeds_ceiling"]}

    for name in sorted(census["arms"]):
        arm = census["arms"][name]
        per_width = {m: c["peak_live_regs"]
                     for m, c in sorted(arm["width_cells"].items())}
        arm_table.add_data(
            name, arm["family"], arm["cell"], arm["kernel_wide_reg_max"],
            arm["reachable_reg_max"], arm["entry_point_reg_max"],
            arm["exceeds_ceiling"], arm["argmax_width"],
            json.dumps(per_width), json.dumps(arm["routing"]))
        for width, cell in sorted(arm["width_cells"].items()):
            cell_table.add_data(
                name, int(width), cell["wrapper"], cell["ipg"],
                cell["rows_per_simd"], cell["uniform"],
                json.dumps(cell["na_cells"]), cell["peak_live_regs"],
                cell["peak_live_values"], cell["allocas"],
                json.dumps(cell["alloca_types"]),
                json.dumps(cell["acc_alloca_types"]), cell["device_loads"],
                cell["air_lines"], json.dumps(cell["predicted"]))
        key = f"rung1/{name}"
        summary[f"{key}/kernel_wide_reg_max"] = arm["kernel_wide_reg_max"]
        summary[f"{key}/reachable_reg_max"] = arm["reachable_reg_max"]
        summary[f"{key}/entry_point_reg_max"] = arm["entry_point_reg_max"]
        summary[f"{key}/exceeds_ceiling"] = arm["exceeds_ceiling"]
    run.log({"rung1/register_census": arm_table,
             "rung1/width_cells": cell_table})

    cov_columns = ["mapping", "expect_exact_cover", "exact_cover", "passed",
                   "pairs_expected", "pairs_written", "written_twice_count",
                   "never_written_count", "out_of_range_count"]
    cov_table = wandb.Table(columns=cov_columns)
    for case in coverage["cases"]:
        cov_table.add_data(*[case[name] for name in cov_columns])
    summary["rung1/coverage_all_passed"] = coverage["all_passed"]
    summary["rung1/coverage_controls_fired"] = sum(
        1 for c in coverage["cases"]
        if not c["expect_exact_cover"] and not c["exact_cover"])
    run.log({"rung1/coverage_proof": cov_table})
    run.summary.update(summary)


# --- rung 2: bitwise parity ----------------------------------------------------

def log_parity(run, artifact: str, ns: str) -> None:
    verdict = json.loads((ARTIFACTS / artifact).read_text())
    columns = ["comparison", "expectation", "cells_compared", "cells_differing",
               "widths_differing", "reachable_widths_differing",
               "unreachable_widths_differing", "bits_differing",
               "first_difference", "route_changes", "route_change_widths",
               "route_change_detail", "passed"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for row in verdict["comparisons"]:
        table.add_data(
            row["comparison"], row["expectation"], row["cells_compared"],
            row["cells_differing"], json.dumps(row["widths_differing"]),
            json.dumps(row["reachable_widths_differing"]),
            json.dumps(row["unreachable_widths_differing"]),
            json.dumps(row["bits_differing"]),
            json.dumps(row["first_difference"]), row["route_changes"],
            json.dumps(row["route_change_widths"]),
            json.dumps(row["route_change_detail"]), row["passed"])
        key = f"{ns}/" + row["comparison"].replace(" ", "_")
        summary[f"{key}/cells_differing"] = row["cells_differing"]
        summary[f"{key}/reachable_widths_differing"] = len(
            row["reachable_widths_differing"])
        summary[f"{key}/route_changes"] = row["route_changes"]
        summary[f"{key}/passed"] = row["passed"]
    summary[f"{ns}/max_reachable_width"] = verdict["max_reachable_width"]
    summary[f"{ns}/all_passed"] = verdict["all_passed"]
    summary[f"{ns}/controls_fired"] = verdict["controls_fired"]
    summary[f"{ns}/controls_total"] = verdict["controls_total"]
    run.log({f"{ns}/parity": table})
    run.summary.update(summary)


def log_rung2_e2e(run, legs: list[pathlib.Path]) -> None:
    """Short end-to-end exact-token runs with row-ledger closure."""
    columns = ["tag", "arm", "tokens", "all_tokens_matched",
               "public_drift_tripwire_passed", "residual_divergence_count",
               "declared_rows_total", "row_ledger_sum", "row_ledger_closes",
               "round_count", "effective_mean_draft_len", "accepted_draft_rate",
               "score"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for path in legs:
        rec = read_leg(path)
        table.add_data(
            rec["tag"], rec["arm"], rec["tokens"], rec["all_tokens_matched"],
            rec["public_drift_tripwire_passed"], rec["residual_divergence_count"],
            rec["declared_rows_total"], rec["row_ledger_sum"],
            rec["row_ledger_closes"], rec["round_count"],
            rec["effective_mean_draft_len"], rec["accepted_draft_rate"],
            rec["score"])
        key = f"rung2e2e/{rec['tag']}"
        summary[f"{key}/all_tokens_matched"] = rec["all_tokens_matched"]
        summary[f"{key}/row_ledger_closes"] = rec["row_ledger_closes"]
    run.log({"rung2/end_to_end_exactness": table})
    run.summary.update(summary)


# --- rung 3: isolated cell timing ---------------------------------------------

def log_rung3(run) -> None:
    metrics = json.loads((ARTIFACTS / "e59-cell-metrics.json").read_text())
    leg_columns = ["tag", "arm", "gpu_gate", "t_ms_by_width", "jit_spread_pct",
                   "bitwise_failures"]
    leg_table = wandb.Table(columns=leg_columns)
    for leg in metrics["legs"]:
        if leg["status"] != "ok":
            continue
        leg_table.add_data(
            leg["tag"], leg["arm"], leg["gpu_gate"]["state"],
            json.dumps(leg["t_ms"]), json.dumps(leg["jit_spread_pct"]),
            json.dumps(leg["bitwise_failures"]))
    run.log({"rung3/legs": leg_table})

    columns = ["pair", "control_arm", "treated_arm", "treated_width",
               "delta_pct", "delta_ms", "bar_pct", "direction",
               "worst_unchanged_width_pct", "pooled_unchanged_pct",
               "reference_pct"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for name, rep in metrics["pairs"].items():
        table.add_data(
            name, rep["control_arm"], rep["treated_arm"], rep["treated_width"],
            rep["delta_pct"], rep["delta_ms"], rep["bar_pct"], rep["direction"],
            rep["bar_inputs"]["worst_unchanged_code_width_pct"],
            rep["pooled_unchanged_code_pct"], rep["reference_pct"])
        summary[f"rung3/{name}/delta_pct"] = rep["delta_pct"]
        summary[f"rung3/{name}/bar_pct"] = rep["bar_pct"]
        summary[f"rung3/{name}/direction"] = rep["direction"]
    gate = metrics["rung3_gate"]
    summary["rung3/gate_state"] = gate["state"]
    summary["rung3/gate_threshold_pct"] = gate["threshold_pct"]
    if "best_pct" in gate:
        summary["rung3/best_net_cell_win_pct"] = gate["best_pct"]
    if "route_for_rung4" in gate:
        summary["rung3/route_for_rung4"] = gate["route_for_rung4"]
    for net, add in metrics.get("additivity", {}).items():
        summary[f"rung3/additivity/{net}/predicted_pct"] = add["predicted_pct"]
        summary[f"rung3/additivity/{net}/measured_pct"] = add["measured_pct"]
        summary[f"rung3/additivity/{net}/residual_pct"] = add["residual_pct"]
    run.log({"rung3/pairs": table})
    run.summary.update(summary)


# --- rung 4: end to end --------------------------------------------------------

def read_meta(path: pathlib.Path) -> dict:
    out: dict = {}
    meta = path / "meta.txt"
    if meta.exists():
        for line in meta.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                out[key] = value
    return out


def read_leg(path: pathlib.Path) -> dict:
    """One --local-iterate invocation: serial control leg plus candidate MTP leg."""
    meta = read_meta(path)
    score = json.loads((path / "score.json").read_text())
    metrics = score["metrics"]

    serial = mtp = None
    for report in sorted((path / "reports").glob("0*-mtp-timed.json")):
        payload = json.loads(report.read_text())
        if payload.get("is_serial_control"):
            serial = payload
        else:
            mtp = payload
    if serial is None or mtp is None:
        raise SystemExit(f"e59_wandb_log: {path} is missing a timed report")

    widths = [d + 1 for d in mtp["effective_draft_lengths"]]
    hist = collections.Counter(widths)
    blocks = mtp.get("block_request_seconds") or []
    ser_blocks = serial.get("block_request_seconds") or []

    def round_cost(report: dict, block_list: list) -> tuple[float, float]:
        """(leg seconds with prefill removed, seed prefill seconds)."""
        prefill = report.get("seed_prefill_seconds")
        if prefill is None:
            return (sum(block_list), float("nan"))
        return (report["decode_seconds"] - prefill, prefill)

    mtp_rounds_s, mtp_prefill_s = round_cost(mtp, blocks)
    ser_rounds_s, ser_prefill_s = round_cost(serial, ser_blocks)
    tokens = int(metrics["decode_tokens"])

    return {
        "tag": meta.get("tag", path.name),
        "arm": meta.get("arm", "?"),
        "tokens": tokens,
        "measured_commit_unwound": meta.get("measured_commit_unwound"),
        "twin_digests": meta.get("twin_digests"),
        "worker_sha256": meta.get("worker_sha256"),
        "metallib_source_fingerprint": meta.get("metallib_source_fingerprint"),
        "cool_gate_requested": meta.get("cool_gate_requested"),
        "warmup_discarded": meta.get("warmup_discarded", "0"),
        "startup_memory_profile": meta.get("startup_memory_profile"),
        "mlx_max_ops_per_buffer": meta.get("mlx_max_ops_per_buffer"),
        "mlx_max_mb_per_buffer": meta.get("mlx_max_mb_per_buffer"),
        "wired_residency_active": meta.get("wired_residency_active"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "stale_metallib_warnings": meta.get("stale_metallib_warnings"),
        "score": score["score"],
        "all_tokens_matched": metrics["all_tokens_matched"],
        "public_drift_tripwire_passed": metrics["public_drift_tripwire_passed"],
        "residual_divergence_count": metrics["residual_divergence_count"],
        "accepted_draft_rate": metrics["accepted_draft_rate"],
        "effective_mean_draft_len": metrics["effective_mean_draft_len"],
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "serial_decode_seconds": serial["decode_seconds"],
        "mtp_decode_seconds": mtp["decode_seconds"],
        "serial_seed_prefill_seconds": ser_prefill_s,
        "mtp_seed_prefill_seconds": mtp_prefill_s,
        "serial_rounds_seconds_prefill_removed": ser_rounds_s,
        "mtp_rounds_seconds_prefill_removed": mtp_rounds_s,
        "serial_seconds_per_token_prefill_removed": ser_rounds_s / tokens,
        "mtp_seconds_per_token_prefill_removed": mtp_rounds_s / tokens,
        "round_count": mtp["round_count"],
        "declared_rows_total": mtp.get("declared_rows_total"),
        "row_ledger_sum": sum(widths),
        "row_ledger_closes": sum(widths) == mtp.get("declared_rows_total"),
        "width_histogram": dict(sorted(hist.items())),
        "p50_block_request_seconds_after_first":
            mtp.get("p50_block_request_seconds_after_first"),
        "first_block_seconds": mtp.get("first_block_seconds"),
    }


LEG_COLUMNS = [
    "tag", "arm", "tokens", "started", "gpu_temp_entry_c", "gpu_temp_exit_c",
    "cool_gate_requested", "all_tokens_matched", "row_ledger_closes",
    "serial_seconds_per_token", "mtp_seconds_per_token",
    "serial_seconds_per_token_prefill_removed",
    "mtp_seconds_per_token_prefill_removed",
    "serial_decode_seconds", "mtp_decode_seconds",
    "serial_seed_prefill_seconds", "mtp_seed_prefill_seconds",
    "serial_rounds_seconds_prefill_removed",
    "mtp_rounds_seconds_prefill_removed",
    "round_count", "declared_rows_total", "row_ledger_sum",
    "effective_mean_draft_len", "accepted_draft_rate", "score",
    "width_histogram", "measured_commit_unwound", "worker_sha256",
    "metallib_source_fingerprint", "stale_metallib_warnings",
    "warmup_discarded", "startup_memory_profile", "mlx_max_ops_per_buffer",
    "mlx_max_mb_per_buffer", "wired_residency_active",
]


def log_rung4_leg(run, path: pathlib.Path, ns: str = "rung4") -> None:
    rec = read_leg(path)
    table = wandb.Table(columns=LEG_COLUMNS)
    table.add_data(*[
        json.dumps(rec[name]) if isinstance(rec[name], dict) else rec[name]
        for name in LEG_COLUMNS])
    run.log({f"{ns}/leg/{rec['tag']}": table})
    run.log({
        f"{ns}/{rec['tag']}/serial_seconds_per_token": rec["serial_seconds_per_token"],
        f"{ns}/{rec['tag']}/mtp_seconds_per_token": rec["mtp_seconds_per_token"],
        f"{ns}/{rec['tag']}/mtp_seconds_per_token_prefill_removed":
            rec["mtp_seconds_per_token_prefill_removed"],
        f"{ns}/{rec['tag']}/score": rec["score"],
    })
    run.summary.update({
        f"{ns}/{rec['tag']}/arm": rec["arm"],
        f"{ns}/{rec['tag']}/all_tokens_matched": rec["all_tokens_matched"],
        f"{ns}/{rec['tag']}/row_ledger_closes": rec["row_ledger_closes"],
        f"{ns}/{rec['tag']}/gpu_temp_entry_c": rec["gpu_temp_entry_c"],
        f"{ns}/{rec['tag']}/width_histogram": json.dumps(rec["width_histogram"]),
    })
    print("logged %s leg %s (arm %s)" % (ns, rec["tag"], rec["arm"]))


def log_rung4(run) -> None:
    metrics = json.loads((ARTIFACTS / "e59-e2e-metrics.json").read_text())
    columns = ["arm", "legs", "serial_seconds_per_token_mean",
               "mtp_seconds_per_token_mean",
               "mtp_seconds_per_token_prefill_removed_mean",
               "seed_prefill_seconds_mean", "round_count",
               "width_histogram", "entry_temp_c", "delta_vs_base_leg_pct",
               "delta_vs_base_round_cost_pct", "serial_leg_delta_pct"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for arm, rec in metrics["arms"].items():
        table.add_data(
            arm, rec["legs"], rec["serial_seconds_per_token_mean"],
            rec["mtp_seconds_per_token_mean"],
            rec["mtp_seconds_per_token_prefill_removed_mean"],
            rec["seed_prefill_seconds_mean"], rec["round_count"],
            json.dumps(rec["width_histogram"]),
            json.dumps(rec["entry_temp_c"]),
            rec.get("delta_vs_base_leg_pct"),
            rec.get("delta_vs_base_round_cost_pct"),
            rec.get("serial_leg_delta_pct"))
        key = f"rung4/{arm}"
        summary[f"{key}/serial_seconds_per_token_mean"] = rec["serial_seconds_per_token_mean"]
        summary[f"{key}/mtp_seconds_per_token_mean"] = rec["mtp_seconds_per_token_mean"]
        summary[f"{key}/delta_vs_base_leg_pct"] = rec.get("delta_vs_base_leg_pct")
        summary[f"{key}/delta_vs_base_round_cost_pct"] = rec.get(
            "delta_vs_base_round_cost_pct")
    for name, value in metrics.get("verdicts", {}).items():
        summary[f"rung4/verdict/{name}"] = (
            json.dumps(value) if isinstance(value, (dict, list)) else value)
    summary["rung4/entry_temperature_spread_c"] = metrics.get(
        "entry_temperature_spread_c")
    summary["rung4/cool_gate_passed_real_gate"] = metrics[
        "cool_gate_passed_real_gate"]
    summary["rung4/gate_qualified_for_timing"] = metrics[
        "gate_qualified_for_timing"]

    # The session's own noise model, one row per same-arm leg pair. This is the
    # only null the rung 4 verdicts use; the 0.0629 % constant is withdrawn.
    null_columns = ["field", "leg_separation", "arm", "tags", "abs_delta_pct"]
    null_table = wandb.Table(columns=null_columns)
    for field, by_sep in metrics["session_null_by_separation_pct"].items():
        for sep, entry in by_sep.items():
            for pair in entry["pairs"]:
                null_table.add_data(field, int(sep), pair["arm"],
                                    json.dumps(pair["tags"]),
                                    pair["abs_delta_pct"])
            summary[f"rung4/null/{field}/sep{sep}/max_abs_delta_pct"] = entry[
                "max_abs_delta_pct"]
    for name, value in metrics["bar_pct"].items():
        summary[f"rung4/bar/{name}"] = value

    reg_columns = ["field", "term", "estimate", "std_error", "t",
                   "estimate_pct_of_base", "residual_dof", "residual_sd"]
    reg_table = wandb.Table(columns=reg_columns)
    for field, fit in metrics["regression_time_by_arm_and_position"].items():
        if not fit.get("fitted"):
            summary[f"rung4/regression/{field}/fitted"] = False
            continue
        summary[f"rung4/regression/{field}/fitted"] = True
        summary[f"rung4/regression/{field}/residual_dof"] = fit["residual_dof"]
        for term, stats in fit["terms"].items():
            reg_table.add_data(field, term, stats["estimate"],
                               stats["std_error"], stats["t"],
                               stats["estimate_pct_of_base"],
                               fit["residual_dof"], fit["residual_sd"])
            if term.startswith("arm["):
                arm = term[4:-1]
                summary[f"rung4/regression/{field}/{arm}/pct_of_base"] = stats[
                    "estimate_pct_of_base"]
                summary[f"rung4/regression/{field}/{arm}/t"] = stats["t"]

    for leg in metrics.get("discarded_warmup_legs", []):
        summary[f"rung4/warmup/{leg['tag']}/gpu_temp_entry_c"] = leg[
            "gpu_temp_entry_c"]
        summary[f"rung4/warmup/{leg['tag']}/mtp_seconds_per_token"] = leg[
            "mtp_seconds_per_token"]

    run.log({"rung4/arms": table,
             "rung4/session_null": null_table,
             "rung4/regression": reg_table})
    run.summary.update(summary)


def log_geometry(run) -> None:
    """The replacement for the withdrawn low-memory grep."""
    path = ARTIFACTS / "e59-geometry-proof.json"
    proof = json.loads(path.read_text())
    columns = ["probe", "requested_profile", "exit_code", "low_memory_notices",
               "precondition_messages", "passed"]
    table = wandb.Table(columns=columns)
    summary: dict = {
        "geometry/all_passed": proof["all_passed"],
        "geometry/physical_memory_gib": proof["physical_memory_gib"],
        "geometry/replaces": proof["replaces"],
    }
    for entry in proof["profile_probes"]:
        table.add_data(*[entry[name] for name in columns])
        summary[f"geometry/profile/{entry['probe']}/passed"] = entry["passed"]
        summary[f"geometry/profile/{entry['probe']}/exit_code"] = entry[
            "exit_code"]
        summary[f"geometry/profile/{entry['probe']}/low_memory_notices"] = entry[
            "low_memory_notices"]
    dose = proof["dose_response"]
    summary["geometry/dose/ran"] = dose["ran"]
    summary["geometry/dose/passed"] = dose["passed"]
    if dose["ran"]:
        summary["geometry/dose/ops8_slower_than_ops50_pct"] = dose[
            "ops8_slower_than_ops50_pct"]
        summary["geometry/dose/min_effect_pct"] = dose["min_effect_pct"]
        dose_columns = ["tag", "requested_ops_per_buffer", "decode_tokens",
                        "mtp_seconds_per_token", "serial_seconds_per_token",
                        "all_tokens_matched", "gpu_temp_entry_c",
                        "gpu_temp_exit_c", "wired_residency_active"]
        dose_table = wandb.Table(columns=dose_columns)
        for leg in dose["legs"]:
            dose_table.add_data(*[leg[name] for name in dose_columns])
        run.log({"geometry/dose": dose_table})
    run.log({"geometry/profile_probes": table})
    run.summary.update(summary)
    print("logged geometry proof all_passed=%s" % proof["all_passed"])


# --- gates ---------------------------------------------------------------------

def log_gates(run) -> None:
    path = ARTIFACTS / "e59-gates.json"
    if not path.exists():
        print("e59_wandb_log: no gate record yet at %s" % path)
        return
    gates = json.loads(path.read_text())
    columns = ["gate", "command", "exit_code", "passed", "note"]
    table = wandb.Table(columns=columns)
    summary: dict = {}
    for gate in gates["gates"]:
        table.add_data(gate["gate"], gate["command"], gate["exit_code"],
                       gate["passed"], gate.get("note", ""))
        summary[f"gates/{gate['gate']}/passed"] = gate["passed"]
    run.log({"gates/summary": table})
    run.summary.update(summary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["rung1", "rung2", "rung2-e2e", "rung2b-leg",
                             "rung3", "rung4-parity", "geometry", "rung4-leg",
                             "rung4", "gates"])
    ap.add_argument("--leg", type=pathlib.Path,
                    help="one run directory for --stage rung4-leg or rung2b-leg")
    ap.add_argument("--legs", type=pathlib.Path, nargs="*", default=[],
                    help="run directories for --stage rung2-e2e")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    run = start_run(args.resume)
    try:
        if args.stage == "rung1":
            log_rung1(run)
        elif args.stage == "rung2":
            log_parity(run, "e59-parity.json", "rung2")
        elif args.stage == "rung4-parity":
            log_parity(run, "e59-parity-rung4.json", "rung4/parity")
        elif args.stage == "geometry":
            log_geometry(run)
        elif args.stage == "rung2-e2e":
            log_rung2_e2e(run, args.legs)
        elif args.stage == "rung3":
            log_rung3(run)
        elif args.stage in ("rung4-leg", "rung2b-leg"):
            if args.leg is None:
                raise SystemExit(f"e59_wandb_log: --stage {args.stage} needs --leg DIR")
            log_rung4_leg(run, args.leg,
                          ns="rung4" if args.stage == "rung4-leg" else "rung2b")
        elif args.stage == "rung4":
            log_rung4(run)
        else:
            log_gates(run)
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
