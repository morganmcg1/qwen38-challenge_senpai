#!/usr/bin/env python3
"""Publish the E95 `target_verify` width census and the E95 rider to W&B.

    usage:
      research/e95_wandb_log.py census LEG@M [LEG@M ...]
      research/e95_wandb_log.py model
      research/e95_wandb_log.py rider --exact DIR --base LEG --cand LEG
      research/e95_wandb_log.py probe [--json PATH]

`census` writes one run per width leg: the identity tuple, the local score
metrics, the host thermal record and the per-phase and per-class dispatch
counters of that leg.

`model` writes the analysis run: the three-term width model, its residuals, the
per-class row-cost regression that prices the idle x-groups, and the ranked
cost weighting of the target weight stream.

`rider` writes the rider run: the 512-token exactness gate, its positive
control, the base-to-rider draft-head dispatch and byte diff, and the ranked
price of the removed dead rows.

`probe` writes the isolated `quantized_matmul` width probe: the measured
per-eval overhead, the read rate at four pack sizes, the refitted width model
on a large and a small tensor, the traffic share of each term, the elasticity
of time to bytes, and what share of the verify phase a byte reduction can
reach.

Every leg in this experiment is a counting and GPU-clock attribution leg, never
a gate-qualified timed arm, so `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged false verbatim on every run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e95_qmv_probe_analysis as probe  # noqa: E402
import e95_verify_census as census  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e95-dispatch-level-dead-work"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

# Fitted on the six census widths below. Reproduce with
#   research/e95_verify_census.py model e95-verify-m3@3 ... e93-gpu-insitu-long@9
WIDTH_MODEL = {
    "fixed_us_per_round": 10_920.0,
    "us_per_weight_pass": 27_377.0,
    "us_per_verify_row": 10_268.0,
    "max_abs_residual_pct": 1.08,
    "out_of_sample_width": 6,
    "out_of_sample_residual_pct": -1.34,
    # Reading `b` as one full pass over the weight stream implies this rate.
    # The E95 probe run refutes that reading: `b` is 0.53 of a pass and the
    # kernel reads at 281.1 GB/s. See job_type `qmv-width-probe`.
    "implied_gb_per_s_if_b_were_one_pass": 526.4,
    "arithmetic_tmac_per_s": 2.50,
}

# Per-class row cost, from the M=5 to M=8 pair. `ns_per_group` is the cost of
# one threadgroup at fixed work; `us_per_mmac` is the cost of the work itself.
ROW_COST = [
    ("gate_up  O=17408", 16.01, 0.3909),
    ("in_proj  O=27648", 16.01, 0.3909),
    ("lm_head  O=248320", 15.92, 0.3888),
    ("qkv      O=7168", 16.46, 0.4019),
    ("out+down O=5120", 37.07, 0.3935),
]

# Ranked cost weighting of the target weight stream, over the two public
# prompts at their measured mean accepted-draft widths.
RANKED_WEIGHTING = {
    "mean_width_beagle": 5.38,
    "mean_width_essays": 6.09,
    "phase_share_pct": 90.8,
    "fixed_share_pct_beagle": 9.0,
    "fixed_share_pct_essays": 8.5,
    "weight_stream_share_pct_beagle": 45.3,
    "weight_stream_share_pct_essays": 42.7,
    "arithmetic_share_pct_beagle": 45.7,
    "arithmetic_share_pct_essays": 48.8,
    "class_share_pct_gate_up_beagle": 40.5,
    "class_share_pct_gate_up_essays": 40.7,
    "class_share_pct_out_down_beagle": 27.4,
    "class_share_pct_out_down_essays": 27.6,
    "class_share_pct_gdn_in_proj_beagle": 14.4,
    "class_share_pct_gdn_in_proj_essays": 14.5,
    "class_share_pct_lm_head": 4.5,
    "class_share_pct_attn": 4.2,
}

# Itemisation of the fixed term `a` at M=5, the width nearest the ranked mean.
# Reproduce with `research/e95_verify_census.py fixed LEG@M ...`.
FIXED_TERM_M5 = [
    ("GDN recurrent step, full [48,128,128] fp32 state",
     48.0, 301.99, 8112.6, 37.2, 74.6, "latency"),
    ("fused residual + RMSNorm", 127.0, 20.81, 1187.1, 17.5, 10.9, "latency"),
    ("GDN recurrent state REPLAY", 6.52, 41.01, 658.4, 62.3, 6.1, "latency"),
    ("GDN prework: causal conv1d, q/k norm, gates",
     48.0, 17.69, 475.3, 37.2, 4.4, "latency"),
    ("q_norm + k_norm + RoPE", 16.0, 1.16, 210.5, 5.5, 1.9, "latency"),
    ("full-attention KV cache write", 32.79, 2.31, 166.5, 13.8, 1.5, "latency"),
    ("MTP top-2 partial + finalize", 2.0, 0.0, 55.4, 0.0, 0.5, "no bytes"),
    ("everything else", 14.94, 0.0, 4.5, 0.0, 0.0, "no bytes"),
]

FIXED_TERM = {
    "a_us_per_round": 10_919.5,
    "gdn_step_us_per_round_low": 6_492.8,
    "gdn_step_us_per_round_high": 9_493.8,
    "gdn_step_us_per_round_mean": 7_759.5,
    "gdn_step_mb_per_round": 301.99,
    "gdn_step_dram_rate_expectation_us": 1_139.6,
    "gdn_step_measured_over_dram_expectation": 6.81,
    "gdn_step_achieved_gb_per_s_low": 31.8,
    "gdn_step_achieved_gb_per_s_high": 46.5,
    "gdn_family_share_of_a_pct_m5": 85.1,
    "a_share_of_verify_phase_pct_beagle": 9.0,
    "a_share_of_round_pct": 8.1,
}

# Draft-head q projection, per marginal draft. The island overwrites 1024 of
# the 12288 q output rows, so the pack only has to produce the other 11264.
Q_ROWS_TOTAL = 12_288
Q_ROWS_ISLAND = 1_024
Q_ROWS_LIVE = Q_ROWS_TOTAL - Q_ROWS_ISLAND
Q_BYTES_PER_ROW = 5120 * 0.5625  # affine-4 group-64: 4 bits + scale + bias
HEAD_WEIGHT_BYTES_PER_CALL = 427_476_480  # class-1 head weight stream, e93


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def read_meta(leg: str) -> dict[str, str]:
    meta = {}
    path = OUT / leg / "meta.txt"
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def gate_flags() -> dict[str, bool]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def identity(meta: dict[str, str]) -> dict[str, object]:
    return {
        "host": HOST,
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "head_provenance_sha256": None,
        "tokens": int(meta.get("tokens", 0)) or None,
        "forced_drafts": int(meta.get("forced_drafts", -1)),
        "local_mode": meta.get("local_mode"),
        "census": meta.get("census") == "1",
        "gputime": meta.get("gputime") == "1",
        "ops_per_buffer": meta.get("ops_per_buffer"),
        "gpu_temp_entry_c": float(meta["gpu_temp_entry_c"])
        if meta.get("gpu_temp_entry_c") else None,
        "gpu_temp_exit_c": float(meta["gpu_temp_exit_c"])
        if meta.get("gpu_temp_exit_c") else None,
        "started": meta.get("started"),
        "finished": meta.get("finished"),
    }


def steady_rounds(leg: str, width: int) -> list[dict]:
    """Rounds of the MTP leg that ran at the forced verify width."""
    records = census.load(OUT / leg / "census.jsonl")
    return [r for r in records
            if r.get("event") == "round" and int(r["width"]) == width]


def leg_phase_table(leg: str, width: int):
    """Per-round dispatch count and encode microseconds for every phase."""
    rounds = steady_rounds(leg, width)
    table: dict[str, dict[str, float]] = {}
    for record in rounds:
        for phase, bucket in record["phases"].items():
            entry = table.setdefault(
                phase, {"dispatches_per_round": 0.0, "commits_per_round": 0.0})
            entry["dispatches_per_round"] += bucket["dispatches"]
            entry["commits_per_round"] += bucket["commits"]
    for entry in table.values():
        entry["dispatches_per_round"] /= len(rounds)
        entry["commits_per_round"] /= len(rounds)
    return len(rounds), table


def phase_shapes(leg: str, width: int, phase: str) -> dict[str, float]:
    """Per-round count of every dispatch shape inside one phase."""
    rounds = steady_rounds(leg, width)
    totals: dict[str, float] = {}
    for record in rounds:
        bucket = record["phases"].get(phase)
        if bucket is None:
            continue
        for shape, count in bucket.get("shapes", {}).items():
            totals[shape] = totals.get(shape, 0.0) + count
    return {shape: total / len(rounds) for shape, total in totals.items()}


def log_census(specs: list[str]) -> None:
    for spec in specs:
        leg, _, width_text = spec.partition("@")
        width = int(width_text)
        meta = read_meta(leg)
        score = read_json(OUT / leg / "score.json")
        metrics = score["metrics"]
        rounds, phases = leg_phase_table(leg, width)

        config = identity(meta)
        config["head_provenance_sha256"] = metrics.get("head_provenance_sha256")
        config["verify_width_m"] = width
        config["leg"] = leg
        config.update(gate_flags())

        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            job_type="width-census-leg", name=f"e95-census-{leg}",
            config=config, reinit=True)
        summary = {
            "rounds": rounds,
            "verify_width_m": width,
            "decode_tokens": metrics["decode_tokens"],
            "all_tokens_matched": metrics["all_tokens_matched"],
            "residual_divergence_count": metrics["residual_divergence_count"],
            "effective_mean_draft_len": metrics["effective_mean_draft_len"],
            "accepted_draft_rate": metrics["accepted_draft_rate"],
            "mtp_decode_speedup_census_serialised": metrics["mtp_decode_speedup"],
        }
        for phase, values in phases.items():
            summary[f"phase/{phase}/dispatches_per_round"] = values[
                "dispatches_per_round"]
            summary[f"phase/{phase}/commits_per_round"] = values[
                "commits_per_round"]
        run.summary.update(summary)
        print(f"{leg}\t{run.id}\t{run.url}")
        run.finish()


def log_model() -> None:
    config = {
        "host": HOST,
        "experiment": "e95-target-verify-width-model",
        "widths_fitted": [3, 4, 5, 8, 9],
        "width_held_out": 6,
        "tokens_per_leg": 384,
    }
    config.update(gate_flags())
    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="width-model", name="e95-width-model", config=config,
        reinit=True)

    run.summary.update({f"model/{k}": v for k, v in WIDTH_MODEL.items()})
    run.summary.update({f"ranked/{k}": v for k, v in RANKED_WEIGHTING.items()})

    rows = wandb.Table(
        columns=["class", "ns_per_threadgroup", "us_per_mmac"], data=ROW_COST)
    run.log({"row_cost_by_class": rows})
    run.summary.update({
        "dead_work/idle_x_groups_are_free": True,
        "dead_work/launch_only_fit_rms_pct": 29.4,
        "dead_work/work_only_fit_rms_pct_low": 0.42,
        "dead_work/work_only_fit_rms_pct_high": 0.52,
        "dead_work/joint_fit_ns_per_threadgroup_low": -0.20,
        "dead_work/joint_fit_ns_per_threadgroup_high": -0.17,
        "dead_work/quantized_cpp_in_editable_paths": False,
    })

    fixed = wandb.Table(
        columns=["class", "dispatches_per_round", "mb_per_round",
                 "us_per_round_m5", "gb_per_s_m5", "share_of_a_pct", "bound"],
        data=FIXED_TERM_M5)
    run.log({"fixed_term_itemisation_m5": fixed})
    run.summary.update({f"fixed/{k}": v for k, v in FIXED_TERM.items()})
    print(f"model\t{run.id}\t{run.url}")
    run.finish()


def log_rider(exact_dir: pathlib.Path, base_leg: str, cand_leg: str) -> None:
    summary_path = exact_dir / "rung0a-summary.json"
    exact = read_json(summary_path)
    meta = exact["meta"]

    base_meta = read_meta(base_leg)
    cand_meta = read_meta(cand_leg)
    base_score = read_json(OUT / base_leg / "score.json")["metrics"]
    cand_score = read_json(OUT / cand_leg / "score.json")["metrics"]

    base_shapes = phase_shapes(base_leg, 6, "draft_head")
    cand_shapes = phase_shapes(cand_leg, 6, "draft_head")

    saved_bytes = int((Q_ROWS_TOTAL - Q_ROWS_LIVE) * Q_BYTES_PER_ROW)
    config = {
        "host": HOST,
        "experiment": "e95-rider-q-row-shrink",
        "candidate_commit": meta.get("commit"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "head_provenance_sha256": meta.get("head_provenance_sha256"),
        "fixture": meta.get("fixture"),
        "golden_sha256": meta.get("golden_sha256"),
        "vendored_metal_fingerprint": meta.get("vendored_metal_fingerprint"),
        "chip": meta.get("chip"),
        "tokens": exact.get("tokens"),
        "mtp_depth": int(meta.get("depth", 0)),
        "dirty_candidate_paths": int(meta.get("dirty_candidate_paths", -1)),
        "base_census_leg": base_leg,
        "candidate_census_leg": cand_leg,
        "base_census_worker_sha256": base_meta.get("worker_sha256"),
        "candidate_census_worker_sha256": cand_meta.get("worker_sha256"),
    }
    config.update(gate_flags())

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="rider-exactness", name="e95-rider-q-row-shrink",
        config=config, reinit=True)

    checks = exact["checks"]
    run.summary.update({
        "exact/passed": exact["passed"],
        "exact/gate_exit": int(exact["gate_exit"]),
        "exact/control_exit": int(exact["control_exit"]),
        "exact/tokens": exact["tokens"],
        "exact/all_tokens_matched": exact["all_tokens_matched"],
        "exact/residual_divergence_count": exact["residual_divergence_count"],
        "exact/parity_all_ok": exact["parity_all_ok"],
        # A positive control that still matches would prove the comparison
        # cannot fail, so the gate is only meaningful when this is false.
        "exact/control_all_tokens_matched":
            exact["control_all_tokens_matched"],
        "exact/positive_control_rejects": checks["positive_control_rejects"],
        "exact/post_eos_continuation": checks["post_eos_continuation"],
        "exact/first_eos_index_in_window": exact["first_eos_index_in_window"],
        "exact/tokens_after_first_eos": exact["tokens_after_first_eos"],
        "exact/declared_rows_total": exact["declared_rows_total"],
        "exact/reference_checked_row_total":
            exact["reference_checked_row_total"],
        "exact/row_ledger_rows": exact["row_ledger_rows"],
        "exact/ledger_equals_declared": checks["ledger_equals_declared"],
        "exact/checked_equals_declared": checks["checked_equals_declared"],
        "exact/emitted_equals_window": checks["emitted_equals_window"],
        "exact/rows_per_token": exact["rows_per_token"],
        "exact/round_count": exact["round_count"],
        "exact/rejected_draft_total": exact["rejected_draft_total"],
        "exact/verify_block_replayed_round_count":
            exact["verify_block_replayed_round_count"],
        "exact/target_cache_offset_final": exact["target_cache_offset_final"],
        "exact/max_rejected_tail_logit_delta":
            exact["max_rejected_tail_logit_delta"],
        "rider/q_rows_before": Q_ROWS_TOTAL,
        "rider/q_rows_after": Q_ROWS_LIVE,
        "rider/q_island_rows": Q_ROWS_ISLAND,
        "rider/q_bytes_before": int(Q_ROWS_TOTAL * Q_BYTES_PER_ROW),
        "rider/q_bytes_after": int(Q_ROWS_LIVE * Q_BYTES_PER_ROW),
        "rider/q_bytes_saved": saved_bytes,
        "rider/head_weight_stream_share_pct":
            100.0 * saved_bytes / HEAD_WEIGHT_BYTES_PER_CALL,
        "census/base_effective_mean_draft_len":
            base_score["effective_mean_draft_len"],
        "census/candidate_effective_mean_draft_len":
            cand_score["effective_mean_draft_len"],
        "census/base_accepted_draft_rate": base_score["accepted_draft_rate"],
        "census/candidate_accepted_draft_rate":
            cand_score["accepted_draft_rate"],
        "census/draft_proposals_identical":
            base_score["effective_mean_draft_len"]
            == cand_score["effective_mean_draft_len"]
            and base_score["accepted_draft_rate"]
            == cand_score["accepted_draft_rate"],
        "census/base_all_tokens_matched": base_score["all_tokens_matched"],
        "census/candidate_all_tokens_matched": cand_score["all_tokens_matched"],
    })

    shape_rows = []
    for shape in sorted(set(base_shapes) | set(cand_shapes)):
        shape_rows.append([
            shape,
            base_shapes.get(shape, 0.0),
            cand_shapes.get(shape, 0.0),
            cand_shapes.get(shape, 0.0) - base_shapes.get(shape, 0.0),
        ])
    run.log({"draft_head_shapes": wandb.Table(
        columns=["shape", "base_per_round", "candidate_per_round", "delta"],
        data=shape_rows)})
    print(f"rider\t{run.id}\t{run.url}")
    run.finish()


def log_probe(path: pathlib.Path) -> None:
    """Ruling 4: is the per-weight-pass term `b` DRAM traffic?

    The probe times one isolated `quantized_matmul` at many widths on two
    tensors that differ only in byte size, so the width model can be refitted
    where nothing else in the model is running. It measures its own per-eval
    overhead instead of solving for it.
    """
    payload = json.loads(path.read_text())
    overhead = payload["eval_overhead_us"]
    rate = probe.report_reads(payload["reads"], overhead)

    cells: dict[int, dict[int, tuple[float, float]]] = {}
    cell_bytes: dict[int, int] = {}
    for cell in payload["cells"]:
        cells.setdefault(cell["outputs"], {})[cell["m"]] = (
            cell["forward_us"], cell["reverse_us"])
        cell_bytes[cell["outputs"]] = cell["packed_bytes"]
    big, small = sorted(cell_bytes, reverse=True)
    fit = {
        outputs: probe.report_tensor(
            outputs, cells[outputs], cell_bytes[outputs], overhead,
            cell_bytes[outputs] / (rate[outputs] * 1e3))
        for outputs in (big, small)
    }

    stream_mb = probe.VERIFY_STREAM_BYTES / 1e6
    predicted_b = fit[big]["b_ns_per_mb"] * stream_mb / 1e3
    predicted_c = fit[big]["c_ns_per_mb"] * stream_mb / 1e3
    one_pass = probe.VERIFY_STREAM_BYTES / (rate[big] * 1e3)

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        job_type="qmv-width-probe", name="e95-qmv-width-probe",
        config={
            "host": HOST,
            "experiment": "e95-ruling4-b-is-not-one-pass",
            "kernel": "qmv_fast_crossrow_affine4_g64_wide",
            "big_outputs": big,
            "small_outputs": small,
            "big_packed_bytes": cell_bytes[big],
            "small_packed_bytes": cell_bytes[small],
            "verify_stream_bytes": probe.VERIFY_STREAM_BYTES,
            "in_model_a_us": probe.IN_MODEL["a"],
            "in_model_b_us": probe.IN_MODEL["b"],
            "in_model_c_us": probe.IN_MODEL["c"],
            **gate_flags(),
        },
        reinit=True)

    metrics = {
        "probe/eval_overhead_us": overhead,
        "probe/read_gb_s_big": rate[big],
        "probe/read_gb_s_small": rate[small],
        "probe/cache_speedup_x": rate[small] / rate[big],
        "probe/traffic_share_b":
            probe.traffic_share(fit[big], fit[small], "b", rate[big], rate[small]),
        "probe/traffic_share_c":
            probe.traffic_share(fit[big], fit[small], "c", rate[big], rate[small]),
        "transfer/b_predicted_us": predicted_b,
        "transfer/b_error_pct":
            100 * (predicted_b - probe.IN_MODEL["b"]) / probe.IN_MODEL["b"],
        "transfer/c_predicted_us": predicted_c,
        "transfer/c_error_pct":
            100 * (predicted_c - probe.IN_MODEL["c"]) / probe.IN_MODEL["c"],
        "reach/mandatory_single_pass_us": one_pass,
    }
    for tag, outputs in (("big", big), ("small", small)):
        for key in ("a", "b", "c", "one_pass_us", "b_over_pass",
                    "b_ns_per_mb", "c_ns_per_mb", "m1_over_pass"):
            metrics[f"fit_{tag}/{key}"] = fit[outputs][key]
    for prompt, spec in probe.RANKED.items():
        phase = (probe.IN_MODEL["a"]
                 + probe.IN_MODEL["b"] * spec["groups"]
                 + probe.IN_MODEL["c"] * spec["mean_m"])
        metrics[f"reach/{prompt}_phase_us"] = phase
        metrics[f"reach/{prompt}_mandatory_pass_pct"] = 100 * one_pass / phase
        metrics[f"reach/{prompt}_qmv_term_pct"] = (
            100 * (phase - probe.IN_MODEL["a"]) / phase)
        metrics[f"reach/{prompt}_non_qmv_fixed_pct"] = (
            100 * probe.IN_MODEL["a"] / phase)
    run.summary.update(metrics)

    ladder = []
    for outputs in (big, small):
        a, b, c = fit[outputs]["a"], fit[outputs]["b"], fit[outputs]["c"]
        for width in sorted(cells[outputs]):
            forward, reverse = cells[outputs][width]
            net = (forward + reverse) / 2 - overhead
            modelled = a + b * probe.groups(width) + c * width
            ladder.append([
                outputs, width, probe.inputs_per_group(width),
                probe.groups(width), forward, reverse,
                100 * (reverse - forward) / forward, net, modelled,
                100 * (net - modelled) / net,
                net / fit[outputs]["one_pass_us"],
                width in probe.MODEL_WIDTHS])
    run.log({"width_ladder": wandb.Table(
        columns=["outputs", "m", "inputs_per_group", "groups", "forward_us",
                 "reverse_us", "drift_pct", "net_us", "fit_us", "residual_pct",
                 "net_over_one_pass", "in_fit"],
        data=ladder)})

    byte_cut = 1.0 - fit[small]["bytes"] / fit[big]["bytes"]
    elasticity = []
    for width in probe.MODEL_WIDTHS:
        big_net = sum(cells[big][width]) / 2 - overhead
        small_net = sum(cells[small][width]) / 2 - overhead
        time_cut = 1.0 - small_net / big_net
        elasticity.append(
            [width, big_net, small_net, time_cut, time_cut / byte_cut])
    run.log({"byte_elasticity": wandb.Table(
        columns=["m", "big_net_us", "small_net_us", "time_cut",
                 "elasticity"],
        data=elasticity)})
    run.summary["reach/byte_cut"] = byte_cut
    run.summary["reach/mean_elasticity"] = (
        sum(row[4] for row in elasticity) / len(elasticity))

    reads = [[entry["outputs"], entry["packed_bytes"],
              probe.weight_bytes(entry["packed_bytes"]), entry["raw_us"],
              entry["raw_us"] - overhead, rate[entry["outputs"]]]
             for entry in payload["reads"]]
    run.log({"read_rates": wandb.Table(
        columns=["outputs", "packed_bytes", "weight_bytes", "raw_us",
                 "net_us", "gb_per_s"],
        data=reads)})

    print(f"probe\t{run.id}\t{run.url}")
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_census = sub.add_parser("census")
    p_census.add_argument("specs", nargs="+")
    sub.add_parser("model")
    p_rider = sub.add_parser("rider")
    p_rider.add_argument("--exact", required=True, type=pathlib.Path)
    p_rider.add_argument("--base", required=True)
    p_rider.add_argument("--cand", required=True)
    p_probe = sub.add_parser("probe")
    p_probe.add_argument(
        "--json", type=pathlib.Path,
        default=pathlib.Path("research/out/e95_qmv_probe.json"))
    args = parser.parse_args()

    if args.command == "census":
        log_census(args.specs)
    elif args.command == "model":
        log_model()
    elif args.command == "probe":
        log_probe(args.json)
    else:
        log_rider(args.exact, args.base, args.cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
