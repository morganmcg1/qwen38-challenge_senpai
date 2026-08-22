#!/usr/bin/env python3
"""Publish the E137 cliff-family attribution to W&B, one run per rung.

    usage: research/e137_wandb_log.py --rung 0|1|2|3|4 [--dry]

E137 asks which dispatch family carries the M=5 -> M=6 verify cost step. Only
rung 2 loads the model and runs the GPU, and even that leg is a routing census
rather than a timing measurement: it runs with `MLX_E120_QMV_PIPELINE_LOG` set
and its per-round times are contaminated by the counter. No rung here is a
gated measurement or a score, so every run publishes
`cool_gate_passed_real_gate`, `gate_qualified_for_timing` and
`official_or_ranked_score` verbatim as false.

FINDING 181 applies to every QMV number below: this host is `applegpu_g16s`
with a 96-register ceiling and the ranked host is g17s at 126, so the routed
QMV curve is host-specific and is labelled `transfer_safe = false`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e137-cliff-family-attribution"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
BASE_SHA = "33ce6a3f478043d168dc74e7322e754b3021d620"
ART = pathlib.Path("research/e137-artifacts")

# F2 section 6 pre-registered the decision rule before item 2R was measured.
CARRIER_THRESHOLD = 0.60
PARTIAL_THRESHOLD = 0.20
# F3 section 3 wrote down the advisor's own prediction for the same number.
PREDICTION_LO = 0.12
PREDICTION_HI = 0.32
# 257 linear cells per target forward: 48+48 GDN, 16+16 full attention,
# 64+64 MLP, 1 lm_head.
CELLS_PER_FORWARD = 257

RUNGS = {
    "0": {
        "run_name": "e137-rung0-cliff-gate-repair",
        "files": ["item0-before.json", "item0-after-pass.json",
                  "item0-polarity.json"],
        "job_type": "static-gate",
        "question":
            "does the entry-point cliff census fail closed, and can three "
            "deliberately damaged polarities each make it fail",
        "command": "senpai/entry-point-cliff-census.sh --json",
    },
    "1": {
        "run_name": "e137-rung1-width-keyed-round-parts",
        "files": ["item1-width-table.json"],
        "job_type": "offline-rekey",
        "question":
            "where inside a round does the M=5 to M=6 step land, and is any "
            "part of it host CPU rather than GPU",
        "command": "python3 research/e137_width_table.py",
    },
    "2": {
        "run_name": "e137-rung2-route-b-coverage-census",
        "files": ["f2-step1-route-b-coverage.json"],
        "job_type": "routing-census",
        "question":
            "does the scored 512-token leg actually route width-6 verify "
            "cells through Route B, or does width 6 silently decline",
        "command": "research/e137_pipeline_census.sh e137pipe512 512",
    },
    "3": {
        "run_name": "e137-rung3-route-b-cost-curve-attribution",
        "files": ["item2-routeb-curve.json", "items-2-3-attribution.json"],
        "job_type": "isolated-cost-curve",
        "question":
            "how much of the in-situ M=5 to M=6 round step is reproduced by "
            "the isolated, dispatch-count-weighted Route B QMV curve",
        "command":
            "research/e137_routeb_curve.sh && "
            "python3 research/e137_cliff_attribution.py",
    },
    "4": {
        "run_name": "e137-rung4-dispatch-bracket-and-falsifications",
        "files": ["items-2-3-attribution.json"],
        "job_type": "offline-arithmetic",
        "question":
            "can the extra SDPA-chunk dispatches at qL >= 6 price the step, "
            "and does the E92 pinned-width ledger place the cliff at the "
            "same boundary",
        "command": "python3 research/e137_cliff_attribution.py",
    },
}


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten("%s/%s" % (prefix, key) if prefix else str(key), sub, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def table(columns: list[str], rows: list[dict]) -> wandb.Table:
    t = wandb.Table(columns=columns)
    for row in rows:
        t.add_data(*[row.get(c) for c in columns])
    return t


def rung0() -> tuple[dict, dict]:
    before = load("item0-before.json")
    after = load("item0-after-pass.json")
    polarity = load("item0-polarity.json")

    cells = []
    for cell in after["cells"]:
        cells.append({k: v for k, v in cell.items()
                      if not isinstance(v, (dict, list))})

    arms = []
    for name, arm in polarity["arms"].items():
        row = {"polarity": name,
               "expected": polarity["expected_verdicts"].get(name),
               "verdict": polarity["verdicts"].get(name)}
        row["agrees"] = row["expected"] == row["verdict"]
        arms.append(row)

    summary = {
        "e137_cliff_gate_failing_polarity_demonstrated":
            polarity["failing_polarity_demonstrated"],
        "gate_verdict_on_current_base": after["verdict"],
        "gate_failures": len(after["failures"]),
        "gate_warnings": len(after["warnings"]),
        "gate_cells": len(after["cells"]),
        "gate_runtime_seconds": after["runtime_seconds"],
        "gate_polarities_agreeing_with_expectation":
            sum(1 for a in arms if a["agrees"]),
        "gate_polarities_total": len(arms),
        "gate_before_repair_emitted_cells": len(before.get("cells", [])),
        "ranked_arch": after["ranked_arch"],
        "toolchain": after["toolchain"],
    }
    flatten("simdgroup_budget", after["simdgroup_budget"], summary)
    flatten("width_histogram", after["width_histogram"], summary)
    return summary, {"gate_cells": table(sorted({k for c in cells for k in c}),
                                         cells),
                     "gate_polarities": table(
                         ["polarity", "expected", "verdict", "agrees"], arms)}


def rung1() -> tuple[dict, dict]:
    payload = load("item1-width-table.json")
    design = payload["designs"]["all"]["width_table"]

    widths = sorted(design, key=int)
    rows = []
    for w in widths:
        seg = design[w]["segments"]
        row = {"width": int(w), "n": seg["round_us"]["n"]}
        for name, stats in seg.items():
            row["%s_mean" % name] = stats["mean"]
            row["%s_median" % name] = stats["median"]
            row["%s_sd" % name] = stats["sd"]
        rows.append(row)

    steps = []
    for lo, hi in zip(widths, widths[1:]):
        if int(hi) != int(lo) + 1:
            continue
        lo_round = design[lo]["segments"]["round_us"]["mean"]
        hi_round = design[hi]["segments"]["round_us"]["mean"]
        lo_cpu = design[lo]["segments"].get("host_thread_cpu_us", {})
        hi_cpu = design[hi]["segments"].get("host_thread_cpu_us", {})
        step = hi_round - lo_round
        cpu_step = (hi_cpu.get("mean", 0.0) - lo_cpu.get("mean", 0.0))
        steps.append({
            "boundary": "%s->%s" % (lo, hi),
            "round_step_us": step,
            "relative_step": step / lo_round,
            "host_cpu_step_us": cpu_step,
            "host_cpu_share_of_step": cpu_step / step if step else None,
            "gpu_share_of_step": 1.0 - (cpu_step / step) if step else None,
        })

    cliff = next(s for s in steps if s["boundary"] == "5->6")
    summary = {
        "item1_rounds": payload["rounds"],
        "item1_legs": len(payload["source_legs"]),
        "item1_tokens_per_leg": payload["tokens_per_leg"],
        "item1_cliff_step_us": cliff["round_step_us"],
        "item1_cliff_relative_step": cliff["relative_step"],
        # FINDING 187. The retracted first headline claimed 43 % non-QMV from
        # a 13-segment host split. `host_thread_cpu_ns` shows the host thread
        # runs for only a small part of a round, so the step is GPU.
        "item1_cliff_host_cpu_step_us": cliff["host_cpu_step_us"],
        "item1_cliff_host_cpu_share_of_step": cliff["host_cpu_share_of_step"],
        "item1_cliff_gpu_share_of_step": cliff["gpu_share_of_step"],
        "item1_first_headline_retracted": True,
        "item1_retraction_note":
            "the 43 % non-QMV headline was withdrawn; the 13-segment host "
            "split cannot attribute a GPU step",
        "chip": payload["chip"],
    }
    flatten("width_histogram", payload["width_histogram"], summary)
    return summary, {
        "item1_width_table": table(sorted({k for r in rows for k in r}), rows),
        "item1_steps": table(
            ["boundary", "round_step_us", "relative_step", "host_cpu_step_us",
             "host_cpu_share_of_step", "gpu_share_of_step"], steps),
    }


def rung2() -> tuple[dict, dict]:
    payload = load("f2-step1-route-b-coverage.json")
    rows = payload["per_width"]
    summary = {k: v for k, v in payload.items()
               if not isinstance(v, (dict, list))}
    summary["cells_per_forward"] = CELLS_PER_FORWARD
    for section in ("warmup_gate", "verdict", "leg", "route"):
        flatten(section, payload.get(section), summary)
    return summary, {"route_b_coverage": table(
        sorted({k for r in rows for k in r}), rows)}


def rung3() -> tuple[dict, dict]:
    attrib = load("items-2-3-attribution.json")
    transfer = attrib["boundary_4_transfer"]
    routed = attrib["isolated_routed"]
    fallback = attrib["isolated_fallback"]

    per_shape = []
    for shape, stats in routed["per_shape"].items():
        disp = stats["dispatches_per_round"]
        m5 = stats["per_call_us"]["5"]["point"]
        m6 = stats["per_call_us"]["6"]["point"]
        step = stats["per_call_step_5_to_6_us"]
        per_shape.append({
            "shape": shape,
            "n": stats["n"],
            "k": stats["k"],
            "dispatches_per_round": disp,
            "m5_us_per_call": m5,
            "m6_us_per_call": m6,
            "step_us_per_call": step["point"],
            "step_ci_lo": step["ci_lo"],
            "step_ci_hi": step["ci_hi"],
            # F3 section 4 asked for fractions, not only microseconds: a
            # common fraction means no family owns the step.
            "relative_step": step["point"] / m5,
            "weighted_step_us": disp * step["point"],
        })
    per_shape.sort(key=lambda r: -r["weighted_step_us"])
    rel = [r["relative_step"] for r in per_shape]

    curve = []
    for width in routed["widths"]:
        w = str(width)
        curve.append({
            "width": width,
            "routed_us": routed["curve_us"][w]["point"],
            "routed_ci_lo": routed["curve_us"][w]["ci_lo"],
            "routed_ci_hi": routed["curve_us"][w]["ci_hi"],
            "fallback_us": fallback["curve_us"][w]["point"],
            "weight_passes": attrib["route_b_config"]
            ["weight_passes_by_width"][w],
        })

    lvr = []
    for boundary, stats in attrib["local_versus_ranked_relative_steps"][
            "steps"].items():
        row = {"boundary": boundary}
        row.update({k: v for k, v in stats.items()
                    if not isinstance(v, (dict, list))})
        lvr.append(row)

    measured = transfer["isolated_over_insitu"]
    summary = {
        "e137_isolated_to_insitu_transfer_at_boundary_4": measured,
        "transfer_ci_lo": transfer["share_ci"][0],
        "transfer_ci_hi": transfer["share_ci"][1],
        "e137_non_qmv_share_of_m5_to_m6_step":
            transfer["implied_non_qmv_share"],
        "non_qmv_share_ci_lo": transfer["implied_non_qmv_share_ci"][0],
        "non_qmv_share_ci_hi": transfer["implied_non_qmv_share_ci"][1],
        "isolated_routed_step_us": transfer["isolated_routed_step_us"]["point"],
        "isolated_routed_step_ci_lo":
            transfer["isolated_routed_step_us"]["ci_lo"],
        "isolated_routed_step_ci_hi":
            transfer["isolated_routed_step_us"]["ci_hi"],
        "insitu_round_step_us": transfer["insitu_round_step_us"],
        "f2_decision_rule_verdict": transfer["f2_decision_rule"]["verdict"],
        "f2_decision_rule_carrier": measured >= CARRIER_THRESHOLD,
        "f2_decision_rule_partial":
            PARTIAL_THRESHOLD <= measured < CARRIER_THRESHOLD,
        # F3 section 3 predicted 0.12 to 0.32 and asked to be falsified.
        "f3_prediction_lo": PREDICTION_LO,
        "f3_prediction_hi": PREDICTION_HI,
        "f3_prediction_falsified": not (
            PREDICTION_LO <= measured <= PREDICTION_HI),
        "route_b_claims_every_cell": True,
        "route_b_bitwise_match_vs_fallback":
            attrib["route_b_config"]["bitwise_match_vs_fallback"],
        "dispatches_per_round": routed["dispatches_per_round"],
        "reps_per_width": routed["samples"]["5"],
        "per_shape_min_relative_step": min(rel),
        "per_shape_max_relative_step": max(rel),
        # F3 section 4 named `mlp.gate_up` as the shape to watch and quoted a
        # 3.1x step from FINDING 52's pre-Route-B curve.
        "mlp_gate_up_relative_step": next(
            r["relative_step"] for r in per_shape
            if r["shape"] == "mlp.gate_up_fused"),
        "largest_weighted_step_shape": per_shape[0]["shape"],
        "transfer_safe": transfer["transfer_safe"],
        "host_register_ceiling": 96,
        "ranked_register_ceiling": 126,
    }
    # The warm-up census cannot say whether scored M=6 cells route, so the
    # verdict is re-derived against the incumbent MLX gate as well.
    robust = attrib.get("routing_robustness")
    if robust:
        summary.update({
            "fallback_over_insitu": robust["fallback_over_insitu"],
            "routing_worst_case_share": robust["worst_case_share"],
            "routing_best_case_share": robust["best_case_share"],
            "both_arms_show_the_step": robust["both_arms_show_the_step"],
            "verdict_identical_under_either_routing":
                robust["verdict_identical_under_either_routing"],
        })
    return summary, {
        "route_b_per_shape_steps": table(
            ["shape", "dispatches_per_round", "m5_us_per_call",
             "m6_us_per_call", "step_us_per_call", "relative_step",
             "weighted_step_us"], per_shape),
        "route_b_curve": table(
            ["width", "routed_us", "fallback_us", "routed_sd",
             "weight_passes"], curve),
        "local_versus_ranked_relative_steps": table(
            sorted({k for r in lvr for k in r}), lvr),
    }


def rung4() -> tuple[dict, dict]:
    attrib = load("items-2-3-attribution.json")
    pricing = attrib["dispatch_pricing"]
    moving = attrib["sdpa_chunk_moving_cliff"]
    e92 = attrib["e92_gpu_interval_ledger"]

    e92_rows = []
    for boundary, stats in e92["steps"].items():
        row = {"boundary": boundary}
        row.update({k: v for k, v in stats.items()
                    if not isinstance(v, (dict, list))})
        e92_rows.append(row)

    summary = {
        "e137_sdpa_dispatch_step_us_bracket_low":
            pricing["boundary_4_bracket_us"]["low"],
        "e137_sdpa_dispatch_step_us_bracket_high":
            pricing["boundary_4_bracket_us"]["high"],
        "sdpa_bracket_share_of_local_step_pct_low":
            pricing["boundary_4_share_of_insitu_step_pct"]["low"],
        "sdpa_bracket_share_of_local_step_pct_high":
            pricing["boundary_4_share_of_insitu_step_pct"]["high"],
        "sdpa_bracket_share_of_ranked_step_pct_low":
            pricing["boundary_4_share_of_ranked_step_pct"]["low"],
        "sdpa_bracket_share_of_ranked_step_pct_high":
            pricing["boundary_4_share_of_ranked_step_pct"]["high"],
        "sdpa_chunk_only_bracket_us_low":
            pricing["sdpa_chunk_only_bracket_us"]["low"],
        "sdpa_chunk_only_bracket_us_high":
            pricing["sdpa_chunk_only_bracket_us"]["high"],
        "dispatch_count_eliminated": True,
        # The second, independent falsification: a fixed `qL >= 6` guard
        # cannot move a cliff by one width, and the E92 pinned-width ledger
        # puts the largest step at 4->5 on an older base.
        "sdpa_guard_diff_lines_vs_e92_base":
            moving["file_identity"]["git_diff_lines_changed"],
        "cliff_position_current_base":
            moving["cliff_position"]["current_base_33ce6a3f"],
        "cliff_position_e92_base": moving["cliff_position"]["e92_base_b5cff751"],
        "sdpa_chunk_eliminated_as_carrier": True,
        "e92_largest_verify_gpu_busy_step": e92["largest_verify_gpu_busy_step"],
        "e92_largest_verify_gpu_busy_step_us":
            e92["largest_verify_gpu_busy_step_us"],
        "e92_width_pinned": e92["width_pinned"],
        "e92_base_sha": e92["base_sha"],
    }
    flatten("dispatch_tax_ns", pricing["dispatch_tax_ns_per_dispatch"], summary)
    flatten("sdpa_chunk", pricing["sdpa_chunk"], summary)
    return summary, {"e92_pinned_width_steps": table(
        sorted({k for r in e92_rows for k in r}), e92_rows)}


BUILDERS = {"0": rung0, "1": rung1, "2": rung2, "3": rung3, "4": rung4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=sorted(RUNGS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    spec = RUNGS[args.rung]
    summary, tables = BUILDERS[args.rung]()
    summary.update({
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
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
                     name=spec["run_name"], job_type=spec["job_type"],
                     config={"experiment": "E137", "rung": args.rung,
                             "base_sha": BASE_SHA, "host": HOST,
                             "pr": 137,
                             "command": spec["command"],
                             "question": spec["question"]})
    for name, value in tables.items():
        run.log({name: value})
    run.summary.update(summary)
    artifact = wandb.Artifact("e137-rung%s" % args.rung, type=spec["job_type"])
    for name in spec["files"]:
        artifact.add_file(str(ART / name))
    run.log_artifact(artifact)
    print(run.url)
    print(run.id)
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
