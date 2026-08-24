#!/usr/bin/env python3
"""Publish the E195 single-pass QMV screen to W&B.

    usage: research/e195_wandb_log.py \
        --gate research/out/e195/screen1/gate.json \
        --probe research/out/e195/screen2/probe.json \
        --ladder research/out/e195/ladder

Every table is `harness=local`. The timed legs ran with
`MLXFAST_LOCAL_COOL_GATE=0` under the standing counterbalanced-session
allowance, so `cool_gate_passed_real_gate=false` and
`gate_qualified_for_timing=false` are carried verbatim and no number here is an
official or ranked score. The primary read is the ABSOLUTE candidate figure
(drafting-round median and `mtp_seconds_per_token`), never the local
serial-to-MTP ratio.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from statistics import median, stdev

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e195_analyze import DEPTH_RE, PLAN_RE, ROUND_RE, SP_RE  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e195-cell-selective-qmv-plan"

CROWN = 3.7291100105909
RECEIPT_A = 3.70784519415395
MUE_MS = 0.567


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_legs(root: pathlib.Path) -> list[dict]:
    legs = []
    for meta_path in sorted(root.glob("*/leg*/meta.txt")):
        meta = dict(
            line.split("=", 1)
            for line in meta_path.read_text().splitlines()
            if "=" in line
        )
        drafting: list[float] = []
        serial: list[float] = []
        accepted: list[int] = []
        proposed = 0
        traced_plan = None
        qmv_sp = 0
        trace = meta_path.parent / "trace.txt"
        if trace.exists():
            for line in trace.read_text().splitlines():
                if not line.startswith("mtp-trace: round="):
                    continue
                round_match = ROUND_RE.search(line)
                depth_match = DEPTH_RE.search(line)
                if not round_match or not depth_match:
                    continue
                plan_match = PLAN_RE.search(line)
                if plan_match:
                    traced_plan = plan_match.group(1)
                sp_match = SP_RE.search(line)
                if sp_match:
                    qmv_sp = int(sp_match.group(1))
                value = int(round_match.group(1)) / 1000.0
                if int(depth_match.group(1)) > 0:
                    drafting.append(value)
                    accepted.append(int(depth_match.group(2)))
                    proposed += int(depth_match.group(1))
                else:
                    serial.append(value)
        score = json.loads(
            (meta_path.parent / "score.json").read_text()
        )["metrics"]
        body = drafting[1:] if len(drafting) > 2 else drafting
        legs.append(
            {
                "leg": int(meta["e195_leg"]),
                "arm": meta["e195_arm"],
                "plan": meta["width_plan"],
                "m": int(meta["verify_width_m"]),
                "rounds": len(drafting),
                "round_ms_p50": median(body),
                "serial_round_ms_p50": median(serial[1:]),
                "accepted_mean": sum(accepted) / len(accepted),
                "accepted_total": sum(accepted),
                "proposed_total": proposed,
                "traced_plan": traced_plan,
                "qmv_single_pass_dispatches": qmv_sp,
                "mtp_s_per_token": score["mtp_seconds_per_token"],
                "serial_s_per_token": score["serial_seconds_per_token"],
                "local_ratio": score["mtp_decode_speedup"],
                "all_tokens_matched": score["all_tokens_matched"],
                "accepted_draft_rate": score["accepted_draft_rate"],
                "residual_divergence_count": score["residual_divergence_count"],
                "decode_tokens": score["decode_tokens"],
                "entry_c": float(meta["gpu_temp_entry"]),
                "exit_c": float(meta["gpu_temp_exit"]),
                "worker_sha256": meta["worker_sha256_before"],
                "worker_digest_stable": meta["worker_digest_stable"],
                "head_sha": meta["head_sha"],
            }
        )
    return sorted(legs, key=lambda leg: leg["leg"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", default="research/out/e195/screen1/gate.json")
    parser.add_argument("--probe", default="research/out/e195/screen2/probe.json")
    parser.add_argument("--ladder", default="research/out/e195/ladder")
    args = parser.parse_args()

    gate = json.loads(pathlib.Path(args.gate).read_text())
    probe = json.loads(pathlib.Path(args.probe).read_text())
    legs = read_legs(pathlib.Path(args.ladder))

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e195-cell-selective-qmv-plan",
        job_type="analysis",
        config={
            "experiment": "e195-cell-selective-qmv-plan",
            "question": (
                "does a per-(cell, width) QMV plan that uses the single-pass "
                "kernel only where it wins beat BOTH the shipped staged plan "
                "and all-cells single pass at m=6, end to end and bit-exactly"
            ),
            "harness": "local",
            "mechanism": (
                "Qwen35QMVKernelVariant staged|singlePass compiled side by "
                "side; Qwen35QMVWidthPlan maps (cell, m) to one of them. "
                "selective = single pass at m=6 for every cell except "
                "mlp.down; selective7 adds lm_head-only single pass at m=7"
            ),
            "arm_switch": "MLX_E195_QMV_WIDTH_PLAN",
            "default_plan_on_branch": "staged",
            "mode": "qwen-mtp-local-iterate",
            "tokens": 256,
            "legs": len(legs),
            "pin": "MLX_E159_FIXED_DRAFT_DEPTH, m = 1 + depth",
            "null_control_width": 5,
            "mue_ms_per_round": MUE_MS,
            "stop_rule": (
                "STOP if selective m=6 gain over staged < 5 ms/round, or if it "
                "does not beat all-cells by >= 1 ms/round. CONTINUE iff "
                ">= 6 ms/round better than staged AND >= all-cells"
            ),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "candidate_head": git("rev-parse", "HEAD"),
            "base_sha": "a813d18400d34b8879822dccbe5b52981324f300",
            "worktree_dirty": bool(git("status", "--porcelain")),
            "host": "apple-m4-pro",
            "ranked_host": "m5-qwen38-27b-mtp",
            "crown_score": CROWN,
            "receipt_a_score": RECEIPT_A,
        },
    )

    gate_table = wandb.Table(
        columns=[
            "cell", "k", "n", "m", "use_table", "elements", "differing",
            "max_ulp", "max_abs_diff", "non_finite", "groups_staged",
            "groups_single", "harness",
        ]
    )
    for row in gate["cells"]:
        gate_table.add_data(
            row["cell"], row["k"], row["n"], row["m"], row["use_table"],
            row["elements"], row["differing"], row["max_ulp"],
            row["max_abs_diff"], row["non_finite"], row["staged_groups"],
            row["single_pass_groups"], "local",
        )

    control_table = wandb.Table(
        columns=["cell", "m", "use_table", "differing", "max_ulp",
                 "max_abs_diff", "harness"]
    )
    for row in gate["positive_control"]:
        control_table.add_data(
            row["cell"], row["m"], row["use_table"], row["differing"],
            row["max_ulp"], row["max_abs_diff"], "local",
        )

    samples = probe["samples"]
    keys = sorted({(s["cell"], s["m"]) for s in samples})
    cell_table = wandb.Table(
        columns=["cell", "k", "n", "m", "staged_us", "single_us", "delta_us",
                 "delta_pct", "groups_staged", "groups_single",
                 "invocations_per_round", "round_delta_ms", "harness"]
    )
    round_delta: dict[int, float] = {}
    for cell, m in keys:
        pick = [s for s in samples if s["cell"] == cell and s["m"] == m]
        staged = median([s["microseconds"] for s in pick if s["arm"] == "staged"])
        single = median(
            [s["microseconds"] for s in pick if s["arm"] == "singlepass"]
        )
        invocations = pick[0]["invocations_per_round"]
        delta_ms = (single - staged) * invocations / 1000.0
        round_delta[m] = round_delta.get(m, 0.0) + delta_ms
        cell_table.add_data(
            cell, pick[0]["k"], pick[0]["n"], m, staged, single,
            single - staged, 100.0 * (single - staged) / staged,
            [s for s in pick if s["arm"] == "staged"][0]["groups"],
            [s for s in pick if s["arm"] == "singlepass"][0]["groups"],
            invocations, delta_ms, "local",
        )

    projection_table = wandb.Table(
        columns=["m", "projected_round_delta_ms", "harness"]
    )
    for m in sorted(round_delta):
        projection_table.add_data(m, round_delta[m], "local")

    leg_table = wandb.Table(
        columns=[
            "leg", "arm", "plan", "traced_plan", "qmv_single_pass_dispatches",
            "m", "rounds", "round_ms_p50", "serial_round_ms_p50",
            "proposed_total", "accepted_total", "accepted_mean",
            "mtp_s_per_token", "serial_s_per_token", "local_ratio",
            "all_tokens_matched", "accepted_draft_rate",
            "residual_divergence_count", "entry_c", "exit_c", "worker_sha256",
            "worker_digest_stable", "harness",
        ]
    )
    for leg in legs:
        leg_table.add_data(
            leg["leg"], leg["arm"], leg["plan"], leg["traced_plan"],
            leg["qmv_single_pass_dispatches"], leg["m"], leg["rounds"],
            leg["round_ms_p50"], leg["serial_round_ms_p50"],
            leg["proposed_total"], leg["accepted_total"],
            leg["accepted_mean"], leg["mtp_s_per_token"],
            leg["serial_s_per_token"], leg["local_ratio"],
            leg["all_tokens_matched"], leg["accepted_draft_rate"],
            leg["residual_divergence_count"], leg["entry_c"], leg["exit_c"],
            leg["worker_sha256"], leg["worker_digest_stable"], "local",
        )

    summary_table = wandb.Table(
        columns=["m", "plan", "legs", "round_ms_p50", "round_ms_sigma",
                 "staged_round_ms", "delta_vs_staged_ms",
                 "delta_vs_staged_pct", "delta_over_mue", "s_per_token",
                 "staged_s_per_token", "token_delta_pct", "harness"]
    )
    effects: dict[tuple[int, str], dict[str, float]] = {}
    for m in sorted({leg["m"] for leg in legs}):
        at_m = [leg for leg in legs if leg["m"] == m]
        staged = [leg for leg in at_m if leg["plan"] == "staged"]
        if not staged:
            continue
        sm = median([leg["round_ms_p50"] for leg in staged])
        st = median([leg["mtp_s_per_token"] for leg in staged])
        for plan in sorted({leg["plan"] for leg in at_m}):
            arm = [leg for leg in at_m if leg["plan"] == plan]
            values = [leg["round_ms_p50"] for leg in arm]
            pm = median(values)
            pt = median([leg["mtp_s_per_token"] for leg in arm])
            sigma = stdev(values) if len(values) > 1 else 0.0
            effects[(m, plan)] = {
                "round_ms": pm,
                "sigma": sigma,
                "delta_ms": pm - sm,
                "token_delta_pct": 100.0 * (pt - st) / st,
            }
            summary_table.add_data(
                m, plan, len(arm), pm, sigma, sm, pm - sm,
                100.0 * (pm - sm) / sm, (pm - sm) / MUE_MS, pt, st,
                100.0 * (pt - st) / st, "local",
            )

    def delta(m: int, plan: str) -> float | None:
        return effects.get((m, plan), {}).get("delta_ms")

    selective_m6 = delta(6, "selective")
    allcells_m6 = delta(6, "singlepass")
    selective_gain = -selective_m6 if selective_m6 is not None else None
    margin_over_allcells = (
        allcells_m6 - selective_m6
        if selective_m6 is not None and allcells_m6 is not None
        else None
    )
    if selective_gain is None or margin_over_allcells is None:
        verdict = "incomplete"
    elif selective_gain >= 6.0 and margin_over_allcells >= 0.0:
        verdict = "continue-stage3"
    elif selective_gain < 5.0 or margin_over_allcells < 1.0:
        verdict = "stop-not-useful"
    else:
        verdict = "unclear"

    seven_gain = delta(7, "selective7")
    seven_baseline = delta(7, "staged")
    if seven_gain is None or seven_baseline is None:
        seven_verdict = "not-measured"
    elif seven_gain < 0.0:
        seven_verdict = "replicated-include-lm-head-m7"
    else:
        seven_verdict = "not-replicated-exclude-lm-head-m7"

    run.log(
        {
            "gate/per_cell": gate_table,
            "gate/positive_control": control_table,
            "probe/per_cell": cell_table,
            "probe/round_projection": projection_table,
            "ladder/legs": leg_table,
            "ladder/effect_by_width_and_plan": summary_table,
        }
    )

    run.summary.update(
        {
            "gate_total_differing": gate["total_differing"],
            "gate_worst_max_ulp": gate["worst_max_ulp"],
            "gate_positive_control_tripped": all(
                row["differing"] > 0 for row in gate["positive_control"]
            ),
            "null_control_delta_ms_m5_selective": delta(5, "selective"),
            "selective_delta_ms_m6": selective_m6,
            "selective_gain_ms_m6": selective_gain,
            "allcells_delta_ms_m6": allcells_m6,
            "selective_minus_allcells_ms_m6": margin_over_allcells,
            "selective_gain_over_mue": (
                selective_gain / MUE_MS if selective_gain is not None else None
            ),
            "selective_token_delta_pct_m6": effects.get(
                (6, "selective"), {}
            ).get("token_delta_pct"),
            "selective7_delta_ms_m7": seven_gain,
            "m7_lm_head_verdict": seven_verdict,
            "probe_round_delta_ms_m7": round_delta.get(7),
            "all_legs_token_matched": all(
                leg["all_tokens_matched"] for leg in legs
            ),
            "all_legs_plan_witnessed": all(
                leg["traced_plan"] == leg["plan"] for leg in legs
            ),
            "staged_legs_zero_single_pass_dispatch": all(
                leg["qmv_single_pass_dispatches"] == 0
                for leg in legs
                if leg["plan"] == "staged"
            ),
            "legs": len(legs),
            "verdict": verdict,
            "harness": "local",
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        }
    )
    print(f"wandb run: {run.url}")
    print(f"wandb run id: {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
