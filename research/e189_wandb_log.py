#!/usr/bin/env python3
"""Publish the E189 single-pass QMV screen to W&B.

    usage: research/e189_wandb_log.py \
        --gate research/out/e189/screen1/gate.json \
        --probe research/out/e189/screen2/probe.json \
        --ladder research/out/e189/ladder

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
from statistics import median

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e189_analyze import DEPTH_RE, ROUND_RE  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e189-single-pass-qmv-screen"

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
        trace = meta_path.parent / "trace.txt"
        if trace.exists():
            for line in trace.read_text().splitlines():
                if not line.startswith("mtp-trace: round="):
                    continue
                round_match = ROUND_RE.search(line)
                depth_match = DEPTH_RE.search(line)
                if not round_match or not depth_match:
                    continue
                value = int(round_match.group(1)) / 1000.0
                if int(depth_match.group(1)) > 0:
                    drafting.append(value)
                    accepted.append(int(depth_match.group(2)))
                else:
                    serial.append(value)
        score = json.loads(
            (meta_path.parent / "score.json").read_text()
        )["metrics"]
        body = drafting[1:] if len(drafting) > 2 else drafting
        legs.append(
            {
                "leg": int(meta["e189_leg"]),
                "arm": meta["e189_arm"],
                "plan": meta["width_plan"],
                "m": int(meta["verify_width_m"]),
                "rounds": len(body),
                "round_ms_p50": median(body),
                "serial_round_ms_p50": median(serial[1:]),
                "accepted_mean": sum(accepted) / len(accepted),
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
    parser.add_argument("--gate", default="research/out/e189/screen1/gate.json")
    parser.add_argument("--probe", default="research/out/e189/screen2/probe.json")
    parser.add_argument("--ladder", default="research/out/e189/ladder")
    args = parser.parse_args()

    gate = json.loads(pathlib.Path(args.gate).read_text())
    probe = json.loads(pathlib.Path(args.probe).read_text())
    legs = read_legs(pathlib.Path(args.ladder))

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name="e189-single-pass-qmv",
        job_type="analysis",
        config={
            "experiment": "e189-single-pass-qmv-screen",
            "question": (
                "can the fused 4-bit QMV serve m=6..8 in one weight-streaming "
                "pass and remove the +35.6 ms/round group-pass step"
            ),
            "harness": "local",
            "mechanism": "Qwen35QMVWidthPlan: IPG=m at m=6,7,8 so G(m)=1",
            "arm_switch": "MLX_E189_QMV_WIDTH_PLAN",
            "default_plan_on_branch": "staged",
            "mode": "qwen-mtp-local-iterate",
            "tokens": 256,
            "legs": len(legs),
            "pin": "MLX_E159_FIXED_DRAFT_DEPTH, m = 1 + depth",
            "null_control_width": 5,
            "mue_ms_per_round": MUE_MS,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "candidate_head": git("rev-parse", "HEAD"),
            "base_sha": "e0c0dec79f0b813e8c9a7caef07e49149e99176e",
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
            "leg", "arm", "plan", "m", "rounds", "round_ms_p50",
            "serial_round_ms_p50", "accepted_mean", "mtp_s_per_token",
            "serial_s_per_token", "local_ratio", "all_tokens_matched",
            "accepted_draft_rate", "residual_divergence_count", "entry_c",
            "exit_c", "worker_sha256", "worker_digest_stable", "harness",
        ]
    )
    for leg in legs:
        leg_table.add_data(
            leg["leg"], leg["arm"], leg["plan"], leg["m"], leg["rounds"],
            leg["round_ms_p50"], leg["serial_round_ms_p50"],
            leg["accepted_mean"], leg["mtp_s_per_token"],
            leg["serial_s_per_token"], leg["local_ratio"],
            leg["all_tokens_matched"], leg["accepted_draft_rate"],
            leg["residual_divergence_count"], leg["entry_c"], leg["exit_c"],
            leg["worker_sha256"], leg["worker_digest_stable"], "local",
        )

    summary_table = wandb.Table(
        columns=["m", "staged_round_ms", "single_round_ms", "delta_ms",
                 "delta_pct", "staged_s_per_token", "single_s_per_token",
                 "token_delta_pct", "delta_over_mue", "harness"]
    )
    effects: dict[int, dict[str, float]] = {}
    for m in sorted({leg["m"] for leg in legs}):
        staged = [leg for leg in legs if leg["m"] == m and leg["plan"] == "staged"]
        single = [
            leg for leg in legs if leg["m"] == m and leg["plan"] == "singlepass"
        ]
        if not staged or not single:
            continue
        sm = median([leg["round_ms_p50"] for leg in staged])
        pm = median([leg["round_ms_p50"] for leg in single])
        st = median([leg["mtp_s_per_token"] for leg in staged])
        pt = median([leg["mtp_s_per_token"] for leg in single])
        effects[m] = {
            "staged_round_ms": sm,
            "single_round_ms": pm,
            "delta_ms": pm - sm,
            "token_delta_pct": 100.0 * (pt - st) / st,
        }
        summary_table.add_data(
            m, sm, pm, pm - sm, 100.0 * (pm - sm) / sm, st, pt,
            100.0 * (pt - st) / st, (pm - sm) / MUE_MS, "local",
        )

    run.log(
        {
            "gate/per_cell": gate_table,
            "gate/positive_control": control_table,
            "probe/per_cell": cell_table,
            "probe/round_projection": projection_table,
            "ladder/legs": leg_table,
            "ladder/effect_by_width": summary_table,
        }
    )

    run.summary.update(
        {
            "gate_total_differing": gate["total_differing"],
            "gate_worst_max_ulp": gate["worst_max_ulp"],
            "gate_positive_control_tripped": all(
                row["differing"] > 0 for row in gate["positive_control"]
            ),
            "null_control_delta_ms_m5": effects.get(5, {}).get("delta_ms"),
            "delta_ms_m6": effects.get(6, {}).get("delta_ms"),
            "delta_ms_m8": effects.get(8, {}).get("delta_ms"),
            "token_delta_pct_m6": effects.get(6, {}).get("token_delta_pct"),
            "token_delta_pct_m8": effects.get(8, {}).get("token_delta_pct"),
            "probe_round_delta_ms_m7": round_delta.get(7),
            "all_legs_token_matched": all(
                leg["all_tokens_matched"] for leg in legs
            ),
            "legs": len(legs),
            "verdict": "not-useful-route1-all-cells",
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
