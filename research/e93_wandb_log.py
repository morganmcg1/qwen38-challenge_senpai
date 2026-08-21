#!/usr/bin/env python3
"""Publish the E93 proposal-head dispatch census to W&B.

    usage: research/e93_wandb_log.py TAG [TAG ...] [--summary]

One run per census leg carries that leg's identity tuple, its local score
metrics, its host-state stratum and its head-dispatch counters. `--summary`
adds one analysis run that carries the per-dispatch census, the per-command
-buffer cost table, the class rollup and the rung-4 arm pricing.

Every leg in this experiment is a counting and GPU-clock attribution leg, not
a timed arm contrast, so `timing_valid`, `cool_gate_passed_real_gate` and
`gate_qualified_for_timing` are logged as false verbatim on every run and the
absence of an entry and exit GPU temperature record is logged explicitly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e93_head_census as census  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e93-per-draft-proposal-head-dispatch-census"
HOST = "apple-m4-pro-applegpu_g16s-20core-48gib"
OUT = pathlib.Path("research/out")

# Per marginal draft, capacity 768. Sources are named in the result report.
DISPATCH_CENSUS = [
    # class, count, kernel, tensor, weight bytes, activation bytes
    (1, 3, "affine_qmv_fast b_4 grid=1x640x1", "fc / o_proj / down_proj", 97_320_960, 0),
    (1, 1, "affine_qmv_fast b_4 grid=1x1536x1", "q_proj", 35_389_440, 0),
    (1, 1, "affine_qmv_fast b_4 grid=1x4352x1", "mlp gate_up fused", 100_270_080, 0),
    (1, 1, "gemv_al grid=64x1x1", "Q precision island dense bf16 [1024,5120]", 10_485_760, 0),
    (1, 1, "gemv_al grid=128x1x1", "K/V exact dense bf16 [2048,5120]", 20_971_520, 0),
    (2, 1, "qwen35_attention_qk_rms_rope_bf16_v1 grid=1792x1x1", "q_norm + k_norm + RoPE", 1_024, 0),
    (2, 2, "gg2_copy grid=256x4x1", "head K and V append, 1 position", 0, 4_096),
    (2, 1, "sdpa_vector_bf16_256_256_nomask_qnt_nc_nosinks grid=24x1x1", "SDPA over head history", 0, 2_097_152),
    (2, 2, "vn_copy grid=196608x1x1 tg=1024x1x1", "head KV cache FULL-ARRAY copy", 0, 6_291_456),
    (3, 1, "qwen35_embed_dual_rms_norm_concat_bf16_v1 grid=2048x1x1", "fused quantized embed + dual RMSNorm", 23_360, 0),
    (3, 1, "qwen35_fused_residual_rms_norm grid=1024x1x1", "post_attention_layernorm", 10_240, 0),
    (3, 2, "rms_looped grid=1024x1x1", "input_layernorm and mtp.norm", 20_480, 0),
    (3, 1, "vv_Add grid=5120x1x1", "attention residual add", 0, 30_720),
    (3, 1, "CV2ISigmoid...contiguous grid=17408x1x1", "SwiGLU silu(gate) * up", 0, 104_448),
    (3, 1, "CV2ISigmoid...strided_2 grid=256x24x1", "attention output sigmoid gate", 0, 36_864),
    (3, 1, "g1_copy grid=17408x1x1", "gate/up split copy", 0, 69_632),
    (4, 1, "affine_qmv_fast b_2 grid=1x12292x1", "draft_lm_head affine-2 [98336,5120]", 157_337_600, 0),
    (4, 1, "draft_top32_partial grid=16384x1x1", "top-32 partial over 98336 logits", 0, 196_672),
    (4, 1, "draft_top32_finalize grid=256x1x1", "top-32 finalize", 0, 65_536),
    (4, 1, "affine_gather_qmv grid=1x1x32", "gatherQuantizedMM over 32 target head rows", 92_160, 0),
    (4, 1, "draft_rerank__98304_149740 grid=32x1x1", "exact rerank over the 32 candidates", 0, 256),
    (5, 1, "scatter_axis bfloat16 int32 none grid=1x1024x1", "replaceExactRows putAlong, 1024 Q island rows", 0, 24_576),
]

# Isolated leg e93-gpu-iso2-d8, per buffer occurrence. Reproduce with
#   research/e93_head_census.py buffers research/out/e93-gpu-iso2-d8/census.jsonl
BUFFER_COSTS = [
    ("draft_lm_head affine-2 + top-32 partial", 994.81, 157_337_600),
    ("mlp gate_up fused + post_attention_layernorm", 416.30, 100_270_080),
    ("down_proj + SwiGLU + gate/up split copy", 211.83, 50_135_040),
    ("q_proj + replaceExactRows putAlong", 154.07, 35_389_440),
    ("K/V exact dense + Q island dense (gemv_al x2)", 127.78, 31_457_280),
    ("fc + input_layernorm + mtp.norm", 128.73, 29_491_200),
    ("o_proj + attention output sigmoid gate", 77.26, 17_694_720),
    ("gatherQuantizedMM + top-32 finalize", 29.64, 92_160),
    ("head K/V append + SDPA over head history", 33.58, 0),
    ("fused embed dual RMSNorm + exact rerank", 10.95, 23_360),
    ("qk_rms_rope + head K/V append", 8.26, 1_024),
    ("mtp.norm + attention residual add", 6.16, 10_240),
]

CLASS_ROLLUP = [
    (1, "weight-streaming GEMV", 7, 1100.0, 264_437_760),
    (4, "readout and rerank", 5, 1027.3, 157_429_760),
    (2, "attention over head history", 6, 68.0, 8_425_984),
    (3, "norms and elementwise", 8, 31.0, 395_744),
    (5, "island scatter", 1, 2.0, 24_576),
]

# ranked % of candidate time = (local delta us / local head pass us) * 6.3 %
HEAD_SHARE_RANKED = 0.063
MARGINAL_DRAFT_US = 2226.5

# Ranked prices are modelled on beagle and essays, the 4th and 5th sorted
# prompts that set the published median. The model is calibrated against the
# ranked increment ox-alpha measured for the Q-row shrink.
ARMS = [
    ("q-row-shrink", "delete 1024 dead q_proj output rows", 12.84, 0.0351, 0.0353,
     "rider; deletion; ranked-proven by the ox-alpha crown"),
    ("head-kv-cache-full-array-copy", "stop the capacity-sized head KV cache copy",
     30.3, 0.0640, 0.0669,
     "PROPOSED; deletion in effect; implementation route must pass the restructuring test"),
    ("lm-head-affine2-rate", "raise the affine-2 readout above 158 GB/s", 334.0, 0.83, 0.83,
     "RESTRUCTURING - reported, explicitly NOT proposed"),
]


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key:
            meta[key.strip()] = value.strip()
    return meta


def host_state(records: list, gate_us: float = census.HOST_PHASE_GATE_US) -> dict:
    out = {}
    rounds = [r for r in records if r.get("event") == "round"]
    by_pid = {}
    for record in rounds:
        by_pid.setdefault(record["pid"], []).append(record)
    for pid, leg in by_pid.items():
        leg = sorted(leg, key=lambda r: r["round"])[1:]
        for phase in {name for r in leg for name in r["phases"]}:
            host = [r["phases"][phase]["dispatch_ns"] / 1000.0
                    for r in leg if phase in r["phases"]]
            disp = [r["phases"][phase]["dispatches"]
                    for r in leg if phase in r["phases"]]
            if not host:
                continue
            clean = [v for v in host if v <= gate_us]
            key = f"host_state/{phase}"
            out[f"{key}/rounds"] = len(host)
            out[f"{key}/clean_rounds"] = len(clean)
            out[f"{key}/dirty_rounds"] = len(host) - len(clean)
            out[f"{key}/host_us_max"] = max(host)
            out[f"{key}/host_us_mean_clean"] = sum(clean) / len(clean) if clean else float("nan")
            out[f"{key}/dispatches_min"] = min(disp)
            out[f"{key}/dispatches_max"] = max(disp)
            out[f"{key}/small_clean_sample"] = len(clean) < 20
    return out


FIRST_CALL = {0: 25, 1: 33}
FIRST_CALL_SATURATED = 37
MARGINAL_DISPATCHES = 27
CAPACITY_GROWTH_DISPATCHES = 6


def head_counters(records: list) -> dict:
    """Head dispatch counts and the residual of the census dispatch model.

    A round issues one first head call plus one marginal call per extra draft.
    `width` is the verified row count, so the draft count is `width - 1`. The
    first call flushes every token committed since the previous head call, so
    its dispatch count depends on the previous round's accepted drafts and
    saturates at two or more rows.
    """
    out = {}
    rounds = [r for r in records if r.get("event") == "round"
              and "draft_head" in r.get("phases", {})]
    samples = []
    for index in range(1, len(rounds)):
        record = rounds[index]
        drafts = record["width"] - 1
        if drafts < 1:
            continue
        previous = rounds[index - 1]["accepted"]
        first = FIRST_CALL.get(previous, FIRST_CALL_SATURATED)
        predicted = first + MARGINAL_DISPATCHES * (drafts - 1)
        samples.append((record["phases"]["draft_head"]["dispatches"], predicted))
    if not samples:
        return out
    residual = [observed - predicted for observed, predicted in samples]
    growth = [v for v in residual if v == CAPACITY_GROWTH_DISPATCHES]
    other = [v for v in residual if v != CAPACITY_GROWTH_DISPATCHES]
    out["head/rounds_modelled"] = len(samples)
    out["head/dispatches_mean"] = sum(s[0] for s in samples) / len(samples)
    out["head/marginal_dispatches_per_draft"] = MARGINAL_DISPATCHES
    out["head/first_call_saturated_dispatches"] = FIRST_CALL_SATURATED
    out["head/kv_capacity_growth_rounds"] = len(growth)
    out["head/model_max_abs_residual_excluding_growth"] = (
        max(abs(v) for v in other) if other else 0)
    # A forced one-operation-per-buffer leg loses exactly the two full-array
    # head KV cache copies of every marginal draft.
    deficits = {-v for v in other if v < 0}
    if deficits:
        out["head/dispatch_deficit_vs_default_build"] = max(deficits)
    return out


def log_leg(tag: str, extra: dict) -> str:
    directory = OUT / tag
    meta = read_meta(directory / "meta.txt")
    score = json.loads((directory / "score.json").read_text())
    records = census.load(str(directory / "census.jsonl"))
    config = {
        "experiment": "e93",
        "leg": tag,
        "harness": "local",
        "host": HOST,
        "role": "student-qwen-askeladd",
        "pr": 95,
        "assignment_id": "qwen38-r1-e93-per-draft-proposal-head-dispatch-census",
        "instrument": "E58/E85 dispatch census + MLX_E80_GPU_TIME Metal command-buffer clock",
        "gpu_temperature_recorded": False,
        "arm_contrast_claimed": False,
        **{k: v for k, v in meta.items()},
        **extra,
    }
    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     job_type="census-leg", name=f"e93-{tag}", config=config,
                     reinit=True)
    summary = {f"score/{k}": v for k, v in score["metrics"].items()}
    summary["score/local_speedup"] = score["score"]
    summary["timing_valid"] = False
    summary["cool_gate_passed_real_gate"] = False
    summary["gate_qualified_for_timing"] = False
    summary["official_or_ranked_score"] = False
    summary.update(host_state(records))
    summary.update(head_counters(records))
    run.summary.update(summary)
    run_id = run.id
    run.finish()
    return run_id


def log_summary(leg_ids: dict, resume_id: str | None = None) -> str:
    config = {
        "experiment": "e93",
        "harness": "local",
        "host": HOST,
        "role": "student-qwen-askeladd",
        "pr": 95,
        "assignment_id": "qwen38-r1-e93-per-draft-proposal-head-dispatch-census",
        "legs": json.dumps(leg_ids),
        "marginal_draft_us": MARGINAL_DRAFT_US,
        "head_share_ranked": HEAD_SHARE_RANKED,
    }
    if not leg_ids:
        config.pop("legs")
    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     job_type="analysis", name="e93-summary", config=config,
                     id=resume_id, resume="allow" if resume_id else None,
                     reinit=True)
    census_table = wandb.Table(
        columns=["class", "per_draft", "kernel", "tensor", "weight_bytes", "activation_bytes"],
        data=[list(row) for row in DISPATCH_CENSUS])
    buffer_table = wandb.Table(
        columns=["buffer", "us_per_draft", "weight_bytes", "gb_per_s"],
        data=[[name, us, b, (b / (us * 1e-6) / 1e9) if b and us else 0.0]
              for name, us, b in BUFFER_COSTS])
    class_table = wandb.Table(
        columns=["class", "name", "dispatches", "us_per_draft", "share_pct", "bytes", "gb_per_s"],
        data=[[c, n, d, us, 100.0 * us / MARGINAL_DRAFT_US, b,
               (b / (us * 1e-6) / 1e9) if b and us else 0.0]
              for c, n, d, us, b in CLASS_ROLLUP])
    arm_table = wandb.Table(
        columns=["arm", "mechanism", "local_us_per_head_call",
                 "ranked_pct_beagle", "ranked_pct_essays",
                 "ranked_pct_beagle_essays_mean", "shape"],
        data=[[a, m, us, beagle, essays, 0.5 * (beagle + essays), s]
              for a, m, us, beagle, essays, s in ARMS])
    run.log({
        "dispatch_census": census_table,
        "buffer_costs": buffer_table,
        "class_rollup": class_table,
        "rung4_arm_pricing": arm_table,
    })
    run.summary.update(SUMMARY_SCALARS)
    run_id = run.id
    run.finish()
    return run_id


SUMMARY_SCALARS = {
        "rung1/marginal_dispatches_per_draft": sum(row[1] for row in DISPATCH_CENSUS),
        "rung1/first_head_call_dispatches_prev_accept_ge2": 37,
        "rung1/first_head_call_dispatches_prev_accept_1": 33,
        "rung1/first_head_call_dispatches_prev_accept_0": 25,
        "rung1/kv_capacity_growth_extra_dispatches": 6,
        "rung1/rounds_modelled_zero_residual": 112,
        "rung1/marginal_dispatches_without_full_array_copy": 25,
        "rung1/head_weight_bytes_per_draft": 421_831_680,
        "rung1/head_artifact_bytes": 427_738_112,
        "rung1/target_owned_bytes_per_draft": 95_040,
        "rung1/activation_bytes_per_draft": 9_035_904,
        "rung2/marginal_draft_us_cap768": MARGINAL_DRAFT_US,
        "rung2/first_head_call_us": 2560.0,
        "rung2/head_pass_us_insitu_long": 2276.8,
        "rung2/head_pass_us_insitu_short": 2267.5,
        "rung2/e85_head_pass_us": 2285.283,
        "rung2/closure_vs_e85": 2276.8 / 2285.283,
        "rung2/mtp_leg_target_verify_share": 0.898,
        "rung2/mtp_leg_draft_head_share": 0.088,
        "rung3/class1_gb_per_s": 240.0,
        "rung3/best_within_host_gb_per_s": 246.2,
        "rung3/lm_head_gb_per_s": 158.2,
        "rung3/lm_head_headroom_us_per_draft": 334.0,
        "rung4/vn_copy_us_per_marginal_draft_cap768": 26.0,
        "rung4/vn_copy_us_per_marginal_draft_ranked_mean_capacity": 30.3,
        "rung4/vn_copy_ranked_pct_beagle": 0.0640,
        "rung4/vn_copy_ranked_pct_essays": 0.0669,
        "rung4/vn_copy_ranked_pct_beagle_essays_mean": 0.0654,
        "rung4/q_row_shrink_ranked_pct_beagle": 0.0351,
        "rung4/q_row_shrink_ranked_pct_essays": 0.0353,
        "rung4/q_row_shrink_ranked_pct_beagle_essays_mean": 0.0352,
        "rung4/q_row_shrink_ranked_measured_pct": 0.035,
        "rung4/model_error_vs_ranked_measurement": 0.014,
        "rung4/lm_head_ranked_pct_if_fully_recovered": 0.83,
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
}


def amend_summary(run_id: str) -> None:
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    for key in list(run.summary.keys()):
        if key.startswith("rung1/"):
            del run.summary[key]
    run.summary.update(SUMMARY_SCALARS)
    run.summary.update()
    print(f"amended summary {run_id}")


def amend(pairs: list[tuple[str, str]]) -> None:
    """Replace the head-dispatch counters on runs that are already published."""
    api = wandb.Api()
    for tag, run_id in pairs:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        records = census.load(str(OUT / tag / "census.jsonl"))
        for key in list(run.summary.keys()):
            if key.startswith("head/"):
                del run.summary[key]
        run.summary.update(head_counters(records))
        run.summary.update()
        print(f"amended {tag} {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--amend", metavar="TAG=RUNID", action="append", default=[])
    parser.add_argument("--amend-summary", metavar="RUNID")
    parser.add_argument("--relog-summary", metavar="RUNID")
    args = parser.parse_args()
    if args.relog_summary:
        log_summary({}, resume_id=args.relog_summary)
        print(f"relogged summary {args.relog_summary}")
        return 0
    if args.amend_summary:
        amend_summary(args.amend_summary)
        return 0
    if args.amend:
        amend([item.split("=", 1) for item in args.amend])
        return 0
    ids = {}
    for tag in args.tags:
        ids[tag] = log_leg(tag, {})
        print(f"{tag} -> https://wandb.ai/{ENTITY}/{PROJECT}/runs/{ids[tag]}")
    if args.summary:
        summary_id = log_summary(ids)
        print(f"summary -> https://wandb.ai/{ENTITY}/{PROJECT}/runs/{summary_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
