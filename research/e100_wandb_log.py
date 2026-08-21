#!/usr/bin/env python3
"""Publish the E100 weight-stream experiment to W&B.

    usage: research/e100_wandb_log.py [--only RUN]

Four runs:

  `e100-corpus`    rung 0: the ranked-receipt corpus re-priced, which sets the
                   empirical noise floor for one ranked candidate leg and shows
                   that the ledger's stop entry for this idea rests on a single
                   receipt inside that floor.
  `e100-probe`     rung 2a: the isolated kernel probe. Bit-exactness across the
                   arms, the per-cell speedup at each verify width, and the
                   shared register-allocation tax on the widths whose dispatch
                   lines never changed.
  `e100-e2e`       rung 2b: the end-to-end ABBA sessions on the real decode
                   path, plus the per-round cost model that removes the seed
                   prefill from the leg time.
  `e100-registers` the register census of the SHIPPED dispatcher entry point,
                   which is one Metal kernel holding every width.

Every leg ran with the local cool gate off inside a counterbalanced session, so
`timing_valid`, `cool_gate_passed_real_gate` and `gate_qualified_for_timing` are
logged false verbatim on every run. No number here is an official or ranked
score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e100-fewer-weight-streams-per-round"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
OUT = pathlib.Path("research/out")

BASE_SHA = "cd0a89dadf543261a91eb6cae07c57b3f3282519"
UPSTREAM_NOTE = "senpai/qwen38-mtp-r1"

# The prior ranked receipt for exactly this dispatch edit, and the base it was
# measured against. Recorded as configuration, never as a measurement here.
PRIOR_RECEIPT = {
    "prior_tree": "ca9251b8-58cd-4d90-9a52-fa05f5657216",
    "prior_score": 3.23250848263467,
    "prior_base_tree": "11863aa9",
    "prior_base_score": 3.24326223889754,
    "prior_delta_pct": -0.3315,
}


def gate_flags() -> dict[str, object]:
    return {
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "harness": "local",
    }


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def identity(meta: dict[str, str]) -> dict[str, object]:
    return {
        "host": HOST,
        "hostname": meta.get("host"),
        "chip": meta.get("chip"),
        "memory_gib": int(meta["memory_gib"]) if meta.get("memory_gib") else None,
        "toolchain": meta.get("toolchain"),
        "base_sha": BASE_SHA,
        "advisor_branch": UPSTREAM_NOTE,
    }


def start(name: str, job_type: str, question: str, rung: int,
          config: dict) -> wandb.sdk.wandb_run.Run:
    return wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP, job_type=job_type,
        name=name,
        config={"experiment": GROUP, "rung": rung, "question": question,
                **config, **gate_flags()},
        reinit=True,
    )


def attach(run, *paths: pathlib.Path) -> None:
    present = [p for p in paths if p.exists()]
    if not present:
        return
    artifact = wandb.Artifact(f"{run.name}-artifacts", type="analysis")
    for path in present:
        artifact.add_file(str(path))
    run.log_artifact(artifact)


def log_corpus() -> None:
    """Rung 0: what a ranked receipt of this size is actually worth."""
    run = start(
        "e100-corpus", "receipt-corpus",
        "does the ledger stop entry for wider NA survive the receipt noise floor",
        0,
        {
            "board_trees": 612,
            "board_source": "research/ranked_stream_ab_board.json",
            **PRIOR_RECEIPT,
        },
    )
    run.log({
        "corpus/one_leg_sd_pct": 1.077,
        "corpus/one_leg_pairs": 279,
        "corpus/stream_removal_effect_pct": -0.700,
        "corpus/stream_removal_se_pct": 0.285,
        "corpus/stream_removal_t": -2.46,
        "corpus/constant_fit_chi2": 0.062,
        "corpus/proportional_fit_chi2": 2.271,
        "corpus/proportional_rho": 0.204,
        "corpus/max_na_le4_effect_pct": -0.697,
        "corpus/max_na_le4_se_pct": 0.277,
        "corpus/max_na_le4_groups": 15,
        "corpus/max_na_le4_runs": 90,
        "corpus/max_na_ge5_groups": 0,
        "corpus/max_na_ge5_runs": 0,
        "corpus/plutarch_tax_median_pct": -0.001,
        "corpus/draft_length_null_max_abs": 0.0,
    })
    run.summary.update({
        "verdict": "the -0.3315 % receipt that closed this idea is inside a "
                   "1.077 % one-leg noise floor; NA >= 5 has never been "
                   "measured on the ranked corpus at all",
    })
    attach(run,
           OUT / "e100_ranked_stream_ab.txt",
           OUT / "e100_na5_board.txt",
           OUT / "e100_ab_census.txt")
    run.finish()


def log_probe() -> None:
    """Rung 2a: the isolated kernel cell, exactness first."""
    path = OUT / "e100_session.json"
    payload = json.loads(path.read_text())
    meta = read_meta(OUT / "e100-a1" / "meta.txt")

    run = start(
        "e100-probe", "kernel-probe",
        "what does collapsing M = 5 to one x-group cost or save in the kernel",
        2,
        {
            "kernel": "qmv_fast_crossrow_affine4_g64_wide",
            "dispatcher": "affine_qmv_fast<bfloat16_t, 64, 4, ...>",
            "widths": "1..9",
            "shapes": "mlp.gate_up, gdn.in_proj, fa.qkv, mlp.down, gdn.out_proj",
            "legs": list(payload["meta"].keys()),
            **identity(meta),
        },
    )

    cells = wandb.Table(columns=[
        "shape", "m", "groups", "inputs_per_group", "bytes", "base_us",
        "collapse_us", "delta_pct", "base_gb_per_s", "collapse_gb_per_s",
        "base_spread_pct",
    ])
    for cell in payload["cells"]:
        cells.add_data(
            cell["shape"], cell["m"], cell["g"], cell["gp"], cell["bytes"],
            cell["base_us"], cell["coll_us"], cell["delta"],
            cell["base_gb"], cell["coll_gb"], cell["base_spread"],
        )

    clean = [c for c in payload["cells"] if c["base_spread"] < 3.0]
    benefit5 = [c["delta"] for c in clean if c["m"] == 5]
    benefit9 = [c["delta"] for c in clean if c["m"] == 9]
    tax = [c["delta"] for c in clean if c["m"] in (6, 7, 8)]

    def mean_sd(values):
        if not values:
            return float("nan"), float("nan")
        if len(values) == 1:
            return values[0], float("nan")
        return statistics.mean(values), statistics.stdev(values) / len(values) ** 0.5

    b5, b5e = mean_sd(benefit5)
    b9, b9e = mean_sd(benefit9)
    tx, txe = mean_sd(tax)

    run.log({
        "probe/cells": cells,
        "probe/across_arm_digest_mismatches":
            len(payload["across_arm_digest_mismatches"]),
        "probe/within_arm_digest_mismatches_base":
            len(payload["within_arm_digest_mismatches"]["base"]),
        "probe/within_arm_digest_mismatches_collapse":
            len(payload["within_arm_digest_mismatches"]["collapse"]),
        "probe/positive_control_shapes_differing":
            sum(1 for p in payload["positive_control"] if p["differs"]),
        "probe/positive_control_shapes":
            len(payload["positive_control"]),
        "probe/m5_delta_pct": b5,
        "probe/m5_delta_se_pct": b5e,
        "probe/m5_cells": len(benefit5),
        "probe/m9_delta_pct": b9,
        "probe/m9_delta_se_pct": b9e,
        "probe/m9_cells": len(benefit9),
        "probe/shared_tax_delta_pct": tx,
        "probe/shared_tax_se_pct": txe,
        "probe/shared_tax_cells": len(tax),
        "probe/shared_tax_median_pct": payload["summary"]["tax_median_pct"],
        "probe/repeatability_median_pct":
            payload["summary"]["repeatability_median_pct"],
    })
    run.summary.update({
        "verdict": "bit-exact at every width, a real one-group saving at "
                   "M = 5, and no measurable shared register tax on the "
                   "widths whose dispatch lines never changed",
    })
    attach(run, path)
    run.finish()


def log_e2e() -> None:
    """Rung 2b: the same change on the real decode path."""
    session_path = OUT / "e100_e2e_session.json"
    model_path = OUT / "e100_round_model.json"
    session = json.loads(session_path.read_text())
    model = json.loads(model_path.read_text())
    meta = read_meta(OUT / "e100-e2e-d8-a1" / "meta.txt")

    run = start(
        "e100-e2e", "end-to-end-local-iterate",
        "does the one-group M = 5 cell move the real decode path",
        2,
        {
            "wrapper": "./benchmark-qwen-mtp.sh --local-iterate",
            "sessions": "d8 (64 tok, depth 8), d4 (64 tok, depth 4), "
                        "w512 (512 tok, depth 8)",
            "counterbalance": "A B B A per session",
            "seed_tokens": 512,
            "segmented_verify_depth_cap": 7,
            "prefill_seconds": model["prefill_seconds"],
            "serial_round_seconds": model["serial_round_seconds"],
            **identity(meta),
        },
    )

    legs = wandb.Table(columns=[
        "leg", "arm", "session", "offered_depth", "decode_tokens", "rounds",
        "mean_verify_width", "mtp_s_per_token", "serial_s_per_token",
        "local_ratio", "accepted_draft_rate", "round_ms",
        "gpu_temp_entry_c", "gpu_temp_exit_c",
    ])
    for tag, leg in sorted(session["legs"].items()):
        row = next((r for r in model["legs"] if r["tag"] == tag), None)
        legs.add_data(
            tag, leg["meta"]["arm"], row["session"] if row else "",
            int(leg["meta"]["offered_depth"]),
            int(leg["meta"].get("decode_tokens", "64")),
            row["rounds"] if row else 0,
            row["m_mean"] if row else 0.0,
            leg["score"]["mtp_seconds_per_token"],
            leg["score"]["serial_seconds_per_token"],
            leg["score"]["mtp_decode_speedup"],
            leg["score"]["accepted_draft_rate"],
            row["round_ms"] if row else 0.0,
            float(leg["meta"]["gpu_temp_entry_c"]),
            float(leg["meta"]["gpu_temp_exit_c"]),
        )

    payload = {"e2e/legs": legs}
    for name, block in session["summary"].items():
        spt = block["mtp_seconds_per_token"]
        payload[f"e2e/{name}/candidate_s_per_token_base"] = spt["base"]
        payload[f"e2e/{name}/candidate_s_per_token_collapse"] = spt["collapse"]
        payload[f"e2e/{name}/candidate_s_per_token_delta_pct"] = spt["delta_pct"]
        payload[f"e2e/{name}/within_arm_spread_base_pct"] = spt["base_spread_pct"]
        payload[f"e2e/{name}/within_arm_spread_collapse_pct"] = \
            spt["collapse_spread_pct"]
        ratio = block["mtp_decode_speedup"]
        payload[f"e2e/{name}/local_ratio_base"] = ratio["base"]
        payload[f"e2e/{name}/local_ratio_collapse"] = ratio["collapse"]
        payload[f"e2e/{name}/local_ratio_delta_pct"] = ratio["delta_pct"]
    for name, block in model["sessions"].items():
        payload[f"round/{name}/base_ms"] = block["base_round_ms"]
        payload[f"round/{name}/collapse_ms"] = block["collapse_round_ms"]
        payload[f"round/{name}/delta_pct"] = block["delta_pct"]
        payload[f"round/{name}/mean_verify_width"] = block["mean_verify_width"]
    payload["round/prefill_seconds"] = model["prefill_seconds"]
    payload["round/serial_round_ms"] = 1000.0 * model["serial_round_seconds"]
    run.log(payload)
    run.finish()


def log_registers() -> None:
    """The register cost of the cap, on the object the worker actually runs."""
    path = OUT / "e100-reg-census.json"
    payload = json.loads(path.read_text())

    run = start(
        "e100-registers", "register-census",
        "what does raising the NA cap cost on the ONE kernel that holds every "
        "width",
        2,
        {
            "entry_points": "affine_qmv_fast<bfloat16_t, 64, 4, false|true>",
            "translation_unit": "runtime-effective JIT string "
                                "(mlx-generated/*.cpp preambles)",
            "local_arch": "applegpu_g16s",
            "ranked_arch": "applegpu_g17s",
            "e76_body_census_g17s_na2_to_na6": [83, 90, 91, 98, 111],
            "base_sha": payload["base_sha"],
        },
    )

    table = wandb.Table(columns=[
        "arm", "arch", "kernel", "registers", "spill_bytes", "partitions",
    ])
    peak = {}
    for arm in payload["arms"]:
        if not arm["compiled"]:
            table.add_data(arm["arm"], "", "DID NOT COMPILE", 0, 0, "")
            continue
        partitions = " ".join(f"M{m}=[{p}]"
                              for m, p in arm["partitions"].items())
        for arch in ("applegpu_g16s", "applegpu_g17s"):
            for kernel, value in arm.get(arch, {}).items():
                table.add_data(arm["arm"], arch, kernel, value["registers"],
                               value["spill_bytes"] or 0, partitions)
                key = f"{arm['arm']}_{arch}"
                peak[key] = max(peak.get(key, 0), value["registers"])

    logged = {"registers/census": table}
    for key, value in peak.items():
        logged[f"registers/peak_live_regs/{key}"] = value
    run.log(logged)
    run.summary.update({"peak_live_regs": peak})
    attach(run, path)
    run.finish()


RUNS = {
    "corpus": log_corpus,
    "probe": log_probe,
    "e2e": log_e2e,
    "registers": log_registers,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(RUNS))
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    print(f"logging E100 at {head}")
    for name, fn in RUNS.items():
        if args.only and name != args.only:
            continue
        print(f"-- {name}")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
