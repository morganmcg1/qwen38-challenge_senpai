#!/usr/bin/env python3
"""E165 Stage 0: publish the closed round budget and the GPU-idle census.

Everything here is `harness=local`. Not one number in this run is an official
or ranked score, and none may be compared with a receipt. Every leg is
DIAGNOSTIC and ungated, so `gate_qualified_for_timing=false` is carried through
verbatim rather than dropped.

WHAT THE RUN IS FOR. The campaign needs to know whether a decode round is idle
or slow, because the two need opposite fixes. The round's blocking eval is a
full barrier, so the window from that barrier to the next `asyncEval` is
PROVABLY idle and every other part of the round has a command buffer enqueued.
That gives a measured lower bound on idle and a measured upper bound on busy,
and the first of those bounds the whole host-reordering family of mechanisms.

COUNTER HAZARD, published here so no downstream reader repeats it.
`verify_build_us` is not host graph construction: under the shipped ladder it
is about 97 % GPU wait (E86). Only `verify_pipeline` is quotable, and only the
`MLX_QWEN_MTP_LADDER=off` leg can separate host encode from verify GPU wall.

Usage:
  python3 research/e165_wandb_log.py --run-name e165-r0-stage0-census
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import wandb

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"

# Advisor F1, RULE 370. Price a per-round mechanism at the observed per-round
# difference at the rows the scored path actually visits, never at the affine
# fit's intercept. These two are recorded so the run carries the conversion it
# was priced with rather than leaving it in prose.
RANKED_DELTA_R_AT_BEAGLE_MS = 0.079
RANKED_DELTA_R_AT_ESSAYS_MS = 0.091
LOCAL_TO_RANKED_INTERCEPT = 3.58


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO).decode().strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--census", default=str(REPO / "research" / "e165-census.json"))
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()

    census = json.loads(pathlib.Path(args.census).read_text())
    legs = census["legs"]
    if not legs:
        raise SystemExit("no legs in the census; nothing to publish")

    # THE PIN GATE. `MLX_E159_FIXED_DRAFT_DEPTH` replaces `draftPolicy`
    # outright, so a leg that reports more than one width did not pin and its
    # components cannot be read as a single-width measurement.
    for leg in legs:
        if leg["rounds"] and len(leg["depths"]) != 1:
            raise SystemExit(
                f"leg {leg['tag']} reports widths {leg['depths']}; the pin "
                "did not hold and the census cannot be published")
        if leg["pinned_depth"] and leg["depths"]:
            if int(leg["pinned_depth"]) != leg["depths"][0]:
                raise SystemExit(
                    f"leg {leg['tag']} pinned {leg['pinned_depth']} but ran "
                    f"width {leg['depths'][0]}")

    config = {
        "experiment": "e165",
        "assignment_id": "e165-per-round-fixed-cost",
        "revision_id": "r0",
        "stage": "0-census",
        "student": "qwen-thorfinn",
        "harness": "local",
        "official_score_produced": False,
        "leg_role": "diagnostic",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "candidate_sha": git("rev-parse", "HEAD"),
        "base_sha": args.base_sha,
        "worker_sha256": legs[0].get("worker_sha256"),
        "ranked_delta_R_at_beagle_ms": RANKED_DELTA_R_AT_BEAGLE_MS,
        "ranked_delta_R_at_essays_ms": RANKED_DELTA_R_AT_ESSAYS_MS,
        "local_to_ranked_intercept_factor": LOCAL_TO_RANKED_INTERCEPT,
        "counter_hazard": (
            "verify_build_us is ~97 % GPU wait under the shipped ladder; only "
            "verify_pipeline is quotable (E86, advisor F1)"),
    }

    run = wandb.init(
        entity=ENTITY, project=PROJECT, name=args.run_name,
        job_type="census", config=config,
        tags=["e165", "stage0", "harness-local", "diagnostic", "ungated"],
    )

    leg_columns = [
        "tag", "pinned_depth", "ladder", "trace", "sync_head", "rounds",
        "full_accept_rounds", "mtp_seconds_per_token",
        "effective_mean_draft_len", "gpu_temp_entry_c", "gpu_temp_exit_c",
    ]
    metric_keys = sorted(
        {k for leg in legs for k in (leg.get("median") or {})})
    leg_table = wandb.Table(columns=leg_columns + metric_keys)
    for leg in legs:
        median = leg.get("median") or {}
        leg_table.add_data(
            *[leg.get(c) for c in leg_columns],
            *[median.get(k) for k in metric_keys],
        )
    run.log({"census/legs": leg_table})

    budget_table = wandb.Table(
        columns=["tag", "pinned_depth", "ladder", "term", "us", "pct_of_round",
                 "device_state"])
    covered = {
        "submit1", "host_chain_build", "submit2", "snapshot",
        "verify_window", "gpu_verify_eval",
    }
    for leg in legs:
        budget = leg.get("budget")
        if not budget:
            continue
        period = budget["round_period"]
        for term, value in budget["terms"].items():
            if value is None:
                continue
            budget_table.add_data(
                leg["tag"], leg["pinned_depth"], leg["ladder"], term, value,
                100.0 * value / period,
                "covered" if term in covered else "provably_idle")
        budget_table.add_data(
            leg["tag"], leg["pinned_depth"], leg["ladder"], "residual",
            budget["residual_us"], 100.0 * budget["residual_us"] / period,
            "unattributed")
        run.summary[f"budget/{leg['tag']}/round_period_us"] = period
        run.summary[f"budget/{leg['tag']}/idle_fraction_lower"] = (
            budget["idle_fraction"])
        run.summary[f"budget/{leg['tag']}/busy_fraction_upper"] = (
            budget["busy_fraction"])
        run.summary[f"budget/{leg['tag']}/residual_pct"] = (
            100.0 * budget["residual_us"] / period)
        run.summary[f"budget/{leg['tag']}/gpu_idle_window_us"] = (
            budget["measured_idle_window_us"])
    run.log({"census/budget": budget_table})

    fit_table = wandb.Table(
        columns=["component", "intercept_us", "per_draft_us", "points"])
    for component, body in census.get("fit", {}).items():
        fit_table.add_data(
            component, body["intercept_us"], body["per_draft_us"],
            json.dumps(body["points"]))
        run.summary[f"fit/{component}/intercept_us"] = body["intercept_us"]
        run.summary[f"fit/{component}/per_draft_us"] = body["per_draft_us"]
    run.log({"census/fit": fit_table})

    contrast_table = wandb.Table(columns=["contrast", "field", "value"])
    for name, body in census.get("contrasts", {}).items():
        for field, value in body.items():
            contrast_table.add_data(name, field, str(value))
            run.summary[f"contrast/{name}/{field}"] = value
    run.log({"census/contrasts": contrast_table})

    print(f"run url: {run.url}")
    print(f"run id: {run.id}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
