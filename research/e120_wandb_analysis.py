#!/usr/bin/env python3
"""Publish the E120 DERIVED analysis to W&B.

    usage: research/e120_wandb_analysis.py --name NAME [--rung 5d-analysis]

`research/e120_wandb_log.py` publishes one probe SESSION: the raw per-cell
timings a single `research/e120_probe.sh` invocation measured. Three such runs
between them hold the whole 7x7 shape-by-width grid (qql6zari, iyzornb9,
by2wpwg5), but the numbers the campaign actually argues from -- the net
microseconds saved per matvec, the per-round price of the shipped gate, and the
ranked percentage those imply -- were only ever files under `research/out/`.

This publishes that second layer, so the chain

    measured cells -> 7x7 grid -> round price -> ranked %

is reproducible from W&B rather than from a working directory that nobody else
can see. It measures nothing: it reads `e120_additivity.py` and
`e120_gate_price.py` output and records it with the provenance of the runs that
produced the cells.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e120-own-the-qmv-dispatch"
HOST = "apple-m4-pro-applegpu_g16s-48gib"
BASE_SHA = "2127858ba770ddc06027205d8df89a8db21d80f5"
BUDGET_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"

# The measured sessions this analysis consumes.
SOURCE_RUNS = {
    "qql6zari": "rung 5d, widths 3,4,8, all seven shapes",
    "iyzornb9": "rung 5d-na, widths 5,6,7,9, four shapes",
    "by2wpwg5": "rung 5d-na2, widths 5,6,7,9, remaining shapes",
}

# E116 primary, 12-leg OLS at R-squared 0.9972.
WIDE_QMV_TO_LEG = 0.6070
WIDE_QMV_TO_LEG_CI = (0.5843, 0.6297)
LEG_TO_RANKED = 0.95

# `tablePays(m:) = m >= 4`. The volume term never binds at any width the gate
# can see, so `m4_and_volume_100k` and a bare `m >= 4` price identically.
SHIPPED_GATE = "m4_and_volume_100k"


def read_json(path: pathlib.Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--rung", default="5d-analysis")
    parser.add_argument(
        "--additivity",
        type=pathlib.Path,
        default=pathlib.Path("research/out/e120-additivity.json"),
    )
    parser.add_argument(
        "--gate-price",
        type=pathlib.Path,
        default=pathlib.Path("research/out/e120-gate-price.json"),
    )
    args = parser.parse_args()

    additivity = read_json(args.additivity)
    gate_price = read_json(args.gate_price)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        job_type="derived-analysis",
        name=args.name,
        config={
            "experiment": GROUP,
            "rung": args.rung,
            "pr": 121,
            "question": (
                "What does the measured 7x7 net-microsecond grid price the "
                "shipped m>=4 table gate at, per round and in ranked percent?"
            ),
            "base_sha": BASE_SHA,
            "budget_base_sha": BUDGET_BASE_SHA,
            "host": HOST,
            "source_runs": SOURCE_RUNS,
            "shipped_gate": SHIPPED_GATE,
            "wide_qmv_to_leg": WIDE_QMV_TO_LEG,
            "wide_qmv_to_leg_ci": list(WIDE_QMV_TO_LEG_CI),
            "leg_to_ranked": LEG_TO_RANKED,
            "analysis_tools": [
                "research/e120_additivity.py",
                "research/e120_gate_price.py",
            ],
            # This run holds no model and starts no GPU work; it re-reads
            # measurements the three source runs already gated and recorded.
            "timing_valid": False,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
            "harness": "local",
            "gpu_seconds_used": 0.0,
            "instrument": "derived analysis, no measurement",
        },
        reinit=True,
    )

    summary: dict[str, object] = {}

    # --- the measured grid, one row per (shape, width) -----------------------
    grid_table = wandb.Table(
        columns=[
            "shape", "width", "k", "n", "base_us", "gain_us", "fill_us",
            "net_us", "gain_pct", "gb_per_s", "source_run",
        ]
    )
    for cell in sorted(additivity["cells"], key=lambda c: (c["shape"], c["m"])):
        gain_pct = (
            100.0 * cell["gain_us"] / cell["base_us"] if cell["base_us"] else None
        )
        grid_table.add_data(
            cell["shape"], cell["m"], cell["k"], cell["n"],
            cell["base_us"], cell["gain_us"], cell["fill_us"],
            cell["net_us"], gain_pct, cell["gb_per_s"],
            ",".join(cell.get("runs", [])),
        )
        summary[f"net_us/{cell['shape']}/m{cell['m']}"] = cell["net_us"]
        summary[f"gain_pct/{cell['shape']}/m{cell['m']}"] = gain_pct
    run.log({"net_us_grid": grid_table})

    # --- the single-group basis the multi-group widths are predicted from ----
    basis_table = wandb.Table(columns=["shape", "na", "gain_us"])
    for shape, by_na in sorted(additivity["basis"].items()):
        for na, gain_us in sorted(by_na.items(), key=lambda kv: int(kv[0])):
            basis_table.add_data(shape, int(na), gain_us)
    run.log({"single_group_basis": basis_table})

    # --- absolute additivity, one row per multi-group prediction -------------
    check_table = wandb.Table(
        columns=[
            "shape", "width", "groups", "pred_gain_us", "meas_gain_us",
            "ratio", "err_us", "err_pct_of_base",
        ]
    )
    for check in sorted(
        additivity["additivity_checks"], key=lambda c: (c["shape"], c["m"])
    ):
        check_table.add_data(
            check["shape"], check["m"], str(check["groups"]),
            check["pred_gain_us"], check["meas_gain_us"], check["ratio"],
            check["err_us"], check["err_pct_of_base"],
        )
    run.log({"additivity_checks": check_table})

    ratios = [c["ratio"] for c in additivity["additivity_checks"]]
    if ratios:
        ordered = sorted(ratios)
        mid = len(ordered) // 2
        summary["additivity/ratio_n"] = len(ratios)
        summary["additivity/ratio_mean"] = sum(ratios) / len(ratios)
        summary["additivity/ratio_median"] = (
            ordered[mid]
            if len(ordered) % 2
            else 0.5 * (ordered[mid - 1] + ordered[mid])
        )
        summary["additivity/ratio_min"] = min(ratios)
        summary["additivity/ratio_max"] = max(ratios)
        # FALSIFIED: an isolated-microsecond sum is not a valid predictor.
        summary["additivity/absolute_additivity_holds"] = False

    # --- per-round price of the shipped gate ---------------------------------
    price_table = wandb.Table(
        columns=["width", "base_us", "net_us", "round_pct", "leg_pct", "ranked_pct"]
    )
    for width in sorted(gate_price, key=int):
        cell = gate_price[width]
        gate = cell["gates"][SHIPPED_GATE]
        leg_pct = gate["pct"] * WIDE_QMV_TO_LEG
        ranked_pct = leg_pct * LEG_TO_RANKED
        price_table.add_data(
            int(width),
            cell["base_us"],
            gate["net_us"],
            gate["pct"],
            leg_pct,
            ranked_pct,
        )
        summary[f"round_base_us/m{width}"] = cell["base_us"]
        summary[f"round_net_us/m{width}"] = gate["net_us"]
        summary[f"round_pct/m{width}"] = gate["pct"]
        summary[f"leg_pct/m{width}"] = leg_pct
        summary[f"ranked_pct/m{width}"] = ranked_pct
    run.log({"gate_price": price_table})

    # --- fractional additivity, the property that DOES hold ------------------
    # Gain percentage is flat across all seven shapes at every width, which is
    # what licenses carrying one per-width percentage across shapes. Absolute
    # microsecond additivity does not hold, which is why the round price is
    # built from percentages and not from a sum of isolated microseconds.
    fractional = wandb.Table(
        columns=["width", "shapes", "mean_gain_pct", "sd_gain_pct"]
    )
    by_width: dict[int, list[float]] = {}
    for cell in additivity["cells"]:
        if cell["shape"].endswith(".small") or not cell["base_us"]:
            continue
        by_width.setdefault(cell["m"], []).append(
            100.0 * cell["gain_us"] / cell["base_us"]
        )
    for width in sorted(by_width):
        values = by_width[width]
        mean = sum(values) / len(values)
        sd = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            if len(values) > 1
            else 0.0
        )
        fractional.add_data(width, len(values), mean, sd)
        summary[f"fractional/mean_gain_pct/m{width}"] = mean
        summary[f"fractional/sd_gain_pct/m{width}"] = sd
    run.log({"fractional_additivity": fractional})

    run.summary.update(summary)

    artifact = wandb.Artifact(
        name="e120-derived-analysis",
        type="analysis",
        description=(
            "E120 net-microsecond grid and shipped-gate round price, derived "
            "from probe runs " + ", ".join(SOURCE_RUNS)
        ),
    )
    artifact.add_file(str(args.additivity))
    artifact.add_file(str(args.gate_price))
    run.log_artifact(artifact)

    print(f"run {run.id}  {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
