#!/usr/bin/env python3
"""Research-only: publish one draft-readout bit/M sweep to W&B.

usage:
  research/log_bits_sweep_wandb.py .mlxfast-private/draft-bits-sweep/<tag> \
      [--group G] [--notes ...]

The sweep JSON produced by QwenQMVCostCurveTests.sweepCompactDraftReadoutOverBits
carries one row per (bits, M) arm plus the host roofline. Both are logged: the
rows as a table and per-arm summary scalars, the roofline as config, so any
achieved-GB/s claim in a report is reachable from a run URL.
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
COLUMNS = ["bits", "m", "weight_bytes", "seconds_per_call", "ms_per_call",
           "seconds_per_call_min", "seconds_per_call_max", "spread_pct",
           "achieved_gb_per_second", "threadgroups_x", "kernel",
           "in_kernel_path", "uses_crossrow_kernel"]


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


def parse_identity(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.split(": ", 1)[-1]
        for field in line.split():
            if "=" in field:
                k, v = field.split("=", 1)
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--group", default="qwen38-r1-e15-draft-readout-3bit")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    doc = json.loads((args.sweep_dir / "bits.json").read_text())
    identity = parse_identity(args.sweep_dir / "identity.txt")
    arms = doc["arms"]
    roofline = doc["roofline"]
    peak = roofline["peak_bandwidth_bytes_per_second"]

    config = {
        "tag": args.sweep_dir.name,
        "arm_order": doc["arm_order"],
        "reps": doc["reps"],
        "inner_calls_per_rep": doc["inner_calls_per_rep"],
        "source": doc["source"],
        "host_model": sh("sysctl", "-n", "hw.model"),
        "host_chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "host_mem_bytes": sh("sysctl", "-n", "hw.memsize"),
        "platform": platform.platform(),
        **{f"device_{k}": v for k, v in doc["device"].items()
           if k != "predicted_vector_limits"},
        **{f"roofline_{k}": v for k, v in roofline.items()},
        **{f"identity_{k}": v for k, v in identity.items()},
    }

    run = wandb.init(project=PROJECT, entity=ENTITY, group=args.group,
                     job_type="qmv-bits-sweep", config=config,
                     notes=args.notes or None)

    table = wandb.Table(columns=COLUMNS)
    for a in arms:
        spread = ((a["seconds_per_call_max"] - a["seconds_per_call_min"])
                  / a["seconds_per_call"] * 100.0)
        table.add_data(
            a["bits"], a["m"], a["weight_bytes"], a["seconds_per_call"],
            a["seconds_per_call"] * 1e3, a["seconds_per_call_min"],
            a["seconds_per_call_max"], spread, a["achieved_gb_per_second"],
            a["threadgroups_x"], a["kernel"], a["in_kernel_path"],
            a["uses_crossrow_kernel"])
        p = f"arm_b{a['bits']}_m{a['m']}"
        run.summary[f"{p}/ms_per_call"] = a["seconds_per_call"] * 1e3
        run.summary[f"{p}/achieved_gb_per_second"] = a["achieved_gb_per_second"]
        run.summary[f"{p}/weight_bytes"] = a["weight_bytes"]
        run.summary[f"{p}/spread_pct"] = spread
        run.summary[f"{p}/in_kernel_path"] = a["in_kernel_path"]
        run.summary[f"{p}/uses_crossrow_kernel"] = a["uses_crossrow_kernel"]
        run.summary[f"{p}/bandwidth_vs_stream_peak"] = (
            a["achieved_gb_per_second"] * 1e9 / peak)
    run.log({"arms": table})

    by_key = {(a["bits"], a["m"]): a for a in arms}
    for m in sorted({a["m"] for a in arms}):
        base = by_key.get((4, m))
        if base is None:
            continue
        for bits in sorted({a["bits"] for a in arms if a["m"] == m}):
            arm = by_key.get((bits, m))
            if arm is None or bits == 4:
                continue
            p = f"delta_b{bits}_vs_b4_m{m}"
            run.summary[f"{p}/byte_ratio"] = (
                arm["weight_bytes"] / base["weight_bytes"] - 1.0)
            run.summary[f"{p}/time_ratio"] = (
                arm["seconds_per_call"] / base["seconds_per_call"] - 1.0)
            run.summary[f"{p}/bandwidth_ratio"] = (
                arm["achieved_gb_per_second"]
                / base["achieved_gb_per_second"] - 1.0)
    run.finish()
    print(run.url)


if __name__ == "__main__":
    main()
