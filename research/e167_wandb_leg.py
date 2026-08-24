#!/usr/bin/env python3
"""E167: publish ONE finished ABBA leg to W&B, immediately.

Why this exists. On 2026-08-24T07:30Z this host was reprovisioned in the
middle of a 16-leg session and every leg was lost, because leg results only
existed under the role directory. A session that logs each leg as it finishes
costs only its remaining legs when that happens again.

The run is resumed by a fixed id, so twelve invocations append to one run.
Each leg is logged twice: as a step-indexed scalar series for the charts, and
as a row in a summary table that survives out of order.

harness=local. These are M4 Pro numbers from the candidate build on both legs
of the local wrapper. They are never an official or ranked score.

Usage:
  python3 research/e167_wandb_leg.py \
      --run-id e167-arms-<session> --leg-dir research/out/e167/arms/leg1-B0 \
      --session research/out/e167/arms/session.txt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"


def parse_kv(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def number(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--leg-dir", required=True, type=pathlib.Path)
    ap.add_argument("--session", required=True, type=pathlib.Path)
    args = ap.parse_args()

    meta = parse_kv(args.leg_dir / "meta.txt")
    session = parse_kv(args.session)
    report_path = args.leg_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    leg = int(meta.get("leg", "0"))
    decode_s = number(str(report.get("decode_seconds")))
    prefill_s = number(str(report.get("seed_prefill_seconds", 0))) or 0.0
    token_count = number(str(report.get("decode_token_count")))
    decode_only_spt = None
    if decode_s is not None and token_count:
        decode_only_spt = (decode_s - prefill_s) / token_count

    row = {
        "leg": leg,
        "arm": meta.get("arm", "?"),
        "utc": meta.get("utc", ""),
        "worker_sha12": meta.get("worker_sha256", "")[:12],
        "witness": meta.get("witness", ""),
        "cool_gate_passed_real_gate": meta.get(
            "cool_gate_passed_real_gate", "false"
        ),
        "gate_qualified_for_timing": meta.get(
            "gate_qualified_for_timing", "false"
        ),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c", ""),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c", ""),
        "all_tokens_matched": bool(report.get("all_tokens_matched", False)),
        "spt": number(str(report.get("parent_measured_seconds_per_token"))),
        "decode_seconds": decode_s,
        "seed_prefill_seconds": prefill_s,
        "decode_only_spt": decode_only_spt,
        "effective_mean_draft_len": number(
            str(report.get("effective_mean_draft_len"))
        ),
        "accepted_draft_total": number(str(report.get("accepted_draft_total"))),
        "round_count": number(str(report.get("round_count"))),
        "timed_exit": meta.get("timed_exit", ""),
    }

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=args.run_id,
        resume="allow",
        name=args.run_id,
        job_type="local-abba-leg",
        tags=["e167", "harness=local", "counter-strip", "abba"],
        config={
            "harness": "local",
            "git_head": session.get("git_head", ""),
            "git_dirty": session.get("git_dirty", ""),
            "tokens": session.get("tokens", ""),
            "depth": session.get("depth", ""),
            "schedule": session.get("schedule", ""),
            "host": session.get("host", ""),
            "golden_sha256": session.get("golden_sha256", ""),
            "assignment": "e167-ship-the-maintained-base",
            "pr": 166,
        },
    )

    arm = row["arm"]
    scalars = {f"leg/{k}": v for k, v in row.items() if isinstance(v, (int, float))}
    scalars.update(
        {
            f"arm/{arm}/decode_only_spt": decode_only_spt,
            f"arm/{arm}/spt": row["spt"],
            f"arm/{arm}/seed_prefill_seconds": prefill_s,
        }
    )
    run.log(scalars, step=leg)

    # One row per leg, appended through the run summary so an interrupted
    # session still shows every leg that finished.
    existing = run.summary.get("legs_json")
    legs = json.loads(existing) if isinstance(existing, str) else []
    legs = [r for r in legs if r.get("leg") != leg] + [row]
    legs.sort(key=lambda r: r.get("leg", 0))
    run.summary["legs_json"] = json.dumps(legs)
    run.summary["legs_logged"] = len(legs)

    run.log(
        {
            "legs_table": wandb.Table(
                columns=list(row.keys()),
                data=[[r.get(c) for c in row.keys()] for r in legs],
            )
        },
        step=leg,
    )

    print(
        f"e167_wandb_leg: logged leg {leg} arm {arm} "
        f"decode_only_spt={decode_only_spt} url={run.url}"
    )
    run.finish()


if __name__ == "__main__":
    main()
