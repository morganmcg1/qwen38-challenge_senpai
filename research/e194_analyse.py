#!/usr/bin/env python3
"""Research-only (qwen38-r1-e194-sdpa-step-removal).

Read one E194 ABBA session directory and price Route 2' in ms per decode round.

    python3 research/e194_analyse.py research/out/e194/session-NAME [--wandb]

Arm 0 is the shipped `today` form, arm 1 is Route 2' (one sequence-major query
buffer). Both arms are the SAME worker binary; only MLX_E194_SEQ_MAJOR_Q moves.

THREE ESTIMATORS, REPORTED TOGETHER.

  pairwise   The primary. The session order is a sequence of adjacent
             (arm0, arm1) pairs in alternating orientation, so a per-pair
             signed difference cancels linear thermal or clock drift to first
             order. n = number of pairs.
  leg_means  Difference of the per-arm means of leg means. Ignores drift, so it
             is the sanity check on the pairwise number, not the headline.
  pooled     Difference of medians over every round of every leg. Immune to
             leg-count smallness, exposed to drift.

STOP RULE (predeclared in the assignment): recovery is the DROP in ms/round of
arm 1 against arm 0. STOP unless recovery - 2 sigma > 0.5 ms/round.

NULL CONTROL. The arm touches only the 6 <= qL <= 9 wide-decode branch, so the
depth-0 serial leg must not move. A serial contrast of the same size as the MTP
contrast falsifies the instrument rather than supporting the mechanism.

Everything here is harness=local, ungated (cool_gate_passed_real_gate=false,
gate_qualified_for_timing=false), and is never a score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def leg_reports(leg_dir: Path) -> tuple[dict | None, dict | None]:
    """Return (serial_report, mtp_report) from the captured trusted CLI reports."""
    serial = mtp = None
    for report_path in sorted((leg_dir / "reports").glob("*.json")):
        try:
            payload = json.loads(report_path.read_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if "parent_measured_seconds_per_token" not in payload:
            continue
        if payload.get("is_serial_control") is True:
            serial = payload
        elif payload.get("is_serial_control") is False:
            mtp = payload
    return serial, mtp


def phase(report: dict) -> dict:
    tokens = report["decode_token_count"]
    decode_seconds = report["decode_seconds"]
    prefill = report.get("seed_prefill_seconds")
    rounds_ms = [1000.0 * v for v in (report.get("block_request_seconds") or [])]
    row = {
        "tokens": tokens,
        "decode_seconds": decode_seconds,
        "prefill_seconds": prefill,
        "charged_seconds_per_token": report["parent_measured_seconds_per_token"],
        "decode_only_seconds_per_token": (
            None if prefill is None else (decode_seconds - prefill) / tokens
        ),
        "round_count": report.get("round_count"),
        "accepted_draft_total": report.get("accepted_draft_total"),
        "rejected_draft_total": report.get("rejected_draft_total"),
        "accepted_draft_rate": report.get("accepted_draft_rate"),
        "declared_rows_total": report.get("declared_rows_total"),
        "reference_checked_row_total": report.get("reference_checked_row_total"),
        "emitted_token_total": report.get("emitted_token_total"),
        "all_tokens_matched": report.get("all_tokens_matched"),
        "residual_divergence_count": report.get("residual_divergence_count"),
        "rounds_ms": rounds_ms,
    }
    # The first block carries the seed prefill and every first-touch allocation,
    # so it is reported but never used in a contrast.
    warm = rounds_ms[1:]
    if warm:
        row["round_ms_median"] = statistics.median(warm)
        row["round_ms_mean"] = statistics.fmean(warm)
        row["round_ms_sd"] = statistics.stdev(warm) if len(warm) > 1 else 0.0
        row["round_ms_first"] = rounds_ms[0]
        row["warm_round_count"] = len(warm)
    return row


def summarize(values: list) -> dict:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0}
    out = {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
    }
    out["sd"] = statistics.stdev(clean) if len(clean) > 1 else 0.0
    out["sem"] = out["sd"] / math.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return out


def stop_rule(recovery_ms: float, sigma_ms: float, threshold: float = 0.5) -> dict:
    lower = recovery_ms - 2.0 * sigma_ms
    return {
        "recovery_ms_per_round": recovery_ms,
        "sigma_ms_per_round": sigma_ms,
        "recovery_minus_2sigma": lower,
        "threshold_ms_per_round": threshold,
        "advance": lower > threshold,
        "verdict": (
            "advance to Stage 3"
            if lower > threshold
            else "STOP: not useful standalone, composition material"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--base-sha", default="80abd5aceaa931dd7827cea2fcd73e3bfc6171f4")
    parser.add_argument("--candidate-sha", default="")
    args = parser.parse_args()

    legs = []
    for leg_dir in sorted(p for p in args.session.iterdir() if p.is_dir()):
        meta_path = leg_dir / "meta.txt"
        if not meta_path.exists():
            continue
        meta = read_meta(meta_path)
        serial, mtp = leg_reports(leg_dir)
        leg = {
            "leg": int(meta.get("e194_leg", 0)),
            "arm": meta.get("e194_arm"),
            "dir": str(leg_dir),
            "exit": meta.get("exit"),
            "gpu_temp_entry": (
                float(meta["gpu_temp_entry"])
                if meta.get("gpu_temp_entry", "unavailable") != "unavailable"
                else None
            ),
            "gpu_temp_exit": (
                float(meta["gpu_temp_exit"])
                if meta.get("gpu_temp_exit", "unavailable") != "unavailable"
                else None
            ),
            "worker_sha256": meta.get("worker_sha256_before"),
            "worker_digest_stable": meta.get("worker_digest_stable"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        }
        score_path = leg_dir / "score.json"
        if score_path.exists():
            score = json.loads(score_path.read_text())
            metrics = score.get("metrics", {})
            leg["score_local_ratio"] = metrics.get("mtp_decode_speedup")
            leg["score_serial_spt"] = metrics.get("serial_seconds_per_token")
            leg["score_mtp_spt"] = metrics.get("mtp_seconds_per_token")
            leg["score_all_tokens_matched"] = metrics.get("all_tokens_matched")
            leg["score_effective_mean_draft_len"] = metrics.get("effective_mean_draft_len")
            leg["score_passed"] = score.get("passed")
        if serial:
            leg["serial"] = phase(serial)
        if mtp:
            leg["mtp"] = phase(mtp)
        legs.append(leg)
    legs.sort(key=lambda row: row["leg"])

    arms = sorted({leg["arm"] for leg in legs if leg.get("arm")})
    per_arm = {}
    for arm in arms:
        rows = [leg for leg in legs if leg["arm"] == arm]
        per_arm[arm] = {
            "legs": [row["leg"] for row in rows],
            "round_ms_mean": summarize([r.get("mtp", {}).get("round_ms_mean") for r in rows]),
            "round_ms_median": summarize(
                [r.get("mtp", {}).get("round_ms_median") for r in rows]
            ),
            "mtp_decode_only_spt": summarize(
                [r.get("mtp", {}).get("decode_only_seconds_per_token") for r in rows]
            ),
            "serial_decode_only_spt": summarize(
                [r.get("serial", {}).get("decode_only_seconds_per_token") for r in rows]
            ),
            "local_ratio_charged": summarize([r.get("score_local_ratio") for r in rows]),
            "gpu_temp_entry": summarize([r.get("gpu_temp_entry") for r in rows]),
            "gpu_temp_exit": summarize([r.get("gpu_temp_exit") for r in rows]),
            "round_count": summarize([r.get("mtp", {}).get("round_count") for r in rows]),
            "accepted_draft_total": summarize(
                [r.get("mtp", {}).get("accepted_draft_total") for r in rows]
            ),
            "rejected_draft_total": summarize(
                [r.get("mtp", {}).get("rejected_draft_total") for r in rows]
            ),
        }

    # RULE 179: the two arms must do the same speculative work. A different
    # round, proposal or acceptance count means the arms are not comparable,
    # whatever the clock says.
    def counter_set(field: str) -> set:
        return {
            leg.get("mtp", {}).get(field)
            for leg in legs
            if leg.get("mtp") is not None
        }

    work_identity = {
        "round_count": sorted(x for x in counter_set("round_count") if x is not None),
        "accepted_draft_total": sorted(
            x for x in counter_set("accepted_draft_total") if x is not None
        ),
        "rejected_draft_total": sorted(
            x for x in counter_set("rejected_draft_total") if x is not None
        ),
        "declared_rows_total": sorted(
            x for x in counter_set("declared_rows_total") if x is not None
        ),
        "emitted_token_total": sorted(
            x for x in counter_set("emitted_token_total") if x is not None
        ),
    }
    work_identity["identical_across_all_legs"] = all(
        len(set(values)) <= 1 for key, values in work_identity.items() if key != "identical_across_all_legs"
    )
    work_identity["all_tokens_matched_every_leg"] = all(
        leg.get("mtp", {}).get("all_tokens_matched") is True for leg in legs if leg.get("mtp")
    )
    work_identity["residual_divergence_total"] = sum(
        (leg.get("mtp", {}).get("residual_divergence_count") or 0) for leg in legs
    )

    # Pairwise estimator over adjacent legs. Each consecutive (2k, 2k+1) pair
    # must contain one leg of each arm; the signed difference is always
    # arm1 - arm0, so alternating pair orientation cancels linear drift.
    def pairwise(field_path: tuple[str, str]) -> dict:
        section, field = field_path
        diffs = []
        pairs = []
        for index in range(0, len(legs) - 1, 2):
            first, second = legs[index], legs[index + 1]
            if {first["arm"], second["arm"]} != {"0", "1"}:
                continue
            by_arm = {leg["arm"]: leg for leg in (first, second)}
            a = by_arm["0"].get(section, {}).get(field)
            b = by_arm["1"].get(section, {}).get(field)
            if a is None or b is None:
                continue
            diffs.append(b - a)
            pairs.append(
                {
                    "legs": [first["leg"], second["leg"]],
                    "orientation": f"{first['arm']}{second['arm']}",
                    "arm0": a,
                    "arm1": b,
                    "delta": b - a,
                }
            )
        stats = summarize(diffs)
        stats["pairs"] = pairs
        return stats

    pooled_rounds = {}
    for arm in arms:
        values = []
        for leg in legs:
            if leg["arm"] != arm or not leg.get("mtp"):
                continue
            values.extend(leg["mtp"]["rounds_ms"][1:])
        pooled_rounds[arm] = values

    contrasts = {}
    if set(arms) == {"0", "1"}:
        pair_mtp = pairwise(("mtp", "round_ms_mean"))
        pair_mtp_median = pairwise(("mtp", "round_ms_median"))
        pair_serial = pairwise(("serial", "decode_only_seconds_per_token"))
        leg_delta = (
            per_arm["1"]["round_ms_mean"]["mean"] - per_arm["0"]["round_ms_mean"]["mean"]
        )
        leg_sigma = math.sqrt(
            per_arm["1"]["round_ms_mean"].get("sem", 0.0) ** 2
            + per_arm["0"]["round_ms_mean"].get("sem", 0.0) ** 2
        )
        pooled_delta = statistics.median(pooled_rounds["1"]) - statistics.median(
            pooled_rounds["0"]
        )
        contrasts = {
            "pairwise_round_ms_mean": pair_mtp,
            "pairwise_round_ms_median": pair_mtp_median,
            "pairwise_serial_decode_spt": pair_serial,
            "leg_means_round_ms": {
                "arm0_mean": per_arm["0"]["round_ms_mean"]["mean"],
                "arm1_mean": per_arm["1"]["round_ms_mean"]["mean"],
                "delta": leg_delta,
                "sigma": leg_sigma,
            },
            "pooled_round_ms_median": {
                "arm0_median": statistics.median(pooled_rounds["0"]),
                "arm1_median": statistics.median(pooled_rounds["1"]),
                "arm0_rounds": len(pooled_rounds["0"]),
                "arm1_rounds": len(pooled_rounds["1"]),
                "delta": pooled_delta,
            },
            "serial_null_control_pct": (
                100.0
                * pair_serial["mean"]
                / per_arm["0"]["serial_decode_only_spt"]["mean"]
                if pair_serial.get("n")
                and per_arm["0"]["serial_decode_only_spt"].get("mean")
                else None
            ),
        }
        contrasts["stop_rule_pairwise"] = stop_rule(
            -pair_mtp["mean"], pair_mtp.get("sem", 0.0)
        )
        contrasts["stop_rule_leg_means"] = stop_rule(-leg_delta, leg_sigma)

    summary = {
        "experiment": "qwen38-r1-e194-sdpa-step-removal",
        "session": str(args.session),
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "predeclared_recovery_ms_per_round": {"low": 0.10, "point": 0.30, "high": 0.65},
        "stop_threshold_ms_per_round": 0.5,
        "base_sha": args.base_sha,
        "candidate_sha": args.candidate_sha,
        "work_identity": work_identity,
        "legs": legs,
        "per_arm": per_arm,
        "contrasts": contrasts,
    }
    out_path = args.session / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "work_identity": work_identity,
                "per_arm": {
                    arm: {
                        "round_ms_mean": stats["round_ms_mean"],
                        "serial_decode_only_spt": stats["serial_decode_only_spt"],
                        "gpu_temp_entry": stats["gpu_temp_entry"],
                    }
                    for arm, stats in per_arm.items()
                },
                "contrasts": {
                    key: (
                        {k: v for k, v in value.items() if k != "pairs"}
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in contrasts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"e194_analyse: wrote {out_path}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=args.run_name or f"e194-{args.session.name}",
            notes=args.notes,
            config={
                "experiment": "qwen38-r1-e194-sdpa-step-removal",
                "harness": "local",
                "mode": "qwen-mtp-local-iterate",
                "arms": {"0": "today", "1": "route2prime-sequence-major-query"},
                "session": str(args.session),
                "base_sha": args.base_sha,
                "candidate_sha": args.candidate_sha,
                "worker_sha256": legs[0].get("worker_sha256") if legs else None,
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
                "phase_trace": 0,
                "added_timing_instrument": "none",
                "timing_source": "trusted-parent-report",
                "predeclared_recovery_ms_per_round_point": 0.30,
                "stop_threshold_ms_per_round": 0.5,
                "host_chip": subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
            },
        )
        columns = [
            "leg",
            "arm",
            "gpu_temp_entry",
            "gpu_temp_exit",
            "round_ms_mean",
            "round_ms_median",
            "round_ms_sd",
            "warm_round_count",
            "mtp_decode_only_spt",
            "serial_decode_only_spt",
            "local_ratio_charged",
            "round_count",
            "accepted_draft_total",
            "rejected_draft_total",
            "all_tokens_matched",
            "worker_sha256",
        ]
        table = wandb.Table(columns=columns)
        for leg in legs:
            mtp = leg.get("mtp", {})
            serial = leg.get("serial", {})
            table.add_data(
                leg.get("leg"),
                leg.get("arm"),
                leg.get("gpu_temp_entry"),
                leg.get("gpu_temp_exit"),
                mtp.get("round_ms_mean"),
                mtp.get("round_ms_median"),
                mtp.get("round_ms_sd"),
                mtp.get("warm_round_count"),
                mtp.get("decode_only_seconds_per_token"),
                serial.get("decode_only_seconds_per_token"),
                leg.get("score_local_ratio"),
                mtp.get("round_count"),
                mtp.get("accepted_draft_total"),
                mtp.get("rejected_draft_total"),
                mtp.get("all_tokens_matched"),
                leg.get("worker_sha256"),
            )
        run.log({"legs": table})

        rounds = wandb.Table(columns=["leg", "arm", "round_index", "round_ms"])
        for leg in legs:
            for index, value in enumerate(leg.get("mtp", {}).get("rounds_ms", [])):
                rounds.add_data(leg.get("leg"), leg.get("arm"), index, value)
        run.log({"rounds": rounds})

        flat = {}
        for arm, stats in per_arm.items():
            for metric, values in stats.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            flat[f"arm{arm}/{metric}/{key}"] = value
        for name, value in contrasts.items():
            if isinstance(value, dict):
                for key, number in value.items():
                    if isinstance(number, (int, float, bool)):
                        flat[f"contrast/{name}/{key}"] = number
            elif isinstance(value, (int, float)):
                flat[f"contrast/{name}"] = value
        flat["work_identity/identical_across_all_legs"] = work_identity[
            "identical_across_all_legs"
        ]
        flat["work_identity/all_tokens_matched_every_leg"] = work_identity[
            "all_tokens_matched_every_leg"
        ]
        run.summary.update(flat)
        run.log(flat)
        print(f"e194_analyse: wandb run {run.url}")
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
