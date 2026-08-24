#!/usr/bin/env python3
"""Research-only (qwen38-r1-e181-local-q-pair-jit-channel).

Read one E181 session directory, split every timed leg into its prefill and
decode-only parts, and contrast the arms.

    python3 research/e181_analyze.py research/out/e181/session-submit-NAME [--wandb]

The trusted parent charges the seed prefill INTO the decode window
(QwenRuntimeMTPDriver.swift:92-100), so `parent_measured_seconds_per_token`
mixes the two phases. The captured CLI reports carry both parts, so the decode
channel is read directly:

    decode_only_seconds = decode_seconds - seed_prefill_seconds

Everything here is harness=local.
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
    """Return (serial_report, mtp_report) from the captured CLI reports."""
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


def phase_split(report: dict) -> dict:
    tokens = report["decode_token_count"]
    decode_seconds = report["decode_seconds"]
    prefill_seconds = report.get("seed_prefill_seconds")
    row = {
        "tokens": tokens,
        "decode_seconds": decode_seconds,
        "charged_seconds_per_token": report["parent_measured_seconds_per_token"],
        "prefill_seconds": prefill_seconds,
        "round_count": report.get("round_count"),
        "accepted_draft_rate": report.get("accepted_draft_rate"),
        "accepted_draft_total": report.get("accepted_draft_total"),
        "rejected_draft_total": report.get("rejected_draft_total"),
        "declared_rows_total": report.get("declared_rows_total"),
        "reference_checked_row_total": report.get("reference_checked_row_total"),
        "emitted_token_total": report.get("emitted_token_total"),
        "all_tokens_matched": report.get("all_tokens_matched"),
        "residual_divergence_count": report.get("residual_divergence_count"),
        "seed_token_count": report.get("seed_token_count"),
    }
    if prefill_seconds is None:
        row["decode_only_seconds"] = None
        row["decode_only_seconds_per_token"] = None
    else:
        decode_only = decode_seconds - prefill_seconds
        row["decode_only_seconds"] = decode_only
        row["decode_only_seconds_per_token"] = decode_only / tokens
    blocks = report.get("block_request_seconds") or []
    if blocks:
        row["block_count"] = len(blocks)
        row["block_p50_seconds"] = statistics.median(blocks)
        row["block_mean_after_first"] = (
            statistics.fmean(blocks[1:]) if len(blocks) > 1 else None
        )
    return row


def summarize(values: list[float]) -> dict:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"n": 0}
    out = {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
        "half_range": (max(clean) - min(clean)) / 2.0,
    }
    out["sd"] = statistics.stdev(clean) if len(clean) > 1 else 0.0
    out["sem"] = out["sd"] / math.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return out


def contrast(a: dict, b: dict) -> dict:
    """Percentage effect of arm b relative to arm a, with a pooled interval."""
    if not a.get("n") or not b.get("n"):
        return {}
    delta = b["mean"] - a["mean"]
    pct = 100.0 * delta / a["mean"]
    sem = math.sqrt(a.get("sem", 0.0) ** 2 + b.get("sem", 0.0) ** 2)
    return {
        "baseline_mean": a["mean"],
        "candidate_mean": b["mean"],
        "delta": delta,
        "delta_pct": pct,
        "pooled_sem": sem,
        "pooled_sem_pct": 100.0 * sem / a["mean"],
        "ci95_pct_low": pct - 196.0 * sem / a["mean"],
        "ci95_pct_high": pct + 196.0 * sem / a["mean"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    legs = []
    for leg_dir in sorted(p for p in args.session.iterdir() if p.is_dir()):
        meta_path = leg_dir / "meta.txt"
        if not meta_path.exists():
            continue
        meta = read_meta(meta_path)
        serial, mtp = leg_reports(leg_dir)
        leg = {
            "leg": int(meta.get("e181_leg", 0)),
            "arm": meta.get("e181_arm"),
            "dir": str(leg_dir),
            "exit": meta.get("exit"),
            "gpu_temp_entry": float(meta["gpu_temp_entry"])
            if meta.get("gpu_temp_entry", "unavailable") != "unavailable"
            else None,
            "gpu_temp_exit": float(meta["gpu_temp_exit"])
            if meta.get("gpu_temp_exit", "unavailable") != "unavailable"
            else None,
            "worker_sha256": meta.get("worker_sha256_before"),
            "worker_digest_stable": meta.get("worker_digest_stable"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        }
        score_path = leg_dir / "score.json"
        if score_path.exists():
            score = json.loads(score_path.read_text())
            leg["score_mtp_decode_speedup"] = score["metrics"]["mtp_decode_speedup"]
            leg["score_serial_spt"] = score["metrics"]["serial_seconds_per_token"]
            leg["score_mtp_spt"] = score["metrics"]["mtp_seconds_per_token"]
        if serial:
            leg["serial"] = phase_split(serial)
        if mtp:
            leg["mtp"] = phase_split(mtp)
        if serial and mtp:
            s_decode = leg["serial"].get("decode_only_seconds_per_token")
            m_decode = leg["mtp"].get("decode_only_seconds_per_token")
            if s_decode and m_decode:
                leg["decode_only_speedup"] = s_decode / m_decode
        legs.append(leg)

    arms = sorted({leg["arm"] for leg in legs if leg.get("arm")})
    per_arm = {}
    for arm in arms:
        rows = [leg for leg in legs if leg["arm"] == arm]
        per_arm[arm] = {
            "legs": [row["leg"] for row in rows],
            "mtp_decode_only_spt": summarize(
                [r.get("mtp", {}).get("decode_only_seconds_per_token") for r in rows]
            ),
            "mtp_prefill_seconds": summarize(
                [r.get("mtp", {}).get("prefill_seconds") for r in rows]
            ),
            "mtp_charged_spt": summarize(
                [r.get("mtp", {}).get("charged_seconds_per_token") for r in rows]
            ),
            "serial_decode_only_spt": summarize(
                [r.get("serial", {}).get("decode_only_seconds_per_token") for r in rows]
            ),
            "serial_prefill_seconds": summarize(
                [r.get("serial", {}).get("prefill_seconds") for r in rows]
            ),
            "decode_only_speedup": summarize(
                [r.get("decode_only_speedup") for r in rows]
            ),
            "local_ratio_charged": summarize(
                [r.get("score_mtp_decode_speedup") for r in rows]
            ),
            "gpu_temp_entry": summarize([r.get("gpu_temp_entry") for r in rows]),
            "accepted_draft_rate": summarize(
                [r.get("mtp", {}).get("accepted_draft_rate") for r in rows]
            ),
            "round_count": summarize([r.get("mtp", {}).get("round_count") for r in rows]),
        }

    contrasts = {}
    if "P" in per_arm:
        for arm in arms:
            if arm == "P":
                continue
            contrasts[f"P_to_{arm}"] = {
                key: contrast(per_arm["P"][key], per_arm[arm][key])
                for key in (
                    "mtp_decode_only_spt",
                    "mtp_prefill_seconds",
                    "mtp_charged_spt",
                    "serial_decode_only_spt",
                    "serial_prefill_seconds",
                    "local_ratio_charged",
                )
            }

    # Warm-only sensitivity. "Drop the cold opening leg, plus one candidate leg
    # for balance" is a judgement call, so enumerate EVERY choice of the dropped
    # candidate leg. If the branch decision survives all of them, the cut is not
    # doing the work; if it does not, the cut is the finding.
    sensitivity = {}
    if "P" in per_arm and len(arms) > 1:
        cold_leg = min(legs, key=lambda leg: leg.get("gpu_temp_entry", math.inf))["leg"]
        keys = (
            "mtp_decode_only_spt",
            "mtp_prefill_seconds",
            "serial_decode_only_spt",
            "serial_prefill_seconds",
        )
        metric_path = {
            "mtp_decode_only_spt": ("mtp", "decode_only_seconds_per_token"),
            "mtp_prefill_seconds": ("mtp", "prefill_seconds"),
            "serial_decode_only_spt": ("serial", "decode_only_seconds_per_token"),
            "serial_prefill_seconds": ("serial", "prefill_seconds"),
        }

        def arm_stats(arm: str, drop: set[int]) -> dict:
            rows = [l for l in legs if l["arm"] == arm and l["leg"] not in drop]
            return {
                key: summarize([r.get(sec, {}).get(field) for r in rows])
                for key, (sec, field) in metric_path.items()
            }

        for arm in arms:
            if arm == "P":
                continue
            cand_legs = [l["leg"] for l in legs if l["arm"] == arm]
            variants = {}
            for extra in [None, *cand_legs]:
                drop_p = {cold_leg}
                drop_c = {cold_leg} | ({extra} if extra is not None else set())
                label = "cold_only" if extra is None else f"cold_plus_{arm}_leg{extra}"
                a = arm_stats("P", drop_p)
                b = arm_stats(arm, drop_c)
                variants[label] = {
                    key: contrast(a[key], b[key]) for key in keys
                }
            decode = [
                v["mtp_decode_only_spt"].get("delta_pct")
                for v in variants.values()
                if v.get("mtp_decode_only_spt")
            ]
            sensitivity[f"P_to_{arm}"] = {
                "cold_leg_dropped": cold_leg,
                "variants": variants,
                "mtp_decode_only_pct_worst_abs": max(decode, key=abs) if decode else None,
                "branch_stable_within_band": (
                    all(abs(d) <= 0.2 for d in decode) if decode else None
                ),
            }

    summary = {
        "session": str(args.session),
        "harness": "local",
        "gate_qualified_for_timing": False,
        "cool_gate_passed_real_gate": False,
        "predeclared_noise_pct": 0.2,
        "legs": legs,
        "per_arm": per_arm,
        "contrasts": contrasts,
        "warm_only_sensitivity": sensitivity,
    }
    out_path = args.session / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"per_arm": per_arm, "contrasts": contrasts}, indent=2, sort_keys=True))
    print(f"e181_analyze: wrote {out_path}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=args.run_name or f"e181-{args.session.name}",
            notes=args.notes,
            config={
                "experiment": "qwen38-r1-e181-local-q-pair-jit-channel",
                "harness": "local",
                "arms": arms,
                "session": str(args.session),
                "cool_gate_passed_real_gate": False,
                "gate_qualified_for_timing": False,
                "phase_trace": 0,
                "added_timing_instrument": "none",
                "timing_source": "trusted-parent-report",
                "predeclared_noise_pct": 0.2,
                "host_chip": subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip(),
                "organizer_sha": "0863b06ac16e26e48fc06e97444095b00feb66d4",
                "base_sha": "38634ebfbf67f8ccc8a5a404acbc15e749d26b7a",
            },
        )
        columns = [
            "leg",
            "arm",
            "gpu_temp_entry",
            "gpu_temp_exit",
            "worker_sha256",
            "mtp_decode_only_spt",
            "mtp_prefill_seconds",
            "mtp_charged_spt",
            "serial_decode_only_spt",
            "serial_prefill_seconds",
            "decode_only_speedup",
            "local_ratio_charged",
            "accepted_draft_rate",
            "round_count",
            "all_tokens_matched",
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
                leg.get("worker_sha256"),
                mtp.get("decode_only_seconds_per_token"),
                mtp.get("prefill_seconds"),
                mtp.get("charged_seconds_per_token"),
                serial.get("decode_only_seconds_per_token"),
                serial.get("prefill_seconds"),
                leg.get("decode_only_speedup"),
                leg.get("score_mtp_decode_speedup"),
                mtp.get("accepted_draft_rate"),
                mtp.get("round_count"),
                mtp.get("all_tokens_matched"),
            )
        run.log({"legs": table})
        flat = {}
        for arm, stats in per_arm.items():
            for metric, values in stats.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flat[f"{arm}/{metric}/{key}"] = value
        for name, values in contrasts.items():
            for metric, value in values.items():
                for key, number in value.items():
                    flat[f"{name}/{metric}/{key}"] = number
        run.summary.update(flat)
        run.log(flat)
        print(f"e181_analyze: wandb run {run.url}")
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
