#!/usr/bin/env python3
"""Compare draft-head readout precision arms on acceptance, fidelity and time.

Each arm directory is one `research/run-draft-bits-arm.sh` run and holds the
captured CLI reports, the amdahl roll-up and the rusage trailer. The first arm
on the command line is the control every other arm is compared against.

Acceptance and token-stream identity are properties of the head's numerics and
do not depend on temperature, so they stay trustworthy on a host whose thermal
floor sits above the 40C cool gate. Wall-clock fields are reported with the
gate outcome attached and must be read with that caveat.

Usage:
    research/draft_bits_arms.py CONTROL_DIR ARM_DIR... [--wandb] [--tag NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path


def load_mtp_report(arm_dir: Path) -> dict:
    reports = arm_dir / "reports"
    best: dict | None = None
    for path in sorted(reports.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "parity_all_ok" not in payload or "decode_seconds" not in payload:
            continue
        if payload.get("mtp_depth", 0) == 0:
            continue
        payload["_source_file"] = path.name
        best = payload
    if best is None:
        raise SystemExit(f"draft_bits_arms: no MTP leg report under {reports}")
    return best


def load_reference_golden(arm_dir: Path) -> tuple[dict, str | None]:
    """The serial golden written by `mtp-verify --generate`.

    `research/capture-cli.sh` copies it out of the scratch run directory that
    benchmark-qwen-mtp.sh deletes. It holds `emitted_tokens` (the serial
    target argmax stream) and one `rows` entry per position carrying the
    top-two tokens and logits. The draft head never runs during this pass, so
    the golden is arm-invariant -- which is exactly what makes it a usable
    anchor for cross-arm token identity.
    """
    for path in sorted((arm_dir / "reports").glob("*-output.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("emitted_tokens"):
            return payload, path.name
    return {}, None


def load_serial_report(arm_dir: Path) -> dict | None:
    for path in sorted((arm_dir / "reports").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("mtp_depth", 0) == 0:
            if "decode_seconds" in payload:
                return payload
    return None


COMPACT_DRAFT_ROWS = 98_336
COMPACT_DRAFT_K = 5_120
COMPACT_DRAFT_GROUP = 64


def draft_head_provenance(arm_dir: Path) -> dict:
    """The arm's requested readout precision and the head bytes it implies.

    The worker cannot report the head it built: on `mtp-timed` its sandbox
    denies file-write*, its stdout is the request protocol, and the parent
    discards its stderr. So the bits come from the arm's recorded request and
    the byte counts from the fixed compact-draft shape, which
    QwenQMVCostCurveTests measures directly on device.
    """
    path = arm_dir / "amdahl.json"
    if not path.exists():
        return {}
    notes = json.loads(path.read_text()).get("provenance", {}).get("notes", "")
    match = re.search(r"MLX_QWEN_MTP_DRAFT_BITS=(\d+)", notes)
    if not match:
        return {}
    bits = int(match.group(1))
    groups = COMPACT_DRAFT_ROWS * COMPACT_DRAFT_K // COMPACT_DRAFT_GROUP
    packed = COMPACT_DRAFT_ROWS * COMPACT_DRAFT_K * bits // 8
    # affine group-64: one fp16 scale and one fp16 bias per group.
    scales = groups * 2 * 2
    return {
        "bits": bits,
        "rows": COMPACT_DRAFT_ROWS,
        "packed_bytes": packed,
        "scale_bytes": scales,
        "total_bytes": packed + scales,
    }


def max_rss_bytes(arm_dir: Path) -> int | None:
    path = arm_dir / "rusage.txt"
    if not path.exists():
        return None
    best = None
    for match in re.finditer(r"(\d+)\s+maximum resident set size", path.read_text()):
        value = int(match.group(1))
        best = value if best is None else max(best, value)
    return best


def thermal_provenance(arm_dir: Path) -> dict:
    """Whether the arm's seconds are gate-qualified, and the temps around it."""
    path = arm_dir / "identity.txt"
    if not path.exists():
        return {}
    out: dict[str, object] = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key not in {"cool_gate", "gpu_temp_c_before", "gpu_temp_c_after"}:
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    out["seconds_are_gate_qualified"] = out.get("cool_gate") == "passed"
    return out


def reference_margins(golden: dict) -> dict:
    """How much logit headroom a draft has to get each position right.

    A noisier draft head can only flip positions whose top-two target logits
    are close, so this distribution bounds the acceptance any precision arm
    can lose. It explains an acceptance delta rather than restating it.
    """
    margins = [
        float(row["top2_logits"][0]) - float(row["top2_logits"][1])
        for row in golden.get("rows", [])
        if len(row.get("top2_logits", ())) >= 2
    ]
    if not margins:
        return {}
    ordered = sorted(margins)
    return {
        "margin_row_count": len(ordered),
        "margin_mean": statistics.fmean(ordered),
        "margin_p10": ordered[len(ordered) // 10],
        "margin_p50": statistics.median(ordered),
        "near_tie_fraction_lt_0p5": sum(m < 0.5 for m in ordered) / len(ordered),
        "near_tie_fraction_lt_2p0": sum(m < 2.0 for m in ordered) / len(ordered),
    }


def draft_depth_histogram(report: dict) -> dict:
    """Where the adaptive depth schedule settled.

    Mean acceptance alone hides a schedule that pulled depth back to protect
    its acceptance rate, which costs tokens per round without ever showing up
    as a worse acceptance number.
    """
    lengths = report.get("effective_draft_lengths") or []
    if not lengths:
        return {}
    counts: dict[int, int] = {}
    for length in lengths:
        counts[int(length)] = counts.get(int(length), 0) + 1
    return {
        "draft_readouts_total": sum(lengths),
        "draft_readouts_per_round": statistics.fmean(lengths),
        "draft_depth_histogram": dict(sorted(counts.items())),
    }


def summarize(arm_dir: Path) -> dict:
    report = load_mtp_report(arm_dir)
    serial = load_serial_report(arm_dir)
    golden, golden_file = load_reference_golden(arm_dir)
    stream = [int(t) for t in golden.get("emitted_tokens", [])]
    head = draft_head_provenance(arm_dir)
    seconds = report.get("parent_measured_seconds_per_token")
    out = {
        "arm_dir": str(arm_dir),
        "arm": arm_dir.name,
        "draft_head_bits": head.get("bits"),
        "draft_head_packed_bytes": head.get("packed_bytes"),
        "draft_head_scale_bytes": head.get("scale_bytes"),
        "draft_head_total_bytes": head.get("total_bytes"),
        "draft_head_rows": head.get("rows"),
        "parity_all_ok": report.get("parity_all_ok"),
        "all_tokens_matched": report.get("all_tokens_matched"),
        "residual_divergence_count": report.get("residual_divergence_count"),
        "accepted_draft_rate": report.get("accepted_draft_rate"),
        "accepted_draft_total": report.get("accepted_draft_total"),
        "rejected_draft_total": report.get("rejected_draft_total"),
        "round_count": report.get("round_count"),
        "effective_mean_draft_len": report.get("effective_mean_draft_len"),
        "ms_per_token": seconds * 1e3 if seconds else None,
        "decode_seconds": report.get("decode_seconds"),
        "ref_golden_file": golden_file,
        "ref_emitted_token_count": len(stream),
        "ref_emitted_stream_sha256": hashlib.sha256(
            json.dumps(stream).encode()
        ).hexdigest(),
        "max_rss_bytes": max_rss_bytes(arm_dir),
        "report_file": report["_source_file"],
    }
    out |= {f"ref_{k}": v for k, v in reference_margins(golden).items()}
    out |= draft_depth_histogram(report)
    out |= thermal_provenance(arm_dir)
    if serial and serial.get("parent_measured_seconds_per_token") and seconds:
        out["local_ratio_serial_over_mtp"] = (
            serial["parent_measured_seconds_per_token"] / seconds
        )
        out["serial_ms_per_token"] = (
            serial["parent_measured_seconds_per_token"] * 1e3
        )
    out["_stream"] = stream
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dirs", type=Path, nargs="+")
    ap.add_argument("--tag", default="draft-head-precision")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    arms = [summarize(d) for d in args.arm_dirs]
    control = arms[0]
    for arm in arms:
        # An arm without a rescued golden is unknown, not divergent.
        arm["ref_stream_identical_to_control"] = (
            arm["_stream"] == control["_stream"]
            if arm["_stream"] and control["_stream"]
            else None
        )
        if control["accepted_draft_rate"] and arm["accepted_draft_rate"]:
            arm["accept_rate_delta_vs_control"] = (
                arm["accepted_draft_rate"] - control["accepted_draft_rate"]
            )
            arm["accept_rate_pct_of_control"] = (
                100.0 * arm["accepted_draft_rate"] / control["accepted_draft_rate"]
            )
        if control["ms_per_token"] and arm["ms_per_token"]:
            arm["ms_per_token_delta_vs_control"] = (
                arm["ms_per_token"] - control["ms_per_token"]
            )
    for arm in arms:
        arm.pop("_stream")

    print(json.dumps({"tag": args.tag, "arms": arms}, indent=2, sort_keys=True))

    header = (
        f"\n{'arm':<28} {'bits':>4} {'MB':>7} {'parity':>7} {'same':>5} "
        f"{'accept':>7} {'vs ctl':>7} {'ms/tok':>8} {'ratio':>6}"
    )
    print(header, file=sys.stderr)
    for arm in arms:
        total = arm["draft_head_total_bytes"]
        print(
            f"{arm['arm']:<28} {str(arm['draft_head_bits']):>4} "
            f"{(total / 1e6 if total else 0):>7.1f} "
            f"{str(arm['parity_all_ok']):>7} "
            f"{str(arm['ref_stream_identical_to_control']):>5} "
            f"{(arm['accepted_draft_rate'] or 0):>7.4f} "
            f"{arm.get('accept_rate_pct_of_control', float('nan')):>6.1f}% "
            f"{(arm['ms_per_token'] or 0):>8.3f} "
            f"{arm.get('local_ratio_serial_over_mtp', float('nan')):>6.3f}",
            file=sys.stderr,
        )

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=f"draft-bits-arms-{args.tag}",
            job_type="analysis",
            group=os.environ.get(
                "WANDB_RUN_GROUP", "qwen38-r1-e6-draft-head-precision"
            ),
            config={"tag": args.tag, "notes": args.notes},
        )
        columns = [
            "arm",
            "bits",
            "packed_bytes",
            "parity_all_ok",
            "ref_stream_identical_to_control",
            "accepted_draft_rate",
            "accept_rate_pct_of_control",
            "effective_mean_draft_len",
            "ms_per_token",
            "local_ratio_serial_over_mtp",
            "max_rss_bytes",
        ]
        table = wandb.Table(columns=columns)
        for arm in arms:
            table.add_data(
                arm["arm"],
                arm["draft_head_bits"],
                arm["draft_head_packed_bytes"],
                arm["parity_all_ok"],
                arm["ref_stream_identical_to_control"],
                arm["accepted_draft_rate"],
                arm.get("accept_rate_pct_of_control"),
                arm["effective_mean_draft_len"],
                arm["ms_per_token"],
                arm.get("local_ratio_serial_over_mtp"),
                arm["max_rss_bytes"],
            )
        run.log({"arms": table})
        flat = {}
        for arm in arms:
            prefix = f"bits{arm['draft_head_bits']}"
            for key, value in arm.items():
                if isinstance(value, (int, float, bool)):
                    flat[f"{prefix}/{key}"] = value
        run.log(flat)
        run.summary.update(flat)
        print(f"WANDB_RUN_URL {run.url}", file=sys.stderr)
        print(f"WANDB_RUN_ID {run.id}", file=sys.stderr)
        run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
