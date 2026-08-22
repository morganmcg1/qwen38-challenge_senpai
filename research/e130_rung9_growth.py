#!/usr/bin/env python3
"""E130 rung 9: how much memory the scored window allocates AFTER the wired
residency ticket is sized.

The ticket in `Qwen36MTPBlockSession.wireResidentWeightsIfEnabled` is

    target = activeMemory * fraction + slackMB << 20     fraction 1.0, slack 64 MiB
    target = min(target, maxRecommendedWorkingSetBytes - 256 MiB)

sized once, at the end of the input-independent warm, and never resized. Every
byte the run allocates after that instant must fit inside the 64 MiB slack or
the driver drops something from the residency set.

F10 predicts 143.75 to 215.75 MiB of post-sizing growth from layer counts, head
dimensions and the token window:

    target KV, 16 full-attention layers   16*4*256*2*2*1024      64.00 MiB
    GDN recurrent state, 48 layers        48*48*128*128*{2,4}  72.0..144.0 MiB
    GDN conv state, 48 layers             48*10240*4*2           3.75 MiB
    head history KV, 1 MTP layer          1*4*256*2*2*1024       4.00 MiB

Those terms are fixed by the config and the token window, both identical on the
ranked M5, so the growth transfers exactly even though the 96 GiB guard keeps
the wired ticket itself off a 48 GiB host.

Usage
-----
    python3 research/e130_rung9_growth.py \
        --leg 512 research/out/e130-rung9-t512/residency.log \
        --leg 128 research/out/e130-rung9-t128/residency.log \
        --out research/e130-artifacts/rung9-allocation-growth.json

Reading the log
---------------
Every worker process of a leg -- reference row generation, serial control and
MTP decode -- opens the same file O_APPEND and tags each line with its pid, so
the reader groups by pid first. `growth` on a sample line is already relative to
the FIRST sizing event of that process, because the sampler closes over that
event's `active`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

MIB = 1 << 20
SHIPPED_SLACK_MIB = 64

# F10's arithmetic, recomputed here from Qwen35Config.swift:225-260 rather than
# copied, so a config change breaks the prediction instead of the report.
FA_LAYERS = 16
KV_HEADS = 4
HEAD_DIM = 256
GDN_LAYERS = 48
GDN_HEADS = 48
GDN_K = 128
GDN_V = 128
GDN_CONV_CHANNELS = 10240
GDN_CONV_KERNEL = 4
MTP_LAYERS = 1
KV_DTYPE_BYTES = 2


def predicted_growth_mib(tokens: int, seed: int = 512) -> dict:
    window = seed + tokens
    target_kv = FA_LAYERS * KV_HEADS * HEAD_DIM * 2 * KV_DTYPE_BYTES * window
    head_kv = MTP_LAYERS * KV_HEADS * HEAD_DIM * 2 * KV_DTYPE_BYTES * window
    conv = GDN_LAYERS * GDN_CONV_CHANNELS * GDN_CONV_KERNEL * KV_DTYPE_BYTES
    recurrent = {
        bytes_per: GDN_LAYERS * GDN_HEADS * GDN_K * GDN_V * bytes_per
        for bytes_per in (2, 4)
    }
    return {
        "token_window": window,
        "target_kv_mib": target_kv / MIB,
        "head_history_kv_mib": head_kv / MIB,
        "gdn_conv_mib": conv / MIB,
        "gdn_recurrent_fp16_mib": recurrent[2] / MIB,
        "gdn_recurrent_fp32_mib": recurrent[4] / MIB,
        "total_fp16_mib": (target_kv + head_kv + conv + recurrent[2]) / MIB,
        "total_fp32_mib": (target_kv + head_kv + conv + recurrent[4]) / MIB,
        # mamba_ssm_dtype is "float32" on this checkpoint, so fp32 is expected.
        "expected_mib": (target_kv + head_kv + conv + recurrent[4]) / MIB,
    }


FIELD = re.compile(r"([a-z_0-9]+)=(-?[0-9]+|true|false|[0-9.]+)")


def parse(path: Path) -> dict[int, dict]:
    processes: dict[int, dict] = {}
    order: list[int] = []
    for raw in path.read_text().splitlines():
        if "e130-residency" not in raw:
            continue
        fields = dict(FIELD.findall(raw))
        pid = int(fields["pid"])
        if pid not in processes:
            processes[pid] = {"pid": pid, "sizing": [], "samples": []}
            order.append(pid)
        record = {
            k: (int(v) if re.fullmatch(r"-?[0-9]+", v) else v)
            for k, v in fields.items()
        }
        if fields["phase"] == "sizing":
            processes[pid]["sizing"].append(record)
        else:
            processes[pid]["samples"].append(record)
    for index, pid in enumerate(order):
        processes[pid]["appearance_index"] = index
    return processes


def summarize(process: dict) -> dict:
    sizing = process["sizing"]
    samples = process["samples"]
    first = sizing[0]
    active_at_sizing = first["active"]
    peak_at_sizing = first["peak"]

    actives = [s["active"] for s in samples]
    peaks = [s["peak"] for s in samples]
    out = {
        "pid": process["pid"],
        "appearance_index": process["appearance_index"],
        "sizing_events": len(sizing),
        "sample_count": len(samples),
        "active_at_sizing": active_at_sizing,
        "peak_at_sizing": peak_at_sizing,
        "active_at_sizing_mib": active_at_sizing / MIB,
        "peak_at_sizing_mib": peak_at_sizing / MIB,
        "slack_mb_shipped": first.get("slack_mb"),
        "maxrec": first.get("maxrec"),
        "physmem": first.get("physmem"),
        "wired_gate_passed": first.get("wired_gate_passed"),
        # Later sizing events are separate sessions in the same process. Their
        # active counts are diagnostic, not part of the first ticket's growth.
        "later_sizing_active_mib": [s["active"] / MIB for s in sizing[1:]],
    }
    if not samples:
        return out
    final_active = actives[-1]
    max_active = max(actives)
    max_peak = max(peaks)
    out.update(
        {
            "elapsed_s_last": samples[-1]["elapsed_s"],
            "active_final_mib": final_active / MIB,
            "active_max_mib": max_active / MIB,
            "peak_max_mib": max_peak / MIB,
            "growth_final_mib": (final_active - active_at_sizing) / MIB,
            "growth_max_mib": (max_active - active_at_sizing) / MIB,
            # Peak is cumulative from process start and is never reset by the
            # worker, so it already contains the pre-sizing warm transient.
            # Report both readings and never quote the literal difference alone.
            "peak_minus_active_at_sizing_mib": (max_peak - active_at_sizing) / MIB,
            "peak_growth_after_sizing_mib": (max_peak - peak_at_sizing) / MIB,
            "exceeds_shipped_slack": (max_active - active_at_sizing) / MIB
            > SHIPPED_SLACK_MIB,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leg",
        nargs=2,
        action="append",
        metavar=("TOKENS", "LOG"),
        required=True,
    )
    parser.add_argument("--out")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    report = {
        "experiment": "e130-rung9",
        "question": "does the scored window allocate more than the 64 MiB wired slack after the ticket is sized",
        "shipped_slack_mib": SHIPPED_SLACK_MIB,
        "instrument": "MLX allocator counters, research-only probe",
        "instrument_is_timing_safe": False,
        "legs": [],
    }

    for tokens_raw, log in args.leg:
        tokens = int(tokens_raw)
        path = Path(log)
        processes = parse(path)
        summaries = [summarize(p) for p in processes.values()]
        summaries.sort(key=lambda s: s["appearance_index"])
        # The leg runs reference generation, then the serial control, then the
        # MTP decode, each in its own worker process, so first appearance order
        # names them.
        roles = ["reference", "serial_control", "mtp_decode"]
        for index, summary in enumerate(summaries):
            summary["role_by_appearance"] = (
                roles[index] if index < len(roles) else f"extra_{index}"
            )
        leg = {
            "decode_tokens": tokens,
            "log": str(path),
            "predicted": predicted_growth_mib(tokens),
            "processes": summaries,
        }
        report["legs"].append(leg)

        print("=" * 78)
        print(f"LEG {tokens} decode tokens   {path}")
        print("=" * 78)
        pred = leg["predicted"]
        print(
            f"  predicted post-sizing growth, F10 arithmetic on a "
            f"{pred['token_window']}-token window:"
        )
        print(f"    target KV, 16 FA layers        {pred['target_kv_mib']:8.2f} MiB")
        print(
            f"    GDN recurrent, 48 layers       "
            f"{pred['gdn_recurrent_fp16_mib']:8.2f} .. "
            f"{pred['gdn_recurrent_fp32_mib']:.2f} MiB"
        )
        print(f"    GDN conv, 48 layers            {pred['gdn_conv_mib']:8.2f} MiB")
        print(f"    head history KV                {pred['head_history_kv_mib']:8.2f} MiB")
        print(
            f"    total                          {pred['total_fp16_mib']:8.2f} .. "
            f"{pred['total_fp32_mib']:.2f} MiB"
        )
        print(f"    expected, mamba_ssm_dtype=float32  {pred['expected_mib']:8.2f} MiB")
        print(f"    shipped slack                  {SHIPPED_SLACK_MIB:8.2f} MiB")
        print()
        header = (
            f"  {'role':<15}{'pid':>7}{'events':>7}{'sizing MiB':>12}"
            f"{'final MiB':>11}{'max MiB':>10}{'growth MiB':>12}"
            f"{'peak-size MiB':>15}"
        )
        print(header)
        for summary in summaries:
            print(
                f"  {summary['role_by_appearance']:<15}{summary['pid']:>7}"
                f"{summary['sizing_events']:>7}"
                f"{summary['active_at_sizing_mib']:>12.2f}"
                f"{summary.get('active_final_mib', float('nan')):>11.2f}"
                f"{summary.get('active_max_mib', float('nan')):>10.2f}"
                f"{summary.get('growth_max_mib', float('nan')):>12.2f}"
                f"{summary.get('peak_growth_after_sizing_mib', float('nan')):>15.2f}"
            )
        print()
        for summary in summaries:
            growth = summary.get("growth_max_mib")
            if growth is None:
                continue
            verdict = (
                "EXCEEDS the 64 MiB slack"
                if growth > SHIPPED_SLACK_MIB
                else "fits inside the 64 MiB slack"
            )
            print(
                f"  {summary['role_by_appearance']:<15} growth {growth:8.2f} MiB "
                f"-> {verdict}"
            )
        print()

    if len(report["legs"]) == 2:
        by_tokens = {leg["decode_tokens"]: leg for leg in report["legs"]}
        print("=" * 78)
        print("TOKEN SCALING: separate the fixed terms from the per-token terms")
        print("=" * 78)
        for role in ("serial_control", "mtp_decode"):
            points = {}
            for tokens, leg in by_tokens.items():
                for process in leg["processes"]:
                    if process["role_by_appearance"] == role and "growth_max_mib" in process:
                        points[tokens] = process["growth_max_mib"]
            if len(points) != 2:
                continue
            (t_lo, g_lo), (t_hi, g_hi) = sorted(points.items())
            per_token_kib = (g_hi - g_lo) * 1024 / (t_hi - t_lo)
            fixed = g_lo - (g_hi - g_lo) * t_lo / (t_hi - t_lo)
            print(
                f"  {role:<15} {t_lo} tok {g_lo:8.2f} MiB   "
                f"{t_hi} tok {g_hi:8.2f} MiB"
            )
            print(
                f"  {'':<15} slope {per_token_kib:8.2f} KiB per decode token, "
                f"fixed intercept {fixed:8.2f} MiB"
            )
            report.setdefault("token_scaling", {})[role] = {
                "growth_mib": {str(t_lo): g_lo, str(t_hi): g_hi},
                "kib_per_decode_token": per_token_kib,
                "fixed_intercept_mib": fixed,
            }
        print()

    verdicts = []
    for leg in report["legs"]:
        for process in leg["processes"]:
            if process["role_by_appearance"] == "mtp_decode" and "growth_max_mib" in process:
                verdicts.append((leg["decode_tokens"], process["growth_max_mib"]))
    if verdicts:
        print("=" * 78)
        print("RUNG 9 VERDICT")
        print("=" * 78)
        for tokens, growth in sorted(verdicts):
            band = predicted_growth_mib(tokens)
            inside = band["total_fp16_mib"] <= growth <= band["total_fp32_mib"]
            print(
                f"  {tokens:>4} tokens: MTP-leg post-sizing growth {growth:8.2f} MiB "
                f"against a {SHIPPED_SLACK_MIB} MiB slack, "
                f"{growth / SHIPPED_SLACK_MIB:.2f}x the allowance; "
                f"F10 band [{band['total_fp16_mib']:.2f}, {band['total_fp32_mib']:.2f}] "
                f"MiB {'CONTAINS' if inside else 'does NOT contain'} it"
            )
        report["verdict"] = {
            "mtp_growth_mib_by_tokens": {str(t): g for t, g in sorted(verdicts)},
            "shipped_slack_mib": SHIPPED_SLACK_MIB,
            "slack_is_too_small": all(
                g > SHIPPED_SLACK_MIB for _, g in verdicts
            ),
        }

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity="wandb-applied-ai-team",
            project="qwen38-mlx-challenge-senpai",
            id="e130rung9",
            name="e130rung9",
            resume="allow",
            config={
                "experiment": "e130-rung9",
                "shipped_slack_mib": SHIPPED_SLACK_MIB,
                "instrument": "mlx-allocator-counters",
                "instrument_is_timing_safe": False,
            },
        )
        flat = {}
        for leg in report["legs"]:
            tokens = leg["decode_tokens"]
            for process in leg["processes"]:
                role = process["role_by_appearance"]
                for key in (
                    "active_at_sizing_mib",
                    "growth_max_mib",
                    "growth_final_mib",
                    "peak_growth_after_sizing_mib",
                ):
                    if key in process:
                        flat[f"e130_rung9_t{tokens}_{role}_{key}"] = process[key]
            flat[f"e130_rung9_t{tokens}_predicted_expected_mib"] = leg["predicted"][
                "expected_mib"
            ]
        if "verdict" in report:
            flat["e130_rung9_slack_is_too_small"] = int(
                report["verdict"]["slack_is_too_small"]
            )
        run.log(flat)
        run.summary.update(flat)
        run.finish()
        print("logged to W&B run e130rung9")


if __name__ == "__main__":
    main()
