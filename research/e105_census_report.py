#!/usr/bin/env python3
"""E105 rung 0 reducer: the three target dispatch families at the live cell.

    usage: research/e105_census_report.py TAG [--width 5] [--json OUT]

Reads `research/out/TAG/census.jsonl`, aggregates the E80 exclusive-GPU-time
ledger over every round snapshot, and reports the GDN prework, q/k norm+RoPE
and KV-cache-write families at the scored width.

For each family it prints the verbatim signature, grid, threadgroup shape,
threadgroup count, waves over the 20 local GPU cores, dispatches per round,
microseconds per dispatch and per round, and the achieved bandwidth against
both DRAM peak and a measured stream roofline. The roofline columns answer
the standing question of whether these dispatches are bandwidth bound: a
family that runs at a few percent of peak is not.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

GPU_CORES = 20
DRAM_PEAK_GB_S = 273.0  # M4 Pro 48 GiB, 273 GB/s
STREAM_ROOFLINE_GB_S = 249.55  # measured achievable stream bandwidth

# Bytes moved per dispatch at the scored width, from the live tensor shapes.
FAMILY_BYTES = {
    "gdn_prework": 412_000,
    "qk_rms_rope": 144_400,
    "kv_write": 20_500,
}

FAMILIES = (
    ("gdn_prework", "GDN prework", "qwen35_packed_gdn_prework"),
    ("qk_rms_rope", "q/k norm + RoPE", "qwen35_attention_qk_rms_rope"),
    ("kv_write", "KV cache write", "gg2_copy"),
)


def parse_shape(text: str, prefix: str) -> tuple[int, int, int] | None:
    for token in text.split():
        if token.startswith(prefix):
            dims = token[len(prefix) :].split("x")
            if len(dims) == 3 and all(d.isdigit() for d in dims):
                return tuple(int(d) for d in dims)  # type: ignore[return-value]
    return None


def width_anchors(tag: str) -> list[tuple[str, int, float]]:
    """GPU-busy time per round at every observed verify width.

    This is the denominator the campaign's family table uses, so it has to be
    reported next to the families themselves.
    """
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    total: dict[str, float] = collections.defaultdict(float)
    rounds: dict[str, set] = collections.defaultdict(set)
    for line in path.read_text().splitlines():
        if '"gputime"' not in line:
            continue
        rec = json.loads(line)
        for key, val in (rec.get("exclusive_kernels") or {}).items():
            w = key.split("|", 1)[0]
            total[w] += val["gpu_ns"]
            rounds[w].add(rec.get("round_last"))
    return [
        (w, len(rounds[w]), total[w] / len(rounds[w]) / 1e3) for w in sorted(total)
    ]


def load(tag: str, width: int, phase: str | None):
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    rounds: set[int] = set()
    agg: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"buffers": 0, "gpu_ns": 0, "min_ns": float("inf"), "max_ns": 0.0}
    )
    for line in path.read_text().splitlines():
        if '"gputime"' not in line:
            continue
        rec = json.loads(line)
        keys = rec.get("exclusive_kernels") or {}
        hit = False
        for key, val in keys.items():
            parts = key.split("|", 2)
            if len(parts) != 3:
                continue
            w, ph, sig = parts
            if w != f"w{width}":
                continue
            if phase is not None and ph != phase:
                continue
            hit = True
            slot = agg[f"{ph}|{sig}"]
            slot["buffers"] += val["buffers"]
            slot["gpu_ns"] += val["gpu_ns"]
            slot["min_ns"] = min(slot["min_ns"], val["min_ns"])
            slot["max_ns"] = max(slot["max_ns"], val["max_ns"])
        if hit:
            rounds.add(rec.get("round_last"))
    return agg, len(rounds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    agg, n_rounds = load(args.tag, args.width, args.phase)
    if not n_rounds:
        print(f"no w{args.width} rounds in {args.tag}", file=sys.stderr)
        return 2

    report = {
        "tag": args.tag,
        "width": args.width,
        "phase": args.phase,
        "rounds": n_rounds,
        "gpu_cores": GPU_CORES,
        "dram_peak_gb_s": DRAM_PEAK_GB_S,
        "stream_roofline_gb_s": STREAM_ROOFLINE_GB_S,
        "families": [],
    }

    total_us_round = 0.0
    total_disp_round = 0.0
    for slug, label, needle in FAMILIES:
        matches = {k: v for k, v in agg.items() if needle in k}
        if not matches:
            continue
        key, val = max(matches.items(), key=lambda kv: kv[1]["gpu_ns"])
        sig = key.split("|", 1)[1]
        grid = parse_shape(sig, "grid=")
        tg = parse_shape(sig, "tg=")
        tg_count = None
        waves = None
        if grid and tg:
            tg_count = 1
            for g, t in zip(grid, tg):
                tg_count *= -(-g // t)
            waves = tg_count / GPU_CORES
        disp_round = val["buffers"] / n_rounds
        us_disp = val["gpu_ns"] / val["buffers"] / 1e3
        us_round = val["gpu_ns"] / n_rounds / 1e3
        nbytes = FAMILY_BYTES[slug]
        gb_s = nbytes / (us_disp * 1e-6) / 1e9
        roofline_us = nbytes / (STREAM_ROOFLINE_GB_S * 1e9) * 1e6
        total_us_round += us_round
        total_disp_round += disp_round
        report["families"].append(
            {
                "family": slug,
                "label": label,
                "signature": sig,
                "grid": grid,
                "threadgroup": tg,
                "threadgroup_count": tg_count,
                "waves_over_cores": waves,
                "dispatches_per_round": disp_round,
                "us_per_dispatch": us_disp,
                "us_per_dispatch_min": val["min_ns"] / 1e3,
                "us_per_dispatch_max": val["max_ns"] / 1e3,
                "us_per_round": us_round,
                "bytes_per_dispatch": nbytes,
                "achieved_gb_s": gb_s,
                "pct_of_dram_peak": 100.0 * gb_s / DRAM_PEAK_GB_S,
                "roofline_us_per_dispatch": roofline_us,
                "roofline_share_of_measured": 100.0 * roofline_us / us_disp,
            }
        )

    report["total_dispatches_per_round"] = total_disp_round
    report["total_us_per_round"] = total_us_round
    anchors = width_anchors(args.tag)
    report["width_anchors"] = anchors
    report["census_round_us"] = next(
        (us for w, _, us in anchors if w == f"w{args.width}"), None
    )

    hdr = (
        f"{'family':16} {'tg':>6} {'tgcnt':>6} {'waves':>6} {'disp/rd':>8} "
        f"{'us/disp':>8} {'us/round':>9} {'GB/s':>7} {'%peak':>6} {'%roof':>6}"
    )
    print(f"E105 rung 0 census  tag={args.tag}  w{args.width} {args.phase}  "
          f"rounds={n_rounds}")
    print(hdr)
    for f in report["families"]:
        tg = "x".join(str(d) for d in f["threadgroup"]) if f["threadgroup"] else "?"
        print(
            f"{f['family']:16} {tg:>6} {f['threadgroup_count']:>6} "
            f"{f['waves_over_cores']:>6.2f} {f['dispatches_per_round']:>8.2f} "
            f"{f['us_per_dispatch']:>8.2f} {f['us_per_round']:>9.2f} "
            f"{f['achieved_gb_s']:>7.1f} {f['pct_of_dram_peak']:>6.1f} "
            f"{f['roofline_share_of_measured']:>6.1f}"
        )
    print(
        f"{'total':16} {'':>6} {'':>6} {'':>6} {total_disp_round:>8.2f} "
        f"{'':>8} {total_us_round:>9.2f}"
    )

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
