#!/usr/bin/env python3
"""E94: chosen-depth histogram and leg summary for one or more traced legs.

usage:
  research/e94_legs.py TAG [TAG ...] [--out research/e94-artifacts/rung1.json]

Reads `research/out/TAG/{meta.txt,score.json,trace.txt}`. The trace holds the
candidate MTP leg only: the serial denominator runs outside the block session.
The chosen-depth histogram is the deliverable; seconds per token is the
confirmation.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROUND_RE = re.compile(
    r"mtp-trace: round=(\d+) d=(\d+) acc=(\d+).*? round_us=(\d+)")
SCORE_KEYS = [
    "decode_tokens", "mtp_depth", "all_tokens_matched",
    "effective_mean_draft_len", "serial_seconds_per_token",
    "mtp_seconds_per_token", "mtp_decode_speedup", "accepted_draft_rate",
    "residual_divergence_count", "head_provenance_sha256",
    "uses_pinned_mtp_head",
]
META_KEYS = [
    "tag", "e94_cap", "e94_arm", "e94_order", "experiment", "tokens",
    "base_sha", "dirty_candidate_paths", "host", "chip", "memory_bytes",
    "worker_sha256", "post_run_worker_sha256", "cli_sha256",
    "post_run_cli_sha256", "metallib_source_fingerprint", "head_dir",
    "cool_gate_passed_real_gate", "gate_qualified_for_timing",
    "official_or_ranked_score", "gpu_temp_entry_c", "gpu_temp_exit_c",
    "started", "finished", "exit", "trace_rounds",
]


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def read_rounds(path: Path) -> list[tuple[int, int, int, int]]:
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROUND_RE.search(line)
        if m:
            rounds.append(tuple(int(g) for g in m.groups()))
    return rounds


def summarise(tag: str, root: Path) -> dict:
    leg = root / tag
    meta = read_meta(leg / "meta.txt")
    score = json.loads((leg / "score.json").read_text())["metrics"]
    rounds = read_rounds(leg / "trace.txt")

    depths = [d for _, d, _, _ in rounds]
    accepted = [a for _, _, a, _ in rounds]
    busy = [u for _, _, _, u in rounds]
    hist = Counter(depths)
    total = len(rounds)
    per_depth = {}
    for depth in sorted(hist):
        sel = [(a, u) for _, d, a, u in rounds if d == depth]
        per_depth[depth] = {
            "rounds": hist[depth],
            "fraction": hist[depth] / total,
            "verify_width": depth + 1,
            "mean_accepted": statistics.fmean(a for a, _ in sel),
            "median_round_us": statistics.median(u for _, u in sel),
            "mean_round_us": statistics.fmean(u for _, u in sel),
            "round_us_share": sum(u for _, u in sel) / sum(busy),
            "tokens_emitted": sum(a + 1 for a, _ in sel),
        }
    tokens_emitted = sum(a + 1 for a in accepted)

    out = {
        "tag": tag,
        "rounds": total,
        "tokens_emitted": tokens_emitted,
        "mean_chosen_depth": statistics.fmean(depths),
        "median_chosen_depth": statistics.median(depths),
        "mean_verify_width": statistics.fmean(d + 1 for d in depths),
        "mean_accepted_per_round": statistics.fmean(accepted),
        "mean_tokens_per_round": tokens_emitted / total,
        "mean_round_us": statistics.fmean(busy),
        "median_round_us": statistics.median(busy),
        "round_us_per_token": sum(busy) / tokens_emitted,
        "first_round_us": busy[0],
        "depth_histogram": per_depth,
        "meta": {k: meta.get(k) for k in META_KEYS},
        "score": {k: score.get(k) for k in SCORE_KEYS},
    }
    return out


def table(legs: list[dict]) -> str:
    caps = sorted({int(l["meta"]["e94_cap"] or 0) for l in legs})
    depths = sorted({d for l in legs for d in l["depth_histogram"]})
    lines = []
    head = ["tag", "cap", "arm", "rounds", "mean_d", "mean_M", "eff_draft",
            "acc_rate", "s/tok", "us/tok(round)", "T_in", "T_out"]
    lines.append(" | ".join(head))
    for leg in legs:
        s = leg["score"]
        m = leg["meta"]
        lines.append(" | ".join([
            leg["tag"], str(m["e94_cap"]), str(m["e94_arm"]),
            str(leg["rounds"]),
            f"{leg['mean_chosen_depth']:.3f}",
            f"{leg['mean_verify_width']:.3f}",
            f"{s['effective_mean_draft_len']:.4f}",
            f"{s['accepted_draft_rate']:.4f}",
            f"{s['mtp_seconds_per_token']:.6f}",
            f"{leg['round_us_per_token']:.0f}",
            f"{float(m['gpu_temp_entry_c'] or 'nan'):.1f}",
            f"{float(m['gpu_temp_exit_c'] or 'nan'):.1f}",
        ]))
    lines.append("")
    lines.append("depth histogram, fraction of rounds")
    lines.append(" | ".join(["tag"] + [f"d={d}" for d in depths]))
    for leg in legs:
        row = [leg["tag"]]
        for d in depths:
            cell = leg["depth_histogram"].get(d)
            row.append(f"{cell['fraction']:.4f}" if cell else "-")
        lines.append(" | ".join(row))
    lines.append("")
    lines.append(f"caps seen: {caps}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--root", default="research/out")
    ap.add_argument("--out")
    args = ap.parse_args()

    root = Path(args.root)
    legs = [summarise(tag, root) for tag in args.tags]
    print(table(legs))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"legs": legs}, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
