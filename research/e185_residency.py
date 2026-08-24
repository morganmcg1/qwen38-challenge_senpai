#!/usr/bin/env python3
"""E185 stage 1 reader: GPU hardware idle inside the decode rounds only.

    usage: research/e185_residency.py research/out/TAG [--json OUT]

`powermetrics -f plist` gives `elapsed_ns` and `idle_ns` per sample from the
same hardware counter, so the idle share of a window is a ratio of two
nanosecond sums. The wall timestamp it prints is truncated to one second,
which is 5.5 sample periods, so the sample stream is placed on the mach uptime
axis by its own cumulative `elapsed_ns` and one fitted offset instead.

The window is the union of the `mtp-anchor:` rounds, so the seed, the serial
leg and the reference generation are outside it by construction. E184 owns the
seed and this reader must never mix it in.

harness=local. Observation only: the leg carries a system-wide sampler and the
in-repo round trace, so its absolute seconds per token are not a timed result.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

NS = 1_000_000_000


@dataclass
class Sample:
    end_mono_ns: int
    elapsed_ns: int
    idle_ns: int
    freq_hz: float

    @property
    def start_mono_ns(self) -> int:
        return self.end_mono_ns - self.elapsed_ns


@dataclass
class Round:
    index: int
    pid: int
    anchors: dict[str, int]

    @property
    def t0(self) -> int:
        return self.anchors["t_round0"]

    @property
    def t1(self) -> int:
        return self.anchors["t_tail_done"]


def read_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def read_plist_stream(path: Path) -> list[tuple[float, int, int, float]]:
    """Return (wall_seconds, elapsed_ns, idle_ns, freq_hz) per sample."""
    blobs = [b for b in path.read_bytes().split(b"\x00") if b.strip()]
    out = []
    for blob in blobs:
        try:
            doc = plistlib.loads(blob)
        except Exception:
            continue
        gpu = doc.get("gpu")
        if gpu is None or "idle_ns" not in gpu:
            continue
        out.append(
            (
                doc["timestamp"].timestamp(),
                int(doc["elapsed_ns"]),
                int(gpu["idle_ns"]),
                float(gpu.get("freq_hz", 0.0)),
            )
        )
    return out


def fit_sample_axis(raw, wall_to_mono_offset_ns: int) -> tuple[list[Sample], float]:
    """Place samples on the mach uptime axis.

    Each sample's own `elapsed_ns` gives an exact relative axis. The printed
    wall timestamp is `floor` to the second, so for the true end time `T_i`,
    `w_i <= T_i < w_i + 1`. With `c_i` the cumulative relative end time, the
    offset `o = T_i - c_i` satisfies `w_i - c_i <= o`, and the tightest
    estimate over many samples is the maximum of that difference.
    """
    cumulative = 0
    rel_end = []
    for _, elapsed_ns, _, _ in raw:
        cumulative += elapsed_ns
        rel_end.append(cumulative)

    diffs = [w - c / NS for (w, _, _, _), c in zip(raw, rel_end)]
    offset_wall = max(diffs)
    residual_span = offset_wall - min(diffs)

    samples = []
    for (_, elapsed_ns, idle_ns, freq_hz), c in zip(raw, rel_end):
        end_wall_ns = int(round((offset_wall + c / NS) * NS))
        samples.append(
            Sample(
                end_mono_ns=end_wall_ns - wall_to_mono_offset_ns,
                elapsed_ns=elapsed_ns,
                idle_ns=idle_ns,
                freq_hz=freq_hz,
            )
        )
    return samples, residual_span


def read_rounds(path: Path) -> list[Round]:
    rounds = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-anchor: "):
            continue
        fields = {}
        for token in line[len("mtp-anchor: ") :].split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                fields[key] = int(value)
            except ValueError:
                pass
        if "t_round0" not in fields or "t_tail_done" not in fields:
            continue
        rounds.append(
            Round(index=fields.get("round", -1), pid=fields.get("pid", -1), anchors=fields)
        )
    return rounds


def largest_round_block(rounds: list[Round], gap_ns: int = 2 * NS) -> list[Round]:
    """Split rounds into contiguous blocks and keep the longest.

    One leg spawns more than one worker and more than one generation phase, so
    the anchors are not a single decode run. Blocks are separated by seconds of
    model loading and reference work, and the decode leg is the longest block.
    """
    if not rounds:
        return []
    ordered = sorted(rounds, key=lambda r: r.t0)
    blocks: list[list[Round]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.t0 - prev.t1 > gap_ns or cur.pid != prev.pid:
            blocks.append([cur])
        else:
            blocks[-1].append(cur)
    return max(blocks, key=lambda b: b[-1].t1 - b[0].t0)


def window_idle(samples: list[Sample], t0: int, t1: int) -> tuple[int, int, int]:
    """Sum `elapsed_ns` and `idle_ns` over samples fully inside [t0, t1)."""
    elapsed = idle = count = 0
    for s in samples:
        if s.start_mono_ns >= t0 and s.end_mono_ns <= t1:
            elapsed += s.elapsed_ns
            idle += s.idle_ns
            count += 1
    return elapsed, idle, count


def phase_profile(samples, block, bins: int = 10):
    """Bin idle by normalised position inside the round.

    A 26 ms sample cannot resolve a 180 µs phase, but 500 samples spread over
    80 rounds can still say whether the idle sits at the round boundary or is
    uniform, which is the part of the location question this stage can answer.
    """
    sums = [[0, 0, 0] for _ in range(bins)]
    for r in block:
        span = r.t1 - r.t0
        if span <= 0:
            continue
        for s in samples:
            mid = s.end_mono_ns - s.elapsed_ns // 2
            if not (r.t0 <= mid < r.t1):
                continue
            pos = (mid - r.t0) / span
            b = min(bins - 1, int(pos * bins))
            sums[b][0] += s.elapsed_ns
            sums[b][1] += s.idle_ns
            sums[b][2] += 1
    return [
        {
            "bin": i,
            "position_lo": i / bins,
            "position_hi": (i + 1) / bins,
            "elapsed_ms": e / 1e6,
            "idle_ms": d / 1e6,
            "idle_pct": (100.0 * d / e) if e else None,
            "samples": n,
        }
        for i, (e, d, n) in enumerate(sums)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--skip-rounds", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    meta = read_meta(out / "meta.txt")
    raw = read_plist_stream(out / "powermetrics.plist")
    if not raw:
        print("e185_residency: no gpu samples", file=sys.stderr)
        return 2

    pre_wall, pre_mono = meta["e185_clock_pre_wall_mono"].split()
    post_wall, post_mono = meta["e185_clock_post_wall_mono"].split()
    offset_pre = int(round(float(pre_wall) * NS)) - int(pre_mono)
    offset_post = int(round(float(post_wall) * NS)) - int(post_mono)
    offset = (offset_pre + offset_post) // 2

    samples, axis_residual_span = fit_sample_axis(raw, offset)

    rounds = read_rounds(out / "trace.txt")
    block = largest_round_block(rounds)
    if len(block) <= args.skip_rounds + 4:
        print("e185_residency: decode block too short", file=sys.stderr)
        return 2
    analysed = block[args.skip_rounds :]

    t0, t1 = analysed[0].t0, analysed[-1].t1
    elapsed_ns, idle_ns, n_samples = window_idle(samples, t0, t1)
    if elapsed_ns == 0:
        print("e185_residency: no samples inside the decode block", file=sys.stderr)
        return 2

    n_rounds = len(analysed)
    covered_ns = t1 - t0
    round_us = [(r.t1 - r.t0) / 1000.0 for r in analysed]

    # The in-repo trace writes the row dump and the anchor line inside the
    # round, after the commit, with no GPU work outstanding. That is instrument
    # cost sitting in an exposed window, so report the idle with and without
    # it rather than pretending the traced round is the production round.
    trace_emit_ns = sum(
        max(0, r.anchors.get("t_tail_done", 0) - r.anchors.get("t_row_trace0", 0))
        for r in analysed
    )

    idle_per_round_us = idle_ns / n_rounds / 1000.0
    idle_pct = 100.0 * idle_ns / elapsed_ns
    corrected_idle_ns = max(0, idle_ns - trace_emit_ns)

    # Alignment sensitivity: shift the whole sample axis by +/- one sample
    # period and by +/- half the wall-timestamp quantum, and report the widest
    # idle share the alignment alone can produce.
    period_ns = int(statistics.median(s.elapsed_ns for s in samples))
    sensitivity = []
    for shift in (-NS // 2, -period_ns, 0, period_ns, NS // 2):
        shifted = [
            Sample(s.end_mono_ns + shift, s.elapsed_ns, s.idle_ns, s.freq_hz)
            for s in samples
        ]
        e, d, _ = window_idle(shifted, t0, t1)
        if e:
            sensitivity.append(
                {"shift_ms": shift / 1e6, "idle_pct": 100.0 * d / e,
                 "idle_us_per_round": d / n_rounds / 1000.0}
            )

    idle_pcts = [s["idle_pct"] for s in sensitivity]

    report = {
        "tag": meta.get("tag"),
        "harness": "local",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "leg_role": meta.get("e185_leg_role"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
        "tokens": meta.get("tokens"),
        "sampler_interval_ms": meta.get("e185_residency_interval_ms"),
        "sample_period_ms_median": period_ns / 1e6,
        "clock_offset_drift_ns": offset_post - offset_pre,
        "axis_residual_span_s": axis_residual_span,
        "rounds_total": len(block),
        "rounds_skipped": args.skip_rounds,
        "rounds_analysed": n_rounds,
        "decode_block_seconds": covered_ns / NS,
        "round_us_median": statistics.median(round_us),
        "round_us_mean": statistics.fmean(round_us),
        "samples_in_window": n_samples,
        "sampled_seconds_in_window": elapsed_ns / NS,
        "window_coverage_pct": 100.0 * elapsed_ns / covered_ns,
        "gpu_idle_pct": idle_pct,
        "gpu_busy_pct": 100.0 - idle_pct,
        "gpu_idle_us_per_round": idle_per_round_us,
        "trace_emit_us_per_round": trace_emit_ns / n_rounds / 1000.0,
        "gpu_idle_us_per_round_minus_trace_emit": corrected_idle_ns / n_rounds / 1000.0,
        "alignment_sensitivity": sensitivity,
        "gpu_idle_pct_alignment_band": [min(idle_pcts), max(idle_pcts)],
        "phase_profile": phase_profile(samples, analysed),
    }

    score_path = out / "score.json"
    if score_path.exists():
        metrics = json.loads(score_path.read_text()).get("metrics", {})
        for key in (
            "mtp_seconds_per_token",
            "serial_seconds_per_token",
            "effective_mean_draft_len",
            "accepted_draft_rate",
            "all_tokens_matched",
            "decode_tokens",
        ):
            if key in metrics:
                report[key] = metrics[key]

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_path:
        Path(args.json_path).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
