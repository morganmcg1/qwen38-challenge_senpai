#!/usr/bin/env python3
"""E165: read the GPU's own residency counters for a leg.

usage:
  research/e165_residency.py research/out/TAG [research/out/TAG2 ...] [--json OUT]

WHY. The host-clock census bounds the GPU-idle fraction but cannot measure it.
A bubble that opens after a command buffer is committed and closes before it
completes is invisible to every clock the session can read, so the census gives
`idle >= host-visible gap` and `busy <= gpu_covered`. `powermetrics` reads the
GPU's residency counters, so it closes that gap from the other side.

WHAT IS REPORTED. Each sample carries an interval, an active residency, an idle
residency and the frequency histogram the active part was spread over. This
splits the advisor's question into its two measurable halves:

  IDLE  = mean idle residency over the decoding window
  SLOW  = where the active residency sits in the frequency histogram

A GPU that is busy 99% of the wall clock at its top frequency and still misses
the stream floor is SLOW. A GPU that is idle 40% of the wall clock is IDLE. The
two numbers are independent and both are measured here.

WINDOW. The sampler runs across model load, warmup and both timed legs, so the
per-sample idle residency is strongly bimodal. The decoding window is taken as
the longest contiguous run of samples below `--busy-threshold` idle residency,
and the report also prints the whole-run figures so the choice of window is
visible rather than hidden.

CONTROLS. `powermetrics-idle-pre.txt` and `powermetrics-idle-post.txt` sample an
idle GPU. They prove the instrument responds, and they bound how much of the
leg's activity belongs to other processes. A leg whose idle baseline is not
quiet is refused, not reported.

harness=local. Nothing here is an official or ranked score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SAMPLE_RE = re.compile(r"\*\*\* Sampled system activity .*?\(([\d.]+)ms elapsed\)")
ACTIVE_FREQ_RE = re.compile(r"^GPU HW active frequency:\s+([\d.]+)\s*MHz")
ACTIVE_RES_RE = re.compile(r"^GPU HW active residency:\s+([\d.]+)%\s*\((.*)\)")
IDLE_RES_RE = re.compile(r"^GPU idle residency:\s+([\d.]+)%")
POWER_RE = re.compile(r"^GPU Power:\s+([\d.]+)\s*mW")
BIN_RE = re.compile(r"([\d.]+)\s*MHz:\s*([\d.]+)%")

# The idle baseline has to be at least this quiet for the leg to be reportable.
IDLE_BASELINE_MIN_PCT = 85.0


def parse_samples(path: Path) -> list[dict]:
    """Split a powermetrics stream into one record per sample."""
    out: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(errors="replace").splitlines():
        m = SAMPLE_RE.search(line)
        if m:
            if cur is not None:
                out.append(cur)
            cur = {"elapsed_ms": float(m.group(1))}
            continue
        if cur is None:
            continue
        m = ACTIVE_FREQ_RE.match(line)
        if m:
            cur["active_freq_mhz"] = float(m.group(1))
            continue
        m = ACTIVE_RES_RE.match(line)
        if m:
            cur["active_pct"] = float(m.group(1))
            cur["freq_bins"] = [
                (float(f), float(p)) for f, p in BIN_RE.findall(m.group(2))
            ]
            continue
        m = IDLE_RES_RE.match(line)
        if m:
            cur["idle_pct"] = float(m.group(1))
            continue
        m = POWER_RE.match(line)
        if m:
            cur["power_mw"] = float(m.group(1))
    if cur is not None:
        out.append(cur)
    return [s for s in out if "idle_pct" in s and "elapsed_ms" in s]


def weighted(samples: list[dict], key: str) -> float:
    """Time-weighted mean, because powermetrics intervals are not uniform."""
    total = sum(s["elapsed_ms"] for s in samples)
    if total <= 0:
        return float("nan")
    return sum(s[key] * s["elapsed_ms"] for s in samples) / total


def busy_runs(samples: list[dict], threshold: float) -> list[tuple[int, int]]:
    """Every contiguous run of decoding samples, in order."""
    runs = []
    start = None
    for i, s in enumerate(samples):
        if s["idle_pct"] < threshold:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(samples)))
    return runs


def freq_histogram(samples: list[dict]) -> list[tuple[float, float]]:
    """Fraction of ACTIVE GPU time spent in each frequency bin."""
    acc: dict[float, float] = {}
    for s in samples:
        for f, pct in s.get("freq_bins", []):
            acc[f] = acc.get(f, 0.0) + pct * s["elapsed_ms"]
    total = sum(acc.values())
    if total <= 0:
        return []
    return sorted((f, v / total * 100.0) for f, v in acc.items() if v > 0)


def read_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def analyse(tag_dir: Path, threshold: float) -> dict:
    samples = parse_samples(tag_dir / "powermetrics.txt")
    if not samples:
        raise SystemExit(f"{tag_dir}: no powermetrics samples")

    control = {}
    for name, fname in (("pre", "powermetrics-idle-pre.txt"),
                        ("post", "powermetrics-idle-post.txt")):
        base = parse_samples(tag_dir / fname)
        control[name] = weighted(base, "idle_pct") if base else float("nan")

    runs = busy_runs(samples, threshold)
    if not runs:
        raise SystemExit(f"{tag_dir}: no busy window below {threshold}% idle")
    segments = []
    for lo, hi in runs:
        seg = samples[lo:hi]
        seconds = sum(s["elapsed_ms"] for s in seg) / 1000.0
        if seconds < 1.0:
            continue
        segments.append({
            "index": len(segments),
            "seconds": seconds,
            "samples": len(seg),
            "idle_pct": weighted(seg, "idle_pct"),
            "active_freq_mhz": weighted(seg, "active_freq_mhz"),
            "power_mw": weighted(seg, "power_mw"),
        })
    lo, hi = max(runs, key=lambda r: sum(s["elapsed_ms"] for s in samples[r[0]:r[1]]))
    window = samples[lo:hi]

    meta = read_meta(tag_dir / "meta.txt")
    score_path = tag_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}

    window_ms = sum(s["elapsed_ms"] for s in window)
    idle_pct = weighted(window, "idle_pct")
    hist = freq_histogram(window)
    top_freq = max((f for f, _ in hist), default=float("nan"))
    at_top = sum(p for f, p in hist if f == top_freq)

    return {
        "busy_segments": segments,
        "tag": tag_dir.name,
        "harness": "local",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "pinned_depth": meta.get("e165_pinned_depth"),
        "worker_sha256": meta.get("worker_sha256"),
        "base_sha": meta.get("base_sha"),
        "samples_total": len(samples),
        "samples_window": len(window),
        "window_seconds": window_ms / 1000.0,
        "busy_threshold_pct": threshold,
        "idle_baseline_pre_pct": control["pre"],
        "idle_baseline_post_pct": control["post"],
        "whole_run_idle_pct": weighted(samples, "idle_pct"),
        # THE MEASURED ANSWER. Not a bound.
        "window_idle_pct": idle_pct,
        "window_busy_pct": 100.0 - idle_pct,
        "window_active_freq_mhz": weighted(window, "active_freq_mhz"),
        "window_power_mw": weighted(window, "power_mw"),
        "window_top_freq_mhz": top_freq,
        "window_pct_active_time_at_top_freq": at_top,
        "freq_histogram_pct_of_active_time": hist,
        "mtp_seconds_per_token": score.get("mtp_seconds_per_token"),
        "serial_seconds_per_token": score.get("serial_seconds_per_token"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--busy-threshold", type=float, default=50.0,
                    help="idle residency %% below which a sample is 'decoding'")
    ap.add_argument("--json")
    args = ap.parse_args()

    results = [analyse(Path(d), args.busy_threshold) for d in args.dirs]

    for r in results:
        print(f"=== {r['tag']}  pinned_d={r['pinned_depth']}  harness=local ===")
        pre, post = r["idle_baseline_pre_pct"], r["idle_baseline_post_pct"]
        quiet = pre >= IDLE_BASELINE_MIN_PCT and post >= IDLE_BASELINE_MIN_PCT
        print(f"  idle baseline        pre {pre:6.2f}%   post {post:6.2f}% "
              f"  [{'ok' if quiet else 'REFUSED: GPU was not quiet'}]")
        print(f"  decoding window      {r['window_seconds']:.2f} s over "
              f"{r['samples_window']} samples of {r['samples_total']}")
        print(f"  GPU IDLE  (measured) {r['window_idle_pct']:7.2f}%")
        print(f"  GPU BUSY  (measured) {r['window_busy_pct']:7.2f}%")
        print(f"  active frequency     {r['window_active_freq_mhz']:7.1f} MHz "
              f"(top bin {r['window_top_freq_mhz']:.0f} MHz, "
              f"{r['window_pct_active_time_at_top_freq']:.1f}% of active time)")
        print(f"  GPU power            {r['window_power_mw']:7.0f} mW")
        print(f"  mtp_seconds_per_token {r['mtp_seconds_per_token']}")
        print("  busy segments (the serial and MTP legs appear separately):")
        for s in r["busy_segments"]:
            print(f"    [{s['index']}] {s['seconds']:7.2f} s  "
                  f"idle {s['idle_pct']:6.2f}%  "
                  f"{s['active_freq_mhz']:6.0f} MHz  {s['power_mw']:6.0f} mW")
        print("  frequency histogram (% of active GPU time):")
        for f, p in r["freq_histogram_pct_of_active_time"]:
            if p >= 0.05:
                print(f"    {f:7.0f} MHz  {p:6.2f}%  {'#' * int(p / 2)}")
        if not quiet:
            print("  VERDICT: not reportable, the idle baseline is contaminated")
        elif r["window_idle_pct"] >= 15.0:
            print("  VERDICT: IDLE is a first-order term in this round")
        else:
            print("  VERDICT: the GPU is busy; the round is SLOW, not IDLE")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
