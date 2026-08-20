#!/usr/bin/env python3
"""E82: measured proposal-head step cost per head, from phase traces.

The advisor made head cost a first-class outcome: on the current frontier
1 % of candidate time is worth 1.000 % of official score. This reads the
`mtp-trace: round=` lines a `--sync-head` leg writes and reports the head
step in ms per draft, so a head can be priced next to its byte split.

`--sync-head` drains the head chain before the verify window, so the chain's
GPU time lands in `draft_build_us` instead of hiding in the trailing
`asyncEval`. Within one round the head path is:

    d_flush_us   host build of the history flush graph
    d_head1_us   host build of head step 1 (flush + first readout)
    d_submit1_us asyncEval of step 1
    d_chain_us   host build of head steps 2..d
    d_submit2_us submit and, under --sync-head, WAIT for the whole chain

so `d_submit2_us` is the head chain's GPU time that the host did not already
overlap, and `draft_build_us` is the whole head phase wall time. Both are
reported: the first is the marginal GPU cost, the second is what the round
actually pays before verify work can start.

  python3 research/e82_head_cost.py research/out/TAG [...] --out FILE
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROUND = re.compile(r"^mtp-trace: round=(\d+) d=(\d+) acc=(\d+) (.*)$")
FIELD = re.compile(r"(\w+)=(-?[\d.]+)")
PHASES = ("draft_build_us", "d_pre_us", "d_flush_us", "d_head1_us",
          "d_submit1_us", "d_chain_us", "d_submit2_us", "verify_build_us",
          "eval_wall_us", "round_us")


def read_meta(tag_dir: Path) -> dict:
    meta = {}
    for line in (tag_dir / "meta.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    return meta


def read_rounds(tag_dir: Path) -> list[dict]:
    rounds = []
    for line in (tag_dir / "trace.txt").read_text().splitlines():
        m = ROUND.match(line)
        if not m:
            continue
        rec = {"round": int(m.group(1)), "d": int(m.group(2)), "acc": int(m.group(3))}
        rec.update({k: float(v) for k, v in FIELD.findall(m.group(4))
                    if k in PHASES})
        rounds.append(rec)
    return rounds


def summarize(tag_dir: Path, warmup: int) -> dict:
    meta = read_meta(tag_dir)
    rounds = [r for r in read_rounds(tag_dir) if r["round"] > warmup and r["d"] > 0]
    if not rounds:
        raise SystemExit(f"{tag_dir}: no drafting rounds after warmup {warmup}")

    # Per-draft normalisation is what makes heads comparable: a better head
    # drafts deeper, so a per-round total would confound cost with schedule.
    per_draft = {
        "head_phase_ms_per_draft": [r["draft_build_us"] / 1000 / r["d"] for r in rounds],
        "chain_gpu_ms_per_draft": [r["d_submit2_us"] / 1000 / r["d"] for r in rounds],
        "host_build_ms_per_draft": [
            (r["d_flush_us"] + r["d_head1_us"] + r["d_chain_us"]) / 1000 / r["d"]
            for r in rounds],
    }
    out = {
        "tag": meta.get("tag", tag_dir.name),
        "head_dir": meta.get("head_dir"),
        "tokens": meta.get("tokens"),
        "sync_head": meta.get("sync_head"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "official_or_ranked_score": meta.get("official_or_ranked_score"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "base_sha": meta.get("base_sha"),
        "chip": meta.get("chip"),
        "drafting_rounds": len(rounds),
        "mean_draft_len": st.mean(r["d"] for r in rounds),
        "mean_accept_per_round": st.mean(r["acc"] for r in rounds),
        "median_round_ms": st.median(r["round_us"] / 1000 for r in rounds),
    }
    for name, xs in per_draft.items():
        out[name] = {"median": st.median(xs), "mean": st.mean(xs), "n": len(xs)}
    for p in PHASES:
        out[f"median_{p}"] = st.median(r[p] for r in rounds)
    return out


def affine_fit(rollup: dict, byte_key: str) -> dict | None:
    """Least-squares `ms = intercept + bytes / bandwidth` over the arm medians.

    Two byte accountings compete. `tensor_bytes` counts only what the head
    artifact ships. `traffic_bytes` adds the compact draft head that the
    runtime derives when the artifact ships no `draft_lm_head`. The fit that
    leaves the smaller residual is the one that describes the hardware.
    """
    pts = [(a, v[byte_key], v["head_phase_ms_per_draft_median"])
           for a, v in rollup.items() if v.get(byte_key)]
    # Distinct byte counts, not distinct arms: identical builds share a point.
    if len({p[1] for p in pts}) < 2:
        return None
    mx = st.mean(p[1] for p in pts)
    my = st.mean(p[2] for p in pts)
    sxx = sum((p[1] - mx) ** 2 for p in pts)
    slope = sum((p[1] - mx) * (p[2] - my) for p in pts) / sxx
    intercept = my - slope * mx
    residuals = {}
    for arm, x, y in pts:
        pred = intercept + slope * x
        residuals[arm] = {
            "bytes": x, "measured_ms": y, "predicted_ms": pred,
            "residual_ms": y - pred, "residual_frac": (y - pred) / y,
        }
    return {
        "byte_key": byte_key,
        "intercept_ms": intercept,
        "ms_per_byte": slope,
        "effective_gb_per_s": 1e-6 / slope if slope > 0 else None,
        "residuals": residuals,
        "max_abs_residual_frac":
            max(abs(r["residual_frac"]) for r in residuals.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--bytes-report", default="research/e82-head-bytes.json")
    ap.add_argument("--out", default="research/e82-head-cost.json")
    args = ap.parse_args()

    legs = [summarize(Path(t), args.warmup) for t in args.tags]
    byte_report = json.loads(Path(args.bytes_report).read_text())
    by_dir = {e["path"]: (arm, e) for arm, e in byte_report["arms"].items()}

    print("leg".ljust(34) + "arm".ljust(13)
          + "head ms/draft  chain GPU  host build   d̄   temps C   gated")
    for leg in legs:
        arm, entry = by_dir.get(leg["head_dir"], ("?", None))
        leg["arm"] = arm
        if entry:
            leg["tree_bytes"] = entry["tree_bytes"]
            leg["tensor_bytes"] = entry["tensor_bytes"]
            leg["traffic_bytes"] = entry["traffic_bytes_per_draft"]
            leg["bytes_per_ms_head_step"] = (
                entry["tensor_bytes"] / leg["head_phase_ms_per_draft"]["median"])
        t0 = float(leg["gpu_temp_entry_c"] or "nan")
        t1 = float(leg["gpu_temp_exit_c"] or "nan")
        print(
            f"{leg['tag'][:33].ljust(34)}{arm.ljust(13)}"
            f"{leg['head_phase_ms_per_draft']['median']:12.3f}"
            f"{leg['chain_gpu_ms_per_draft']['median']:11.3f}"
            f"{leg['host_build_ms_per_draft']['median']:11.3f}"
            f"{leg['mean_draft_len']:6.2f}"
            f"  {t0:.1f}->{t1:.1f}"
            f"  {leg['gate_qualified_for_timing']}")

    by_arm: dict[str, list[dict]] = {}
    for leg in legs:
        by_arm.setdefault(leg["arm"], []).append(leg)
    print("\n=== per arm (median across legs) ===")
    print("arm".ljust(13) + "legs  head ms/draft   artifact B/ms    traffic B/ms")
    rollup = {}
    for arm, group in by_arm.items():
        ms = st.median(g["head_phase_ms_per_draft"]["median"] for g in group)
        tb = group[0].get("tensor_bytes")
        tr = group[0].get("traffic_bytes")
        rollup[arm] = {
            "legs": [g["tag"] for g in group],
            "head_phase_ms_per_draft_median": ms,
            "head_phase_ms_per_draft_legs":
                [g["head_phase_ms_per_draft"]["median"] for g in group],
            "chain_gpu_ms_per_draft_median":
                st.median(g["chain_gpu_ms_per_draft"]["median"] for g in group),
            "tensor_bytes": tb,
            "traffic_bytes": tr,
            "artifact_bytes_per_ms": tb / ms if tb else None,
            "traffic_bytes_per_ms": tr / ms if tr else None,
            "gate_qualified_for_timing":
                sorted({g["gate_qualified_for_timing"] for g in group}),
        }
        print(f"{arm.ljust(13)}{len(group):5d}{ms:15.3f}"
              + (f"{tb / ms:16,.0f}{tr / ms:16,.0f}" if tb
                 else f"{'unknown head dir':>32}"))

    fits = {key: affine_fit(rollup, key) for key in
            ("tensor_bytes", "traffic_bytes")}
    print("\n=== which byte accounting predicts the measured head step? ===")
    for key, fit in fits.items():
        if not fit:
            continue
        print(f"{key}: {fit['effective_gb_per_s']:.1f} GB/s"
              f"  fixed {fit['intercept_ms']:.3f} ms"
              f"  max |residual| {100 * fit['max_abs_residual_frac']:.1f}%")
        for arm, r in fit["residuals"].items():
            print(f"    {arm.ljust(13)}measured {r['measured_ms']:6.3f}"
                  f"  predicted {r['predicted_ms']:6.3f}"
                  f"  {100 * r['residual_frac']:+6.1f}%")

    Path(args.out).write_text(json.dumps(
        {"legs": legs, "by_arm": rollup, "byte_law_fits": fits}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
