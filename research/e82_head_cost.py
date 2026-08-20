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

import numpy as np

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


def read_head_provenance(tag_dir: Path) -> dict:
    """`uses_pinned_mtp_head` reads true whichever head is attached, because
    `benchmark-qwen-mtp.sh:280` provisions only the organizer head. Only
    `head_provenance_sha256` discriminates, so every leg must carry it."""
    path = tag_dir / "score.json"
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text()).get("metrics", {})
    return {
        "head_provenance_sha256": metrics.get("head_provenance_sha256"),
        "uses_pinned_mtp_head": metrics.get("uses_pinned_mtp_head"),
    }


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
        **read_head_provenance(tag_dir),
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


def precision_fit(rollup: dict) -> dict | None:
    """`ms = sum_c bytes_c / B_c`, one effective bandwidth per precision class.

    Every single-bandwidth law fails on this head set, and the controlled pair
    says why: `pinned` and `master-bf16` differ only in the readout, so their
    marginal 125,870,080 bytes over 0.147 ms imply 856 GB/s on a machine whose
    measured peak is 226 GB/s. A byte of 2-bit weight and a byte of BF16 weight
    are not the same purchase, because the 2-bit byte carries eight values to
    unpack instead of half of one. Pricing each class separately is the
    smallest model that can be true.

    Read the caveat with the numbers: three distinct byte vectors and three
    classes leave zero degrees of freedom, so this fit is a solved system, not
    a tested one. The `q2` coefficient also absorbs the shortlist stage's
    top32/gather/rerank dispatch overhead, which no byte count represents.
    """
    pts = [(a, v["stream_bytes_by_precision"], v["head_phase_ms_per_draft_median"])
           for a, v in rollup.items() if v.get("stream_bytes_by_precision")]
    classes = sorted({c for _, s, _ in pts for c in s})
    rows = {tuple(s.get(c, 0) for c in classes) for _, s, _ in pts}
    dof = len(rows) - len(classes)
    if dof < 0:
        return None
    A = np.array([[s.get(c, 0) for c in classes] for _, s, _ in pts], float)
    y = np.array([t for _, _, t in pts], float)
    ms_per_byte, *_ = np.linalg.lstsq(A, y, rcond=None)
    residuals = {}
    for (arm, s, measured), pred in zip(pts, A @ ms_per_byte):
        residuals[arm] = {
            "measured_ms": measured, "predicted_ms": float(pred),
            "residual_ms": measured - float(pred),
            "residual_frac": (measured - float(pred)) / measured,
        }
    return {
        "classes": classes,
        "distinct_byte_vectors": len(rows),
        "degrees_of_freedom": dof,
        "effective_gb_per_s": {c: (1e-6 / m if m > 0 else None)
                               for c, m in zip(classes, ms_per_byte)},
        "ms_per_byte": {c: float(m) for c, m in zip(classes, ms_per_byte)},
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
            leg["stream_bytes_by_precision"] = entry["head_stream_bytes_by_precision"]
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
            "stream_bytes_by_precision": group[0].get("stream_bytes_by_precision"),
            "artifact_bytes_per_ms": tb / ms if tb else None,
            "traffic_bytes_per_ms": tr / ms if tr else None,
            "gate_qualified_for_timing":
                sorted({g["gate_qualified_for_timing"] for g in group}),
            "head_provenance_sha256":
                sorted({g.get("head_provenance_sha256") for g in group}),
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

    pfit = precision_fit(rollup)
    if pfit:
        print("\n=== one bandwidth per precision class"
              f" (dof={pfit['degrees_of_freedom']},"
              f" max |residual| {100 * pfit['max_abs_residual_frac']:.1f}%) ===")
        for c in pfit["classes"]:
            print(f"    {c.ljust(6)}{pfit['effective_gb_per_s'][c]:8.1f} GB/s")
        for arm, r in pfit["residuals"].items():
            print(f"    {arm.ljust(13)}measured {r['measured_ms']:6.3f}"
                  f"  predicted {r['predicted_ms']:6.3f}"
                  f"  {100 * r['residual_frac']:+6.1f}%")

    Path(args.out).write_text(json.dumps(
        {"legs": legs, "by_arm": rollup, "byte_law_fits": fits,
         "precision_fit": pfit}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
