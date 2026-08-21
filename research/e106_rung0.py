#!/usr/bin/env python3
"""E106 rung 0 -- replicate the Finding 36 per-dispatch fixed-cost law.

    usage: research/e106_rung0.py TAG [TAG ...] [--width 5] [--json OUT]

Input is `research/out/TAG/census.jsonl` from
`research/e96_census_leg.sh TAG DRAFTS TOKENS 0`, which pins
`MLX_E58_BUFFER_LIMIT_OPS=0` and `MLX_E58_BUFFER_LIMIT_MB=1` so one command
buffer carries exactly one dispatch and a buffer interval is that dispatch's
exclusive GPU time.

A census leg is never a timing leg. Host wall clock is invalid because the
census swizzle serialises every dispatch; only Metal's GPU clock counts.

The byte model is the transformed affine-4 group-64 weight stream, which is
`0.5625 * K * N` bytes for one pass over one projection. It reproduces the
organiser's 14,412,349,440-byte stream exactly (checked below), and it is the
same model the Finding 36 fit used: weights only, activations reported
separately.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys

HIDDEN = 5120
VOCAB = 248_320
ATTN_LAYERS = 16
GDN_LAYERS = 48
LAYERS = ATTN_LAYERS + GDN_LAYERS
ATTN_V_DIM = 6144
GDN_V_DIM = 6144
MLP_INTERMEDIATE = 17_408
GPU_CORES = 20

# Finding 36, fitted by the advisor on the E96 round census.
LAW_F_US = 9.90
LAW_S_US_PER_GB = 3670.2

SHAPE_RE = re.compile(
    r"^(?P<kernel>\S+) grid=(?P<gx>\d+)x(?P<gy>\d+)x(?P<gz>\d+)"
    r" tg=(?P<tx>\d+)x(?P<ty>\d+)x(?P<tz>\d+)$")


def affine4(k: int, n: int) -> int:
    """affine 4-bit group-64: 4 bits per weight plus a bf16 scale and bias."""
    return n * k // 2 + 4 * (n * k // 64)


# out width -> (family label, [K of every tensor in the family], tensors/pass)
STREAM_CLASSES = {
    14_336: ("fa.qkv", [HIDDEN] * ATTN_LAYERS),
    16_480: ("gdn.in_proj", [HIDDEN] * GDN_LAYERS),
    34_816: ("mlp.gate_up", [HIDDEN] * LAYERS),
    5_120: ("gdn.out_proj + fa.o_proj + mlp.down",
            [GDN_V_DIM] * GDN_LAYERS + [ATTN_V_DIM] * ATTN_LAYERS
            + [MLP_INTERMEDIATE] * LAYERS),
    VOCAB: ("lm_head", [HIDDEN]),
}
CLEAN_FAMILIES = ("lm_head", "mlp.gate_up", "gdn.in_proj", "fa.qkv")

WEIGHT_STREAM_BYTES = sum(
    sum(affine4(k, n) for k in ks) for n, (_label, ks) in STREAM_CLASSES.items())


def mean_bytes_per_dispatch(out_width: int) -> float:
    _label, ks = STREAM_CLASSES[out_width]
    return sum(affine4(k, out_width) for k in ks) / len(ks)


def load(tag: str):
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e106_rung0: no census at {path}")
    rounds = collections.Counter()
    kernels: dict[str, dict] = {}
    phases: dict[str, dict] = {}
    shapes_seen: dict[int, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") == "round":
            width = rec["width"]
            for phase, bucket in rec.get("phases", {}).items():
                for shape, count in bucket.get("shapes", {}).items():
                    shapes_seen[width][f"{phase}|{shape}"].append(count)
            continue
        if rec.get("event") != "gputime":
            continue
        for width in {k.split("|", 1)[0] for k in rec.get("by_width_phase", {})}:
            rounds[width] += rec.get("rounds", 1)
        for key, v in rec.get("exclusive_kernels", {}).items():
            e = kernels.setdefault(key, {"buffers": 0, "gpu_ns": 0.0,
                                         "min_ns": float("inf"), "max_ns": 0.0,
                                         "sum_sq_ns": 0.0})
            e["buffers"] += v["buffers"]
            e["gpu_ns"] += v["gpu_ns"]
            e["min_ns"] = min(e["min_ns"], v["min_ns"])
            e["max_ns"] = max(e["max_ns"], v["max_ns"])
            e["sum_sq_ns"] += v.get("sum_sq_ns", 0.0)
        for key, v in rec.get("by_width_phase", {}).items():
            e = phases.setdefault(key, {"dispatches": 0, "gpu_ns": 0.0,
                                        "buffers": 0})
            e["dispatches"] += v.get("dispatches", 0)
            e["buffers"] += v.get("buffers", 0)
            e["gpu_ns"] += v.get("gpu_ns", 0)
    return dict(rounds), kernels, phases, shapes_seen


def fit(points):
    """Least squares of us = F + GB * S over (GB, us) pairs."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((p[1] - mean_y) ** 2 for p in points)
    ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return intercept, slope, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--phase", default="target_verify")
    ap.add_argument("--json")
    args = ap.parse_args()

    assert WEIGHT_STREAM_BYTES == 14_412_349_440, WEIGHT_STREAM_BYTES
    payload: dict[str, object] = {}

    for tag in args.tags:
        rounds, kernels, phases, shapes_seen = load(tag)
        wkey = f"w{args.width}"
        n_rounds = rounds.get(wkey, 0)
        print(f"=== {tag}   width M={args.width}   rounds={n_rounds} ===")
        print("rounds by width:", dict(sorted(rounds.items())))
        if not n_rounds:
            continue

        print("\nGPU busy per round, by phase (census GPU clock):")
        round_us = 0.0
        round_dispatches = 0.0
        for key in sorted(phases):
            if not key.startswith(wkey + "|"):
                continue
            us = phases[key]["gpu_ns"] / 1e3 / n_rounds
            disp = phases[key]["dispatches"] / n_rounds
            print(f"  {key:<28} {us:12.1f} us/round  {disp:9.2f} dispatches/round")
            if key.endswith("|outside"):
                continue
            round_us += us
            round_dispatches += disp
        print(f"  ROUND TOTAL (excl. outside)   {round_us:12.1f} us/round  "
              f"{round_dispatches:9.2f} dispatches/round")

        # --- per-shape table -------------------------------------------------
        rows = []
        for key, v in kernels.items():
            width_key, phase, shape = key.split("|", 2)
            if width_key != wkey:
                continue
            match = SHAPE_RE.match(shape)
            if not match:
                continue
            gx, gy, gz = (int(match.group(g)) for g in ("gx", "gy", "gz"))
            tx, ty, tz = (int(match.group(g)) for g in ("tx", "ty", "tz"))
            per_round = v["buffers"] / n_rounds
            us_disp = v["gpu_ns"] / 1e3 / v["buffers"]
            us_round = v["gpu_ns"] / 1e3 / n_rounds
            threadgroups = gx * gy * gz
            out_width = gy * 8
            is_qmv = match.group("kernel").startswith("affine_qmv_fast")
            family = None
            gb_disp = 0.0
            if is_qmv and out_width in STREAM_CLASSES:
                family = STREAM_CLASSES[out_width][0]
                gb_disp = mean_bytes_per_dispatch(out_width) / 1e9
            law_us = LAW_F_US + gb_disp * LAW_S_US_PER_GB if gb_disp else 0.0
            rows.append({
                "phase": phase, "kernel": match.group("kernel"),
                "grid": (gx, gy, gz), "tg": (tx, ty, tz),
                "dispatches": v["buffers"], "per_round": per_round,
                "us_per_dispatch": us_disp, "us_per_round": us_round,
                "min_us": v["min_ns"] / 1e3, "max_us": v["max_ns"] / 1e3,
                "threadgroups": threadgroups,
                "threads_per_tg": tx * ty * tz,
                "waves_20_cores": threadgroups / GPU_CORES,
                "family": family, "gb_per_dispatch": gb_disp,
                "law_us": law_us,
                "resid_us": us_disp - law_us if law_us else 0.0,
            })
        rows.sort(key=lambda r: -r["us_per_round"])

        print(f"\nExclusive per-dispatch GPU cost, M={args.width}, "
              f"one dispatch per command buffer:")
        head = (f"  {'us/rnd':>9} {'n/rnd':>6} {'us/disp':>8} {'min':>7} "
                f"{'max':>7} {'TGs':>7} {'TG/core':>7} {'GB/disp':>8} "
                f"{'law us':>8} {'resid':>8}  kernel")
        print(head)
        for r in rows:
            if r["us_per_round"] < 1.0:
                continue
            label = (f"{r['kernel']} grid={r['grid'][0]}x{r['grid'][1]}x"
                     f"{r['grid'][2]} tg={r['tg'][0]}x{r['tg'][1]}x{r['tg'][2]}"
                     f" [{r['phase']}]")
            print(f"  {r['us_per_round']:9.1f} {r['per_round']:6.2f} "
                  f"{r['us_per_dispatch']:8.2f} {r['min_us']:7.2f} "
                  f"{r['max_us']:7.2f} {r['threadgroups']:7d} "
                  f"{r['waves_20_cores']:7.1f} {r['gb_per_dispatch']:8.5f} "
                  f"{r['law_us']:8.2f} {r['resid_us']:8.2f}  {label[:96]}")

        # --- streaming families ----------------------------------------------
        fam = collections.defaultdict(
            lambda: {"dispatches": 0, "gpu_ns": 0.0, "gb": 0.0, "grids": set(),
                     "tgs": set(), "threadgroups": set(), "phases": set()})
        for r in rows:
            # The target pass is the family measurement. The draft head issues
            # the same kernel on the same weights with grid.x = 1, so pooling
            # the two would mix a one-row dispatch into a five-row family.
            if not r["family"] or r["phase"] != args.phase:
                continue
            if r["grid"][0] != args.width:
                continue
            f = fam[r["family"]]
            f["dispatches"] += r["dispatches"]
            f["gpu_ns"] += r["us_per_dispatch"] * r["dispatches"] * 1e3
            f["gb"] += r["gb_per_dispatch"] * r["dispatches"]
            f["grids"].add(r["grid"])
            f["tgs"].add(r["tg"])
            f["threadgroups"].add(r["threadgroups"])
            f["phases"].add(r["phase"])

        print("\nStreaming families:")
        print(f"  {'family':<36} {'disp/rnd':>8} {'GB/disp':>9} {'us/disp':>9} "
              f"{'GB/s':>8} {'TGs':>7}")
        fitted = []
        table = {}
        for name, f in sorted(fam.items(), key=lambda kv: -kv[1]["gpu_ns"]):
            disp_round = f["dispatches"] / n_rounds
            gb_disp = f["gb"] / f["dispatches"]
            us_disp = f["gpu_ns"] / 1e3 / f["dispatches"]
            rate = gb_disp / (us_disp * 1e-6) if us_disp else 0.0
            tgs = sorted(f["threadgroups"])
            print(f"  {name:<36} {disp_round:8.1f} {gb_disp:9.5f} "
                  f"{us_disp:9.2f} {rate:8.1f} {str(tgs):>7}")
            table[name] = {
                "dispatches_per_round": disp_round, "gb_per_dispatch": gb_disp,
                "us_per_dispatch": us_disp, "gb_per_s": rate,
                "us_per_round": f["gpu_ns"] / 1e3 / n_rounds,
                "threadgroups": tgs, "grids": sorted(map(list, f["grids"])),
                "threadgroup_dims": sorted(map(list, f["tgs"])),
                "phases": sorted(f["phases"]),
            }
            if name in CLEAN_FAMILIES:
                fitted.append((gb_disp, us_disp))

        if len(fitted) >= 2:
            f_us, s_us, r2 = fit(fitted)
            print(f"\n  refit on {len(fitted)} clean families:  "
                  f"F = {f_us:.2f} us/dispatch   S = {s_us:.1f} us/GB "
                  f"= {1e6 / s_us:.1f} GB/s   R^2 = {r2:.8f}")
            print(f"  shipped Finding 36 law:      F = {LAW_F_US:.2f} "
                  f"us/dispatch   S = {LAW_S_US_PER_GB:.1f} us/GB "
                  f"= {1e6 / LAW_S_US_PER_GB:.1f} GB/s")
            print(f"\n  {'family':<36} {'us/disp':>9} {'refit':>9} "
                  f"{'resid':>8} {'resid %':>8} {'peak% after F':>14}")
            for name, entry in table.items():
                pred = f_us + entry["gb_per_dispatch"] * s_us
                resid = entry["us_per_dispatch"] - pred
                after_f = (entry["gb_per_dispatch"]
                           / ((entry["us_per_dispatch"] - f_us) * 1e-6)
                           if entry["us_per_dispatch"] > f_us else 0.0)
                entry["refit_us"] = pred
                entry["resid_us"] = resid
                entry["resid_pct"] = 100.0 * resid / entry["us_per_dispatch"]
                entry["gb_per_s_after_f"] = after_f
                entry["excess_us_per_round"] = resid * entry[
                    "dispatches_per_round"]
                entry["excess_pct_of_round"] = (
                    100.0 * resid * entry["dispatches_per_round"] / round_us
                    if round_us else 0.0)
                print(f"  {name:<36} {entry['us_per_dispatch']:9.2f} "
                      f"{pred:9.2f} {resid:8.2f} {entry['resid_pct']:8.3f} "
                      f"{after_f / (273e9 / 1e9) * 100:13.1f}%")
            anomaly = table.get("gdn.out_proj + fa.o_proj + mlp.down")
            if anomaly:
                print(f"\n  N=5120 excess: {anomaly['resid_us']:.2f} us per "
                      f"dispatch, {anomaly['excess_us_per_round']:.1f} us per "
                      f"round, {anomaly['excess_pct_of_round']:.3f} % of the "
                      f"{round_us:.0f} us census round")
            payload[tag] = {
                "rounds": n_rounds, "width": args.width,
                "round_us": round_us, "round_dispatches": round_dispatches,
                "fit": {"F_us": f_us, "S_us_per_gb": s_us, "r2": r2},
                "families": table,
                "shapes": [
                    {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in r.items()} for r in rows],
            }
        print()

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
