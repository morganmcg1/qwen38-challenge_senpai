#!/usr/bin/env python3
"""Summarize the E44 section 7.3 paired A/B microbenchmark and log it to W&B.

Every claim gets an interval. The pre-registered detection threshold for this
experiment is MDE(exact, df=4) = 0.5040 %; the achieved interval is computed from
the observed pairwise spread and reported next to it, so a null result can be
distinguished from an underpowered one.

    research/e44_ab_summary.py .mlxfast-private/e44-qmv-ab/TAG [--wandb]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

# Pre-registered in the assignment. Not recomputed from the data.
PREREG_MDE_PCT = 0.5040
# The assignment's promotion bar for this mechanism.
BAR_PCT = 5.0
# Widths the candidate actually replaces; 1..3 are the untouched-width guard.
TOUCHED = range(4, 10)

# Student t, df = 4. Two-sided 95 % critical value, and the one-sided 80 %-power
# companion. Hardcoded so the summary has no scipy dependency.
T_975_DF4 = 2.7764451051977987
T_800_DF4 = 0.9409645
MDE_FACTOR_DF4 = (T_975_DF4 + T_800_DF4) / math.sqrt(5)


def load(run_dir: pathlib.Path) -> tuple[dict, dict]:
    payload = json.loads((run_dir / "ab.json").read_text())
    identity: dict[str, str] = {}
    for line in (run_dir / "identity.txt").read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            identity[key.strip()] = value.strip()
    return payload, identity


def paired_stats(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else float("nan")
    half = T_975_DF4 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n_pairs": n,
        "mean_pct": mean,
        "sd_pct": sd,
        "ci95_lo_pct": mean - half,
        "ci95_hi_pct": mean + half,
        "achieved_mde_pct": MDE_FACTOR_DF4 * sd if n > 1 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    payload, identity = load(args.run_dir)
    rows = payload["measurements"]

    fidelity = [r for r in rows if r["kind"] == "fidelity"]
    timing = [r for r in rows if r["kind"] == "timing"]

    shapes = sorted({r["shape"] for r in timing})
    widths = sorted({r["m"] for r in timing})

    print(f"device        : {payload['device']}  ({payload['architecture']})")
    print(f"function      : {payload['function']}")
    print(f"design        : {payload['order']}, pairs={payload['pairs']}, "
          f"reps={payload['reps']}, inner={payload['inner']}")
    print(f"base arm      : {identity.get('base_metal_sha256', '?')[:16]} "
          f"@ {identity.get('base_sha', '?')[:8]}")
    print(f"cand arm      : {identity.get('cand_metal_sha256', '?')[:16]} "
          f"@ {identity.get('head', '?')[:8]}")
    print(f"thermal       : real_gate={identity.get('cool_gate_passed_real_gate')} "
          f"gate_qualified={identity.get('gate_qualified_for_timing')} "
          f"entry={identity.get('gpu_temp_c_entry')}C "
          f"exit={identity.get('gpu_temp_c_exit')}C")
    print(f"pre-registered MDE(exact, df=4) = {PREREG_MDE_PCT:.4f} %   "
          f"bar = {BAR_PCT:.1f} %")

    print("\n--- fidelity vs exact double reference (before any timing) ---")
    print(f"{'shape':<26}{'M':>3}{'base max_rel':>15}{'cand max_rel':>15}"
          f"{'cand rms_rel':>15}")
    worst_cand = 0.0
    for r in fidelity:
        worst_cand = max(worst_cand, r["cand_max_rel"])
        print(f"{r['shape']:<26}{r['m']:>3}{r['base_max_rel']:>15.3e}"
              f"{r['cand_max_rel']:>15.3e}{r['cand_rms_rel']:>15.3e}")
    print(f"worst candidate max_rel = {worst_cand:.3e}")

    summary: dict[str, float] = {}
    table: list[dict] = []
    print("\n--- paired timing, candidate minus base ---")
    for shape in shapes:
        print(f"\n{shape}")
        print(f"{'M':>3}{'base us':>11}{'cand us':>11}{'delta %':>10}"
              f"{'sd %':>8}{'95% CI':>20}{'MDE %':>9}  verdict")
        for m in widths:
            pairs = [r for r in timing if r["shape"] == shape and r["m"] == m]
            if not pairs:
                continue
            base_us = statistics.fmean(p["base_s"] for p in pairs) * 1e6
            cand_us = statistics.fmean(p["cand_s"] for p in pairs) * 1e6
            deltas = [100.0 * (p["cand_s"] - p["base_s"]) / p["base_s"]
                      for p in pairs]
            st = paired_stats(deltas)
            speedup = -st["mean_pct"]  # positive means the candidate is faster
            if st["ci95_hi_pct"] < 0.0:
                verdict = "faster"
            elif st["ci95_lo_pct"] > 0.0:
                verdict = "slower"
            else:
                verdict = "null"
            if m in TOUCHED and speedup >= BAR_PCT and st["ci95_hi_pct"] < 0.0:
                verdict += " CLEARS BAR"
            tag = "" if m in TOUCHED else " (guard)"
            print(f"{m:>3}{base_us:>11.2f}{cand_us:>11.2f}"
                  f"{st['mean_pct']:>+10.3f}{st['sd_pct']:>8.3f}"
                  f"  [{st['ci95_lo_pct']:+7.3f},{st['ci95_hi_pct']:+7.3f}]"
                  f"{st['achieved_mde_pct']:>9.3f}  {verdict}{tag}")
            row = {"shape": shape, "m": m, "base_us": base_us,
                   "cand_us": cand_us, "speedup_pct": speedup, **st}
            table.append(row)
            key = f"{shape}/M{m}"
            summary[f"{key}/speedup_pct"] = speedup
            summary[f"{key}/ci95_lo_pct"] = -st["ci95_hi_pct"]
            summary[f"{key}/ci95_hi_pct"] = -st["ci95_lo_pct"]
            summary[f"{key}/base_us"] = base_us
            summary[f"{key}/cand_us"] = cand_us

    touched = [r for r in table if r["m"] in TOUCHED]
    guard = [r for r in table if r["m"] not in TOUCHED]
    best = max(touched, key=lambda r: r["speedup_pct"], default=None)
    print("\n--- decision ---")
    if touched:
        mean_touched = statistics.fmean(r["speedup_pct"] for r in touched)
        print(f"mean speedup over replaced widths M in [4, 9]: "
              f"{mean_touched:+.3f} %")
        summary["mean_speedup_touched_pct"] = mean_touched
    if guard:
        mean_guard = statistics.fmean(r["speedup_pct"] for r in guard)
        # M in 1..3 run byte-identical code in both arms, so the true effect
        # there is exactly zero. Their spread is therefore a direct measurement
        # of the harness noise floor, which is worth more than any assumed sd:
        # no effect smaller than this is believable no matter what the paired
        # interval says.
        floor = (statistics.stdev([r["speedup_pct"] for r in guard])
                 if len(guard) > 1 else float("nan"))
        print(f"mean effect on untouched-width guard M in [1, 3]: "
              f"{mean_guard:+.3f} % (expected exactly 0; identical code in "
              f"both arms)")
        print(f"empirical noise floor from the guard: sd={floor:.3f} % over "
              f"{len(guard)} zero-effect measurement(s), "
              f"worst |effect|="
              f"{max(abs(r['speedup_pct']) for r in guard):.3f} %")
        summary["mean_effect_guard_pct"] = mean_guard
        summary["guard_noise_floor_sd_pct"] = floor
    if best:
        print(f"best replaced width: {best['shape']} M={best['m']} "
              f"{best['speedup_pct']:+.3f} % "
              f"[{-best['ci95_hi_pct']:+.3f}, {-best['ci95_lo_pct']:+.3f}]")
        summary["best_speedup_pct"] = best["speedup_pct"]
        # Three outcomes, not two. Reporting "NOT CLEARED" for an underpowered
        # session and for a genuinely small effect would invite reading a
        # measurement failure as a mechanism failure.
        above_bar = best["speedup_pct"] >= BAR_PCT
        resolved = best["ci95_hi_pct"] < 0.0  # interval excludes no-change
        # A best-width pass is necessary and not sufficient: E27 shipped a
        # correct per-width table, won at the widths it targeted, and still lost
        # score to a cost charged elsewhere. So a single winning width can never
        # carry the verdict while the replaced widths regress on net.
        regressed = [r for r in touched
                     if r["speedup_pct"] < 0.0 and r["ci95_lo_pct"] > 0.0]
        net_regression = mean_touched < 0.0
        named = ", ".join("{} M={} {:+.2f} %".format(
            r["shape"].split("_")[0], r["m"], r["speedup_pct"])
            for r in regressed)
        print(f"replaced widths that regress with a resolved interval: "
              f"{len(regressed)}/{len(touched)}"
              + (f"  ({named})" if regressed else ""))
        summary["regressed_touched_widths"] = float(len(regressed))
        summary["net_regression_over_touched"] = float(net_regression)
        clears = above_bar and resolved and not net_regression
        if above_bar and resolved and net_regression:
            verdict = (f"BEST-WIDTH ONLY -> M={best['m']} clears the bar but the "
                       f"replaced widths are {mean_touched:+.3f} % on net with "
                       f"{len(regressed)} resolved regressions. Not bankable as "
                       f"dispatched; the winning widths must be isolated first")
        elif clears:
            verdict = "CLEARED -> exactness work is authorised"
        elif above_bar and not resolved:
            verdict = (f"UNRESOLVED -> point estimate is above the bar but the "
                       f"95 % interval does not exclude no-change; "
                       f"achieved MDE {best['achieved_mde_pct']:.3f} % vs "
                       f"pre-registered {PREREG_MDE_PCT:.4f} %. Underpowered, "
                       f"not negative: add pairs before concluding")
        elif resolved:
            verdict = ("NOT CLEARED -> effect is real but below the bar")
        else:
            verdict = ("NOT CLEARED -> no resolved effect at this power; "
                       f"achieved MDE {best['achieved_mde_pct']:.3f} % vs "
                       f"pre-registered {PREREG_MDE_PCT:.4f} %")
        print(f"{BAR_PCT:.1f} % bar: {verdict}")
        summary["clears_bar"] = float(clears)
        summary["best_above_bar"] = float(above_bar)
        summary["best_interval_resolved"] = float(resolved)
    summary["worst_cand_max_rel"] = worst_cand

    # The pre-registered mechanism was weight-stream halving. It predicted the
    # win at M=5..8 and larger on mlp_down; the data contradicts both. The
    # surviving explanation is a fixed 8-row MMA tile: candidate cost flat in M
    # up to 8, base cost rising, so the sign of the effect is set by where those
    # two curves cross. Record the flatness so that claim is auditable evidence
    # rather than narrative.
    cost_model = []
    for shape in sorted({r["shape"] for r in table}):
        cand_p = [r["cand_us"] for r in table
                  if r["shape"] == shape and 4 <= r["m"] <= 8]
        base_p = [r["base_us"] for r in table
                  if r["shape"] == shape and 4 <= r["m"] <= 8]
        if len(cand_p) < 2:
            continue
        mean_p = statistics.fmean(cand_p)
        sd_p = statistics.stdev(cand_p)
        base_rise = 100.0 * (base_p[-1] / base_p[0] - 1.0)
        key = shape.split("_")[0]
        summary[f"cost_model/{key}/cand_plateau_us"] = mean_p
        summary[f"cost_model/{key}/cand_plateau_cv_pct"] = 100.0 * sd_p / mean_p
        summary[f"cost_model/{key}/base_rise_m4_to_m8_pct"] = base_rise
        cost_model.append([shape, mean_p, sd_p, 100.0 * sd_p / mean_p, base_rise])
    print("\n--- cost model: is the candidate flat in M? ---")
    for shape, mean_p, sd_p, cv, rise in cost_model:
        print(f"{shape:24s} cand M=4..8 plateau {mean_p:8.2f} us  "
              f"cv {cv:5.2f} %   base rise M4->M8 {rise:+6.1f} %")

    if args.wandb:
        import wandb
        run = wandb.init(
            project="qwen38-mlx-challenge-senpai",
            entity="wandb-applied-ai-team",
            job_type="microbenchmark",
            name=f"e44-sgmm-qmv-ab-{args.run_dir.name}",
            tags=["e44", "simdgroup-matrix", "qmv_fast", "affine4-g64",
                  "section-7.3", "paired-abba", "microbenchmark"],
            config={
                **{f"identity/{k}": v for k, v in identity.items()},
                "device": payload["device"],
                "architecture": payload["architecture"],
                "function": payload["function"],
                "order": payload["order"],
                "pairs": payload["pairs"],
                "reps": payload["reps"],
                "inner": payload["inner"],
                "prereg_mde_pct": PREREG_MDE_PCT,
                "bar_pct": BAR_PCT,
                "touched_widths": list(TOUCHED),
                # Preserved verbatim: this is a counterbalanced ungated local
                # arm, which is directional causal evidence and never a score.
                "cool_gate_passed_real_gate":
                    identity.get("cool_gate_passed_real_gate"),
                "gate_qualified_for_timing":
                    identity.get("gate_qualified_for_timing"),
            },
        )
        cols = ["shape", "m", "base_us", "cand_us", "speedup_pct", "sd_pct",
                "ci95_lo_pct", "ci95_hi_pct", "achieved_mde_pct", "n_pairs"]
        wandb.log({
            "per_width": wandb.Table(
                columns=cols,
                data=[[r[c] for c in cols] for r in table]),
            "fidelity": wandb.Table(
                columns=["shape", "m", "base_max_rel", "cand_max_rel",
                         "cand_rms_rel", "cand_vs_base_max_rel"],
                data=[[r["shape"], r["m"], r["base_max_rel"],
                       r["cand_max_rel"], r["cand_rms_rel"],
                       r["cand_vs_base_max_rel"]] for r in fidelity]),
            "raw_pairs": wandb.Table(
                columns=["shape", "m", "pair", "base_s", "cand_s",
                         "session_elapsed_s"],
                data=[[r["shape"], r["m"], r["pair"], r["base_s"], r["cand_s"],
                       r.get("session_elapsed_s", float("nan"))]
                      for r in timing]),
            "cost_model": wandb.Table(
                columns=["shape", "cand_plateau_us", "cand_plateau_sd_us",
                         "cand_plateau_cv_pct", "base_rise_m4_to_m8_pct"],
                data=cost_model),
            **summary,
        })
        run.summary.update(summary)
        print(f"\nW&B run: {run.url}  id={run.id}")
        run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
