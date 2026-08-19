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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e44_air_summary import dispatch_table_from_header  # noqa: E402

# Pre-registered in the assignment. Not recomputed from the data.
PREREG_MDE_PCT = 0.5040
# The assignment's promotion bar for this mechanism.
BAR_PCT = 5.0
# Widths the candidate actually replaces; 1..3 are the untouched-width guard.
TOUCHED = range(4, 10)

# Score identity: raw_p = serial_leg / mtp_leg. askeladd's E42 measured both QMV
# shares causally and bit-exactly.
#
# 🔴🔴 CORRECTED, ledger 176 (edward E50, merged 26fd0ac). The canonical model now
# lives in `research/qmv_score_leverage.py`; keep this file consistent with it.
#
# The RANKED harness times two different binaries -- `--baseline` is a prebuilt
# tree at /opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current and `--candidate`
# is this workspace -- and scores
#   baseline_serial_seconds_per_token_mean / candidate_mtp_seconds_per_token_mean.
# So d ln(serial)/dx == 0 for everything we can edit, and PSI_SERIAL does NOT
# enter a ranked price. A UNIFORM change is worth +PSI_MTP per 1 %, not
# PSI_MTP - PSI_SERIAL = -0.1789. It is POSITIVE, and 3.77x larger in magnitude.
#
# For THIS file the correction matters in one specific place, and it is adverse:
# the ceiling term below is a genuinely uniform effect (one shared register
# allocation, all widths including M=1). It was priced with the -0.1789
# coefficient, which made a uniform SLOWDOWN look almost free -- and briefly even
# beneficial. It is not. A uniform slowdown costs PSI_MTP per 1 %, so the E44
# ceiling bound and anything gated on it (thorfinn's E46/E49 register trades) must
# be re-derived with the larger coefficient.
PSI_MTP = 0.6736
PSI_SERIAL = 0.8525     # LOCAL-ONLY. Retained for the local two-leg ratio only.
# %score per 1 % of QMV cost removed, by harness. Gated is harness-invariant.
LEV_UNIFORM_RANKED = PSI_MTP                # what a submission actually scores
LEV_UNIFORM_LOCAL = PSI_MTP - PSI_SERIAL    # what --local-iterate's ratio shows

# Student t by df: two-sided 95 % critical value and the one-sided 80 %-power
# companion. Hardcoded so the summary has no scipy dependency. df=4 keeps the
# exact constants r1 published, so its numbers are unmoved by this generalisation.
T_975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.7764451051977987,
         5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157,
         10: 2.228139, 11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787,
         15: 2.131450, 16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024,
         20: 2.085963}
T_800 = {1: 1.376382, 2: 1.060660, 3: 0.978472, 4: 0.9409645, 5: 0.919544,
         6: 0.905703, 7: 0.896030, 8: 0.888890, 9: 0.883404, 10: 0.879058,
         11: 0.875530, 12: 0.872609, 13: 0.870152, 14: 0.868055, 15: 0.866245,
         16: 0.864667, 17: 0.863279, 18: 0.862049, 19: 0.860951, 20: 0.859964}


def t_quantiles(df: int) -> tuple[float, float]:
    if df in T_975:
        return T_975[df], T_800[df]
    return 1.959964, 0.841621  # normal limit


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
    t975, t800 = t_quantiles(n - 1)
    mean = statistics.fmean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else float("nan")
    half = t975 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n_pairs": n,
        "mean_pct": mean,
        "sd_pct": sd,
        "ci95_lo_pct": mean - half,
        "ci95_hi_pct": mean + half,
        "achieved_mde_pct": ((t975 + t800) / math.sqrt(n) * sd
                             if n > 1 else float("nan")),
    }


def zero_effect_cells(run_dir: pathlib.Path) -> list[dict]:
    """Per-cell effects from an A/A session, where the true effect is 0 at every
    width. r1 read its floor from three cheap guard widths only; this reads it
    from every width the decision is actually made at."""
    payload, identity = load(run_dir)
    # The whole claim of a control is that its true effect is zero, which holds
    # only if the two arms are the same bytes. A control whose arms differ would
    # quietly inflate the floor and hide a real regression behind it.
    base_sha, cand_sha = (identity.get("base_metal_sha256"),
                          identity.get("cand_metal_sha256"))
    if base_sha is None or cand_sha is None or base_sha != cand_sha:
        raise SystemExit(
            f"{run_dir}: control arms are not identical bytes "
            f"(base={base_sha}, cand={cand_sha}). This is not an A/A control.")
    timing = [r for r in payload["measurements"] if r["kind"] == "timing"]
    cells = []
    for shape in sorted({r["shape"] for r in timing}):
        for m in sorted({r["m"] for r in timing if r["shape"] == shape}):
            pairs = [r for r in timing if r["shape"] == shape and r["m"] == m]
            st = paired_stats([100.0 * (r["cand_s"] - r["base_s"]) / r["base_s"]
                               for r in pairs])
            cells.append({"shape": shape, "m": m, "effect_pct": -st["mean_pct"],
                          **st})
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--touched", default="4,5,6,7,8,9",
                        help="widths the candidate replaces (default: r1's "
                             "all-widths arm)")
    parser.add_argument("--control", type=pathlib.Path,
                        help="A/A session directory; its cells have a true "
                             "effect of exactly zero and set the floor")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--artifact", action="append", default=[],
                        type=pathlib.Path,
                        help="file to attach to the W&B run; repeatable")
    parser.add_argument("--extra-summary", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra scalar to record, e.g. a compile-only "
                             "register readout; repeatable")
    args = parser.parse_args()
    touched_widths = {int(w) for w in args.touched.split(",") if w.strip()}
    touched_label = ", ".join(str(w) for w in sorted(touched_widths))

    payload, identity = load(args.run_dir)
    rows = payload["measurements"]

    # `--touched` is a claim about which widths the candidate arm actually
    # changes, and it silently poisons the guard if it is wrong: widths that did
    # change get averaged into a "zero-effect" floor. The two arm sources are in
    # the run directory, so the claim is checked against them rather than
    # trusted.
    arms = (args.run_dir / "base.metal", args.run_dir / "cand.metal")
    if all(a.is_file() for a in arms):
        base_dispatch = dispatch_table_from_header(arms[0])
        cand_dispatch = dispatch_table_from_header(arms[1])
        changed = {m for m, cell in base_dispatch.items()
                   if cand_dispatch.get(m) != cell}
        if changed != touched_widths:
            raise SystemExit(
                f"{args.run_dir}: --touched says {sorted(touched_widths)} but "
                f"the arm sources differ at {sorted(changed)}. Refusing: a "
                f"width that changed cannot serve as a zero-effect guard.")
        print(f"touched widths verified against both arm sources: "
              f"{sorted(changed) if changed else 'none (A/A control)'}")

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
            if m in touched_widths and speedup >= BAR_PCT and st["ci95_hi_pct"] < 0.0:
                verdict += " CLEARS BAR"
            tag = "" if m in touched_widths else " (guard)"
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

    control_cells = zero_effect_cells(args.control) if args.control else []
    floor_pct = float("nan")
    if control_cells:
        floor_pct = max(abs(c["effect_pct"]) for c in control_cells)
        resolved = [c for c in control_cells
                    if c["ci95_lo_pct"] > 0.0 or c["ci95_hi_pct"] < 0.0]
        print("\n--- A/A control: same design, both arms are the base bytes ---")
        print(f"{'shape':<26}{'M':>3}{'effect %':>11}{'sd %':>8}"
              f"{'MDE %':>9}  resolved on a true zero")
        for c in control_cells:
            flag = ("YES" if (c["ci95_lo_pct"] > 0.0 or c["ci95_hi_pct"] < 0.0)
                    else "no")
            print(f"{c['shape']:<26}{c['m']:>3}{c['effect_pct']:>+11.3f}"
                  f"{c['sd_pct']:>8.3f}{c['achieved_mde_pct']:>9.3f}  {flag}")
        print(f"true effect is exactly 0 in all {len(control_cells)} cells: "
              f"worst |effect| = {floor_pct:.3f} %, "
              f"sd of effects = "
              f"{statistics.stdev([c['effect_pct'] for c in control_cells]):.3f} %, "
              f"intervals excluding zero = {len(resolved)}/{len(control_cells)}")
        print(f"RESOLUTION FLOOR taken from this control: {floor_pct:.3f} % "
              f"-- no effect below it is believable whatever its interval says")

    touched = [r for r in table if r["m"] in touched_widths]
    guard = [r for r in table if r["m"] not in touched_widths]
    guard_label = ", ".join(str(m) for m in sorted({r["m"] for r in guard}))
    best = max(touched, key=lambda r: r["speedup_pct"], default=None)
    print("\n--- decision ---")
    if touched:
        mean_touched = statistics.fmean(r["speedup_pct"] for r in touched)
        print(f"mean speedup over replaced widths M in {{{touched_label}}}: "
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
        print(f"mean effect on untouched-width guard M in {{{guard_label}}}: "
              f"{mean_guard:+.3f} % (identical code in both arms; any effect "
              f"is the shared register allocation, not the cell)")
        print(f"empirical noise floor from the guard: sd={floor:.3f} % over "
              f"{len(guard)} zero-effect measurement(s), "
              f"worst |effect|="
              f"{max(abs(r['speedup_pct']) for r in guard):.3f} %")
        summary["mean_effect_guard_pct"] = mean_guard
        summary["guard_noise_floor_sd_pct"] = floor
        if control_cells:
            over = [r for r in guard if abs(r["speedup_pct"]) > floor_pct]
            print(f"untouched widths whose |effect| exceeds the "
                  f"{floor_pct:.3f} % control floor: {len(over)}/{len(guard)}"
                  + ("".join("  ({} M={} {:+.3f} %)".format(
                      r["shape"].split("_")[0], r["m"], r["speedup_pct"])
                      for r in over) if over else ""))
            summary["control_floor_pct"] = floor_pct
            summary["guard_cells_over_control_floor"] = float(len(over))
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
    # Flatness is only a claim about widths the MMA cell actually serves with a
    # single 8-row tile. Averaging a scalar cell into the plateau would make the
    # narrow arm look ragged for a reason that has nothing to do with the tile.
    plateau_widths = sorted(m for m in touched_widths if m <= 8)
    cost_model = []
    for shape in sorted({r["shape"] for r in table}):
        cand_p = [r["cand_us"] for r in table
                  if r["shape"] == shape and r["m"] in plateau_widths]
        base_p = [r["base_us"] for r in table
                  if r["shape"] == shape and r["m"] in plateau_widths]
        if len(cand_p) < 2:
            continue
        mean_p = statistics.fmean(cand_p)
        sd_p = statistics.stdev(cand_p)
        base_rise = 100.0 * (base_p[-1] / base_p[0] - 1.0)
        spread = 100.0 * (max(cand_p) - min(cand_p)) / max(cand_p)
        key = shape.split("_")[0]
        summary[f"cost_model/{key}/cand_plateau_us"] = mean_p
        summary[f"cost_model/{key}/cand_plateau_cv_pct"] = 100.0 * sd_p / mean_p
        summary[f"cost_model/{key}/cand_plateau_spread_pct"] = spread
        summary[f"cost_model/{key}/base_rise_pct"] = base_rise
        cost_model.append([shape, mean_p, sd_p, 100.0 * sd_p / mean_p, spread,
                           base_rise])
    span = (f"M={plateau_widths[0]}..{plateau_widths[-1]}"
            if plateau_widths else "none")
    print(f"\n--- cost model: is the candidate flat over its single-tile "
          f"widths {span}? ---")
    for shape, mean_p, sd_p, cv, spread, rise in cost_model:
        print(f"{shape:24s} cand plateau {mean_p:8.2f} us  cv {cv:5.2f} %  "
              f"spread {spread:5.2f} %   base rise over {span} {rise:+6.1f} %")

    # 🔴 Ledger 176: the two halves NO LONGER have opposite score signs. Under the
    # retracted local model the width term was +0.6736 and the ceiling term
    # -0.1789, so summing them was a sign error. On ranked BOTH are +0.6736,
    # because the serial leg is a pinned separate binary in either case.
    #
    # They are still reported separately, and still never summed, for a different
    # and now stronger reason: they carry different SHARES (`f` for the touched
    # widths, the whole kernel for the shared register allocation) and different
    # evidence quality. Same coefficient is not same quantity.
    if touched:
        gated = PSI_MTP
        # 🔴 RANKED coefficient. Was PSI_MTP - PSI_SERIAL = -0.1789 until ledger
        # 176; the serial leg is a pinned separate binary and cannot respond.
        uniform = LEV_UNIFORM_RANKED
        mean_touched = statistics.fmean(r["speedup_pct"] for r in touched)
        print("\n--- score decomposition (psi_mtp=%.4f; RANKED, ledger 176) ---"
              % PSI_MTP)
        print("    uniform coefficient %+.4f (was %+.4f under the retracted "
              "local model)" % (LEV_UNIFORM_RANKED, LEV_UNIFORM_LOCAL))
        print(f"width term   M in {{{touched_label}}}: MTP leg only, the serial "
              f"leg never dispatches these widths")
        print(f"             dScore = {gated:+.3f} % per 1 % of MTP-leg QMV cost "
              f"removed; mean win here = {mean_touched:+.3f} %")
        print(f"             ->  dScore = {gated * mean_touched:+.3f} % x f, "
              f"f = share of MTP-leg QMV cost dispatched at these widths")
        row = "  ".join(f"f={f:.2f}: {gated * mean_touched * f:+.3f} %"
                        for f in (0.05, 0.10, 0.25, 0.50))
        print(f"             {row}")
        print(f"             f is UNIDENTIFIED (E43): this is a sensitivity "
              f"table, not a prediction")
        if guard:
            mean_guard = statistics.fmean(r["speedup_pct"] for r in guard)
            print(f"ceiling term M in {{{guard_label}}}: uniform, it acts through "
                  f"one shared register allocation and so also speeds up M=1")
            # 🔴 Ledger 176. The old rationale here was "ADVERSE, because the
            # serial leg is more QMV-dominated than the candidate leg" -- that was
            # the retracted local model and it is deleted. The coefficient is per
            # 1 % of QMV cost REMOVED, so it is positive; this term is adverse
            # because the ceiling change ADDS cost (registers rise, occupancy
            # falls, every width slows), not because the coefficient is negative.
            print(f"             dScore = {uniform:+.3f} % per 1 % of cost "
                  f"REMOVED; this term ADDS cost, so its contribution is "
                  f"negative")
            print(f"             magnitude is {abs(LEV_UNIFORM_RANKED / LEV_UNIFORM_LOCAL):.2f}x "
                  f"the retracted local model's -- a uniform slowdown is NOT "
                  f"nearly free, which is how -0.1789 made it look")
            worst_guard = max(abs(r["speedup_pct"]) for r in guard)
            # Two different floors, and the conservative one wins. The A/A
            # control measures pure measurement noise between IDENTICAL
            # binaries. The guard cells share their cell source but are compiled
            # into a DIFFERENT binary, so they also carry register-allocation
            # and code-layout scatter. Bounding a ceiling effect that is claimed
            # to act through exactly that binary difference therefore requires
            # the larger of the two, or the bound quietly assumes away the very
            # variability it is meant to cover.
            bound = max(worst_guard, floor_pct) if floor_pct == floor_pct \
                else worst_guard
            if floor_pct == floor_pct:
                print(f"             floors: A/A control (identical binaries) "
                      f"{floor_pct:.3f} %, untouched-width guard (different "
                      f"binaries) {worst_guard:.3f} % -- using the larger")
            print(f"             measured mean effect {mean_guard:+.3f} %, "
                  f"below the {bound:.3f} % floor, so it is BOUNDED and not "
                  f"measured: |dScore| <= {abs(uniform) * bound:.4f} %")
            summary["score/width_term_per_1pct"] = gated
            summary["score/ceiling_term_per_1pct"] = uniform
            summary["score/width_term_at_f1_pct"] = gated * mean_touched

    for pair in args.extra_summary:
        key, _, value = pair.partition("=")
        summary[key.strip()] = float(value)

    if args.wandb:
        import wandb
        run = wandb.init(
            project="qwen38-mlx-challenge-senpai",
            entity="wandb-applied-ai-team",
            job_type="microbenchmark",
            name=args.wandb_name or f"e44-sgmm-qmv-ab-{args.run_dir.name}",
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
                "control_run_dir": str(args.control) if args.control else None,
                "bar_pct": BAR_PCT,
                "touched_widths": sorted(touched_widths),
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
                         "cand_plateau_cv_pct", "cand_plateau_spread_pct",
                         "base_rise_pct"],
                data=cost_model),
            **summary,
        })
        if control_cells:
            wandb.log({"aa_control": wandb.Table(
                columns=["shape", "m", "effect_pct", "sd_pct", "ci95_lo_pct",
                         "ci95_hi_pct", "n_pairs"],
                data=[[c["shape"], c["m"], c["effect_pct"], c["sd_pct"],
                       c["ci95_lo_pct"], c["ci95_hi_pct"], c["n_pairs"]]
                      for c in control_cells])})
        for path in args.artifact:
            wandb.save(str(path), base_path=str(path.parent), policy="now")
        run.summary.update(summary)
        print(f"\nW&B run: {run.url}  id={run.id}")
        run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
