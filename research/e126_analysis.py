#!/usr/bin/env python3
"""E126 rung 1: price Route B's kernel-side prize against the SHIPPED base.

    research/e126_analysis.py research/out/e126-rung1/rate.json \
        --model research/e126-artifacts/rung0-model.json \
        --out research/e126-artifacts/rung1-summary.json

harness=local. The session is counterbalanced and ungated, so it supports a
relative claim only and reports no score.

The estimator, the warm-up rule, the round weights, the cost weighting, the
bootstrap and the three void checks are E121's, imported unchanged, so this
session can be compared with E121 rung 2 and with E123 directly. Only the
reference arm changes: arm 0 is `share_off`, the pre-E121 shape.

Three contrasts matter:

  gain(n_sums_free  vs share_off)  cross-instrument replication of thorfinn's
                                   5.88 % at NA=4
  gain(n_sums_free  vs share_on)   PRIMARY. what Route B can still win on the
                                   base that actually ships
  gain(n_sums_loaded vs share_on)  the same question for the mix-faithful arm

and one derived quantity:

  O = gain(share_on vs share_off) / gain(n_sums_free vs share_off)

which is the share of Route B's kernel-side prize that E121 already collects.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e121_analysis import (  # noqa: E402
    FAMILY_CALLS, IMPLAUSIBLE_GBS, LEG_TRANSFER, RANKED_TRANSFER,
    ROUND_WEIGHTS, bootstrap, collect, fidelity, gains, med, thermal,
)

BASE_ARM = "share_off"
SHIPPED_ARM = "share_on"
RANKED_ARCH = "applegpu_g17s"
ARMS = ("share_off", "n_sums_free", "n_nosums_e123", "n_sums_loaded",
        "share_on")
DIAG_ARMS = ("n_sums_free", "n_nosums_e123", "n_sums_loaded")

# NA=2 is unreachable: `affine_qmv_fast` routes M=2 to a separate function and
# no `_m` instantiation produces `wide<2>`. NA=5 folds the shipped gate off, so
# `share_on` is byte-identical to `share_off` there and its reading is noise.
UNREACHABLE = (2,)
GATE_OFF = (2, 5)

# Pre-registered in `research/e126-results.md` before this session ran.
PREREGISTERED = {
    "free_vs_off_na4_pct": (6.4, (4.5, 8.5)),
    "primary_free_vs_on_na4_pct": (4.7, (4.0, 6.6)),
    "e123_vs_off_na4_pct": (6.4, (5.0, 7.5)),
    "loaded_vs_on_na4_pct": (1.7, (0.5, 3.5)),
    "overlap_O_na4": (0.25, (0.15, 0.35)),
    "primary_round_weighted_pct": (3.7, (3.0, 5.0)),
}
THORFINN_GAIN = {3: 2.20, 4: 5.88, 5: 6.52}
STOP_DISAGREEMENT = 2.0


def cells_of(doc: dict, warmup: int) -> dict:
    return collect(doc, warmup)


def pooled(cells: dict, shapes, widths, arm: str, ref: str):
    out: dict[int, list[float]] = {}
    for width in widths:
        acc: list[float] = []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is not None:
                acc.extend(gains(cell, arm, ref))
        if acc:
            out[width] = acc
    return out


def cost_weighted(cells: dict, shapes, widths, arm: str, ref: str):
    """Per width, shape effects combined by their share of round cost."""
    per_width: dict[int, float] = {}
    for width in widths:
        num = den = 0.0
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is None or shape not in FAMILY_CALLS:
                continue
            base = med([b[ref] for b in cell["blocks"] if b.get(ref)])
            weight = FAMILY_CALLS[shape] * base
            num += weight * med(gains(cell, arm, ref))
            den += weight
        if den:
            per_width[width] = num / den
    return per_width


def round_weighted(per_width: dict[int, float], pin_zero=()) -> tuple:
    """Round-weighted percent with unreachable and gated widths pinned to 0."""
    total = sum(ROUND_WEIGHTS.values())
    value = 0.0
    covered = 0.0
    for width, weight in ROUND_WEIGHTS.items():
        if width in pin_zero:
            covered += weight
            continue
        if width not in per_width:
            continue
        value += weight * per_width[width]
        covered += weight
    return value / total * total / 1.0, covered / total


def bandwidth_rows(doc: dict, warmup: int) -> list[dict]:
    rows = []
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup:
            continue
        base = row["seconds"].get(BASE_ARM)
        if not base:
            continue
        rows.append({
            "shape": row["shape"], "m": row["m"], "block": row["block"],
            "base_gbs": row["read_bytes"] / base / 1e9,
            "seconds": row["seconds"],
        })
    return rows


def per_arm_bandwidth(doc: dict, warmup: int) -> dict:
    """Achieved GB/s of every arm in every cell, pooled over surviving blocks.

    `read_bytes` is the same for all arms in a cell, so this is a rescaling of
    the arm times. It is published per arm because a phi-axis placement needs
    each arm's own distance below peak, not just the reference arm's.
    """
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    read_bytes: dict[tuple[str, int], int] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup:
            continue
        key = (row["shape"], row["m"])
        read_bytes[key] = row["read_bytes"]
        bucket = cells.setdefault(key, {})
        for arm, sec in row["seconds"].items():
            if sec:
                bucket.setdefault(arm, []).append(row["read_bytes"] / sec / 1e9)
    out = {}
    for (shape, m), bucket in sorted(cells.items()):
        out["%s|NA%d" % (shape, m)] = {
            "shape": shape, "na": m, "read_bytes": read_bytes[(shape, m)],
            "blocks": max(len(v) for v in bucket.values()),
            "gbs": {arm: med(v) for arm, v in sorted(bucket.items())},
        }
    return out


def ols(xs: list[float], ys: list[float]) -> dict | None:
    """Slope, intercept and Pearson r of y on x."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    return {"n": n, "slope_pct_per_gbs": slope,
            "intercept_pct": my - slope * mx,
            "r": sxy / (sxx ** 0.5 * syy ** 0.5),
            "gbs_min": min(xs), "gbs_max": max(xs)}


def covariate(doc: dict, warmup: int, arm: str, ref: str) -> dict:
    """Achieved GB/s as a continuous covariate, at two nesting levels.

    `within_cell` holds shape AND width fixed and regresses across the three
    surviving blocks. Achieved rate barely moves inside one cell, so those
    slopes are a noise floor, not a bandwidth response.

    `within_width` holds width fixed and uses the five scored shapes as the
    points. That is the level at which achieved rate actually varies, from
    about 78 to 212 GB/s, and it is the level that answers the question
    thorfinn asked: is this mechanism bandwidth sensitive? Shape is never
    pooled with width, because the two contrasts would be confounded.
    """
    cell_xy: dict[tuple[str, int], tuple[list, list]] = {}
    for row in bandwidth_rows(doc, warmup):
        secs = row["seconds"]
        if not (secs.get(ref) and secs.get(arm)):
            continue
        key = (row["shape"], row["m"])
        xs, ys = cell_xy.setdefault(key, ([], []))
        xs.append(row["base_gbs"])
        ys.append(100.0 * (1.0 - secs[arm] / secs[ref]))

    within_cell = {}
    for (shape, m), (xs, ys) in sorted(cell_xy.items()):
        fit = ols(xs, ys)
        if fit is not None:
            within_cell["%s|NA%d" % (shape, m)] = fit

    by_width: dict[int, tuple[list, list, list]] = {}
    for (shape, m), (xs, ys) in sorted(cell_xy.items()):
        wx, wy, names = by_width.setdefault(m, ([], [], []))
        wx.append(statistics.fmean(xs))
        wy.append(med(ys))
        names.append(shape)
    within_width = {}
    for m, (wx, wy, names) in sorted(by_width.items()):
        fit = ols(wx, wy)
        if fit is not None:
            fit["points"] = [
                {"shape": s, "base_gbs": x, "effect_pct": y}
                for s, x, y in zip(names, wx, wy)]
            within_width["NA%d" % m] = fit

    cell_slopes = [v["slope_pct_per_gbs"] for v in within_cell.values()]
    return {"within_cell": within_cell,
            "within_cell_median_slope": med(cell_slopes),
            "within_cell_max_abs_slope": max((abs(s) for s in cell_slopes),
                                             default=float("nan")),
            "within_width": within_width}


def void_checks(doc: dict, cells: dict, shapes, widths,
                controls: list[str]) -> dict:
    """E121's three checks, with the byte-identical control replaced.

    E126 has no scaffold arm. The equivalent zero-reading control is
    `share_on` at NA=5, where the shipped `if constexpr` gate folds off and the
    rung 0 census proved the machine text is byte-identical to `share_off`.
    """
    verdicts: list[str] = []
    fast = []
    for row in doc["measurements"]:
        if row["kind"] != "timing":
            continue
        for arm, secs in row["seconds"].items():
            rate = row["read_bytes"] / secs / 1e9
            if rate > IMPLAUSIBLE_GBS:
                fast.append("%s M=%d %s block %d: %.0f GB/s" % (
                    row["shape"], row["m"], arm, row["block"], rate))
    if fast:
        verdicts.append("implied bandwidth above %.0f GB/s in %d rows"
                        % (IMPLAUSIBLE_GBS, len(fast)))

    control = None
    if 5 in widths:
        values = pooled(cells, shapes, [5], SHIPPED_ARM, BASE_ARM).get(5, [])
        control = med(values)
        if values and abs(control) > 0.50:
            verdicts.append(
                "share_on at NA=5 is byte-identical text and reads %+.3f %%"
                % control)

    undetected = [c for c in controls if "detected=False" in c]
    if undetected:
        verdicts.append("%d of %d positive controls did not fire"
                        % (len(undetected), len(controls)))

    return {"void": bool(verdicts), "reasons": verdicts,
            "implausible_row_count": len(fast),
            "implausible_rows": fast[:20],
            "gate_folded_control_arm": "%s at NA=5" % SHIPPED_ARM,
            "gate_folded_control_pct": control,
            "control_tolerance_pct": 0.50}


def per_arm_thermal(doc: dict) -> dict:
    """Entry and exit temperature per timed cell, and the entry spread.

    The palindrome puts every arm at every position inside one block, so no arm
    owns a temperature. What can still differ is the entry temperature of the
    CELLS an arm is averaged over, and that is what this reports.
    """
    rows = [r for r in doc["measurements"] if r["kind"] == "thermal"]
    by_width: dict[int, dict] = {}
    for row in rows:
        entry = by_width.setdefault(row["m"], {"entry": [], "exit": []})
        entry["entry"].append(row["gpu_temp_entry_c"])
        entry["exit"].append(row["gpu_temp_exit_c"])
    out = {}
    for width, values in sorted(by_width.items()):
        out["NA%d" % width] = {
            "cells": len(values["entry"]),
            "entry_min_c": min(values["entry"]),
            "entry_max_c": max(values["entry"]),
            "entry_spread_c": max(values["entry"]) - min(values["entry"]),
            "exit_max_c": max(values["exit"]),
        }
    return out


def task4_revision(loaded_vs_off: dict, loaded_vs_on: dict) -> dict:
    """Re-price thorfinn's rung 5e using the MEASURED basis, not the assumed one.

    Rung 0 assumed his 5.88 % at M=4 came from an `n_sums_free`-like arm. This
    session measures `n_sums_loaded`, the mix-faithful arm, at 5.311 % against
    `share_off` at NA=4, which reproduces his number to 11 %. `n_sums_free`
    reads 9.279 % and does not. So the faithful arm is his basis.

    His absolute replica-dispatch cost does not change when E121 is present, so
    the marginal gain is scaled by the measured surviving fraction
    `gain(loaded vs share_on) / gain(loaded vs share_off)` and the replica cost
    is subtracted after.
    """
    thorfinn_net = {3: -0.03, 4: 3.46, 5: 4.40}
    per_width = {}
    for width, gain in THORFINN_GAIN.items():
        off = loaded_vs_off.get(width)
        on = loaded_vs_on.get(width)
        if off is None or on is None or off == 0:
            continue
        surviving = on / off
        replica = gain - thorfinn_net[width]
        marginal = gain * surviving
        per_width[width] = {
            "thorfinn_gain_on_share_off_pct": gain,
            "measured_surviving_fraction": surviving,
            "marginal_gain_pct": marginal,
            "replica_cost_pp": replica,
            "marginal_net_pct": marginal - replica,
        }

    def rw(pick) -> float:
        return sum(ROUND_WEIGHTS[w] * pick(w) for w in per_width
                   if w in ROUND_WEIGHTS)

    base_net = sum(ROUND_WEIGHTS[w] * thorfinn_net[w] for w in per_width)
    e121_net = rw(lambda w: per_width[w]["marginal_net_pct"])
    return {
        "per_width": per_width,
        "share_off_base_round_net_pct": base_net,
        "share_off_base_leg_pct": base_net * LEG_TRANSFER,
        "e121_base_round_net_pct": e121_net,
        "e121_base_leg_pct": e121_net * LEG_TRANSFER,
        "e121_base_ranked_pct": e121_net * LEG_TRANSFER * RANKED_TRANSFER,
        "fraction_of_route_b_leg_value_removed_by_e121":
            1.0 - e121_net / base_net if base_net else float("nan"),
    }


def slot_position(doc: dict, warmup_blocks: int, widths) -> dict:
    """Bound harness defect 30, the slot-position bias a short palindrome keeps.

    A palindrome of five arms is ten slots, so arm 0 holds the two extreme
    slots and the last arm holds the two middle ones. Askeladd reports that a
    convex post-sample fall can then separate byte-identical sources.

    Two bounds are available in this session. The palindrome asymmetry
    `slot[9-i] - slot[i]` cancels the arm effect and exposes any monotone
    drift. The convex part is confounded with the arm effect at every width
    except NA=5, where the shipped gate folds off and `share_on` is
    byte-identical to `share_off` while sitting in the opposite slot pair.
    """
    arms = doc["arms"]
    n_slots = len(arms) * 2
    out = {"slots_per_block": n_slots, "order": doc.get("order")}
    per_width = {}
    for width in widths:
        rows = [m for m in doc["measurements"]
                if m["kind"] == "timing" and m["m"] == width
                and m["block"] >= warmup_blocks]
        asym = {}
        for i, arm in enumerate(arms):
            vals = []
            for m in rows:
                mid = statistics.median(m["slots"])
                vals.append(100.0 * (m["slots"][n_slots - 1 - i]
                                     - m["slots"][i]) / mid)
            asym[arm] = {
                "slots": [i, n_slots - 1 - i],
                "median_pp": statistics.median(vals),
                "max_abs_pp": max(abs(v) for v in vals),
            }
        per_width[width] = {"blocks": len(rows), "palindrome_asymmetry": asym}
    out["per_width"] = per_width
    out["max_abs_median_asymmetry_pp"] = max(
        abs(a["median_pp"]) for w in per_width.values()
        for a in w["palindrome_asymmetry"].values())
    return out


def residency(census: dict, arch: str, widths) -> dict:
    """Resident simdgroups of each ISOLATED arm kernel on the timed device.

    The isolated harness compiles one width per kernel, so each arm gets its
    own register allocation and an arm can win or lose an occupancy step that
    the shipped kernel never sees: `affine_qmv_fast` inlines every width into
    one kernel with one allocation (Rule 56). A contrast whose two arms differ
    here is measuring occupancy as well as instructions, and must be read with
    the shipped-entry census, not instead of it.
    """
    from e121_arms import simdgroups  # noqa: PLC0415

    table = {}
    for arm, row in census["arms"].items():
        for width in widths:
            value = row.get(arch, {}).get(str(width))
            if value is None:
                continue
            table.setdefault(arm, {})[width] = {
                "registers": value["registers"],
                "simdgroups": simdgroups(arch, value["registers"]),
            }
    confounded = []
    for arm in DIAG_ARMS + (SHIPPED_ARM,):
        for ref in (BASE_ARM, SHIPPED_ARM):
            if arm == ref:
                continue
            for width in widths:
                a = table.get(arm, {}).get(width)
                b = table.get(ref, {}).get(width)
                if a is None or b is None or a["simdgroups"] == b["simdgroups"]:
                    continue
                confounded.append({
                    "contrast": "%s_vs_%s" % (arm, ref), "width": width,
                    "arm_simdgroups": a["simdgroups"],
                    "ref_simdgroups": b["simdgroups"],
                    "residency_ratio": a["simdgroups"] / b["simdgroups"],
                })
    return {"architecture": arch, "isolated_per_arm": table,
            "occupancy_confounded_contrasts": confounded}


def score_band(name: str, value: float) -> str:
    point, (lo, hi) = PREREGISTERED[name]
    inside = lo <= value <= hi
    return "%s  predicted %.2f [%.2f, %.2f]  measured %.3f  %s" % (
        name, point, lo, hi, value, "IN BAND" if inside else "MISS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate", type=pathlib.Path)
    ap.add_argument("--model", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--warmup-blocks", type=int, default=1)
    args = ap.parse_args()

    doc = json.loads(args.rate.read_text())
    cells = cells_of(doc, args.warmup_blocks)
    shapes = sorted({s for s, _ in cells})
    widths = sorted({m for _, m in cells})
    failures, controls, checked = fidelity(doc)

    contrasts = {}
    for arm in ARMS:
        if arm == BASE_ARM:
            continue
        contrasts["%s_vs_share_off" % arm] = {
            "per_width_pct": {w: med(v) for w, v in
                              pooled(cells, shapes, widths, arm,
                                     BASE_ARM).items()},
            "cost_weighted_per_width_pct": cost_weighted(
                cells, shapes, widths, arm, BASE_ARM),
            "per_shape_per_width_pct": {
                "%s|NA%d" % (s, w): med(gains(cells[(s, w)], arm, BASE_ARM))
                for (s, w) in sorted(cells)},
        }
    for arm in DIAG_ARMS:
        contrasts["%s_vs_share_on" % arm] = {
            "per_width_pct": {w: med(v) for w, v in
                              pooled(cells, shapes, widths, arm,
                                     SHIPPED_ARM).items()},
            "cost_weighted_per_width_pct": cost_weighted(
                cells, shapes, widths, arm, SHIPPED_ARM),
            "per_shape_per_width_pct": {
                "%s|NA%d" % (s, w): med(gains(cells[(s, w)], arm, SHIPPED_ARM))
                for (s, w) in sorted(cells)},
        }

    on_vs_off = contrasts["share_on_vs_share_off"]["per_width_pct"]
    free_vs_off = contrasts["n_sums_free_vs_share_off"]["per_width_pct"]
    loaded_vs_off = contrasts["n_sums_loaded_vs_share_off"]["per_width_pct"]
    overlap = {}
    for width in widths:
        if width in GATE_OFF:
            overlap[width] = 0.0
            continue
        prize = free_vs_off.get(width)
        if prize:
            overlap[width] = on_vs_off[width] / prize

    primary = contrasts["n_sums_free_vs_share_on"]["per_width_pct"]
    primary_rw, coverage = round_weighted(
        {w: (0.0 if w in UNREACHABLE else v) for w, v in primary.items()},
        pin_zero=UNREACHABLE)

    replication = {}
    for width, measured in free_vs_off.items():
        expected = THORFINN_GAIN.get(width)
        if expected:
            ratio = measured / expected if expected else float("nan")
            replication["NA%d" % width] = {
                "thorfinn_pct": expected, "measured_pct": measured,
                "ratio": ratio,
                "stop_rule_fires": bool(
                    ratio > STOP_DISAGREEMENT or ratio < 1.0 / STOP_DISAGREEMENT),
            }

    samples = pooled(cells, shapes, widths, "n_sums_free", SHIPPED_ARM)
    ci_lo, ci_hi = bootstrap(samples, drop=UNREACHABLE)

    summary = {
        "harness": "local",
        "experiment": "e126-price-route-b-on-the-shipped-base",
        "reference_arm": BASE_ARM,
        "shipped_arm": SHIPPED_ARM,
        "shapes": shapes,
        "widths": widths,
        "warmup_blocks_discarded": args.warmup_blocks,
        "contrasts": contrasts,
        "overlap_O_per_width": overlap,
        "primary_free_vs_on_per_width_pct": primary,
        "primary_round_weighted_pct": primary_rw,
        "primary_round_weighted_ci95": [ci_lo, ci_hi],
        "primary_round_weight_coverage": coverage,
        "faithful_loaded_vs_on_per_width_pct":
            contrasts["n_sums_loaded_vs_share_on"]["per_width_pct"],
        "faithful_loaded_vs_off_per_width_pct": loaded_vs_off,
        "thorfinn_replication": replication,
        "task4_revision": task4_revision(
            loaded_vs_off,
            contrasts["n_sums_loaded_vs_share_on"]["per_width_pct"]),
        "per_arm_bandwidth": per_arm_bandwidth(doc, args.warmup_blocks),
        "slot_position": slot_position(doc, args.warmup_blocks, widths),
        "bandwidth_covariate_free_vs_off": covariate(
            doc, args.warmup_blocks, "n_sums_free", BASE_ARM),
        "bandwidth_covariate_free_vs_on": covariate(
            doc, args.warmup_blocks, "n_sums_free", SHIPPED_ARM),
        "bandwidth_covariate_on_vs_off": covariate(
            doc, args.warmup_blocks, SHIPPED_ARM, BASE_ARM),
        "validity": void_checks(doc, cells, shapes, widths, controls),
        "exactness_failures": failures,
        "exactness_checks": checked,
        "positive_controls": controls,
        "thermal": thermal(doc),
        "thermal_per_width": per_arm_thermal(doc),
        "leg_transfer": LEG_TRANSFER,
        "ranked_transfer": RANKED_TRANSFER,
        "preregistered": {k: {"point": p, "band": list(b)}
                          for k, (p, b) in PREREGISTERED.items()},
    }

    if args.model is not None:
        summary["rung0_model"] = json.loads(args.model.read_text())
    if args.census is not None:
        census = json.loads(args.census.read_text())
        arch = doc.get("architecture", "applegpu_g16s")
        summary["isolated_residency"] = residency(census, arch, widths)
        # The ranked M5 is `applegpu_g17s`. Nothing here was timed on it, so
        # this table transfers a cost observation only, never a measurement.
        summary["isolated_residency_ranked"] = residency(
            census, RANKED_ARCH, widths)

    print("E126 rung 1, harness=local. reference arm %s, %d shapes, widths %s"
          % (BASE_ARM, len(shapes), widths))
    print("\nPer width, percent FASTER than share_off (pooled median):")
    print("  %-22s %s" % ("arm", "".join("     NA%d" % w for w in widths)))
    for arm in ARMS[1:]:
        row = contrasts["%s_vs_share_off" % arm]["per_width_pct"]
        print("  %-22s %s" % (arm, "".join(
            "%8.3f" % row.get(w, float("nan")) for w in widths)))

    print("\nPer width, percent FASTER than share_on (the shipped base):")
    for arm in DIAG_ARMS:
        row = contrasts["%s_vs_share_on" % arm]["per_width_pct"]
        print("  %-22s %s" % (arm, "".join(
            "%8.3f" % row.get(w, float("nan")) for w in widths)))

    print("\nOverlap O = gain(share_on) / gain(n_sums_free), both vs share_off")
    print("  %-22s %s" % ("O", "".join(
        "%8.3f" % overlap.get(w, float("nan")) for w in widths)))

    print("\nReplication of thorfinn's grid, gain(n_sums_free vs share_off)")
    for key, row in sorted(replication.items()):
        print("  %-6s thorfinn %6.3f  measured %7.3f  ratio %5.2f  stop=%s"
              % (key, row["thorfinn_pct"], row["measured_pct"], row["ratio"],
                 row["stop_rule_fires"]))

    t4 = summary["task4_revision"]
    print("\nTask 4. thorfinn rung 5e re-priced on the E121 base, using the "
          "measured\nsurviving fraction of the mix-faithful arm n_sums_loaded")
    for width, row in sorted(t4["per_width"].items()):
        print("  NA%d  thorfinn %+6.2f  surviving %.3f  marginal %+6.2f  "
              "replica %5.2f  net %+6.2f"
              % (width, row["thorfinn_gain_on_share_off_pct"],
                 row["measured_surviving_fraction"], row["marginal_gain_pct"],
                 row["replica_cost_pp"], row["marginal_net_pct"]))
    print("  share_off base: round net %+.3f %%, leg %+.3f %%"
          % (t4["share_off_base_round_net_pct"], t4["share_off_base_leg_pct"]))
    print("  E121 base:      round net %+.3f %%, leg %+.3f %%, ranked %+.3f %%"
          % (t4["e121_base_round_net_pct"], t4["e121_base_leg_pct"],
             t4["e121_base_ranked_pct"]))
    print("  E121 removes %.1f %% of Route B's leg value"
          % (100.0 * t4["fraction_of_route_b_leg_value_removed_by_e121"]))

    print("\nPre-registered bands")
    scored = {
        "free_vs_off_na4_pct": free_vs_off.get(4),
        "primary_free_vs_on_na4_pct": primary.get(4),
        "e123_vs_off_na4_pct":
            contrasts["n_nosums_e123_vs_share_off"]["per_width_pct"].get(4),
        "loaded_vs_on_na4_pct":
            contrasts["n_sums_loaded_vs_share_on"]["per_width_pct"].get(4),
        "overlap_O_na4": overlap.get(4),
        "primary_round_weighted_pct": primary_rw,
    }
    for name, value in scored.items():
        if value is not None:
            print("  " + score_band(name, value))
    summary["preregistered_scored"] = scored

    print("\nPrimary round weighted %.3f %% CI95 [%.3f, %.3f] coverage %.3f"
          % (primary_rw, ci_lo, ci_hi, coverage))
    print("Predicted leg %.3f %%, ranked %.3f %%"
          % (primary_rw * LEG_TRANSFER,
             primary_rw * LEG_TRANSFER * RANKED_TRANSFER))

    print("\nBandwidth covariate. within_cell holds shape AND width fixed and "
          "is a noise floor;\nwithin_width holds width fixed and uses the five "
          "shapes, where the rate really varies.")
    for label, key in (("free vs off", "bandwidth_covariate_free_vs_off"),
                       ("free vs on", "bandwidth_covariate_free_vs_on"),
                       ("on vs off", "bandwidth_covariate_on_vs_off")):
        cov = summary[key]
        print("  %-12s within_cell median %+.5f %%/(GB/s), max |slope| %.5f"
              % (label, cov["within_cell_median_slope"],
                 cov["within_cell_max_abs_slope"]))
        for width, fit in cov["within_width"].items():
            print("               %-5s slope %+.5f %%/(GB/s)  r=%+.3f  "
                  "n=%d  rate %.0f-%.0f GB/s"
                  % (width, fit["slope_pct_per_gbs"], fit["r"], fit["n"],
                     fit["gbs_min"], fit["gbs_max"]))

    print("\nAchieved GB/s per arm per cell, pooled over %d surviving blocks"
          % (max(c["blocks"] for c in summary["per_arm_bandwidth"].values())))
    print("  %-34s %s" % ("cell", "".join("%14s" % a for a in ARMS)))
    for key, cell in summary["per_arm_bandwidth"].items():
        print("  %-34s %s" % (key, "".join(
            "%14.1f" % cell["gbs"][a] if a in cell["gbs"] else "%14s" % "?"
            for a in ARMS)))

    sp = summary["slot_position"]
    print("\nHarness defect 30. %d slots per block, %s order, so arm 0 holds "
          "the two\nextreme slots and %s holds the two middle ones. "
          "Palindrome asymmetry\nslot[n-1-i] minus slot[i] cancels the arm "
          "effect and exposes monotone drift."
          % (sp["slots_per_block"], sp["order"], ARMS[-1]))
    print("  %-16s %s" % ("arm", "".join("     NA%d" % w for w in widths)))
    for arm in ARMS:
        print("  %-16s %s" % (arm, "".join(
            "%+8.3f" % sp["per_width"][w]["palindrome_asymmetry"][arm]
            ["median_pp"] for w in widths)))
    print("  max |median asymmetry| %.3f pp over all arms and widths"
          % sp["max_abs_median_asymmetry_pp"])
    folded = contrasts["share_on_vs_share_off"]["per_width_pct"].get(5)
    if folded is not None:
        print("  the convex part is identified only at NA=5, where the "
              "shipped gate folds\n  off and share_on is byte-identical to "
              "share_off in the opposite slot pair:\n  extreme minus middle "
              "= %+.3f pp" % -folded)

    if "isolated_residency" in summary:
        for key, label in (("isolated_residency", "timed"),
                           ("isolated_residency_ranked", "ranked, not timed")):
            res = summary[key]
            print("\nIsolated-kernel residency on %s (%s), registers / "
                  "resident simdgroups" % (res["architecture"], label))
            print("  %-16s %s" % ("arm", "".join("      NA%d" % w
                                                 for w in widths)))
            for arm in ARMS:
                row = res["isolated_per_arm"].get(arm, {})
                print("  %-16s %s" % (arm, "".join(
                    "%4d/%-4d" % (row[w]["registers"], row[w]["simdgroups"])
                    if w in row else "     ?  " for w in widths)))
        res = summary["isolated_residency"]
        bad = res["occupancy_confounded_contrasts"]
        print("  %d contrast cells differ in residency and are therefore "
              "occupancy confounded:" % len(bad))
        for row in bad:
            print("    %-32s NA%d  %d vs %d simdgroups (%.3fx)"
                  % (row["contrast"], row["width"], row["arm_simdgroups"],
                     row["ref_simdgroups"], row["residency_ratio"]))

    print("\nThermal, per width. cool_gate_passed_real_gate=%s "
          "gate_qualified_for_timing=%s"
          % (summary["thermal"].get("cool_gate_passed_real_gate"),
             summary["thermal"].get("gate_qualified_for_timing")))
    for width, row in summary["thermal_per_width"].items():
        print("  %-6s cells %d  entry %.2f-%.2f C (spread %.2f)  exit max %.2f"
              % (width, row["cells"], row["entry_min_c"], row["entry_max_c"],
                 row["entry_spread_c"], row["exit_max_c"]))

    print("\nvoid=%s  reasons=%s" % (summary["validity"]["void"],
                                     summary["validity"]["reasons"]))
    print("exactness_failures=%s  checks=%d  positive_controls=%d"
          % (failures, checked, len(controls)))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True)
                            + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
