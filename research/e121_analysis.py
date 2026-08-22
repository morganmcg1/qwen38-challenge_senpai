#!/usr/bin/env python3
"""E121 rung 2: price the gated cross-simdgroup chunk-sum share.

    research/e121_analysis.py research/out/e121-rung2/rate.json \
        --census research/e121-artifacts/rung0-census.json \
        --out research/e121-artifacts/rung2-summary.json

Sign convention, which is the advisor's and not E110's: a POSITIVE percentage
means the arm is FASTER than `a_base`. The estimator is the paired per-block
ratio inside one counterbalanced palindrome block, so it cancels the drift the
palindrome cannot, and its spread across blocks is the instrument's own noise.
The first block of every cell is discarded as a cold start, which is the E110
protocol: a naive mean over all blocks moved `b_barrier` by five percentage
points in that session.

The headline is round-weighted over the realised verify-width histogram, and it
is reported twice: over all widths, and with NA=5 excluded and its coverage
stated, because the shared path is gated off at NA=5 by `if constexpr`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics

# Share of the streaming term spent in a one-group pass of each width.
ROUND_WEIGHTS = {2: 0.024, 3: 0.275, 4: 0.667, 5: 0.034}

# E116's measured kernel-to-leg transfer, then rule 34's local-to-ranked factor.
LEG_TRANSFER = 0.6070
RANKED_TRANSFER = 0.95

# Campaign Rule 59's submission bar, on the predicted ranked effect.
SUBMISSION_BAR_PCT = 0.50

# Pre-registered before timing, from E118's price ladder. ALU class only; the
# ladder has no threadgroup and no barrier class, so the exchange is UNPRICED.
LADDER_ALU_SLOPE = {2: 0.0043, 3: 0.0676, 4: 0.0940, 5: 0.0793}
PREREGISTERED = {
    "ladder_alu_only_na4_pct": 3.76,
    "na4_band_if_branch": (1.3, 1.6),
    "na4_band_if_predicated": (0.0, 0.3),
    "round_weighted_band": (1.0, 1.2),
    "exchange_cost_na4_pp": 0.801,
    "ceiling_na4_pct": 2.266,
}


def collect(doc: dict, warmup_blocks: int) -> dict:
    cells: dict[tuple[str, int], dict] = {}
    for row in doc["measurements"]:
        if row["kind"] != "timing" or row["block"] < warmup_blocks:
            continue
        cell = cells.setdefault((row["shape"], row["m"]), {
            "temp": [], "blocks": []})
        cell["temp"].append(row["gpu_temp_entry_c"])
        cell["blocks"].append(row["seconds"])
    return cells


def gains(cell: dict, arm: str, ref: str = "a_base") -> list[float]:
    """Per-block percent speedup of `arm` over `ref`; positive means faster."""
    return [100.0 * (1.0 - b[arm] / b[ref])
            for b in cell["blocks"] if b.get(ref) and b.get(arm)]


def med(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def ladder(cells: dict, shapes: list[str], widths: list[int],
           arm: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for width in widths:
        pooled: list[float] = []
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is not None:
                pooled.extend(gains(cell, arm))
        if pooled:
            out[width] = pooled
    return out


def weighted(per_width: dict[int, float], drop: tuple[int, ...] = ()) -> tuple:
    """Round-weighted percent and the share of round weight it covers."""
    use = {w: v for w, v in per_width.items()
           if w in ROUND_WEIGHTS and w not in drop}
    if not use:
        return float("nan"), 0.0
    coverage = sum(ROUND_WEIGHTS[w] for w in use)
    total = sum(ROUND_WEIGHTS[w] for w in ROUND_WEIGHTS)
    value = sum(ROUND_WEIGHTS[w] * v for w, v in use.items()) / coverage
    return value, coverage / total


def bootstrap(per_width_samples: dict[int, list[float]], drop: tuple = (),
              draws: int = 4000, seed: int = 121) -> tuple[float, float]:
    """Percentile CI on the round-weighted number, resampling blocks."""
    rng = random.Random(seed)
    use = {w: v for w, v in per_width_samples.items()
           if w in ROUND_WEIGHTS and w not in drop and v}
    if not use:
        return float("nan"), float("nan")
    coverage = sum(ROUND_WEIGHTS[w] for w in use)
    out = []
    for _ in range(draws):
        acc = 0.0
        for width, values in use.items():
            pick = [values[rng.randrange(len(values))] for _ in values]
            acc += ROUND_WEIGHTS[width] * med(pick)
        out.append(acc / coverage)
    out.sort()
    return out[int(0.025 * draws)], out[int(0.975 * draws)]


def fidelity(doc: dict) -> tuple[list[str], list[str], int]:
    failures, controls, checked = [], [], 0
    for row in doc["measurements"]:
        if row["kind"] == "fidelity":
            for arm in row["arms"]:
                checked += 1
                if arm["exact_required"] and not arm["bit_identical"]:
                    failures.append("%s M=%d %s: %d/%d differ" % (
                        row["shape"], row["m"], arm["arm"], arm["differing"],
                        arm["total"]))
        elif row["kind"] == "positive_control":
            controls.append("%s M=%d %s: %d/%d differ, detected=%s" % (
                row["shape"], row["m"], row["arm"], row["differing"],
                row["total"], row["detected"]))
    return failures, controls, checked


def thermal(doc: dict) -> dict:
    entry = [r["gpu_temp_entry_c"] for r in doc["measurements"]
             if r["kind"] == "thermal"]
    exits = [r["gpu_temp_exit_c"] for r in doc["measurements"]
             if r["kind"] == "thermal"]
    if not entry:
        return {}
    return {"entry_min_c": min(entry), "entry_max_c": max(entry),
            "entry_spread_c": max(entry) - min(entry),
            "exit_min_c": min(exits), "exit_max_c": max(exits),
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rate", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--warmup-blocks", type=int, default=1)
    args = ap.parse_args()

    doc = json.loads(args.rate.read_text())
    cells = collect(doc, args.warmup_blocks)
    shapes = sorted({s for s, _ in cells})
    widths = sorted({w for _, w in cells})
    arms = [a for a in doc["arms"] if a != "a_base"]

    print("=== E121 rung 2, harness=local, ungated by design ===")
    print("shapes %d  widths %s  arms %d  blocks kept %d of %d, "
          "first %d discarded"
          % (len(shapes), widths, len(doc["arms"]),
             doc["pairs"] - args.warmup_blocks, doc["pairs"],
             args.warmup_blocks))
    print("positive percent means FASTER than a_base")
    print()

    failures, controls, checked = fidelity(doc)
    print("=== exactness, every cell ===")
    print("  cells checked: %d" % checked)
    print("  exactness failures on arms required to be exact: %d"
          % len(failures))
    for line in failures:
        print("    FAIL %s" % line)
    for line in controls:
        print("  positive control %s" % line)
    undetected = [c for c in controls if "detected=False" in c]
    if undetected:
        print("  *** POSITIVE CONTROL DID NOT FIRE, session is void ***")
    print()

    heat = thermal(doc)
    if heat:
        print("=== thermal ===")
        print("  entry C: min %.1f max %.1f spread %.1f   exit C: min %.1f "
              "max %.1f" % (heat["entry_min_c"], heat["entry_max_c"],
                            heat["entry_spread_c"], heat["exit_min_c"],
                            heat["exit_max_c"]))
        print("  cool_gate_passed_real_gate=false  "
              "gate_qualified_for_timing=false")
        print()

    summary = {"shapes": shapes, "widths": widths, "arms": {},
               "thermal": heat, "exactness_failures": failures,
               "positive_controls": controls,
               "preregistered": PREREGISTERED,
               "warmup_blocks_discarded": args.warmup_blocks}

    print("=== per width, paired median percent faster than a_base ===")
    header = "  %-17s" % "arm" + "".join("   NA%d" % w for w in widths)
    print(header + "     all    ex-NA5  cover")
    for arm in arms:
        samples = ladder(cells, shapes, widths, arm)
        per_width = {w: med(v) for w, v in samples.items()}
        allw, cov_all = weighted(per_width)
        ex5, cov_ex5 = weighted(per_width, drop=(5,))
        lo, hi = bootstrap(samples, drop=(5,))
        row = "  %-17s" % arm
        row += "".join("  %+.3f" % per_width.get(w, float("nan"))
                       for w in widths)
        row += "   %+.3f  %+.3f  %.3f" % (allw, ex5, cov_ex5)
        print(row)
        summary["arms"][arm] = {
            "per_width_pct": per_width,
            "round_weighted_pct": allw,
            "round_weighted_ex_na5_pct": ex5,
            "coverage_ex_na5": cov_ex5,
            "ci95_ex_na5": [lo, hi],
            "blocks_per_width": {w: len(v) for w, v in samples.items()},
        }
    print()

    print("=== headline cells and the shipped frame ===")
    for arm in arms:
        info = summary["arms"][arm]
        na4 = info["per_width_pct"].get(4, float("nan"))
        ex5 = info["round_weighted_ex_na5_pct"]
        lo, hi = info["ci95_ex_na5"]
        leg = ex5 * info["coverage_ex_na5"] * LEG_TRANSFER
        ranked = leg * RANKED_TRANSFER
        info["predicted_leg_pct"] = leg
        info["predicted_ranked_pct"] = ranked
        info["clears_submission_bar"] = ranked >= SUBMISSION_BAR_PCT
        print("  %-17s NA4 %+.3f   ex-NA5 %+.3f [%+.3f, %+.3f]   "
              "leg %+.3f   ranked %+.3f%s"
              % (arm, na4, ex5, lo, hi, leg, ranked,
                 "  CLEARS RULE 59" if ranked >= SUBMISSION_BAR_PCT else ""))
    print()

    print("=== pre-registered predictions, written before timing ===")
    print("  ladder ALU-only at NA4: %+.2f %%  (predicted to over-predict)"
          % PREREGISTERED["ladder_alu_only_na4_pct"])
    print("  branch fork:     NA4 in [%.1f, %.1f] if issue is skipped"
          % PREREGISTERED["na4_band_if_branch"])
    print("  predication fork: NA4 in [%.1f, %.1f] if the add still issues"
          % PREREGISTERED["na4_band_if_predicated"])
    print("  round-weighted:  [%.1f, %.1f] %%"
          % PREREGISTERED["round_weighted_band"])
    for arm in ("g_split_pred", "g_min_ask"):
        if arm not in summary["arms"]:
            continue
        na4 = summary["arms"][arm]["per_width_pct"].get(4)
        if na4 is None:
            continue
        gap = PREREGISTERED["ladder_alu_only_na4_pct"] - na4
        print("  %-14s NA4 measured %+.3f %%, ladder over-predicts by %+.3f pp"
              % (arm, na4, gap))
    print()

    if args.census and args.census.exists():
        cen = json.loads(args.census.read_text())["arms"]
        print("=== rung 0 census, carried forward ===")
        for arm in ["a_base"] + arms:
            if arm not in cen:
                continue
            entry = cen[arm].get("entry", {})
            line = "  %-17s" % arm
            for arch in ("applegpu_g16s", "applegpu_g17s"):
                got = entry.get(arch)
                if got:
                    line += "  %s R=%d%s sg=%d" % (
                        arch[-4:], got["registers"],
                        "s%d" % got["spill_bytes"] if got["spill_bytes"]
                        else "", got["simdgroups"])
            print(line)
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("wrote %s" % args.out)

    return 1 if failures or undetected else 0


if __name__ == "__main__":
    raise SystemExit(main())
