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

# Dispatches of each probe shape in one decode round, from E71's `FAMILY_SHAPE`,
# which reconciles against the ledger's 14,412,349,440 B of quantized weight.
# `fa_o_proj` shares `gdn_out_proj`'s shape exactly, so its 16 calls join that
# row. `lm_head` (k=5120, n=248320, one call) has no probe shape; the nearest
# measured shape is `mlp_gate_up`, same k and also large n, so it is carried as
# 248320/34816 equivalent calls of that shape.
FAMILY_CALLS = {
    "mlp_gate_up_k5120_n34816": 64 + 248320 / 34816,
    "mlp_down_k17408_n5120": 64,
    "gdn_out_proj_k6144_n5120": 48 + 16,
    "gdn_in_proj_k5120_n16480": 48,
    "fa_qkv_k5120_n14336": 16,
}

# M4 Pro DRAM peak. Every shape here streams tens of megabytes of weights once,
# far past any cache, so an implied rate above this is not a fast kernel: it is
# a dispatch that did no work. A faulted command buffer retires immediately and
# reports microseconds, which is how a silent out-of-bounds write in the probe's
# own entry point looked like a 23 TB/s result.
DRAM_PEAK_GBS = 273.0
IMPLAUSIBLE_GBS = 1.2 * DRAM_PEAK_GBS

# `a_scaffold` is byte-identical machine text to `a_base` on both architectures
# at every width, proved by digest in the rung-0 census. It therefore has to
# read zero. A larger reading means the estimator, not the arm, is moving.
CONTROL_ARM = "a_scaffold"
CONTROL_TOLERANCE_PCT = 0.50

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


def cost_ladder(cells: dict, shapes: list[str], widths: list[int],
                arm: str) -> tuple[dict[int, float], dict]:
    """Per width, the shape effects combined by their share of round cost.

    Pooling blocks across shapes weights every shape equally, which is what
    E118 reported and so is the frame the +1.0 % bar was set in. It is not the
    shipped frame: `program.md` requires nonuniform per-cell effects to be
    weighted by current-tree cost before they are summed. One dispatch of a
    shape costs its measured `a_base` time, and a round issues `FAMILY_CALLS`
    of them, so that product is the weight.
    """
    per_width: dict[int, float] = {}
    share: dict = {}
    for width in widths:
        num = den = 0.0
        for shape in shapes:
            cell = cells.get((shape, width))
            if cell is None or shape not in FAMILY_CALLS:
                continue
            base = med([b["a_base"] for b in cell["blocks"] if b.get("a_base")])
            weight = FAMILY_CALLS[shape] * base
            num += weight * med(gains(cell, arm))
            den += weight
            share.setdefault(width, {})[shape] = weight
        if den:
            per_width[width] = num / den
            share[width] = {s: w / den for s, w in share[width].items()}
    return per_width, share


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


def validity(doc: dict, cells: dict, shapes: list[str], widths: list[int],
             controls: list[str]) -> dict:
    """Three independent ways this instrument can lie, checked before reading it.

    The first attempt at this rung passed exit 0 and produced a clean-looking
    table that was entirely fictional: a bounds bug wrote past the operand
    buffers, faulted the command buffer, and every later dispatch retired
    without running. These checks turn that class of failure into a loud void
    instead of a plausible number.
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

    ctrl = ladder(cells, shapes, widths, CONTROL_ARM)
    ctrl_width = {w: med(v) for w, v in ctrl.items()}
    ctrl_off = {w: v for w, v in ctrl_width.items()
                if abs(v) > CONTROL_TOLERANCE_PCT}
    if ctrl_off:
        verdicts.append("%s, which is byte-identical text, reads %s"
                        % (CONTROL_ARM,
                           ", ".join("NA%d %+.3f %%" % (w, v)
                                     for w, v in sorted(ctrl_off.items()))))

    undetected = [c for c in controls if "detected=False" in c]
    if undetected:
        verdicts.append("%d of %d positive controls did not fire"
                        % (len(undetected), len(controls)))

    return {"void": bool(verdicts), "reasons": verdicts,
            "implausible_rows": fast[:20],
            "implausible_row_count": len(fast),
            "control_arm": CONTROL_ARM,
            "control_per_width_pct": ctrl_width,
            "control_tolerance_pct": CONTROL_TOLERANCE_PCT,
            "dram_peak_gbs": DRAM_PEAK_GBS}


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
    print()

    valid = validity(doc, cells, shapes, widths, controls)
    print("=== instrument validity, checked before the numbers are read ===")
    print("  implied bandwidth over %.0f GB/s: %d timing rows"
          % (IMPLAUSIBLE_GBS, valid["implausible_row_count"]))
    for line in valid["implausible_rows"]:
        print("    %s" % line)
    print("  %s per width: %s" % (CONTROL_ARM, ", ".join(
        "NA%d %+.3f" % (w, v)
        for w, v in sorted(valid["control_per_width_pct"].items()))))
    if valid["void"]:
        print("  *** SESSION IS VOID ***")
        for reason in valid["reasons"]:
            print("    %s" % reason)
    else:
        print("  all three validity checks PASS")
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
               "positive_controls": controls, "validity": valid,
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
        cost_width, cost_share = cost_ladder(cells, shapes, widths, arm)
        cost_all, _ = weighted(cost_width)
        summary["arms"][arm] = {
            "per_width_pct": per_width,
            "round_weighted_pct": allw,
            "round_weighted_ex_na5_pct": ex5,
            "coverage_ex_na5": cov_ex5,
            "ci95_ex_na5": [lo, hi],
            "cost_weighted_per_width_pct": cost_width,
            "cost_weighted_round_pct": cost_all,
            "blocks_per_width": {w: len(v) for w, v in samples.items()},
        }
        summary["cost_share"] = cost_share
    print()

    print("=== the same effects, shapes weighted by round cost ===")
    print(header + "     all   vs pooled")
    for arm in arms:
        info = summary["arms"][arm]
        cost_width = info["cost_weighted_per_width_pct"]
        cost_all = info["cost_weighted_round_pct"]
        row = "  %-17s" % arm
        row += "".join("  %+.3f" % cost_width.get(w, float("nan"))
                       for w in widths)
        row += "   %+.3f   %+.3f" % (
            cost_all, cost_all - info["round_weighted_pct"])
        print(row)
    print("  cost share at NA4: %s" % ", ".join(
        "%s %.3f" % (s.split("_k")[0], v)
        for s, v in sorted(summary["cost_share"].get(4, {}).items())))
    print()

    print("=== headline cells and the shipped frame ===")
    for arm in arms:
        info = summary["arms"][arm]
        na4 = info["per_width_pct"].get(4, float("nan"))
        ex5 = info["round_weighted_ex_na5_pct"]
        lo, hi = info["ci95_ex_na5"]
        # The gate leaves NA=5 byte-identical to the base, so the all-width
        # number already carries NA=5 at its true zero and needs no coverage
        # rescale. `ex5` is kept only to compare against E118's frame.
        kernel = info["cost_weighted_round_pct"]
        leg = kernel * LEG_TRANSFER
        ranked = leg * RANKED_TRANSFER
        clears = ranked >= SUBMISSION_BAR_PCT and not valid["void"]
        info["predicted_leg_pct"] = leg
        info["predicted_ranked_pct"] = ranked
        info["clears_submission_bar"] = clears
        print("  %-17s NA4 %+.3f   ex-NA5 %+.3f [%+.3f, %+.3f]   "
              "kernel %+.3f   leg %+.3f   ranked %+.3f%s"
              % (arm, na4, ex5, lo, hi, kernel, leg, ranked,
                 "  CLEARS RULE 59" if clears else ""))
    if valid["void"]:
        print("  none of the above may be read: the session is void")
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
        print("  shipped entry point affine_qmv_fast, and the NA=5 body")
        for arm in ["a_base"] + arms:
            if arm not in cen:
                continue
            line = "  %-17s" % arm
            for arch in ("applegpu_g16s", "applegpu_g17s"):
                got = cen[arm].get(arch, {})
                entry, na5 = got.get("entry"), got.get("5")
                if entry:
                    line += "  %s entry R=%d%s" % (
                        arch[-4:], entry["registers"],
                        "s%d" % entry["spill_bytes"] if entry["spill_bytes"]
                        else "")
                if na5:
                    line += " na5=%s" % na5["text_sha8"]
            print(line)
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("wrote %s" % args.out)

    return 1 if failures or valid["void"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
