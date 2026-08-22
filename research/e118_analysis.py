#!/usr/bin/env python3
"""E118: reduce the metadata-load arms, and name the binding resource.

    research/e118_analysis.py --rate research/out/e118-full/rate.json \
        --census research/e118-artifacts/census.json \
        --out research/e118-artifacts/summary.json

The identified-set columns need four board facts per prompt. CAMPAIGN RULE 40
forbids an analysis script reading a gitignored host-local path for a number
that reaches the report, so those facts are extracted ONCE into a committed
slice and only the slice is read afterwards:

    research/e118_analysis.py --extract-receipt /tmp/yukon-board/full.json \
        --receipt b8b8b860 --slice research/e118-artifacts/e114_receipt_slice.json

Sign convention throughout: a POSITIVE percentage means the arm is FASTER than
`a_base`. E114's arm tables use the opposite sign, so the two are never mixed in
one column here.

harness=local everywhere. Every number below comes from a standalone probe that
compiles its own copy of the kernel. Finding 28 says the `quantized` metallib is
dead for the scored worker and only the worker binary carries the arm, so
nothing here is an end-to-end measurement of the shipped path.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scoring_weights as sw  # noqa: E402

HEADLINE_SHAPE = "mlp_gate_up_k5120_n34816"
WARMUP_BLOCKS = 1
KILL_RULE_PCT = 0.5
# From the assignment: the arms whose mechanism the primary metric may rank.
# `p_prefetch_w` is bit exact but spills on g16s at the widths that matter, so
# it is reported beside them and never inside the headline.
PROMOTION_ARMS = ("s_bcast", "s_bcast_all", "s_bcast_scale", "p_split_meta",
                  "g_pack32", "s_bcast_pack32")
DIAGNOSTIC_ARMS = ("n_nosums", "l_loadonly")


# --- the committed receipt slice ----------------------------------------------

def extract_receipt(board: str, prefix: str, out: pathlib.Path) -> int:
    """Write the four board facts per prompt that the identified set needs."""
    import e114_width_recovery as wr

    rec = wr.load_receipt(board, prefix)
    keep = ("mean_width", "p_width1", "rounds", "round_us", "mtp_us_per_token",
            "raw", "mean_draft_len", "zero_draft_rounds")
    slim = {"id": rec["id"], "status": rec["status"],
            "published": rec["published"], "source_board": board,
            "prompts": {name: {k: p[k] for k in keep}
                        for name, p in rec["prompts"].items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim, indent=2, sort_keys=True) + "\n")
    print("wrote %s for receipt %s" % (out, rec["id"]))
    return 0


# --- reduction ----------------------------------------------------------------

def paired_pct(rate: dict) -> dict:
    """Per (shape, NA, arm) percent faster than `a_base`, block by block."""
    arms = rate["arms"]
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rate["measurements"]:
        if row.get("kind") != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        key = (row["shape"], row["m"])
        seconds = row["seconds"]
        bucket = cells.setdefault(key, {a: [] for a in arms})
        base = seconds[arms[0]]
        for arm in arms:
            bucket[arm].append(100.0 * (base - seconds[arm]) / base)
    return cells


def absolute_us(rate: dict) -> dict:
    """Per (shape, NA, arm) median absolute microseconds, same blocks."""
    arms = rate["arms"]
    cells: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rate["measurements"]:
        if row.get("kind") != "timing" or row["block"] < WARMUP_BLOCKS:
            continue
        bucket = cells.setdefault((row["shape"], row["m"]), {a: [] for a in arms})
        for arm in arms:
            bucket[arm].append(row["seconds"][arm] * 1e6)
    return {k: {a: statistics.median(v) for a, v in b.items()}
            for k, b in cells.items()}


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "median": statistics.median(values), "mean": mean,
            "sd": sd, "sem": sd / math.sqrt(n) if n > 1 else 0.0,
            "min": min(values), "max": max(values)}


def forward_reverse_gap(rate: dict) -> dict:
    """Harness defect 16 residual: slot `a` against slot `2N - 1 - a`.

    A fixed cost paid by the first timed slot of a block does not cancel in the
    palindrome mean, so it survives as a positive forward-minus-reverse gap on
    arm 0 and on nothing else. E115 saw +61.6 % on arm 0 with every other arm
    under 0.4 %. Both the all-block and the post-warm-up figure are reported,
    because the first is what the fix has to remove.
    """
    arms = rate["arms"]
    n = len(arms)
    allb: dict[str, list[float]] = {a: [] for a in arms}
    kept: dict[str, list[float]] = {a: [] for a in arms}
    for row in rate["measurements"]:
        if row.get("kind") != "timing":
            continue
        slots = row["slots"]
        for i, arm in enumerate(arms):
            fwd, rev = slots[i], slots[2 * n - 1 - i]
            gap = 100.0 * (fwd - rev) / rev
            allb[arm].append(gap)
            if row["block"] >= WARMUP_BLOCKS:
                kept[arm].append(gap)
    return {arm: {"all_blocks_median_pct": statistics.median(allb[arm]),
                  "all_blocks_max_abs_pct": max(abs(v) for v in allb[arm]),
                  "post_warmup_median_pct": statistics.median(kept[arm]),
                  "post_warmup_max_abs_pct": max(abs(v) for v in kept[arm])}
            for arm in arms}


def fidelity_rows(rate: dict) -> dict:
    exact_failures, control_failures, diag_seen = [], [], []
    for row in rate["measurements"]:
        if row.get("kind") == "fidelity":
            for entry in row["arms"]:
                if entry["exact_required"] and not entry["bit_identical"]:
                    exact_failures.append(
                        {"shape": row["shape"], "m": row["m"], **entry})
                if not entry["exact_required"]:
                    diag_seen.append(entry["arm"])
        elif row.get("kind") == "positive_control" and not row["detected"]:
            control_failures.append(row)
    return {"exact_failures": exact_failures,
            "control_failures": control_failures,
            "diagnostic_arms_seen": sorted(set(diag_seen))}


# --- weighting ----------------------------------------------------------------

def identified_range(delta: dict[int, float], slice_path: pathlib.Path,
                     rates: dict) -> tuple[float, float]:
    """Exact extremum of the published-weighted arm value over E114's set."""
    import e114_rerank as rr

    rec = json.loads(slice_path.read_text())
    rec["prompts"] = {k: dict(v) for k, v in rec["prompts"].items()}
    mix = rr.prompt_mix(rec)
    policy, _ = rr.load_policy_shapes("research/e114-artifacts/rung1b.json")
    vsets, _ = rr.build(rec, rates, True, policy)
    lo = rr.arm_range(delta, vsets, mix, rec, rates, hi=False)
    hi = rr.arm_range(delta, vsets, mix, rec, rates, hi=True)
    return lo, hi


def point_shapes(delta: dict[int, float]) -> dict[str, float]:
    """The four E114 candidate shapes. Every one FAILED its own rung-0 gate."""
    table = json.loads(
        pathlib.Path("research/e114-artifacts/rung1.json").read_text())
    out = {}
    for shape, shift in table["delta_weights"].items():
        weights = {na: sw.STANDING_WEIGHTS[na] + shift[str(na)]
                   for na in sw.NA_CELLS}
        out[shape] = sw.weighted(delta, weights)
    return out


# --- report -------------------------------------------------------------------

def report(rate_path: pathlib.Path, census_path: pathlib.Path | None,
           slice_path: pathlib.Path | None, out_path: pathlib.Path | None,
           shape: str) -> int:
    rate = json.loads(rate_path.read_text())
    arms = rate["arms"]
    cells = paired_pct(rate)
    us = absolute_us(rate)
    widths = sorted({m for (s, m) in cells if s == shape})
    shapes = sorted({s for (s, _) in cells})

    print("=" * 96)
    print("E118 - the metadata-load instruction axis of the wide affine-4 QMV")
    print("=" * 96)
    print("harness=local   device %s   architecture %s" %
          (rate["device"], rate["architecture"]))
    print("standalone probe, own kernel copy: NOT an end-to-end measurement")
    print("blocks per cell %d, first %d discarded, palindrome order, "
          "cool_gate_passed_real_gate=false, gate_qualified_for_timing=false"
          % (rate["pairs"], WARMUP_BLOCKS))
    print("sign convention: POSITIVE percent means FASTER than a_base")

    fid = fidelity_rows(rate)
    print("\n-- fidelity")
    print("   exact-arm failures  : %d" % len(fid["exact_failures"]))
    print("   positive-control failures: %d" % len(fid["control_failures"]))
    print("   diagnostic arms (difference expected): %s"
          % ", ".join(fid["diagnostic_arms_seen"]))

    gaps = forward_reverse_gap(rate)
    print("\n-- harness defect 16 residual, forward slot against reverse slot")
    print("   %-16s %12s %12s %12s %12s"
          % ("arm", "all med %", "all |max| %", "kept med %", "kept |max| %"))
    for arm in arms:
        g = gaps[arm]
        print("   %-16s %12.3f %12.3f %12.3f %12.3f"
              % (arm, g["all_blocks_median_pct"], g["all_blocks_max_abs_pct"],
                 g["post_warmup_median_pct"], g["post_warmup_max_abs_pct"]))

    print("\n-- %s, percent faster than a_base, median over kept blocks "
          "(sem in brackets)" % shape)
    header = "   %-16s" % "arm" + "".join("  %18s" % ("NA%d" % m)
                                          for m in widths)
    print(header)
    per_arm_na: dict[str, dict[int, float]] = {}
    stats: dict[str, dict[int, dict]] = {}
    for arm in arms:
        line = "   %-16s" % arm
        per_arm_na[arm], stats[arm] = {}, {}
        for m in widths:
            st = summarise(cells[(shape, m)][arm])
            per_arm_na[arm][m] = st["median"]
            stats[arm][m] = st
            line += "  %+10.3f (%.3f)" % (st["median"], st["sem"])
        print(line)

    print("\n-- absolute microseconds, %s, median over kept blocks" % shape)
    print("   %-16s" % "arm" + "".join("  %10s" % ("NA%d" % m) for m in widths))
    for arm in arms:
        print("   %-16s" % arm
              + "".join("  %10.1f" % us[(shape, m)][arm] for m in widths))

    # --- Finding 44 placement -------------------------------------------------
    print("\n-- Finding 44 placement on %s: a_base against its own load "
          "ceiling" % shape)
    print("   %4s %12s %12s %10s" % ("NA", "a_base us", "l_loadonly us",
                                     "gap %"))
    f44 = {}
    for m in widths:
        base_us, load_us = us[(shape, m)]["a_base"], us[(shape, m)]["l_loadonly"]
        gap = 100.0 * (base_us - load_us) / load_us
        f44[m] = {"a_base_us": base_us, "l_loadonly_us": load_us,
                  "gap_pct": gap}
        print("   %4d %12.1f %12.1f %10.2f" % (m, base_us, load_us, gap))
    weights = {na: sw.STANDING_WEIGHTS[na] for na in sw.NA_CELLS}
    if set(widths) == set(sw.NA_CELLS):
        f44_weighted = sw.weighted({m: f44[m]["gap_pct"] for m in widths},
                                   weights)
        print("   round weighted gap: %+.2f %%" % f44_weighted)
    else:
        f44_weighted = float("nan")

    # --- the headline ---------------------------------------------------------
    print("\n-- round-weighted percent faster than a_base, %s, standing "
          "weights %s" % (shape, sw.STANDING_WEIGHTS))
    rows = {}
    for arm in arms:
        if arm == "a_base" or set(widths) != set(sw.NA_CELLS):
            continue
        value = sw.weighted(per_arm_na[arm], weights)
        row = {"standing_pct": value, "na": per_arm_na[arm],
               "role": ("diagnostic" if arm in DIAGNOSTIC_ARMS
                        else "promotion" if arm in PROMOTION_ARMS
                        else "other"),
               "points": point_shapes(per_arm_na[arm])}
        if slice_path is not None:
            lo, hi = identified_range(per_arm_na[arm], slice_path,
                                      sw.ONE_GROUP_GBPS)
            rlo, rhi = identified_range(per_arm_na[arm], slice_path,
                                        sw.RANKED_ONE_GROUP_GBPS)
            row["identified_local"] = [lo, hi]
            row["identified_ranked"] = [rlo, rhi]
        rows[arm] = row
    order = sorted(rows, key=lambda a: -rows[a]["standing_pct"])
    print("   %-16s %10s %10s %22s %22s"
          % ("arm", "role", "standing", "identified local", "identified ranked"))
    for arm in order:
        r = rows[arm]
        loc = ("[%+7.3f, %+7.3f]" % tuple(r["identified_local"])
               if "identified_local" in r else "n/a")
        rnk = ("[%+7.3f, %+7.3f]" % tuple(r["identified_ranked"])
               if "identified_ranked" in r else "n/a")
        print("   %-16s %10s %+10.3f %22s %22s"
              % (arm, r["role"][:10], r["standing_pct"], loc, rnk))

    print("\n   the four point shapes are DIAGNOSTIC: every one failed E114's "
          "own rung-0 gate")
    print("   %-16s %10s %10s %10s %10s" % ("arm", "maxent", "gt1", "gt2",
                                            "policy"))
    for arm in order:
        p = rows[arm]["points"]
        print("   %-16s %+10.3f %+10.3f %+10.3f %+10.3f"
              % (arm, p["maxent"], p["gt1"], p["gt2"], p["policy"]))

    best, best_value = None, float("-inf")
    for arm in PROMOTION_ARMS:
        if arm in rows and rows[arm]["standing_pct"] > best_value:
            best, best_value = arm, rows[arm]["standing_pct"]
    print("\n   PRIMARY METRIC e118_best_bit_exact_arm_round_weighted_pct_"
          "faster_vs_a_base = %+.4f  (%s)" % (best_value, best))
    print("   kill rule %+.2f %% -> %s"
          % (KILL_RULE_PCT,
             "CLEARED" if best_value >= KILL_RULE_PCT else "NOT CLEARED, null"))

    # --- every other shape ----------------------------------------------------
    print("\n-- every shape, round-weighted percent faster than a_base, "
          "standing weights")
    print("   %-16s" % "arm" + "".join("  %26s" % s for s in shapes))
    per_shape = {}
    for arm in arms:
        if arm == "a_base":
            continue
        line = "   %-16s" % arm
        per_shape[arm] = {}
        for s in shapes:
            ws = sorted({m for (ss, m) in cells if ss == s})
            if set(ws) != set(sw.NA_CELLS):
                line += "  %26s" % "-"
                continue
            table = {m: statistics.median(cells[(s, m)][arm]) for m in ws}
            value = sw.weighted(table, weights)
            per_shape[arm][s] = {"weighted_pct": value, "na": table}
            line += "  %+26.3f" % value
        print(line)

    # --- the discriminator ----------------------------------------------------
    print("\n-- discriminator")
    s_b = rows.get("s_bcast", {}).get("standing_pct", float("nan"))
    s_ba = rows.get("s_bcast_all", {}).get("standing_pct", float("nan"))
    p_sm = rows.get("p_split_meta", {}).get("standing_pct", float("nan"))
    n_ns = rows.get("n_nosums", {}).get("standing_pct", float("nan"))
    bar = KILL_RULE_PCT
    if s_b >= bar and p_sm < bar:
        verdict = "load-issue port"
    elif s_b >= bar and p_sm >= bar:
        verdict = "memory latency"
    elif n_ns >= bar and s_b < bar and p_sm < bar:
        verdict = "total instruction issue or ALU"
    else:
        verdict = "no arm cleared the bar; the probe does not select a resource"
    print("   s_bcast %+.3f   s_bcast_all %+.3f   p_split_meta %+.3f   "
          "n_nosums %+.3f" % (s_b, s_ba, p_sm, n_ns))
    print("   binding resource selected by the data: %s" % verdict)

    payload = {
        "harness": "local",
        "device": rate["device"], "architecture": rate["architecture"],
        "shape": shape, "widths": widths, "warmup_blocks": WARMUP_BLOCKS,
        "pairs": rate["pairs"], "ramp_ms": rate.get("ramp_ms"),
        "sign_convention": "positive percent means FASTER than a_base",
        "standing_weights": sw.STANDING_WEIGHTS,
        "fidelity": fid, "forward_reverse_gap_pct": gaps,
        "per_arm_na_pct": per_arm_na,
        "per_arm_na_stats": stats,
        "absolute_us": {"%s|NA%d" % k: v for k, v in us.items()},
        "finding44": {"per_na": f44, "round_weighted_gap_pct": f44_weighted},
        "weighted": rows, "per_shape": per_shape,
        "primary_metric": {
            "name": "e118_best_bit_exact_arm_round_weighted_pct_faster_vs_"
                    "a_base",
            "value": best_value, "arm": best, "kill_rule_pct": KILL_RULE_PCT,
            "cleared": bool(best_value >= KILL_RULE_PCT)},
        "discriminator": {"s_bcast": s_b, "s_bcast_all": s_ba,
                          "p_split_meta": p_sm, "n_nosums": n_ns,
                          "verdict": verdict},
    }
    if census_path is not None and census_path.exists():
        payload["census"] = json.loads(census_path.read_text())
        print_census(payload["census"])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % out_path)
    return 0


def print_census(census: dict) -> None:
    widths = census["widths"]
    print("\n-- AIR device loads per entry point (and simd_shuffle calls)")
    print("   %-16s %s" % ("arm", "  ".join("%9s" % ("NA%d" % na)
                                            for na in widths)))
    for arm, row in census["arms"].items():
        cells = []
        for na in widths:
            c = row["air"].get(str(na), {})
            sh = c.get("shuffles", 0)
            cells.append("%9s" % ("%s+%dsh" % (c.get("device_loads", "?"), sh)
                                  if sh else str(c.get("device_loads", "?"))))
        print("   %-16s %s" % (arm, "  ".join(cells)))
    for arch in (census["local_arch"], census["ranked_arch"]):
        print("\n-- %s registers / spill bytes / machine text bytes" % arch)
        for arm, row in census["arms"].items():
            cells = []
            for na in widths:
                v = row.get(arch, {}).get(str(na))
                if v is None:
                    cells.append("NA%d=?" % na)
                    continue
                spill = v["spill_bytes"] or 0
                cells.append("NA%d=%s%s/%s" % (na, v["registers"],
                                               "s%d" % spill if spill else "",
                                               v["text_bytes"]))
            print("   %-16s %s" % (arm, "  ".join(cells)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=pathlib.Path)
    ap.add_argument("--census", type=pathlib.Path)
    ap.add_argument("--slice", type=pathlib.Path,
                    default=pathlib.Path(
                        "research/e118-artifacts/e114_receipt_slice.json"))
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--shape", default=HEADLINE_SHAPE)
    ap.add_argument("--extract-receipt")
    ap.add_argument("--receipt", default="b8b8b860")
    args = ap.parse_args()

    if args.extract_receipt:
        return extract_receipt(args.extract_receipt, args.receipt, args.slice)
    if args.rate is None:
        ap.error("--rate is required unless --extract-receipt is given")
    slice_path = args.slice if args.slice and args.slice.exists() else None
    return report(args.rate, args.census, slice_path, args.out, args.shape)


if __name__ == "__main__":
    raise SystemExit(main())
