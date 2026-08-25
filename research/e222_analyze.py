#!/usr/bin/env python3
"""Turn an E222 timed session into per-width nets and a pooled ms/round price.

    usage: research/e222_analyze.py SESSION_DIR [--phase screen] [--out FILE]

`SESSION_DIR` is a `research/out/e222/TAG` directory written by
`research/e222_session.sh`.

Method. Each unit is timed at chain lengths 1..4 inside every ABBA block, so a
per-block least-squares fit of `microseconds ~ intercept + slope * chain`
separates the device cost of one dispatch (`slope`) from the fixed host cost of
one `eval` boundary (`intercept`). The six per-block slopes are then combined
with a Student-t interval, which is what carries the block-to-block thermal
drift into the reported uncertainty instead of hiding it.

Pooling. FINDING 564's width census gives the rounds observed at each MTP
width. The `cap7` census migrates the whole cap-8 `m = 9` mass to `m = 8`,
because a cap-8 nine-row round is an eight-draft round and an eight-draft round
at cap 7 becomes a seven-draft, eight-row round.

RULE 407. Register legality is judged on `g17s`, the ranked generation. An arm
that is `g17s`-clean and `g16s`-spilling is ranked-legal, and its LOCAL time is
a one-sided lower bound on its ranked benefit, because only the local leg pays
the spill. Such a net is reported and labelled, never silently pooled as if it
were two-sided.

harness=local-microbench. No number here is a whole-leg or a ranked score.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent

BASELINE_ARM = "staged_r4"

# FINDING 564 pooled width census, cap 8: rounds observed at each MTP width
# over five prompts.
CAP8_ROUNDS = {2: 3, 3: 61, 4: 128, 5: 166, 6: 5, 7: 7, 8: 10, 9: 257}
# E224 reverts the maintained base to `segmentedVerifyDepthCap = 7`, so `m = 9`
# has no ranked exposure and its round mass reassigns to `m = 8`.
CAP7_ROUNDS = {**CAP8_ROUNDS, 8: CAP8_ROUNDS[8] + CAP8_ROUNDS[9], 9: 0}

# FINDING 577: the register budget is generation-dependent.
G16S_BUDGET, G17S_BUDGET = 96, 126
GEOMETRY = HERE / "e222-artifacts" / "e222-geometry.json"

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def fused_registers() -> dict[tuple[str, int], dict]:
    """RULE 407 register probe pinned to the EXACT timed instantiation.

    Every timed fused arm uses the `both` schedule (lazy packed load plus late
    scale/bias read) and launches with `useTable: true`, so the timed binary is
    the `_tbl` instantiation. Reading the probe here, rather than copying its
    numbers into this file, means a later knob or compiler change that moves an
    arm above its budget kills the arm in the report instead of going unnoticed.
    """
    blob = json.loads(GEOMETRY.read_text())["geometries"]
    out = {}
    for rows in (2, 4):
        for m in (6, 7, 8, 9):
            entry = blob[f"e222_fused_m{m}_r{rows}_both_tbl"]
            compile_ = entry["compile"]
            assert entry["geometry"]["lazy"] and entry["geometry"]["late_sb"]
            assert not entry["geometry"]["seq"]
            assert entry["geometry"]["m"] == m
            assert entry["geometry"]["rows"] == rows
            out[(f"fused_r{rows}", m)] = {
                "instantiation": f"e222_fused_m{m}_r{rows}_both_tbl",
                "g16s_registers": compile_["g16s"]["registers"],
                "g16s_spill_bytes": compile_["g16s"]["spill_bytes"],
                "g16s_text_sha8": compile_["g16s"]["text_sha8"],
                "g17s_registers": compile_["g17s"]["registers"],
                "g17s_spill_bytes": compile_["g17s"]["spill_bytes"],
                "g17s_text_sha8": compile_["g17s"]["text_sha8"],
                "air_fp_sequence_sha8": compile_["air"]["fp_sequence_sha8"],
            }
    return out


FUSED_REGISTERS = fused_registers()


def fit_slope(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Ordinary least squares of `y ~ intercept + slope * x`."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    return slope, my - slope * mx


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = (T95.get(len(values) - 1, 1.96) * statistics.stdev(values)
            / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def unit_fits(samples: list[dict]) -> dict[str, dict]:
    """One record per timed unit, with the six per-block slope estimates."""
    grouped: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    facts: dict[str, dict] = {}
    for row in samples:
        label = row["label"]
        grouped[label][row["block"]].append(
            (float(row["chain"]), float(row["microseconds"])))
        facts.setdefault(label, row)

    out = {}
    for label, blocks in grouped.items():
        per_block = []
        for _, points in sorted(blocks.items()):
            if len(points) >= 2:
                per_block.append(fit_slope(points))
        slopes = [s for s, _ in per_block]
        mean, lo, hi = mean_ci(slopes)
        row = facts[label]
        out[label] = {
            "label": label, "cell": row["cell"], "m": row["m"],
            "arm": row["arm"], "rows_per_simd": row["rows_per_simd"],
            "simdgroups": row["simdgroups"], "x_groups": row["x_groups"],
            "invocations_per_round": row["invocations_per_round"],
            "k": row["k"], "n": row["n"],
            "stream_bytes": row["stream_bytes"],
            "pass_stream_bytes": row["pass_stream_bytes"],
            "activation_read_bytes": row["activation_read_bytes"],
            "xsums_read_bytes": row["xsums_read_bytes"],
            "blocks": len(per_block),
            "slope_us": mean, "slope_ci95": [lo, hi],
            "slope_per_block": slopes,
            "intercept_us": statistics.fmean([i for _, i in per_block]),
            "mean_us_per_unit": statistics.fmean(
                [r["microseconds_per_unit"] for r in samples
                 if r["label"] == label]),
        }
    return out


def ranked_legal(arm: str, m: int) -> dict:
    """RULE 407: legality is a `g17s` question; `g16s` decides comparability."""
    probe = FUSED_REGISTERS.get((arm, m))
    if probe is None:
        # The shipped staged plan and its single-pass sibling are what the
        # router already emits, so their legality is the router's own.
        return {"ranked_legal": True, "two_sided": True, "register_probe": None,
                "g17s_headroom": None}
    legal = probe["g17s_spill_bytes"] == 0 and (
        probe["g17s_registers"] <= G17S_BUDGET)
    local_clean = probe["g16s_spill_bytes"] == 0 and (
        probe["g16s_registers"] <= G16S_BUDGET)
    return {
        "ranked_legal": legal,
        "two_sided": legal and local_clean,
        "g17s_headroom": G17S_BUDGET - probe["g17s_registers"],
        "register_probe": probe,
    }


def width_nets(fits: dict[str, dict], widths: list[int],
               arms: list[str]) -> list[dict]:
    """Per-width net ms/round of each arm against the shipped staged plan.

    A cell contributes `delta_us * invocations_per_round`, because the round
    issues that cell that many times.
    """
    by_key = {(f["cell"], f["m"], f["arm"]): f for f in fits.values()}
    cells = sorted({f["cell"] for f in fits.values()})
    nets = []
    for m in widths:
        for arm in arms:
            if arm == BASELINE_ARM:
                continue
            per_cell, total, lo_total, hi_total, complete = [], 0.0, 0.0, 0.0, True
            for cell in cells:
                base = by_key.get((cell, m, BASELINE_ARM))
                cand = by_key.get((cell, m, arm))
                if base is None or cand is None:
                    complete = False
                    continue
                inv = base["invocations_per_round"]
                delta = base["slope_us"] - cand["slope_us"]
                # Independent per-block fits, so the two intervals add in
                # quadrature rather than linearly.
                bh = (base["slope_ci95"][1] - base["slope_ci95"][0]) / 2
                ch = (cand["slope_ci95"][1] - cand["slope_ci95"][0]) / 2
                half = math.hypot(bh, ch)
                ms = delta * inv / 1000.0
                per_cell.append({
                    "cell": cell, "invocations_per_round": inv,
                    "baseline_us": base["slope_us"],
                    "candidate_us": cand["slope_us"],
                    "delta_us": delta, "delta_us_ci95_half": half,
                    "ms_per_round": ms,
                    "baseline_dispatches": base["x_groups"],
                    "candidate_dispatches": cand["x_groups"],
                })
                total += ms
                lo_total += (delta - half) * inv / 1000.0
                hi_total += (delta + half) * inv / 1000.0
            nets.append({
                "m": m, "arm": arm, "complete": complete,
                "ms_per_round": total,
                "ms_per_round_ci95": [lo_total, hi_total],
                "per_cell": per_cell, **ranked_legal(arm, m),
            })
    return nets


def pool(nets: list[dict], census: dict[int, int], arms: list[str],
         name: str) -> list[dict]:
    total_rounds = sum(census.values())
    by_key = {(n["m"], n["arm"]): n for n in nets}
    out = []
    for arm in arms:
        if arm == BASELINE_ARM:
            continue
        terms, pooled, lo, hi, illegal = [], 0.0, 0.0, 0.0, []
        for m, rounds in sorted(census.items()):
            net = by_key.get((m, arm))
            if net is None or rounds == 0:
                continue
            if not net["ranked_legal"]:
                illegal.append(m)
                continue
            share = rounds / total_rounds
            terms.append({
                "m": m, "rounds": rounds, "share": share,
                "at_width_ms_per_round": net["ms_per_round"],
                "pooled_term": net["ms_per_round"] * share,
                "two_sided": net["two_sided"],
            })
            pooled += net["ms_per_round"] * share
            lo += net["ms_per_round_ci95"][0] * share
            hi += net["ms_per_round_ci95"][1] * share
        out.append({
            "census": name, "arm": arm, "total_rounds": total_rounds,
            "pooled_ms_per_round": pooled,
            "pooled_ms_per_round_ci95": [lo, hi],
            "round_mass_priced": sum(t["rounds"] for t in terms),
            "widths_dropped_illegal": illegal,
            "all_terms_two_sided": all(t["two_sided"] for t in terms),
            "terms": terms,
        })
    return out


def argmax_plan(nets: list[dict], census: dict[int, int],
                name: str) -> dict:
    """Best ranked-legal arm at each width, which is what a width-keyed router
    would actually emit."""
    total_rounds = sum(census.values())
    chosen, pooled, lo, hi = [], 0.0, 0.0, 0.0
    for m, rounds in sorted(census.items()):
        if rounds == 0:
            continue
        legal = [n for n in nets
                 if n["m"] == m and n["ranked_legal"] and n["complete"]]
        if not legal:
            continue
        best = max(legal, key=lambda n: n["ms_per_round"])
        if best["ms_per_round"] <= 0:
            continue
        share = rounds / total_rounds
        chosen.append({
            "m": m, "arm": best["arm"], "rounds": rounds,
            "at_width_ms_per_round": best["ms_per_round"],
            "pooled_term": best["ms_per_round"] * share,
            "two_sided": best["two_sided"],
        })
        pooled += best["ms_per_round"] * share
        lo += best["ms_per_round_ci95"][0] * share
        hi += best["ms_per_round_ci95"][1] * share
    return {
        "census": name, "plan": "per-width argmax over ranked-legal arms",
        "pooled_ms_per_round": pooled,
        "pooled_ms_per_round_ci95": [lo, hi],
        "round_mass_priced": sum(c["rounds"] for c in chosen),
        "total_rounds": total_rounds,
        "all_terms_two_sided": all(c["two_sided"] for c in chosen),
        "widths": chosen,
    }


def per_chain(samples: list[dict]) -> dict:
    """Sign-stability check that does not depend on the linear chain model.

    The chain fit has large negative intercepts on the wide cells, which means
    a chained dispatch costs more than an isolated one and the fitted slope is
    an upper bound on the marginal cost. A verdict that only the slope supports
    would be a verdict about the fit. This reports the plain mean cost of one
    dispatch at each chain length, where `chain = 1` is the closest analogue of
    a decode round that issues the cell once between other kernels.
    """
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in samples:
        grouped[(row["cell"], row["m"], row["arm"], row["chain"])].append(
            row["microseconds_per_unit"])
    means = {k: statistics.fmean(v) for k, v in grouped.items()}
    chains = sorted({k[3] for k in means})
    rows = []
    for (cell, m, arm, chain), value in sorted(means.items()):
        if arm == BASELINE_ARM:
            continue
        base = means.get((cell, m, BASELINE_ARM, chain))
        if base is None:
            continue
        rows.append({
            "cell": cell, "m": m, "arm": arm, "chain": chain,
            "baseline_us": base, "candidate_us": value,
            "ratio_candidate_over_baseline": value / base,
            "delta_us": base - value,
        })
    stability = []
    for cell in sorted({r["cell"] for r in rows}):
        for m in sorted({r["m"] for r in rows}):
            for arm in sorted({r["arm"] for r in rows}):
                seen = [r for r in rows
                        if r["cell"] == cell and r["m"] == m
                        and r["arm"] == arm]
                if len(seen) != len(chains):
                    continue
                signs = {r["delta_us"] > 0 for r in seen}
                stability.append({
                    "cell": cell, "m": m, "arm": arm,
                    "faster_at_every_chain": signs == {True},
                    "slower_at_every_chain": signs == {False},
                    "sign_stable": len(signs) == 1,
                    "ratios": [r["ratio_candidate_over_baseline"]
                               for r in seen],
                })
    return {"chains": chains, "rows": rows, "sign_stability": stability}


def cellwise_ceiling(samples: list[dict], census: dict[int, int],
                     name: str, chain: int) -> dict:
    """Most generous ceiling this experiment can reach.

    A per-cell, per-width router that keeps the shipped plan wherever fusion
    loses and takes the best ranked-legal fused arm wherever it wins. Nothing
    inside the declared mechanism can beat this, so if it misses the promotion
    bar the mechanism is dead rather than mistuned.
    """
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in samples:
        if row["chain"] != chain:
            continue
        grouped[(row["cell"], row["m"], row["arm"])].append(
            row["microseconds_per_unit"])
    means = {k: statistics.fmean(v) for k, v in grouped.items()}
    inv = {(r["cell"]): r["invocations_per_round"] for r in samples}
    cells = sorted({k[0] for k in means})
    total_rounds = sum(census.values())
    picks, pooled = [], 0.0
    for m, rounds in sorted(census.items()):
        if rounds == 0:
            continue
        width_ms = 0.0
        for cell in cells:
            base = means.get((cell, m, BASELINE_ARM))
            if base is None:
                continue
            best_arm, best_ms = BASELINE_ARM, 0.0
            for arm in ("fused_r2", "fused_r4"):
                if not ranked_legal(arm, m)["ranked_legal"]:
                    continue
                cand = means.get((cell, m, arm))
                if cand is None:
                    continue
                ms = (base - cand) * inv[cell] / 1000.0
                if ms > best_ms:
                    best_arm, best_ms = arm, ms
            width_ms += best_ms
            picks.append({"m": m, "cell": cell, "arm": best_arm,
                          "ms_per_round": best_ms})
        pooled += width_ms * rounds / total_rounds
    return {
        "census": name, "chain": chain,
        "plan": "per-cell per-width argmax, shipped plan kept where fusion "
                "loses",
        "pooled_ms_per_round": pooled,
        "total_rounds": total_rounds,
        "picks": picks,
    }


def calibration(fits: dict[str, dict], samples: list[dict]) -> dict:
    """The `staged_r4` to `single_r4` delta at `m = 6`.

    Both geometries are register-legal at that width and the shipped router
    already picks `singlePass` there for `mlp.gate_up` and `gdn.in_proj`, so
    this delta is the real fusion saving with no register confound. The desk
    model priced the whole experiment off `a + b * stream`; this measurement
    validates or refutes it.

    The shipped staged plan is ONE dispatch whose grid x carries the group
    index (`Qwen35.swift:2185`, `grid: (groups * 32, ...)`), so `a` never
    applied and the only term at stake is `b * stream`. Removing a group pass
    removes a weight read that ran CONCURRENTLY with the pass that keeps it,
    which is a different quantity from the first-touch DRAM stream `b` was
    fitted on.
    """
    e219_a_us = 45.724
    e219_b_us_per_mib = 1.9292
    chain1: dict[tuple, list[float]] = defaultdict(list)
    for row in samples:
        if row["chain"] == 1:
            chain1[(row["cell"], row["m"], row["arm"])].append(
                row["microseconds_per_unit"])
    chain1_means = {k: statistics.fmean(v) for k, v in chain1.items()}
    rows = []
    for cell in sorted({f["cell"] for f in fits.values()}):
        base = next((f for f in fits.values()
                     if f["cell"] == cell and f["m"] == 6
                     and f["arm"] == BASELINE_ARM), None)
        cand = next((f for f in fits.values()
                     if f["cell"] == cell and f["m"] == 6
                     and f["arm"] == "single_r4"), None)
        if base is None or cand is None:
            continue
        saved_mib = ((base["pass_stream_bytes"] - cand["pass_stream_bytes"])
                     / 1048576.0)
        predicted = e219_a_us + e219_b_us_per_mib * saved_mib
        measured = base["slope_us"] - cand["slope_us"]
        bh = (base["slope_ci95"][1] - base["slope_ci95"][0]) / 2
        ch = (cand["slope_ci95"][1] - cand["slope_ci95"][0]) / 2
        isolated = (chain1_means[(cell, 6, BASELINE_ARM)]
                    - chain1_means[(cell, 6, "single_r4")])
        rows.append({
            "cell": cell,
            "baseline_group_passes": base["x_groups"],
            "candidate_group_passes": cand["x_groups"],
            "stream_saved_mib": saved_mib,
            "predicted_b_term_us": e219_b_us_per_mib * saved_mib,
            "measured_slope_us": measured,
            "measured_slope_ci95_half": math.hypot(bh, ch),
            "measured_isolated_us": isolated,
            "implied_marginal_b_us_per_mib":
                isolated / saved_mib if saved_mib else None,
            "fraction_of_e219_b":
                (isolated / saved_mib) / e219_b_us_per_mib
                if saved_mib else None,
        })
    implied = [r["implied_marginal_b_us_per_mib"] for r in rows
               if r["implied_marginal_b_us_per_mib"] is not None]
    return {
        "question":
            "is a redundant CONCURRENT weight group pass worth E219's fitted "
            "b, which was measured on first-touch DRAM streams?",
        "shipped_staged_plan_is_one_dispatch": True,
        "shipped_group_index_source": "grid x, Qwen35.swift:2185",
        "e219_a_us_per_dispatch_not_applicable": e219_a_us,
        "e219_b_us_per_mib": e219_b_us_per_mib,
        "per_cell": rows,
        "implied_marginal_b_us_per_mib_mean":
            statistics.fmean(implied) if implied else None,
        "implied_marginal_b_us_per_mib_max": max(implied) if implied else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    parser.add_argument("--phase", default="screen")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    session = pathlib.Path(args.session_dir)
    blob = json.loads((session / f"{args.phase}.json").read_text())
    samples = blob["samples"]
    fits = unit_fits(samples)
    widths = sorted({f["m"] for f in fits.values()})
    arms = list(dict.fromkeys(blob["arms"]))

    nets = width_nets(fits, widths, arms)
    temps = [v for v in blob["gpu_temperature_c"].values() if v > 0]
    entries = [v for k, v in blob["gpu_temperature_c"].items()
               if k.endswith("_entry") and v > 0]

    report = {
        "probe": "e222-dequant-value-sharing",
        "phase": args.phase,
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "baseline_arm": BASELINE_ARM,
        "arms": arms,
        "widths": widths,
        "blocks": blob["blocks"],
        "chains": blob["chains"],
        "abba_counterbalanced": True,
        "gpu_temperature_c": blob["gpu_temperature_c"],
        "gpu_temperature_min_c": min(temps),
        "gpu_temperature_max_c": max(temps),
        "gpu_temperature_spread_c": max(temps) - min(temps),
        "block_entry_temperature_spread_c": max(entries) - min(entries),
        "register_budgets": {"g16s": G16S_BUDGET, "g17s": G17S_BUDGET},
        "register_probe_timed_instantiations": {
            f"{arm}/m{m}": v for (arm, m), v in FUSED_REGISTERS.items()},
        "fusion_saving_calibration": calibration(fits, samples),
        "per_chain": per_chain(samples),
        "cellwise_ceiling": {
            f"{c}/chain{ch}": cellwise_ceiling(samples, cen, c, ch)
            for c, cen in (("cap7", CAP7_ROUNDS), ("cap8", CAP8_ROUNDS))
            for ch in blob["chains"]},
        "unit_fits": sorted(fits.values(), key=lambda f: f["label"]),
        "width_nets": nets,
        "pooled": {
            "cap7": pool(nets, CAP7_ROUNDS, arms, "cap7"),
            "cap8": pool(nets, CAP8_ROUNDS, arms, "cap8"),
        },
        "argmax_plan": {
            "cap7": argmax_plan(nets, CAP7_ROUNDS, "cap7"),
            "cap8": argmax_plan(nets, CAP8_ROUNDS, "cap8"),
        },
        "width_census": {"cap7": CAP7_ROUNDS, "cap8": CAP8_ROUNDS},
    }

    out = pathlib.Path(args.out) if args.out else (
        HERE / "e222-artifacts" / f"e222-{args.phase}-analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    cal = report["fusion_saving_calibration"]
    print(f"== fusion saving calibration, m = 6, {BASELINE_ARM} -> single_r4")
    print(f"   E219 fitted b = {cal['e219_b_us_per_mib']:.4f} us/MiB; the "
          "shipped staged plan is ONE dispatch, so `a` never applied")
    for r in cal["per_cell"]:
        print(f"  {r['cell']:<14} {r['baseline_group_passes']}->"
              f"{r['candidate_group_passes']} group pass  "
              f"saved {r['stream_saved_mib']:8.2f} MiB  "
              f"b predicts {r['predicted_b_term_us']:8.1f} us  "
              f"isolated {r['measured_isolated_us']:8.1f} us  "
              f"marginal b {r['implied_marginal_b_us_per_mib']:+7.3f} "
              f"({r['fraction_of_e219_b']:+6.1%} of E219 b)")
    print(f"  mean marginal b "
          f"{cal['implied_marginal_b_us_per_mib_mean']:+.3f} us/MiB, "
          f"best cell {cal['implied_marginal_b_us_per_mib_max']:+.3f}")

    print("\n== per-width net ms/round against the shipped staged plan")
    for n in nets:
        flag = "ranked-only" if (n["ranked_legal"] and not n["two_sided"]) \
            else ("ILLEGAL" if not n["ranked_legal"] else "two-sided")
        print(f"  m{n['m']} {n['arm']:<12} {n['ms_per_round']:+9.3f} "
              f"[{n['ms_per_round_ci95'][0]:+8.3f},"
              f"{n['ms_per_round_ci95'][1]:+8.3f}]  {flag}")

    for census in ("cap7", "cap8"):
        print(f"\n== pooled, {census} census")
        for p in report["pooled"][census]:
            print(f"  {p['arm']:<12} {p['pooled_ms_per_round']:+9.3f} "
                  f"[{p['pooled_ms_per_round_ci95'][0]:+8.3f},"
                  f"{p['pooled_ms_per_round_ci95'][1]:+8.3f}]  "
                  f"mass {p['round_mass_priced']}/{p['total_rounds']}  "
                  f"dropped {p['widths_dropped_illegal']}")
        a = report["argmax_plan"][census]
        print(f"  {'ARGMAX':<12} {a['pooled_ms_per_round']:+9.3f} "
              f"[{a['pooled_ms_per_round_ci95'][0]:+8.3f},"
              f"{a['pooled_ms_per_round_ci95'][1]:+8.3f}]  "
              + " ".join(f"m{c['m']}={c['arm']}" for c in a["widths"]))

    print("\n== sign stability across chain lengths "
          "(candidate/baseline, chains 1..4)")
    for s in report["per_chain"]["sign_stability"]:
        verdict = "FASTER at every chain" if s["faster_at_every_chain"] else (
            "slower at every chain" if s["slower_at_every_chain"]
            else "sign flips")
        print(f"  {s['cell']:<14} m{s['m']} {s['arm']:<12} "
              + " ".join(f"{r:5.2f}" for r in s["ratios"])
              + f"  {verdict}")

    print("\n== most generous ceiling: per-cell per-width argmax, shipped "
          "plan kept where fusion loses")
    for key in sorted(report["cellwise_ceiling"]):
        c = report["cellwise_ceiling"][key]
        print(f"  {key:<14} {c['pooled_ms_per_round']:+9.3f} ms/round")

    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
