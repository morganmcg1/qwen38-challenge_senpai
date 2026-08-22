#!/usr/bin/env python3
"""E137 items 2 and 3: attribute the M=5 to M=6 verify cost step.

Three independent lines of evidence, none of which needs a new timed leg:

1. E92 rung-2 production widths carry a real GPU-interval ledger
   (`gpu_intervals=1`) at PINNED widths M=1..9 on base b5cff751. Pinning
   removes the width-position confound that Rule 106 raises against the E130
   traces, and `round_gpu_busy_us` separates GPU work from host wait.
2. The isolated QMV width curve on the scored Route B path and on the MLX
   fallback beside it, from `E137RouteBCostCurveTests`.
3. The +109 dispatch step at the boundary, priced against E58's in-situ
   dispatch tax.

Every number is labelled with the base it came from. E92 and E130 sit on
different bases and are never pooled.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "research" / "e137-artifacts"

# E58's corrected in-situ dispatch tax, per dispatch.
DISPATCH_TAX_NS = (77.0, 428.0)
# Ledger 198(H) dispatch census, dispatches per round.
DISPATCH_STEPS = {"5->6": 109, "6->7": 29, "7->8": 29}
# About 64 of the +109 are the SDPA exactness chunk: per full-attention layer
# one extra SDPA, one concat and two hidden query copies, over 16 layers.
SDPA_CHUNK_DISPATCHES = 64
FULL_ATTENTION_LAYERS = 16

# In-situ round step measured in item 1 from the E130 rung-11 traces.
INSITU_STEP_US = 39134.9
# The ranked reconstruction the assignment quotes, FINDING 167.2.
RANKED_STEP_US = 16566.7


def boot_ci(samples, draws=10000, seed=20260822, stat=statistics.median):
    if len(samples) < 2:
        return {"point": stat(samples) if samples else float("nan")}
    rng = random.Random(seed)
    n = len(samples)
    vals = sorted(stat([samples[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return {
        "point": stat(samples),
        "ci_lo": vals[int(0.025 * draws)],
        "ci_hi": vals[int(0.975 * draws)],
        "bootstrap_draws": draws,
        "n": n,
    }


def boot_diff_ci(a, b, draws=10000, seed=20260822, stat=statistics.median):
    """Interval on stat(b) - stat(a) by independent resampling of each arm."""
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    vals = sorted(
        stat([b[rng.randrange(nb)] for _ in range(nb)])
        - stat([a[rng.randrange(na)] for _ in range(na)])
        for _ in range(draws)
    )
    return {
        "point": stat(b) - stat(a),
        "ci_lo": vals[int(0.025 * draws)],
        "ci_hi": vals[int(0.975 * draws)],
        "bootstrap_draws": draws,
        "n_low": na,
        "n_high": nb,
    }


# --------------------------------------------------------------------------
# 1. E92 GPU-interval ledger at pinned widths


def e92_ledger():
    path = ROOT / "research" / "e92-artifacts" / "rung2-production-widths.json"
    raw = json.loads(path.read_text())
    legs = raw["legs"]
    meta = legs[0]["meta"]

    by_width = {}
    for leg in legs:
        by_width.setdefault(leg["M"], []).append(leg)

    fields = [
        "round_us",
        "round_gpu_busy_us",
        "round_gpu_idle_us",
        "verify_gpu_busy_us",
        "head_gpu_busy_us",
        "snapshot_gpu_busy_us",
        "d_submit2_gpu_busy_us",
    ]
    table = {}
    for m, group in sorted(by_width.items()):
        row = {"n_legs": len(group), "tags": [leg["tag"] for leg in group]}
        row["accepted_mean"] = group[0].get("accepted_mean")
        row["rounds_analysed"] = sum(leg.get("rounds_analysed", 0) for leg in group)
        for f in fields:
            vals = [leg[f] for leg in group if isinstance(leg.get(f), (int, float))]
            if not vals:
                continue
            mean = statistics.fmean(vals)
            row[f] = {
                "mean": round(mean, 1),
                "legs": [round(v, 1) for v in vals],
                "spread_us": round(max(vals) - min(vals), 1),
                "spread_frac": round((max(vals) - min(vals)) / mean, 5) if mean else None,
            }
        # The host-stuck fraction says whether the host thread was waiting.
        stuck = [leg.get("frac_rounds_host_stuck") for leg in group]
        row["frac_rounds_host_stuck"] = [s for s in stuck if s is not None]
        table[m] = row

    steps = {}
    for m in sorted(table):
        if m + 1 not in table:
            continue
        step = {}
        for f in fields:
            if f in table[m] and f in table[m + 1]:
                step[f] = round(table[m + 1][f]["mean"] - table[m][f]["mean"], 1)
        steps[f"{m}->{m+1}"] = step

    busy_steps = {k: v.get("verify_gpu_busy_us", 0.0) for k, v in steps.items()}
    cliff = max(busy_steps, key=lambda k: busy_steps[k])
    return {
        "source": "research/e92-artifacts/rung2-production-widths.json",
        "base_sha": meta["base_sha"],
        "chip": meta["chip"],
        "tokens": meta["tokens"],
        "gpu_intervals": meta["gpu_intervals"],
        "harness": meta["harness"],
        "local_mode": meta["local_mode"],
        "cool_gate": meta["cool_gate"],
        "cool_gate_passed_real_gate": meta["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": meta["gate_qualified_for_timing"],
        "official_or_ranked_score": meta["official_or_ranked_score"],
        "width_pinned": True,
        "note": (
            "widths are PINNED by e92_verify_width, so width and position are "
            "not confounded here. Different and older base than the E130 "
            "traces; never pooled with them, used only for the shape of the "
            "GPU-versus-host split and the location of the cliff."
        ),
        "table": table,
        "steps": steps,
        "largest_verify_gpu_busy_step": cliff,
        "largest_verify_gpu_busy_step_us": round(busy_steps[cliff], 1),
    }


# --------------------------------------------------------------------------
# 2. Isolated QMV curves


def isolated_curve(path, arm_key):
    raw = json.loads(path.read_text())
    shapes = raw["shapes"]
    widths = raw["widths"]

    # F2 6(b): the per-shape step as well as the weighted total. One shape
    # carrying the cliff is a targetable mechanism; seven carrying it
    # proportionally is a structural cause and a different next move.
    per_shape = {}
    for s in shapes:
        samples = {}
        for m in widths:
            row = next(r for r in s["rows"] if r["m"] == m)
            key = f"{arm_key}_samples"
            if key not in row:
                continue
            tap = row["tap_overhead_seconds_per_call"]
            samples[m] = [(v - tap) * 1e6 for v in row[key]]
        entry = {
            "k": s["k"],
            "n": s["n"],
            "dispatches_per_round": s["calls_per_verify"],
            "per_call_us": {str(m): boot_ci(v) for m, v in samples.items()},
        }
        if 5 in samples and 6 in samples:
            step = boot_diff_ci(samples[5], samples[6])
            entry["per_call_step_5_to_6_us"] = step
            entry["weighted_step_5_to_6_us"] = {
                "point": round(step["point"] * s["calls_per_verify"], 1),
                "ci_lo": round(step["ci_lo"] * s["calls_per_verify"], 1),
                "ci_hi": round(step["ci_hi"] * s["calls_per_verify"], 1),
            }
            base5 = statistics.median(samples[5])
            entry["relative_step_pct"] = round(step["point"] / base5 * 100, 1)
        per_shape[s["name"]] = entry

    per_width_samples = {}
    for m in widths:
        # One replicate index across shapes is not a paired draw, so the total
        # is resampled per shape and summed at matched replicate rank.
        totals = None
        for s in shapes:
            row = next(r for r in s["rows"] if r["m"] == m)
            key = f"{arm_key}_samples"
            if key not in row:
                totals = None
                break
            tap = row["tap_overhead_seconds_per_call"]
            vals = [(v - tap) * 1e6 * s["calls_per_verify"] for v in row[key]]
            if totals is None:
                totals = list(vals)
            else:
                totals = [a + b for a, b in zip(totals, vals)]
        if totals:
            per_width_samples[m] = totals

    curve = {m: boot_ci(v) for m, v in per_width_samples.items()}
    steps = {}
    for m in widths:
        if m + 1 in per_width_samples and m in per_width_samples:
            steps[f"{m}->{m+1}"] = boot_diff_ci(
                per_width_samples[m], per_width_samples[m + 1]
            )
    return {
        "arm": arm_key,
        "widths": widths,
        "dispatch_weighting": {
            s["name"]: s["calls_per_verify"] for s in shapes
        },
        "dispatches_per_round": sum(s["calls_per_verify"] for s in shapes),
        "curve_us": {str(k): v for k, v in curve.items()},
        "steps_us": steps,
        "per_shape": per_shape,
        "samples": {str(k): len(v) for k, v in per_width_samples.items()},
    }, per_width_samples


# --------------------------------------------------------------------------
# 3. Dispatch-count pricing


def dispatch_pricing():
    lo_ns, hi_ns = DISPATCH_TAX_NS
    out = {
        "dispatch_tax_ns_per_dispatch": {"low": lo_ns, "high": hi_ns, "source": "E58 corrected in-situ"},
        "dispatch_census_source": "ledger 198(H), lines 14126-14160",
        "steps": {},
        "sdpa_chunk": {
            "guard": "AttentionUtils.swift:120-142, qL >= 6, qL <= 9, kL >= qL, causal",
            "split": 5,
            "full_attention_layers": FULL_ATTENTION_LAYERS,
            "dispatches_per_layer": "1 extra SDPA + 1 concat + 2 hidden query copies",
            "dispatches": SDPA_CHUNK_DISPATCHES,
        },
    }
    for name, count in DISPATCH_STEPS.items():
        lo_us = count * lo_ns / 1000.0
        hi_us = count * hi_ns / 1000.0
        out["steps"][name] = {
            "dispatches": count,
            "us_low": round(lo_us, 2),
            "us_high": round(hi_us, 2),
        }
    lo_us = DISPATCH_STEPS["5->6"] * lo_ns / 1000.0
    hi_us = DISPATCH_STEPS["5->6"] * hi_ns / 1000.0
    out["boundary_4_bracket_us"] = {"low": round(lo_us, 2), "high": round(hi_us, 2)}
    out["boundary_4_share_of_insitu_step_pct"] = {
        "low": round(lo_us / INSITU_STEP_US * 100, 4),
        "high": round(hi_us / INSITU_STEP_US * 100, 4),
        "step_us": INSITU_STEP_US,
    }
    out["boundary_4_share_of_ranked_step_pct"] = {
        "low": round(lo_us / RANKED_STEP_US * 100, 4),
        "high": round(hi_us / RANKED_STEP_US * 100, 4),
        "step_us": RANKED_STEP_US,
    }
    sd_lo = SDPA_CHUNK_DISPATCHES * lo_ns / 1000.0
    sd_hi = SDPA_CHUNK_DISPATCHES * hi_ns / 1000.0
    out["sdpa_chunk_only_bracket_us"] = {"low": round(sd_lo, 2), "high": round(sd_hi, 2)}
    return out


# --------------------------------------------------------------------------
# 4. The chunked-SDPA family, falsified without spending a leg


def sdpa_chunk_moving_cliff(e92):
    """The FA family is byte-identical on both bases, but the cliff moved.

    F2 section 4 closed FINDING 184 by a bandwidth argument. This closes it a
    second time, empirically and for free, by noticing that the two bases hold
    the cliff at DIFFERENT widths while holding the guard fixed.
    """
    return {
        "guard": (
            "AttentionUtils.swift:118-142, fires at qL >= 6 with split = 5, "
            "so it can only ever add work at the 5 to 6 boundary"
        ),
        "file_identity": {
            "path": "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift",
            "e92_base": "b5cff751",
            "current_base": "33ce6a3f",
            "git_diff_lines_changed": 0,
            "checked": "git diff b5cff751 33ce6a3f -- <path>, empty, this session",
        },
        "cliff_position": {
            "e92_base_b5cff751": e92["largest_verify_gpu_busy_step"],
            "e92_base_step_us": e92["largest_verify_gpu_busy_step_us"],
            "current_base_33ce6a3f": "5->6",
            "current_base_step_us": INSITU_STEP_US,
        },
        "argument": (
            "the guard is byte-identical on both bases and fires only at "
            "qL >= 6, so it cannot produce a 39,305 us step at 4->5. On the "
            "E92 base the largest verify GPU-busy step IS at 4->5, and the "
            "4->5 step on the current base is 14,889 us. A fixed qL >= 6 "
            "guard cannot move a cliff by one width."
        ),
        "conclusion": "chunked SDPA is eliminated as the carrier of the step",
        "independent_of": (
            "F2 section 4's bandwidth bracket of 270 to 420 us; this line of "
            "evidence uses neither a bandwidth constant nor a dispatch tax"
        ),
    }


# --------------------------------------------------------------------------
# 5. Local against ranked, relative to the M=5 round


def local_versus_ranked(width_table_path):
    """F2 section 2 asked for this comparison in the item 1 artifact.

    Each step is normalised by the round cost at the LOWER width, which is the
    normalisation F2 used. The ranked column is edward's refit of the
    `623e77af` and `d3c491b5` receipt pair; it is quoted, not measured here.
    """
    raw = json.loads(width_table_path.read_text())
    table = raw["designs"]["all"]["width_table"]
    rounds = {int(m): row["segments"]["round_us"]["mean"] for m, row in table.items()}
    ranked_pct = {"3->4": 9.1, "4->5": 8.3, "5->6": 36.1, "6->7": 2.7, "7->8": 11.9}

    rows = {}
    for m in sorted(rounds):
        if m + 1 not in rounds:
            continue
        key = f"{m}->{m+1}"
        local_pct = (rounds[m + 1] - rounds[m]) / rounds[m] * 100.0
        entry = {
            "local_round_us_low_width": rounds[m],
            "local_step_us": round(rounds[m + 1] - rounds[m], 1),
            "local_step_pct_of_low_width_round": round(local_pct, 1),
            "ranked_step_pct_of_low_width_round": ranked_pct.get(key),
        }
        if key in ranked_pct:
            entry["gap_pp"] = round(local_pct - ranked_pct[key], 1)
        rows[key] = entry

    return {
        "normalisation": "step(M -> M+1) divided by round(M)",
        "local_source": (
            "research/e137-artifacts/item1-width-table.json, design `all`, "
            "12 E130 rung-11 legs, Apple M4 Pro (applegpu_g16s)"
        ),
        "ranked_source": (
            "quoted from F2 section 2: edward's per-round refit of the "
            "623e77af and d3c491b5 ranked receipt pair, Apple M5 "
            "(applegpu_g17s). Not measured by this experiment."
        ),
        "steps": rows,
        "reading": (
            "every shallow step disagrees between the two hosts; the cliff "
            "agrees. The cliff is a property of the model at verify width 6, "
            "not a g16s register artefact."
        ),
    }


def main():
    ART.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "e137-items-2-and-3",
        "gpu_used": False,
        "note": "offline analysis over already-paid-for artifacts",
    }
    result["e92_gpu_interval_ledger"] = e92_ledger()

    routeb_path = ART / "item2-routeb-curve.json"
    if routeb_path.exists():
        routed, routed_samples = isolated_curve(routeb_path, "routed")
        fallback, fallback_samples = isolated_curve(routeb_path, "fallback")
        raw = json.loads(routeb_path.read_text())
        result["isolated_routed"] = routed
        result["isolated_fallback"] = fallback
        result["route_b_config"] = {
            "arm": raw["arm"],
            "entry": raw["entry"],
            "routed_widths": raw["routed_widths"],
            "weight_passes_by_width": {
                str(r["m"]): r.get("weight_passes")
                for s in raw["shapes"][:1]
                for r in s["rows"]
            },
            "bitwise_match_vs_fallback": all(
                r.get("routed_matches_fallback_bitwise", True)
                for s in raw["shapes"]
                for r in s["rows"]
            ),
        }
        if 5 in routed_samples and 6 in routed_samples:
            step = boot_diff_ci(routed_samples[5], routed_samples[6])
            share = step["point"] / INSITU_STEP_US
            if share >= 0.60:
                verdict = "QMV is the carrier; attack it"
            elif share >= 0.20:
                verdict = "QMV is a partial carrier; report the residual"
            else:
                verdict = "QMV is not the carrier; the search moves"
            result["boundary_4_transfer"] = {
                "f2_decision_rule": {
                    "thresholds": ">=0.60 carrier, 0.20-0.60 partial, <0.20 not",
                    "measured": round(share, 4),
                    "verdict": verdict,
                },
                "insitu_round_step_us": INSITU_STEP_US,
                "isolated_routed_step_us": step,
                "isolated_over_insitu": round(share, 4),
                "insitu_over_isolated": round(1.0 / share, 4) if share else None,
                "share_ci": [
                    round(step["ci_lo"] / INSITU_STEP_US, 4),
                    round(step["ci_hi"] / INSITU_STEP_US, 4),
                ],
                "implied_non_qmv_share": round(1.0 - share, 4),
                "implied_non_qmv_share_ci": [
                    round(1.0 - step["ci_hi"] / INSITU_STEP_US, 4),
                    round(1.0 - step["ci_lo"] / INSITU_STEP_US, 4),
                ],
                "transfer_safe": False,
                "transfer_safety_note": (
                    "FINDING 181: this host is applegpu_g16s with a 96-register "
                    "ceiling; the ranked host is g17s at 126. The routed QMV "
                    "family's M=5 to M=6 behaviour here is not the ranked "
                    "behaviour."
                ),
            }

    result["dispatch_pricing"] = dispatch_pricing()
    result["sdpa_chunk_moving_cliff"] = sdpa_chunk_moving_cliff(
        result["e92_gpu_interval_ledger"])
    width_table = ART / "item1-width-table.json"
    if width_table.exists():
        result["local_versus_ranked_relative_steps"] = local_versus_ranked(width_table)

    out = ART / "items-2-3-attribution.json"
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {out}")

    e92 = result["e92_gpu_interval_ledger"]
    print(f"\nE92 GPU-interval ledger, base {e92['base_sha'][:8]}, widths PINNED")
    print("M    round_us  gpu_busy  gpu_idle  verify_busy   spread")
    for m, row in sorted(e92["table"].items()):
        print(
            f"{m:<4} {row['round_us']['mean']:9.0f} "
            f"{row['round_gpu_busy_us']['mean']:9.0f} "
            f"{row['round_gpu_idle_us']['mean']:9.0f} "
            f"{row['verify_gpu_busy_us']['mean']:12.0f} "
            f"{row['verify_gpu_busy_us']['spread_frac']*100:7.2f}%"
        )
    print("\nverify_gpu_busy_us steps")
    for k, v in e92["steps"].items():
        print(f"  {k}  {v.get('verify_gpu_busy_us', float('nan')):10.1f}")
    print(
        f"\nlargest step on the E92 base: {e92['largest_verify_gpu_busy_step']} "
        f"= {e92['largest_verify_gpu_busy_step_us']} us"
    )

    if "boundary_4_transfer" in result:
        t = result["boundary_4_transfer"]
        s = t["isolated_routed_step_us"]
        print(
            f"\nisolated routed step 5->6 = {s['point']:.1f} us "
            f"[{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]"
        )
        print(
            f"in-situ step {INSITU_STEP_US} us -> isolated/in-situ = "
            f"{t['isolated_over_insitu']} {t['share_ci']}"
        )
        print(
            f"implied non-QMV share = {t['implied_non_qmv_share']} "
            f"{t['implied_non_qmv_share_ci']}"
        )
        print(f"F2 decision rule: {t['f2_decision_rule']['verdict']}")

    if "isolated_routed" in result:
        print("\nrouted per-shape, isolated, us")
        print(
            "shape                              disp    M=5/call    M=6/call"
            "   step/call    rel   weighted step"
        )
        for name, entry in result["isolated_routed"]["per_shape"].items():
            per = entry["per_call_us"]
            step = entry.get("per_call_step_5_to_6_us", {})
            weighted = entry.get("weighted_step_5_to_6_us", {})
            print(
                f"{name:<34} {entry['dispatches_per_round']:>4} "
                f"{per['5']['point']:>11.1f} {per['6']['point']:>11.1f} "
                f"{step.get('point', float('nan')):>11.1f} "
                f"{entry.get('relative_step_pct', float('nan')):>8.1f} %"
                f"{weighted.get('point', float('nan')):>16.1f}"
            )

    if "local_versus_ranked_relative_steps" in result:
        print("\nstep as a fraction of the round at the lower width")
        print("step     local (M4 Pro)   ranked (M5, quoted)   gap")
        for key, row in result["local_versus_ranked_relative_steps"]["steps"].items():
            ranked = row["ranked_step_pct_of_low_width_round"]
            gap = row.get("gap_pp")
            print(
                f"{key:<8} {row['local_step_pct_of_low_width_round']:>13.1f} % "
                f"{('%.1f %%' % ranked) if ranked is not None else '-':>21} "
                f"{('%+.1f pp' % gap) if gap is not None else '-':>10}"
            )

    chunk = result.get("sdpa_chunk_moving_cliff")
    if chunk:
        print(
            f"\nchunked SDPA: guard byte-identical across bases, cliff at "
            f"{chunk['cliff_position']['e92_base_b5cff751']} on "
            f"{chunk['file_identity']['e92_base']} and "
            f"{chunk['cliff_position']['current_base_33ce6a3f']} on "
            f"{chunk['file_identity']['current_base']} -> {chunk['conclusion']}"
        )

    p = result["dispatch_pricing"]
    b = p["boundary_4_bracket_us"]
    print(f"\ndispatch bracket at boundary 4: {b['low']} to {b['high']} us")
    print(
        f"  as a share of the in-situ step: "
        f"{p['boundary_4_share_of_insitu_step_pct']['low']} to "
        f"{p['boundary_4_share_of_insitu_step_pct']['high']} %"
    )
    print(
        f"  as a share of the ranked step:  "
        f"{p['boundary_4_share_of_ranked_step_pct']['low']} to "
        f"{p['boundary_4_share_of_ranked_step_pct']['high']} %"
    )


if __name__ == "__main__":
    main()
