#!/usr/bin/env python3
"""E109 rung 0: reduce one blocked A/B session to an effect with a confidence
interval, and state what the protocol can and cannot resolve.

    usage: research/e109_ab_report.py SESSION_DIR [--json OUT]
                                      [--known-dose LABEL=UNITS ...]

THE ENDPOINT. Every leg reports `block_request_seconds`, one parent-measured
wall-clock time per decode round. The leg statistic is a 10 %-trimmed mean over
those rounds after dropping round 0. That number IS a decode-only round time,
so a percentage quoted against it satisfies campaign rule 34 by construction --
there is no `wall / rounds` denominator here and no seed model to get wrong.

THE ESTIMATOR. Arms are paired inside a block:

    d_b = stat(arm, block b) - stat(control, block b)

and the reported effect is mean(d_b) with a Student-t 95 % interval over the k
estimate blocks. Block 0 is a thermal conditioning block and never enters d.

WHAT THE HALF-WIDTH MEANS. Run two BYTE-IDENTICAL arms through the same
protocol and the interval must cover zero; its half-width is the smallest
effect this protocol can separate from noise. That is the number the campaign
needs, and it is reported in microseconds per round and as a percentage of the
measured control round.

NOISE DECOMPOSITION. Two variances drive that half-width and they are bought
differently:

    var(d_b) = 2 * (sigma_leg^2 + sigma_round^2 / n_rounds)

`sigma_round / sqrt(n_rounds)` is the within-leg standard error and falls as
the token window grows. `sigma_leg` is a per-leg offset -- clock and thermal
state, allocation luck -- and only more blocks reduce it. The report separates
them so the next student knows whether to buy tokens or blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

# Two-sided 95 % Student-t quantiles by degrees of freedom.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
TRIM = 0.10
BAR_PCT = 0.20


def t95(df: int) -> float:
    return T95.get(df, 1.960)


def trimmed_mean(values: list[float], frim: float = TRIM) -> float:
    ordered = sorted(values)
    cut = int(len(ordered) * frim)
    kept = ordered[cut: len(ordered) - cut] if cut else ordered
    return statistics.fmean(kept)


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def load_legs(session: pathlib.Path) -> list[dict]:
    legs = []
    for leg_dir in sorted(session.glob("b*-*")):
        report_path = leg_dir / "report.json"
        meta_path = leg_dir / "meta.txt"
        if not report_path.exists() or not meta_path.exists():
            continue
        report = json.loads(report_path.read_text())
        meta = read_meta(meta_path)
        rounds = report["block_request_seconds"][1:]
        us = [r * 1e6 for r in rounds]
        legs.append(
            {
                "dir": leg_dir.name,
                "block": int(meta["block"]),
                "arm": meta["arm_label"],
                "arm_env": meta.get("arm_env", ""),
                "round_us_trimmed": trimmed_mean(us),
                "round_us_mean": statistics.fmean(us),
                "round_us_median": statistics.median(us),
                "round_us_sem": statistics.stdev(us) / math.sqrt(len(us)),
                "round_us_sd": statistics.stdev(us),
                "rounds_used": len(us),
                "round_count": report["round_count"],
                "first_round_us": report["block_request_seconds"][0] * 1e6,
                "leg_seconds": report["decode_seconds"],
                "seconds_per_token": report["parent_measured_seconds_per_token"],
                "mean_draft": report["effective_mean_draft_len"],
                "matched": report["all_tokens_matched"],
                "entry_c": float(meta["gpu_temp_entry_c"] or "nan"),
                "exit_c": float(meta["gpu_temp_exit_c"] or "nan"),
                "wall_seconds": float(meta.get("leg_wall_seconds", "nan")),
            }
        )
    return legs


def contrast(diffs: list[float], control_round_us: float) -> dict:
    k = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if k > 1 else float("nan")
    sem = sd / math.sqrt(k) if k > 1 else float("nan")
    half = t95(k - 1) * sem if k > 1 else float("nan")
    return {
        "blocks": k,
        "effect_us_per_round": mean,
        "effect_pct": 100.0 * mean / control_round_us,
        "sd_of_block_diffs_us": sd,
        "sem_us": sem,
        "half_width_us": half,
        "half_width_pct": 100.0 * half / control_round_us,
        "ci95_us": [mean - half, mean + half],
        "ci95_pct": [
            100.0 * (mean - half) / control_round_us,
            100.0 * (mean + half) / control_round_us,
        ],
        "significant_at_95": bool(abs(mean) > half),
        "block_diffs_us": diffs,
        "single_pair_spread_pct": [100.0 * d / control_round_us for d in diffs],
    }


def reduce_session(session: pathlib.Path, known_dose: dict[str, float]) -> dict:
    legs = load_legs(session)
    if not legs:
        raise SystemExit(f"no legs under {session}")
    design = json.loads((session / "design.json").read_text())
    control = design["control_arm"]

    estimate = [leg for leg in legs if leg["block"] > 0]
    by_block: dict[int, dict[str, dict]] = {}
    for leg in estimate:
        by_block.setdefault(leg["block"], {})[leg["arm"]] = leg

    arms = []
    for leg in legs:
        if leg["arm"] not in arms:
            arms.append(leg["arm"])

    control_stats = [
        leg["round_us_trimmed"] for leg in estimate if leg["arm"] == control
    ]
    control_round_us = statistics.fmean(control_stats)

    contrasts = {}
    for arm in arms:
        if arm == control:
            continue
        diffs = [
            block[arm]["round_us_trimmed"] - block[control]["round_us_trimmed"]
            for block in by_block.values()
            if arm in block and control in block
        ]
        if not diffs:
            continue
        entry = contrast(diffs, control_round_us)
        control_env = next(
            leg["arm_env"] for leg in legs if leg["arm"] == control)
        arm_env = next(leg["arm_env"] for leg in legs if leg["arm"] == arm)
        entry["arm_env"] = arm_env
        entry["byte_identical_to_control"] = arm_env == control_env
        entry["clears_bar"] = bool(
            entry["half_width_pct"] < BAR_PCT and entry["significant_at_95"]
        )
        contrasts[arm] = entry

    # Noise decomposition. var(d) = 2 (sigma_leg^2 + sigma_round^2 / n_rounds).
    within_sem = statistics.median([leg["round_us_sem"] for leg in estimate])
    null_arms = [a for a, c in contrasts.items()
                 if c["byte_identical_to_control"]]
    sd_null = (
        statistics.fmean([contrasts[a]["sd_of_block_diffs_us"] for a in null_arms])
        if null_arms else float("nan")
    )
    leg_var = (sd_null ** 2) / 2.0 - within_sem ** 2 if null_arms else float("nan")
    sigma_leg = math.sqrt(leg_var) if leg_var == leg_var and leg_var > 0 else 0.0

    dose = {}
    for arm, units in known_dose.items():
        if arm not in contrasts:
            continue
        per_unit = contrasts[arm]["effect_us_per_round"] / units
        dose[arm] = {
            "units": units,
            "us_per_unit": per_unit,
            "us_per_unit_ci95": [
                (contrasts[arm]["effect_us_per_round"] - contrasts[arm]["half_width_us"]) / units,
                (contrasts[arm]["effect_us_per_round"] + contrasts[arm]["half_width_us"]) / units,
            ],
            "predicts_other_arms_us": {
                other: per_unit * known_dose[other]
                for other in known_dose if other != arm
            },
        }

    entries = [leg["entry_c"] for leg in estimate if leg["entry_c"] == leg["entry_c"]]
    wall = sum(leg["wall_seconds"] for leg in legs
               if leg["wall_seconds"] == leg["wall_seconds"])
    per_leg = wall / len(legs)
    blocks = design["estimate_blocks"]

    return {
        "session": str(session),
        "design": design,
        "harness": "local",
        "control_round_us": control_round_us,
        "control_round_us_sd_across_blocks": (
            statistics.stdev(control_stats) if len(control_stats) > 1 else float("nan")
        ),
        "bar_pct": BAR_PCT,
        "bar_us": BAR_PCT / 100.0 * control_round_us,
        "arms": arms,
        "contrasts": contrasts,
        "known_dose": dose,
        "resolution": {
            "null_arms": null_arms,
            "half_width_us": (
                statistics.fmean([contrasts[a]["half_width_us"] for a in null_arms])
                if null_arms else float("nan")
            ),
            "half_width_pct": (
                statistics.fmean([contrasts[a]["half_width_pct"] for a in null_arms])
                if null_arms else float("nan")
            ),
        },
        "noise": {
            "within_leg_sem_us_median": within_sem,
            "rounds_per_leg": statistics.median([leg["rounds_used"] for leg in estimate]),
            "round_sd_us_median": statistics.median(
                [leg["round_us_sd"] for leg in estimate]),
            "sd_of_null_block_diffs_us": sd_null,
            "sigma_leg_us": sigma_leg,
            "leg_share_of_pair_variance": (
                sigma_leg ** 2 / (sigma_leg ** 2 + within_sem ** 2)
                if null_arms and (sigma_leg or within_sem) else float("nan")
            ),
        },
        "cost": {
            "legs": len(legs),
            "session_wall_seconds": wall,
            "mean_leg_wall_seconds": per_leg,
            "minutes_per_two_arm_decision": (2 * (blocks + 1) * per_leg) / 60.0,
        },
        "integrity": {
            "all_legs_matched": all(leg["matched"] for leg in legs),
            "distinct_round_counts": sorted({leg["round_count"] for leg in legs}),
            "distinct_mean_draft": sorted({leg["mean_draft"] for leg in legs}),
            "entry_c_min": min(entries) if entries else None,
            "entry_c_max": max(entries) if entries else None,
            "entry_c_spread": (max(entries) - min(entries)) if entries else None,
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "official_or_ranked_score": False,
        },
        "legs": legs,
    }


def render(out: dict) -> str:
    res = out["resolution"]
    noise = out["noise"]
    lines = [
        f"E109 protocol report -- {out['session']}",
        f"  control round {out['control_round_us']:,.1f} us"
        f"   bar {out['bar_pct']:.2f} % = {out['bar_us']:,.1f} us",
        f"  blocks {out['design']['estimate_blocks']}"
        f" (+{out['design']['conditioning_blocks']} conditioning)"
        f"  tokens {out['design']['tokens']}"
        f"  rounds/leg {noise['rounds_per_leg']:.0f}",
        "",
        f"{'arm':<8} {'effect us':>10} {'effect %':>9} {'half us':>9}"
        f" {'half %':>8} {'sig':>4} {'null':>5}",
    ]
    for arm, c in out["contrasts"].items():
        lines.append(
            f"{arm:<8} {c['effect_us_per_round']:>10.1f} {c['effect_pct']:>9.3f}"
            f" {c['half_width_us']:>9.1f} {c['half_width_pct']:>8.3f}"
            f" {'yes' if c['significant_at_95'] else 'no':>4}"
            f" {'yes' if c['byte_identical_to_control'] else 'no':>5}"
        )
    lines += [
        "",
        f"  RESOLUTION (null half-width) {res['half_width_us']:,.1f} us"
        f" = {res['half_width_pct']:.3f} %"
        f"   {'CLEARS' if res['half_width_pct'] < out['bar_pct'] else 'ABOVE'}"
        f" the {out['bar_pct']:.2f} % bar",
        f"  within-leg SEM {noise['within_leg_sem_us_median']:.1f} us"
        f"   per-round SD {noise['round_sd_us_median']:.1f} us"
        f"   leg offset sigma {noise['sigma_leg_us']:.1f} us"
        f"   leg share of pair variance"
        f" {100 * noise['leg_share_of_pair_variance']:.1f} %",
        f"  cost {out['cost']['minutes_per_two_arm_decision']:.1f} min"
        f" per two-arm decision"
        f"   ({out['cost']['mean_leg_wall_seconds']:.1f} s/leg,"
        f" {out['cost']['legs']} legs)",
    ]
    for arm, d in out["known_dose"].items():
        lines.append(
            f"  dose {arm}: {d['units']:.0f} units ->"
            f" {d['us_per_unit']:.4f} us/unit"
            f" (95 % {d['us_per_unit_ci95'][0]:.4f} ... {d['us_per_unit_ci95'][1]:.4f});"
            f" predicts " + ", ".join(
                f"{k} {v:,.1f} us" for k, v in d["predicts_other_arms_us"].items())
        )
    integ = out["integrity"]
    lines += [
        f"  integrity matched={integ['all_legs_matched']}"
        f" rounds={integ['distinct_round_counts']}"
        f" draft={integ['distinct_mean_draft']}"
        f" entry C {integ['entry_c_min']:.2f}...{integ['entry_c_max']:.2f}"
        f" (spread {integ['entry_c_spread']:.2f})",
        "  cool_gate_passed_real_gate=false gate_qualified_for_timing=false"
        " official_or_ranked_score=false",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--json")
    parser.add_argument("--known-dose", action="append", default=[])
    args = parser.parse_args()

    known = {}
    for item in args.known_dose:
        label, _, units = item.partition("=")
        known[label] = float(units)

    out = reduce_session(pathlib.Path(args.session), known)
    print(render(out))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
