#!/usr/bin/env python3
"""E116 rung 2: how much of an injected GPU dose does a decode round absorb?

    usage: research/e116_absorption_report.py LEG_DIR [LEG_DIR ...]
               --dose-unit-us US [--json OUT]

ESTIMAND.

    alpha = d(round us) / d(injected dose us)

`alpha = 1.00` means the round is perfectly serial: every microsecond of extra
GPU work lands on the round wall, so only byte or ALU reduction can help and
the overlap class has no target at the round level. `alpha < 1` means the round
has slack that a restructuring could claim.

DESIGN. askeladd's E109 v2 within-leg alternating estimator, reused unchanged:
this file imports `pair_rounds`, `triple_rounds`, `summarise` and `t95` from
`research/e109_v2_report.py` rather than restating them. The dose alternates
round by round, so the per-leg offset that carried 97.9 % of E109 v1's pair
variance cancels inside every pair.

WHAT E116 CHANGES. In E109 the dose flag lived inside the model forward, so the
instrumented boundary saw 380 width-1 forwards and only 12 of 77 timed rounds:
`round_alignment_verified` was false and the instrument was unusable for
wide-verify arms. E116 rung 0b moved the switch into `generateRound`, where it
is read once per round. The mapping from round index to dose is therefore exact
by construction:

    round index i (0-based) is dosed  <=>  i is odd

which is the same convention `e109_v2_report.pair_rounds` assumes. A leg that
also carries `MLX_QWEN_MTP_TRACE=1` writes one `mtp-trace: e116 dose` line per
round; this reader checks that witness against the parent's own
`effective_draft_lengths` and reports `round_alignment_verified`.

THE NULL ARM IS NOT OPTIONAL. A decode round could carry period-2 structure of
its own, and the alternating estimator would report it whether or not a dose
was applied. `alpha` is therefore built from the DIFFERENCE between the dosed
arm and the `k = 0` arm, both run through the identical estimator with the
identical hypothetical assignment.

RULE 34. Every percent printed here divides the E109 v2 control round, which
is the mean parent-measured `block_request_seconds` over rounds 1..R-1 of the
same legs. That frame is named in the output as `e109_v2_control_round_us`.

RULE 37. One dose unit is one `mlp.gate_up`-shaped affine 4-bit group-64 QMV at
M = 1. `--dose-unit-us` must be the M=1 census rate. It is not a scored-width
rate and nothing here may be read as one.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e109_v2_report import (  # noqa: E402
    pair_rounds,
    read_meta,
    summarise,
    t95,
    triple_rounds,
    within_leg_drift_us_per_round,
)


def dose_of(arm_env: str) -> int:
    for token in arm_env.replace(",", " ").split():
        key, _, value = token.partition("=")
        if key == "MLX_E116_DOSE":
            return int(value)
    return 0


def alternating(arm_env: str) -> bool:
    return "MLX_E116_DOSE_ALTERNATE=1" in arm_env


def read_dose_witness(path: pathlib.Path) -> list[dict] | None:
    """One `mtp-trace: e116 dose` line per round, written by the worker."""
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("mtp-trace: e116 dose "):
            continue
        fields = {}
        for token in line.split()[3:]:
            key, _, value = token.partition("=")
            fields[key] = value
        rows.append({"round": int(fields["round"]),
                     "dosed": int(fields["dosed"]),
                     "units": int(fields["units"]),
                     "width": int(fields["width"])})
    return rows or None


def histogram(values: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def analyse(leg_dir: pathlib.Path) -> dict:
    report = json.loads((leg_dir / "report.json").read_text())
    meta = read_meta(leg_dir / "meta.txt")
    arm_env = meta.get("arm_env", "")

    us = [s * 1e6 for s in report["block_request_seconds"]]
    widths = list(report["effective_draft_lengths"])
    rounds = len(us)
    if len(widths) != rounds:
        raise SystemExit(
            f"{leg_dir}: {rounds} round times but {len(widths)} widths")

    dose = dose_of(arm_env)
    alt = alternating(arm_env)
    # The switch lives in the round loop, so the mapping is exact.
    dosed = [i % 2 == 1 for i in range(rounds)] if alt else [True] * rounds

    witness = read_dose_witness(leg_dir / "trace.txt")
    alignment = None
    if witness:
        tail = witness[-rounds:]
        observed_widths = [row["width"] for row in tail]
        expected_widths = [k + 1 for k in widths]
        witness_dosed = [row["dosed"] == 1 for row in tail]
        matched = observed_widths == expected_widths
        alignment = {
            "witness_rounds": len(witness),
            "round_count": rounds,
            "one_witness_line_per_round": len(witness) == rounds,
            "width_histogram_observed": histogram(observed_widths),
            "width_histogram_expected": histogram(expected_widths),
            "width_one_lines": sum(1 for w in observed_widths if w == 1),
            "width_fingerprint_matched": matched,
            "witness_dosed_agrees_with_parity": witness_dosed == dosed,
            "dosed_rounds": sum(witness_dosed),
            "units_when_dosed": sorted(
                {row["units"] for row in tail if row["dosed"] == 1}),
            "verify_block_replayed_round_count": report[
                "verify_block_replayed_round_count"],
        }
        if matched:
            dosed = witness_dosed

    pairs = pair_rounds(us, widths, dosed)
    pair_stats = summarise([p["difference_us"] for p in pairs])
    triples = triple_rounds(us, widths, dosed)
    triple_stats = summarise([t["estimate_us"] for t in triples])
    control_round_us = statistics.fmean(us[1:]) if rounds > 1 else float("nan")

    return {
        "leg": leg_dir.name,
        "harness": "local",
        "arm_label": meta.get("arm_label"),
        "arm_env": arm_env,
        "dose_units": dose,
        "alternating": alt,
        "git_head": meta.get("git_head"),
        "git_dirty_build": meta.get("git_dirty_build"),
        "worker_sha256": meta.get("worker_sha256"),
        "golden_sha256": meta.get("golden_sha256"),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "leg_wall_seconds": meta.get("leg_wall_seconds"),
        "all_tokens_matched": report["all_tokens_matched"],
        "round_count": rounds,
        "seconds_per_token": report.get("parent_measured_seconds_per_token"),
        "width_histogram": histogram([k + 1 for k in widths]),
        "dose_alignment": alignment,
        "pairs": pair_stats["n"],
        "pairs_by_width": histogram([p["width"] for p in pairs]),
        "e109_v2_control_round_us": control_round_us,
        "within_leg_drift_us_per_round": within_leg_drift_us_per_round(
            us, widths),
        "paired_difference_mean_us": pair_stats["mean_us"],
        "paired_difference_sd_us": pair_stats["sd_us"],
        "paired_difference_sem_us": pair_stats["sem_us"],
        "half_width_us": pair_stats["half_width_us"],
        "drift_cancelled_triples": triple_stats["n"],
        "drift_cancelled_mean_us": triple_stats["mean_us"],
        "drift_cancelled_sem_us": triple_stats["sem_us"],
        "drift_cancelled_half_width_us": triple_stats["half_width_us"],
    }


def arm_mean(legs: list[dict], key: str) -> dict:
    values = [leg[key] for leg in legs]
    n = len(values)
    mean = statistics.fmean(values) if n else float("nan")
    sd = statistics.stdev(values) if n > 1 else float("nan")
    return {"legs": n, "mean_us": mean, "sd_us": sd,
            "sem_us": sd / math.sqrt(n) if n > 1 else float("nan")}


def contrast(dosed: list[dict], null: list[dict], key: str) -> dict:
    a, b = arm_mean(dosed, key), arm_mean(null, key)
    delta = a["mean_us"] - b["mean_us"]
    variance, df = 0.0, 0
    for side in (a, b):
        if side["legs"] > 1 and not math.isnan(side["sem_us"]):
            variance += side["sem_us"] ** 2
            df += side["legs"] - 1
    half = t95(df) * math.sqrt(variance) if df > 0 else float("nan")
    return {"dosed": a, "null": b, "dose_minus_null_us": delta,
            "half_width_us": half, "df": df,
            "excludes_zero": bool(df > 0 and abs(delta) > half)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="+")
    ap.add_argument("--dose-unit-us", type=float, required=True,
                    help="measured M=1 census microseconds per dose unit")
    ap.add_argument("--json")
    args = ap.parse_args()

    results = [analyse(pathlib.Path(p)) for p in args.legs]
    dosed = [r for r in results if r["dose_units"] > 0]
    null = [r for r in results if r["dose_units"] == 0]
    if not dosed or not null:
        raise SystemExit("e116_absorption_report: need a dosed arm and a "
                         "k=0 null arm")

    doses = sorted({r["dose_units"] for r in dosed})
    if len(doses) != 1:
        raise SystemExit(f"e116_absorption_report: mixed doses {doses}")
    dose_units = doses[0]
    injected_us = dose_units * args.dose_unit_us
    control = statistics.fmean(r["e109_v2_control_round_us"] for r in results)

    out = {
        "harness": "local",
        "experiment": "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": 2,
        "estimator_1": "e109 v2 mean of equal-width neighbouring round pairs",
        "estimator_2": "e109 v2 equal-width DUD/UDU triples, drift cancelled",
        "round_frame": "e116 rung 2 = mean parent block_request_seconds over "
                       "rounds 1..R-1 of these mtp-timed legs",
        "e109_v2_control_round_us": control,
        "dose_units": dose_units,
        "dose_unit_us_m1_census": args.dose_unit_us,
        "injected_us_per_dosed_round": injected_us,
        "injected_percent_of_round": 100.0 * injected_us / control,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "legs": results,
    }

    for label, key in (("pair", "paired_difference_mean_us"),
                       ("triple", "drift_cancelled_mean_us")):
        c = contrast(dosed, null, key)
        alpha = c["dose_minus_null_us"] / injected_us
        half_alpha = c["half_width_us"] / injected_us
        out[f"alpha_{label}"] = {
            **c,
            "alpha": alpha,
            "alpha_half_width": half_alpha,
            "alpha_ci95": [alpha - half_alpha, alpha + half_alpha],
            "round_is_serial_alpha_1": bool(
                not math.isnan(half_alpha) and abs(alpha - 1.0) <= half_alpha),
            "slack_alpha_below_0_90": bool(
                not math.isnan(half_alpha) and alpha + half_alpha < 0.90),
        }

    out["round_alignment_verified"] = all(
        leg["dose_alignment"]
        and leg["dose_alignment"]["width_fingerprint_matched"]
        and leg["dose_alignment"]["one_witness_line_per_round"]
        and leg["dose_alignment"]["witness_dosed_agrees_with_parity"]
        for leg in results if leg["dose_alignment"] is not None) and any(
        leg["dose_alignment"] is not None for leg in results)

    print("E116 rung 2 -- round absorption of a known GPU dose   harness=local")
    print("  ungated: cool_gate_passed_real_gate=false,"
          " gate_qualified_for_timing=false, official_or_ranked_score=false")
    print(f"  round frame: {out['round_frame']}")
    print(f"  e109_v2_control_round_us = {control:,.0f}")
    print(f"  dose k={dose_units} at {args.dose_unit_us:.2f} us/unit (M=1)"
          f" = {injected_us:,.1f} us"
          f" = {out['injected_percent_of_round']:.3f} % of the round")
    print()
    print(f"{'leg':<14} {'arm':<6} {'k':>3} {'matched':>8} {'rounds':>7}"
          f" {'pairs':>6} {'round us':>10} {'pair us':>10} {'+-95% us':>9}"
          f" {'triple us':>10} {'entry C':>8} {'exit C':>7}")
    for r in sorted(results, key=lambda x: x["leg"]):
        print(f"{r['leg']:<14} {str(r['arm_label']):<6} {r['dose_units']:>3}"
              f" {str(r['all_tokens_matched']):>8} {r['round_count']:>7}"
              f" {r['pairs']:>6} {r['e109_v2_control_round_us']:>10,.0f}"
              f" {r['paired_difference_mean_us']:>+10.1f}"
              f" {r['half_width_us']:>9.1f}"
              f" {r['drift_cancelled_mean_us']:>+10.1f}"
              f" {str(r['gpu_temp_entry_c']):>8}"
              f" {str(r['gpu_temp_exit_c']):>7}")
    print()
    for label in ("pair", "triple"):
        a = out[f"alpha_{label}"]
        print(f"  {label:<7} dosed {a['dosed']['mean_us']:+9.1f} us"
              f" (n={a['dosed']['legs']}),"
              f" null {a['null']['mean_us']:+9.1f} us"
              f" (n={a['null']['legs']});"
              f" difference {a['dose_minus_null_us']:+9.1f}"
              f" +-{a['half_width_us']:.1f} us")
        print(f"          alpha = {a['alpha']:.3f}"
              f"  95% CI [{a['alpha_ci95'][0]:.3f}, {a['alpha_ci95'][1]:.3f}]"
              f"  excludes_zero={a['excludes_zero']}")
    print()
    print(f"  round_alignment_verified = {out['round_alignment_verified']}")
    for leg in results:
        align = leg["dose_alignment"]
        if not align:
            continue
        print(f"    {leg['leg']}: witness lines {align['witness_rounds']}"
              f" for {align['round_count']} rounds,"
              f" width-1 lines {align['width_one_lines']},"
              f" fingerprint {align['width_fingerprint_matched']},"
              f" dosed rounds {align['dosed_rounds']},"
              f" units {align['units_when_dosed']}")
        print(f"      realised width histogram"
              f" {align['width_histogram_observed']}")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
