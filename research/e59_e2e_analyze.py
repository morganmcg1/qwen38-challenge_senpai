#!/usr/bin/env python3
"""E59 rung 4 analysis: score the counterbalanced end-to-end legs.

Reads every run directory under `.mlxfast-private/e59-e2e/runs/*`, groups the
legs by arm, and reports the candidate move on both bases the assignment names:

  * leg basis        -- the whole timed MTP leg, seed prefill included, which
                        is what `--local-iterate` divides by the token count;
  * round-cost basis -- the same leg with the measured seed prefill removed,
                        which is the basis that converts to rank (189(D)).

  python3 research/e59_e2e_analyze.py --out research/e59-artifacts/e59-e2e-metrics.json

Exit code 0 means every pre-registered rung 4 check passed. Exit code 2 means a
stop rule fired.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e59_wandb_log import read_leg  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / ".mlxfast-private/e59-e2e/runs"
PREREG = REPO / "research/e59-artifacts/e59-prereg.json"

BASE_ARM = "shipped"
# Pre-registered in PR 62: the local same-build null floor for a --local-iterate
# leg. The session's own base-to-base replicate spread can only raise the bar.
NULL_FLOOR_PCT = 0.0629
# Pre-registered route prediction, both bases.
PREDICTED_LEG_PCT = -0.34
PREDICTED_ROUND_COST_PCT = -0.44
# Pre-registered additive shape of the ceiling dose under a multiplicative null.
CEILING_ADDITIVE_LOW_PCT = 0.99
CEILING_ADDITIVE_HIGH_PCT = 2.33


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def spread_pct(values: list[float]) -> float:
    """Peak-to-peak replicate spread as a percent of the mean."""
    if len(values) < 2:
        return 0.0
    return 100.0 * (max(values) - min(values)) / mean(values)


def collect() -> list[dict]:
    legs = []
    for path in sorted(RUNS.glob("*")):
        if not (path / "score.json").exists():
            continue
        rec = read_leg(path)
        rec["order"] = rec["started"] or path.name
        legs.append(rec)
    legs.sort(key=lambda r: r["order"])
    return legs


def summarise(arm: str, legs: list[dict]) -> dict:
    hists = [tuple(sorted(leg["width_histogram"].items())) for leg in legs]
    temps = [leg["gpu_temp_entry_c"] for leg in legs]
    return {
        "arm": arm,
        "legs": len(legs),
        "tags": [leg["tag"] for leg in legs],
        "serial_seconds_per_token_mean": mean(
            leg["serial_seconds_per_token"] for leg in legs),
        "serial_seconds_per_token_all": [
            leg["serial_seconds_per_token"] for leg in legs],
        "mtp_seconds_per_token_mean": mean(
            leg["mtp_seconds_per_token"] for leg in legs),
        "mtp_seconds_per_token_all": [
            leg["mtp_seconds_per_token"] for leg in legs],
        "mtp_seconds_per_token_prefill_removed_mean": mean(
            leg["mtp_seconds_per_token_prefill_removed"] for leg in legs),
        "mtp_seconds_per_token_prefill_removed_all": [
            leg["mtp_seconds_per_token_prefill_removed"] for leg in legs],
        "serial_seconds_per_token_prefill_removed_mean": mean(
            leg["serial_seconds_per_token_prefill_removed"] for leg in legs),
        "seed_prefill_seconds_mean": mean(
            leg["mtp_seed_prefill_seconds"] for leg in legs),
        "serial_seed_prefill_seconds_mean": mean(
            leg["serial_seed_prefill_seconds"] for leg in legs),
        "mtp_decode_seconds_mean": mean(leg["mtp_decode_seconds"] for leg in legs),
        "serial_decode_seconds_mean": mean(
            leg["serial_decode_seconds"] for leg in legs),
        "score_mean": mean(leg["score"] for leg in legs),
        "round_count": legs[0]["round_count"],
        "round_count_identical_across_legs": len(
            {leg["round_count"] for leg in legs}) == 1,
        "width_histogram": legs[0]["width_histogram"],
        "width_histogram_identical_across_legs": len(set(hists)) == 1,
        "declared_rows_total": legs[0]["declared_rows_total"],
        "row_ledger_closes": all(leg["row_ledger_closes"] for leg in legs),
        "all_tokens_matched": all(leg["all_tokens_matched"] for leg in legs),
        "public_drift_tripwire_passed": all(
            leg["public_drift_tripwire_passed"] for leg in legs),
        "accepted_draft_rate_mean": mean(
            leg["accepted_draft_rate"] for leg in legs),
        "effective_mean_draft_len": legs[0]["effective_mean_draft_len"],
        "entry_temp_c": temps,
        "exit_temp_c": [leg["gpu_temp_exit_c"] for leg in legs],
        "cool_gate_requested": sorted({leg["cool_gate_requested"] for leg in legs}),
        "replicate_spread_pct": {
            "serial_leg": spread_pct([leg["serial_seconds_per_token"] for leg in legs]),
            "mtp_leg": spread_pct([leg["mtp_seconds_per_token"] for leg in legs]),
            "mtp_round_cost": spread_pct(
                [leg["mtp_seconds_per_token_prefill_removed"] for leg in legs]),
        },
    }


def delta_pct(candidate: float, base: float) -> float:
    return 100.0 * (candidate / base - 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-e2e-metrics.json")
    args = ap.parse_args()

    legs = collect()
    if not legs:
        raise SystemExit("e59_e2e_analyze: no legs under %s" % RUNS)

    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for leg in legs:
        by_arm[leg["arm"]].append(leg)

    arms = {arm: summarise(arm, rows) for arm, rows in by_arm.items()}
    if BASE_ARM not in arms:
        raise SystemExit("e59_e2e_analyze: no %s legs; the contrast has no base"
                         % BASE_ARM)
    base = arms[BASE_ARM]

    # The bar is the larger of the pre-registered null floor and this session's
    # own base-to-base replicate spread. A session that drifted cannot be
    # rescued by the smaller pre-registered number.
    bar = {
        "leg": max(NULL_FLOOR_PCT, base["replicate_spread_pct"]["mtp_leg"]),
        "round_cost": max(NULL_FLOOR_PCT,
                          base["replicate_spread_pct"]["mtp_round_cost"]),
        "serial": max(NULL_FLOOR_PCT, base["replicate_spread_pct"]["serial_leg"]),
    }

    for arm, rec in arms.items():
        rec["delta_vs_base_leg_pct"] = delta_pct(
            rec["mtp_seconds_per_token_mean"], base["mtp_seconds_per_token_mean"])
        rec["delta_vs_base_round_cost_pct"] = delta_pct(
            rec["mtp_seconds_per_token_prefill_removed_mean"],
            base["mtp_seconds_per_token_prefill_removed_mean"])
        rec["serial_leg_delta_pct"] = delta_pct(
            rec["serial_seconds_per_token_mean"],
            base["serial_seconds_per_token_mean"])
        rec["seed_prefill_delta_pct"] = delta_pct(
            rec["seed_prefill_seconds_mean"], base["seed_prefill_seconds_mean"])
        rec["exceeds_bar_leg"] = abs(rec["delta_vs_base_leg_pct"]) > bar["leg"]
        rec["exceeds_bar_round_cost"] = (
            abs(rec["delta_vs_base_round_cost_pct"]) > bar["round_cost"])
        rec["exceeds_bar_serial"] = abs(rec["serial_leg_delta_pct"]) > bar["serial"]

    entry_temps = [float(t) for leg in legs
                   if (t := leg["gpu_temp_entry_c"]) not in (None, "")]
    real_gate = all(leg["cool_gate_requested"] == "1" for leg in legs)

    verdicts: dict = {}

    # Stop rule 4: a ceiling dose that moves one leg but not the other means the
    # ceiling cost is width dependent, and the additive pricing model that the
    # deficit arithmetic rests on does not hold.
    ceiling = arms.get("ceil_only")
    if ceiling is not None:
        moved_mtp = ceiling["exceeds_bar_leg"]
        moved_serial = ceiling["exceeds_bar_serial"]
        verdicts["ceiling_moved_mtp_leg"] = moved_mtp
        verdicts["ceiling_moved_serial_leg"] = moved_serial
        verdicts["ceiling_cost_is_width_dependent"] = moved_mtp != moved_serial
        verdicts["stop_rule_4_fired"] = moved_mtp != moved_serial
        verdicts["ceiling_dose_leg_pct"] = ceiling["delta_vs_base_leg_pct"]
        verdicts["ceiling_dose_serial_pct"] = ceiling["serial_leg_delta_pct"]
        verdicts["ceiling_dose_inside_prereg_additive_band"] = bool(
            CEILING_ADDITIVE_LOW_PCT <= ceiling["delta_vs_base_leg_pct"]
            <= CEILING_ADDITIVE_HIGH_PCT)

    route = [arm for arm in arms if arm.startswith("m5_")]
    for arm in route:
        rec = arms[arm]
        verdicts[f"{arm}_leg_pct"] = rec["delta_vs_base_leg_pct"]
        verdicts[f"{arm}_round_cost_pct"] = rec["delta_vs_base_round_cost_pct"]
        verdicts[f"{arm}_beats_bar_leg"] = bool(
            rec["delta_vs_base_leg_pct"] < -bar["leg"])
        verdicts[f"{arm}_beats_bar_round_cost"] = bool(
            rec["delta_vs_base_round_cost_pct"] < -bar["round_cost"])
        verdicts[f"{arm}_vs_prereg_leg_pct"] = round(
            rec["delta_vs_base_leg_pct"] - PREDICTED_LEG_PCT, 4)
        verdicts[f"{arm}_vs_prereg_round_cost_pct"] = round(
            rec["delta_vs_base_round_cost_pct"] - PREDICTED_ROUND_COST_PCT, 4)

    verdicts["all_arms_token_exact"] = all(
        rec["all_tokens_matched"] for rec in arms.values())
    verdicts["all_arms_row_ledger_closes"] = all(
        rec["row_ledger_closes"] for rec in arms.values())
    verdicts["all_arms_same_width_histogram"] = len(
        {json.dumps(rec["width_histogram"], sort_keys=True)
         for rec in arms.values()}) == 1

    payload = {
        "base_arm": BASE_ARM,
        "arms": arms,
        "leg_order": [
            {"tag": leg["tag"], "arm": leg["arm"], "started": leg["started"],
             "gpu_temp_entry_c": leg["gpu_temp_entry_c"],
             "gpu_temp_exit_c": leg["gpu_temp_exit_c"]}
            for leg in legs],
        "bar_pct": bar,
        "bar_inputs": {
            "prereg_null_floor_pct": NULL_FLOOR_PCT,
            "base_replicate_spread_pct": base["replicate_spread_pct"],
        },
        "predicted_pct": {
            "leg_basis": PREDICTED_LEG_PCT,
            "round_cost_basis": PREDICTED_ROUND_COST_PCT,
            "ceiling_additive_band": [CEILING_ADDITIVE_LOW_PCT,
                                      CEILING_ADDITIVE_HIGH_PCT],
        },
        "entry_temperature_spread_c": (
            round(max(entry_temps) - min(entry_temps), 2) if entry_temps else None),
        "entry_temperature_c": entry_temps,
        "entry_temperature_sd_c": (
            round(statistics.pstdev(entry_temps), 3) if len(entry_temps) > 1
            else None),
        "cool_gate_passed_real_gate": real_gate,
        "gate_qualified_for_timing": real_gate,
        "official_or_ranked_score": False,
        "verdicts": verdicts,
        "prereg": json.loads(PREREG.read_text()),
    }

    print("E59 rung 4 legs, in run order:")
    print("  %-22s %-12s %-20s %-7s %-7s" %
          ("tag", "arm", "started", "in C", "out C"))
    for leg in legs:
        print("  %-22s %-12s %-20s %-7s %-7s"
              % (leg["tag"], leg["arm"], leg["started"],
                 leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"]))

    print("\nper arm (mean over legs)")
    print("  %-12s %-3s %-12s %-12s %-12s %-10s"
          % ("arm", "n", "serial s/tok", "mtp s/tok", "mtp round", "prefill s"))
    for arm, rec in arms.items():
        print("  %-12s %-3d %-12.6f %-12.6f %-12.6f %-10.4f"
              % (arm, rec["legs"], rec["serial_seconds_per_token_mean"],
                 rec["mtp_seconds_per_token_mean"],
                 rec["mtp_seconds_per_token_prefill_removed_mean"],
                 rec["seed_prefill_seconds_mean"]))

    print("\ndeltas vs %s   (bar: leg %.4f %%, round-cost %.4f %%, serial %.4f %%)"
          % (BASE_ARM, bar["leg"], bar["round_cost"], bar["serial"]))
    print("  %-12s %-12s %-14s %-12s"
          % ("arm", "leg %", "round-cost %", "serial %"))
    for arm, rec in arms.items():
        if arm == BASE_ARM:
            continue
        print("  %-12s %-+12.4f %-+14.4f %-+12.4f"
              % (arm, rec["delta_vs_base_leg_pct"],
                 rec["delta_vs_base_round_cost_pct"], rec["serial_leg_delta_pct"]))
    print("  round-cost basis is the one that converts to rank (189(D)).")

    print("\nVERDICTS")
    for name, value in verdicts.items():
        print("  %-42s %s" % (name, value))

    rc = 2 if verdicts.get("stop_rule_4_fired") else 0
    if not verdicts["all_arms_token_exact"]:
        print("\nHARD STOP: an arm did not match the golden token stream.")
        rc = 2
    if not verdicts["all_arms_row_ledger_closes"]:
        print("\nHARD STOP: an arm's row ledger does not close.")
        rc = 2

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print("\nwrote %s" % out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
