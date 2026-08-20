#!/usr/bin/env python3
"""E59 rung 4 analysis: score the palindrome end-to-end legs.

Reads the rung 4 run directories under `.mlxfast-private/e59-e2e/runs/*`,
groups the legs by arm, and reports the candidate move on both bases the
assignment names:

  * leg basis        -- the whole timed MTP leg, seed prefill included, which
                        is what `--local-iterate` divides by the token count;
  * round-cost basis -- the same leg with the measured seed prefill removed,
                        which is the basis that converts to rank (189(D)).

  python3 research/e59_e2e_analyze.py --out research/e59-artifacts/e59-e2e-metrics.json

Three things decide the verdicts, and all three are session-local by design.

1.  The noise bar is this session's own same-arm spread, taken at the leg
    separation the contrast actually has. The old `0.0629 %` constant is
    withdrawn: it came from adjacent legs, and ledger 198 measured the same-
    binary spread growing with separation (0.0032 % adjacent, 0.1147 % three
    apart, 0.1634 % five apart). A palindrome measures all three separations
    for free, so the session supplies its own bar.

2.  The arm effect is an ordinary least-squares fit of `time ~ arm +
    leg_position`. A two-block difference of means confounds the arm with its
    position; the regression carries position as a covariate and reports the
    residual degrees of freedom honestly.

3.  The decision is taken on the CELL the leg implies, not on the leg. This
    host's width histogram is not the ranked one: it under-weights M=5 and M=6
    by about 2.04x, so a small leg number cannot refute a mechanism the ranked
    mixture weights far more heavily. The leg effect is therefore divided by
    the M=5 share of leg round cost and compared with the cell gate.

Exit code 0 means every preregistered rung 4 check passed. Exit code 2 means a
stop rule fired.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e59_ols import fit_arm_position  # noqa: E402
from e59_wandb_log import read_leg  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / ".mlxfast-private/e59-e2e/runs"
PREREG = REPO / "research/e59-artifacts/e59-rung4-prereg.json"
PREREG_AMENDMENT = REPO / "research/e59-artifacts/e59-rung4-prereg-amendment.json"
CELLS = REPO / "research/e59-artifacts/e59-rung4-cells.json"

BASE_ARM = "shipped"
# `m5_rbx` was dropped from the leg session on the advisor's 2026-08-20T02:43Z
# instruction. `t55` turns M=5 into a single {5} group, so after `t55` there is
# nothing left at M=5 for a row-block route to win, and `m5_rbx` alone would
# move the leg by about -0.16 %, at or below the session null.
TREATMENT_ARMS = ("t55",)

# Round cost by width from E1, and the width histogram of the E60 512-token
# candidate leg. Their product is the divisor the advisor preregistered: the
# share of leg round cost that this host actually spends at M=5.
E1_ROUND_US = {4: 91288, 5: 115691, 6: 134668, 7: 154169, 8: 172827, 9: 184970}
E60_LEG_ROUNDS = {4: 5, 5: 7, 6: 17, 7: 3, 8: 3, 9: 41}
LOCAL_M5_SHARE = 0.0668

# Cell gate, on the cell the leg implies. Preregistered by the advisor to
# replace the old `-2.0 %` leg threshold, which this host cannot resolve.
ADVANCE_IMPLIED_CELL_PCT = -6.0
KILL_IMPLIED_CELL_PCT = -2.0

# What stage A measured on the whole-table cell palindrome, and what it
# predicts for this leg once the local M=5 share and the QMV share of the leg
# are applied. `QMV_SHARE_OF_LEG` is the ranked figure from
# `research/dilution_basis.py`; on this host it is an estimate, so the
# prediction is reported and never gates anything.
STAGE_A_CELL_PCT = {"t55": -20.209, "m5_rbx": -13.431}
QMV_SHARE_OF_LEG = 0.82127
PREDICTED_LEG_PCT = {
    arm: round(pct * LOCAL_M5_SHARE * QMV_SHARE_OF_LEG, 4)
    for arm, pct in STAGE_A_CELL_PCT.items()
}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def spread_pct(values: list[float]) -> float:
    """Peak-to-peak replicate spread as a percent of the mean."""
    if len(values) < 2:
        return 0.0
    return 100.0 * (max(values) - min(values)) / mean(values)


def collect(tag_prefix: str) -> list[dict]:
    legs = []
    for path in sorted(RUNS.glob("*")):
        if not path.name.startswith(tag_prefix):
            continue
        if not (path / "score.json").exists():
            continue
        rec = read_leg(path)
        rec["order"] = rec["started"] or path.name
        legs.append(rec)
    legs.sort(key=lambda r: r["order"])
    return legs


def same_arm_spreads_by_separation(legs: list[dict], field: str) -> dict:
    """Every same-arm leg pair, keyed by how far apart the two legs ran.

    This is the session's own noise model. Each entry compares two legs that
    ran the identical binary, so anything it reports is drift plus measurement
    noise and nothing else.
    """
    pairs = collections.defaultdict(list)
    for left, right in itertools.combinations(legs, 2):
        if left["arm"] != right["arm"]:
            continue
        separation = right["leg_position"] - left["leg_position"]
        delta = 100.0 * abs(right[field] - left[field]) / left[field]
        pairs[separation].append(
            {
                "arm": left["arm"],
                "tags": [left["tag"], right["tag"]],
                "abs_delta_pct": delta,
            }
        )
    return {
        str(sep): {
            "pairs": entries,
            "max_abs_delta_pct": max(e["abs_delta_pct"] for e in entries),
        }
        for sep, entries in sorted(pairs.items())
    }


def ols_arm_position(legs: list[dict], field: str) -> dict:
    """Fit `value ~ arm + leg_position` with the base arm as the reference."""
    fit = fit_arm_position(
        [{"arm": leg["arm"], "position": leg["leg_position"],
          "value": leg[field], "label": leg["tag"]}
         for leg in legs],
        BASE_ARM,
    )
    fit["field"] = field
    return fit


def m5_share_of_round_cost(histogram: dict) -> dict:
    """Price a measured width histogram with the E1 round costs.

    The gate divisor is preregistered, so this never replaces it. It says
    whether this session's own mixture matches the one the divisor came from,
    which is the only way to see that the conversion is still honest.
    """
    priced = {}
    for width, rounds in histogram.items():
        m = int(width)
        cost = E1_ROUND_US.get(m)
        if cost is None:
            continue
        priced[m] = cost * int(rounds)
    total = sum(priced.values())
    if not total:
        return {"available": False}
    return {
        "available": True,
        "priced_round_us_by_width": priced,
        "total_priced_round_us": total,
        "m5_share": priced.get(5, 0) / total,
        "widths_without_e1_cost": sorted(
            int(w) for w in histogram if int(w) not in E1_ROUND_US),
    }


def implied_cell(leg_pct: float, round_cost_pct: float) -> dict:
    """Convert one leg effect into the M=5 cell effect it implies.

    Two divisions, both reported, both using the preregistered divisor.
    `from_leg` is the advisor's literal rule and is deliberately conservative:
    the raw leg carries seed prefill and every non-QMV round cost, neither of
    which the M=5 kernel can move, so dividing it by an M=5 share of round cost
    understates the cell. `from_round_cost` removes the measured prefill first
    and is the sharper read. The gate reads the conservative one.
    """
    return {
        "divisor": LOCAL_M5_SHARE,
        "raw_leg_pct": leg_pct,
        "raw_round_cost_pct": round_cost_pct,
        "implied_cell_pct": leg_pct / LOCAL_M5_SHARE,
        "implied_cell_pct_from_round_cost": round_cost_pct / LOCAL_M5_SHARE,
    }


def cell_gate(implied_cell_pct: float, sign_stable: bool) -> str:
    if implied_cell_pct <= ADVANCE_IMPLIED_CELL_PCT and sign_stable:
        return "advance"
    if implied_cell_pct <= KILL_IMPLIED_CELL_PCT and sign_stable:
        return "report_only"
    return "kill"


def summarise(arm: str, legs: list[dict]) -> dict:
    hists = [tuple(sorted(leg["width_histogram"].items())) for leg in legs]
    return {
        "arm": arm,
        "legs": len(legs),
        "tags": [leg["tag"] for leg in legs],
        "leg_positions": [leg["leg_position"] for leg in legs],
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
        "entry_temp_c": [leg["gpu_temp_entry_c"] for leg in legs],
        "exit_temp_c": [leg["gpu_temp_exit_c"] for leg in legs],
        "cool_gate_requested": sorted({leg["cool_gate_requested"] for leg in legs}),
        "replicate_spread_pct": {
            "serial_leg": spread_pct(
                [leg["serial_seconds_per_token"] for leg in legs]),
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
    ap.add_argument("--tag-prefix", default="e59-r4-",
                    help="only analyse legs whose run directory starts with this")
    args = ap.parse_args()

    every_leg = collect(args.tag_prefix)
    if not every_leg:
        raise SystemExit("e59_e2e_analyze: no %s* legs under %s"
                         % (args.tag_prefix, RUNS))

    warmups = [leg for leg in every_leg if leg["warmup_discarded"] == "1"]
    legs = [leg for leg in every_leg if leg["warmup_discarded"] != "1"]
    if not legs:
        raise SystemExit("e59_e2e_analyze: every leg is a discarded warm-up")
    # Positions number the timed set only, so the regression covariate
    # describes the counterbalanced sequence and not the throwaway leg.
    for position, leg in enumerate(legs, start=1):
        leg["leg_position"] = position

    by_arm: dict[str, list[dict]] = collections.defaultdict(list)
    for leg in legs:
        by_arm[leg["arm"]].append(leg)

    arms = {arm: summarise(arm, rows) for arm, rows in by_arm.items()}
    if BASE_ARM not in arms:
        raise SystemExit("e59_e2e_analyze: no %s legs; the contrast has no base"
                         % BASE_ARM)
    base = arms[BASE_ARM]

    fields = ("mtp_seconds_per_token",
              "mtp_seconds_per_token_prefill_removed",
              "serial_seconds_per_token")
    nulls = {field: same_arm_spreads_by_separation(legs, field)
             for field in fields}

    def bar_for(field: str, separation: int) -> float:
        """Largest same-arm spread at this separation or wider.

        Wider separations bound narrower ones: if two identical legs five apart
        differ by x, a contrast three apart cannot claim a tighter bar without
        new evidence.
        """
        candidates = [entry["max_abs_delta_pct"]
                      for sep, entry in nulls[field].items()
                      if int(sep) >= separation]
        return max(candidates) if candidates else float("nan")

    # In a palindrome every arm shares the same mean position, so an arm-vs-base
    # contrast spans the whole session and the widest same-arm separation is the
    # honest bar.
    separations = [int(sep) for sep in nulls["mtp_seconds_per_token"]]
    widest = max(separations) if separations else 1
    bar = {
        "leg": bar_for("mtp_seconds_per_token", widest),
        "round_cost": bar_for("mtp_seconds_per_token_prefill_removed", widest),
        "serial": bar_for("serial_seconds_per_token", widest),
        "separation_used": widest,
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

    regressions = {
        field: ols_arm_position(legs, field)
        for field in ("mtp_seconds_per_token",
                      "mtp_seconds_per_token_prefill_removed")
    }

    entry_temps = [float(t) for leg in legs
                   if (t := leg["gpu_temp_entry_c"]) not in (None, "")]
    real_gate = all(leg["cool_gate_requested"] == "1" for leg in legs)

    leg_fit = regressions["mtp_seconds_per_token"]

    verdicts: dict = {}
    conversions: dict = {}
    for arm in TREATMENT_ARMS:
        rec = arms.get(arm)
        if rec is None:
            continue
        if leg_fit.get("fitted") and f"arm[{arm}]" in leg_fit["terms"]:
            term = leg_fit["terms"][f"arm[{arm}]"]
            verdicts[f"{arm}_leg_pct_regression"] = term["estimate_pct_of_base"]
            verdicts[f"{arm}_leg_t_regression"] = term["t"]
        verdicts[f"{arm}_leg_pct"] = rec["delta_vs_base_leg_pct"]
        verdicts[f"{arm}_round_cost_pct"] = rec["delta_vs_base_round_cost_pct"]
        verdicts[f"{arm}_beats_bar_leg"] = bool(
            rec["delta_vs_base_leg_pct"] < -bar["leg"])
        verdicts[f"{arm}_vs_prereg_leg_pct"] = round(
            rec["delta_vs_base_leg_pct"] - PREDICTED_LEG_PCT[arm], 4)

        # The gate. Every timed leg of this arm must move the same way, or the
        # sign is not stable and the conversion means nothing.
        per_leg = [delta_pct(leg["mtp_seconds_per_token"],
                             base["mtp_seconds_per_token_mean"])
                   for leg in by_arm[arm]]
        sign_stable = all(p < 0 for p in per_leg) or all(p > 0 for p in per_leg)
        conv = implied_cell(rec["delta_vs_base_leg_pct"],
                            rec["delta_vs_base_round_cost_pct"])
        conv["per_leg_pct"] = per_leg
        conv["sign_stable_across_palindrome"] = sign_stable
        conv["gate"] = cell_gate(conv["implied_cell_pct"], sign_stable)
        conv["stage_a_measured_cell_pct"] = STAGE_A_CELL_PCT.get(arm)
        conversions[arm] = conv

        verdicts[f"{arm}_implied_cell_pct"] = conv["implied_cell_pct"]
        verdicts[f"{arm}_implied_cell_pct_from_round_cost"] = conv[
            "implied_cell_pct_from_round_cost"]
        verdicts[f"{arm}_sign_stable"] = sign_stable
        verdicts[f"{arm}_cell_gate"] = conv["gate"]

    verdicts["all_arms_token_exact"] = all(
        rec["all_tokens_matched"] for rec in arms.values())
    verdicts["all_arms_row_ledger_closes"] = all(
        rec["row_ledger_closes"] for rec in arms.values())
    verdicts["all_arms_same_width_histogram"] = len(
        {json.dumps(rec["width_histogram"], sort_keys=True)
         for rec in arms.values()}) == 1
    verdicts["all_legs_wired_residency_inactive"] = all(
        leg["wired_residency_active"] == "false" for leg in legs)
    verdicts["all_legs_ranked_geometry"] = all(
        leg["mlx_max_ops_per_buffer"] == "50"
        and leg["mlx_max_mb_per_buffer"] == "512"
        and leg["startup_memory_profile"] == "full" for leg in legs)
    verdicts["warmup_leg_declared_and_dropped"] = len(warmups) >= 1

    # Arm certification by Mach-O section digest. A routing-only arm must leave
    # `__TEXT,__text` identical and move `__TEXT,__cstring`, because the whole
    # change is one template argument inside a JIT kernel source string. The
    # whole-file digest cannot say this: it also moves with LC_UUID and the
    # code-signature slots on every relink.
    digests = {
        "text_by_arm": {arm: sorted({leg["worker_text_sha256"] for leg in rows})
                        for arm, rows in by_arm.items()},
        "cstring_by_arm": {
            arm: sorted({leg["worker_cstring_sha256"] for leg in rows})
            for arm, rows in by_arm.items()},
        "file_by_arm": {arm: sorted({leg["worker_sha256"] for leg in rows})
                        for arm, rows in by_arm.items()},
    }
    digests["available"] = all(
        d is not None for arm in digests["text_by_arm"]
        for d in digests["text_by_arm"][arm] + digests["cstring_by_arm"][arm])
    if digests["available"]:
        verdicts["each_arm_has_one_text_digest"] = all(
            len(v) == 1 for v in digests["text_by_arm"].values())
        verdicts["each_arm_has_one_cstring_digest"] = all(
            len(v) == 1 for v in digests["cstring_by_arm"].values())
        verdicts["arms_share_one_text_digest"] = len(
            {v[0] for v in digests["text_by_arm"].values() if v}) == 1
        verdicts["arms_have_distinct_cstring_digests"] = len(
            {v[0] for v in digests["cstring_by_arm"].values() if v}) == len(by_arm)

    # The divisor audit. The gate divisor is preregistered from the advisor's
    # E60 512-token histogram; this prices the histogram the session actually
    # produced with the same E1 round costs and reports the difference.
    measured_share = {arm: m5_share_of_round_cost(rec["width_histogram"])
                      for arm, rec in arms.items()}
    prereg_share = m5_share_of_round_cost(E60_LEG_ROUNDS)

    payload = {
        "base_arm": BASE_ARM,
        "tag_prefix": args.tag_prefix,
        "arms": arms,
        "leg_order": [
            {"tag": leg["tag"], "arm": leg["arm"],
             "leg_position": leg["leg_position"], "started": leg["started"],
             "gpu_temp_entry_c": leg["gpu_temp_entry_c"],
             "gpu_temp_exit_c": leg["gpu_temp_exit_c"],
             "wired_residency_active": leg["wired_residency_active"],
             "mlx_max_ops_per_buffer": leg["mlx_max_ops_per_buffer"],
             "metallib_source_fingerprint": leg["metallib_source_fingerprint"]}
            for leg in legs],
        "discarded_warmup_legs": [
            {"tag": leg["tag"], "arm": leg["arm"],
             "gpu_temp_entry_c": leg["gpu_temp_entry_c"],
             "gpu_temp_exit_c": leg["gpu_temp_exit_c"],
             "mtp_seconds_per_token": leg["mtp_seconds_per_token"]}
            for leg in warmups],
        "session_null_by_separation_pct": nulls,
        "bar_pct": bar,
        "regression_time_by_arm_and_position": regressions,
        "worker_section_digests": digests,
        "cell_conversion": conversions,
        "m5_share_audit": {
            "divisor_used": LOCAL_M5_SHARE,
            "prereg_e60_histogram": E60_LEG_ROUNDS,
            "prereg_share_recomputed": prereg_share,
            "measured_share_by_arm": measured_share,
            "e1_round_us": E1_ROUND_US,
        },
        "predicted_pct": {
            "leg_basis": PREDICTED_LEG_PCT,
            "stage_a_cell_pct": STAGE_A_CELL_PCT,
            "qmv_share_of_leg": QMV_SHARE_OF_LEG,
            "advance_implied_cell_pct": ADVANCE_IMPLIED_CELL_PCT,
            "kill_implied_cell_pct": KILL_IMPLIED_CELL_PCT,
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
        "prereg_amendment": json.loads(PREREG_AMENDMENT.read_text()),
        "stage_a_cells": (json.loads(CELLS.read_text())["routes"]
                          if CELLS.exists() else None),
    }

    print("E59 rung 4 legs, in run order:")
    print("  %-24s %-3s %-10s %-20s %-7s %-7s"
          % ("tag", "pos", "arm", "started", "in C", "out C"))
    for leg in warmups:
        print("  %-24s %-3s %-10s %-20s %-7s %-7s  (discarded warm-up)"
              % (leg["tag"], "-", leg["arm"], leg["started"],
                 leg["gpu_temp_entry_c"], leg["gpu_temp_exit_c"]))
    for leg in legs:
        print("  %-24s %-3d %-10s %-20s %-7s %-7s"
              % (leg["tag"], leg["leg_position"], leg["arm"], leg["started"],
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

    print("\nsession null, same arm, same binary, by leg separation")
    for field, table in nulls.items():
        for sep, entry in table.items():
            arms_seen = ",".join(sorted({p["arm"] for p in entry["pairs"]}))
            print("  %-38s sep %-2s %-12s max |d| %.4f %%"
                  % (field, sep, arms_seen, entry["max_abs_delta_pct"]))

    print("\ndeltas vs %s   (bar at separation %d: leg %.4f %%, round-cost %.4f %%,"
          " serial %.4f %%)"
          % (BASE_ARM, bar["separation_used"], bar["leg"], bar["round_cost"],
             bar["serial"]))
    print("  %-12s %-12s %-14s %-12s" % ("arm", "leg %", "round-cost %", "serial %"))
    for arm, rec in arms.items():
        if arm == BASE_ARM:
            continue
        print("  %-12s %-+12.4f %-+14.4f %-+12.4f"
              % (arm, rec["delta_vs_base_leg_pct"],
                 rec["delta_vs_base_round_cost_pct"], rec["serial_leg_delta_pct"]))
    print("  round-cost basis is the one that converts to rank (189(D)).")

    print("\nregression  value ~ arm + leg_position")
    for field, fit in regressions.items():
        if not fit.get("fitted"):
            print("  %-38s NOT FITTED: %s" % (field, fit.get("reason")))
            continue
        print("  %s  (n=%d, residual dof=%d, residual sd=%.3e)"
              % (field, fit["n"], fit["residual_dof"], fit["residual_sd"]))
        for name, term in fit["terms"].items():
            pct = term["estimate_pct_of_base"]
            pct_text = "" if pct is None else "  (%+.4f %% of base)" % pct
            t_text = "  t=%+.4f" % term["t"] if term["t"] is not None else ""
            print("    %-18s %+.6e +/- %.3e%s%s"
                  % (name, term["estimate"], term["std_error"], t_text, pct_text))

    print("\nM=5 share of leg round cost, priced with the E1 round costs")
    print("  %-22s %s" % ("preregistered divisor", LOCAL_M5_SHARE))
    if prereg_share["available"]:
        print("  %-22s %.5f  (E60 512-token histogram %s)"
              % ("recomputed", prereg_share["m5_share"], E60_LEG_ROUNDS))
    for arm, share in measured_share.items():
        if share["available"]:
            print("  %-22s %.5f  (this session, %s)"
                  % ("measured " + arm, share["m5_share"],
                     arms[arm]["width_histogram"]))

    print("\nCELL GATE   implied_cell_pct = leg_pct / %.4f" % LOCAL_M5_SHARE)
    print("  advance if <= %.1f %% with a stable sign, report-only down to"
          " %.1f %%, kill above it"
          % (ADVANCE_IMPLIED_CELL_PCT, KILL_IMPLIED_CELL_PCT))
    for arm, conv in conversions.items():
        print("  %-10s leg %+.4f %% -> cell %+.3f %%   (round-cost basis"
              " %+.4f %% -> cell %+.3f %%)"
              % (arm, conv["raw_leg_pct"], conv["implied_cell_pct"],
                 conv["raw_round_cost_pct"],
                 conv["implied_cell_pct_from_round_cost"]))
        print("  %-10s per-leg %s  sign stable %s  GATE %s"
              % ("", ["%+.4f" % p for p in conv["per_leg_pct"]],
                 conv["sign_stable_across_palindrome"], conv["gate"].upper()))
        if conv["stage_a_measured_cell_pct"] is not None:
            print("  %-10s stage A measured the same cell at %+.3f %%"
                  % ("", conv["stage_a_measured_cell_pct"]))

    if digests["available"]:
        print("\nworker section digests (first 12 hex)")
        for arm in sorted(digests["text_by_arm"]):
            print("  %-10s __text %s  __cstring %s"
                  % (arm,
                     ",".join(d[:12] for d in digests["text_by_arm"][arm]),
                     ",".join(d[:12] for d in digests["cstring_by_arm"][arm])))

    print("\nVERDICTS")
    for name, value in verdicts.items():
        print("  %-42s %s" % (name, value))

    rc = 0
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
