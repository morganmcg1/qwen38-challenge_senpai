#!/usr/bin/env python3
"""E158 R1.C -- harvest the precision-island ABBA legs into one artifact.

    usage: python3 research/e158_r1_harvest.py [--out PATH]

Reads whichever of the two sessions have run:

  perprompt  .mlxfast-private/e128/runs-e158r1-<id>-<slot>-<arm>/<id>/
             `mtp-timed` legs. Candidate MTP leg only, so
             `parent_measured_seconds_per_token` is the leg RULE 177 prices.
             Ungated, ABBA-counterbalanced, entry and exit temperature kept.

  gated      research/out/e158r1g-<slot>-<arm>/
             `--local-iterate` legs behind the real 40 C gate, on the one public
             fixture. Carries the serial control leg as well, so the local
             serial-to-MTP ratio is available. That ratio is admissible for this
             change only because the causal path is confined to the candidate
             MTP leg: the serial control leg never drafts.

ABBA estimator. The slot order is all, none, none, all. Slots 1 and 4 average to
position 2.5 and slots 2 and 3 average to position 2.5, so a linear drift in
session position cancels exactly in mean(none) - mean(all).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = ["all", "none", "none", "all"]
PROMPTS = ["beagle_a", "essays_montaigne", "benchfixture"]
CANARY_PROMPT = "plutarch_lives"
SWEEP_PROMPT = "beagle_a"
SWEEP_DEPTHS = [1, 2, 3, 4, 6, 8]

# Advisor F2, 2026-08-23: score the weighted effect as
# 0.474 * (beagle) + 0.526 * (a top-four-like prompt). `beagle` is the measured
# median anchor at weight 0.4741. The top-four-like prompt must draft like the
# four prompts that scramble across ranks 5-8, which means edl about 5.0-6.1;
# `benchfixture` sits at 6.36 and `essays_montaigne` at 3.44, so `benchfixture`
# is the qualifying proxy and the essays pairing is reported beside it as the
# harder-text sensitivity.
WEIGHTED_PAIRS = {
    "beagle_plus_top_four_like": {"beagle_a": 0.4741, "benchfixture": 0.5259},
    "beagle_plus_essays": {"beagle_a": 0.4741, "essays_montaigne": 0.5259},
}

# Absolute-mtp noise floor for this host, in percent (advisor, 2026-08-23).
NOISE_FLOOR_PCT = 0.039


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def read_json(path: pathlib.Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def witness(directory: pathlib.Path) -> str | None:
    path = directory / "witness.txt"
    if not path.is_file():
        return None
    lines = sorted(
        {
            line.strip()
            for line in path.read_text().splitlines()
            if line.startswith("qwen-mtp-island-arm: ")
        }
    )
    if len(lines) != 1:
        return None
    return lines[0]


def timed_leg(
    session: str,
    prompt: str,
    slot: int,
    arm: str,
    runs_dir: str,
    witness_dir: str,
) -> dict | None:
    """One `mtp-timed` leg, in the F6-corrected form.

    Advisor ERROR 202: `effective_mean_draft_len` is the mean number of drafts
    PROPOSED per round, which is the schedule's decision, not the accepted
    length. A round emits `1 + accepted` tokens, so the correct arithmetic is

        rounds = N - acceptedDraftTotal
        R      = decodeSeconds / rounds
        a      = acceptedDraftTotal / rounds
        q      = effective_mean_draft_len
        mtp    = R / (1 + a)

    `q` and `a` are kept in separately named fields on purpose; conflating them
    is the error this replaces. `rounds_identity_holds` records the falsifiable
    check `rounds + acceptedDraftTotal == N` rather than assuming it, because
    the driver reports `rounds` independently.
    """
    run = ROOT / ".mlxfast-private/e128" / runs_dir / prompt
    report = read_json(run / "report.json")
    meta = read_meta(run / "meta.txt")
    if report is None:
        return None
    mtp = report["parent_measured_seconds_per_token"]
    q = report["effective_mean_draft_len"]
    rounds = report["round_count"]
    accepted = report["accepted_draft_total"]
    n_tokens = report["decode_token_count"]
    return {
        "session": session,
        "prompt": prompt,
        "slot": slot,
        "arm": arm,
        "witness": witness(ROOT / ".mlxfast-private/e158r1" / witness_dir),
        "mtp_seconds_per_token": mtp,
        "seconds_per_round_R": report["decode_seconds"] / rounds,
        "mean_accepted_per_round_a": accepted / rounds,
        "mean_proposed_per_round_q": q,
        "alpha_accept_fraction": report["accepted_draft_rate"],
        "rounds_identity_holds": rounds + accepted == n_tokens,
        "decode_seconds": report["decode_seconds"],
        "decode_token_count": n_tokens,
        "round_count": rounds,
        "non_drafting_round_count": report.get("non_drafting_round_count"),
        "effective_mean_draft_len": q,
        "accepted_draft_rate": report["accepted_draft_rate"],
        "accepted_draft_total": accepted,
        "rejected_draft_total": report["rejected_draft_total"],
        "all_tokens_matched": report["all_tokens_matched"],
        "residual_divergence_count": report["residual_divergence_count"],
        "p50_block_request_seconds": report["p50_block_request_seconds"],
        "head_provenance_sha256": (report.get("head_provenance") or {}).get(
            "sha256"
        ),
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
        "timing_valid": meta.get("timing_valid"),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
        "official_or_ranked_score": meta.get("official_or_ranked_score"),
        "base_sha": meta.get("base_sha"),
        "worker_sha256": meta.get("worker_sha256"),
        "cli_sha256": meta.get("cli_sha256"),
        "golden_sha256": meta.get("golden_sha256"),
        "prompt_sha256": meta.get("prompt_sha256"),
        "host": meta.get("host"),
        "chip": meta.get("chip"),
    }


def perprompt_legs() -> list[dict]:
    legs = []
    for prompt in PROMPTS:
        for slot, arm in enumerate(ARMS, start=1):
            leg = timed_leg(
                "perprompt",
                prompt,
                slot,
                arm,
                f"runs-e158r1-{prompt}-{slot}-{arm}",
                f"perprompt-{prompt}-{slot}-{arm}",
            )
            if leg is not None:
                legs.append(leg)
    return legs


def canary_legs() -> list[dict]:
    legs = []
    for arm in ("all", "none"):
        leg = timed_leg(
            "canary",
            CANARY_PROMPT,
            1,
            arm,
            f"runs-e158r1-canary-{arm}",
            f"canary-{CANARY_PROMPT}-{arm}",
        )
        if leg is not None:
            legs.append(leg)
    return legs


def sweep_legs() -> list[dict]:
    """Within-prompt offered-depth sweep, both arms.

    Only rows per round move. The head, prompt, golden and host are fixed, so
    this is the arm of the experiment that can separate the fixed per-round cost
    from the per-row cost.
    """
    legs = []
    for depth in SWEEP_DEPTHS:
        for arm in ("all", "none"):
            leg = timed_leg(
                "depthsweep",
                SWEEP_PROMPT,
                depth,
                arm,
                f"runs-e158r1-sweep-d{depth}-{arm}",
                f"sweep-d{depth}-{arm}",
            )
            if leg is not None:
                leg["offered_depth"] = depth
                leg["rows_per_round"] = 1.0 + leg["mean_proposed_per_round_q"]
                legs.append(leg)
    return legs


def gated_legs() -> list[dict]:
    legs = []
    for slot, arm in enumerate(ARMS, start=1):
        out = ROOT / "research/out" / f"e158r1g-{slot}-{arm}"
        score = read_json(out / "score.json")
        meta = read_meta(out / "meta.txt")
        if score is None:
            continue
        metrics = score["metrics"]
        # `effective_mean_draft_len` counts drafts PROPOSED per round. The F6
        # identity needs drafts ACCEPTED per round, which is that count scaled
        # by the acceptance rate. Using the proposed count here overstates R.
        q = metrics["effective_mean_draft_len"]
        a = q * metrics["accepted_draft_rate"]
        tokens = metrics["decode_tokens"]
        rounds = tokens / (1.0 + a)
        legs.append(
            {
                "session": "gated",
                "prompt": "benchfixture",
                "slot": slot,
                "arm": arm,
                "witness": witness(
                    ROOT / ".mlxfast-private/e158r1" / f"gated-{slot}-{arm}"
                ),
                "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                "mean_proposed_per_round_q": q,
                "mean_accepted_per_round_a": a,
                "rows_per_round": 1.0 + q,
                "round_count_derived": rounds,
                "accepted_draft_total_derived": rounds * a,
                "seconds_per_round_R": metrics["mtp_seconds_per_token"]
                * (1.0 + a),
                "peak_ram_gb": metrics.get("peak_ram_gb"),
                "process_resident_memory_gb": metrics.get(
                    "process_resident_memory_gb"
                ),
                "non_drafting_round_count": metrics.get(
                    "non_drafting_round_count"
                ),
                "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                "decode_token_count": metrics["decode_tokens"],
                "effective_mean_draft_len": metrics["effective_mean_draft_len"],
                "accepted_draft_rate": metrics["accepted_draft_rate"],
                "all_tokens_matched": metrics["all_tokens_matched"],
                "residual_divergence_count": metrics["residual_divergence_count"],
                "head_provenance_sha256": metrics["head_provenance_sha256"],
                "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
                "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
                "cool_gate_passed_real_gate": meta.get(
                    "cool_gate_passed_real_gate"
                ),
                "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
                "official_or_ranked_score": meta.get("official_or_ranked_score"),
                "island_arm_requested": meta.get("island_arm_requested"),
                "base_sha": meta.get("base_sha"),
                "worker_sha256": meta.get("worker_sha256"),
                "cli_sha256": meta.get("cli_sha256"),
                "host": meta.get("host"),
                "chip": meta.get("chip"),
                "started": meta.get("started"),
                "finished": meta.get("finished"),
            }
        )
    return legs


def abba(legs: list[dict], field: str) -> dict | None:
    by_arm: dict[str, list[float]] = {"all": [], "none": []}
    for leg in legs:
        value = leg.get(field)
        if value is None:
            return None
        by_arm[leg["arm"]].append(float(value))
    if len(by_arm["all"]) != 2 or len(by_arm["none"]) != 2:
        return None
    mean_all = statistics.fmean(by_arm["all"])
    mean_none = statistics.fmean(by_arm["none"])
    delta = mean_none - mean_all
    return {
        "all": by_arm["all"],
        "none": by_arm["none"],
        "mean_all": mean_all,
        "mean_none": mean_none,
        "delta_none_minus_all": delta,
        "delta_pct_of_all": 100.0 * delta / mean_all if mean_all else None,
        "within_arm_spread_all": abs(by_arm["all"][0] - by_arm["all"][1]),
        "within_arm_spread_none": abs(by_arm["none"][0] - by_arm["none"][1]),
    }


def published_gain_split(cell: dict) -> dict | None:
    """Split the published gain into its cost factor and its acceptance factor.

    Advisor F6:

        published gain = (R_old / R_new) * (1 + a_new) / (1 + a_old) - 1

    with `old` = arm `all` and `new` = arm `none`. Because `1 + a = N / rounds`,
    the acceptance factor equals `rounds_old / rounds_new`, so the split is
    reported in both forms. The `rounds` form uses two integers the driver
    reports directly and needs no derived rate.

    The product is algebraically the plain mtp ratio. The split does not change
    the number; it says WHICH factor moved, which is the whole point. A cost win
    and an acceptance win have different transfer risk to a hidden prompt.
    """
    r = cell.get("seconds_per_round_R")
    a = cell.get("mean_accepted_per_round_a")
    rounds = cell.get("round_count")
    if not r or not a:
        return None
    cost_factor = r["mean_all"] / r["mean_none"]
    accept_factor = (1.0 + a["mean_none"]) / (1.0 + a["mean_all"])
    out = {
        "rounds_identity_holds": cell.get("rounds_identity_holds"),
        "R_pct_change": 100.0 * (r["mean_none"] / r["mean_all"] - 1.0),
        "a_pct_change": 100.0 * (a["mean_none"] / a["mean_all"] - 1.0)
        if a["mean_all"]
        else None,
        "published_pct_from_cost": 100.0 * (cost_factor - 1.0),
        "published_pct_from_acceptance": 100.0 * (accept_factor - 1.0),
        "published_pct_total": 100.0 * (cost_factor * accept_factor - 1.0),
    }
    if rounds and rounds["mean_none"]:
        out["accept_factor_from_rounds"] = (
            rounds["mean_all"] / rounds["mean_none"]
        )
        out["published_pct_per_extra_accepted_token"] = (
            100.0 / rounds["mean_all"]
        )
    return out


def round_cost_fit(points: list[tuple[float, float]]) -> dict | None:
    """Least squares `R = s + h * rows`, and `rho = 8h/s`.

    `points` are `(rows_per_round, seconds_per_round)`. Advisor F7 asks for
    `rho` "if separable". Separability needs the offered depth to move, so this
    is only meaningful on the depth sweep, where the prompt, head, golden and
    host are held fixed and only rows per round change. A fit across different
    prompts is confounded: there `q` moves because the schedule reacted to
    different text.
    """
    if len(points) < 3:
        return None
    n = len(points)
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if not sxx:
        return None
    h = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    s = mean_y - h * mean_x
    resid = [y - (s + h * x) for x, y in points]
    sst = sum((y - mean_y) ** 2 for _, y in points)
    return {
        "n_points": n,
        "fixed_seconds_per_round_s": s,
        "marginal_seconds_per_row_h": h,
        "rho_8h_over_s": 8.0 * h / s if s else None,
        "r_squared": 1.0 - sum(v * v for v in resid) / sst if sst else None,
        "max_abs_residual": max(abs(v) for v in resid),
        "rows_range": [min(x for x, _ in points), max(x for x, _ in points)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="research/e158-artifacts/r1-abba.json"
    )
    args = parser.parse_args()

    legs = perprompt_legs() + canary_legs() + sweep_legs() + gated_legs()
    if not legs:
        print("e158_r1_harvest: no legs found", file=sys.stderr)
        return 1

    summary: dict = {
        "experiment": "e158-r1-precision-island-price",
        "harness": "local",
        "noise_floor_pct_absolute_mtp": NOISE_FLOOR_PCT,
        "weighted_pairs": WEIGHTED_PAIRS,
        "abba_order": ARMS,
        "legs": legs,
        "per_prompt": {},
        "gated": {},
    }

    for prompt in PROMPTS:
        group = [
            leg
            for leg in legs
            if leg["session"] == "perprompt" and leg["prompt"] == prompt
        ]
        if len(group) != 4:
            continue
        summary["per_prompt"][prompt] = {
            "mtp_seconds_per_token": abba(group, "mtp_seconds_per_token"),
            "seconds_per_round_R": abba(group, "seconds_per_round_R"),
            "mean_accepted_per_round_a": abba(group, "mean_accepted_per_round_a"),
            "mean_proposed_per_round_q": abba(group, "mean_proposed_per_round_q"),
            "alpha_accept_fraction": abba(group, "alpha_accept_fraction"),
            "rounds_identity_holds": all(
                leg["rounds_identity_holds"] for leg in group
            ),
            "non_drafting_round_count": sorted(
                {leg["non_drafting_round_count"] for leg in group}
            ),
            "accepted_draft_total": abba(group, "accepted_draft_total"),
            "round_count": abba(group, "round_count"),
            "rejected_draft_total": abba(group, "rejected_draft_total"),
            "drafting_bit_identical_across_arms": len(
                {
                    (
                        leg["accepted_draft_total"],
                        leg["rejected_draft_total"],
                        leg["round_count"],
                    )
                    for leg in group
                }
            )
            == 1,
            "all_tokens_matched": all(leg["all_tokens_matched"] for leg in group),
            "witnesses_ok": all(
                leg["witness"] is not None
                and leg["witness"].startswith(
                    f"qwen-mtp-island-arm: {leg['arm']} "
                )
                for leg in group
            ),
        }
        summary["per_prompt"][prompt]["published_gain_split"] = (
            published_gain_split(summary["per_prompt"][prompt])
        )

    sweep = [leg for leg in legs if leg["session"] == "depthsweep"]
    if sweep:
        summary["depth_sweep"] = {
            "prompt": SWEEP_PROMPT,
            "role": "separates the fixed per-round cost s from the per-row cost "
            "h by moving only the offered depth inside one prompt",
            "arms": {},
        }
        for arm in ("all", "none"):
            arm_legs = [leg for leg in sweep if leg["arm"] == arm]
            if not arm_legs:
                continue
            summary["depth_sweep"]["arms"][arm] = {
                "fit": round_cost_fit(
                    [
                        (leg["rows_per_round"], leg["seconds_per_round_R"])
                        for leg in arm_legs
                    ]
                ),
                "points": [
                    {
                        "offered_depth": leg["offered_depth"],
                        "q": leg["mean_proposed_per_round_q"],
                        "a": leg["mean_accepted_per_round_a"],
                        "alpha": leg["alpha_accept_fraction"],
                        "rows_per_round": leg["rows_per_round"],
                        "round_count": leg["round_count"],
                        "accepted_draft_total": leg["accepted_draft_total"],
                        "non_drafting_round_count": leg[
                            "non_drafting_round_count"
                        ],
                        "seconds_per_round_R": leg["seconds_per_round_R"],
                        "mtp_seconds_per_token": leg["mtp_seconds_per_token"],
                        "all_tokens_matched": leg["all_tokens_matched"],
                        "witness": leg["witness"],
                    }
                    for leg in arm_legs
                ],
            }
        fits = {
            arm: cell["fit"]
            for arm, cell in summary["depth_sweep"]["arms"].items()
            if cell.get("fit")
        }
        if "all" in fits and "none" in fits:
            # B1 removes a read that happens once per PROPOSAL SLOT, so it must
            # land in h. A change that lands in s instead would falsify the
            # stated mechanism even if the end-to-end number were unchanged.
            summary["depth_sweep"]["mechanism_check"] = {
                "h_pct_change_none_vs_all": 100.0
                * (
                    fits["none"]["marginal_seconds_per_row_h"]
                    / fits["all"]["marginal_seconds_per_row_h"]
                    - 1.0
                ),
                "s_pct_change_none_vs_all": 100.0
                * (
                    fits["none"]["fixed_seconds_per_round_s"]
                    / fits["all"]["fixed_seconds_per_round_s"]
                    - 1.0
                ),
                "predicts_saving_in_h_not_s": True,
            }

    # The cost side transfers to an unseen prompt; the acceptance side is a
    # numerics perturbation that helped on two local prompts and hurt on the
    # canary, so it is reported separately and never folded into the headline.
    clean = [
        prompt
        for prompt, cell in summary["per_prompt"].items()
        if cell.get("drafting_bit_identical_across_arms")
    ]
    summary["cost_only_estimate"] = {
        "clean_cells": clean,
        "note": "cells where both arms produced identical accepted, rejected "
        "and round counts, so the arm contrast is pure round cost",
        "published_pct": {
            prompt: summary["per_prompt"][prompt]["published_gain_split"][
                "published_pct_from_cost"
            ]
            for prompt in clean
        },
    }

    summary["median_weighted_candidate_leg"] = {}
    for name, weights in WEIGHTED_PAIRS.items():
        weighted = 0.0
        complete = True
        for prompt, weight in weights.items():
            cell = (
                summary["per_prompt"].get(prompt, {}).get("mtp_seconds_per_token")
            )
            if not cell:
                complete = False
                continue
            weighted += weight * cell["delta_pct_of_all"]
        if not complete:
            continue
        gain = -weighted / 100.0
        summary["median_weighted_candidate_leg"][name] = {
            "weights": weights,
            "delta_pct_of_all": weighted,
            "gain_fraction_g": gain,
            "published_pct_rule176": 100.0 * gain / (1.0 - gain)
            if gain < 1.0
            else None,
            "beats_noise_floor": abs(weighted) > NOISE_FLOOR_PCT,
        }

    canary = {
        leg["arm"]: leg for leg in legs if leg["session"] == "canary"
    }
    if canary:
        summary["canary"] = {
            "prompt": CANARY_PROMPT,
            "role": "head-quality detector only; median weight 0.0033 and needs "
            "+181 % before the published median moves, so it is never a target",
            "arms": {
                arm: {
                    "effective_mean_draft_len": leg["effective_mean_draft_len"],
                    "non_drafting_round_count": leg["non_drafting_round_count"],
                    "round_count": leg["round_count"],
                    "accepted_draft_rate": leg["accepted_draft_rate"],
                    "mtp_seconds_per_token": leg["mtp_seconds_per_token"],
                    "all_tokens_matched": leg["all_tokens_matched"],
                    "witness": leg["witness"],
                }
                for arm, leg in canary.items()
            },
        }
        if "all" in canary and "none" in canary:
            base_edl = canary["all"]["effective_mean_draft_len"]
            summary["canary"]["edl_pct_change_none_vs_all"] = (
                100.0
                * (canary["none"]["effective_mean_draft_len"] / base_edl - 1.0)
                if base_edl
                else None
            )
            summary["canary"]["non_drafting_delta_none_minus_all"] = (
                canary["none"]["non_drafting_round_count"]
                - canary["all"]["non_drafting_round_count"]
            )

    gated = [leg for leg in legs if leg["session"] == "gated"]
    if len(gated) == 4:
        summary["gated"] = {
            "mtp_seconds_per_token": abba(gated, "mtp_seconds_per_token"),
            "serial_seconds_per_token": abba(gated, "serial_seconds_per_token"),
            "mtp_decode_speedup": abba(gated, "mtp_decode_speedup"),
            "effective_mean_draft_len": abba(gated, "effective_mean_draft_len"),
            "accepted_draft_rate": abba(gated, "accepted_draft_rate"),
            "all_tokens_matched": all(leg["all_tokens_matched"] for leg in gated),
            "gate_qualified_for_timing": all(
                leg["gate_qualified_for_timing"] == "true" for leg in gated
            ),
            "witnesses_ok": all(
                leg["witness"] is not None
                and leg["witness"].startswith(
                    f"qwen-mtp-island-arm: {leg['arm']} "
                )
                for leg in gated
            ),
        }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            key: summary[key]
            for key in (
                "per_prompt",
                "canary",
                "depth_sweep",
                "gated",
                "cost_only_estimate",
                "median_weighted_candidate_leg",
            )
            if key in summary
        },
        indent=2,
        sort_keys=True,
    ))
    print(f"e158_r1_harvest: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
