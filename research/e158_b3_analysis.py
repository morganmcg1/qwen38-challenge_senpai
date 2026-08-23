#!/usr/bin/env python3
"""E158 R1 B3 -- gated ABBA analysis for the precision-island arms.

Reads the four gated legs harvested into research/e158-artifacts/r1-abba.json
and answers the advisor's B3 asks:

  * `e158_r1_local_pct` on ABSOLUTE candidate MTP seconds per token, with a
    within-session sigma, reported against the 0.039 % historical floor.
  * the exact F6 field set (q, a, R, round_count, accepted_draft_total) and a
    ledger-closure audit.
  * the F7 accuracy/cost split. The two-column form uses MEASURED R only, so it
    is exact and has zero residual. The three-column form needs a per-prompt
    (s, h) fit, which B1 only ran on beagle_a, so it is reported as a labelled
    extrapolation and not as a result.
  * F8 resident-memory high-water mark from the external sampler.

The serial leg never reads the proposal head, so it is a positive control: the
island selector must not move it. That also makes the local ratio admissible
here under program.md, because the causal path is confined to the candidate
MTP leg.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "research/e158-artifacts"
DECODE_TOKENS = 512
NOISE_FLOOR_PCT = 0.039

# B1 depth-sweep fit, beagle_a only. Used ONLY for the labelled extrapolation.
BEAGLE_FIT = {
    "all": {"s": 0.04358370, "h": 0.01745739},
    "none": {"s": 0.04348383, "h": 0.01734174},
}
# Dead draft-head buffers found by source inspection this round.
DEAD_DRAFT_HEAD_BYTES = 157_337_600


def ledger(leg: dict) -> dict:
    """Exact round ledger. edl and the accept rate are exact rationals here,
    so rounds, proposed, accepted and rejected all come out integral."""
    q = leg["effective_mean_draft_len"]
    rate = leg["accepted_draft_rate"]
    a = q * rate
    rounds = DECODE_TOKENS / (1.0 + a)
    proposed = q * rounds
    accepted = a * rounds
    return {
        "mean_proposed_per_round_q": q,
        "accepted_draft_rate": rate,
        "mean_accepted_per_round_a": a,
        "round_count": rounds,
        "round_count_is_integral": abs(rounds - round(rounds)) < 1e-6,
        "draft_proposed_total": proposed,
        "accepted_draft_total": accepted,
        "rejected_draft_total": proposed - accepted,
        "seconds_per_round_R": leg["mtp_seconds_per_token"] * (1.0 + a),
        # program.md ledger identity: every decoded token is either a round's
        # primary token or an accepted draft.
        "ledger_residual": rounds + accepted - DECODE_TOKENS,
    }


def arm_stats(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "values": values}


def welch(a: dict, b: dict) -> dict:
    """b - a, pooled over two equal-size arms."""
    pooled_sd = math.sqrt((a["sd"] ** 2 + b["sd"] ** 2) / 2.0)
    se = pooled_sd * math.sqrt(1.0 / a["n"] + 1.0 / b["n"])
    diff = b["mean"] - a["mean"]
    return {
        "delta": diff,
        "delta_pct_of_baseline": 100.0 * diff / a["mean"],
        "pooled_sd": pooled_sd,
        "standard_error": se,
        "standard_error_pct_of_baseline": 100.0 * se / a["mean"],
        "sigma_multiple": abs(diff) / se if se else None,
        "degrees_of_freedom": a["n"] + b["n"] - 2,
    }


def main() -> int:
    harvest = json.loads((ART / "r1-abba.json").read_text())
    gated = [leg for leg in harvest["legs"] if leg["session"] == "gated"]
    if len(gated) != 4:
        print(f"e158_b3: expected 4 gated legs, found {len(gated)}")
        return 1

    per_leg = []
    for leg in sorted(gated, key=lambda x: x["slot"]):
        row = {
            "slot": leg["slot"],
            "arm": leg["arm"],
            "witness": leg["witness"],
            "head_provenance_sha256": leg["head_provenance_sha256"],
            "worker_sha256": leg["worker_sha256"],
            "cli_sha256": leg["cli_sha256"],
            "base_sha": leg["base_sha"],
            "gate_qualified_for_timing": leg["gate_qualified_for_timing"],
            "cool_gate_passed_real_gate": leg["cool_gate_passed_real_gate"],
            "gpu_temp_entry_c": float(leg["gpu_temp_entry_c"]),
            "gpu_temp_exit_c": float(leg["gpu_temp_exit_c"]),
            "mtp_seconds_per_token": leg["mtp_seconds_per_token"],
            "serial_seconds_per_token": leg["serial_seconds_per_token"],
            "mtp_decode_speedup": leg["mtp_decode_speedup"],
            "all_tokens_matched": leg["all_tokens_matched"],
            "residual_divergence_count": leg["residual_divergence_count"],
            "worker_peak_rss_gb": leg["worker_peak_rss_gb"],
            "worker_peak_rss_samples": leg["worker_peak_rss_samples"],
            # Only the per-round trace carries this, and the --local-iterate
            # path writes score.json without it.
            "non_drafting_round_count": leg["non_drafting_round_count"],
        }
        row.update(ledger(leg))
        per_leg.append(row)

    def by_arm(field: str) -> dict[str, dict]:
        return {
            arm: arm_stats([r[field] for r in per_leg if r["arm"] == arm])
            for arm in ("all", "none")
        }

    mtp = by_arm("mtp_seconds_per_token")
    serial = by_arm("serial_seconds_per_token")
    ratio = by_arm("mtp_decode_speedup")
    rss = by_arm("worker_peak_rss_gb")
    R = by_arm("seconds_per_round_R")
    acc = by_arm("mean_accepted_per_round_a")
    q = by_arm("mean_proposed_per_round_q")

    primary = welch(mtp["all"], mtp["none"])
    primary["vs_noise_floor_pct"] = NOISE_FLOOR_PCT
    primary["effect_over_floor"] = (
        abs(primary["delta_pct_of_baseline"]) / NOISE_FLOOR_PCT
    )
    primary["within_session_sigma_exceeds_floor"] = (
        primary["standard_error_pct_of_baseline"] > NOISE_FLOOR_PCT
    )

    # Exact two-column split on measured quantities.
    d_ln_R = math.log(R["none"]["mean"] / R["all"]["mean"])
    d_ln_rows = math.log(
        (1.0 + acc["none"]["mean"]) / (1.0 + acc["all"]["mean"])
    )
    d_ln_t = math.log(mtp["none"]["mean"] / mtp["all"]["mean"])
    split_exact = {
        "cost_per_round_pct": 100.0 * d_ln_R,
        "accepted_per_round_pct": -100.0 * d_ln_rows,
        "model_total_pct": 100.0 * (d_ln_R - d_ln_rows),
        "measured_total_pct": 100.0 * d_ln_t,
        "residual_pct": 100.0 * (d_ln_t - (d_ln_R - d_ln_rows)),
        "basis": "measured R and measured a only; identity t = R / (1 + a)",
    }

    # Labelled extrapolation: attribute part of the cost column to the extra
    # rows offered, using the beagle_a marginal row cost h. benchfixture has a
    # different seed length, so its (s, h) is NOT this fit.
    rows_all = 1.0 + q["all"]["mean"]
    rows_none = 1.0 + q["none"]["mean"]
    row_cost_delta = BEAGLE_FIT["none"]["h"] * (rows_none - rows_all)
    mech_delta = (
        BEAGLE_FIT["none"]["s"] - BEAGLE_FIT["all"]["s"]
    ) + (BEAGLE_FIT["none"]["h"] - BEAGLE_FIT["all"]["h"]) * rows_all
    modelled_R_all = (
        BEAGLE_FIT["all"]["s"] + BEAGLE_FIT["all"]["h"] * rows_all
    )
    split_extrapolated = {
        "status": "EXTRAPOLATION, not a result",
        "reason": (
            "beagle_a (s, h) applied to benchfixture. Seed length differs, so "
            "the KV cache and therefore both s and h differ. The modelled R "
            "misses measured R by "
            f"{100.0 * (modelled_R_all / R['all']['mean'] - 1.0):.2f} %."
        ),
        "schedule_extra_rows_pct": 100.0 * row_cost_delta / R["all"]["mean"],
        "mechanism_at_fixed_rows_pct": 100.0 * mech_delta / modelled_R_all,
        "modelled_seconds_per_round_all": modelled_R_all,
        "measured_seconds_per_round_all": R["all"]["mean"],
    }

    out = {
        "experiment": "e158-r1-b3-gated-abba",
        "harness": "local",
        "prompt": "benchfixture",
        "decode_tokens": DECODE_TOKENS,
        "arms_in_order": [r["arm"] for r in per_leg],
        "counterbalanced": [r["arm"] for r in per_leg] == [
            "all", "none", "none", "all"
        ],
        "official_or_ranked_score": False,
        "per_leg": per_leg,
        "identity_tuple_uniform": {
            "worker_sha256": sorted({r["worker_sha256"] for r in per_leg}),
            "cli_sha256": sorted({r["cli_sha256"] for r in per_leg}),
            "head_provenance_sha256": sorted(
                {r["head_provenance_sha256"] for r in per_leg}
            ),
            "base_sha": sorted({r["base_sha"] for r in per_leg}),
            "base_sha_note": (
                "slot 1 predates a research-only analysis commit; the "
                "benchmark executes none of those files and the worker, CLI "
                "and metallib digests are identical on all four legs"
            ),
        },
        "correctness": {
            "all_tokens_matched": all(r["all_tokens_matched"] for r in per_leg),
            "residual_divergence_total": sum(
                r["residual_divergence_count"] for r in per_leg
            ),
            "ledger_closes_on_every_leg": all(
                abs(r["ledger_residual"]) < 1e-6 for r in per_leg
            ),
            "round_counts_integral": all(
                r["round_count_is_integral"] for r in per_leg
            ),
        },
        "thermal": {
            "all_gate_qualified": all(
                r["gate_qualified_for_timing"] == "true" for r in per_leg
            ),
            "entry_temp_c": [r["gpu_temp_entry_c"] for r in per_leg],
            "entry_temp_spread_c": max(r["gpu_temp_entry_c"] for r in per_leg)
            - min(r["gpu_temp_entry_c"] for r in per_leg),
        },
        "primary_absolute_mtp_seconds_per_token": {
            "all": mtp["all"],
            "none": mtp["none"],
            **primary,
        },
        "serial_control": {
            "all": serial["all"],
            "none": serial["none"],
            **welch(serial["all"], serial["none"]),
            "expectation": (
                "null; the serial leg decodes at depth 0 and never reads the "
                "proposal head"
            ),
        },
        "local_ratio": {
            "all": ratio["all"],
            "none": ratio["none"],
            **welch(ratio["all"], ratio["none"]),
        },
        "f6_fields_by_arm": {
            "seconds_per_round_R": {k: v["mean"] for k, v in R.items()},
            "mean_accepted_per_round_a": {k: v["mean"] for k, v in acc.items()},
            "mean_proposed_per_round_q": {k: v["mean"] for k, v in q.items()},
            "round_count": {
                arm: [r["round_count"] for r in per_leg if r["arm"] == arm]
                for arm in ("all", "none")
            },
            "accepted_draft_total": {
                arm: [
                    r["accepted_draft_total"] for r in per_leg if r["arm"] == arm
                ]
                for arm in ("all", "none")
            },
            "non_drafting_round_count": (
                "unavailable on the --local-iterate path; score.json carries "
                "no per-round trace"
            ),
        },
        "f7_split_exact_two_column": split_exact,
        "f7_split_three_column_extrapolated": split_extrapolated,
        "f8_resident_memory": {
            "worker_peak_rss_gb_by_arm": {k: v["mean"] for k, v in rss.items()},
            **welch(rss["all"], rss["none"]),
            "high_water_gb": max(r["worker_peak_rss_gb"] for r in per_leg),
            "source": "external-ps-sampler-2s",
            "dead_draft_head_bytes": DEAD_DRAFT_HEAD_BYTES,
            "dead_draft_head_pct_of_high_water": 100.0
            * DEAD_DRAFT_HEAD_BYTES
            / (max(r["worker_peak_rss_gb"] for r in per_leg) * 1024**3),
        },
    }

    # The ranked score is a median over eight hidden prompts, so a mechanism
    # whose sign flips per prompt cannot be priced from one prompt. Pull every
    # matched total already in the harvest.
    cross: dict[str, dict[str, list[float]]] = {}
    for leg in harvest["legs"]:
        if leg["session"] not in ("perprompt", "canary"):
            continue
        cross.setdefault(leg["prompt"], {}).setdefault(leg["arm"], []).append(
            leg["mtp_seconds_per_token"]
        )
    cross_pct = {
        prompt: 100.0
        * (statistics.fmean(v["none"]) / statistics.fmean(v["all"]) - 1.0)
        for prompt, v in sorted(cross.items())
        if v.get("all") and v.get("none")
    }
    cross_pct["benchfixture_gated_512"] = primary["delta_pct_of_baseline"]
    values = sorted(cross_pct.values())
    out["cross_prompt_none_minus_all_pct"] = {
        "by_prompt": cross_pct,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "sign_flips": min(values) < 0.0 < max(values),
        "note": (
            "ungated perprompt/canary totals plus the gated benchfixture "
            "total; negative favours the shipped default arm none"
        ),
    }

    path = ART / "b3-gated.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"arms in order        {out['arms_in_order']} "
          f"counterbalanced={out['counterbalanced']}")
    print(f"gate qualified       {out['thermal']['all_gate_qualified']} "
          f"entry spread {out['thermal']['entry_temp_spread_c']:.2f} C")
    print(f"tokens matched       {out['correctness']['all_tokens_matched']} "
          f"divergences {out['correctness']['residual_divergence_total']}")
    print(f"ledger closes        "
          f"{out['correctness']['ledger_closes_on_every_leg']}")
    print()
    for r in per_leg:
        print(f"  slot {r['slot']} {r['arm']:<5} "
              f"mtp {r['mtp_seconds_per_token']:.9f}  "
              f"R {r['seconds_per_round_R']:.6f}  "
              f"rounds {r['round_count']:.0f}  "
              f"acc {r['accepted_draft_total']:.0f}  "
              f"rej {r['rejected_draft_total']:.0f}  "
              f"rss {r['worker_peak_rss_gb']:.3f} GB")
    print()
    print(f"e158_r1_local_pct    {primary['delta_pct_of_baseline']:+.4f} % "
          f"+/- {primary['standard_error_pct_of_baseline']:.4f} % (1 sigma) "
          f"= {primary['sigma_multiple']:.1f} sigma")
    print(f"  vs {NOISE_FLOOR_PCT} % floor  "
          f"{primary['effect_over_floor']:.1f}x")
    print(f"serial control       "
          f"{out['serial_control']['delta_pct_of_baseline']:+.4f} % "
          f"({out['serial_control']['sigma_multiple']:.1f} sigma)")
    print()
    print("F7 exact two-column split (none - all):")
    print(f"  cost per round      "
          f"{split_exact['cost_per_round_pct']:+.4f} %")
    print(f"  accepted per round  "
          f"{split_exact['accepted_per_round_pct']:+.4f} %")
    print(f"  model total         "
          f"{split_exact['model_total_pct']:+.4f} %")
    print(f"  measured total      "
          f"{split_exact['measured_total_pct']:+.4f} %")
    print(f"  residual            {split_exact['residual_pct']:+.6f} %")
    print()
    print(f"F8 high-water        {out['f8_resident_memory']['high_water_gb']:.3f} GB"
          f"  arm delta "
          f"{out['f8_resident_memory']['delta_pct_of_baseline']:+.4f} %")
    print(f"  dead draft head     {DEAD_DRAFT_HEAD_BYTES:,} B = "
          f"{out['f8_resident_memory']['dead_draft_head_pct_of_high_water']:.2f}"
          f" % of high water")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
