#!/usr/bin/env python3
"""Price the E149 Arm A shipped -> leaf16 contrast from the gated ABBA session.

ESTIMATOR. Within one prompt and one replicate the order is A B B A, so both
arms sit at mean position 2.5 and monotone drift cancels to first order. The
reported contrast is therefore the within-replicate one,

    pct = 100 * (mean(B legs) / mean(A legs) - 1)

and never a pooled arm mean across replicates.

FRAMES (Rule 144). Every number below names its frame.

    decode  parent_measured_seconds_per_token. The mechanism lives here.
    total   (seed_prefill_seconds + decode_seconds) / decode_token_count.
            The ranked leg times seed processing and decoding in one window,
            and no leaf-width change touches the seed, so the total frame is
            the conservative one.
    harness local on every line. Both legs run the candidate binary and the
            arm is a run-time selector confined to the candidate MTP leg, so
            nothing cancels; but a local percentage is still not a ranked
            percentage, which is why the per-round absolute is reported too.

DECOMPOSITION (Rule 115). In the decode frame seconds per token is
N * round_us / 512 exactly, so

    spt_B / spt_A = (N_B / N_A) * (round_us_B / round_us_A)

The first term is the scheduler's response to the acceptance change, which is
deterministic and transfers to the ranked harness unchanged. The second is the
per-round cost of the mechanism, an absolute microsecond figure that must be
re-priced over the ranked round rather than carried across as a percentage.
Both terms are reported separately, because Arm A is expected to move both:
halving the leaves halves the coarse probe pass (term 2) and changing which
candidates reach the rerank changes acceptance (term 1).

F235 FACE VALUE. The campaign's per-round multiplier is
+1.0000 % of published median per 515.2 us/round saved. This script reports the
face-value conversion of its own measured per-round delta and does not apply a
transfer discount; the discount, if any, belongs in the write-up where it can
be argued.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Rule 116: the published median rides on these two prompts. benchfixture is
# the depth-price witness and carries no ranked weight.
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}
F235_US_PER_ROUND_PER_PCT = 515.2
DECODE_TOKENS = 512

SHIP_BENCH_ROUNDS = 78
SHIP_BENCH_EDL = "6.358974358974359"
PB6_BENCH_EDL = "5.853658536585366"


def read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key] = value
    return meta


def collect(runs: Path) -> list[dict]:
    legs = []
    for meta_path in sorted(runs.glob("*/*/meta.txt")):
        meta = read_meta(meta_path)
        if meta.get("e149_leg_role") != "timed":
            continue
        report_path = meta_path.parent / "report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        decode_s = report["decode_seconds"]
        seed_s = report.get("seed_prefill_seconds", 0.0)
        tokens = report["decode_token_count"]
        rounds = report["round_count"]
        lengths = report.get("effective_draft_lengths") or []
        legs.append(
            {
                "slot": meta_path.parent.parent.name,
                "prompt": meta["prompt_id"],
                "arm": meta["e149_arm_requested"],
                "replicate": int(meta["e149_replicate"]),
                "position": int(meta["e149_position"]),
                "leaf": int(meta["e149_leaf_exported"]),
                "depth_price_arm": meta["e149_depth_price_arm_exported"],
                "gate_status": meta["e149_cool_gate_status"],
                "gate_qualified": meta["e149_gate_qualified_for_timing"],
                "witness": meta["e149_witness"],
                "entry_c": float(meta["gpu_temp_entry_c"] or "nan"),
                "exit_c": float(meta["gpu_temp_exit_c"] or "nan"),
                "worker_sha256": meta["worker_sha256"],
                "matched": meta["all_tokens_matched"] == "true",
                "divergences": int(meta["residual_divergence_count"]),
                "rounds": rounds,
                "edl_text": meta["effective_mean_draft_len"],
                "edl": report["effective_mean_draft_len"],
                "accept_rate": report["accepted_draft_rate"],
                "accepted": report["accepted_draft_total"],
                "rejected": report["rejected_draft_total"],
                "non_drafting_rounds": report.get("non_drafting_round_count"),
                "decode_spt": report["parent_measured_seconds_per_token"],
                "total_spt": (seed_s + decode_s) / tokens,
                "decode_seconds": decode_s,
                "seed_prefill_seconds": seed_s,
                "decode_round_us": 1e6 * decode_s / rounds,
                "total_round_us": 1e6 * (seed_s + decode_s) / rounds,
                # Verify width is the drafted length plus the primary token.
                "width_hist": {
                    str(w + 1): lengths.count(w)
                    for w in sorted(set(lengths))
                },
            }
        )
    return legs


def contrast(a_legs: list[dict], b_legs: list[dict], key: str) -> float:
    a = statistics.fmean(leg[key] for leg in a_legs)
    b = statistics.fmean(leg[key] for leg in b_legs)
    return 100.0 * (b / a - 1.0)


def blocks(legs: list[dict]) -> dict:
    out = {}
    for leg in legs:
        out.setdefault((leg["prompt"], leg["replicate"]), []).append(leg)
    return out


def ci95(values: list[float]) -> tuple[float, float, float]:
    """Mean and a 2-sigma interval on the mean. n is small and honest about it."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - 2.0 * sem, mean + 2.0 * sem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="s1")
    parser.add_argument("--runs", default=".mlxfast-private/e128/runs-e149a")
    parser.add_argument("--out", default="research/e149-armA.json")
    args = parser.parse_args()

    legs = collect(Path(args.runs))
    if not legs:
        raise SystemExit(f"e149_armA_report: no timed legs under {args.runs}")

    per_block = []
    for (prompt, rep), group in sorted(blocks(legs).items()):
        a_legs = [leg for leg in group if leg["arm"] == "shipped"]
        b_legs = [leg for leg in group if leg["arm"] == "leaf16"]
        if len(a_legs) != 2 or len(b_legs) != 2:
            continue
        ship_rounds = {leg["rounds"] for leg in a_legs}
        cand_rounds = {leg["rounds"] for leg in b_legs}
        per_block.append(
            {
                "prompt": prompt,
                "replicate": rep,
                "decode_pct": contrast(a_legs, b_legs, "decode_spt"),
                "total_pct": contrast(a_legs, b_legs, "total_spt"),
                "decode_round_us_shipped": statistics.fmean(
                    leg["decode_round_us"] for leg in a_legs),
                "decode_round_us_leaf16": statistics.fmean(
                    leg["decode_round_us"] for leg in b_legs),
                "total_round_us_shipped": statistics.fmean(
                    leg["total_round_us"] for leg in a_legs),
                "total_round_us_leaf16": statistics.fmean(
                    leg["total_round_us"] for leg in b_legs),
                "round_count_shipped": sorted(ship_rounds),
                "round_count_leaf16": sorted(cand_rounds),
                "round_count_ratio": (
                    statistics.fmean(leg["rounds"] for leg in b_legs)
                    / statistics.fmean(leg["rounds"] for leg in a_legs)),
                "edl_shipped": sorted({leg["edl_text"] for leg in a_legs}),
                "edl_leaf16": sorted({leg["edl_text"] for leg in b_legs}),
                "accept_rate_shipped": statistics.fmean(
                    leg["accept_rate"] for leg in a_legs),
                "accept_rate_leaf16": statistics.fmean(
                    leg["accept_rate"] for leg in b_legs),
                "within_arm_spread_pct_shipped": 100.0 * (
                    max(leg["decode_spt"] for leg in a_legs)
                    / min(leg["decode_spt"] for leg in a_legs) - 1.0),
                "within_arm_spread_pct_leaf16": 100.0 * (
                    max(leg["decode_spt"] for leg in b_legs)
                    / min(leg["decode_spt"] for leg in b_legs) - 1.0),
                "deterministic_shipped": len(ship_rounds) == 1,
                "deterministic_leaf16": len(cand_rounds) == 1,
            }
        )
        block = per_block[-1]
        block["decode_round_us_delta"] = (
            block["decode_round_us_leaf16"] - block["decode_round_us_shipped"])
        block["total_round_us_delta"] = (
            block["total_round_us_leaf16"] - block["total_round_us_shipped"])
        block["accept_rate_delta_pp"] = 100.0 * (
            block["accept_rate_leaf16"] - block["accept_rate_shipped"])
        block["f235_face_pct_of_median"] = (
            -block["decode_round_us_delta"] / F235_US_PER_ROUND_PER_PCT)

    by_prompt = {}
    for block in per_block:
        by_prompt.setdefault(block["prompt"], []).append(block)

    prompt_summary = {}
    for prompt, group in by_prompt.items():
        decode_mean, decode_lo, decode_hi = ci95(
            [b["decode_pct"] for b in group])
        total_mean, total_lo, total_hi = ci95([b["total_pct"] for b in group])
        prompt_summary[prompt] = {
            "n_blocks": len(group),
            "decode_pct": decode_mean,
            "decode_pct_ci95": [decode_lo, decode_hi],
            "total_pct": total_mean,
            "total_pct_ci95": [total_lo, total_hi],
            "decode_round_us_delta": statistics.fmean(
                b["decode_round_us_delta"] for b in group),
            "total_round_us_delta": statistics.fmean(
                b["total_round_us_delta"] for b in group),
            "accept_rate_delta_pp": statistics.fmean(
                b["accept_rate_delta_pp"] for b in group),
            "round_count_ratio": statistics.fmean(
                b["round_count_ratio"] for b in group),
            "f235_face_pct_of_median": statistics.fmean(
                b["f235_face_pct_of_median"] for b in group),
        }

    ranked_prompts = [p for p in MEDPAIR if p in prompt_summary]
    weight_total = sum(MEDPAIR[p] for p in ranked_prompts)
    medpair = {}
    if ranked_prompts:
        for field in ("decode_pct", "total_pct", "decode_round_us_delta",
                      "total_round_us_delta", "accept_rate_delta_pp",
                      "f235_face_pct_of_median"):
            medpair[field] = sum(
                MEDPAIR[p] * prompt_summary[p][field] for p in ranked_prompts
            ) / weight_total

    all_block_decode = [b["decode_pct"] for b in per_block]
    unweighted_mean, unweighted_lo, unweighted_hi = ci95(all_block_decode)

    entry_temps = [leg["entry_c"] for leg in legs
                   if not math.isnan(leg["entry_c"])]
    width_hist = {}
    for leg in legs:
        arm = leg["arm"]
        target = width_hist.setdefault(arm, {})
        for width, count in leg["width_hist"].items():
            target[width] = target.get(width, 0) + count

    summary = {
        "experiment": "e149-arm-a-leaf16-shipped-vocabulary",
        "harness": "local",
        "label": args.label,
        "estimator": "within-prompt within-replicate ABBA contrast, A B B A",
        "gate": "real 40C via benchmark.sh --local-cool-gate-only",
        "depth_price_arm": sorted({leg["depth_price_arm"] for leg in legs}),
        "e149_leaf16_depth_price_arm": "ship",
        "n_timed_legs": len(legs),
        "worker_sha256": sorted({leg["worker_sha256"] for leg in legs}),
        "witness_states": sorted({leg["witness"] for leg in legs}),
        "gate_states": sorted({leg["gate_status"] for leg in legs}),
        "all_gate_qualified": all(
            leg["gate_qualified"] == "true" for leg in legs),
        "e149_leaf16_divergences": sum(leg["divergences"] for leg in legs),
        "all_tokens_matched": all(leg["matched"] for leg in legs),
        "entry_temp_c_min": min(entry_temps) if entry_temps else None,
        "entry_temp_c_max": max(entry_temps) if entry_temps else None,
        "entry_temp_c_spread": (max(entry_temps) - min(entry_temps))
        if entry_temps else None,
        "per_block": per_block,
        "per_prompt": prompt_summary,
        "medpair_weighted": medpair,
        "e149_leaf16_local_pct": medpair.get("total_pct"),
        "e149_leaf16_local_pct_decode_frame": medpair.get("decode_pct"),
        "unweighted_block_mean_decode_pct": unweighted_mean,
        "unweighted_block_mean_decode_pct_ci95": [unweighted_lo, unweighted_hi],
        "e149_leaf16_round_cost_us": medpair.get("decode_round_us_delta"),
        "e149_leaf16_round_cost_us_total_frame": medpair.get(
            "total_round_us_delta"),
        "width_histogram": width_hist,
        "f235_us_per_round_per_pct": F235_US_PER_ROUND_PER_PCT,
    }

    bench = prompt_summary.get("benchfixture")
    bench_legs = [leg for leg in legs
                  if leg["prompt"] == "benchfixture" and leg["arm"] == "shipped"]
    summary["depth_price_witness"] = {
        "prompt": "benchfixture",
        "expected_ship_rounds": SHIP_BENCH_ROUNDS,
        "expected_ship_edl": SHIP_BENCH_EDL,
        "pb6_edl_tripwire": PB6_BENCH_EDL,
        "observed_shipped_rounds": sorted({leg["rounds"] for leg in bench_legs}),
        "observed_shipped_edl": sorted({leg["edl_text"] for leg in bench_legs}),
        "passed": bool(bench_legs)
        and all(leg["rounds"] == SHIP_BENCH_ROUNDS for leg in bench_legs)
        and all(leg["edl_text"] == SHIP_BENCH_EDL for leg in bench_legs),
    }
    if bench is not None:
        summary["benchfixture_witness_prompt_decode_pct"] = bench["decode_pct"]

    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
