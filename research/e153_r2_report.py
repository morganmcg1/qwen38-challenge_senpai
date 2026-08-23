#!/usr/bin/env python3
"""Price the E153 R2 split -> merged wide-decode SDPA contrast.

ESTIMATOR. Within one prompt and one replicate the leg order is A B B A, so
both arms sit at mean position 2.5 and monotone thermal drift cancels to first
order. The reported contrast is the within-block one,

    pct = 100 * (mean(B legs) / mean(A legs) - 1)

with A = split (the shipped two `scaledDotProductAttention` calls) and
B = merged (the custom kernel). Negative means the merged arm is faster.

WHY THIS CONTRAST IS UNUSUALLY CLEAN. The merged kernel was measured BITWISE
identical to the split at all 24 exactness cells, so the two arms must decode
the same tokens, make the same accept/reject decisions, and therefore produce
the same `round_count` and `effective_mean_draft_len`. This script asserts that
identity per block and voids any block that breaks it. Consequence: unlike
E149 Arm A, the Rule 115 schedule term is exactly 1 by construction and the
whole measured difference is per-round kernel cost.

FRAMES (Rule 144). Every number names its frame.

    decode  parent_measured_seconds_per_token. The mechanism lives here.
    total   (seed_prefill_seconds + decode_seconds) / decode_token_count.
            The ranked leg times seed processing and decoding in one window.
            Full attention also runs during seed prefill, but at qL > 9 the
            merged kernel is not reached, so the seed is arm-invariant and the
            total frame is the conservative one.
    harness local on every line. Both legs run one binary and the arm is a
            run-time selector confined to the candidate MTP leg.

RULE 134. 524.5 us/round per 1 % of published median, total-leg frame, as
corrected by thorfinn's E135 work. The carried 515.2 is 1.8 % low. Both the
corrected conversion and the legacy one are reported so the ledger can be
reconciled, but the corrected constant is the headline.

RANKED TRANSFER. R2 is width-gated at qL >= 6, so a local per-round saving
must be re-priced by the ranked probability that a round is eligible, not
carried across as a percentage. That is `RANKED_P_M_GE_6`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Rule 116: the published median rides on these two prompts. benchfixture
# carries no ranked weight and is reported as a witness only.
MEDPAIR = {"beagle_a": 0.478, "essays_montaigne": 0.522}
# Rule 134, corrected on 0cf1637e, total-leg frame.
US_PER_ROUND_PER_PCT = 524.5
US_PER_ROUND_PER_PCT_LEGACY = 515.2
# E149 C1: ranked P(M >= 6), the fraction of ranked rounds the guard admits.
RANKED_P_M_GE_6 = 0.5861
# E149 C1 headline prediction for this mechanism, ranked frame.
C1_PREDICTED_US_PER_ROUND = 154.4987
C1_PREDICTED_SE = 14.1513
# Gated per-leg standard deviation, measured on this host.
GATED_LEG_SD_PCT = 0.052
DECODE_TOKENS = 512
FULL_ATTENTION_LAYERS = 16


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
        if meta.get("e153_leg_role") != "timed":
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
                "arm": meta["e153_arm_requested"],
                "arm_witnessed": meta.get("e153_arm_witnessed", "none"),
                "replicate": int(meta["e153_replicate"]),
                "position": int(meta["e153_position"]),
                "gate_status": meta["e153_cool_gate_status"],
                "gate_qualified": meta["e153_gate_qualified_for_timing"],
                "witness": meta["e153_witness"],
                "entry_c": float(meta.get("gpu_temp_entry_c") or "nan"),
                "exit_c": float(meta.get("gpu_temp_exit_c") or "nan"),
                "worker_sha256": meta.get("worker_sha256", ""),
                "matched": meta.get("all_tokens_matched") == "true",
                "divergences": int(meta.get("residual_divergence_count", 0)),
                "rounds": rounds,
                "edl_text": meta.get("effective_mean_draft_len", ""),
                "edl": report["effective_mean_draft_len"],
                "accept_rate": report["accepted_draft_rate"],
                "decode_spt": report["parent_measured_seconds_per_token"],
                "total_spt": (seed_s + decode_s) / tokens,
                "decode_seconds": decode_s,
                "seed_prefill_seconds": seed_s,
                "decode_round_us": 1e6 * decode_s / rounds,
                "total_round_us": 1e6 * (seed_s + decode_s) / rounds,
                "width_hist": {
                    str(w + 1): lengths.count(w) for w in sorted(set(lengths))
                },
            }
        )
    return legs


def contrast(a_legs: list[dict], b_legs: list[dict], key: str) -> float:
    a = statistics.fmean(leg[key] for leg in a_legs)
    b = statistics.fmean(leg[key] for leg in b_legs)
    return 100.0 * (b / a - 1.0)


def delta_us(a_legs: list[dict], b_legs: list[dict], key: str) -> float:
    a = statistics.fmean(leg[key] for leg in a_legs)
    b = statistics.fmean(leg[key] for leg in b_legs)
    return b - a


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - 2.0 * sem, mean + 2.0 * sem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="s1")
    parser.add_argument("--runs", default=".mlxfast-private/e128/runs-e153r2")
    parser.add_argument("--out", default="research/e153-r2-abba.json")
    args = parser.parse_args()

    legs = collect(Path(args.runs))
    if not legs:
        raise SystemExit(f"e153_r2_report: no timed legs under {args.runs}")

    groups: dict[tuple[str, int], list[dict]] = {}
    for leg in legs:
        groups.setdefault((leg["prompt"], leg["replicate"]), []).append(leg)

    per_block = []
    schedule_violations = []
    for (prompt, rep), group in sorted(groups.items()):
        a_legs = [leg for leg in group if leg["arm"] == "split"]
        b_legs = [leg for leg in group if leg["arm"] == "merged"]
        if len(a_legs) != 2 or len(b_legs) != 2:
            continue
        # Bitwise identity forces one schedule across both arms.
        rounds_seen = {leg["rounds"] for leg in group}
        edl_seen = {leg["edl_text"] for leg in group}
        hist_seen = {json.dumps(leg["width_hist"], sort_keys=True) for leg in group}
        schedule_identical = (
            len(rounds_seen) == 1 and len(edl_seen) == 1 and len(hist_seen) == 1
        )
        if not schedule_identical:
            schedule_violations.append(
                {
                    "prompt": prompt,
                    "replicate": rep,
                    "round_counts": sorted(rounds_seen),
                    "edl": sorted(edl_seen),
                    "width_hists": sorted(hist_seen),
                }
            )
        per_block.append(
            {
                "prompt": prompt,
                "replicate": rep,
                "schedule_identical_across_arms": schedule_identical,
                "decode_pct": contrast(a_legs, b_legs, "decode_spt"),
                "total_pct": contrast(a_legs, b_legs, "total_spt"),
                "decode_us_per_round": delta_us(a_legs, b_legs, "decode_round_us"),
                "total_us_per_round": delta_us(a_legs, b_legs, "total_round_us"),
                "rounds": sorted(rounds_seen),
                "edl": sorted(edl_seen),
                "accept_rate_delta_pp": 100.0
                * (
                    statistics.fmean(leg["accept_rate"] for leg in b_legs)
                    - statistics.fmean(leg["accept_rate"] for leg in a_legs)
                ),
                "split_decode_round_us": statistics.fmean(
                    leg["decode_round_us"] for leg in a_legs
                ),
                "merged_decode_round_us": statistics.fmean(
                    leg["decode_round_us"] for leg in b_legs
                ),
                "width_hist": a_legs[0]["width_hist"],
                "arms_witnessed": sorted({leg["arm_witnessed"] for leg in group}),
                "gate_states": sorted({leg["gate_status"] for leg in group}),
                "witness_states": sorted({leg["witness"] for leg in group}),
            }
        )

    if not per_block:
        raise SystemExit("e153_r2_report: no complete A B B A block")

    # Rule 116 medpair weighting over the two ranked-like prompts.
    def medpair(key: str) -> float:
        total_w = 0.0
        acc = 0.0
        for block in per_block:
            w = MEDPAIR.get(block["prompt"], 0.0)
            if w:
                acc += w * block[key]
                total_w += w
        return acc / total_w if total_w else float("nan")

    medpair_decode_pct = medpair("decode_pct")
    medpair_total_pct = medpair("total_pct")
    medpair_decode_us = medpair("decode_us_per_round")
    medpair_total_us = medpair("total_us_per_round")

    block_decode = [b["decode_pct"] for b in per_block]
    mean_decode, lo_decode, hi_decode = ci95(block_decode)

    # Ranked transfer. The local session already runs the ranked-like medpair
    # prompts, whose realised P(M >= 6) is close to the ranked value, so the
    # local per-round saving is reported both raw and re-priced.
    local_us = abs(medpair_total_us)
    ranked_us_same_eligibility = local_us
    eligible_round_us = (
        local_us / RANKED_P_M_GE_6 if RANKED_P_M_GE_6 else float("nan")
    )

    result = {
        "probe": "e153_r2_merged_sdpa_abba",
        "harness": "local",
        "label": args.label,
        "n_timed_legs": len(legs),
        "n_blocks": len(per_block),
        "sign_convention": "negative pct = merged arm faster than split arm",
        "arm_a": "split (MLX_E153_MERGED_SDPA=0)",
        "arm_b": "merged (compiled default)",
        "all_tokens_matched": all(leg["matched"] for leg in legs),
        "e153_merged_sdpa_divergences": sum(leg["divergences"] for leg in legs),
        "witness_states": sorted({leg["witness"] for leg in legs}),
        "arms_witnessed": sorted({leg["arm_witnessed"] for leg in legs}),
        "gate_states": sorted({leg["gate_status"] for leg in legs}),
        "all_gate_qualified": all(
            leg["gate_qualified"] == "true" for leg in legs
        ),
        "worker_sha256": sorted({leg["worker_sha256"] for leg in legs}),
        "entry_c_min": min(leg["entry_c"] for leg in legs),
        "entry_c_max": max(leg["entry_c"] for leg in legs),
        "entry_c_spread": max(leg["entry_c"] for leg in legs)
        - min(leg["entry_c"] for leg in legs),
        "schedule_identical_all_blocks": not schedule_violations,
        "schedule_violations": schedule_violations,
        "per_block": per_block,
        "e153_merged_sdpa_local_pct": medpair_total_pct,
        "e153_merged_sdpa_local_pct_decode_frame": medpair_decode_pct,
        "e153_merged_sdpa_round_cost_us": medpair_decode_us,
        "e153_merged_sdpa_round_cost_us_total_frame": medpair_total_us,
        "unweighted_block_decode_pct": mean_decode,
        "unweighted_block_decode_pct_ci95": [lo_decode, hi_decode],
        "gated_leg_sd_pct": GATED_LEG_SD_PCT,
        "sigma_vs_gated_floor_total": medpair_total_pct
        / (GATED_LEG_SD_PCT * math.sqrt(0.478**2 + 0.522**2)),
        "rule134_us_per_round_per_pct": US_PER_ROUND_PER_PCT,
        "rule134_face_value_pct": local_us / US_PER_ROUND_PER_PCT,
        "rule134_face_value_pct_legacy_515_2": local_us
        / US_PER_ROUND_PER_PCT_LEGACY,
        "ranked_p_m_ge_6": RANKED_P_M_GE_6,
        "implied_eligible_round_saving_us": eligible_round_us,
        "c1_predicted_us_per_round": C1_PREDICTED_US_PER_ROUND,
        "c1_predicted_se": C1_PREDICTED_SE,
        "c1_prediction_ratio": (
            ranked_us_same_eligibility / C1_PREDICTED_US_PER_ROUND
            if C1_PREDICTED_US_PER_ROUND
            else float("nan")
        ),
        "c1_prediction_z": (
            (ranked_us_same_eligibility - C1_PREDICTED_US_PER_ROUND)
            / C1_PREDICTED_SE
            if C1_PREDICTED_SE
            else float("nan")
        ),
        "full_attention_layers": FULL_ATTENTION_LAYERS,
    }

    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
