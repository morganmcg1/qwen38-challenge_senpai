#!/usr/bin/env python3
"""E130-S part 1: is the ranked runner state the same thing as harness defect 19?

Harness defect 19 (ledger 263.x, "Harness defect 19 observed live") is EXTERNAL
INTERRUPTION OF A WHOLE TIMED REGION on a local machine: `fa.qkv` at M=3 read
1254.7 us forward against 300.0 us reverse in FOUR OF EIGHT blocks. It is a
per-region, random, strictly-inflating contaminant.

FINDING 150 / E130 rung 8 measured a different-looking object: a two-level
offset of 930.9 us per drafting round that is shared by every receipt in a
cluster, across six disjoint solver accounts.

The advisor asked me to test defect 19 against that model using recorded
per-round times. My rung-2 artifacts hold kernel microbenchmarks only, with no
per-round decode series, so that route is blocked. This script uses a stronger
route that needs no new data: the ranked receipts already carry EIGHT
INDEPENDENT TIMED PROMPTS per submission, which is exactly the axis on which a
per-region interruption and a per-run state disagree.

    defect 19    the contaminant hits individual timed regions at random, so
                 the eight per-prompt offsets inside one receipt DISAGREE, and
                 it hits whatever region is timed, so the SERIAL leg is hit too
    run state    the offset is a property of the run, so the eight per-prompt
                 offsets inside one receipt AGREE, and only the candidate leg
                 that owns the state moves

Three discriminators, all computable from the board:

    1. intraclass correlation of the per-prompt offset across receipts
    2. within-receipt scatter against the between-cluster step
    3. the serial leg as a co-timed control on the same host

harness = ranked.

Usage:
    python3 research/e130_defect19.py \
        --board /tmp/yukon-board/full.json \
        --ref cf79f7df \
        --out research/e130-artifacts/rung8b-defect19.json
"""

from __future__ import annotations

import argparse
import json
import math
import random

from e130_state_model import ORDER, TOKENS, load_board, prompts, structure

# The six receipts FINDING 150 / rung 8 classified, in the two clusters it
# found. Cluster 0 sits at the reference level, cluster 1 is one step above it.
CLUSTER0 = ["3b376ba2", "c63eaa21", "48423d09"]
CLUSTER1 = ["1986338b", "d3c491b5", "2d02ef0b"]

# rung 8, research/e130-artifacts/rung8-state-model.json
STEP_US_PER_DRAFTING_ROUND = 930.9

# ledger 263.x, the live defect-19 observation
DEFECT19_BLOCKS_HIT = 4
DEFECT19_BLOCKS_TOTAL = 8
DEFECT19_INFLATED_US = 1254.7
DEFECT19_CLEAN_US = 300.0


def offsets(board: dict, member: str, ref: str, leg: str) -> list[dict]:
    """Per-prompt offset of `member` against `ref`, in us per drafting round.

    This whole schedule family declines to draft on `plutarch`, so that prompt
    has zero drafting rounds and no defined ratio. It is kept as a separate
    zero-dose control instead of being dropped.
    """
    pm, pr = prompts(board[member]), prompts(board[ref])
    key = "mtp_seconds_per_token_mean" if leg == "candidate" else "serial_seconds_per_token_mean"
    rows = []
    for name in ORDER:
        em, er = pm[name], pr[name]
        st = structure(em)
        delta_us = (em[key] - er[key]) * TOKENS * 1e6
        dr = st["drafting_rounds"]
        rows.append(
            {
                "prompt": name,
                "drafting_rounds": dr,
                "delta_us": delta_us,
                "delta_us_per_drafting_round": (delta_us / dr) if dr > 0 else None,
            }
        )
    return rows


def icc1(groups: list[list[float]]) -> dict:
    """One-way random-effects ICC over equal-sized groups.

    ICC near 1 means the offset belongs to the receipt. ICC near 0 means it
    belongs to the individual timed prompt, which is what a per-region
    interruption produces.
    """
    r = len(groups)
    k = len(groups[0])
    flat = [v for g in groups for v in g]
    grand = sum(flat) / len(flat)
    means = [sum(g) / k for g in groups]
    ss_between = k * sum((m - grand) ** 2 for m in means)
    ss_within = sum((v - m) ** 2 for g, m in zip(groups, means) for v in g)
    ms_between = ss_between / (r - 1)
    ms_within = ss_within / (r * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return {
        "icc": (ms_between - ms_within) / denom if denom else float("nan"),
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_ratio": ms_between / ms_within if ms_within else float("inf"),
        "sd_between_receipt_means": math.sqrt(ss_between / (k * (r - 1))),
        "sd_within_receipt": math.sqrt(ms_within),
        "df_between": r - 1,
        "df_within": r * (k - 1),
    }


def two_way(groups: list[list[float]]) -> dict:
    """Receipt x prompt decomposition of the per-drafting-round offset.

    A pure per-drafting-round constant predicts no prompt main effect. Any
    reproducible prompt structure is a second term the law does not yet carry.
    """
    r = len(groups)
    k = len(groups[0])
    flat = [v for g in groups for v in g]
    grand = sum(flat) / len(flat)
    rec_means = [sum(g) / k for g in groups]
    pr_means = [sum(g[j] for g in groups) / r for j in range(k)]
    ss_receipt = k * sum((m - grand) ** 2 for m in rec_means)
    ss_prompt = r * sum((m - grand) ** 2 for m in pr_means)
    ss_total = sum((v - grand) ** 2 for v in flat)
    ss_resid = ss_total - ss_receipt - ss_prompt
    df_resid = (r - 1) * (k - 1)
    ms_prompt = ss_prompt / (k - 1)
    ms_resid = ss_resid / df_resid
    return {
        "grand_mean_us": grand,
        "prompt_means_us": pr_means,
        "ss_receipt": ss_receipt,
        "ss_prompt": ss_prompt,
        "ss_residual": ss_resid,
        "prompt_share_of_total": ss_prompt / ss_total if ss_total else float("nan"),
        "f_prompt": ms_prompt / ms_resid if ms_resid else float("inf"),
        "df_prompt": k - 1,
        "df_residual": df_resid,
        "residual_sd_us": math.sqrt(ms_resid),
    }


def skewness(xs: list[float]) -> float:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    if var <= 0:
        return float("nan")
    return (sum((x - m) ** 3 for x in xs) / n) / var**1.5


def defect19_null(groups: list[list[float]], trials: int, seed: int) -> dict:
    """Simulate defect 19 and ask how often it reproduces the observed ICC.

    The null keeps every receipt's measured mean offset. It only changes HOW
    that mean is produced: instead of every prompt carrying it, a Bernoulli
    subset of prompts carries an inflated value and the rest carry nothing,
    at the hit rate the ledger observed live (four blocks in eight).
    """
    rng = random.Random(seed)
    k = len(groups[0])
    means = [sum(g) / k for g in groups]
    q = DEFECT19_BLOCKS_HIT / DEFECT19_BLOCKS_TOTAL
    observed = icc1(groups)["icc"]
    iccs = []
    for _ in range(trials):
        sim = []
        for m in means:
            hits = [1 if rng.random() < q else 0 for _ in range(k)]
            nhit = sum(hits)
            # preserve the receipt mean exactly; an all-miss draw carries none
            amp = (m * k / nhit) if nhit else 0.0
            sim.append([amp if h else 0.0 for h in hits])
        iccs.append(icc1(sim)["icc"])
    iccs.sort()
    ge = sum(1 for v in iccs if v >= observed)
    return {
        "trials": trials,
        "seed": seed,
        "hit_rate": q,
        "observed_icc": observed,
        "null_icc_median": iccs[len(iccs) // 2],
        "null_icc_p95": iccs[int(0.95 * len(iccs))],
        "null_icc_max": iccs[-1],
        "p_value_icc_ge_observed": ge / trials,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="/tmp/yukon-board/full.json")
    ap.add_argument("--ref", default="cf79f7df")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=130)
    ap.add_argument("--out", default="research/e130-artifacts/rung8b-defect19.json")
    args = ap.parse_args()

    board = load_board(args.board)
    members = CLUSTER0 + CLUSTER1
    missing = [m for m in members + [args.ref] if m not in board]
    if missing:
        raise SystemExit(f"board is missing per-prompt rows for {missing}")

    result = {
        "harness": "ranked",
        "question": (
            "Is the FINDING 150 runner state the same phenomenon as harness "
            "defect 19, external interruption of a timed region?"
        ),
        "reference": args.ref,
        "cluster0": CLUSTER0,
        "cluster1": CLUSTER1,
        "step_us_per_drafting_round": STEP_US_PER_DRAFTING_ROUND,
        "defect19_live_observation": {
            "source": "campaign-ledger.md, Harness defect 19 observed live",
            "tensor": "fa.qkv at M=3",
            "blocks_hit": DEFECT19_BLOCKS_HIT,
            "blocks_total": DEFECT19_BLOCKS_TOTAL,
            "inflated_us": DEFECT19_INFLATED_US,
            "clean_us": DEFECT19_CLEAN_US,
            "inflation_ratio": DEFECT19_INFLATED_US / DEFECT19_CLEAN_US,
        },
        "legs": {},
    }

    for leg in ("candidate", "serial"):
        per_receipt = {}
        groups = []
        zero_dose = {}
        for m in members:
            rows = offsets(board, m, args.ref, leg)
            zero_dose[m] = [
                {"prompt": r["prompt"], "delta_us": r["delta_us"]}
                for r in rows
                if r["drafting_rounds"] == 0
            ]
            vals = [
                r["delta_us_per_drafting_round"]
                for r in rows
                if r["delta_us_per_drafting_round"] is not None
            ]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            resid = [v - mean for v in vals]
            per_receipt[m] = {
                "cluster": 0 if m in CLUSTER0 else 1,
                "prompts": rows,
                "mean_us_per_drafting_round": mean,
                "sd_us_per_drafting_round": sd,
                "sd_over_step": sd / STEP_US_PER_DRAFTING_ROUND,
                "residual_skewness": skewness(resid),
                "residual_max_over_sd": max(resid) / sd if sd else float("nan"),
            }
            groups.append(vals)

        stat = icc1(groups)
        pooled_within = stat["sd_within_receipt"]
        leg_out = {
            "per_receipt": per_receipt,
            "anova": stat,
            "pooled_within_receipt_sd": pooled_within,
            "step_over_pooled_within_sd": STEP_US_PER_DRAFTING_ROUND / pooled_within
            if pooled_within
            else float("inf"),
            "cluster_means": {
                "cluster0": sum(
                    per_receipt[m]["mean_us_per_drafting_round"] for m in CLUSTER0
                )
                / len(CLUSTER0),
                "cluster1": sum(
                    per_receipt[m]["mean_us_per_drafting_round"] for m in CLUSTER1
                )
                / len(CLUSTER1),
            },
            "zero_drafting_control": {
                "per_receipt": zero_dose,
                "cluster0_mean_delta_us": sum(
                    e["delta_us"] for m in CLUSTER0 for e in zero_dose[m]
                )
                / sum(len(zero_dose[m]) for m in CLUSTER0),
                "cluster1_mean_delta_us": sum(
                    e["delta_us"] for m in CLUSTER1 for e in zero_dose[m]
                )
                / sum(len(zero_dose[m]) for m in CLUSTER1),
            },
        }
        if leg == "candidate":
            leg_out["defect19_null"] = defect19_null(groups, args.trials, args.seed)
            leg_out["cluster1_two_way"] = two_way(groups[len(CLUSTER0) :])
        result["legs"][leg] = leg_out

    cand = result["legs"]["candidate"]
    ser = result["legs"]["serial"]
    cand_step = cand["cluster_means"]["cluster1"] - cand["cluster_means"]["cluster0"]
    ser_step = ser["cluster_means"]["cluster1"] - ser["cluster_means"]["cluster0"]

    # Every prompt except plutarch has non_drafting_round_count 0, so rounds and
    # drafting rounds are the same number there. plutarch is the ONLY prompt that
    # separates a per-round law from a per-drafting-round law, and it carries 449
    # rounds with zero of them drafting.
    zc = cand["zero_drafting_control"]
    plut_rounds = TOKENS / (
        prompts(board[args.ref])["plutarch"]["effective_mean_draft_len"] + 1.0
    )
    plut_obs = zc["cluster1_mean_delta_us"] - zc["cluster0_mean_delta_us"]
    plut_pred_per_round = plut_rounds * cand_step
    plut_scatter = max(abs(e["delta_us"]) for m in members for e in zc["per_receipt"][m])

    result["verdict"] = {
        "candidate_icc": cand["anova"]["icc"],
        "candidate_step_us_per_drafting_round": cand_step,
        "serial_step_us_per_drafting_round": ser_step,
        "serial_step_over_candidate_step": ser_step / cand_step if cand_step else float("nan"),
        "p_value_defect19_null": cand["defect19_null"]["p_value_icc_ge_observed"],
        "per_round_rejection": {
            "note": (
                "plutarch is the only prompt with drafting rounds != rounds, so it "
                "alone separates the per-round law from the per-drafting-round law"
            ),
            "plutarch_rounds": plut_rounds,
            "observed_step_us": plut_obs,
            "predicted_step_us_if_per_round": plut_pred_per_round,
            "observed_over_predicted": plut_obs / plut_pred_per_round
            if plut_pred_per_round
            else float("nan"),
            "largest_single_plutarch_delta_us": plut_scatter,
        },
    }

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)

    print(f"reference {args.ref}   step {STEP_US_PER_DRAFTING_ROUND} us/drafting round")
    for leg in ("candidate", "serial"):
        lg = result["legs"][leg]
        print(f"\n--- {leg} leg")
        print(
            f"  ICC {lg['anova']['icc']:+.4f}   F {lg['anova']['f_ratio']:.1f}"
            f"   within-receipt sd {lg['pooled_within_receipt_sd']:.1f} us"
        )
        print(
            f"  cluster means {lg['cluster_means']['cluster0']:+8.1f}"
            f" / {lg['cluster_means']['cluster1']:+8.1f} us"
        )
        for m in CLUSTER0 + CLUSTER1:
            e = lg["per_receipt"][m]
            print(
                f"    {m}  c{e['cluster']}  mean {e['mean_us_per_drafting_round']:+8.1f}"
                f"  sd {e['sd_us_per_drafting_round']:7.1f}"
                f"  sd/step {e['sd_over_step']:.3f}"
                f"  skew {e['residual_skewness']:+.2f}"
            )
    d = cand["defect19_null"]
    print(
        f"\ndefect-19 null: median ICC {d['null_icc_median']:+.4f}"
        f"   p95 {d['null_icc_p95']:+.4f}   max {d['null_icc_max']:+.4f}"
        f"   p(ICC >= observed) {d['p_value_icc_ge_observed']:.5f}"
    )
    tw = cand["cluster1_two_way"]
    print(
        f"\ncluster-1 receipt x prompt: prompt effect F {tw['f_prompt']:.2f}"
        f" on {tw['df_prompt']},{tw['df_residual']} df"
        f"   prompt share of SS {100 * tw['prompt_share_of_total']:.1f} %"
        f"   residual sd {tw['residual_sd_us']:.0f} us"
    )
    pr = result["verdict"]["per_round_rejection"]
    print(
        f"\nper-round law on plutarch ({pr['plutarch_rounds']:.0f} rounds, 0 drafting):"
        f" observed {pr['observed_step_us']:+.0f} us"
        f" against predicted {pr['predicted_step_us_if_per_round']:+.0f} us"
        f"   ratio {pr['observed_over_predicted']:+.4f}"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
