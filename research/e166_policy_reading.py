#!/usr/bin/env python3
"""E166 steps 1 and 2 -- read the shipped depth policy, then price it.

Step 1 writes the closed form of the objective `Qwen36MTPBlockSession`
actually minimises, converts its dimensionless price into both harnesses, and
replays the shipped schedule under a substituted ranked price.

Step 2 converts the E159 pinned-width accepted-draft rates into per-token
accept probabilities and tests whether acceptance falls with depth.

No GPU. Every number here is arithmetic on constants read out of the source or
out of a sealed leg.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

MAX_DEPTH = 8                     # Qwen36MTPLimits.maxDepth
SEGMENTED_VERIFY_DEPTH_CAP = 7    # Qwen36MTPBlockSession:1060
SHIP_H = 0.18                     # headStepCostRatio, :892
EMA_PRIOR = [0.85 * 0.98 ** i for i in range(MAX_DEPTH)]   # :856
EMA_ALPHA = 0.15                  # :858

# harness=ranked, FINDING 352 / receipt 5a9f130a: R = s + h*M, M = 1 + d.
RANKED_S = 16.1585
RANKED_H = 5.3350
# harness=local, E159 destepped affine fit (LAW 354 pass removed at M>=6).
LOCAL_DESTEP_S = 47.756
LOCAL_DESTEP_H = 8.6457
# harness=local, E159 raw measured curve, M = 2..8.
E159_R = {2: 69.403, 3: 71.674, 4: 78.124, 5: 91.501,
          6: 126.952, 7: 138.722, 8: 146.332}
# harness=local, E159 accepted-draft rate at pinned depth d = M - 1.
E159_ACCEPT_RATE = {1: 0.9807, 2: 0.9545, 3: 0.9475, 4: 0.9243,
                    5: 0.8685, 6: 0.8758, 7: 0.8642}


def price_ratio(s: float, h: float) -> float:
    """The dimensionless constant the policy compares against.

    The policy prices a round as `C(d) = V * (1 + ratio*d)` with `V` the cost
    of a round that proposes nothing, so the ratio an affine cost law implies
    is its marginal row over its own `d = 0` round, `h / (s + h)`.
    """
    return h / (s + h)


def choose_depth(p_vector, h_marginal, cap=SEGMENTED_VERIFY_DEPTH_CAP):
    """`costModelDepth` with a uniform price, reproduced exactly (:1091-1111)."""
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= p_vector[depth]
        threshold = h_marginal * (1.0 + expected) / (1.0 + h_marginal * depth)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def clamp_margin(p_vector, margin):
    """The two top-2 confidence clamps at positions 0 and 1 (:1090-1098)."""
    out = list(p_vector)
    if margin is not None and not math.isnan(margin):
        out[0] = min(out[0], 1.0 / (1.0 + math.exp(-margin / 2.0)))
        out[1] = min(out[1], 1.0 / (1.0 + math.exp(-margin / 3.0)))
    return out


def solve_q(target_sum: float, depth: int) -> float:
    """q with sum_{k=1..depth} q^k == target_sum."""
    lo, hi = 1e-6, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = sum(mid ** k for k in range(1, depth + 1))
        if total < target_sum:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def step2():
    rows = []
    for d, rate in sorted(E159_ACCEPT_RATE.items()):
        accepted = rate * d
        q = solve_q(accepted, d)
        rows.append({"pinned_depth": d, "rows_M": d + 1,
                     "accepted_draft_rate": rate,
                     "accepted_count": accepted,
                     "implied_q": q})
    qs = [r["implied_q"] for r in rows]
    mean = sum(qs) / len(qs)
    sd = (sum((q - mean) ** 2 for q in qs) / (len(qs) - 1)) ** 0.5
    deep = [r["implied_q"] for r in rows if r["pinned_depth"] >= 4]
    shallow = [r["implied_q"] for r in rows if r["pinned_depth"] <= 3]
    # chain reach per position, from the increments of accepted count
    chain = []
    prev = 0.0
    for r in rows:
        chain.append(r["accepted_count"] - prev)
        prev = r["accepted_count"]
    return {
        "table": rows,
        "q_mean": mean, "q_sd": sd, "q_spread_pct": 100 * (max(qs) - min(qs)) / mean,
        "q_mean_shallow_d1_3": sum(shallow) / len(shallow),
        "q_mean_deep_d4_7": sum(deep) / len(deep),
        "q_collapses_with_depth": (sum(deep) / len(deep)) < 0.9 * (sum(shallow) / len(shallow)),
        "chain_reach_increments": chain,
    }


def parse_trace(path):
    """Read one round per `sched=` trace line: margin, EMAs, walked depths."""
    rounds = []
    line_re = re.compile(
        r"arm=(?P<arm>\S+) m=(?P<margin>\S+) streak=(?P<streak>\d+) "
        r"cap=(?P<cap>\d+) ema=(?P<ema>[0-9.,]+) sched=(?P<sched>\S*)")
    anchor_re = re.compile(r"mtp-anchor: round=(?P<round>\d+) d=(?P<d>\d+) acc=(?P<acc>\d+)")
    anchors = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            hit = anchor_re.search(line)
            if hit:
                anchors.append((int(hit["round"]), int(hit["d"]), int(hit["acc"])))
            hit = line_re.search(line)
            if not hit:
                continue
            margin = float("nan") if hit["margin"] in ("nan", "-nan") else float(hit["margin"])
            rounds.append({
                "arm": hit["arm"],
                "margin": margin,
                "streak": int(hit["streak"]),
                "cap": int(hit["cap"]),
                "ema": [float(x) for x in hit["ema"].split(",")],
                "walk": hit["sched"],
            })
    return rounds, anchors


def replay(rounds, anchors):
    """Open-loop counterfactual: hold every EMA at its shipped value and swap
    only the price constant. The EMAs are a closed loop in a real run, so this
    is a first-order reading, not a prediction of a live arm."""
    ranked_ratio = price_ratio(RANKED_S, RANKED_H)
    ship, ranked = Counter(), Counter()
    per_round = []
    for entry in rounds:
        p_vector = clamp_margin(entry["ema"], entry["margin"])
        cap = min(entry["cap"], SEGMENTED_VERIFY_DEPTH_CAP)
        d_ship = choose_depth(p_vector, SHIP_H, cap)
        d_ranked = choose_depth(p_vector, ranked_ratio, cap)
        ship[d_ship] += 1
        ranked[d_ranked] += 1
        per_round.append((d_ship, d_ranked))
    realised = Counter(a[1] for a in anchors)
    n = max(len(per_round), 1)
    return {
        "rounds_traced": len(per_round),
        "ranked_price_ratio": ranked_ratio,
        "ship_price_ratio": SHIP_H,
        "hist_ship_replayed": dict(sorted(ship.items())),
        "hist_ranked_replayed": dict(sorted(ranked.items())),
        "hist_realised_from_anchor": dict(sorted(realised.items())),
        "mean_depth_ship_replayed": sum(d for d, _ in per_round) / n,
        "mean_depth_ranked_replayed": sum(d for _, d in per_round) / n,
        "mean_depth_realised": (sum(a[1] for a in anchors) / len(anchors)) if anchors else None,
        "replay_reproduces_shipped_walk": all(
            d == r for (d, _), r in zip(per_round, [a[1] for a in anchors])
        ) if len(anchors) == len(per_round) else None,
    }


# harness=ranked, E166 brief: per-token accept probability solved from the
# exact ranked accept ledger, and the shipped offered depth on the same row.
RANKED_PROMPTS = {
    "beagle": (0.9341, 4.38), "essays": (0.9647, 5.09),
    "republic": (0.9661, 4.99), "medicine": (0.9639, 5.26),
    "botany": (0.9598, 6.15), "travel": (0.6974, 2.65),
    "drama": (0.6030, 2.30), "plutarch": (0.3158, 1.16),
}


def global_min_depth(q, s, h, cap=SEGMENTED_VERIFY_DEPTH_CAP):
    best_d, best_f = 0, (s + h) / 1.0
    accepted = 0.0
    for d in range(1, cap + 1):
        accepted += q ** d
        f = (s + h * (1 + d)) / (1.0 + accepted)
        if f < best_f:
            best_d, best_f = d, f
    return best_d, best_f


def prompt_table():
    """Does the SHIPPED price already choose the depth the ranked model wants?

    If the policy's own estimator saw the prompt's measured flat `q`, the
    shipped greedy rule would choose `d_ship_greedy`. Compare that with the
    ranked global optimum and with the depth the policy actually offers.
    """
    ranked_ratio = price_ratio(RANKED_S, RANKED_H)
    rows = []
    for name, (q, shipped_edl) in RANKED_PROMPTS.items():
        p_vector = [q] * MAX_DEPTH
        d_star, _ = global_min_depth(q, RANKED_S, RANKED_H)
        rows.append({
            "prompt": name, "q": q, "shipped_edl": shipped_edl,
            "d_star_ranked_global": d_star,
            "d_ship_greedy_at_flat_q": choose_depth(p_vector, SHIP_H),
            "d_ranked_greedy_at_flat_q": choose_depth(p_vector, ranked_ratio),
            "accepted_at_cap7": sum(q ** k for k in range(1, 8)),
        })
    return rows


def flat_q_map():
    """Where the two prices disagree, on a flat-acceptance family."""
    ranked_ratio = price_ratio(RANKED_S, RANKED_H)
    out = []
    q = 0.30
    while q <= 0.995 + 1e-9:
        p_vector = [q] * MAX_DEPTH
        out.append({"q": round(q, 3),
                    "d_ship": choose_depth(p_vector, SHIP_H),
                    "d_ranked": choose_depth(p_vector, ranked_ratio)})
        q += 0.05
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", help="phase-trace file from one adaptive leg")
    parser.add_argument("--json", help="write the full result here")
    args = parser.parse_args()

    ranked_ratio = price_ratio(RANKED_S, RANKED_H)
    local_ratio = price_ratio(LOCAL_DESTEP_S, LOCAL_DESTEP_H)
    ranked_round0 = RANKED_S + RANKED_H
    local_round0 = LOCAL_DESTEP_S + LOCAL_DESTEP_H

    result = {
        "experiment": "e166-ranked-depth-optimum",
        "step1_constants": {
            "headStepCostRatio": SHIP_H,
            "depthPriceArm": "ship (uniform)",
            "segmentedVerifyDepthCap": SEGMENTED_VERIFY_DEPTH_CAP,
            "maxDepth": MAX_DEPTH,
            "ema_prior": EMA_PRIOR,
            "ema_alpha": EMA_ALPHA,
        },
        "step1_price_translation": {
            "harness=ranked": {
                "law": "R(M) = 16.1585 + 5.3350*M",
                "round_at_d0_ms": ranked_round0,
                "marginal_row_ms": RANKED_H,
                "implied_ratio": ranked_ratio,
                "shipped_ratio_in_ms": SHIP_H * ranked_round0,
                "shipped_over_truth": SHIP_H / ranked_ratio,
            },
            "harness=local": {
                "law": "R(M) = 47.756 + 8.6457*M  (E159 destepped)",
                "round_at_d0_ms": local_round0,
                "marginal_row_ms": LOCAL_DESTEP_H,
                "implied_ratio": local_ratio,
                "shipped_ratio_in_ms": SHIP_H * local_round0,
                "shipped_over_truth": SHIP_H / local_ratio,
            },
            "harness=local_raw_with_law354_step": {
                "mean_marginal_row_ms": (E159_R[8] - E159_R[2]) / 6.0,
                "note": "raw curve, LAW 354 weight-stream step still inside it",
            },
        },
        "step1_flat_q_map": flat_q_map(),
        "step1_prompt_table": prompt_table(),
        "step2": step2(),
    }

    if args.trace:
        rounds, anchors = parse_trace(args.trace)
        result["step1_replay"] = replay(rounds, anchors)

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
