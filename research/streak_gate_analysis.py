#!/usr/bin/env python3
"""Closed-form reachability of the segmented deep-round regime.

Zero-GPU analysis of the gate the assignment asks about. It models the exact
shipped logic in `Qwen36MTPBlockSession.swift`:

  * `costModelDepth` (~L596-614) -- greedy marginal-depth rule under a width cap.
  * width cap (~L578-584): `fullAcceptStreak >= segmentedStreakGate ? 8 : 4`.
  * streak update (~L1057-1058): `acceptedCount == drafts.count ? +1 : 0`,
    reached ONLY by drafting rounds -- the `draftCount == 0` path returns at
    ~L773, so a non-drafting round leaves the streak frozen.

The question it settles: on a prompt whose per-draft acceptance is `p`, is the
deep regime reachable at all, and does it pay when reached?
"""

from __future__ import annotations

import argparse
import json
import random

H = 0.20  # headStepCostRatio (owned by qwen-edward; read, never retuned)
SHALLOW_CAP = 4  # sdpaWidthWallDepthCap
DEEP_CAP = 8  # segmentedVerifyDepthCap


def chosen_depth(p: float, cap: int, h: float = H) -> int:
    """Exact port of `costModelDepth` for a uniform per-position EMA of `p`.

    The depth-0 logit-margin clamp is omitted: it only ever lowers p, and no
    steady-state statement should depend on one round's margin.
    """
    reach = 1.0
    expected = 0.0
    depth = 0
    while depth < cap:
        reach *= p
        threshold = h * (1.0 + expected) / (1.0 + depth * h)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def expected_accepted(p: float, d: int) -> float:
    return sum(p**i for i in range(1, d + 1))


def tokens_per_cost(p: float, d: int, h: float = H) -> float:
    """Round throughput in tokens per target-forward-equivalent.

    Cost `1 + d*h` is the same currency the shipped cost model's threshold uses.
    """
    return (1.0 + expected_accepted(p, d)) / (1.0 + d * h)


def rounds_to_streak(q: float, k: int) -> float:
    """Expected Bernoulli(q) trials to first observe k consecutive successes."""
    if q <= 0.0:
        return float("inf")
    if q >= 1.0:
        return float(k)
    return (1.0 - q**k) / (q**k * (1.0 - q))


def deep_run_length(q_deep: float) -> float:
    """Consecutive deep rounds once the gate opens (it stays open while full)."""
    if q_deep >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - q_deep)


def advisor_grid(
    acceptances: list[float] | None = None,
    depths: list[int] | None = None,
    streaks: tuple[int, ...] = (1, 2, 3),
) -> list[dict]:
    """Per-round full-accept probability and expected rounds to a given streak.

    A round only increments `fullAcceptStreak` when EVERY draft is accepted, so
    the streak is a Bernoulli process with success probability `p ** depth`.
    """
    acceptances = acceptances or [1.00, 0.90, 0.80, 0.70, 0.60]
    depths = depths or [2, 4, 6, 8]
    rows = []
    for p in acceptances:
        for d in depths:
            q = p**d
            rows.append(
                {
                    "p": p,
                    "depth": d,
                    "full_accept_prob": q,
                    **{
                        f"rounds_to_streak_{k}": rounds_to_streak(q, k)
                        for k in streaks
                    },
                }
            )
    return rows


def analyse(p: float, gate: int) -> dict:
    d_shallow = chosen_depth(p, SHALLOW_CAP)
    d_deep = chosen_depth(p, DEEP_CAP)
    cap_binds = d_deep > d_shallow

    q_open = p**d_shallow if d_shallow > 0 else 0.0
    q_deep = p**d_deep if d_deep > 0 else 0.0

    to_open = rounds_to_streak(q_open, gate)
    run = deep_run_length(q_deep) if cap_binds else 0.0
    duty = run / (run + to_open) if cap_binds and to_open != float("inf") else 0.0

    return {
        "p": p,
        "gate": gate,
        "chosen_depth_cap4": d_shallow,
        "chosen_depth_cap8": d_deep,
        "cap8_binds": cap_binds,
        "q_full_accept_shallow": q_open,
        "q_full_accept_deep": q_deep,
        "expected_rounds_to_open": to_open,
        "expected_consecutive_deep_rounds": run,
        "deep_duty_cycle": duty,
        "tokens_per_cost_shallow": tokens_per_cost(p, d_shallow),
        "tokens_per_cost_deep": tokens_per_cost(p, d_deep),
        "deep_pays": tokens_per_cost(p, d_deep) > tokens_per_cost(p, d_shallow),
    }


def implied_p(target_speedup: float, h: float = H) -> dict:
    """Smallest uniform p whose best achievable depth reaches `target_speedup`.

    Speedup vs serial is tokens-per-cost, so this inverts the same model.
    """
    best = None
    p = 0.50
    while p <= 0.9995:
        d = chosen_depth(p, DEEP_CAP)
        s = tokens_per_cost(p, d, h)
        if s >= target_speedup:
            best = {"p": round(p, 4), "depth": d, "modelled_speedup": s}
            break
        p += 0.0005
    return best or {"p": None, "note": "unreachable under this cost model"}


ALPHA = 0.15  # acceptEMAAlpha (read-only for this assignment)
OPTIMISM_TARGET = 0.95  # capped optimism transfer on a fully-accepted round
MAX_DEPTH = 8  # Qwen36MTPLimits.maxDepth
PRIOR = [0.85 * 0.98**i for i in range(MAX_DEPTH)]


def depth_from_ema(ema: list, cap: int, h: float = H) -> int:
    """`costModelDepth` driven by a full per-position EMA vector."""
    reach = 1.0
    expected = 0.0
    depth = 0
    while depth < cap:
        reach *= ema[depth]
        threshold = h * (1.0 + expected) / (1.0 + depth * h)
        if not reach > threshold:
            break
        expected += reach
        depth += 1
    return depth


def simulate_trajectory(
    p: float,
    gate: int,
    tokens: int,
    seed: int = 0,
    deep_cap: int = DEEP_CAP,
    shallow_cap: int = SHALLOW_CAP,
    ema_gate: float = None,
    ema_gate_index: int = 4,
) -> dict:
    """Replay the shipped EMA/streak/width-cap loop token-for-token.

    `p` is the true per-position acceptance. p == 1.0 replays the local
    copy fixture deterministically; p < 1.0 draws leading-run acceptances.
    `ema_gate` adds the candidate disjunct `positionAcceptEMA[i] >= t`.
    """
    rng = random.Random(seed)
    ema = list(PRIOR)
    streak = 0
    emitted = 0
    rounds = []
    while emitted < tokens:
        streak_in = streak
        deep = streak_in >= gate
        if ema_gate is not None and ema[ema_gate_index] >= ema_gate:
            deep = True
        cap = deep_cap if deep else shallow_cap
        depth = depth_from_ema(ema, cap)
        ema_in = list(ema)
        if depth == 0:
            emitted += 1
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "streak_in": streak_in,
                    "cap": cap,
                    "depth": 0,
                    "accepted": 0,
                    "width": 1,
                    "emitted": emitted,
                    "ema_in": ema_in,
                }
            )
            continue  # non-drafting round leaves `fullAcceptStreak` frozen
        accepted = depth
        for i in range(depth):
            if p < 1.0 and rng.random() >= p:
                accepted = i
                break
        for i in range(accepted):
            ema[i] += ALPHA * (1.0 - ema[i])
        if accepted < depth:
            ema[accepted] += ALPHA * (0.0 - ema[accepted])
        elif depth < MAX_DEPTH:
            ema[depth] += ALPHA * (OPTIMISM_TARGET - ema[depth])
        streak = streak + 1 if accepted == depth else 0
        emitted += accepted + 1
        rounds.append(
            {
                "round": len(rounds) + 1,
                "streak_in": streak_in,
                "cap": cap,
                "depth": depth,
                "accepted": accepted,
                "width": accepted + 1,
                "emitted": emitted,
                "ema_in": ema_in,
            }
        )
    widths = {}
    depths = {}
    for r in rounds:
        widths[r["width"]] = widths.get(r["width"], 0) + 1
        depths[r["depth"]] = depths.get(r["depth"], 0) + 1
    first_deep = next((r for r in rounds if r["depth"] == MAX_DEPTH), None)
    return {
        "p": p,
        "gate": gate,
        "tokens": tokens,
        "rounds": len(rounds),
        "depth_histogram": dict(sorted(depths.items())),
        "width_histogram": dict(sorted(widths.items())),
        "mean_depth": sum(r["depth"] for r in rounds) / max(len(rounds), 1),
        "tokens_per_round": rounds[-1]["emitted"] / len(rounds) if rounds else 0.0,
        "first_depth8_round": first_deep["round"] if first_deep else None,
        "first_depth8_token": first_deep["emitted"] if first_deep else None,
        "width9_rounds": widths.get(MAX_DEPTH + 1, 0),
        "trajectory": rounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--acceptance",
        type=float,
        nargs="*",
        default=[1.00, 0.95, 0.90, 0.80, 0.70, 0.60],
    )
    ap.add_argument("--gates", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--depth-grid", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--promoted-score", type=float, default=2.9042110287045)
    ap.add_argument("--out")
    args = ap.parse_args()

    print("## Table 1 - cost-model chosen depth (uniform EMA = p, h = %.2f)" % H)
    print()
    print("| p | depth @cap4 | depth @cap8 | does cap 8 bind? |")
    print("|---:|---:|---:|:--|")
    for p in args.acceptance:
        d4 = chosen_depth(p, SHALLOW_CAP)
        d8 = chosen_depth(p, DEEP_CAP)
        print(
            "| %.2f | %d | %d | %s |"
            % (p, d4, d8, "**yes**" if d8 > d4 else "no - cap never reached")
        )

    print()
    print("## Table 2 - advisor grid: full-accept probability and streak cost")
    print()
    print("| p | realized depth d | q = p^d | E[rounds to streak 1] "
          "| E[rounds to streak 2] | E[rounds to streak 3] |")
    print("|---:|---:|---:|---:|---:|---:|")
    grid = advisor_grid(args.acceptance, args.depth_grid)
    for row in grid:
        print(
            "| %.2f | %d | %.4f | %.1f | %.1f | %.1f |"
            % (
                row["p"],
                row["depth"],
                row["full_accept_prob"],
                row["rounds_to_streak_1"],
                row["rounds_to_streak_2"],
                row["rounds_to_streak_3"],
            )
        )

    print()
    print("## Table 3 - two-regime duty cycle (gate opens at cap4 depth, "
          "deep rounds run at cap8 depth)")
    print()
    print("| p | gate | d@cap4 | d@cap8 | E[rounds to open] "
          "| E[consecutive deep] | deep duty cycle | deep pays? |")
    print("|---:|---:|---:|---:|---:|---:|---:|:--|")
    records = []
    for p in args.acceptance:
        for gate in args.gates:
            r = analyse(p, gate)
            records.append(r)
            print(
                "| %.2f | %d | %d | %d | %s | %s | %s | %s |"
                % (
                    p,
                    gate,
                    r["chosen_depth_cap4"],
                    r["chosen_depth_cap8"],
                    "n/a" if not r["cap8_binds"] else "%.1f" % r["expected_rounds_to_open"],
                    "n/a" if not r["cap8_binds"] else "%.2f" % r["expected_consecutive_deep_rounds"],
                    "n/a" if not r["cap8_binds"] else "%.1f%%" % (100 * r["deep_duty_cycle"]),
                    "n/a" if not r["cap8_binds"] else ("yes" if r["deep_pays"] else "**no**"),
                )
            )

    print()
    print("## Table 4 - round throughput by depth (tokens per target-equivalent)")
    print()
    header = "| p | " + " | ".join("d=%d" % d for d in range(0, 9)) + " | argmax |"
    print(header)
    print("|---:|" + "---:|" * 9 + "---:|")
    for p in args.acceptance:
        vals = [tokens_per_cost(p, d) for d in range(0, 9)]
        best = max(range(9), key=lambda d: vals[d])
        print(
            "| %.2f | " % p
            + " | ".join("%.3f" % v for v in vals)
            + " | **d=%d** |" % best
        )

    print()
    print("## Implied per-draft acceptance of the promoted frontier")
    print()
    print(
        "Promoted official score %.4f -> %s"
        % (args.promoted_score, json.dumps(implied_p(args.promoted_score)))
    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "h": H,
                    "shallow_cap": SHALLOW_CAP,
                    "deep_cap": DEEP_CAP,
                    "advisor_grid": grid,
                    "records": records,
                    "implied_p_promoted": implied_p(args.promoted_score),
                },
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
