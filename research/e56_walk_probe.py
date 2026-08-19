#!/usr/bin/env python3
"""Read both depth rules directly instead of describing them.

The first E56 session collapsed the sched arm onto verify width 4 in 98.5% of
rounds. This probe showed why: the old price charged a per-OPERATION QMV ratio
inside a whole-round walk, which made the 4 -> 5 and 8 -> 9 steps unreachable at
EVERY acceptance rate, so that arm was an unconditional width cap and not a
walk. The probe now rebuilds the repaired round-level table from the live
constants and reports, for each rule, the payable steps and the stop depth as a
function of the acceptance probability a round sees.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

SOURCE = SESSION.read_text(encoding="utf-8")


def swift_scalar(name: str) -> float:
    match = re.search(rf"let {name} = ([0-9.]+)", SOURCE)
    if not match:
        raise SystemExit(f"e56_walk_probe: {name} not found")
    return float(match.group(1))


HEAD_STEP_COST_RATIO = swift_scalar("headStepCostRatio")


def inputs_per_group() -> dict[int, int]:
    """Read the shipped table out of the schedule, not out of this file."""
    match = re.search(
        r"verifyInputsPerGroup: \[Int: Int\] = \[(.*?)\]", SOURCE, re.S)
    if not match:
        raise SystemExit("e56_walk_probe: verifyInputsPerGroup not found")
    return {int(k): int(v) for k, v in re.findall(r"(\d+)\s*:\s*(\d+)",
                                                  match.group(1))}


def measured_round_seconds() -> list[float]:
    match = re.search(r"measuredRoundSeconds = \[(.*?)\]", SOURCE, re.S)
    if not match:
        raise SystemExit("e56_walk_probe: measuredRoundSeconds not found")
    return [float(value) for value in re.findall(r"[0-9.]+", match.group(1))]


def stream_cost_ratio() -> float:
    """Reproduce `verifyStreamCostRatio` from the measured round curve."""
    seconds = measured_round_seconds()
    marginal = [b - a for a, b in zip(seconds, seconds[1:])]
    return marginal[1] / ((marginal[0] + marginal[2]) / 2.0)


def streams(width: int, ipg: dict[int, int]) -> int:
    group = ipg.get(width)
    return 1 if group is None else -(-width // group)


def cost_table(ipg: dict[int, int]) -> tuple[list[float], list[float]]:
    ratio = stream_cost_ratio()
    crosses = [streams(d + 2, ipg) > streams(d + 1, ipg) for d in range(8)]
    boundaries = sum(crosses)
    rows = len(crosses)
    within = rows * HEAD_STEP_COST_RATIO / (
        rows - boundaries + boundaries * ratio)
    marginal = [within * ratio if cross else within for cross in crosses]
    cumulative = [1.0]
    for step in marginal:
        cumulative.append(cumulative[-1] + step)
    return marginal, cumulative


def walk(p: float, marginal: list[float], cumulative: list[float],
         scalar: bool, cap: int = 8) -> int:
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= p
        if scalar:
            threshold = (HEAD_STEP_COST_RATIO * (1.0 + expected)
                         / (1.0 + depth * HEAD_STEP_COST_RATIO))
        else:
            threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
        if reach <= threshold:
            break
        expected += reach
        depth += 1
    return depth


def main() -> None:
    ipg = inputs_per_group()
    marginal, cumulative = cost_table(ipg)

    print(f"inputsPerGroup read from source: {ipg}")
    print(f"measuredRoundSeconds read from source: {measured_round_seconds()}")
    print(f"verifyStreamCostRatio {stream_cost_ratio():.6f}"
          f"  headStepCostRatio {HEAD_STEP_COST_RATIO}\n")
    print("depth  width step  crosses a stream boundary  marginal  cumulative")
    for depth in range(8):
        crosses = streams(depth + 2, ipg) > streams(depth + 1, ipg)
        print(f"  {depth}      {depth + 1}->{depth + 2}       "
              f"{str(crosses):<25} {marginal[depth]:.6f}  "
              f"{cumulative[depth]:.6f}")

    print("\nPayability: a step is CLOSED at every acceptance rate when")
    print("marginal[d] * (d + 1) / cumulative[d] >= 1, because the walk needs")
    print("reach > that and reach <= 1 always.")
    for depth in range(8):
        required = marginal[depth] * (depth + 1) / cumulative[depth]
        state = "CLOSED" if required >= 1.0 else "open"
        print(f"  d={depth} (width {depth + 1}->{depth + 2})"
              f"  required reach floor {required:.4f}  {state}")

    print("\nStop depth as a function of a constant acceptance probability.")
    print("Verify width = drafts + 1; cap 8 (streak-qualified).")
    print("    p     base drafts  base W   sched drafts  sched W")
    for p in (0.995, 0.99, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
        base = walk(p, marginal, cumulative, scalar=True)
        sched = walk(p, marginal, cumulative, scalar=False)
        print(f"  {p:.3f}      {base}          {base + 1}"
              f"           {sched}            {sched + 1}")

    print("\nThreshold trajectory at p=0.99, the measured accept rate.")
    reach, expected = 1.0, 0.0
    for depth in range(8):
        reach *= 0.99
        sched_threshold = (marginal[depth] * (1.0 + expected)
                           / cumulative[depth])
        base_threshold = (HEAD_STEP_COST_RATIO * (1.0 + expected)
                          / (1.0 + depth * HEAD_STEP_COST_RATIO))
        print(f"  d={depth} reach={reach:.4f}"
              f"  sched_threshold={sched_threshold:.4f}"
              f" {'BREAK' if reach <= sched_threshold else 'extend'}"
              f"   base_threshold={base_threshold:.4f}"
              f" {'BREAK' if reach <= base_threshold else 'extend'}")
        if reach <= sched_threshold:
            break
        expected += reach

    print("\nSmallest p at which each rule still crosses the 4->5 boundary")
    print("(i.e. still reaches 4 drafts / verify width 5):")
    for scalar in (True, False):
        crossing = None
        for step in range(10000, 0, -1):
            p = step / 10000
            if walk(p, marginal, cumulative, scalar=scalar) >= 4:
                crossing = p
            else:
                break
        label = "base (scalar h)" if scalar else "sched (stream-aware)"
        print(f"  {label:<22} {crossing if crossing else 'never'}")


if __name__ == "__main__":
    main()
