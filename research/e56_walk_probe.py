#!/usr/bin/env python3
"""Explain the measured width histogram from the two walks themselves.

The E56 session collapsed the sched arm onto verify width 4 in 98.5% of rounds.
That is a far larger behavioural change than the counterfactual predicted, so
the walk has to be read directly rather than described. This re-implements both
threshold rules from the live constants and reports where each one stops as a
function of the acceptance probability the round sees.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

HEAD_STEP_COST_RATIO = 0.18
REFERENCE_HEAD_STEP_RATIO = 0.039819
STREAM_SURCHARGE_RATIO = 27.532 / 9.624


def inputs_per_group() -> dict[int, int]:
    """Read the shipped table out of the schedule, not out of this file."""
    text = SESSION.read_text(encoding="utf-8")
    match = re.search(
        r"verifyInputsPerGroup: \[Int: Int\] = \[(.*?)\]", text, re.S)
    if not match:
        raise SystemExit("e56_walk_probe: verifyInputsPerGroup not found")
    return {int(k): int(v) for k, v in re.findall(r"(\d+)\s*:\s*(\d+)",
                                                  match.group(1))}


def streams(width: int, ipg: dict[int, int]) -> int:
    group = ipg.get(width)
    return 1 if group is None else -(-width // group)


def cost_table(ipg: dict[int, int]) -> tuple[list[float], list[float]]:
    raw = [1.0 + (STREAM_SURCHARGE_RATIO
                  if streams(d + 2, ipg) > streams(d + 1, ipg) else 0.0)
           for d in range(8)]
    scale = (HEAD_STEP_COST_RATIO - REFERENCE_HEAD_STEP_RATIO) / (
        sum(raw) / len(raw))
    marginal = [REFERENCE_HEAD_STEP_RATIO + scale * value for value in raw]
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
    print(f"stream surcharge ratio {STREAM_SURCHARGE_RATIO:.6f}\n")
    print("depth  width step  crosses a stream boundary  marginal  cumulative")
    for depth in range(8):
        crosses = streams(depth + 2, ipg) > streams(depth + 1, ipg)
        print(f"  {depth}      {depth + 1}->{depth + 2}       "
              f"{str(crosses):<25} {marginal[depth]:.6f}  "
              f"{cumulative[depth]:.6f}")

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
