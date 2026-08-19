#!/usr/bin/env python3
"""Read the depth rules directly instead of describing them.

  python3 research/e56_walk_probe.py            # report every arm
  python3 research/e56_walk_probe.py --check    # guard one patched arm

The first E56 session collapsed the sched arm onto verify width 4 in 98.5% of
rounds. This probe showed why: the old price charged a per-OPERATION QMV ratio
inside a whole-round walk, which made the 4 -> 5 and 8 -> 9 steps unreachable at
EVERY acceptance rate, so that arm was an unconditional width cap and not a
walk. It now rebuilds the round-level table for each E56 arm from the live
constants and reports, per arm, the payable steps and the stop depth as a
function of the acceptance probability a round sees.

`--check` reads the schedule file as it currently stands, which is how
`e56_build_arms.sh` proves that a patched arm still declares its own closures.
"""
from __future__ import annotations

import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SESSION = ROOT / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

# The four arms of the E56 revision. `base` is the shipped scalar rule; the
# others price the crossings they name and nothing else.
ARMS = {
    "base": None,
    "s45": {5},
    "s89": {9},
    "sfull": {5, 9},
}


def read_source() -> str:
    return SESSION.read_text(encoding="utf-8")


def swift_scalar(source: str, name: str) -> float:
    match = re.search(rf"let {name} = ([0-9.]+)", source)
    if not match:
        raise SystemExit(f"e56_walk_probe: {name} not found")
    return float(match.group(1))


def inputs_per_group(source: str) -> dict[int, int]:
    """Read the shipped table out of the schedule, not out of this file."""
    match = re.search(
        r"verifyInputsPerGroup: \[Int: Int\] = \[(.*?)\]", source, re.S)
    if not match:
        raise SystemExit("e56_walk_probe: verifyInputsPerGroup not found")
    return {int(k): int(v) for k, v in re.findall(r"(\d+)\s*:\s*(\d+)",
                                                  match.group(1))}


def priced_boundary_widths(source: str) -> set[int]:
    match = re.search(r"pricedBoundaryWidths: Set<Int> = \[(.*?)\]", source)
    if not match:
        raise SystemExit("e56_walk_probe: pricedBoundaryWidths not found")
    return {int(value) for value in re.findall(r"\d+", match.group(1))}


def declared_closed_steps(source: str) -> list[int]:
    match = re.search(r"declaredClosedDepthSteps: \[Int\] = \[(.*?)\]", source)
    if not match:
        raise SystemExit("e56_walk_probe: declaredClosedDepthSteps not found")
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


def measured_round_seconds(source: str) -> list[float]:
    match = re.search(r"measuredRoundSeconds = \[(.*?)\]", source, re.S)
    if not match:
        raise SystemExit("e56_walk_probe: measuredRoundSeconds not found")
    return [float(value) for value in re.findall(r"[0-9.]+", match.group(1))]


def stream_cost_ratio(source: str) -> float:
    """Reproduce `verifyStreamCostRatio` from the measured round curve."""
    seconds = measured_round_seconds(source)
    marginal = [b - a for a, b in zip(seconds, seconds[1:])]
    return marginal[1] / ((marginal[0] + marginal[2]) / 2.0)


def streams(width: int, ipg: dict[int, int]) -> int:
    group = ipg.get(width)
    return 1 if group is None else -(-width // group)


def cost_table(source: str, priced: set[int] | None,
               depth_cap: int = 8) -> tuple[list[float], list[float]]:
    """Rebuild `marginalCostRatio` and `cumulativeCostRatio` for one arm.

    `priced=None` is the shipped scalar rule, whose table is flat.
    """
    h = swift_scalar(source, "headStepCostRatio")
    if priced is None:
        marginal = [h] * depth_cap
    else:
        ipg = inputs_per_group(source)
        ratio = stream_cost_ratio(source)
        crosses = [(d + 2) in priced
                   and streams(d + 2, ipg) > streams(d + 1, ipg)
                   for d in range(depth_cap)]
        count = sum(crosses)
        within = depth_cap * h / (depth_cap - count + count * ratio)
        marginal = [within * ratio if cross else within for cross in crosses]
    cumulative = [1.0]
    for step in marginal:
        cumulative.append(cumulative[-1] + step)
    return marginal, cumulative


def closed_steps(marginal: list[float], cumulative: list[float]) -> list[int]:
    """Steps no acceptance rate can take.

    The walk extends while `reach > marginal[d] * (1 + expected) /
    cumulative[d]`. `reach` never exceeds 1 and `expected >= d * reach`, so the
    step is unreachable whenever `marginal[d] * (d + 1) >= cumulative[d]`.
    """
    return [d for d in range(len(marginal))
            if marginal[d] * (d + 1) / cumulative[d] >= 1.0]


def walk(p: float, marginal: list[float], cumulative: list[float],
         cap: int = 8) -> int:
    reach, expected, depth = 1.0, 0.0, 0
    while depth < cap:
        reach *= p
        threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
        if reach <= threshold:
            break
        expected += reach
        depth += 1
    return depth


def check(source: str) -> int:
    """Guard the arm the checkout currently holds."""
    priced = priced_boundary_widths(source)
    declared = declared_closed_steps(source)
    marginal, cumulative = cost_table(source, priced)
    closed = closed_steps(marginal, cumulative)
    ipg = inputs_per_group(source)
    flat = sorted(w for w in priced if streams(w, ipg) <= streams(w - 1, ipg))
    print(f"e56_walk_probe --check: pricedBoundaryWidths={sorted(priced)} "
          f"declaredClosedDepthSteps={declared} computed_closed={closed}")
    failures = []
    if closed != declared:
        failures.append(
            f"closed steps {closed} do not match the declaration {declared}")
    if flat:
        failures.append(
            f"widths {flat} are priced but add no weight stream in the live "
            "dispatch table")
    mean = sum(marginal) / len(marginal)
    if abs(mean - swift_scalar(source, "headStepCostRatio")) > 1e-12:
        failures.append(f"mean price {mean} is off headStepCostRatio")
    for failure in failures:
        print(f"e56_walk_probe: FAIL {failure}")
    return 1 if failures else 0


def report(source: str) -> None:
    ipg = inputs_per_group(source)
    h = swift_scalar(source, "headStepCostRatio")
    print(f"inputsPerGroup read from source: {ipg}")
    print(f"measuredRoundSeconds read from source: "
          f"{measured_round_seconds(source)}")
    print(f"verifyStreamCostRatio {stream_cost_ratio(source):.6f}"
          f"  headStepCostRatio {h}")
    print(f"checkout arm: pricedBoundaryWidths="
          f"{sorted(priced_boundary_widths(source))} "
          f"declaredClosedDepthSteps={declared_closed_steps(source)}\n")

    tables = {name: cost_table(source, priced)
              for name, priced in ARMS.items()}

    print("Price of the depth -> depth + 1 step, as a fraction of a depth-0 "
          "round.")
    print("Every arm has the same MEAN price, so no arm is an h retune.")
    print("  depth  width step  crossing  " + "  ".join(
        f"{name:>9}" for name in ARMS))
    for depth in range(8):
        crossing = streams(depth + 2, ipg) > streams(depth + 1, ipg)
        row = "  ".join(f"{tables[name][0][depth]:9.6f}" for name in ARMS)
        print(f"    {depth}      {depth + 1}->{depth + 2}      "
              f"{str(crossing):<9} {row}")
    means = "  ".join(f"{sum(tables[name][0]) / 8:9.6f}" for name in ARMS)
    print(f"   mean                          {means}")

    print("\nPayable steps. A step is CLOSED at every acceptance rate when")
    print("marginal[d] * (d + 1) / cumulative[d] >= 1, because the walk needs")
    print("reach > that and reach <= 1 always.")
    for name in ARMS:
        marginal, cumulative = tables[name]
        floors = [marginal[d] * (d + 1) / cumulative[d] for d in range(8)]
        closed = closed_steps(marginal, cumulative)
        print(f"  {name:<6} required reach floor "
              f"{[round(value, 4) for value in floors]}")
        print(f"         closed steps {closed}"
              + (f"  -> verify width capped at {closed[0] + 1}"
                 if closed else ""))

    print("\nStop depth against a constant per-draft acceptance probability.")
    print("Cell values are the verify width the round runs (drafts + 1).")
    print("      p    " + "  ".join(f"{name:>7}" for name in ARMS))
    for p in (0.995, 0.99, 0.98, 0.95, 0.90, 0.875, 0.85, 0.835, 0.80, 0.75,
              0.70, 0.60):
        widths = "  ".join(f"{walk(p, *tables[name]) + 1:7d}" for name in ARMS)
        print(f"  {p:.3f}  {widths}")

    print("\nAcceptance rates that matter: local fixture 0.99, "
          "beagle 0.8351, medicine 0.8750.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the arm currently in the checkout")
    args = parser.parse_args()
    source = read_source()
    if args.check:
        raise SystemExit(check(source))
    report(source)


if __name__ == "__main__":
    main()
