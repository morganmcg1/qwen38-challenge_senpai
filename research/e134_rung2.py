#!/usr/bin/env python3
"""E134 rung 2. Price depth policies that answer the rung-1 inversion.

harness=local instrument. Zero GPU.

Rung 1 found that the shipped estimator is not merely weak at the depth-4
boundary; on `medicine_hist` it is inverted, with a fit-free margin AUC of
0.2281 and a shipped-reach AUC of 0.1097. The cause is selection: the
depth-4 question is only asked in rounds that already accepted drafts 0-3,
so round-start optimism is anti-informative about the next boundary.

This module prices arms that answer that specific failure, through the same
E128 replayer that produced the published arm table, and validates every arm
by leaving one ranked prompt out of the hyperparameter choice.

The replayer resamples a recorded round's acceptance capability onto a target
prompt's survival curve and carries the recorded margin with it. This module
additionally carries the recorded round's full per-position margin VECTOR, so
a policy may read the previous round's margin shape. That transfer is weaker
than a live run: the vector is the recorded round's, while the accepted count
is the simulated one. Every number below is therefore a replayed estimate and
not a measured speedup.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e128_price import (  # noqa: E402
    DECODE_TOKENS, PROMPT_NAMES, RANKED_PROMPTS, load_board_receipt,
    median_of, pooled_positions, ranked_round_us, shift_p_vector, survival)
from e128_replay import (  # noqa: E402
    EMA_PRIOR, MAX_DEPTH, PRICE_CUMULATIVE, PRICE_MARGINAL,
    SEGMENTED_VERIFY_DEPTH_CAP, record_accept_outcome)
from e134_rung1 import parse_trace  # noqa: E402


def walk(ema, margin, offer, adjust=None, ctx=None, force=None, price=None):
    """The shipped walk with one hook on the per-step probability.

    `adjust(depth, p, ctx) -> p` is the ONLY departure from the shipped
    `cost_model_depth`. With `adjust=None` every value is bit identical to the
    shipped walk, so the `ship` arm below is a true baseline and not a
    reimplementation that happens to agree.

    `force(depth, ctx) -> bool | None` replaces the continue test at one
    boundary. It is not implementable: rung 3 uses it to give the walk perfect
    knowledge at a single named boundary and nowhere else.

    `price` is an optional `(marginal, cumulative)` pair that replaces the
    shipped depth price. It is a pure two-vector substitution, so an arm built
    on it is one edit to `makeUniformDepthPrice`.
    """
    marginal, cumulative = price or (PRICE_MARGINAL, PRICE_CUMULATIVE)
    cap = min(min(offer, MAX_DEPTH), SEGMENTED_VERIFY_DEPTH_CAP)
    if cap <= 0:
        return 0
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        if adjust is not None:
            p = adjust(depth, p, ctx)
        reach *= p
        verdict = None if force is None else force(depth, ctx)
        if verdict is None:
            threshold = marginal[depth] * (1.0 + expected) / cumulative[depth]
            verdict = reach > threshold
        if not verdict:
            break
        expected += reach
        depth += 1
    return depth


def slope_of(rows) -> float:
    """Least-squares slope of a per-position margin vector."""
    n = len(rows)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(rows) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(rows))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


# ------------------------------------------------------------------- the arms

def make_arm(name: str, k: float):
    """Every arm is one `adjust` hook, so each is implementable in Swift.

    `deep` arms act only at boundaries at or beyond `DEEP`, which is where
    rung 1 located the inversion. Shallow boundaries keep shipped behaviour.
    """
    deep = 4
    if name == "ship":
        return None
    if name == "shrinkdeep":
        # The estimator's inputs are anti-informative at deep boundaries, so
        # stop trusting them and fall back to the running base rate.
        def adjust(depth, p, ctx):
            if depth < deep:
                return p
            return (1.0 - k) * p + k * ctx["base_rate"]
        return adjust
    if name == "slopedeep":
        # `prev_margin_slope` is the one input rung 1 found still correctly
        # oriented at depth 4, AUC 0.7359 [0.6001 0.8717] on medicine_hist.
        def adjust(depth, p, ctx):
            if depth < deep:
                return p
            bump = k * math.tanh(ctx["prev_slope"])
            return min(0.999, max(0.001, p + bump))
        return adjust
    if name == "kmdeep":
        # Replace the shipped per-step estimate with the leg's own realised
        # conditional acceptance at that position.
        def adjust(depth, p, ctx):
            if depth < deep:
                return p
            emp = ctx["km"](depth)
            if emp is None:
                return p
            return (1.0 - k) * p + k * emp
        return adjust
    if name == "capdeep":
        # A pure control: never draft past `deep` unless the running base rate
        # is above `k`. No new information at all.
        def adjust(depth, p, ctx):
            if depth < deep:
                return p
            return p if ctx["base_rate"] >= k else 0.0
        return adjust
    raise SystemExit("unknown arm %r" % name)


def oracle_depth(offer: int, capability: int) -> int:
    """`e128_price.make_policy("oracle")`, the ceiling for any depth rule.

    It is not implementable. It is here so the framework can be checked
    against the published F10.1 value of +8.5248 percent.
    """
    cap_depth = min(offer, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
    best, best_cost = 0, ranked_round_us(1)
    for depth in range(1, cap_depth + 1):
        cost = ranked_round_us(depth + 1) / (1.0 + min(capability, depth))
        if cost < best_cost:
            best, best_cost = depth, cost
    return best


ARM_GRID = {
    "ship": [0.0],
    "oracle": [0.0],
    "shrinkdeep": [0.15, 0.30, 0.50, 0.70, 1.00],
    "slopedeep": [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20],
    "kmdeep": [0.25, 0.50, 0.75, 1.00],
    "capdeep": [0.70, 0.80, 0.85, 0.90, 0.95],
}
CONTROL_ARMS = ("ship", "oracle")


# --------------------------------------------------------------- the simulator

class VectorSampler:
    """`RoundSampler` that also carries the recorded per-position margins."""

    def __init__(self, rounds, p_fixture, p_target, seed=128):
        self.rounds = rounds
        self.s_fix = survival(p_fixture)
        self.s_target = survival(p_target)
        self.rng = random.Random(seed)
        self.cursor = self.rng.randrange(len(rounds))

    def draw(self):
        record = self.rounds[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.rounds)
        a, d = record["accepted"], record["depth"]
        high = self.s_fix[min(a, len(self.s_fix) - 1)]
        low = 0.0 if a >= d else self.s_fix[min(a + 1, len(self.s_fix) - 1)]
        u = self.rng.uniform(low, high)
        capability = 0
        for k in range(1, len(self.s_target)):
            if u <= self.s_target[k]:
                capability = k
            else:
                break
        return capability, record["margin"], record.get("rows", [])


def simulate(adjust, sampler: VectorSampler, windows: int,
             tokens: int = DECODE_TOKENS, oracle: bool = False,
             force=None, price=None) -> dict:
    total_us = total_tokens = 0.0
    total_depth = total_rounds = total_accepted = total_drafted = 0
    per_window = []
    depth_counts = [0] * (MAX_DEPTH + 2)
    cap_counts = [0] * (MAX_DEPTH + 2)
    boundary_asked = [0] * (MAX_DEPTH + 2)
    boundary_needed = [0] * (MAX_DEPTH + 2)
    for _ in range(windows):
        ema = list(EMA_PRIOR)
        emitted = 0
        window_us = 0.0
        prev_rows = []
        reached = [0] * (MAX_DEPTH + 1)
        passed = [0] * (MAX_DEPTH + 1)
        accepted_total = drafted_total = 0

        def km(depth, reached=reached, passed=passed):
            if reached[depth] < 8:
                return None
            return (passed[depth] + 0.5) / (reached[depth] + 1.0)

        while emitted < tokens:
            remaining = tokens - emitted
            offer = max(1, min(MAX_DEPTH, remaining - 1))
            capability, margin, rows = sampler.draw()
            ctx = {
                "prev_slope": slope_of(prev_rows),
                "prev_rows": prev_rows,
                "base_rate": (accepted_total / drafted_total
                              if drafted_total else 0.85),
                "km": km,
                "capability": capability,
                "offer": offer,
            }
            depth = (oracle_depth(offer, capability) if oracle
                     else walk(ema, margin, offer, adjust, ctx, force,
                               price))
            accepted = min(capability, depth)
            depth_counts[min(depth, MAX_DEPTH + 1)] += 1
            cap_counts[min(capability, MAX_DEPTH + 1)] += 1
            walk_cap = min(offer, MAX_DEPTH, SEGMENTED_VERIFY_DEPTH_CAP)
            for j in range(min(depth + 1, walk_cap)):
                boundary_asked[j] += 1
                if capability > j:
                    boundary_needed[j] += 1
            for j in range(depth):
                reached[j] += 1
                if j < accepted:
                    passed[j] += 1
            window_us += ranked_round_us(depth + 1)
            emitted += min(1 + accepted, remaining)
            total_depth += depth
            total_rounds += 1
            total_accepted += accepted
            total_drafted += depth
            accepted_total += accepted
            drafted_total += depth
            ema = record_accept_outcome(ema, accepted, depth)
            prev_rows = list(rows[:accepted + 1]) if rows else []
        total_us += window_us
        total_tokens += tokens
        per_window.append(window_us / tokens)
    return {
        "us_per_token": total_us / total_tokens,
        "us_per_token_sem": (statistics.stdev(per_window)
                             / math.sqrt(len(per_window))
                             if len(per_window) > 1 else 0.0),
        "mean_depth": total_depth / total_rounds,
        "accept_rate": (total_accepted / total_drafted
                        if total_drafted else 0.0),
        "rounds": total_rounds,
        "depth_counts": depth_counts,
        "cap_counts": cap_counts,
        "boundary_asked": boundary_asked,
        "boundary_needed": boundary_needed,
    }


# ------------------------------------------------------------------- the panel

def build_legs(accept: pathlib.Path, runs: pathlib.Path) -> tuple[dict, dict]:
    """Published E128 legs, with each round's margin vector attached.

    Attachment is keyed by the round number the trace itself prints, never by
    list position, because the published `rounds_detail` and a fresh parse do
    not always hold the same number of rounds. The gate then re-checks the
    accepted count and the scalar margin on every attached round, so a slipped
    or mismatched attachment is reported rather than silently priced.
    """
    legs = {leg["prompt_id"]: leg
            for leg in json.loads(accept.read_text())["legs"]}
    gate = {"attached": 0, "accept_mismatch": 0, "margin_mismatch": 0,
            "unmatched": 0, "legs": 0}
    for name, leg in legs.items():
        path = runs / name / "trace.txt"
        if not path.is_file():
            continue
        rich, _ = parse_trace(path)
        by_round = {record["round"]: record for record in rich}
        gate["legs"] += 1
        for target in leg["rounds_detail"]:
            source = by_round.get(target["round"])
            if source is None:
                gate["unmatched"] += 1
                target["rows"] = []
                continue
            if source["acc"] != target["accepted"]:
                gate["accept_mismatch"] += 1
            if abs(source["margin"] - target["margin"]) > 1e-6:
                gate["margin_mismatch"] += 1
            target["rows"] = source["rows"]
            gate["attached"] += 1
    return legs, gate


def fit_transfer(factory, p_fixture, target, windows) -> tuple:
    """`e128_price.fit_delta` against this module's simulator.

    Identical bisection and identical clamping. It is duplicated only because
    the published version calls the two-value sampler directly, and this
    module's sampler returns the margin vector as a third value.
    """
    def value_at(delta):
        return simulate(None, factory(shift_p_vector(p_fixture, delta)),
                        windows)["mean_depth"]

    low, high = -8.0, 12.0
    d_low, d_high = value_at(low), value_at(high)
    if target <= d_low:
        return low, d_low, d_high
    if target >= d_high:
        return high, d_low, d_high
    for _ in range(40):
        mid = 0.5 * (low + high)
        if value_at(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high), d_low, d_high


def prompt_panel(legs: dict, windows: int, fit_windows: int,
                 seed: int) -> dict:
    """One transfer-fitted sampler factory and shipped baseline per prompt."""
    panel = {}
    for prompt, spec in RANKED_PROMPTS.items():
        fixture = spec["fixture"]
        chosen = [legs[name] for name in fixture if name in legs]
        if len(chosen) != len(fixture):
            print("skip %s: missing %r" % (
                prompt, [n for n in fixture if n not in legs]))
            continue
        p_fixture = pooled_positions(chosen)
        rounds = [r for leg in chosen for r in leg["rounds_detail"]]

        def factory(p_target, rounds=rounds, p_fixture=p_fixture):
            return VectorSampler(rounds, p_fixture, p_target, seed=seed)

        delta, _, _ = fit_transfer(factory, p_fixture, spec["depth"],
                                   fit_windows)
        p_target = shift_p_vector(p_fixture, delta)
        base = simulate(None, factory(p_target), windows)
        panel[prompt] = {"factory": factory, "p_target": p_target,
                         "delta": delta, "ship": base,
                         "target_depth": spec["depth"]}
    return panel


def arm_ratios(panel: dict, name: str, k: float, windows: int) -> dict:
    """Candidate seconds-per-token ratio against the shipped arm, per prompt."""
    oracle = name == "oracle"
    adjust = None if oracle else make_arm(name, k)
    out = {}
    for prompt, entry in panel.items():
        run = simulate(adjust, entry["factory"](entry["p_target"]), windows,
                       oracle=oracle)
        out[prompt] = {
            "ratio": run["us_per_token"] / entry["ship"]["us_per_token"],
            "mean_depth": run["mean_depth"],
            "accept_rate": run["accept_rate"]}
    return out


def signal_survival(panel: dict, windows: int, depth: int = 4) -> dict:
    """Does the replayer still CONTAIN the signal rung 1 measured?

    `VectorSampler` collapses a round to one scalar capability and then sets
    `accepted = min(capability, depth)`, so the per-position conditional is
    the target survival curve by construction. If that erases the link between
    the previous round's margin shape and acceptance at one boundary, then no
    arm built on that link can be priced here, whatever the arm table says.

    This measures the link inside the simulator on exactly the rounds the
    boundary applies to, so it is directly comparable with the rung-1 AUC.
    """
    out = {}
    for prompt, entry in panel.items():
        sampler = entry["factory"](entry["p_target"])
        scores, labels = [], []
        ema = list(EMA_PRIOR)
        prev_rows = []
        for _ in range(windows * 200):
            capability, margin, rows = sampler.draw()
            chosen = walk(ema, margin, MAX_DEPTH)
            accepted = min(capability, chosen)
            if chosen > depth and accepted >= depth:
                scores.append(slope_of(prev_rows))
                labels.append(1.0 if accepted > depth else 0.0)
            ema = record_accept_outcome(ema, accepted, chosen)
            prev_rows = list(rows[:accepted + 1]) if rows else []
        pos = sum(labels)
        if len(labels) < 30 or pos in (0, len(labels)):
            out[prompt] = None
            continue
        pairs = concordance(scores, labels)
        out[prompt] = {"auc": pairs, "n": len(labels), "rate": pos / len(labels)}
    return out


def concordance(scores, labels) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        stop = index
        while (stop + 1 < len(order)
               and scores[order[stop + 1]] == scores[order[index]]):
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for j in range(index, stop + 1):
            ranks[order[j]] = shared
        index = stop + 1
    pos = sum(1 for v in labels if v > 0.5)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum = sum(r for r, v in zip(ranks, labels) if v > 0.5)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def median_pct(receipt: dict, ratios: dict) -> float:
    """Replayed ranked median gain against the shipped arm, in percent."""
    raws, ship = [], []
    for prompt, entry in receipt["per_prompt"].items():
        ratio = ratios.get(prompt, {}).get("ratio", 1.0)
        raws.append(entry["serial"] / (entry["candidate"] * ratio))
        ship.append(entry["serial"] / entry["candidate"])
    return (median_of(raws) / median_of(ship) - 1.0) * 100.0


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path,
                    default=here.parent / ".mlxfast-private/e128/runs-forced")
    ap.add_argument("--accept", type=pathlib.Path,
                    default=here / "e128-artifacts/rung1-forced.json")
    ap.add_argument("--board", type=pathlib.Path,
                    default=pathlib.Path("/tmp/yukon-board/full.json"))
    ap.add_argument("--receipt", default="d3c491b5")
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--fit-windows", type=int, default=24)
    ap.add_argument("--offered-depth", type=int, default=7)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", type=pathlib.Path,
                    default=here / "e134-artifacts/rung2-arms.json")
    args = ap.parse_args()

    print("harness=local instrument  E134 rung 2  zero GPU")
    print("replayed ranked median against receipt %s\n" % args.receipt)

    legs, gate = build_legs(args.accept, args.runs)
    print("## attachment gate")
    print("  legs                                  %d" % gate["legs"])
    print("  rounds with a margin vector attached  %d" % gate["attached"])
    print("  accepted-count mismatches             %d"
          % gate["accept_mismatch"])
    print("  scalar-margin mismatches              %d"
          % gate["margin_mismatch"])
    print("  rounds with no matching trace round   %d" % gate["unmatched"])
    if gate["accept_mismatch"] or gate["margin_mismatch"] or gate["unmatched"]:
        print("  ATTACHMENT IS NOT PROVEN - every number below is void")

    receipt = load_board_receipt(args.board, args.receipt)
    seeds = [args.seed + i for i in range(args.seeds)]

    # Every arm is compared with the shipped arm under the SAME sampler seed,
    # so common random numbers cancel most of the replay variance. The error
    # bar that matters is therefore the spread of that paired difference
    # across seeds, not the spread of the shipped arm on its own.
    table = {(n, k): [] for n, grid in ARM_GRID.items() for k in grid}
    ratios_by_seed = {seed: {} for seed in seeds}
    panels = {}
    for seed in seeds:
        panel = prompt_panel(legs, args.windows, args.fit_windows, seed)
        panels[seed] = panel
        for name, grid in ARM_GRID.items():
            for k in grid:
                ratios = arm_ratios(panel, name, k, args.windows)
                ratios_by_seed[seed][(name, k)] = ratios
                table[(name, k)].append(median_pct(receipt, ratios))

    panel = panels[seeds[0]]
    print("\n## transfer fit, one per ranked prompt, at seed %d" % seeds[0])
    print("%-10s %8s %10s %10s %9s" % (
        "prompt", "delta", "tgt depth", "sim depth", "weight"))
    for prompt, entry in panel.items():
        print("%-10s %8.4f %10.3f %10.3f %9.4f" % (
            prompt, entry["delta"], entry["target_depth"],
            entry["ship"]["mean_depth"], RANKED_PROMPTS[prompt]["weight"]))

    print("\n## every arm at every grid point, IN SAMPLE, over %d seeds"
          % len(seeds))
    print("   `sd` is the paired seed-to-seed spread and is the noise floor.")
    print("%-12s %7s %12s %9s %9s" % (
        "arm", "k", "median pct", "sd", "t"))
    summary = {}
    for name, grid in ARM_GRID.items():
        for k in grid:
            values = table[(name, k)]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            se = sd / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            summary[(name, k)] = {"mean": mean, "sd": sd, "se": se,
                                  "values": values}
            print("%-12s %7.2f %+12.4f %9.4f %9.2f" % (
                name, k, mean, sd, mean / se if se else float("nan")))

    print("\n## leave-one-prompt-out, which is the headline")
    print("   For each ranked prompt the grid point is chosen WITHOUT that")
    print("   prompt, then applied to it, at every seed. The median therefore")
    print("   contains no in-sample grid choice for any prompt.")
    print("%-12s %12s %12s %9s %10s" % (
        "arm", "in-sample", "held-out", "sd", "gap"))
    held = {}
    for name, grid in ARM_GRID.items():
        if name in CONTROL_ARMS:
            continue
        in_sample = max(summary[(name, k)]["mean"] for k in grid)
        per_seed, chosen = [], {}
        for seed in seeds:
            ratios = {}
            for prompt in panel:
                best, best_k = None, None
                for k in grid:
                    others = {p: r for p, r
                              in ratios_by_seed[seed][(name, k)].items()
                              if p != prompt}
                    value = median_pct(receipt, others)
                    if best is None or value > best:
                        best, best_k = value, k
                chosen.setdefault(prompt, []).append(best_k)
                ratios[prompt] = ratios_by_seed[seed][(name, best_k)][prompt]
            per_seed.append(median_pct(receipt, ratios))
        mean = statistics.fmean(per_seed)
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
        held[name] = {"in_sample": in_sample, "held_out": mean,
                      "held_out_sd": sd, "held_out_values": per_seed,
                      "chosen_k": {p: v for p, v in chosen.items()}}
        print("%-12s %+12.4f %+12.4f %9.4f %+10.4f" % (
            name, in_sample, mean, sd, mean - in_sample))

    print("\n## instrument check: does the replayer still hold the signal?")
    print("   AUC of `prev_margin_slope` for acceptance at depth 4, measured")
    print("   INSIDE the simulator. Rung 1 measured 0.7359 on medicine_hist")
    print("   from the real trace. A value near 0.5000 means the replayer has")
    print("   erased the effect and cannot price any arm that uses it.")
    print("%-10s %8s %8s %8s" % ("prompt", "auc", "n", "rate"))
    survival_check = signal_survival(panel, args.windows)
    for prompt, entry in survival_check.items():
        if entry is None:
            print("%-10s %8s" % (prompt, "-"))
            continue
        print("%-10s %8.4f %8d %8.3f" % (
            prompt, entry["auc"], entry["n"], entry["rate"]))

    print("\n## controls")
    for name in CONTROL_ARMS:
        entry = summary[(name, ARM_GRID[name][0])]
        print("  %-10s %+10.4f  sd %.4f" % (name, entry["mean"], entry["sd"]))
    print("  F10.1 published oracle  +8.5248  (the framework check)")

    best = max(held.items(), key=lambda kv: kv[1]["held_out"], default=None)
    if best is not None:
        value = best[1]["held_out"]
        print("\n## verdict")
        print("  best held-out arm                 %s" % best[0])
        print("  e134_replayed_ranked_median_pct   %+.4f +- %.4f"
              % (value, best[1]["held_out_sd"]))
        print("  shipping bar                      +0.5000")
        print("  close-the-axis bar                +0.3000")
        print("  VERDICT: %s" % (
            "ADVANCE to rung 4" if value >= 0.50
            else "KEEP OPEN, below the shipping bar" if value >= 0.30
            else "CLOSE THE AXIS, nothing clears +0.30"))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "harness": "local instrument", "gpu_used": False,
        "receipt": receipt["id"], "receipt_score": receipt["score"],
        "windows": args.windows, "seeds": seeds, "attachment_gate": gate,
        "in_sample": {"%s@%s" % (n, k): v for (n, k), v in summary.items()},
        "held_out": held,
        "signal_survival": survival_check,
        "transfer": {p: {"delta": e["delta"],
                         "simulated_depth": e["ship"]["mean_depth"],
                         "target_depth": e["target_depth"]}
                     for p, e in panel.items()},
    }, indent=2) + "\n")
    print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
