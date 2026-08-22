#!/usr/bin/env python3
"""E140: replace the first-break depth walk with an explicit argmax.

`harness=local instrument`. Zero GPU.

The shipped `costModelDepth` takes a step exactly when that one step raises
the ratio `E / C` of expected emitted tokens over round cost, and it stops at
the first step that does not. That greedy returns the global argmax only when
`E / C` is quasiconcave in depth. Under the flat shipped price it is, and this
module proves that as a gate. Under the E134 item 2 measured ranked cost curve
it is not, because the step into verify width 7 is one tenth of the step into
width 6, so the ratio dips at the cliff and recovers immediately after it.

This module owns the policy, the price tables and the falsification gates.
`e140_cells.py` prices the 2x2 on the E128 replayer.

Gate, run by `main`:

  1. The two walks are compared on the 617 recorded shipped-trace rounds, on
     bit-identical round-start state, so no simulator divergence can hide a
     disagreement.
  2. Under the flat shipped price the disagreement count must be exactly 0.
  3. Under the measured curve the disagreement count must be greater than 0.
     That is the positive control: it proves the comparison is able to fail.

Usage:
  python3 e140_lookahead.py --json e140-artifacts/gate-cellC.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
from e128_price import MAX_DEPTH, ranked_price_table  # noqa: E402
from e128_replay import (  # noqa: E402
    PRICE_CUMULATIVE, PRICE_MARGINAL, SEGMENTED_VERIFY_DEPTH_CAP,
)
from e134_rung1 import parse_trace  # noqa: E402
from e134_rung2 import walk  # noqa: E402
from e134_rung3 import OUR_CURVE, boundary_price  # noqa: E402

MEASURED_CURVE_PATH = HERE / "e134-artifacts/item2-measured-curve.json"

# The four fitted forms of the E134 item 2 inversion. `pre_arm` is the
# uncorrected `slopeonly_b6` fit of our own receipts; the other three differ
# only in how the uniform templating term is attributed.
CURVE_FORMS = ("pre_arm", "per_round", "per_drafting_round", "proportional")

SHIPPED_TIER = 1.45
SHIPPED_CLIFF = 4  # `marginal[4]` prices the step into verify width 6


def load_curves(path: pathlib.Path = MEASURED_CURVE_PATH) -> dict:
    """The four curve forms, keyed exactly as `CURVE_FORMS` names them."""
    blob = json.loads(path.read_text())
    curves = {"pre_arm": dict(OUR_CURVE, name="ours_pre_arm")}
    for form, entry in blob["curves"].items():
        if form != "pre_arm":
            curves[form] = entry["curve"]
    return curves, blob["best_form"]


def walk_argmax(ema, margin, offer, price=None):
    """The shipped walk's objective, maximised over every feasible depth.

    Every input to the per-step probability is the shipped input: the same
    `positionAcceptEMA` entry, and the same depth-0 and depth-1 margin clamps
    at the same divisors. The ONLY change is that the loop no longer breaks at
    the first non-improving step; it records the value of stopping at every
    depth and returns the best.

    `value` at depth `d` is `E_d / C_d` with `E_d = 1 + sum_{i<d} reach_i` the
    expected emitted tokens and `C_d = cumulative[d]` the round cost. The
    comparison is strict, so a tie keeps the shallower depth, which is the
    same tie-break the shipped `guard reach > threshold` makes.
    """
    _, cumulative = price or (PRICE_MARGINAL, PRICE_CUMULATIVE)
    cap = min(min(offer, MAX_DEPTH), SEGMENTED_VERIFY_DEPTH_CAP)
    if cap <= 0:
        return 0
    best, best_value = 0, 1.0 / cumulative[0]
    reach, expected, depth = 1.0, 0.0, 0
    have_margin = not math.isnan(margin)
    while depth < cap:
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        reach *= p
        expected += reach
        depth += 1
        value = (1.0 + expected) / cumulative[depth]
        if value > best_value:
            best_value, best = value, depth
    return best


def value_profile(ema, margin, offer, price=None):
    """`E_d / C_d` at every feasible depth, for the convexity diagnostic."""
    _, cumulative = price or (PRICE_MARGINAL, PRICE_CUMULATIVE)
    cap = min(min(offer, MAX_DEPTH), SEGMENTED_VERIFY_DEPTH_CAP)
    values = [1.0 / cumulative[0]]
    reach, expected = 1.0, 0.0
    have_margin = not math.isnan(margin)
    for depth in range(cap):
        p = ema[depth]
        scale = {0: 2.0, 1: 3.0}.get(depth)
        if scale is not None and have_margin:
            p = min(p, 1.0 / (1.0 + math.exp(-margin / scale)))
        reach *= p
        expected += reach
        values.append((1.0 + expected) / cumulative[depth + 1])
    return values


def is_quasiconcave(values, tol: float = 0.0) -> bool:
    """True when the sequence never rises again after it has fallen."""
    fell = False
    for index in range(1, len(values)):
        if values[index] < values[index - 1] - tol:
            fell = True
        elif fell and values[index] > values[index - 1] + tol:
            return False
    return True


# ------------------------------------------------------------- price tables

def flat_price():
    """The shipped uniform 0.18 table, exactly as `e128_replay` holds it."""
    return list(PRICE_MARGINAL), list(PRICE_CUMULATIVE)


def curve_price(curve):
    """`ranked_price_table` under one installed curve. No global left behind."""
    saved = e128_price.CURVE
    e128_price.CURVE = curve
    try:
        marginal, cumulative = ranked_price_table()
    finally:
        e128_price.CURVE = saved
    return list(marginal), list(cumulative)


def pb6_price(tier: float = SHIPPED_TIER, cliff: int = SHIPPED_CLIFF):
    """`makeBoundaryDepthPrice(enteringVerifyWidth: 6, tier: 1.45)`."""
    marginal, cumulative, _ = boundary_price(tier, cliff)
    return list(marginal), list(cumulative)


def prefix_consistency(price) -> float:
    """Largest gap between `cumulative` and the prefix sums of `marginal`.

    `walk` reads `marginal[d]` and `cumulative[d]`; `walk_argmax` reads
    `cumulative[d + 1]` instead of the pair. The two are the same quantity
    only if the table is internally consistent, and the shipped
    `makeUniformDepthPrice` builds `cumulative` from a closed form rather
    than by accumulation, so the two forms can differ by an ulp. A tie broken
    by that ulp would be a real cell-C disagreement, which is exactly what
    this number bounds.
    """
    marginal, cumulative = price
    worst, running = 0.0, cumulative[0]
    for depth, step in enumerate(marginal):
        running += step
        if depth + 1 < len(cumulative):
            worst = max(worst, abs(running - cumulative[depth + 1]))
    return worst


def price_shape(price, label, curve=None):
    marginal, cumulative = price
    rows = {"label": label,
            "marginal": list(marginal),
            "cumulative": list(cumulative),
            "prefix_gap": prefix_consistency(price),
            "total": sum(marginal)}
    if curve is not None:
        saved = e128_price.CURVE
        e128_price.CURVE = curve
        try:
            us = [e128_price.ranked_round_us(m)
                  for m in range(1, MAX_DEPTH + 2)]
        finally:
            e128_price.CURVE = saved
        rows["round_us"] = us
        rows["step_us"] = [us[i + 1] - us[i] for i in range(len(us) - 1)]
    return rows


# ------------------------------------------------------- the recorded rounds

def load_recorded(directory: pathlib.Path) -> tuple[list[dict], dict]:
    """Every recorded shipped-trace round, minus each leg's final round.

    The final round of a leg is clamped by the tokens still owed to the fixed
    512-token window rather than by the walk, so no depth rule can change it.
    `e134_fm1_audit` drops the same rounds for the same reason, which is why
    this returns 617 rounds from 622 across the five archived legs.
    """
    records, gates = [], {}
    if not directory.is_dir():
        raise SystemExit("no recorded traces at %s" % directory)
    for leg in sorted(p for p in directory.iterdir() if p.is_dir()):
        trace = leg / "trace.txt"
        if not trace.is_file():
            continue
        rounds, gate = parse_trace(trace)
        gates[leg.name] = gate
        for index, record in enumerate(rounds):
            if index == len(rounds) - 1:
                continue
            records.append(dict(record, leg=leg.name))
    return records, gates


def compare_walks(records, price) -> dict:
    """Greedy against argmax on bit-identical recorded round-start state."""
    out = {"rounds": len(records), "disagreements": 0, "deeper": 0,
           "shallower": 0, "pairs": {}, "non_quasiconcave": 0,
           "greedy_depths": [0] * (MAX_DEPTH + 1),
           "argmax_depths": [0] * (MAX_DEPTH + 1),
           "examples": []}
    for record in records:
        offer = record["cap"]
        greedy = walk(record["ema"], record["margin"], offer, price=price)
        best = walk_argmax(record["ema"], record["margin"], offer, price=price)
        out["greedy_depths"][greedy] += 1
        out["argmax_depths"][best] += 1
        values = value_profile(record["ema"], record["margin"], offer,
                               price=price)
        if not is_quasiconcave(values):
            out["non_quasiconcave"] += 1
        if greedy == best:
            continue
        out["disagreements"] += 1
        if best > greedy:
            out["deeper"] += 1
        else:
            out["shallower"] += 1
        key = "%d->%d" % (greedy, best)
        out["pairs"][key] = out["pairs"].get(key, 0) + 1
        if len(out["examples"]) < 6:
            out["examples"].append({
                "leg": record["leg"], "round": record["round"],
                "offer": offer, "greedy": greedy, "argmax": best,
                "acc": record["acc"],
                "value_profile": values})
    return out


def report(label, result) -> None:
    print("\n## %s" % label)
    print("rounds inspected                 %d" % result["rounds"])
    print("depth disagreements              %d" % result["disagreements"])
    print("  argmax deeper than greedy      %d" % result["deeper"])
    print("  argmax shallower than greedy   %d" % result["shallower"])
    print("rounds with a non-quasiconcave E/C profile   %d"
          % result["non_quasiconcave"])
    print("greedy depth histogram   %s" % " ".join(
        "%d:%d" % (d, n) for d, n in enumerate(result["greedy_depths"]) if n))
    print("argmax depth histogram   %s" % " ".join(
        "%d:%d" % (d, n) for d, n in enumerate(result["argmax_depths"]) if n))
    if result["pairs"]:
        print("transitions              %s" % " ".join(
            "%s x%d" % (k, v) for k, v in sorted(result["pairs"].items())))
    for case in result["examples"]:
        print("   %s round %d offer %d  greedy %d -> argmax %d  acc %d"
              % (case["leg"], case["round"], case["offer"], case["greedy"],
                 case["argmax"], case["acc"]))
        print("      E/C  %s" % " ".join(
            "%.5f" % v for v in case["value_profile"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", type=pathlib.Path,
                    default=HERE.parent / ".mlxfast-private/e128/runs-shipped")
    ap.add_argument("--json", type=pathlib.Path,
                    default=HERE / "e140-artifacts/gate-cellC.json")
    args = ap.parse_args()

    print("harness=local instrument  E140 cell C gate  zero GPU")
    curves, best_form = load_curves()
    print("measured curve forms: %s ; E134 best form %s"
          % (", ".join(CURVE_FORMS), best_form))

    records, gates = load_recorded(args.shipped)
    print("\n## trace parse gate")
    total_bad = 0
    for name, gate in sorted(gates.items()):
        total_bad += gate["row_count_bad"] + gate["margin_identity_bad"]
        print("%-20s rounds %4d  row-count bad %d  margin-identity bad %d  "
              "sched max err %.2e" % (
                  name, gate["rounds"], gate["row_count_bad"],
                  gate["margin_identity_bad"], gate["sched_max_abs_error"]))
    print("scored rounds after dropping each leg's budget-clamped final round"
          "  %d" % len(records))

    prices = {"flat 0.18 (cell A/C price)": flat_price(),
              "pb6 tier 1.45 (cell E price)": pb6_price()}
    for form in CURVE_FORMS:
        prices["measured %s (cell B/D price)" % form] = curve_price(
            curves[form])

    print("\n## price tables in force")
    print("%-34s %9s %9s   %s" % ("price", "total", "prefix gap", "marginal"))
    shapes = {}
    for label, price in prices.items():
        curve = None
        for form in CURVE_FORMS:
            if label.endswith("(cell B/D price)") and form in label:
                curve = curves[form]
        shapes[label] = price_shape(price, label, curve)
        print("%-34s %9.4f %9.1e   %s" % (
            label, shapes[label]["total"], shapes[label]["prefix_gap"],
            " ".join("%.4f" % v for v in price[0])))

    results = {}
    for label, price in prices.items():
        results[label] = compare_walks(records, price)
        report(label, results[label])

    flat_label = "flat 0.18 (cell A/C price)"
    gate_value = results[flat_label]["disagreements"]
    controls = {label: results[label]["disagreements"]
                for label in prices if label.startswith("measured")}
    control_ok = any(v > 0 for v in controls.values())

    print("\n## verdict")
    print("e140_cellC_depth_disagreements   %d   (must be exactly 0)"
          % gate_value)
    print("positive control, measured curves: %s" % ", ".join(
        "%s=%d" % (k.split()[1], v) for k, v in controls.items()))
    print("the comparison can fail:          %s" % control_ok)
    verdict = ("PASS" if gate_value == 0 and control_ok
               else "FAIL - repair before pricing cells B and D")
    print("VERDICT: %s" % verdict)

    payload = {
        "harness": "local instrument", "gpu_used": False,
        "shipped_traces": str(args.shipped),
        "trace_gates": gates,
        "scored_rounds": len(records),
        "trace_parse_bad": total_bad,
        "price_shapes": shapes,
        "walk_comparison": {k: {kk: vv for kk, vv in v.items()
                                if kk != "examples"}
                            for k, v in results.items()},
        "examples": {k: v["examples"] for k, v in results.items()},
        "e140_cellC_depth_disagreements": gate_value,
        "positive_control_disagreements": controls,
        "positive_control_can_fail": control_ok,
        "verdict": verdict,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print("\nwrote %s" % args.json)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
