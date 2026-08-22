"""E134 gates A and B (F5), read off the persisted replay artifacts.

Gate A asks the replayer to reproduce a ranked receipt it was never fitted to.
The flat-h family is the right instrument because one isolated ranked A/B pair
exists for it: a1326b4b against 036fd9ca, byte-identical except
headStepCostRatio, measuring h = 0.16 at -1.164 percent.

  A1  the argmax of the flat family lands at h = 0.18
  A2  h = 0.16 scores negative, inside the band -2.5 to -0.4

Gate B re-scores the boundary family under the curve that thorfinn's
`{6:6, 7:7}` one-pass table would produce.

Usage:
  python3 e134_gates.py --json e134-artifacts/gates-ab.json
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "e134-artifacts")

# The isolated ranked A/B on headStepCostRatio, the only external ranked
# observation of this family. Everything else in the tree is a local or
# replayed number and cannot calibrate the replayer.
RANKED_H016 = -1.164
RANKED_BAND = (-2.5, -0.4)
SHIPPED_H = 0.18


def load(name):
    with open(os.path.join(ART, name)) as handle:
        return json.load(handle)


def grid_of(doc):
    return {row["weight"]: row for row in doc["grid"]}


def interpolate_at_depth(rows, depth):
    """Flat-family score at a chosen mean depth, linear between neighbours.

    The matched-depth control the arm has to beat. A boundary-targeted arm is
    only interesting if it beats the flat price that reaches the same average
    depth, because reaching a different depth is a level effect and the level
    axis is closed.
    """
    ordered = sorted(rows, key=lambda r: r["mean_depth"])
    if depth <= ordered[0]["mean_depth"] or depth >= ordered[-1]["mean_depth"]:
        return None
    for lo, hi in zip(ordered, ordered[1:]):
        if lo["mean_depth"] <= depth <= hi["mean_depth"]:
            span = hi["mean_depth"] - lo["mean_depth"]
            if span == 0:
                return lo["median_pct"]
            frac = (depth - lo["mean_depth"]) / span
            return lo["median_pct"] + frac * (hi["median_pct"] - lo["median_pct"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    flat_ours = load("gateA-flat-ours.json")
    flat_board = load("gateA-flat-board.json")
    tier_ours = load("gateA-tier4-ours.json")
    b4 = load("gateB-cliff4-onepass67.json")
    b6 = load("gateB-cliff6-onepass67.json")

    report = {"harness": "ranked", "gate_a": {}, "gate_b": {}}

    print("=" * 76)
    print("E134 GATE A -- can the replayer reproduce a ranked receipt it has")
    print("never seen? harness=ranked model, zero GPU.")
    print("=" * 76)

    gate = flat_ours["gate"]
    print(
        f"\npopulation: {gate['legs']} legs, {gate['attached']} attached rounds, "
        f"{gate['unmatched']} unmatched, {gate['accept_mismatch']} accept "
        f"mismatches, {gate['margin_mismatch']} margin mismatches"
    )
    print(f"protocol: {flat_ours['windows']} windows, {flat_ours['seeds']} seeds, "
          f"leave-one-prompt-out\n")

    print(f"{'h':>8}{'ours %':>12}{'board %':>12}{'mean depth':>13}{'accept':>10}")
    go, gb = grid_of(flat_ours), grid_of(flat_board)
    for h in sorted(go):
        print(
            f"{h:>8.2f}{go[h]['median_pct']:>12.4f}{gb[h]['median_pct']:>12.4f}"
            f"{go[h]['mean_depth']:>13.3f}{go[h]['accept_rate']:>10.3f}"
        )

    for label, grid in (("ours", go), ("board", gb)):
        argmax = max(grid, key=lambda h: grid[h]["median_pct"])
        h016 = grid[0.16]["median_pct"]
        a1 = abs(argmax - SHIPPED_H) < 1e-9
        a2 = RANKED_BAND[0] <= h016 <= RANKED_BAND[1]
        report["gate_a"][label] = {
            "argmax_h": argmax,
            "h016_pct": h016,
            "ranked_h016_pct": RANKED_H016,
            "abs_error_pp": abs(h016 - RANKED_H016),
            "A1_pass": a1,
            "A2_pass": a2,
            "pass": a1 and a2,
        }
        print(
            f"\n{label} curve:"
            f"\n  A1 argmax at h = {argmax:.2f}  ->  {'PASS' if a1 else 'FAIL'}"
            f"\n  A2 h = 0.16 reads {h016:+.4f} %, band {RANKED_BAND[0]} to "
            f"{RANKED_BAND[1]}, ranked target {RANKED_H016:+.3f} %"
            f"  ->  {'PASS' if a2 else 'FAIL'}"
            f"\n     absolute error against the ranked receipt: "
            f"{abs(h016 - RANKED_H016):.4f} pp"
        )

    print("\n## supporting checks, which do not gate\n")
    for label, grid in (("ours", go), ("board", gb)):
        mono = grid[0.14]["median_pct"] < grid[0.15]["median_pct"] < grid[0.16][
            "median_pct"
        ]
        print(
            f"{label:>6}: h=0.14 and h=0.15 worse than h=0.16  -> "
            f"{'PASS' if mono else 'FAIL'}"
            f"   ({grid[0.14]['median_pct']:+.4f} < "
            f"{grid[0.15]['median_pct']:+.4f} < {grid[0.16]['median_pct']:+.4f})"
        )
        print(
            f"{label:>6}: h=0.32 negative -> "
            f"{'PASS' if grid[0.32]['median_pct'] < 0 else 'FAIL'}"
            f"   ({grid[0.32]['median_pct']:+.4f} %, "
            f"against a ranked reading near -3 %)"
        )
        report["gate_a"][label]["h032_pct"] = grid[0.32]["median_pct"]
        report["gate_a"][label]["monotone_below_shipped"] = mono

    # The decisive contrast. A boundary-targeted arm must beat the flat price
    # that reaches the SAME mean depth, or it is only a level effect.
    print("\n## the decisive contrast: boundary-targeted against matched depth\n")
    gt = grid_of(tier_ours)
    print(f"{'tier':>8}{'median %':>12}{'mean depth':>13}"
          f"{'flat @ same depth':>20}{'separation':>13}")
    contrasts = []
    for tier in sorted(gt):
        depth = gt[tier]["mean_depth"]
        flat_here = interpolate_at_depth(flat_ours["grid"], depth)
        if flat_here is None:
            continue
        sep = gt[tier]["median_pct"] - flat_here
        contrasts.append(
            {"tier": tier, "median_pct": gt[tier]["median_pct"],
             "mean_depth": depth, "flat_at_same_depth_pct": flat_here,
             "separation_pp": sep}
        )
        print(
            f"{tier:>8.4f}{gt[tier]['median_pct']:>12.4f}{depth:>13.3f}"
            f"{flat_here:>20.4f}{sep:>13.4f}"
        )
    report["gate_a"]["matched_depth_contrast"] = contrasts

    print("\n" + "=" * 76)
    print("E134 GATE B -- re-score under the post-{6:6, 7:7} curve")
    print("=" * 76)

    for label, doc, index in (("index 4", b4, 4), ("index 6", b6, 6)):
        rows = doc["grid"]
        argmax = max(rows, key=lambda r: r["median_pct"])
        lofo = doc["cliff_lofo"]
        report["gate_b"][f"cliff{index}"] = {
            "cliff_ratio": doc["cliff_ratio"],
            "argmax_w": argmax["weight"],
            "argmax_pct": argmax["median_pct"],
            "held_out_pct": lofo["held_out"],
            "in_sample_pct": lofo["in_sample"],
            "sd": lofo["sd"],
        }
        print(
            f"\n{label}: step ratio {doc['cliff_ratio']:.4f}, "
            f"marginal[{index}] = 0.18 * (1 + w * {doc['cliff_ratio'] - 1:.4f})"
        )
        print(f"{'w':>8}{'median %':>12}{'mean depth':>13}{'accept':>10}")
        for row in rows:
            print(
                f"{row['weight']:>8.4f}{row['median_pct']:>12.4f}"
                f"{row['mean_depth']:>13.3f}{row['accept_rate']:>10.3f}"
            )
        print(
            f"  argmax w = {argmax['weight']:.4f} at {argmax['median_pct']:+.4f} % ; "
            f"held out {lofo['held_out']:+.4f} %, sd {lofo['sd']:.4f}"
        )

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
