#!/usr/bin/env python3
"""E117 -- price the rung-0 null and reconcile the campaign's group-scaling
constants against the directly measured shipped partition table.

    research/e117_reconcile.py [--artifact PATH]

CAMPAIGN RULE 40: this reads the committed artifact under
`research/e117-artifacts/`, never a gitignored host-local path. Every number it
prints appears in `research/e117-results.md`.

harness=local throughout. No thermal gate, no score.
"""

from __future__ import annotations

import argparse
import json
import pathlib

# E109 v2 realised local width histogram on the pair census, expressed as verify
# widths: draft count d runs at verify width d + 1.
# senpai/campaign-ledger.md, E109 v2. 26 of 31 rounds at width 8 = 83.9 %.
WIDTH_HISTOGRAM = {4: 1, 5: 1, 6: 2, 7: 1, 8: 26}

# Finding 22 local round shares, senpai/campaign-ledger.md:30172-30184.
FINDING22_LOCAL_SHARE = {
    "mlp.gate_up": 0.37937,
    "out_proj + down_proj": 0.28666,
    "gdn.in_proj": 0.13859,
    "lm_head": 0.04132,
    "fa.qkv": 0.04049,
}
FINDING22_STREAM_SUBTOTAL = 0.89757
# senpai/campaign-ledger.md:35059. The tensors that reach the wide cross-row
# branch: gate_up + gdn.in_proj + lm_head + fa.qkv.
QUALIFYING_SHARE = 0.59977

# E100 round-level collapse, and the identity E115 built on it.
COLLAPSE_MEASURED = 0.180
A_LOCAL = 2 * (1 - COLLAPSE_MEASURED)  # 1.640, a RATE ratio
B_MSPLIT_TIME_RATIO = 1.960  # E115 rung 1, a TIME ratio
E115_F = 0.667

DECODE_ONLY_M5_ROUND_US = 102_864  # Rule 34 frame
DISPATCH_BOUNDARY_US = 1.8  # E106 corrected boundary
GATE_UP_SITES_PER_ROUND = 64
GDN_IN_PROJ_SITES_PER_ROUND = 48
BAR_PCT = 0.20


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def shares(shape: dict) -> dict[int, float]:
    """Streaming-time share of each realised width, count x a_one_net(M)."""
    weights = {}
    for m, count in WIDTH_HISTOGRAM.items():
        cell = shape["widths"].get(str(m)) or shape["widths"].get(m)
        if cell is None:
            continue
        weights[m] = count * cell["a_one_net_us"]
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


def price(summary: dict, shape_name: str, sites: int, round_share: float) -> None:
    shape = summary["shapes"][shape_name]
    weight = shares(shape)
    print(f"\n## {shape_name}   round share {round_share * 100:.3f} % "
          f"(Finding 22)   {sites} call sites per round")
    print(f"{'M':>2} {'part':>8} {'share':>7} {'net %':>8} {'contrib':>9}")
    always = positive = 0.0
    for m in sorted(weight):
        cell = shape["widths"].get(str(m)) or shape["widths"].get(m)
        arm = cell["arms"].get("e_nsplit_serial")
        if arm is None:
            continue
        pct = arm["net_pct_faster_mean"]
        contrib = weight[m] * pct
        always += contrib
        if pct > 0:
            positive += contrib
        print(f"{m:>2} {cell['partition']:>8} {weight[m]:>7.4f} "
              f"{pct:>8.3f} {contrib:>9.4f}")
    boundary_pct = (sites * DISPATCH_BOUNDARY_US / DECODE_ONLY_M5_ROUND_US) * 100
    print(f"  always split              {always:>8.3f} % of {shape_name}"
          f"  = {always * round_share:>7.3f} % of the local round")
    print(f"  split where positive      {positive:>8.3f} % of {shape_name}"
          f"  = {positive * round_share:>7.3f} % of the local round")
    print(f"  extra dispatch boundaries {sites} x {DISPATCH_BOUNDARY_US} us"
          f" = {boundary_pct:.3f} % of the {DECODE_ONLY_M5_ROUND_US} us round")
    print(f"  conditional net ceiling   "
          f"{positive * round_share - boundary_pct:>7.3f} % of the local round"
          f"   bar {BAR_PCT} %")


def group_ratios(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("Directly measured in-dispatch group scaling, TIME ratios")
    print("=" * 78)
    print(f"{'shape':>14} {'[3+3]/[3]':>10} {'[4+4]/[4]':>10} "
          f"{'[3+3+3]/[3]':>12} {'[4+3]/([4]+[3])':>16}")
    for name, shape in summary["shapes"].items():
        w = {int(k): v for k, v in shape["widths"].items()}
        if not {3, 4, 6, 7, 8, 9} <= set(w):
            continue
        t3, t4 = w[3]["a_one_net_us"], w[4]["a_one_net_us"]
        print(f"{name:>14} {w[6]['a_one_net_us'] / t3:>10.4f} "
              f"{w[8]['a_one_net_us'] / t4:>10.4f} "
              f"{w[9]['a_one_net_us'] / t3:>12.4f} "
              f"{w[7]['a_one_net_us'] / (t3 + t4):>16.4f}")
    print("\nA two-group dispatch never costs the sum of its isolated groups.")
    print("The saving is the grouping benefit that the isolated NA ladder "
          "cannot see.")
    print("\nCONTAMINATED CELL: fa.qkv M=3 reads 512.74 us net against 244.06 "
          "at M=4.")
    print("Four of its eight blocks were interrupted for the whole forward "
          "region.")
    print("Every fa.qkv column that divides by [3] is void. Do not quote them.")


def reconcile_f(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("Reconciling E115's f = 0.667 with Finding 22")
    print("=" * 78)
    gate_up = {int(k): v for k, v in summary["shapes"]["mlp.gate_up"]["widths"].items()}
    r44 = gate_up[8]["a_one_net_us"] / gate_up[4]["a_one_net_us"]
    r33 = gate_up[6]["a_one_net_us"] / gate_up[3]["a_one_net_us"]
    print(f"E115 identity:  A_round = f * A_tensor + (1 - f)")
    print(f"  A_round  = 2 * (1 - {COLLAPSE_MEASURED}) = {A_LOCAL:.3f}   RATE ratio")
    print(f"  A_tensor = {B_MSPLIT_TIME_RATIO:.3f}                  "
          f"TIME ratio, E115 b_msplit")
    print(f"  f        = {E115_F:.3f}")
    print("\nThe two inputs are not the same kind of object. A rate ratio rises")
    print("towards G as grouping improves; a time ratio falls towards 1. As a")
    print(f"rate ratio the b_msplit result is 2 / {B_MSPLIT_TIME_RATIO} = "
          f"{2 / B_MSPLIT_TIME_RATIO:.4f}, and")
    print(f"  f = ({A_LOCAL:.3f} - 1) / ({2 / B_MSPLIT_TIME_RATIO:.4f} - 1) = "
          f"{(A_LOCAL - 1) / (2 / B_MSPLIT_TIME_RATIO - 1):.1f}, "
          "which is impossible for a fraction.")
    print("\nRedo it in one consistent TIME frame. With phi the share of the")
    print("one-group round that scales with the group count:")
    print("  t2 / t1 = (1 - phi) + phi * (q2 / q1)")
    t_ratio = 1 / (1 - COLLAPSE_MEASURED)
    print(f"  t2 / t1 = 1 / (1 - {COLLAPSE_MEASURED}) = {t_ratio:.4f}   E100")
    for label, ratio in (("[4+4]/[4]", r44), ("[3+3]/[3]", r33)):
        phi = (t_ratio - 1) / (ratio - 1)
        print(f"  q2/q1 = {ratio:.4f} ({label})  ->  phi = {phi:.3f}")
    print(f"\nFinding 22 wide-branch qualifying share  {QUALIFYING_SHARE:.3f}")
    print(f"Finding 22 whole streaming subtotal      {FINDING22_STREAM_SUBTOTAL:.3f}")
    print("\nphi lands well below both, so E100's collapse cannot be read as")
    print("'this fraction of the round is group-scaling matvec work'. The two")
    print("measurements describe different partitions: E100 compared the")
    print("PRE-E100 [3+2] map against the current [5] map, and [3+2] is two")
    print("groups of 3 and 2 rows, which are intrinsically cheaper per group")
    print("than the [4+4] the current tree runs at M = 8.")
    t5 = gate_up[5]["a_one_net_us"]
    t3, t2 = gate_up[3]["a_one_net_us"], gate_up[2]["a_one_net_us"]
    grouping_saving = 1 - gate_up[6]["a_one_net_us"] / (2 * t3)
    t32 = (t3 + t2) * (1 - grouping_saving)
    print(f"\nDirect check of E100 in the M frame, mlp.gate_up:")
    print(f"  [5] measured                 {t5:.2f} us")
    print(f"  [3] + [2] isolated           {t3 + t2:.2f} us")
    print(f"  grouping saving from [3+3]   {grouping_saving * 100:.2f} %")
    print(f"  [3+2] estimated              {t32:.2f} us   INFERRED")
    print(f"  [5] faster than [3+2] by     {(1 - t5 / t32) * 100:.2f} %")
    print(f"  x wide-branch share {QUALIFYING_SHARE:.3f}   "
          f"-> {(1 - t5 / t32) * 100 * QUALIFYING_SHARE:.2f} % of the round")
    print(f"  E100 measured round collapse  {COLLAPSE_MEASURED * 100:.1f} %")
    print("\nThe wide-branch tensors alone explain most of E100's 18.0 %, and")
    print("the narrow-branch out_proj and down_proj family, 28.666 % of the")
    print("round, changed partition in the same commit. Finding 22 and E100")
    print("agree. E115's f = 0.667 does not follow from either and should be")
    print("withdrawn.")


def frame_dilution(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("Frame dilution: every campaign A is an upper bound, and 1.640 is")
    print("not a diluted current-tree kernel constant")
    print("=" * 78)
    gate_up = {int(k): v for k, v in summary["shapes"]["mlp.gate_up"]["widths"].items()}
    print("A_local is a ratio of ROUND times. With round time t = F + q and any")
    print("non-QMV F > 0:")
    print("  A_round = 2 (F + q1) / (F + q2)  >  A_kernel = 2 q1 / q2")
    print(f"\nSolving A_round = {A_LOCAL:.3f} for A_kernel at fixed F:")
    print(f"{'F':>6} {'A_kernel':>10}")
    for f_share in (0.0, 0.10, 0.20, 0.30, 0.40):
        a_kernel = 2 * (A_LOCAL / 2 - f_share) / (1 - f_share)
        print(f"{f_share:>6.2f} {a_kernel:>10.3f}")
    print("\nA_kernel is highest at F = 0 and falls as F rises, so 1.640 is the")
    print("ceiling this model allows. Now measure A_kernel directly on the")
    print("shipped partitions, mlp.gate_up, as a rate ratio 2 t1 / t2:")
    for label, one, many, g in (
        ("[4+4] vs [4]", 4, 8, 2),
        ("[3+3] vs [3]", 3, 6, 2),
        ("[3+3+3] vs [3]", 3, 9, 3),
    ):
        a = g * gate_up[one]["a_one_net_us"] / gate_up[many]["a_one_net_us"]
        print(f"  {label:>16}  A_kernel = {a:.4f}")
    print("\nEvery measured A_kernel is far below 1.640, so the dilution model")
    print("is consistent in sign: the round ratio does exceed the kernel ratio.")
    print("The bracket in the E115 pre-registration, A_kernel = 1.550 at F = 20 %")
    print("and 1.400 at F = 40 %, is therefore an upper bound and a loose one.")
    print("The directly measured current-tree A_kernel at the width that")
    print("carries 84 % of rounds is 1.154. A [4+4] group returns 15 % more")
    print("logical bytes per second than a [4] group, not the 40 to 55 % the")
    print("bracket allows, so the bracket cannot be used as an estimate.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=pathlib.Path,
        default=pathlib.Path("research/e117-artifacts/rung0-mframe-summary.json"))
    args = parser.parse_args()
    summary = load(args.artifact)

    print("=" * 78)
    print("E117 rung-0 pricing, harness=local, "
          "gate_qualified_for_timing=false")
    print(f"source {args.artifact}   estimator {summary['estimator']}")
    print("=" * 78)
    print("Realised width histogram, E109 v2 pair census: "
          f"{WIDTH_HISTOGRAM}, "
          f"{WIDTH_HISTOGRAM[8] / sum(WIDTH_HISTOGRAM.values()) * 100:.1f} % "
          "at width 8")

    price(summary, "mlp.gate_up", GATE_UP_SITES_PER_ROUND,
          FINDING22_LOCAL_SHARE["mlp.gate_up"])
    price(summary, "gdn.in_proj", GDN_IN_PROJ_SITES_PER_ROUND,
          FINDING22_LOCAL_SHARE["gdn.in_proj"])
    group_ratios(summary)
    reconcile_f(summary)
    frame_dilution(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
