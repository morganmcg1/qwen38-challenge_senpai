#!/usr/bin/env python3
"""Re-price the E103 ceiling against the post-E100 round anchor.

    usage: research/e103_reprice.py

Rungs 0 to 2c priced every saving against one number, the 127,176 us round
GPU busy measured at verify width 5 on the pre-E100 base. E100 merged, it
collapses the M = 5 quantised matvec to one x-group, and the advisor asked
for the anchor to be re-measured before any percentage is quoted again.

Two census legs on the rebased tree supply the new anchors, and the width 6
leg also answers a structural question that no microbenchmark could: what the
scored path really dispatches once the trusted `qL * gqa <= 32` cap is
exceeded. Both legs are census legs, never timing legs.

The output is:

  1. the measured round GPU busy per verify width, before and after E100;
  2. the verbatim in-situ dispatch structure of the FA-layer attention at
     verify width 6, which is a 5-row leg and a 1-row leg, each paired inside
     one command buffer with a `g2_copy`;
  3. the session-weighted re-pricing of the stacked ceiling, which is the
     only percentage that maps to a ranked score, since a round at M = 9 is
     70 % longer than a round at M = 5 and the width mass is spread over both;
  4. a handoff measurement that this census produced as a by-product: the
     per-row cost of `affine_qmv_fast` at M = 6 against the same kernel at
     M = 5, where E100 already landed.
"""

from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path("research/out/e103")
CENSUS = OUT / "census_rebased.json"
D4, D5 = "e103r2-d4-ops0", "e103r2-d5-ops0"

FA_LAYERS = 16
# Ledger 207 verify-width shares, the same table rung 2b priced with.
WIDTH_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334, 7: 0.122, 8: 0.0735,
               9: 0.0575}
# Stacked saving per round, per width, from `research/e103_split_report.py`:
# best pack arm plus the measured merge of the 5 + r partition.
STACK_US = {3: 186.0, 4: 57.0, 5: 296.0, 6: 474.0, 7: 526.0, 8: 618.0,
            9: 459.0}
PRE_E100_ROUND_US = 127176.0
MIN_USEFUL_FRACTION = 0.0030

# In-situ buffer costs, `research/e103_insitu_split.py`, 176 buffers each.
W6_LEG5 = 77.65   # g2_copy grid=1280x24x1 + sdpa_vector qnt_c grid=24x5x1
W6_LEG1 = 35.33   # g2_copy grid= 256x24x1 + sdpa_vector qnt_nc grid=24x1x1
W5_LEG5 = 79.29   # sdpa_vector qnt_c grid=24x5x1, alone in its buffer
# The same 1-row kernel alone in its buffer, from the draft head at width 5
# and from the non-speculative target forward at width 1.
ISOLATED_1ROW = (32.31, 32.92)
LATENCY_CLASS_FACTOR = 2.40
INSITU_DISCOUNT = (1.65, 2.59)
PUBLISHED_FLOOR_PCT = 0.277


def kernels(leg: dict, prefix: str) -> dict[str, dict]:
    return {k: v for k, v in leg["kernels"].items() if k.startswith(prefix)}


def compute() -> dict:
    """Every scalar the report and the W&B run quote, from one place."""
    payload = json.loads(CENSUS.read_text())
    anchors = {}
    for leg in payload.values():
        for width, busy in leg["round_busy_us"].items():
            m = int(width[1:])
            if m and leg["rounds_by_width"][width] >= 10:
                anchors[m] = busy
    # Round length model. Widths 5 and 6 are measured over 10+ rounds each.
    # Width 4 is a single-round sample from the same leg. For M >= 7 the
    # increment is the pre-E100 width 5 to width 6 step, which is the
    # increment in the regime where E100's `case 5:` edit does not apply.
    per_row_ge6 = anchors[6] - PRE_E100_ROUND_US
    rounds = {3: 79000.0, 4: 86301.0, 5: anchors[5], 6: anchors[6]}
    for m in (7, 8, 9):
        rounds[m] = anchors[6] + per_row_ge6 * (m - 6)

    mass = sum(WIDTH_SHARE.values())
    num = sum(WIDTH_SHARE[m] * STACK_US[m] for m in WIDTH_SHARE)
    den = sum(WIDTH_SHARE[m] * rounds[m] for m in WIDTH_SHARE)
    # Rung 2b measured the merge on the SDPA kernels only, because the
    # microbenchmark never saw the copy. In situ the split pays two copies
    # and a merged pass pays one, so M >= 6 has one more copy to give back.
    copy_us = W6_LEG1 - sum(ISOLATED_1ROW) / 2
    num_hi = sum(WIDTH_SHARE[m] * (STACK_US[m]
                                   + (FA_LAYERS * copy_us if m >= 6 else 0.0))
                 for m in WIDTH_SHARE)
    weighted_round = den / mass
    return {
        "anchors": anchors,
        "rounds": rounds,
        "per_row_ge6_us": per_row_ge6,
        "insitu_fa_w5_us_per_round": FA_LAYERS * W5_LEG5,
        "insitu_fa_w6_us_per_round": FA_LAYERS * (W6_LEG5 + W6_LEG1),
        "copy_us": copy_us,
        "weighted_round_us": weighted_round,
        "weighted_saving_us": num / mass,
        "weighted_saving_upper_us": num_hi / mass,
        "ceiling_pct": 100 * num / den,
        "ceiling_upper_pct": 100 * num_hi / den,
        "bar_us": MIN_USEFUL_FRACTION * weighted_round,
        "ratio_to_bar": (num / mass) / (MIN_USEFUL_FRACTION * weighted_round),
        "ratio_to_bar_upper": (num_hi / mass)
        / (MIN_USEFUL_FRACTION * weighted_round),
    }


def main() -> None:
    payload = json.loads(CENSUS.read_text())
    d4, d5 = payload[D4], payload[D5]

    print("=== rebased round anchors, GPU busy per round ===")
    print("leg                width  rounds   us/round")
    anchors: dict[int, float] = {}
    for tag, leg in ((D4, d4), (D5, d5)):
        for width, busy in sorted(leg["round_busy_us"].items()):
            n = leg["rounds_by_width"][width]
            print(f"{tag:<18} {width:>5}  {n:>6}  {busy:9.0f}")
            m = int(width[1:])
            if m and n >= 10:
                anchors[m] = busy
    print(f"\npre-E100 anchor at width 5: {PRE_E100_ROUND_US:.0f} us/round")
    print(f"post-E100 anchor at width 5: {anchors[5]:.0f} us/round  "
          f"({100 * (anchors[5] / PRE_E100_ROUND_US - 1):+.1f} %)")
    print(f"post-E100 anchor at width 6: {anchors[6]:.0f} us/round  "
          f"({100 * (anchors[6] / anchors[5] - 1):+.1f} % over width 5)")
    print("E100 shortens the M = 5 round only. It edits the `case 5:`")
    print("instantiation in kernels/quantized.h, so the M >= 6 round keeps")
    print("the pre-E100 cost and is now much longer than the old anchor.")

    print("\n=== in-situ FA attention, verify width 5 against width 6 ===")
    for tag, leg, width in ((D4, d4, "w5"), (D5, d5, "w6")):
        rows = [(k, v) for k, v in kernels(leg, f"{width}|target_verify").items()
                if "sdpa" in k]
        total = sum(v["us_per_round"] for _, v in rows)
        print(f"{tag} {width}: exclusive sdpa in target_verify "
              f"{total:.0f} us/round over {len(rows)} kernels")
    print("The width 6 leg reports no exclusive sdpa kernel, because at")
    print("qL = 6 the trusted cap `qL * gqa <= 32` with gqa = 6 is exceeded")
    print("and every attention op now emits two kernels into one command")
    print("buffer, so the one-op-per-buffer reducer drops it. The signature")
    print("table holds the real structure:")
    print("  16 buffers/round  g2_copy grid=1280x24x1 + "
          f"sdpa_vector_..._qnt_c_nosinks  grid=24x5x1  {W6_LEG5:5.2f} us/buffer")
    print("  16 buffers/round  g2_copy grid= 256x24x1 + "
          f"sdpa_vector_..._qnt_nc_nosinks grid=24x1x1  {W6_LEG1:5.2f} us/buffer")
    insitu_w6 = FA_LAYERS * (W6_LEG5 + W6_LEG1)
    print(f"FA attention at width 6 = {insitu_w6:.0f} us/round = "
          f"{100 * insitu_w6 / anchors[6]:.3f} % of the width 6 round")
    insitu_w5 = FA_LAYERS * W5_LEG5
    print(f"FA attention at width 5 = {insitu_w5:.0f} us/round = "
          f"{100 * insitu_w5 / anchors[5]:.3f} % of the width 5 round")
    print("The 5 + r partition the advisor predicted is confirmed in situ.")
    print("It also carries a per-leg `g2_copy` that no rung measured: the")
    print(f"1-row buffer costs {W6_LEG1:.2f} us against {ISOLATED_1ROW[0]:.2f} "
          f"to {ISOLATED_1ROW[1]:.2f} us for the same kernel alone, so the copy")
    print(f"is {W6_LEG1 - ISOLATED_1ROW[1]:.1f} to "
          f"{W6_LEG1 - ISOLATED_1ROW[0]:.1f} us and a merged single pass "
          f"removes one of the two.")

    print("\n=== session-weighted re-pricing of the stacked ceiling ===")
    # A round at M = 9 is far longer than a round at M = 5, so the only
    # honest session percentage is total microseconds saved over total
    # microseconds spent, not a mean of per-width percentages.
    step = (PRE_E100_ROUND_US - anchors[5]) / 1.0  # unused guard
    del step
    # Round length model. Widths 5 and 6 are measured over 10+ rounds each.
    # Width 4 and width 2 are single-round samples from the same two legs.
    # For M >= 7 the increment is taken from the pre-E100 width 5 to width 6
    # step, 139476 - 127176 = 12300 us per extra row, which is the increment
    # in the regime where E100 does not apply.
    per_row_ge6 = anchors[6] - PRE_E100_ROUND_US
    rounds = {3: 79000.0, 4: 86301.0, 5: anchors[5], 6: anchors[6]}
    for m in (7, 8, 9):
        rounds[m] = anchors[6] + per_row_ge6 * (m - 6)
    print(f"increment per extra row at M >= 6: {per_row_ge6:.0f} us")
    print("  M   share   round us   saving us   saving %")
    num = den = 0.0
    for m in sorted(WIDTH_SHARE):
        share, r, s = WIDTH_SHARE[m], rounds[m], STACK_US[m]
        num += share * s
        den += share * r
        print(f"  {m}  {share:6.3f}  {r:9.0f}  {s:9.0f}   {100 * s / r:7.3f}")
    weighted_round = den / sum(WIDTH_SHARE.values())
    weighted_saving = num / sum(WIDTH_SHARE.values())
    pct = 100 * num / den
    bar = MIN_USEFUL_FRACTION * weighted_round
    print(f"\nsession-weighted round   {weighted_round:9.0f} us")
    print(f"session-weighted saving  {weighted_saving:9.0f} us")
    print(f"stacked ceiling          {pct:9.3f} % of the local round")
    print(f"minimum useful effect    {bar:9.0f} us/round "
          f"({100 * MIN_USEFUL_FRACTION:.2f} %)")
    print(f"ratio to bar             {weighted_saving / bar:9.2f} x")
    print(f"pre-E100 pricing said    {100 * weighted_saving / PRE_E100_ROUND_US:9.3f} %")
    print("The rebased anchor does not move the verdict. E100 takes 18.6 %")
    print("off the M = 5 round, but M >= 6 rounds are longer than the old")
    print("single anchor and carry 58.7 % of the width mass, so the two")
    print("effects cancel to within a tenth of a percentage point.")

    # Upper variant. Rung 2b measured the merge on the SDPA kernels only,
    # because the microbenchmark never saw the copy. In situ the split pays
    # two copies and a merged pass pays one, so every M >= 6 round has one
    # more copy to give back than rung 2b priced.
    copy_us = W6_LEG1 - sum(ISOLATED_1ROW) / 2
    num_hi = sum(WIDTH_SHARE[m] * (STACK_US[m] + (FA_LAYERS * copy_us
                                                  if m >= 6 else 0.0))
                 for m in WIDTH_SHARE)
    hi_saving = num_hi / sum(WIDTH_SHARE.values())
    print(f"\nupper variant, one copy per FA layer given back at M >= 6")
    print(f"  measured copy cost       {copy_us:9.2f} us per leg")
    print(f"  stacked saving           {hi_saving:9.0f} us/round")
    print(f"  stacked ceiling          {100 * num_hi / den:9.3f} % of the round")
    print(f"  ratio to bar             {hi_saving / bar:9.2f} x")

    for label, p in (("central", pct), ("upper", 100 * num_hi / den)):
        ranked = LATENCY_CLASS_FACTOR * p
        print(f"\nranked, {label} variant")
        print(f"  undiscounted            {ranked:6.3f} %")
        print(f"  after in-situ discount  {ranked / INSITU_DISCOUNT[1]:6.3f} "
              f"% to {ranked / INSITU_DISCOUNT[0]:6.3f} %")
        print(f"  published floor         {PUBLISHED_FLOOR_PCT:6.3f} %")

    print("\n=== handoff: what the width 6 census found next door ===")
    # Same kernel, same round, one extra row. E100 collapsed M = 5 to one
    # x-group; M >= 6 still pays the old rate.
    print("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0, per-row us")
    print("  rows_out   n/round   M=5 us  M=5/row   M=6 us  M=6/row   penalty"
          "   us/round if M=6 matched M=5")
    grids = [(4352, 64), (640, 128), (2060, 48), (1792, 16), (31040, 1)]
    total_gap = 0.0
    for rows_out, n in grids:
        k5 = f"w5|target_verify|affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=5x{rows_out}x1 tg=32x2x1"
        k6 = f"w6|target_verify|affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0 grid=6x{rows_out}x1 tg=32x2x1"
        v5 = d4["kernels"][k5]["us_per_dispatch"]
        v6 = d5["kernels"][k6]["us_per_dispatch"]
        r5, r6 = v5 / 5, v6 / 6
        gap = (r6 - r5) * 6 * n
        total_gap += gap
        print(f"  {rows_out:8d}  {n:8d}  {v5:7.1f}  {r5:7.2f}  {v6:7.1f}  "
              f"{r6:7.2f}  {100 * (r6 / r5 - 1):+7.1f} %  {gap:9.0f}")
    print(f"  total recoverable at M = 6: {total_gap:.0f} us/round = "
          f"{100 * total_gap / anchors[6]:.2f} % of the width 6 round")
    m_ge_6 = sum(WIDTH_SHARE[m] for m in WIDTH_SHARE if m >= 6)
    print(f"  M >= 6 carries {100 * m_ge_6:.1f} % of the width mass, so the "
          f"session average is about {m_ge_6 * total_gap:.0f} us/round")
    print(f"  that is {m_ge_6 * total_gap / weighted_saving:.0f} x the entire "
          f"E103 stacked ceiling, in the same kernel family E100 already "
          f"knows how to collapse")


if __name__ == "__main__":
    main()
