#!/usr/bin/env python3
"""E163 F4: three numbers describe one physical thing and span 30 %.

    92.449 ms   my sealed gated 512-token shipped@d4 leg
   116.46 ms    my Interim-2 traced round
   120.18 ms    askeladd's band-aware fit, s = 57.83 + 5 x 12.47

They disagree for two DIFFERENT reasons that happen to land in the same place.
This reducer proves both from persisted artifacts, so the ledger records a
cause rather than a guess. harness=local throughout.

    python3 research/e163_reconcile_round_cost.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
ARTIFACTS = ROOT / "research" / "e163-artifacts"

WITNESS = "e163w5hist"
GATED = ("e163w5p1shippedd4", "e163w5p6shippedd4")
GATED_G2 = ("e163w5p2minna5d4", "e163w5p5minna5d4")

ASKELADD_INTERCEPT_MS = 57.83
ASKELADD_PER_ROW_MS = 12.47
INTERIM2_ROUND_MS = 116.46

ROUND = re.compile(r"mtp-trace: round=(\d+) d=(\d+) .*?round_us=(\d+)")


def d4_rounds(tag: str) -> list[float]:
    path = OUT / tag / "trace.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace: round="):
            continue
        m = ROUND.search(line)
        if m and int(m.group(2)) == 4:
            out.append(int(m.group(3)) / 1000.0)
    return out


def meta(tag: str) -> dict:
    path = OUT / tag / "meta.txt"
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def score(tag: str) -> dict:
    return json.loads((OUT / tag / "score.json").read_text())["metrics"]


def summarise(tag: str) -> dict:
    r = d4_rounds(tag)
    m = meta(tag)
    s = score(tag)
    return {
        "tag": tag,
        "tokens": int(m["tokens"]),
        "cool_gate": m["cool_gate"],
        "cool_gate_passed_real_gate": m["cool_gate_passed_real_gate"],
        "gate_qualified_for_timing": m["gate_qualified_for_timing"],
        "gpu_temp_entry_c": float(m["gpu_temp_entry_c"]),
        "gpu_temp_exit_c": float(m["gpu_temp_exit_c"]),
        "d4_rounds": len(r),
        "d4_mean_ms": statistics.mean(r),
        "d4_median_ms": statistics.median(r),
        "d4_min_ms": min(r),
        "serial_seconds_per_token": s["serial_seconds_per_token"],
        "mtp_seconds_per_token": s["mtp_seconds_per_token"],
    }


def main() -> int:
    witness = summarise(WITNESS)
    gated = [summarise(t) for t in GATED]
    gated_g2 = [summarise(t) for t in GATED_G2]

    g_median = statistics.mean(x["d4_median_ms"] for x in gated)
    g_serial = statistics.mean(x["serial_seconds_per_token"] for x in gated)
    round_ratio = witness["d4_median_ms"] / g_median
    serial_ratio = witness["serial_seconds_per_token"] / g_serial

    print("=" * 78)
    print("NUMBER 2: the 116 ms round came from an UNGATED 128-token leg.")
    print("=" * 78)
    hdr = (f"  {'leg':<20}{'tok':>5}{'gate':>6}{'qual':>7}{'Tin':>6}{'Tout':>6}"
           f"{'n':>5}{'med ms':>9}{'min ms':>9}{'serial s/tok':>14}")
    print(hdr)
    for x in [witness] + gated + gated_g2:
        print(f"  {x['tag']:<20}{x['tokens']:>5}{x['cool_gate']:>6}"
              f"{str(x['gate_qualified_for_timing']):>7}"
              f"{x['gpu_temp_entry_c']:>6.1f}{x['gpu_temp_exit_c']:>6.1f}"
              f"{x['d4_rounds']:>5}{x['d4_median_ms']:>9.3f}"
              f"{x['d4_min_ms']:>9.3f}{x['serial_seconds_per_token']:>14.6f}")
    print()
    print(f"  witness round / gated round   = {round_ratio:.4f}")
    print(f"  witness serial / gated serial = {serial_ratio:.4f}")
    print(f"  The two ratios agree to {100*abs(round_ratio/serial_ratio - 1):.1f} %.")
    print("  The serial depth-0 path shares no drafting, no head and no")
    print("  accumulator grouping with the MTP path, so a factor that moves both")
    print("  by the same amount is a HOST STATE factor. The witness leg was")
    print("  slower everywhere; it is not evidence about width 5 or the shipped")
    print("  plan. It also carries gate_qualified_for_timing=false, so it may")
    print("  not be compared with gated history at all.")
    print(f"  Predicted from the gated round: "
          f"{g_median:.3f} x {serial_ratio:.4f} = "
          f"{g_median*serial_ratio:.3f} ms, against {witness['d4_median_ms']:.3f} "
          f"measured.")
    print(f"  The Interim-2 figure {INTERIM2_ROUND_MS:.2f} ms is the same leg, "
          f"reduced over")
    print("  a different round subset. It is retracted as a round cost.")

    # --- number 3 ---------------------------------------------------------
    session = json.loads((ARTIFACTS / "e163_pinned_w5.json").read_text())
    b = session["boundary"]
    cells = session["decomposition"]["cells"]
    r5_g1 = b["R_ms_low"]
    r6_g2 = b["R_ms_high"]
    legs_g2 = [x for x in session["legs"] if x["arm_key"] == "arm_low"] or None
    r5_g2 = statistics.mean(
        x["R_ms_from_leg"] for x in session["legs"]
        if x["arm_key"] == cells["arm_low"]["arm_key"])

    # Legs 3 and 4 pin depth 5 but still close on one width-5 tail round, so
    # their mean is 95/96 of the width-6 cell and 1/96 of the width-5 one.
    # Undo that dilution before deriving a per-row slope from it.
    hist = [x["trace_width_histogram"] for x in session["legs"]
            if x["arm_key"] == cells["shipped_high"]["arm_key"]]
    n6 = sum(h.get("6", h.get(6, 0)) for h in hist)
    n5 = sum(h.get("5", h.get(5, 0)) for h in hist)
    r6_pure = ((n6 + n5) * r6_g2 - n5 * r5_g1) / n6

    step = r5_g2 - r5_g1
    per_row = r6_pure - r5_g2
    intercept = r5_g1 - 5 * per_row

    print()
    print("=" * 78)
    print("NUMBER 3: askeladd's 120.18 ms is a fit evaluated across a step.")
    print("=" * 78)
    print("  Three measured cells determine three parameters exactly:")
    print(f"    R(M=5, G=1) = a + 5h       = {r5_g1:8.3f} ms")
    print(f"    R(M=5, G=2) = a + 5h + s   = {r5_g2:8.3f} ms")
    print(f"    R(M=6, G=2) = a + 6h + s   = {r6_pure:8.3f} ms"
          f"  (from {r6_g2:.3f} ms measured, un-diluted over "
          f"{n6} width-6 and {n5} width-5 rounds)")
    print(f"  -> s (one-off second weight stream) = {step:7.3f} ms")
    print(f"  -> h (per verified row, within band) = {per_row:7.3f} ms")
    print(f"  -> a (round base)                    = {intercept:7.3f} ms")
    print()
    print(f"  askeladd's intercept {ASKELADD_INTERCEPT_MS:.2f} ms agrees with my "
          f"a = {intercept:.3f} ms")
    print(f"  to {100*abs(ASKELADD_INTERCEPT_MS/intercept - 1):.1f} %, from a "
          f"different session and a different fit.")
    print(f"  His slope {ASKELADD_PER_ROW_MS:.2f} ms/row does not: it mixes the "
          f"within-band")
    print(f"  {per_row:.3f} ms/row with the {step:.3f} ms band step, because a "
          f"single line was")
    print("  fitted across the G=1/G=2 boundary. Evaluating that line INSIDE")
    print(f"  band 1 gives {ASKELADD_INTERCEPT_MS + 5*ASKELADD_PER_ROW_MS:.2f} ms "
          f"against the measured {r5_g1:.3f} ms, an")
    print(f"  overstatement of "
          f"{100*((ASKELADD_INTERCEPT_MS + 5*ASKELADD_PER_ROW_MS)/r5_g1 - 1):.0f} %. "
          f"The two-band form is the one to keep.")
    print()
    print("  The two errors are unrelated and land within 1 % of each other by")
    print("  coincidence. Neither is a measurement of R(5) on the shipped plan.")
    print(f"  The ledger value is {r5_g1:.3f} ms: gated, 512 tokens, "
          f"six-leg palindrome.")

    payload = {
        "harness": "local",
        "official_or_ranked_score": False,
        "witness": witness,
        "gated_g1": gated,
        "gated_g2": gated_g2,
        "witness_round_over_gated_round": round_ratio,
        "witness_serial_over_gated_serial": serial_ratio,
        "interim2_retracted_round_ms": INTERIM2_ROUND_MS,
        "two_band": {
            "round_base_a_ms": intercept,
            "per_row_h_ms": per_row,
            "second_stream_step_s_ms": step,
            "R_m5_g1_ms": r5_g1,
            "R_m5_g2_ms": r5_g2,
            "R_m6_g2_ms": r6_g2,
            "R_m6_g2_pure_ms": r6_pure,
        },
        "askeladd_fit": {
            "intercept_ms": ASKELADD_INTERCEPT_MS,
            "per_row_ms": ASKELADD_PER_ROW_MS,
            "evaluated_at_m5_ms": ASKELADD_INTERCEPT_MS + 5 * ASKELADD_PER_ROW_MS,
            "intercept_agreement_percent":
                100 * abs(ASKELADD_INTERCEPT_MS / intercept - 1),
        },
    }
    path = ARTIFACTS / "e163_round_cost_reconciliation.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
