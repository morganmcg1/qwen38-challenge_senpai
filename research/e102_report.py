#!/usr/bin/env python3
"""E102: print the reconciliation tables from the rung-1 census JSON.

    python3 research/e102_report.py research/out/e102/regs.json
"""

from __future__ import annotations

import json
import sys

G16, G17 = "applegpu_g16s", "applegpu_g17s"

# E77-corrected occupancy law, quoted from the E76 retraction feedback:
# S_ranked(R) = floor(496 KiB / (128 R)); Omega(S) = (32 / S) ** gamma.
GAMMA = 0.01346
BUDGET = 496 * 1024


def occupancy(registers: int) -> tuple[int, float]:
    s = BUDGET // (128 * registers)
    return s, (32.0 / s) ** GAMMA


def main() -> int:
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1
                          else "research/out/e102/regs.json"))
    raw = data.get("cells_pipeline_raw") or ""
    if raw:
        print(raw.splitlines()[0])

    print("\n== dispatcher scope: the shipped affine_qmv_fast entry point ==")
    print(f"{'arm':<20}{'AIRpk':>6}{'AIRln':>7}"
          f"{'g16sR':>7}{'g16sSp':>7}{'g16sB':>7}"
          f"{'g17sR':>7}{'g17sSp':>7}{'g17sB':>7}"
          f"{'maxTPTG':>8}{'execW':>6}{'tgmem':>6}")
    base = None
    for name, rec in data["arms"].items():
        air = rec["air_entry_scope"]
        pipe = (rec["pipeline_local"].get("per_entry") or {}).get(
            "affine_qmv_fast_bfloat16_t_64_4_false", {})
        row = (f"{name:<20}{air['peak_live_regs']:>6}{air['air_lines']:>7}"
               f"{rec[G16]['registers']:>7}{rec[G16]['spill_bytes']:>7}"
               f"{rec[G16]['text_bytes']:>7}"
               f"{rec[G17]['registers']:>7}{rec[G17]['spill_bytes']:>7}"
               f"{rec[G17]['text_bytes']:>7}"
               f"{pipe.get('max_total_threads_per_threadgroup', '-'):>8}"
               f"{pipe.get('thread_execution_width', '-'):>6}"
               f"{pipe.get('static_threadgroup_memory_bytes', '-'):>6}")
        print(row)
        if base is None:
            base = rec

    print("\n== bare-body scope: the E76 and E97 instrument ==")
    print(f"{'cell':<16}{'NA':>3}{'acc':>4}{'AIRpk':>6}"
          f"{'g16sR':>7}{'g16sSp':>7}{'g16sB':>7}"
          f"{'g17sR':>7}{'g17sSp':>7}{'g17sB':>7}")
    for name, rec in data["cells"].items():
        air = rec["air_cell_scope"]
        print(f"{name:<16}{rec['na']:>3}{rec['accumulators']:>4}"
              f"{air['peak_live_regs']:>6}"
              f"{rec[G16]['registers']:>7}{rec[G16]['spill_bytes']:>7}"
              f"{rec[G16]['text_bytes']:>7}"
              f"{rec[G17]['registers']:>7}{rec[G17]['spill_bytes']:>7}"
              f"{rec[G17]['text_bytes']:>7}")

    print("\n== does the dispatcher equal the widest body it inlines? ==")
    # Arms that inline only the shipped vec<float, NA> `_wide` family. The
    # `_wideN` arms J and N use float[4][NA] accumulators, which this
    # bare-body census does not instantiate, so they have no comparison cell.
    widest = {"A_shipped": 4, "B_ca9251b8": 5, "C_m5_only": 5, "D_fact2b": 6,
              "E_dead_m9_body": 4, "F_dead_m9_case": 4, "G_prune_both_m9": 4,
              "H_prune_narrow": 4, "I_prune_all_dead": 4,
              "K_ctrlA_59b321ee": 4, "L_ca9251b8_real": 5}
    for name, rec in data["arms"].items():
        if name not in widest:
            continue
        cell = data["cells"].get(f"e102_cell_na{widest[name]}")
        if not cell:
            continue
        same = all(rec[a]["registers"] == cell[a]["registers"]
                   for a in (G16, G17))
        print(f"{name:<20} widest NA={widest[name]}  "
              f"dispatcher {rec[G16]['registers']}/{rec[G17]['registers']}  "
              f"body {cell[G16]['registers']}/{cell[G17]['registers']}  "
              f"{'MATCH' if same else 'DIFFER'}")

    print("\n== E77-corrected occupancy price of the g17s step, harness=ranked model ==")
    a17 = data["arms"]["A_shipped"][G17]["registers"]
    s_a, w_a = occupancy(a17)
    for name, rec in data["arms"].items():
        r = rec[G17]["registers"]
        s, w = occupancy(r)
        print(f"{name:<20} g17s R={r:>4}  S={s:>3}  "
              f"Omega ratio vs A = {w / w_a:+.6f} -> {100 * (w / w_a - 1):+.4f} %")
    print(f"(A: R={a17}, S={s_a})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
