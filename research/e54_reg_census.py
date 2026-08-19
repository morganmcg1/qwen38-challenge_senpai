#!/usr/bin/env python3
"""E54 Law D discriminator: kernel-wide register footprint of every E54 arm. No GPU.

The advisor's Law D says an NA=5 group carries the register footprint of NA=8
(`sizeof(vec<float,5>) == 32`), that the allocation is shared by every width
because the file has one `[[kernel]]` and inline `METAL_FUNC` helpers, and that
this is why E27 won every isolated cell and still lost 0.3321 % of board score.

Two readouts, and only one of them is a Law D instrument:

  REAL TABLE (`shipped` vs `e27_full`) is the Law D instrument. Both arms keep
  all seven width cases, so `kernel_wide_reg_max` is the max over the same seven
  cases the campaign's 108 was measured over, and the entry readout inlines all
  of them. E27's revert recorded 129 -> 108 and 183 -> 163; this reproduces or
  refutes that from source.

  ISOLATED arms (`iso_*`) delete every case but one, so their max is over a
  single width and is NOT comparable to 108. They answer a different question:
  what does the treated cell itself cost. Reported, never compared to the
  ceiling.

Arms come from `research/e54_arms.py`, so the census measures exactly the source
that was timed rather than a re-derivation of it.

  python3 research/e54_reg_census.py --out research/e54-artifacts/e54-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e46_reg_census import (  # noqa: E402
    CEILING,
    CELL,
    ENTRIES,
    HEADER,
    HEADER_REL,
    WIDTHS,
    compile_probe,
    ipg_table,
    na_cells,
    streams,
)
from e54_arms import ARMS, apply_arm  # noqa: E402

REPO = HERE.parent
# The real-table pair is the only Law D instrument; isolated arms are cells.
REAL_TABLE = ("shipped", "e27_full")
PAIRS = [
    ("P1", "iso_m5_ipg3", "iso_m5_ipg5"),
    ("P2", "iso_m7_ipg4", "iso_m7_ipg5"),
    ("P3", "iso_m8_ipg4", "iso_m8_ipg5"),
    ("P4", "shipped", "e27_full"),
]
# E27's revert acknowledgement, quoted in the E54 brief.
E27_RECORDED = {"kernel_wide_reg_max": 129, "entry_point_reg_max": 183}
TIP_RECORDED = {"kernel_wide_reg_max": 108, "entry_point_reg_max": 163}


def census_arm(name: str, workdir: pathlib.Path) -> dict:
    shadow = workdir / name
    header_dst = shadow / HEADER_REL
    header_dst.parent.mkdir(parents=True, exist_ok=True)
    text = apply_arm(HEADER.read_text(), name)
    header_dst.write_text(text)

    table = ipg_table(text)
    present = [m for m in WIDTHS if table[m]]
    out = {
        "arm": name,
        "family": ARMS[name]["family"],
        "cell": ARMS[name]["cell"],
        "ipg_table": table,
        "streams": streams(table),
        "present_widths": present,
        "real_table": len(present) == len(WIDTHS),
    }

    cells = {}
    for m in present:
        res = compile_probe(shadow, "cell_m%d" % m,
                            {"E46_CELL_M": m, "E46_CELL_IPG": table[m]}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        cells[m] = dict(res["functions"][CELL], ipg=table[m],
                        na_cells=na_cells(m, table[m]))

    entry = compile_probe(shadow, "entry", {}, ENTRIES)
    if entry["status"] != "ok":
        return dict(out, status="entry_%s" % entry["status"],
                    error=entry.get("error"))

    width_max = max(c["peak_live_regs"] for c in cells.values())
    return dict(
        out, status="ok", width_cells=cells, entry=entry["functions"],
        kernel_wide_reg_max=width_max,
        argmax_width=max(cells, key=lambda m: cells[m]["peak_live_regs"]),
        entry_point_reg_max=max(f["peak_live_regs"]
                                for f in entry["functions"].values()),
        # Only meaningful when every width case is present.
        exceeds_ceiling=(width_max > CEILING) if len(present) == len(WIDTHS)
        else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="research/e54-artifacts/e54-reg-census.json")
    ap.add_argument("--arms", default=",".join(
        [a for a in ARMS if not ARMS[a].get("never_time")]))
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        results = {n: census_arm(n, workdir) for n in names}

    bad = {n: r["status"] for n, r in results.items() if r["status"] != "ok"}
    if bad:
        print("CENSUS FAILED: %s" % bad)
        for n in bad:
            print("  %s: %s" % (n, results[n].get("error")))
        return 1

    print("E54 REGISTER CENSUS  (ceiling %d, tip entry %d)"
          % (CEILING, TIP_RECORDED["entry_point_reg_max"]))
    print("  %-22s %-14s %6s %6s  %s"
          % ("arm", "table", "kmax", "entry", "per-width regs M=3..9"))
    for n in names:
        r = results[n]
        per = " ".join("%d:%s" % (m, r["width_cells"][m]["peak_live_regs"]
                                  if m in r["width_cells"] else "-")
                       for m in WIDTHS)
        print("  %-22s %-14s %6d %6d  %s"
              % (n, "REAL" if r["real_table"] else "isolated",
                 r["kernel_wide_reg_max"], r["entry_point_reg_max"], per))

    print("\nLAW D INSTRUMENT  (real table only)")
    base, comp = (results[k] for k in REAL_TABLE)
    d_kmax = comp["kernel_wide_reg_max"] - base["kernel_wide_reg_max"]
    d_entry = comp["entry_point_reg_max"] - base["entry_point_reg_max"]
    print("  shipped   kmax %3d  entry %3d   (campaign records %d / %d)"
          % (base["kernel_wide_reg_max"], base["entry_point_reg_max"],
             TIP_RECORDED["kernel_wide_reg_max"],
             TIP_RECORDED["entry_point_reg_max"]))
    print("  e27_full  kmax %3d  entry %3d   (E27 revert records %d / %d)"
          % (comp["kernel_wide_reg_max"], comp["entry_point_reg_max"],
             E27_RECORDED["kernel_wide_reg_max"],
             E27_RECORDED["entry_point_reg_max"]))
    print("  delta     kmax %+d  entry %+d" % (d_kmax, d_entry))

    reproduces = (base["kernel_wide_reg_max"] == TIP_RECORDED["kernel_wide_reg_max"]
                  and comp["kernel_wide_reg_max"] == E27_RECORDED["kernel_wide_reg_max"])
    law_d_active = d_kmax > 0
    print("\n  reproduces_recorded_129_to_108 = %s" % reproduces)
    print("  LAW D VERDICT: %s"
          % ("SUPPORTED -- the real table's shared register max RISES when the "
             "NA=5 cells are added" if law_d_active else
             "NOT SUPPORTED -- the real table's shared register max does not rise"))

    print("\nPER-PAIR CELL FOOTPRINT")
    for tag, a, b in PAIRS:
        if a not in results or b not in results:
            continue
        ra, rb = results[a], results[b]
        print("  %-3s %-14s kmax %3d entry %3d  ->  %-14s kmax %3d entry %3d"
              "   (%+d / %+d)"
              % (tag, a, ra["kernel_wide_reg_max"], ra["entry_point_reg_max"],
                 b, rb["kernel_wide_reg_max"], rb["entry_point_reg_max"],
                 rb["kernel_wide_reg_max"] - ra["kernel_wide_reg_max"],
                 rb["entry_point_reg_max"] - ra["entry_point_reg_max"]))

    payload = {
        "ceiling": CEILING,
        "recorded_tip": TIP_RECORDED,
        "recorded_e27": E27_RECORDED,
        "arms": results,
        "law_d": {
            "instrument": "real-table shipped vs e27_full",
            "delta_kernel_wide_reg_max": d_kmax,
            "delta_entry_point_reg_max": d_entry,
            "reproduces_recorded_129_to_108": reproduces,
            "supported": law_d_active,
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
