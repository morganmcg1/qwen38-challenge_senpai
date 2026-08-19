#!/usr/bin/env python3
"""E59 rung 1: does `<T,5,5>` at rows_per_simd = 2 fit under the 108 floor? No GPU.

This is the hard gate. If the kernel maximum of a real-table arm exceeds 108,
that arm cannot ship and the QMV width-table direction closes with it.

Three published register laws are settled here, and they disagree:

  r=4, askeladd's E55 law   20 + 21*max(NA) + 4*[two distinct NA group sizes]
  r=2, ledger 183(C)        16 + 15*max(NA)   -> 91 at NA=5
  r=2, E44 anchors          15 + 17*max(NA)   -> 100 at NA=5

Both r=2 fits sit under the floor, so the route survives either way, but only a
measurement says which fit is right. The loser gets retracted.

Every width case is censused, not only M=5, because the widths share one
`[[kernel]]` allocation: a route that halves M=5 and perturbs M=9 has not
bought anything. The per-width wrapper and IPG are read from the patched
dispatch table itself, so the census measures the cell the table really reaches.

`peak_live_values` and `allocas` are reported for every cell, because that pair
is what settled the mixed-versus-uniform `+4` term. A row-block cell holds two
sequential accumulator lifetimes and might or might not allocate twice.

  python3 research/e59_reg_census.py --out research/e59-artifacts/e59-reg-census.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e46_reg_census import (  # noqa: E402
    CEILING,
    ENTRIES,
    HEADER,
    HEADER_REL,
    INCLUDE,
    air_stats,
    na_cells,
)
from e59_arms import ARMS, apply_arm, routing_table  # noqa: E402

REPO = HERE.parent
CELL_PROBE = REPO / "research/e59_cell_probe.metal"
ENTRY_PROBE = REPO / "research/e46_entry_probe.metal"
CELL = "e59_cell"
# The scored session verifies at most one primary token plus eight drafts.
MAX_REACHABLE_WIDTH = 9


def predict(na_values: list[int], rows_per_simd: int) -> dict:
    top = max(na_values)
    mixed = len(set(na_values)) > 1
    if rows_per_simd == 4:
        return {"e55_law_r4": 20 + 21 * top + (4 if mixed else 0)}
    return {"ledger_183c_r2": 16 + 15 * top, "e44_anchors_r2": 15 + 17 * top}


def compile_probe(shadow: pathlib.Path, tag: str, probe: pathlib.Path,
                  defines: dict[str, str], wanted: tuple[str, ...]) -> dict:
    ll = shadow / ("%s.ll" % tag)
    ll_o3 = shadow / ("%s.o3.ll" % tag)
    flags = ["-D%s=%s" % (k, v) for k, v in defines.items()]
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", *flags,
         "-I", str(shadow), "-I", str(INCLUDE), "-S", str(probe), "-o", str(ll)],
        capture_output=True, text=True)
    if emit.returncode != 0:
        return {"status": "compile_failed",
                "error": emit.stderr.strip().splitlines()[-8:]}
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)],
        capture_output=True, text=True)
    if opt.returncode != 0:
        return {"status": "metal_opt_failed",
                "error": opt.stderr.strip().splitlines()[-8:]}

    lines = ll_o3.read_text().splitlines()
    found = {}
    for name in wanted:
        body, inside = [], False
        for line in lines:
            if line.startswith("define ") and ("@%s(" % name) in line:
                inside = True
            elif inside and line == "}":
                inside = False
            elif inside:
                body.append(line)
        if not body:
            return {"status": "kernel_not_found", "missing": name}
        found[name] = air_stats(body)
    return {"status": "ok", "functions": found}


def census_arm(name: str, workdir: pathlib.Path) -> dict:
    shadow = workdir / name
    header_dst = shadow / HEADER_REL
    header_dst.parent.mkdir(parents=True, exist_ok=True)
    text = apply_arm(HEADER.read_text(), name)
    header_dst.write_text(text)

    table = routing_table(text)
    out = {
        "arm": name,
        "family": ARMS[name]["family"],
        "cell": ARMS[name]["cell"],
        "routing": {str(m): v for m, v in sorted(table.items())},
    }

    cells = {}
    for m, route in sorted(table.items()):
        if "_m" not in route["wrapper"]:
            continue  # M=2 keeps the promoted pair kernel, which takes no IPG
        res = compile_probe(
            shadow, "cell_m%d" % m, CELL_PROBE,
            {"E59_CELL_M": str(m), "E59_CELL_IPG": str(route["ipg"]),
             "E59_CELL_WRAPPER": route["wrapper"]}, (CELL,))
        if res["status"] != "ok":
            return dict(out, status="cell_%d_%s" % (m, res["status"]),
                        error=res.get("error"))
        na = na_cells(m, route["ipg"])
        cells[m] = dict(res["functions"][CELL], **route, na_cells=na,
                        uniform=len(set(na)) == 1,
                        predicted=predict(na, route["rows_per_simd"]))

    entry = compile_probe(shadow, "entry", ENTRY_PROBE, {}, ENTRIES)
    if entry["status"] != "ok":
        return dict(out, status="entry_%s" % entry["status"],
                    error=entry.get("error"))

    reachable = {m: c for m, c in cells.items() if m <= MAX_REACHABLE_WIDTH}
    width_max = max(c["peak_live_regs"] for c in cells.values())
    return dict(
        out, status="ok", width_cells=cells, entry=entry["functions"],
        kernel_wide_reg_max=width_max,
        argmax_width=max(cells, key=lambda m: cells[m]["peak_live_regs"]),
        reachable_reg_max=max(c["peak_live_regs"] for c in reachable.values()),
        entry_point_reg_max=max(f["peak_live_regs"]
                                for f in entry["functions"].values()),
        exceeds_ceiling=width_max > CEILING,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e59-artifacts/e59-reg-census.json")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    with tempfile.TemporaryDirectory(prefix="e59-reg-census-") as tmp:
        results = {n: census_arm(n, pathlib.Path(tmp)) for n in names}

    bad = {n: r["status"] for n, r in results.items() if r["status"] != "ok"}
    if bad:
        print("CENSUS FAILED: %s" % bad)
        for n in bad:
            print("  %s: %s" % (n, results[n].get("error")))
        return 1

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    widths = sorted({m for r in results.values() for m in r["width_cells"]})
    print("E59 REGISTER CENSUS   head=%s   legality floor=%d" % (head[:8], CEILING))
    print("  %-22s %6s %6s %6s   %s"
          % ("arm", "kmax", "reach", "entry",
             " ".join("M%d" % m for m in widths)))
    for n in names:
        r = results[n]
        per = " ".join(
            ("%3d" % r["width_cells"][m]["peak_live_regs"]) if m in r["width_cells"]
            else "  -" for m in widths)
        flag = "  OVER FLOOR" if r["exceeds_ceiling"] else ""
        print("  %-22s %6d %6d %6d   %s%s"
              % (n, r["kernel_wide_reg_max"], r["reachable_reg_max"],
                 r["entry_point_reg_max"], per, flag))

    print("\nM=5 CELL DETAIL   (the cell under test)")
    print("  %-22s %-26s %4s %5s %5s %6s %7s  %s"
          % ("arm", "wrapper", "rps", "regs", "vals", "alloca", "acc", "law"))
    for n in names:
        cell = results[n]["width_cells"].get(5)
        if not cell:
            continue
        laws = " ".join("%s=%d" % (k.split("_")[0], v)
                        for k, v in sorted(cell["predicted"].items()))
        print("  %-22s %-26s %4d %5d %5d %6d %7s  %s"
              % (n, cell["wrapper"].replace("qmv_fast_crossrow_affine4_g64", "..."),
                 cell["rows_per_simd"], cell["peak_live_regs"],
                 cell["peak_live_values"], cell["allocas"],
                 ",".join(cell["acc_alloca_types"]) or "-", laws))

    print("\nUNTREATED WIDTHS vs `shipped`   (a route must not perturb them)")
    if "shipped" in results:
        base = results["shipped"]["width_cells"]
        for n in names:
            if n == "shipped":
                continue
            moved = {m: (base[m]["peak_live_regs"], c["peak_live_regs"])
                     for m, c in results[n]["width_cells"].items()
                     if m in base and m != 5
                     and c["peak_live_regs"] != base[m]["peak_live_regs"]}
            print("  %-22s %s" % (n, moved or "all untreated widths unmoved"))

    payload = {
        "head": head,
        "ceiling": CEILING,
        "pipeline": "metal -O2 -S | metal-opt -passes=default<O3>",
        "arms": {n: results[n] for n in names},
        "any_exceeds_ceiling": any(r["exceeds_ceiling"] for r in results.values()),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
