#!/usr/bin/env python3
"""E54 step 0: prove what `vec<float, 5>` is before timing anything that uses it.

The wide multi-row QMV helper accumulates in `typedef vec<float, NA> VF` and
asserts `NA <= 4`. MSL specifies `vec<T, N>` only for N in {2, 3, 4}, so the
NA = 5 arms of E49 and E54 rest on a type the specification does not name. This
script answers four questions with the compiler, not with prose:

  1. does `vec<float, 5>` compile at all?
  2. `sizeof` and `alignof`, bisected by `static_assert`: exactly one candidate
     compiles, and a wrong candidate is a hard compile error, so the answer
     cannot be a misparse of a log line;
  3. what the accumulator array becomes in AIR (`[4 x <N x float>]` and its
     alignment), for NA = 2..8, which is where the layout cost of an
     unrepresentable width shows up;
  4. whether the real scored cell `qmv_fast_crossrow_affine4_g64_m<T, M, IPG>`
     compiles at IPG = 5 on a relaxed header, and what its register heuristic
     and accumulator allocas are next to the shipped IPG.

No GPU. The lane-correctness half of the proof is bitwise and runs on the real
kernel through `research/e54_parity.sh`.

  python3 research/e54_vec5_proof.py --out research/e54-artifacts/vec5-proof.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e46_reg_census import HEADER, HEADER_REL, compile_probe  # noqa: E402
from e54_arms import relax_asserts  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "research/e54_vec5_probe.metal"
CELL = "e46_cell"
NA_RANGE = list(range(2, 9))
SIZEOF_CANDIDATES = [8, 12, 16, 20, 24, 28, 32, 40, 64]
ALIGNOF_CANDIDATES = [4, 8, 12, 16, 20, 32, 64]
# (M, IPG) pairs: the shipped cell and the NA=5 cell for every width E54 times,
# plus M=9 from E49 so the four-cell table has one uniform census.
CELLS = [(5, 3), (5, 5), (7, 4), (7, 5), (8, 4), (8, 5), (9, 3), (9, 5)]

ALLOCA_VEC = re.compile(r"alloca \[(\d+) x <(\d+) x float>\], align (\d+)")


def compile_vec_probe(na: int, defines: dict[str, int],
                      out: pathlib.Path) -> tuple[bool, str]:
    flags = ["-D%s=%d" % (k, v) for k, v in defines.items()]
    res = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2",
         "-DE54_NA=%d" % na, *flags, "-S", str(PROBE), "-o", str(out)],
        capture_output=True, text=True)
    return res.returncode == 0, res.stderr.strip()


def bisect(na: int, key: str, candidates: list[int],
           workdir: pathlib.Path) -> dict:
    """Exactly one candidate must compile; anything else is reported as-is."""
    accepted = []
    errors = {}
    for value in candidates:
        ok, err = compile_vec_probe(
            na, {key: value}, workdir / ("bisect_na%d_%s.ll" % (na, key)))
        if ok:
            accepted.append(value)
        else:
            errors[value] = err.splitlines()[-1] if err else ""
    return {"accepted": accepted, "unique": len(accepted) == 1,
            "value": accepted[0] if len(accepted) == 1 else None,
            "candidates": candidates,
            "rejected_example": next(iter(errors.values()), None)}


def vec_layout(workdir: pathlib.Path) -> dict:
    out = {}
    for na in NA_RANGE:
        ll = workdir / ("layout_na%d.ll" % na)
        ok, err = compile_vec_probe(na, {}, ll)
        if not ok:
            out[na] = {"compiles": False,
                       "error": err.splitlines()[-3:] if err else []}
            continue
        hit = ALLOCA_VEC.search(ll.read_text())
        row = {"compiles": True,
               "sizeof": bisect(na, "E54_SIZEOF", SIZEOF_CANDIDATES, workdir),
               "alignof": bisect(na, "E54_ALIGNOF", ALIGNOF_CANDIDATES, workdir)}
        if hit:
            rows, lanes, align = (int(hit.group(i)) for i in (1, 2, 3))
            row["acc_alloca"] = "[%d x <%d x float>]" % (rows, lanes)
            row["acc_alloca_align"] = align
            row["lanes"] = lanes
        sz = row["sizeof"]["value"]
        if sz is not None:
            row["payload_bytes"] = 4 * na
            row["padding_bytes"] = sz - 4 * na
            row["float_slots_occupied"] = sz // 4
        out[na] = row
    return out


def real_cell_census(workdir: pathlib.Path) -> dict:
    """The scored cell at IPG=5 versus its shipped IPG, on a relaxed header."""
    shadow = workdir / "relaxed_header"
    dst = shadow / HEADER_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(relax_asserts(HEADER.read_text()))

    out = {}
    for m, ipg in CELLS:
        res = compile_probe(shadow, "cell_m%d_ipg%d" % (m, ipg),
                            {"E46_CELL_M": m, "E46_CELL_IPG": ipg}, (CELL,))
        key = "<T,%d,%d>" % (m, ipg)
        if res["status"] != "ok":
            out[key] = {"status": res["status"], "error": res.get("error")}
            continue
        fn = res["functions"][CELL]
        tail = m % ipg
        out[key] = {
            "status": "ok",
            "peak_live_regs": fn["peak_live_regs"],
            "allocas": fn["allocas"],
            "acc_alloca_types": fn["acc_alloca_types"],
            "device_loads": fn["device_loads"],
            "air_lines": fn["air_lines"],
            "working_groups": (m + ipg - 1) // ipg,
            "group_na_values": sorted({ipg} | ({max(tail, 2)} if tail else set())),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e54-artifacts/vec5-proof.json")
    args = ap.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="e54-vec5-"))
    payload = {
        "question": "what does vec<float,5> compile to, and does the scored "
                    "cell compile at IPG=5?",
        "probe": str(PROBE.relative_to(REPO)),
        "toolchain": subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True, text=True).stderr.strip().splitlines()[:2],
        "vec_layout": vec_layout(workdir),
        "real_cells": real_cell_census(workdir),
        "workdir": str(workdir),
    }

    print("vec<float,NA> layout, from the Metal front end")
    print("  %-4s %-9s %-9s %-9s %-22s %s"
          % ("NA", "compiles", "sizeof", "alignof", "acc alloca", "align"))
    for na, row in sorted(payload["vec_layout"].items()):
        if not row["compiles"]:
            print("  %-4d %-9s %s" % (na, "NO", row.get("error")))
            continue
        print("  %-4d %-9s %-9s %-9s %-22s %s"
              % (na, "yes", row["sizeof"]["value"], row["alignof"]["value"],
                 row.get("acc_alloca"), row.get("acc_alloca_align")))

    print("\nthe real scored cell, relaxed header, compile only")
    print("  %-12s %-8s %-8s %-8s %-10s %s"
          % ("cell", "regs", "allocas", "groups", "group NA", "acc alloca types"))
    for key, row in payload["real_cells"].items():
        if row["status"] != "ok":
            print("  %-12s %s %s" % (key, row["status"], row.get("error")))
            continue
        print("  %-12s %-8d %-8d %-8d %-10s %s"
              % (key, row["peak_live_regs"], row["allocas"],
                 row["working_groups"], row["group_na_values"],
                 row["acc_alloca_types"]))

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
