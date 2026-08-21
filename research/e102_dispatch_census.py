#!/usr/bin/env python3
"""E102 rung 0b: what input-rows-per-group every submission tree actually dispatches.

`e102_na_census.py` keys on the shared `_wide` template bound, so it can only
see trees that widened THAT template. Several trees instead added a private
kernel with the row count baked into its name -- `wide5_direct`,
`two_banks_rows2_direct` -- and those are invisible to a `static_assert(NA...)`
scan while still putting five input rows in one weight pass. This census reads
the `ntg.x` switch instead, so a tree is classified by what it dispatches.

For each distinct `quantized.h` blob it reports, per `case N`, the callee, the
callee's input-rows-per-group (IPG) and the callee's `rows_per_simd`.

    python3 research/e102_dispatch_census.py
"""

from __future__ import annotations

import collections
import json
import re
import subprocess

HDR = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"

SWITCH_HEAD = re.compile(
    r"if \(!batched && group_size == 64 && bits == 4 && out_vec_size >= 1024\)")
CASE = re.compile(r"^\s*case (\d+):\s*$", re.M)
CALL = re.compile(r"(qmv_fast_[A-Za-z0-9_]+)\s*<\s*T\s*,?\s*([^>]*)>")
FUNC_DEF = re.compile(
    r"template <([^>]*)>\s*\nMETAL_FUNC void (qmv_fast_[A-Za-z0-9_]+)\(")
NO_TMPL_DEF = re.compile(r"^METAL_FUNC void (qmv_fast_[A-Za-z0-9_]+)\(", re.M)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout


def func_bodies(text: str) -> dict[str, str]:
    """Map every qmv_fast_* definition to its body text."""
    starts = []
    for m in re.finditer(r"METAL_FUNC void (qmv_fast_[A-Za-z0-9_]+)\(", text):
        starts.append((m.start(), m.group(1)))
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        out[name] = text[pos:end]
    return out


def ipg_of(name: str, targs: str, bodies: dict[str, str]) -> str:
    """Input rows per weight pass for one dispatch callee."""
    args = [a.strip() for a in targs.split(",") if a.strip()]
    body = bodies.get(name, "")
    rps = re.findall(r"constexpr int rows_per_simd = ([A-Za-z0-9_]+)", body)
    rps_s = rps[0] if rps else "?"
    if name.endswith("_g64_m") and len(args) >= 2:
        return f"IPG={args[1]} rps={rps_s}"
    if name.endswith("_g64_mN") and len(args) >= 2:
        return f"IPG={args[1]} rps={rps_s}"
    if name.endswith("_g64") and len(args) >= 1:
        return f"IPG={args[0]} rps={rps_s}"
    # Private kernels bake the row count in. Recover it from the callee body:
    # the widest `vec<float, K>` or an explicit NA constant.
    na = re.findall(r"vec<float, (\d+)>", body)
    inner = re.findall(r"qmv_fast_crossrow_affine4_g64_wide<\s*T,\s*(\d+)", body)
    baked = sorted({int(x) for x in na + inner})
    return f"IPG~{baked or '?'} rps={rps_s} args={args}"


def switch_map(text: str) -> list[tuple[int, str]]:
    m = SWITCH_HEAD.search(text)
    if not m:
        return []
    region = text[m.start():m.start() + 9000]
    # Only the >= 4096 arm matters: the scored projections are all >= 4096 wide.
    hi = region.find("if (out_vec_size >= 4096)")
    if hi < 0:
        return []
    lo_marker = region.find("} else {", hi)
    region = region[hi:lo_marker if lo_marker > 0 else len(region)]
    bodies_out = []
    cases = list(CASE.finditer(region))
    for i, c in enumerate(cases):
        end = cases[i + 1].start() if i + 1 < len(cases) else len(region)
        chunk = region[c.start():end]
        call = CALL.search(chunk)
        bodies_out.append((int(c.group(1)),
                           f"{call.group(1)}<T,{call.group(2)}>" if call else "?"))
    return bodies_out


def main() -> None:
    subs = json.load(open("/tmp/e102-na-census.json"))["subs"]
    by_blob = collections.defaultdict(list)
    for s in subs:
        by_blob[s["hdr"]].append(s["sub"])

    rows = []
    for blob, members in by_blob.items():
        if blob is None:
            rows.append({"blob": blob, "n": len(members), "subs": members,
                         "dispatch": None})
            continue
        text = git("cat-file", "blob", blob)
        bodies = func_bodies(text)
        dispatch = []
        for case_m, callee in switch_map(text):
            name = callee.split("<")[0]
            targs = callee.split("<T,", 1)[1].rstrip(">") if "<T," in callee else ""
            dispatch.append({"M": case_m, "callee": name,
                             "shape": ipg_of(name, targs, bodies)})
        rows.append({"blob": blob, "n": len(members), "subs": members,
                     "dispatch": dispatch})

    key = lambda r: json.dumps(r["dispatch"], sort_keys=True)
    groups = collections.defaultdict(lambda: {"n": 0, "subs": []})
    for r in rows:
        g = groups[key(r)]
        g["n"] += r["n"]
        g["subs"].extend(r["subs"])

    print(f"distinct >=4096 dispatch tables: {len(groups)}")
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        d = json.loads(k)
        print(f"\nn={g['n']:4d}  example {g['subs'][0][:8]}")
        if not d:
            print("    (no >=4096 crossrow switch found)")
            continue
        for e in d:
            print(f"    M={e['M']}: {e['callee']:<52s} {e['shape']}")
    json.dump({k: v for k, v in groups.items()},
              open("/tmp/e102-dispatch-census.json", "w"))


if __name__ == "__main__":
    main()
