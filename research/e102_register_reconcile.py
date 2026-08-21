#!/usr/bin/env python3
"""E102 rung 1: measure the QMV DISPATCHER, not the bare kernel body.

Three instruments have published three different numbers for the same code:

  E97  AIR `peak_live_regs`, bare `_wide` body        NA = 5 -> 125
  E76  `metal-tt` registers, bare `_wide` body        NA = 5 -> 98
  E98  AIR `peak_live_regs`, whole entry point        155 - 163

They are not in conflict once the scope is stated. `peak_live_regs` counts SSA
values live at a point in architecture-INDEPENDENT AIR, before the AGX backend
has run, so it cannot know any machine's register file. `metal-tt` runs the
real per-generation backend and reports what that generation allocates. And the
entry point inlines every arm of the `ntg.x` switch into ONE `[[kernel]]`, so
its allocation is a max over branches and is not comparable to a per-cell
number at all.

E76 censused the bare body. The scored object is the dispatcher. This module
compiles the real shipped entry point, `affine_qmv_fast<bfloat16_t, 64, 4,
false>`, from the runtime-effective JIT string, patches only the dispatch table
inside that string, and reports four instruments per arm on both generations.
The runtime pipeline probe reads the SAME metallib that `metal-tt` reads, so
the arbiter and the static number describe one object.

Arms:

    A  shipped                  NA <= 4, <T,5,3> <T,9,3>
    B  the ca9251b8 table       NA <= 5, <T,5,5> <T,9,5>
    C  M = 5 only               NA <= 5, <T,5,5>, <T,9,3> kept
    D  the FACT 2b table        NA <= 6, <T,5,5> <T,6,6>
    E  dead M = 9 body removed  case label kept, body replaced by `break`
    F  dead M = 9 case removed  case label deleted as well
    G  M = 9 pruned from BOTH switch branches

`segmentedVerifyDepthCap = 7` caps the draft count at 7, so `M = draftCount + 1`
never exceeds 8 and `case 9` cannot execute in the scored worker. E, F and G
price what that dead instantiation costs the widths we do run. E against F
separates a dead-code effect from a switch-shape effect.

    python3 research/e102_register_reconcile.py --out research/out/e102/regs.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jit_string_compile import PREAMBLES, host_name, preamble  # noqa: E402
from agx_crossarch import LOCAL_ARCH, RANKED_ARCH, build_metallib, translate  # noqa: E402
from e46_reg_census import air_stats  # noqa: E402

REPO = HERE.parent
ENTRY_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
ENTRY = host_name(ENTRY_CELL)

NA_BOUND = ('static_assert(NA >= 2 && NA <= 4, '
            '"wide multi-row QMV supports NA in [2, 4]");')
WIDE_CALL = "qmv_fast_crossrow_affine4_g64_m<T, {m}, {ipg}, true>"
NARROW_CALL = "qmv_fast_crossrow_affine4_g64<T, {m}>"
SHIPPED_IPG = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}


def base_preambles() -> str:
    return "".join(preamble(stem, None) for stem in PREAMBLES)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"e102: anchor {label} matched {count} sites, expected 1")
    return text.replace(old, new)


def drop_case(text: str, call: str, label: str, keep_label: bool) -> str:
    """Remove one `case N:` arm, located by the unique call it contains."""
    if text.count(call) != 1:
        raise SystemExit(f"e102: case anchor {label} matched "
                         f"{text.count(call)} sites, expected 1")
    at = text.index(call)
    head = text.rindex("case ", 0, at)
    tail = text.index("return;", at) + len("return;")
    if keep_label:
        label_end = text.index(":", head) + 1
        new = text[head:label_end] + "\n          break;"
    else:
        new = ""
    return text[:head] + new + text[tail:]


# name -> (NA bound, {M: new IPG}, [(call template, M, keep case label)])
ARMS: dict[str, tuple[int, dict[int, int], list]] = {
    "A_shipped": (4, {}, []),
    "B_ca9251b8": (5, {5: 5, 9: 5}, []),
    "C_m5_only": (5, {5: 5}, []),
    "D_fact2b": (6, {5: 5, 6: 6}, []),
    "E_dead_m9_body": (4, {}, [(WIDE_CALL, 9, True)]),
    "F_dead_m9_case": (4, {}, [(WIDE_CALL, 9, False)]),
    "G_prune_both_m9": (4, {}, [(WIDE_CALL, 9, False), (NARROW_CALL, 9, False)]),
    # The narrow branch serves 1024 <= out_vec_size < 4096. The smallest scored
    # quantised projection is n = 5120 (E70 2.2, E74 rung 3), so nothing in the
    # scored path enters it. Case 2 shares its instantiation with the reachable
    # wide case 2, so dropping cases 3..9 removes every dead pair instantiation.
    "H_prune_narrow": (4, {}, [(NARROW_CALL, m, False) for m in range(3, 10)]),
    "I_prune_all_dead": (4, {}, [(NARROW_CALL, m, False) for m in range(3, 10)]
                        + [(WIDE_CALL, 9, False)]),
}

CELL_WRAPPER = """
[[kernel]] void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {{
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, {na}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      int(tid.x) * {na}, int(tid.y) * 8 + int(simd_gid) * 4, simd_lid);
}}
"""


def arm_source(na_bound: int, table: dict[int, int], drops: list) -> str:
    text = base_preambles()
    if na_bound != 4:
        text = replace_once(
            text, NA_BOUND,
            f'static_assert(NA >= 2 && NA <= {na_bound}, '
            f'"wide multi-row QMV supports NA in [2, {na_bound}]");',
            "na_bound")
    for m, ipg in sorted(table.items()):
        text = replace_once(text,
                            WIDE_CALL.format(m=m, ipg=SHIPPED_IPG[m]),
                            WIDE_CALL.format(m=m, ipg=ipg),
                            f"table_M{m}")
    for template, m, keep in drops:
        call = template.format(m=m, ipg=SHIPPED_IPG[m])
        text = drop_case(text, call, f"case_M{m}", keep)
    return text + (f'\ntemplate [[host_name("{ENTRY}")]] [[kernel]] '
                   f"decltype({ENTRY_CELL}) {ENTRY_CELL};\n")


def cells_source(widths: list[int]) -> tuple[str, dict]:
    """Bare-body cell wrappers: the E76 and E97 scope, on this same string."""
    text = replace_once(
        base_preambles(), NA_BOUND,
        f'static_assert(NA >= 2 && NA <= {max(widths)}, "probe-only NA bound");',
        "na_bound")
    labels, parts = {}, [text]
    for na in widths:
        name = f"e102_cell_na{na}"
        parts.append(CELL_WRAPPER.format(name=name, na=na))
        labels[name] = {"na": na, "rows_per_simd": 4, "accumulators": 4 * na}
    return "".join(parts), labels


def air_census(source: str, entries: list[str], workdir: pathlib.Path) -> dict:
    src = workdir / "air.metal"
    src.write_text(source)
    ll, ll_o3 = workdir / "air.ll", workdir / "air.o3.ll"
    emit = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-O2", "-S",
         str(src), "-o", str(ll)], capture_output=True, text=True)
    if emit.returncode != 0:
        return {e: {"status": "compile_failed",
                    "error": emit.stderr.strip().splitlines()[-8:]}
                for e in entries}
    opt = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal-opt", "-passes=default<O3>", "-S",
         str(ll), "-o", str(ll_o3)], capture_output=True, text=True)
    if opt.returncode != 0:
        return {e: {"status": "metal_opt_failed",
                    "error": opt.stderr.strip().splitlines()[-8:]}
                for e in entries}
    lines = ll_o3.read_text().splitlines()
    out = {}
    for entry in entries:
        body, inside = [], False
        for line in lines:
            if line.startswith("define ") and f"@{entry}(" in line:
                inside = True
            elif inside and line == "}":
                inside = False
            elif inside:
                body.append(line)
        out[entry] = dict(air_stats(body), status="ok") if body else \
            {"status": "entry_not_found"}
    return out


PIPE_FIELD = re.compile(r"([a-z_]+)=([^\s]+)")


def pipeline_probe(lib: pathlib.Path, entries: list[str],
                   workdir: pathlib.Path) -> dict:
    binary = workdir / "pipeline_probe"
    build = subprocess.run(
        ["clang", "-fobjc-arc", "-O2", "-framework", "Metal", "-framework",
         "Foundation", "-o", str(binary),
         str(REPO / "research/e102_pipeline_probe.m")],
        capture_output=True, text=True)
    if build.returncode != 0:
        return {"status": "probe_build_failed",
                "error": build.stderr.strip().splitlines()[-6:]}
    run = subprocess.run([str(binary), str(lib), *entries],
                         capture_output=True, text=True)
    if run.returncode != 0:
        return {"status": "probe_failed",
                "error": (run.stderr or run.stdout).strip().splitlines()[-6:]}
    per_entry = {}
    for line in run.stdout.splitlines():
        fields = dict(PIPE_FIELD.findall(line))
        for entry in entries:
            if f"function={entry}" in line:
                per_entry[entry] = fields
    return {"status": "ok", "raw": run.stdout, "per_entry": per_entry}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--arch", nargs="+", default=[LOCAL_ARCH, RANKED_ARCH])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--widths", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--keep", type=pathlib.Path,
                    help="keep the generated arm sources here")
    ap.add_argument("--skip-cells", action="store_true")
    args = ap.parse_args()

    result = {"entry_cell": ENTRY_CELL, "architectures": args.arch,
              "arms": {}, "cells": {}}

    with tempfile.TemporaryDirectory(prefix="e102-") as tmp:
        root = pathlib.Path(tmp)
        for name in args.arms:
            bound, table, drops = ARMS[name]
            work = root / name
            work.mkdir()
            source = arm_source(bound, table, drops)
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / f"{name}.metal").write_text(source)
            record = {"na_bound": bound, "table": table,
                      "dropped_cases": [m for _, m, _ in drops],
                      "source_bytes": len(source)}
            record["air_entry_scope"] = air_census(source, [ENTRY], work)[ENTRY]
            lib = build_metallib(source, work)
            for arch in args.arch:
                record[arch] = translate(lib, arch, work)[ENTRY]
            record["pipeline_local"] = pipeline_probe(lib, [ENTRY], work)
            result["arms"][name] = record
            print(f"[arm] {name} done", flush=True)

        if not args.skip_cells:
            work = root / "cells"
            work.mkdir()
            source, labels = cells_source(args.widths)
            names = list(labels)
            air = air_census(source, names, work)
            lib = build_metallib(source, work)
            tt = {arch: translate(lib, arch, work) for arch in args.arch}
            pipe = pipeline_probe(lib, names, work)
            for name in names:
                result["cells"][name] = dict(
                    labels[name], air_cell_scope=air.get(name),
                    pipeline_local=pipe.get("per_entry", {}).get(name),
                    **{arch: tt[arch][name] for arch in args.arch})
            result["cells_pipeline_raw"] = pipe.get("raw")
            print("[cells] done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    report(result, args.arch)
    print(f"wrote {args.out}")
    return 0


def _pipe(rec: dict | None, key: str) -> str:
    return (rec or {}).get(key, "-")


def report(result: dict, arches: list[str]) -> None:
    if result["arms"]:
        print("\n== dispatcher scope: the whole affine_qmv_fast entry point ==")
        head = f"{'arm':<20}{'AIRpeak':>8}{'AIRline':>8}"
        for a in arches:
            tag = a.replace("applegpu_", "")
            head += f"{tag + 'reg':>9}{tag + 'spl':>8}{tag + 'code':>10}"
        head += f"{'maxTPTG':>9}{'execW':>7}{'tgmem':>7}"
        print(head)
        for name, rec in result["arms"].items():
            air = rec["air_entry_scope"]
            row = (f"{name:<20}{air.get('peak_live_regs', '-'):>8}"
                   f"{air.get('air_lines', '-'):>8}")
            for arch in arches:
                r = rec[arch]
                row += (f"{r['registers']:>9}{r['spill_bytes']:>8}"
                        f"{r['text_sha8']:>10}")
            p = rec["pipeline_local"].get("per_entry", {}).get(ENTRY)
            row += (f"{_pipe(p, 'max_total_threads_per_threadgroup'):>9}"
                    f"{_pipe(p, 'thread_execution_width'):>7}"
                    f"{_pipe(p, 'static_threadgroup_memory_bytes'):>7}")
            print(row)

    if result["cells"]:
        print("\n== bare-body scope: the E76 and E97 instrument ==")
        head = f"{'cell':<18}{'NA':>4}{'acc':>5}{'AIRpeak':>8}"
        for a in arches:
            tag = a.replace("applegpu_", "")
            head += f"{tag + 'reg':>9}{tag + 'spl':>8}"
        head += f"{'maxTPTG':>9}"
        print(head)
        for name, rec in result["cells"].items():
            air = rec.get("air_cell_scope") or {}
            row = (f"{name:<18}{rec['na']:>4}{rec['accumulators']:>5}"
                   f"{air.get('peak_live_regs', '-'):>8}")
            for arch in arches:
                r = rec[arch]
                row += f"{r['registers']:>9}{r['spill_bytes']:>8}"
            row += (f"{_pipe(rec.get('pipeline_local'), 'max_total_threads_per_threadgroup'):>9}")
            print(row)


if __name__ == "__main__":
    raise SystemExit(main())
