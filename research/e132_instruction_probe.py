#!/usr/bin/env python3
"""Find a static instruction count for an AGX kernel we cannot disassemble.

E132-F2 makes the static instruction count of the inner k-loop the headline
number, ahead of resident simdgroups. `xcrun metal-objdump` on this toolchain
lists `agx1/agx2/agx3` as known targets but refuses every disassembly request,
so the instruction stream cannot be read directly. Two channels remain, and
this module calibrates both against kernels whose instruction count is known by
construction.

  channel 1  a `__GPU_METADATA` FlatBuffer field that tracks the count.
             `research/agx_crossarch.py` already calibrated field 0 as
             registers and field 14 as spill bytes the same way, so the same
             search is applied to every field.
  channel 2  `__TEXT,__text` size. AGX is a variable-length encoding, so bytes
             are not instructions, but if bytes per added instruction is
             stable across very different instruction kinds then text size
             divides into a usable instruction count.

`calibrate` builds a ladder of kernels that differ by an exactly known number
of machine instructions and reports, for each candidate channel, the fitted
slope, the intercept and the residual. A channel is accepted only when the
slope is constant across every ladder, which is what makes the derived count a
measurement rather than a guess.

  python3 research/e132_instruction_probe.py calibrate
  python3 research/e132_instruction_probe.py fields --arch applegpu_g17s
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402

ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)


# --------------------------------------------------------------------------
# ladders whose instruction delta is known by construction
# --------------------------------------------------------------------------

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

# One dependent fma per rung. The chain is serial so nothing vectorises across
# rungs, `-fno-fast-math` stops the compiler from reassociating it, and the
# accumulator stays in one register, so rung n holds exactly n more fma
# instructions than rung 0 and nothing else changes.
FMA_LADDER = """kernel void k_fma{n}(
    device float* o, device const float* a,
    uint i [[thread_position_in_grid]]) {{
  float v = a[i];
  float c = a[i + 1];
{body}  o[i] = v;
}}
"""

# One dependent float multiply per rung. Same shape, different instruction, and
# still unfoldable because float multiplication is not associative under
# `-fno-fast-math`. Two float ladders that return the same slope is what proves
# a channel counts instructions rather than one particular opcode.
FMUL_LADDER = """kernel void k_fmul{n}(
    device float* o, device const float* a,
    uint i [[thread_position_in_grid]]) {{
  float v = a[i];
  float c = a[i + 1];
{body}  o[i] = v;
}}
"""

# One extra device load plus one dependent add per rung, so two instructions
# per rung against the one of the other ladders. Loads are the instruction kind
# a spill emits, so this ladder is the one that matters most for E132, and a
# channel that returns slope 2 here and slope 1 above is counting instructions
# rather than arithmetic.
LOAD_LADDER = """kernel void k_load{n}(
    device float* o, device const float* a,
    uint i [[thread_position_in_grid]]) {{
  float v = a[i];
{body}  o[i] = v;
}}
"""

LADDERS = {
    "fma": (FMA_LADDER, "  v = fma(v, c, v);\n"),
    "fmul": (FMUL_LADDER, "  v = v * c;\n"),
    "load": (LOAD_LADDER, "  v = v + a[i + %d];\n"),
}
LADDER_EXPECTED = {"fma": 1, "fmul": 1, "load": 2}
RUNGS = (0, 8, 16, 24, 32, 40, 48)

# A pressure ladder built from named scalars, not from an array. An indexed
# `float v[n]` is demoted whole to the thread stack as soon as the index is not
# a constant, which reports a frame of `4n + 16` bytes for every rung and
# measures array placement rather than register spill. Distinct named values
# are register candidates, so the allocator really has to choose, and each rung
# adds exactly `ROUNDS` fma instructions of identical shape.
PRESSURE_ROUNDS = 3
PRESSURE_WIDTHS = tuple(range(16, 209, 16))


def pressure_kernel(n: int) -> str:
    """One kernel holding `n` named live floats across a serial fma chain."""
    decls = "".join("  float v%d = a[i + %d];\n" % (j, j) for j in range(n))
    work = ""
    for r in range(PRESSURE_ROUNDS):
        work += "".join(
            "  v%d = fma(v%d, v%d, v%d);\n"
            % (j, j, (j + 5 + r) % n, (j + 7 + r) % n) for j in range(n))
    total = " + ".join("v%d" % j for j in range(n))
    return ("kernel void k_p%d(\n"
            "    device float* o, device const float* a,\n"
            "    uint i [[thread_position_in_grid]]) {\n"
            "%s%s  o[i] = %s;\n}\n" % (n, decls, work, total))


# `research/agx_crossarch.py` calibrated the frame as `4 * slots + 16`. The
# `onset` ladder below falsifies that: every frame the allocator emits is a
# multiple of 16 bytes and a kernel that spills nothing reports exactly 0, so
# the frame is allocated on demand and 16-byte aligned. A frame therefore fixes
# the slot count only to within a band of four.
BYTES_PER_SLOT = 4


def slot_band(frame_bytes: int) -> tuple[int, int]:
    """Spilled 4-byte slots consistent with a 16-byte-aligned frame."""
    if frame_bytes <= 0:
        return (0, 0)
    return (max(1, frame_bytes // BYTES_PER_SLOT - 3),
            frame_bytes // BYTES_PER_SLOT)


def ladder_source(kind: str, n: int) -> str:
    template, step = LADDERS[kind]
    if "%d" in step:
        body = "".join(step % (j + 1) for j in range(n))
    else:
        body = step * n
    return template.format(n=n, body=body)


def ladder_library(kind: str) -> tuple[str, list[str]]:
    parts = [ladder_source(kind, n) for n in RUNGS]
    return "\n".join([PRELUDE] + parts), ["k_%s%d" % (kind, n) for n in RUNGS]


# --------------------------------------------------------------------------
# raw metadata field access
# --------------------------------------------------------------------------

def all_fields(metallib: pathlib.Path, arch: str, workdir: pathlib.Path,
               names: list[str]) -> dict[str, dict[int, int]]:
    """Every FlatBuffer field of the widest table, per kernel.

    `agx_crossarch.kernel_records` returns only the four fields the register
    census needs. Calibration has to see all of them.
    """
    out: dict[str, dict[int, int]] = {}
    for index, name in enumerate(names):
        script = workdir / ("probe_%s_%d.mtlp-json" % (arch, index))
        script.write_text(json.dumps({"pipelines": {"compute_pipelines": [
            {"compute_function": name}]}}))
        archive = workdir / ("probe_%s_%d.mtlp" % (arch, index))
        done = subprocess.run(
            ["xcrun", "metal-tt", "-arch", arch, str(metallib), str(script),
             "-o", str(archive)], capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit("metal-tt failed for %s %s:\n%s"
                             % (arch, name, done.stderr))
        blob = archive.read_bytes()
        compute = agx.section(blob, "__TEXT,__compute")
        starts = agx.find_mach_headers(compute)
        if len(starts) != 1:
            raise SystemExit("%s %s: %d objects in a one-kernel script"
                             % (arch, name, len(starts)))
        inner = compute[starts[0]:]
        metadata = agx.section(inner, "__GPU_METADATA,__compute")
        tables = agx.flatbuffer_tables(metadata)
        kernel = max(tables, key=len)
        text = agx.section(inner, "__TEXT,__text") or b""
        fields = dict(kernel)
        fields[-1] = len(text)  # text size, tracked as a pseudo-field
        out[name] = fields
    return out


# --------------------------------------------------------------------------
# AIR control-flow analysis
# --------------------------------------------------------------------------

LABEL = re.compile(r"^([%\w.\-]+):")
BRANCH = re.compile(r"\blabel %([\w.\-]+)")
DEFINE = re.compile(r"^define .*@([\w.$]+)\(")
INTRINSIC = re.compile(r"@(air\.[\w.]+)")


def emit_air(source: str, workdir: pathlib.Path) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "air.metal"
    src.write_text(source)
    out = workdir / "air.ll"
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", str(src), "-o", str(out)],
        check=True, capture_output=True)
    return out.read_text()


def air_functions(text: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Every defined function as an ordered list of (block label, lines)."""
    functions: dict[str, list[tuple[str, list[str]]]] = {}
    current: list[tuple[str, list[str]]] | None = None
    label, lines = "entry", []
    for raw in text.splitlines():
        start = DEFINE.match(raw)
        if start:
            current = []
            functions[start.group(1)] = current
            label, lines = "entry", []
            continue
        if current is None:
            continue
        if raw.startswith("}"):
            current.append((label, lines))
            current = None
            continue
        hit = LABEL.match(raw)
        if hit:
            current.append((label, lines))
            label, lines = hit.group(1), []
            continue
        body = raw.split(";")[0].strip()
        if body:
            lines.append(body)
    return functions


def successors(lines: list[str]) -> list[str]:
    terminator = lines[-1] if lines else ""
    if not terminator.startswith("br ") and not terminator.startswith("switch"):
        return []
    return BRANCH.findall(terminator)


def dominators(blocks: list[tuple[str, list[str]]]) -> dict[str, set[str]]:
    """Iterative dominator sets. The first block is the entry."""
    names = [name for name, _ in blocks]
    edges = {name: successors(lines) for name, lines in blocks}
    predecessors: dict[str, list[str]] = {name: [] for name in names}
    for name, outs in edges.items():
        for target in outs:
            if target in predecessors:
                predecessors[target].append(name)
    entry = names[0]
    universe = set(names)
    dom = {name: (universe if name != entry else {entry}) for name in names}
    changed = True
    while changed:
        changed = False
        for name in names[1:]:
            if not predecessors[name]:
                new = {name}
            else:
                new = set.intersection(
                    *(dom[p] for p in predecessors[name])) | {name}
            if new != dom[name]:
                dom[name] = new
                changed = True
    return dom


def natural_loops(blocks: list[tuple[str, list[str]]]) -> list[dict]:
    """Every natural loop, outermost first, with its instruction census.

    A back edge runs from a latch to a header that dominates it. The loop body
    is the header plus every block that reaches the latch without leaving
    through the header. Loops sharing a header are merged, so one `for` gives
    one loop however many `continue` edges it has.
    """
    body_of = {name: lines for name, lines in blocks}
    order = {name: i for i, (name, _) in enumerate(blocks)}
    edges = {name: successors(lines) for name, lines in blocks}
    predecessors: dict[str, list[str]] = {name: [] for name, _ in blocks}
    for name, outs in edges.items():
        for target in outs:
            if target in predecessors:
                predecessors[target].append(name)
    dom = dominators(blocks)

    merged: dict[str, set[str]] = {}
    for latch, outs in edges.items():
        for header in outs:
            if header not in dom or header not in dom[latch]:
                continue
            members = {header, latch}
            stack = [latch] if latch != header else []
            while stack:
                node = stack.pop()
                for prior in predecessors[node]:
                    if prior not in members:
                        members.add(prior)
                        stack.append(prior)
            merged.setdefault(header, set()).update(members)

    loops = []
    for header, members in merged.items():
        loops.append({
            "header": header,
            "blocks": sorted(members, key=lambda b: order[b]),
            "census": census_lines(
                [line for name in sorted(members, key=lambda b: order[b])
                 for line in body_of[name]]),
        })
    for loop in loops:
        loop["depth"] = sum(1 for other in loops
                            if other is not loop
                            and set(loop["blocks"]) < set(other["blocks"]))
    loops.sort(key=lambda l: (l["depth"], -len(l["blocks"])))
    return loops


def census_lines(lines: list[str]) -> dict[str, int]:
    """Classify AIR instructions. `phi` is excluded from the machine count.

    A `phi` is a name for a value that already lives in a register, so it
    usually costs no instruction. It is reported separately rather than
    silently dropped.
    """
    counts = {"total": 0, "phi": 0, "load": 0, "store": 0, "call": 0,
              "fma": 0, "float_alu": 0, "int_alu": 0, "gep": 0,
              "branch": 0, "other": 0}
    for line in lines:
        head = line.split("=")[-1].strip().split(" ")[0] if "=" in line \
            else line.split(" ")[0]
        counts["total"] += 1
        if head == "phi":
            counts["phi"] += 1
        elif head in ("load", "atomic"):
            counts["load"] += 1
        elif head == "store":
            counts["store"] += 1
        elif head in ("br", "switch", "ret"):
            counts["branch"] += 1
        elif head == "getelementptr":
            counts["gep"] += 1
        elif head in ("call", "tail"):
            counts["call"] += 1
            hit = INTRINSIC.search(line)
            if hit and hit.group(1).startswith("air.fma"):
                counts["fma"] += 1
        elif head in ("fadd", "fmul", "fsub", "fdiv", "fneg", "fpext",
                      "fptrunc", "fcmp"):
            counts["float_alu"] += 1
        elif head in ("add", "sub", "mul", "shl", "lshr", "ashr", "and",
                      "or", "xor", "icmp", "zext", "sext", "trunc",
                      "select", "bitcast", "insertelement",
                      "extractelement", "shufflevector"):
            counts["int_alu"] += 1
        else:
            counts["other"] += 1
    counts["machine"] = counts["total"] - counts["phi"] - counts["branch"]
    return counts


def slope(xs: list[int], ys: list[int]) -> tuple[float, float, float]:
    """Least-squares slope, intercept and maximum absolute residual."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return 0.0, my, 0.0
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    b = my - m * mx
    worst = max(abs(y - (m * x + b)) for x, y in zip(xs, ys))
    return m, b, worst


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def calibrate(out: pathlib.Path) -> int:
    result: dict = {
        "schema_version": 1,
        "gpu_used": False,
        "timing_valid": False,
        "harness": "compile_only",
        "tool": "research/e132_instruction_probe.py calibrate",
        "rungs": list(RUNGS),
        "ladders": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for kind in LADDERS:
            source, names = ladder_library(kind)
            lib = agx.build_metallib(source, workdir / kind)
            for arch in ARCHES:
                fields = all_fields(lib, arch, workdir / kind, names)
                shared = set.intersection(
                    *(set(fields[n]) for n in names))
                fits = {}
                for key in sorted(shared):
                    ys = [fields[n][key] for n in names]
                    if len(set(ys)) == 1:
                        continue
                    m, b, worst = slope(list(RUNGS), ys)
                    fits[key] = {
                        "slope": round(m, 4),
                        "intercept": round(b, 2),
                        "max_abs_residual": round(worst, 3),
                        "values": ys,
                    }
                result["ladders"].setdefault(kind, {})[arch] = fits

    print("E132 instruction-count channel calibration")
    print("rungs (extra instructions by construction): %s" % (list(RUNGS),))
    print()
    for kind in LADDERS:
        for arch in ARCHES:
            fits = result["ladders"][kind][arch]
            print("ladder %-5s %-16s  moving fields: %s"
                  % (kind, arch, sorted(fits) or "none"))
            for key in sorted(fits):
                fit = fits[key]
                label = "text_bytes" if key == -1 else "field %d" % key
                print("    %-12s slope %8.4f  intercept %8.2f  "
                      "max|resid| %6.3f  %s"
                      % (label, fit["slope"], fit["intercept"],
                         fit["max_abs_residual"], fit["values"]))
        print()

    # A channel is usable only when its slope is the same on every ladder.
    verdict = {}
    for arch in ARCHES:
        keys = set.intersection(*(set(result["ladders"][k][arch])
                                  for k in LADDERS))
        for key in sorted(keys):
            slopes = {k: result["ladders"][k][arch][key]["slope"]
                      for k in LADDERS}
            resid = max(result["ladders"][k][arch][key]["max_abs_residual"]
                        for k in LADDERS)
            # An instruction counter returns the constructed instruction delta
            # on every ladder: 1 per rung for the two arithmetic ladders and 2
            # per rung for the load ladder, which adds a load and an add.
            per_instruction = [slopes[k] / LADDER_EXPECTED[k] for k in LADDERS]
            spread = max(per_instruction) - min(per_instruction)
            verdict.setdefault(arch, {})[key] = {
                "slopes_per_ladder": slopes,
                "units_per_instruction": [round(v, 4)
                                          for v in per_instruction],
                "unit_spread": round(spread, 4),
                "max_abs_residual": round(resid, 3),
                "instruction_counter": bool(
                    abs(statistics.fmean(per_instruction) - 1.0) < 0.02
                    and spread < 0.02 and resid < 0.5),
            }
    result["verdict"] = verdict
    print("channels moving on ALL THREE ladders")
    for arch in ARCHES:
        for key, row in sorted(verdict.get(arch, {}).items()):
            label = "text_bytes" if key == -1 else "field %d" % key
            print("  %-16s %-12s slopes %s  units/instr %s  spread %.4f  "
                  "counter=%s"
                  % (arch, label, row["slopes_per_ladder"],
                     row["units_per_instruction"], row["unit_spread"],
                     row["instruction_counter"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote %s" % out)
    return 0


def spillcost(out: pathlib.Path) -> int:
    """Bytes of machine code one spilled slot costs, on each architecture."""
    source = "\n".join([PRELUDE] + [pressure_kernel(n)
                                    for n in PRESSURE_WIDTHS])
    names = ["k_p%d" % n for n in PRESSURE_WIDTHS]
    rows: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx.build_metallib(source, workdir / "p")
        for arch in ARCHES:
            found = agx.translate(lib, arch, workdir / "p",
                                  select=lambda n: n in set(names))
            for width, name in zip(PRESSURE_WIDTHS, names):
                record = found[name]
                spill = record["spill_bytes"]
                rows.setdefault(arch, []).append({
                    "width": width,
                    "registers": record["registers"],
                    "spill_bytes": spill,
                    "spill_slots_band": list(slot_band(spill)),
                    # The band is at most four wide against slot counts of
                    # 28 and up here, so the upper end drives the fit and the
                    # residual error stays inside the reported min-to-max
                    # spread.
                    "spill_slots": slot_band(spill)[1],
                    "text_bytes": record["text_bytes"],
                })

    result = {
        "schema_version": 1,
        "gpu_used": False,
        "timing_valid": False,
        "harness": "compile_only",
        "tool": "research/e132_instruction_probe.py spillcost",
        "widths": list(PRESSURE_WIDTHS),
        "rows": rows,
        "fit": {},
    }
    print("E132 spill-code cost ladder: same work per live value, "
          "rising pressure")
    for arch in ARCHES:
        table = rows[arch]
        clean = [r for r in table if r["spill_bytes"] == 0]
        dirty = [r for r in table if r["spill_bytes"] > 0]
        print("\n%s" % arch)
        print("  %-7s %-6s %-7s %-7s %-10s %s"
              % ("width", "regs", "spill_B", "slots", "text_B", "excess_B"))
        if len(clean) < 3 or not dirty:
            print("  ladder does not straddle the spill threshold")
            continue
        m, b, worst = slope([r["width"] for r in clean],
                            [r["text_bytes"] for r in clean])
        per_slot = []
        for row in table:
            expected = m * row["width"] + b
            excess = row["text_bytes"] - expected
            row["expected_text_bytes"] = round(expected, 1)
            row["excess_bytes"] = round(excess, 1)
            if row["spill_slots"] > 0:
                per_slot.append(excess / row["spill_slots"])
            print("  %-7d %-6d %-7d %-7d %-10d %+.1f"
                  % (row["width"], row["registers"], row["spill_bytes"],
                     row["spill_slots"], row["text_bytes"], excess))
        fit = {
            "no_spill_bytes_per_live_value": round(m, 3),
            "no_spill_intercept": round(b, 1),
            "no_spill_max_abs_residual": round(worst, 1),
            "bytes_per_spill_slot_mean": round(statistics.fmean(per_slot), 2),
            "bytes_per_spill_slot_min": round(min(per_slot), 2),
            "bytes_per_spill_slot_max": round(max(per_slot), 2),
        }
        result["fit"][arch] = fit
        print("  no-spill line: %.3f bytes per live value, intercept %.1f, "
              "max|resid| %.1f" % (m, b, worst))
        print("  spill code: %.2f bytes per slot "
              "(range %.2f to %.2f over %d spilling rungs)"
              % (fit["bytes_per_spill_slot_mean"],
                 fit["bytes_per_spill_slot_min"],
                 fit["bytes_per_spill_slot_max"], len(per_slot)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote %s" % out)
    return 0


def fields_command(arch: str) -> int:
    """Dump every metadata field of the shipped NA=8 body, for inspection."""
    import e132_wide_matvec as wm  # noqa: PLC0415
    swift = wm.swift_at(None)
    header = wm.header_at(swift)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        parts, names = [], []
        for na in (7, 8):
            for table in (True, False):
                name = "body_na%d_%s" % (na, "t" if table else "f")
                parts.append(wm.entry_point(name, wm.na_body(na, table), table))
                names.append(name)
        lib = agx.build_metallib(wm.library(header, parts), workdir / "f")
        found = all_fields(lib, arch, workdir / "f", names)
    keys = sorted(set().union(*(set(v) for v in found.values())))
    print("%-16s %s" % ("field", " ".join("%14s" % n for n in names)))
    for key in keys:
        label = "text_bytes" if key == -1 else "field %d" % key
        row = [found[n].get(key, "-") for n in names]
        if len(set(row)) == 1:
            continue
        print("%-16s %s" % (label, " ".join("%14s" % v for v in row)))
    return 0


def loops_command(out: pathlib.Path, widths=(6, 7, 8, 9)) -> int:
    """AIR inner-k-loop instruction census for every rung-1 candidate.

    The `i` loop inside a k block is fully unrolled, so after `-O2` the only
    loop left in `qwen_e120_qmv_wide` is the k loop. Its AIR instruction count
    is the static work the source asks for, per k block, before register
    allocation. It cannot see spill, which is why it is reported next to the
    machine-code size rather than instead of it.
    """
    import e132_wide_matvec as wm  # noqa: PLC0415
    import e132_wide_source as ws  # noqa: PLC0415

    swift = wm.swift_at(None)
    header = wm.header_at(swift)
    ws.assert_faithful(header)

    rows: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for name, (flags, _note) in wm.VARIANTS.items():
            patched = ws.patched_header(header, **flags)
            for table in (True, False):
                arm = "sumtable" if table else "notable"
                for na in widths:
                    kernel = "body_na%d" % na
                    part = wm.entry_point(
                        kernel, wm.na_body(na, table,
                                           rows_per_simd=ws.rows_per_simd(
                                               na, **flags)), table)
                    source = "\n".join([wm.e120.PRELUDE, patched, "", part])
                    text = emit_air(
                        source, workdir / ("%s_%s_%d" % (name, arm, na)))
                    functions = air_functions(text)
                    # `-O2` leaves the wide body as its own AIR function when
                    # the entry point calls it once. When it does inline it,
                    # the entry point itself is the only place the k loop can
                    # be, so that is the fallback.
                    match = [f for f in functions
                             if "qwen_e120_qmv_wide" in f] or [kernel]
                    if len(match) != 1 or match[0] not in functions:
                        raise SystemExit("%s %s NA=%d: %d AIR functions match"
                                         % (name, arm, na, len(match)))
                    loops = natural_loops(functions[match[0]])
                    if not loops:
                        raise SystemExit("%s %s NA=%d: no loop in AIR"
                                         % (name, arm, na))
                    inner = loops[0]
                    per_element = ws.output_elements_per_thread(na, **flags)
                    rows.setdefault(name, {}).setdefault(arm, {})[na] = {
                        "air_inner_loop_blocks": len(inner["blocks"]),
                        "air_inner_loop_instructions": inner["census"]["machine"],
                        "air_inner_loop_loads": inner["census"]["load"],
                        "air_inner_loop_fma": inner["census"]["fma"],
                        "air_inner_loop_float_alu": inner["census"]["float_alu"],
                        "air_inner_loop_int_alu": inner["census"]["int_alu"],
                        "output_elements_per_thread": per_element,
                        "air_inner_loop_instructions_per_output_element":
                            round(inner["census"]["machine"] / per_element, 2),
                        "loop_count": len(loops),
                    }

    result = {
        "schema_version": 1,
        "gpu_used": False,
        "timing_valid": False,
        "harness": "compile_only",
        "tool": "research/e132_instruction_probe.py loops",
        "note": "AIR is pre-register-allocation and therefore carries no "
                "spill code. Read it with the machine-code numbers from "
                "research/e132_wide_matvec.py rung1.",
        "widths": list(widths),
        "rows": rows,
    }
    for arm in ("sumtable", "notable"):
        print("\nAIR inner k-loop census [%s]   "
              "instructions (instructions per output element)" % arm)
        head = "%-13s %s" % ("variant", " ".join("%18s" % ("NA=%d" % na)
                                                 for na in widths))
        print(head)
        print("-" * len(head))
        for name in wm.VARIANTS:
            cells = []
            for na in widths:
                cell = rows[name][arm][na]
                cells.append("%6d (%7.2f)" % (
                    cell["air_inner_loop_instructions"],
                    cell["air_inner_loop_instructions_per_output_element"]))
            print("%-13s %s" % (name, " ".join("%18s" % c for c in cells)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote %s" % out)
    return 0


# The coarse `spillcost` ladder steps by 16 live values and never produces a
# frame between 0 and 128 bytes, so it cannot distinguish `frame = 4 * slots +
# 16` from a 16-byte-aligned `frame = 16 * ceil(slots / 4)`. The two models
# disagree exactly at a 16-byte frame, which is what the best NA=8 candidate
# reports, so the smallest frames have to be measured directly.
ONSET_WIDTHS = tuple(range(80, 135))


def onset(out: pathlib.Path) -> int:
    """Smallest non-zero spill frames, to fix the frame-to-slot law."""
    source = "\n".join([PRELUDE] + [pressure_kernel(n)
                                    for n in ONSET_WIDTHS])
    names = ["k_p%d" % n for n in ONSET_WIDTHS]
    rows: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx.build_metallib(source, workdir / "o")
        for arch in ARCHES:
            found = agx.translate(lib, arch, workdir / "o",
                                  select=lambda n: n in set(names))
            for width, name in zip(ONSET_WIDTHS, names):
                record = found[name]
                rows.setdefault(arch, []).append({
                    "width": width,
                    "registers": record["registers"],
                    "spill_bytes": record["spill_bytes"],
                    "text_bytes": record["text_bytes"],
                })

    verdicts = {}
    print("E132 spill-onset ladder: does a 16-byte frame mean zero slots?")
    for arch in ARCHES:
        frames = sorted({r["spill_bytes"] for r in rows[arch]
                         if r["spill_bytes"] > 0})
        smallest = frames[0] if frames else None
        first = next((r for r in rows[arch] if r["spill_bytes"] > 0), None)
        last_clean = None
        for row in rows[arch]:
            if row["spill_bytes"] == 0:
                last_clean = row
        # `4 * slots + 16` predicts frames of 20, 24, 28 just past the onset.
        # 16-byte alignment predicts 16 as the smallest possible frame.
        aligned = bool(frames) and all(f % 16 == 0 for f in frames)
        law = ("16 * ceil(slots / 4), so a 16-byte frame holds 1 to 4 slots"
               if aligned else "4 * slots + 16, so a 16-byte frame holds 0 "
               "slots")
        verdicts[arch] = {
            "distinct_frames": frames[:12],
            "smallest_non_zero_frame": smallest,
            "all_frames_16_byte_aligned": aligned,
            "law": law,
            "onset_width": first["width"] if first else None,
            "widest_clean_width": last_clean["width"] if last_clean else None,
            "widest_clean_registers": (last_clean["registers"]
                                       if last_clean else None),
        }
        print("\n%s" % arch)
        print("  onset at width %s, widest clean width %s at %s registers"
              % (verdicts[arch]["onset_width"],
                 verdicts[arch]["widest_clean_width"],
                 verdicts[arch]["widest_clean_registers"]))
        print("  smallest non-zero frame %s bytes" % smallest)
        print("  distinct frames %s" % frames[:12])
        print("  law: %s" % law)
        near = [r for r in rows[arch]
                if 0 < r["spill_bytes"] <= 64][:8]
        for row in near:
            print("    width %3d  regs %3d  frame %4d  text %6d"
                  % (row["width"], row["registers"], row["spill_bytes"],
                     row["text_bytes"]))

    result = {
        "schema_version": 1,
        "gpu_used": False,
        "timing_valid": False,
        "harness": "compile_only",
        "tool": "research/e132_instruction_probe.py onset",
        "question": ("does a 16-byte spill frame mean zero spilled slots, "
                     "or 1 to 4 slots under 16-byte frame alignment?"),
        "widths": list(ONSET_WIDTHS),
        "rows": rows,
        "verdicts": verdicts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("\nwrote %s" % out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    five = sub.add_parser("onset")
    five.add_argument("--out", type=pathlib.Path,
                      default=pathlib.Path("research/e132-artifacts/"
                                           "spill-onset.json"))
    one = sub.add_parser("calibrate")
    one.add_argument("--out", type=pathlib.Path,
                     default=pathlib.Path("research/e132-artifacts/"
                                          "instruction-channel.json"))
    two = sub.add_parser("fields")
    two.add_argument("--arch", default=agx.RANKED_ARCH)
    three = sub.add_parser("spillcost")
    three.add_argument("--out", type=pathlib.Path,
                       default=pathlib.Path("research/e132-artifacts/"
                                            "spill-code-cost.json"))
    four = sub.add_parser("loops")
    four.add_argument("--out", type=pathlib.Path,
                      default=pathlib.Path("research/e132-artifacts/"
                                           "air-inner-loop.json"))
    args = parser.parse_args(argv)
    if args.command == "onset":
        return onset(args.out)
    if args.command == "calibrate":
        return calibrate(args.out)
    if args.command == "spillcost":
        return spillcost(args.out)
    if args.command == "loops":
        return loops_command(args.out)
    return fields_command(args.arch)


if __name__ == "__main__":
    raise SystemExit(main())
