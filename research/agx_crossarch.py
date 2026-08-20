#!/usr/bin/env python3
"""Read per-kernel register and spill numbers for a GPU generation we do not have.

`xcrun metal` stops at AIR, which is architecture-independent, so it cannot
answer register questions for any generation. `xcrun metal-tt` goes further: it
runs the real AGX backend for a named architecture and writes a GPU executable.
macOS ships one backend per Apple GPU generation on every Mac, so a G16 host can
translate for `applegpu_g17s`, the ranked runner's generation.

The translated binary carries a `__GPU_METADATA` section holding an undocumented
FlatBuffer. Two of its fields are what we need, and both are calibrated in
`selftest` against kernels whose register pressure is known by construction:

  registers   the per-kernel register count
  spill_bytes present only when the kernel spills, and equal to 4 * N + 16 for
              a kernel holding N live floats

No public field reports occupancy, so this module does not report it.

  python3 research/agx_crossarch.py selftest
  python3 research/agx_crossarch.py census --metallib X.metallib --arch g16s g17s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

# The generation this host runs, and the generation the ranked runner runs.
LOCAL_ARCH = "applegpu_g16s"
RANKED_ARCH = "applegpu_g17s"

# Field indices inside the third FlatBuffer table of `__GPU_METADATA`. They are
# undocumented, so `selftest` re-derives them from known kernels on every run.
REGISTER_FIELD = 0
SPILL_FIELD = 14
SPILL_MIRROR_FIELD = 41

SECTION_COMMAND = 25  # LC_SEGMENT_64


def mach_sections(blob: bytes) -> list[tuple[str, int, int]]:
    count, = struct.unpack_from("<I", blob, 16)
    offset, found = 32, []
    for _ in range(count):
        command, size = struct.unpack_from("<II", blob, offset)
        if command == SECTION_COMMAND:
            n_sections, = struct.unpack_from("<I", blob, offset + 64)
            cursor = offset + 72
            for _ in range(n_sections):
                name = blob[cursor:cursor + 16].rstrip(b"\x00").decode()
                segment = blob[cursor + 16:cursor + 32].rstrip(b"\x00").decode()
                length, = struct.unpack_from("<Q", blob, cursor + 40)
                start, = struct.unpack_from("<I", blob, cursor + 48)
                found.append((f"{segment},{name}", start, length))
                cursor += 80
        offset += size
    return found


def section(blob: bytes, want: str) -> bytes | None:
    for name, start, length in mach_sections(blob):
        if name == want:
            return blob[start:start + length]
    return None


def flatbuffer_tables(blob: bytes) -> list[dict[int, int]]:
    """Recover every plausible FlatBuffer table as {field index: value}.

    A FlatBuffer table starts with a signed offset back to its vtable. The
    vtable stores its own size and the table size, so a candidate is accepted
    only when both agree and both stay in range.
    """
    size = len(blob)
    tables = []
    for table in range(4, size - 4, 2):
        back, = struct.unpack_from("<i", blob, table)
        if back < 4 or back > 4096 or table - back < 0:
            continue
        vtable = table - back
        if vtable + 4 > size:
            continue
        vtable_size, table_size = struct.unpack_from("<HH", blob, vtable)
        if vtable_size != back or not 4 <= vtable_size <= 512:
            continue
        if not 4 <= table_size <= 512 or table + table_size > size:
            continue
        fields = {}
        for index in range((vtable_size - 4) // 2):
            at, = struct.unpack_from("<H", blob, vtable + 4 + 2 * index)
            if at == 0:
                continue
            if at + 4 <= table_size:
                fields[index] = struct.unpack_from("<I", blob, table + at)[0]
            elif at + 2 <= table_size:
                fields[index] = struct.unpack_from("<H", blob, table + at)[0]
            else:
                fields[index] = blob[table + at]
        tables.append(fields)
    return tables


def kernel_names(metallib: pathlib.Path) -> list[str]:
    listing = subprocess.run(
        ["xcrun", "metal-nm", "--defined-only", str(metallib)],
        capture_output=True, text=True, check=True)
    names = [line.split()[-1] for line in listing.stdout.splitlines() if line.strip()]
    if not names:
        raise SystemExit(f"no compute functions in {metallib}")
    return names


def translate(metallib: pathlib.Path, arch: str,
              workdir: pathlib.Path) -> dict[str, dict]:
    """Run the real AGX backend for `arch` and read one record per kernel.

    Each kernel is translated on its own. `__GPU_METADATA` carries no kernel
    name, and the order of the objects inside `__TEXT,__compute` is the
    linker's, not the pipeline script's: translating a multi-kernel script and
    zipping the two lists silently mispairs records. Requesting one kernel per
    script costs one process per kernel and removes the ambiguity entirely.
    """
    found = {}
    for index, name in enumerate(kernel_names(metallib)):
        script = workdir / f"pipeline_{index}.mtlp-json"
        script.write_text(json.dumps({"pipelines": {"compute_pipelines": [
            {"compute_function": name}]}}))
        out = workdir / f"{arch}_{index}.mtlp"
        done = subprocess.run(
            ["xcrun", "metal-tt", "-arch", arch, str(metallib), str(script),
             "-o", str(out)],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"metal-tt failed for {arch} {name}:\n{done.stderr}")
        records = kernel_records(out.read_bytes())
        if len(records) != 1:
            raise SystemExit(
                f"{arch} {name}: {len(records)} kernels in a one-kernel script")
        found[name] = records[0]
    return found


def kernel_records(blob: bytes) -> list[dict]:
    """Every embedded GPU executable in a translated pipeline archive.

    `metal-tt` writes one outer Mach-O whose `__TEXT,__compute` section holds
    the per-kernel GPU objects back to back, so the objects are reached through
    that section rather than by scanning the archive.
    """
    compute = section(blob, "__TEXT,__compute")
    if compute is None:
        raise SystemExit("translated archive has no __TEXT,__compute section")
    records = []
    for start in find_mach_headers(compute):
        inner = compute[start:]
        metadata = section(inner, "__GPU_METADATA,__compute")
        if metadata is None:
            continue
        tables = flatbuffer_tables(metadata)
        if len(tables) < 3:
            continue
        kernel = max(tables, key=len)
        text = section(inner, "__TEXT,__text") or b""
        records.append({
            "registers": kernel.get(REGISTER_FIELD),
            "spill_bytes": kernel.get(SPILL_FIELD, 0),
            "spill_mirror": kernel.get(SPILL_MIRROR_FIELD, 0),
            # Machine-code size and digest. Two kernels that report the same
            # register count are only really the same if these agree too, so
            # this is what separates a genuine tie from a mispaired record.
            "text_bytes": len(text),
            "text_sha8": hashlib.sha256(text).hexdigest()[:8],
        })
    return records


MACH_MAGIC = b"\xcf\xfa\xed\xfe"


def find_mach_headers(blob: bytes) -> list[int]:
    found, at = [], blob.find(MACH_MAGIC)
    while at != -1:
        found.append(at)
        at = blob.find(MACH_MAGIC, at + 4)
    return found


SELFTEST_KERNEL = """kernel void k_v{n}(
    device float* o, device const float* a,
    uint i [[thread_position_in_grid]]) {{
  float v[{n}];
  for (int j = 0; j < {n}; ++j) v[j] = a[i * {n} + j];
  for (int r = 0; r < 3; ++r)
    for (int j = 0; j < {n}; ++j)
      v[j] = fma(v[j], v[(j + 5) % {n}], v[(j + 7) % {n}]);
  float s = 0;
  for (int j = 0; j < {n}; ++j) s += v[j];
  o[i] = s;
}}
"""
SELFTEST_WIDTHS = [8, 24, 32, 48, 64, 96]


def build_metallib(source: str, workdir: pathlib.Path,
                   include: pathlib.Path | None = None) -> pathlib.Path:
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "probe.metal"
    src.write_text(source)
    air = workdir / "probe.air"
    flags = ["-std=metal4.0", "-O2", "-fno-fast-math"]
    if include is not None:
        flags += ["-I", str(include)]
    subprocess.run(["xcrun", "-sdk", "macosx", "metal", *flags, "-c",
                    str(src), "-o", str(air)], check=True, capture_output=True)
    lib = workdir / "probe.metallib"
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air),
                    "-o", str(lib)], check=True, capture_output=True)
    return lib


def selftest() -> int:
    """Calibrate the two fields against kernels of known register pressure.

    Each `k_v{n}` holds n live floats across a dependent fma chain, so pressure
    rises with n and the backend must start spilling at some width. The fields
    are only usable if the spill field appears exactly when spilling starts and
    matches the 4 * n + 16 frame the source implies.
    """
    source = ("#include <metal_stdlib>\nusing namespace metal;\n"
              + "".join(SELFTEST_KERNEL.format(n=n) for n in SELFTEST_WIDTHS))
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = build_metallib(source, workdir)
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            found = translate(lib, arch, workdir)
            print(f"  {arch}")
            for width in SELFTEST_WIDTHS:
                record = found[f"k_v{width}"]
                spill = record["spill_bytes"]
                # A kernel holding n live floats that spills its whole working
                # set needs a 4 * n byte frame plus a fixed 16 byte header.
                predicted = 4 * width + 16
                note = ""
                if spill:
                    note = "ok" if spill == predicted else f"EXPECTED {predicted}"
                    if spill != predicted:
                        failures.append(
                            f"{arch} n={width}: spill {spill} != {predicted}")
                if record["spill_mirror"] != spill:
                    failures.append(f"{arch} n={width}: mirror disagrees")
                print(f"    n={width:<3} registers={record['registers']:<4} "
                      f"spill_bytes={spill:<4} {note}")
            if found[f"k_v{SELFTEST_WIDTHS[0]}"]["spill_bytes"]:
                failures.append(f"{arch}: narrowest kernel already spills")
            if not found[f"k_v{SELFTEST_WIDTHS[-1]}"]["spill_bytes"]:
                failures.append(f"{arch}: widest kernel does not spill")
            # A record is only attributable if reordering the source cannot
            # move it. Rebuilding with the kernels emitted in reverse must
            # return each kernel its own machine code, not a permutation.
            reversed_source = ("#include <metal_stdlib>\nusing namespace metal;\n"
                               + "".join(SELFTEST_KERNEL.format(n=n)
                                         for n in reversed(SELFTEST_WIDTHS)))
            shuffled = translate(
                build_metallib(reversed_source, workdir / f"rev_{arch}"),
                arch, workdir / f"rev_{arch}")
            moved = [f"k_v{w}" for w in SELFTEST_WIDTHS
                     if shuffled[f"k_v{w}"]["text_sha8"] != found[f"k_v{w}"]["text_sha8"]]
            for name in moved:
                failures.append(
                    f"{arch} {name}: record moved when the source was "
                    "reordered, so records are not attributable")
            print(f"    permutation guard: "
                  f"{'PASS' if not moved else 'FAIL ' + ','.join(moved)} "
                  f"({len(SELFTEST_WIDTHS)} kernels re-emitted in reverse)")
    for line in failures:
        print(f"SELFTEST FAILED {line}")
    print(f"selftest: {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


def scalar_kernel(n: int) -> str:
    """A kernel holding n live floats in named scalars, never in an array.

    `SELFTEST_KERNEL` keeps its working set in `float v[n]` indexed by a
    non-constant expression, so past a certain width the front end stops
    promoting the array and the resulting frame reports as a spill. That
    measures a promotion heuristic rather than the register file. Naming every
    value removes the array, so the only thing that can force a frame here is
    genuinely running out of registers.
    """
    load = "".join(f"  float v{j} = a[i * {n} + {j}];\n" for j in range(n))
    work = ""
    for _ in range(3):
        work += "".join(
            f"  v{j} = fma(v{j}, v{(j + 5) % n}, v{(j + 7) % n});\n"
            for j in range(n))
    total = " + ".join(f"v{j}" for j in range(n))
    return (f"kernel void k_s{n}(device float* o, device const float* a,\n"
            f"    uint i [[thread_position_in_grid]]) {{\n"
            f"{load}{work}  o[i] = {total};\n}}\n")


def wall(out: pathlib.Path | None) -> int:
    """Locate the register budget by sweeping until the backend starts spilling.

    A census number means nothing without the budget it is measured against,
    so this reports the widest kernel that still fits and the highest register
    count reached without a frame, in the same units the census prints.
    """
    widths = list(range(8, 97, 4))
    source = ("#include <metal_stdlib>\nusing namespace metal;\n"
              + "".join(scalar_kernel(n) for n in widths))
    result = {"live_floats": widths, "sweep": {}, "budget": {}}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = build_metallib(source, workdir)
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            found = translate(lib, arch, workdir)
            result["sweep"][arch] = found
            print(f"  {arch}")
            for n in widths:
                record = found[f"k_s{n}"]
                print(f"    floats={n:<3} registers={record['registers']:<4} "
                      f"spill_bytes={record['spill_bytes']}")
            clean = [(n, found[f"k_s{n}"]["registers"]) for n in widths
                     if not found[f"k_s{n}"]["spill_bytes"]]
            last, peak = max(clean, key=lambda pair: pair[1])
            first_frame = next((n for n in widths
                                if found[f"k_s{n}"]["spill_bytes"]), None)
            result["budget"][arch] = {
                "max_registers_without_a_frame": peak,
                "at_live_floats": last,
                "first_live_float_count_with_a_frame": first_frame,
            }
            print(f"  {arch}: highest register count reached without a frame "
                  f"= {peak} at {last} live floats, ratio {peak / last:.3f} "
                  "per float")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    budget = sub.add_parser("wall")
    budget.add_argument("--out", type=pathlib.Path)
    census = sub.add_parser("census")
    census.add_argument("--metallib", type=pathlib.Path, required=True)
    census.add_argument("--arch", nargs="+", default=[LOCAL_ARCH, RANKED_ARCH])
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest()
    if args.command == "wall":
        return wall(args.out)

    with tempfile.TemporaryDirectory() as tmp:
        for arch in args.arch:
            found = translate(args.metallib, arch, pathlib.Path(tmp))
            print(arch, json.dumps(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
