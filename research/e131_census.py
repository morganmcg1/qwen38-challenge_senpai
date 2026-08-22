#!/usr/bin/env python3
"""E131 rung 0: every body the wide-QMV entry point can inline, and what binds it.

Ledger 274 F107 established that `affine_qmv_fast` compiles every width into
ONE Metal function, so its register allocation is the maximum over all inlined
bodies and every dispatch runs at that occupancy. E125 censused four `_wide<NA>`
bodies and the entry point. That is not the whole entry point: the same function
also carries the non-`_m` cross-row family, the generic MLX `qmv_fast_impl`
fall-through, and - in other template cells - the affine-2 single-row readout
and `adjust_matrix_offsets`.

Two frames are produced, because they answer different questions:

  isolated bodies   one wrapper entry point per body, so each body's own
                    register cost is visible. Never a live kernel.
  entry points      the real instantiations, plus counterfactual copies with
                    named call sites deleted. These are ground truth: they are
                    what the register allocator actually sees.

The counterfactuals answer the question alphonse's E130 is blocked on: what
becomes the binding register count once the M=5 call site is gone.

This is a compile-only census. It runs the real AGX backend for both
architectures through `xcrun metal-tt` and never touches the GPU, so it carries
no timing, thermal or gate claim. Nothing in the tracked kernel tree is edited:
every counterfactual is written into a scratch directory.

    python3 research/e131_census.py --outdir research/e131-artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402
from e123_arms import SIMDGROUP_BUDGET, simdgroups  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHES = (agx.LOCAL_ARCH, agx.RANKED_ARCH)

SCORED_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, false>"
BITS2_CELL = "affine_qmv_fast<bfloat16_t, 64, 2, false>"
BATCHED_CELL = "affine_qmv_fast<bfloat16_t, 64, 4, true>"

# The `out_vec_size >= 4096` switch, isolated by span. The `< 4096` switch calls
# the pair kernel under a byte-identical name, so a global replace would hit
# both.
WIDE_OPEN = "    if (out_vec_size >= 4096) {\n"
WIDE_CLOSE = "    } else {\n"

CASE5_M = """        case 5:
          qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""
CASE2_PAIR = """        case 2:
          qmv_fast_crossrow_affine4_g64<T, 2>(
              w, scales, biases, x, y, in_vec_size, out_vec_size,
              tid, simd_gid, simd_lid);
          return;
"""

# The live `switch (ntg.x)` in the wide branch, read out of the emitted base.
# `M` is the draft width the parent asks for; `IPG` is the inputs-per-group the
# `_wide` body is instantiated at, and `NA` is the set of `_wide` widths the
# body can reach at run time.
LIVE_WIDE_DISPATCH = {
    2: {"callee": "qmv_fast_crossrow_affine4_g64<T, 2>", "na": ()},
    3: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 3, 3, true>", "na": (3,)},
    4: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 4, 4, true>", "na": (4,)},
    5: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 5, 5, true>", "na": (5,)},
    6: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>", "na": (3,)},
    7: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>", "na": (4, 3)},
    8: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>", "na": (4,)},
    9: {"callee": "qmv_fast_crossrow_affine4_g64_m<T, 9, 3, true>", "na": (3,)},
}

STD_ARGS = """    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]"""

STD_CALL = """      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);"""

ADJUST_WRAPPER = """
[[kernel]] void e131_iso_adjust(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& x_batch_ndims [[buffer(6)]],
    const constant int* x_shape [[buffer(7)]],
    const constant int64_t* x_strides [[buffer(8)]],
    const constant int& w_batch_ndims [[buffer(9)]],
    const constant int* w_shape [[buffer(10)]],
    const constant int64_t* w_strides [[buffer(11)]],
    const constant int64_t* s_strides [[buffer(12)]],
    const constant int64_t* b_strides [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<bfloat16_t>(
      x, w, scales, biases, y, out_vec_size * M, x_batch_ndims, x_shape,
      x_strides, w_batch_ndims, w_shape, w_strides, s_strides, b_strides, tid);
  // The offsets are the whole body, so they have to be observed or the
  // optimizer deletes the census subject.
  y[0] = x[0] + scales[0] + biases[0] + bfloat16_t(w[0] & 1u);
}
"""

ISO_TEMPLATE = """
[[kernel]] void e131_iso_%(name)s(
%(args)s) {
%(body)s
}
"""


def iso_kernels() -> dict[str, str]:
    """One wrapper entry point per inlinable body, keyed by census cell name."""
    cells: dict[str, str] = {}
    for m in range(2, 10):
        cells["xr%d" % m] = ISO_TEMPLATE % {
            "name": "xr%d" % m, "args": STD_ARGS,
            "body": "  qmv_fast_crossrow_affine4_g64<bfloat16_t, %d>(\n%s"
                    % (m, STD_CALL)}
    for m, entry in LIVE_WIDE_DISPATCH.items():
        if not entry["na"]:
            continue
        callee = entry["callee"].replace("T", "bfloat16_t")
        cells["m%d" % m] = ISO_TEMPLATE % {
            "name": "m%d" % m, "args": STD_ARGS,
            "body": "  %s(\n%s" % (callee, STD_CALL)}
    for na in range(2, 6):
        cells["wide%d" % na] = ISO_TEMPLATE % {
            "name": "wide%d" % na, "args": STD_ARGS,
            "body": ("  const int first_m = int(tid.x) * %d;\n"
                     "  const int out_row = int(tid.y) * 8 + int(simd_gid) * 4;\n"
                     "  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %d, true>(\n"
                     "      w, scales, biases, x, y, in_vec_size, out_vec_size,\n"
                     "      first_m, out_row, simd_lid);" % (na, na))}
    cells["single2"] = ISO_TEMPLATE % {
        "name": "single2", "args": STD_ARGS,
        "body": "  qmv_fast_singlerow_affine2_g64<bfloat16_t>(\n%s" % STD_CALL}
    for bits in (2, 4):
        cells["impl%d" % bits] = ISO_TEMPLATE % {
            "name": "impl%d" % bits, "args": STD_ARGS,
            "body": "  qmv_fast_impl<bfloat16_t, 64, %d>(\n%s" % (bits, STD_CALL)}
    cells["adjust"] = ADJUST_WRAPPER
    return cells


def host_name(cell: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", cell).strip("_")


def instantiation(cell: str) -> str:
    return ('\ntemplate [[host_name("%s")]] [[kernel]] decltype(%s) %s;\n'
            % (host_name(cell), cell, cell))


def emit_preamble(scratch: pathlib.Path) -> str:
    """The runtime-effective JIT string with the trailing instantiation removed."""
    path = scratch / "base_raw.metal"
    subprocess.run(
        [sys.executable, str(ROOT / "research/jit_string_compile.py"),
         "--emit", str(path), "--", SCORED_CELL],
        check=True, cwd=str(ROOT), capture_output=True)
    text = path.read_text()
    tail = instantiation(SCORED_CELL)
    if not text.endswith(tail):
        raise SystemExit("e131_census: emitted base does not end in the "
                         "expected instantiation; the emitter moved")
    return text[:-len(tail)]


def expect(text: str, needle: str, count: int, label: str) -> None:
    seen = text.count(needle)
    if seen != count:
        raise SystemExit("e131_census: %s matched %d times, expected %d. The "
                         "base moved; re-read the kernel before censusing it."
                         % (label, seen, count))


def wide_span(text: str) -> tuple[int, int]:
    open_at = text.index(WIDE_OPEN)
    return open_at, text.index(WIDE_CLOSE, open_at)


def drop_in_wide(text: str, block: str, label: str) -> str:
    start, end = wide_span(text)
    inner = text[start:end]
    expect(inner, block, 1, "wide-branch %s" % label)
    return text[:start] + inner.replace(block, "") + text[end:]


def drop_everywhere(text: str, block: str, count: int, label: str) -> str:
    expect(text, block, count, label)
    return text.replace(block, "")


# Counterfactual entry points. Each is (cell, patch), and each patch asserts its
# own site count, so a base that moves fails loudly instead of quietly censusing
# the shipped kernel twice under two names.
def cf_none(text: str) -> str:
    return text


def cf_no_m5(text: str) -> str:
    return drop_in_wide(text, CASE5_M, "case 5")


def cf_no_m5_no_wide_pair(text: str) -> str:
    return drop_in_wide(cf_no_m5(text), CASE2_PAIR, "case 2")


def cf_no_m5_no_pair(text: str) -> str:
    return drop_everywhere(cf_no_m5(text), CASE2_PAIR, 2, "both case 2 blocks")


def cf_no_narrow(text: str) -> str:
    """Delete the whole `out_vec_size < 4096` switch, keeping M=5."""
    start, end = wide_span(text)
    close = text.index("\n    }\n  }\n", end)
    return text[:end] + "    } else {\n" + text[close:]


def cf_no_m5_no_narrow(text: str) -> str:
    return cf_no_narrow(cf_no_m5(text))


VARIANTS = (
    ("entry_shipped", SCORED_CELL, cf_none),
    ("entry_no_m5", SCORED_CELL, cf_no_m5),
    ("entry_no_m5_no_wide_pair", SCORED_CELL, cf_no_m5_no_wide_pair),
    ("entry_no_m5_no_pair", SCORED_CELL, cf_no_m5_no_pair),
    ("entry_no_m5_no_narrow", SCORED_CELL, cf_no_m5_no_narrow),
    ("entry_bits2", BITS2_CELL, cf_none),
    ("entry_batched", BATCHED_CELL, cf_none),
)


def census_library(source: str, workdir: pathlib.Path, tag: str,
                   select) -> dict[str, dict]:
    lib = agx.build_metallib(source, workdir / tag)
    rows: dict[str, dict] = {}
    for arch in ARCHES:
        for kernel, record in agx.translate(lib, arch, workdir / tag,
                                            select=select).items():
            rows.setdefault(kernel, {})[arch] = {
                "registers": record.get("registers"),
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "text_sha8": record.get("text_sha8"),
                "simdgroups": simdgroups(record.get("registers"), arch),
            }
    return rows


# The bodies each entry-point variant can still reach at run time. A body that
# only survives in a branch the front end instantiates but the optimizer proves
# unreachable is NOT listed: it cannot hold a register.
XR_ALL = tuple("xr%d" % n for n in range(2, 10))
M_ALL = tuple("m%d" % m for m in (3, 4, 5, 6, 7, 8, 9))
LIVE_BODIES = {
    "entry_shipped": XR_ALL + M_ALL + ("impl4",),
    "entry_no_m5": XR_ALL + tuple(m for m in M_ALL if m != "m5") + ("impl4",),
    "entry_no_m5_no_wide_pair":
        XR_ALL + tuple(m for m in M_ALL if m != "m5") + ("impl4",),
    "entry_no_m5_no_pair":
        tuple(x for x in XR_ALL if x != "xr2")
        + tuple(m for m in M_ALL if m != "m5") + ("impl4",),
    "entry_no_m5_no_narrow":
        ("xr2",) + tuple(m for m in M_ALL if m != "m5") + ("impl4",),
    "entry_bits2": ("single2", "impl2"),
    "entry_batched": ("impl4", "adjust"),
}


def floor_law(bodies: dict, entries: dict) -> dict:
    """Is each entry point allocated for the widest body it can still reach?"""
    checks: dict[str, dict] = {}
    for name, live in LIVE_BODIES.items():
        for arch in ARCHES:
            widest = max(live, key=lambda b: bodies[b][arch]["registers"])
            entry = entries[name][arch]["registers"]
            checks["%s/%s" % (name, arch)] = {
                "entry_registers": entry,
                "widest_live_body": widest,
                "widest_live_body_registers": bodies[widest][arch]["registers"],
                "entry_simdgroups": entries[name][arch]["simdgroups"],
                "holds": entry == bodies[widest][arch]["registers"],
            }
    return checks


def answers(bodies: dict, entries: dict, law: dict) -> dict:
    """The three questions rung 0 has to answer, in machine-readable form."""
    out: dict = {}
    for tag, name in (("q1_current", "entry_shipped"),
                      ("q2_after_na5_removal", "entry_no_m5"),
                      ("q3_after_na5_and_pair_removal", "entry_no_m5_no_pair")):
        for arch in ARCHES:
            key = "%s/%s" % (tag, arch)
            row = entries[name][arch]
            check = law["%s/%s" % (name, arch)]
            out[key] = {
                "registers": row["registers"],
                "simdgroups": row["simdgroups"],
                "set_by": check["widest_live_body"],
                "text_bytes": row["text_bytes"],
            }
    shipped = entries["entry_shipped"][agx.RANKED_ARCH]
    no_m5 = entries["entry_no_m5"][agx.RANKED_ARCH]
    out["q2_g17s_residency_gain_pct"] = round(
        100.0 * no_m5["simdgroups"] / shipped["simdgroups"] - 100.0, 3)
    out["q2_binding_registers_g17s"] = no_m5["registers"]
    out["q2_alphonse_c_a_alive"] = no_m5["registers"] < 94
    out["floor_law_holds_everywhere"] = all(c["holds"] for c in law.values())
    return out


def toolchain() -> str:
    done = subprocess.run(["xcrun", "metal", "--version"], capture_output=True,
                          text=True, check=True)
    lines = [l for l in (done.stdout + done.stderr).splitlines() if l.strip()]
    return lines[0] if lines else "unknown"


def cliff(arch: str, registers: int) -> dict:
    """Where a register count sits between two floor-division cliffs."""
    budget = SIMDGROUP_BUDGET[arch]
    groups = budget // registers
    return {
        "registers": registers,
        "simdgroups": groups,
        # The largest register count that still yields `groups + 1`, and the
        # reduction needed to reach it.
        "registers_for_next_step": budget // (groups + 1),
        "registers_to_gain_one": registers - budget // (groups + 1),
        "registers_to_lose_one": budget // groups + 1 - registers,
        "gain_pct": 100.0 * (groups + 1) / groups - 100.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="research/e131-artifacts")
    ap.add_argument("--scratch", default="")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with tempfile.TemporaryDirectory(dir=args.scratch or None) as tmp:
        scratch = pathlib.Path(tmp)
        preamble = emit_preamble(scratch)

        iso = iso_kernels()
        iso_source = preamble + "".join(iso.values())
        iso_names = {"e131_iso_%s" % name: name for name in iso}
        print("iso bodies: %d" % len(iso))
        iso_rows = census_library(
            iso_source, scratch, "iso", lambda n: n in iso_names)
        bodies = {iso_names[k]: v for k, v in iso_rows.items()}

        entries: dict[str, dict] = {}
        digests: dict[str, str] = {}
        for name, cell, patch in VARIANTS:
            text = patch(preamble) + instantiation(cell)
            digest = hashlib.sha256(text.encode()).hexdigest()[:12]
            if digest in digests.values() and name != "entry_shipped":
                same = [k for k, v in digests.items() if v == digest]
                raise SystemExit("e131_census: %s is byte-identical to %s"
                                 % (name, same))
            digests[name] = digest
            (scratch / ("%s.metal" % name)).write_text(text)
            want = host_name(cell)
            rows = census_library(text, scratch, name, lambda n: n == want)
            if want not in rows:
                raise SystemExit("e131_census: %s did not emit %s" % (name, want))
            entries[name] = rows[want]
            print("%-26s sha=%s  %s" % (
                name, digest,
                "  ".join("%s %s regs / %s sg" % (
                    arch.replace("applegpu_", ""),
                    entries[name][arch]["registers"],
                    entries[name][arch]["simdgroups"]) for arch in ARCHES)))

    payload = {
        "experiment": "E131",
        "rung": 0,
        "harness": "local",
        "timing_valid": False,
        "gpu_used": False,
        "official_or_ranked_score": False,
        "base_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, check=True).stdout.strip(),
        "toolchain": toolchain(),
        "simdgroup_budget": SIMDGROUP_BUDGET,
        "scored_cell": SCORED_CELL,
        "live_wide_dispatch": {str(k): v for k, v in LIVE_WIDE_DISPATCH.items()},
        "live_bodies": {k: list(v) for k, v in LIVE_BODIES.items()},
        "law": floor_law(bodies, entries),
        "answers": answers(bodies, entries, floor_law(bodies, entries)),
        "bodies": bodies,
        "entries": entries,
        "entry_digests": digests,
        "cliffs": {
            arch: {name: cliff(arch, row[arch]["registers"])
                   for name, row in list(bodies.items()) + list(entries.items())}
            for arch in ARCHES},
        "wall_seconds": round(time.time() - started, 1),
    }
    out = outdir / "rung0-census.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote %s in %.1f s" % (out, payload["wall_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
