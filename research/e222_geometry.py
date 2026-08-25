#!/usr/bin/env python3
"""E222 stage 0: geometry enumeration and compile-only screen for dequant-value
sharing across the two token groups of a G>=2 staged QMV entry.

E219 (FINDING 573) measured that the second token group of a G = 2 entry
re-pays the whole `b` term. `b` is the per-weight-byte work that does not scale
with NA: the packed load, the nibble extraction and the nibble-to-float
conversion. E220 (FINDING 575) closed the MSL integer path, so the only
remaining dequant lever is to stop RE-COMPUTING those values in the second
group.

Two families can share a computed value:

  fused    one thread owns both token groups for its weight rows, the value
           stays in a register and feeds two fma chains. Accumulators become
           rows * (NA_g0 + NA_g1).
  tgshare  the threadgroup covers both token groups over the same weight rows.
           Each simdgroup extracts its share of the tile's (row, i) units,
           writes the values to threadgroup memory, and every simdgroup then
           reads the whole tile for its own token group.

This screen spends zero GPU seconds. It answers three questions before any
timed leg:

  1. Legality. Real AGX register and spill numbers for the local `g16s`
     generation and the ranked `g17s` generation, plus the threadgroup byte
     count against the 32 KiB limit.
  2. The rows ladder. FINDING 574 is INFERRED from a rows = 4 fit; this census
     compiles rows in {1, 2, 4, 8} x NA in [2, 9] and measures it.
  3. The reachable share of `b`. The FINDING 573 ceiling of 11.614 ms/round
     prices PERFECT reuse, where the consuming group pays nothing. The AIR
     census counts what each geometry's consumer actually pays, so the ceiling
     can be scaled by the measured op ratio instead of assumed.

  python3 research/e222_geometry.py [--out PATH]
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

import agx_crossarch  # noqa: E402
import e220_census  # noqa: E402

REPO = HERE.parent
ARTIFACTS = HERE / "e222-artifacts"
ARCHES = ("applegpu_g16s", "applegpu_g17s")

# FINDING 573 final census, and the measured floors E216/E217 established.
CEILING_POOLED_MS = 11.614
CEILING_POOLED_CI95U_MS = 16.3
PAIRING_FLOOR_POOLED_MS = -1.323
RAW_BYTE_STAGING_FLOOR_POOLED_MS = -5.866
PROMOTION_POOLED_MS = 1.0
STOP_POOLED_MS = 0.3

# The threadgroup memory a single threadgroup may declare.
TG_LIMIT_BYTES = 32 * 1024

# The staged plan's G = 2 widths, as (m, ipg) -> (NA of group 0, NA of group 1).
G2_SPLITS = {6: (3, 3), 7: (4, 3), 8: (4, 4), 9: (5, 4)}

PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

# --- family 1: the shipped body with `rows_per_simd` opened up ---------------

ROWS_ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_out_row =
        int(qmv_tid.y) * (2 * {rows}) + int(qmv_sgid) * {rows};
    qwen_e222_rows<{na}, {rows}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x) * {na}, qmv_out_row, qmv_lid);
}}
"""

# --- family 2: register reuse, one thread owns both token groups -------------

FUSED_TEMPLATE_PATH = HERE / "e222_fused.msl"

# Single source of truth: Tests/MLXFastTests/E222ValueSharingTests.swift JITs
# the same text for its timed arms, so the screened geometry and the timed
# geometry cannot diverge.
FUSED_HEADER = FUSED_TEMPLATE_PATH.read_text()

FUSED_ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    device atomic_uint* census [[buffer(9)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_out_row =
        int(qmv_tid.y) * ({sg} * {rows}) + int(qmv_sgid) * {rows};
    qwen_e222_fused<{na0}, {na1}, {rows}, {lazy}, {late_sb}, {seq},
                    {table}, false>(
        w, scales, biases, x, xsums, y, census,
        qmv_k, qmv_n, qmv_stride, qmv_out_row, qmv_lid);
}}
"""

# --- family 3: threadgroup-memory value staging ------------------------------
#
# The tile is `ROWS_TG` weight rows. Its `(row, i)` units are split evenly over
# the `SG` simdgroups, so every simdgroup extracts the same number of units and
# then reads the whole tile back for its own token group. Reading back its own
# units costs one extra vector load per unit and is what keeps the `i` order,
# and therefore each output's fma chain, identical to the shipped body.
#
# `PHASES` splits the k-block's four `i` values into equal staging phases. Two
# phases halve the threadgroup footprint and double the barrier count.

TGSHARE_HEADER = """
    template <int NA, int ROWS_TG, int SG, int PHASES, typename VT,
              bool PRODUCE, bool USE_TABLE>
    inline void qwen_e222_tgshare(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device float* xsums,
        device bfloat16_t* y,
        threadgroup VT* vals,
        const int in_vec_size,
        const int out_vec_size,
        const int sums_stride,
        int first_m,
        int tg_row0,
        int row_slot,
        uint simd_lid,
        uint sgid
    ) {
        typedef vec<float, NA> VF;
        constexpr int rows_per_simd = 4;
        constexpr int values_per_thread = 16;
        constexpr int block_size = values_per_thread * 32;
        constexpr int bytes_per_lane = 8;
        constexpr int i_per_phase = 4 / PHASES;
        constexpr int units_per_sg = (ROWS_TG * i_per_phase) / SG;
        const int in_vec_size_w = in_vec_size / 2;
        const int in_vec_size_g = in_vec_size / 64;

        VF acc[rows_per_simd];
        for (int r = 0; r < rows_per_simd; r++) {
            acc[r] = VF(0.0f);
        }

        for (int k = 0; k < in_vec_size; k += block_size) {
            thread float scale_local[rows_per_simd];
            thread float bias_local[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                const int row = tg_row0 + row_slot + r;
                const int group_index =
                    row * in_vec_size_g + k / 64 + int(simd_lid) / 4;
                scale_local[r] = scales[group_index];
                bias_local[r] = biases[group_index];
            }

            VF sums = VF(0.0f);
            if (USE_TABLE) {
                const device float* st =
                    xsums + ((k / block_size) * 32 + int(simd_lid)) *
                    sums_stride + first_m;
                for (int m = 0; m < NA; m++) {
                    sums[m] = st[m];
                }
            }
            VF partial[rows_per_simd];
            for (int r = 0; r < rows_per_simd; r++) {
                partial[r] = VF(0.0f);
            }

            for (int phase = 0; phase < PHASES; phase++) {
                for (int u = 0; PRODUCE && u < units_per_sg; u++) {
                    const int unit = int(sgid) * units_per_sg + u;
                    const int slot = unit / i_per_phase;
                    const int i = phase * i_per_phase + unit % i_per_phase;
                    const device uint16_t* ws =
                        reinterpret_cast<const device uint16_t*>(
                            reinterpret_cast<const device uint8_t*>(w) +
                            (tg_row0 + slot) * in_vec_size_w + k / 2 +
                            simd_lid * bytes_per_lane);
                    const uint16_t p = ws[i];
                    const int at =
                        ((slot * i_per_phase + unit % i_per_phase) * 32 +
                         int(simd_lid)) * 4;
                    vals[at + 0] = static_cast<VT>(p & 0x000f);
                    vals[at + 1] = static_cast<VT>((p >> 4) & 0x000f);
                    vals[at + 2] = static_cast<VT>((p >> 8) & 0x000f);
                    vals[at + 3] = static_cast<VT>((p >> 12) & 0x000f);
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int j = 0; j < i_per_phase; j++) {
                    const int i = phase * i_per_phase + j;
                    VF a0, a1, a2, a3;
                    for (int m = 0; m < NA; m++) {
                        const device bfloat16_t* xm =
                            x + (first_m + m) * in_vec_size + k +
                            simd_lid * values_per_thread + 4 * i;
                        const vec<bfloat16_t, 4> xv =
                            *reinterpret_cast<
                                const device vec<bfloat16_t, 4>*>(xm);
                        a0[m] = static_cast<float>(xv[0]);
                        a1[m] = static_cast<float>(xv[1]);
                        a2[m] = static_cast<float>(xv[2]);
                        a3[m] = static_cast<float>(xv[3]);
                        if (!USE_TABLE) {
                            sums[m] += xv[0] + xv[1] + xv[2] + xv[3];
                        }
                    }
                    for (int r = 0; r < rows_per_simd; r++) {
                        const int at =
                            (((row_slot + r) * i_per_phase + j) * 32 +
                             int(simd_lid)) * 4;
                        const vec<VT, 4> v =
                            *reinterpret_cast<const threadgroup vec<VT, 4>*>(
                                vals + at);
                        partial[r] += (a0 * static_cast<float>(v[0]) +
                                       a1 * static_cast<float>(v[1]) +
                                       a2 * static_cast<float>(v[2]) +
                                       a3 * static_cast<float>(v[3]));
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (int r = 0; r < rows_per_simd; r++) {
                acc[r] += scale_local[r] * partial[r] + sums * bias_local[r];
            }
        }

        for (int r = 0; r < rows_per_simd; r++) {
            for (int m = 0; m < NA; m++) {
                const float reduced = simd_sum(acc[r][m]);
                if (simd_lid == 0) {
                    y[(first_m + m) * out_vec_size +
                      tg_row0 + row_slot + r] =
                        static_cast<bfloat16_t>(reduced);
                }
            }
        }
    }
"""

TGSHARE_ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    threadgroup {vt} qmv_vals[{tg_values}];
    constexpr int slots_per_group = {rows_tg} / 4;
    const int qmv_row_slot = int(qmv_sgid) % slots_per_group * 4;
    const int qmv_group = int(qmv_sgid) / slots_per_group;
    qwen_e222_tgshare<{na}, {rows_tg}, {sg}, {phases}, {vt}, {produce}, false>(
        w, scales, biases, x, xsums, y, qmv_vals,
        qmv_k, qmv_n, qmv_stride,
        qmv_group * {na}, int(qmv_tid.y) * {rows_tg}, qmv_row_slot,
        qmv_lid, qmv_sgid);
}}
"""

VT_BYTES = {"float": 4, "bfloat16_t": 2, "uchar": 1}


def rows_header(live: str) -> str:
    """The shipped body with `rows_per_simd` promoted to a template argument.

    The substitution is textual, so the `R = 4` instance is the shipped source
    and serves as the control for every other rung of the ladder.
    """
    header = live.replace(
        "template <int NA, bool USE_TABLE>\n    inline void qwen_e120_qmv_wide(",
        "template <int NA, int R, bool USE_TABLE>\n"
        "    inline void qwen_e222_rows(",
    )
    header = header.replace(
        "constexpr int rows_per_simd = 4;", "constexpr int rows_per_simd = R;")
    if "qwen_e222_rows" not in header or "rows_per_simd = R" not in header:
        raise SystemExit("the shipped body no longer matches the rows rewrite")
    # Only the wide template is needed; the `qwen_e120_qmv_m` dispatcher below
    # it still refers to the original symbol.
    return header.split("template <int M, int IPG, bool USE_TABLE>")[0]


def geometries() -> list[dict]:
    """Every geometry this screen prices, with its static properties."""
    out: list[dict] = []

    for rows in (1, 2, 4, 8):
        for na in range(2, 10):
            for table in (False, True):
                out.append({
                    "family": "rows",
                    "name": (f"e222_rows{rows}_na{na}"
                             f"_{'tbl' if table else 'rec'}"),
                    "rows": rows,
                    "na": na,
                    "table": table,
                    "simdgroups": 2,
                    "rows_per_tg": 2 * rows,
                    "tg_bytes": 0,
                    "barriers_per_k_block": 0,
                    "role": "ladder" if rows != 4 else "control",
                })

    # A fused threadgroup always spans the shipped eight weight rows, so the
    # simdgroup count follows `rows` and no E216 coalescing loss applies.
    for m, (na0, na1) in G2_SPLITS.items():
        for rows in (1, 2, 4):
            for lazy, late_sb, tag in ((False, False, "base"),
                                       (True, False, "lazyw"),
                                       (False, True, "latesb"),
                                       (True, True, "both")):
                for seq in (False, True):
                    for table in (False, True):
                        out.append({
                            "family": "fused",
                            "name": (f"e222_fused_m{m}_r{rows}_{tag}"
                                     f"{'_seq' if seq else ''}"
                                     f"_{'tbl' if table else 'rec'}"),
                            "rows": rows,
                            "na0": na0,
                            "na1": na1,
                            "m": m,
                            "lazy": lazy,
                            "late_sb": late_sb,
                            "seq": seq,
                            "table": table,
                            "simdgroups": 8 // rows,
                            "rows_per_tg": 8,
                            "tg_bytes": 0,
                            "barriers_per_k_block": 0,
                            "role": "candidate1_seq" if seq else "candidate1",
                        })

    nas = sorted({na for pair in G2_SPLITS.values() for na in pair})
    for vt in ("float", "bfloat16_t", "uchar"):
        for rows_tg, sg, phases in ((8, 4, 1), (8, 4, 2), (4, 2, 1)):
            values = rows_tg * (4 // phases) * 32 * 4
            for na in nas:
                out.append({
                    "family": "tgshare",
                    "name": (f"e222_tg_{vt.replace('16_t', '')}_r{rows_tg}"
                             f"_sg{sg}_p{phases}_na{na}"),
                    "vt": vt,
                    "rows_tg": rows_tg,
                    "sg": sg,
                    "phases": phases,
                    "na": na,
                    "produce": True,
                    "tg_values": values,
                    "tg_bytes": values * VT_BYTES[vt],
                    "barriers_per_k_block": 2 * phases,
                    "rows_per_tg": rows_tg,
                    "simdgroups": sg,
                    "role": "candidate2",
                })

    # The ideal-consumer anchor: the same body with the producer removed, so
    # its instruction count is the common work plus the threadgroup loads. It
    # is not implementable; it exists to measure `b` by subtraction.
    for na in nas:
        out.append({
            "family": "tgshare",
            "name": f"e222_anchor_na{na}",
            "vt": "float",
            "rows_tg": 8,
            "sg": 4,
            "phases": 1,
            "na": na,
            "produce": False,
            "tg_values": 8 * 4 * 32 * 4,
            "tg_bytes": 8 * 4 * 32 * 4 * VT_BYTES["float"],
            "barriers_per_k_block": 2,
            "rows_per_tg": 8,
            "simdgroups": 4,
            "role": "anchor",
        })
    return out


def entry_for(geo: dict) -> str:
    if geo["family"] == "rows":
        return ROWS_ENTRY.format(
            name=geo["name"], na=geo["na"], rows=geo["rows"],
            table="true" if geo["table"] else "false")
    if geo["family"] == "fused":
        return FUSED_ENTRY.format(
            name=geo["name"], na0=geo["na0"], na1=geo["na1"],
            rows=geo["rows"], sg=geo["simdgroups"],
            lazy="true" if geo["lazy"] else "false",
            late_sb="true" if geo["late_sb"] else "false",
            seq="true" if geo["seq"] else "false",
            table="true" if geo["table"] else "false")
    return TGSHARE_ENTRY.format(
        name=geo["name"], na=geo["na"], rows_tg=geo["rows_tg"], sg=geo["sg"],
        phases=geo["phases"], vt=geo["vt"], tg_values=geo["tg_values"],
        produce="true" if geo["produce"] else "false")


def source_for(geos: list[dict], live: str) -> str:
    header = PREAMBLE + rows_header(live) + FUSED_HEADER + TGSHARE_HEADER
    return header + "\n" + "\n".join(entry_for(geo) for geo in geos)


def compile_one(geo: dict, live: str) -> dict:
    """Compile one geometry alone: AIR opcodes plus real AGX registers.

    One kernel per translation unit keeps the register number attributable and
    keeps a geometry that fails to build from hiding the rest of the table.
    """
    source = source_for([geo], live)
    record: dict = {"builds": True}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        try:
            air = e220_census.emit_air(source, workdir)
        except subprocess.CalledProcessError as exc:
            return {"builds": False,
                    "error": exc.stderr.decode()[-1500:]}
        bodies = e220_census.functions(air)
        # The entry point is a stub; the geometry's k loop lives in the
        # inlineable template body, which is the largest function in the unit.
        symbol, body = max(bodies.items(), key=lambda kv: len(kv[1]))
        if len(body) < 50:
            return {"builds": False, "error": "no template body found in AIR"}
        record["air_symbol"] = symbol[:60]
        record["air"] = e220_census.op_stats(body)
        record["air"]["threadgroup_load"] = sum(
            1 for line in body
            if re.search(r"=\s*load\s.*addrspace\(3\)", line))
        record["air"]["threadgroup_store"] = sum(
            1 for line in body
            if re.match(r"\s*store\s", line) and "addrspace(3)" in line)
        record["air"]["barrier_calls"] = sum(
            1 for line in body if "barrier" in line)
        try:
            lib = agx_crossarch.build_metallib(source, workdir)
        except subprocess.CalledProcessError as exc:
            record["builds"] = False
            record["error"] = exc.stderr.decode()[-1500:]
            return record
        for arch in ARCHES:
            found = agx_crossarch.translate(lib, arch, workdir)
            record[arch.replace("applegpu_", "")] = found.get(geo["name"], {})
    return record


def unit_cost_model() -> dict:
    """Price the `b` work per `(row, i)` unit, per token-group pair.

    `b` is the per-weight-byte work: the packed load, the integer extraction
    and the nibble-to-float conversion. One `(row, i)` unit is one packed
    `uint16` and the four values inside it. The AIR loops stay rolled, so the
    whole-body instruction count cannot be differenced between geometries;
    these are per-unit counts read from the source and cross-checked against
    the dequant-block census the compiler emits.

    Shipped: the unit is paid twice, once per token group.
    Ideal:   the unit is paid once and both groups read it for free.
    fused:   the unit is paid once and both chains read it from a register,
             which is the ideal.
    tgshare: the unit is paid once with an added vector store, and each token
             group then pays a vector load, plus a conversion back to float
             when the stored type is not `float`.
    """
    unit = {"device_load": 1, "int_ops": 3, "convert": 4}
    base_unit = sum(unit.values())
    shipped = 2 * base_unit
    ideal = base_unit
    model = {
        "ops_per_unit_shipped_two_groups": shipped,
        "ops_per_unit_ideal_two_groups": ideal,
        "shipped_unit_breakdown": unit,
        "families": {
            "fused": {
                "ops_per_unit_two_groups": ideal,
                "share_of_ideal": 1.0,
                "why": ("the value never leaves the register that produced "
                        "it, so the second chain adds no access cost"),
            },
            # SEQ runs the two groups as two phases of each `i` step, so only
            # one group's activation vectors are live at a time. The packed
            # word is loaded once either way, but the integer extraction and
            # the nibble-to-float conversion appear twice in the source. This
            # entry prices the PESSIMISTIC case, in which the compiler
            # rematerialises both instead of keeping four floats live per row.
            # The AIR census decides which case a build actually is.
            "fused:seq": {
                "ops_per_unit_two_groups": (
                    unit["device_load"] + 2 * unit["int_ops"]
                    + 2 * unit["convert"]),
                "share_of_ideal": round(
                    (shipped
                     - (unit["device_load"] + 2 * unit["int_ops"]
                        + 2 * unit["convert"]))
                    / (shipped - ideal), 4),
                "why": ("register relief bought by re-deriving the nibbles "
                        "for the second group; a lower bound on the shared "
                        "work the geometry keeps"),
            },
        },
    }
    for vt, size in VT_BYTES.items():
        convert_back = 0 if vt == "float" else 4
        producer = base_unit + 1
        consumer = 1 + convert_back
        total = producer + 2 * consumer
        model["families"][f"tgshare:{vt}"] = {
            "ops_per_unit_two_groups": total,
            "producer_ops_per_unit": producer,
            "consumer_ops_per_unit": consumer,
            "storage_bytes_per_value": size,
            "share_of_ideal": round((shipped - total) / (shipped - ideal), 4),
            "why": ("only float storage avoids paying the nibble-to-float "
                    "conversion once per token group"),
        }
    for name, entry in model["families"].items():
        entry["predicted_pooled_ms"] = round(
            CEILING_POOLED_MS * entry["share_of_ideal"], 3)
    return model


def verdict(geo: dict, record: dict, model: dict) -> dict:
    """Legality and the cost floor a geometry must clear."""
    out: dict = {"legal": True, "reasons": []}
    if not record.get("builds", False):
        out["legal"] = False
        out["reasons"].append("does not compile")
        return out
    for arch in ("g16s", "g17s"):
        entry = record.get(arch) or {}
        spill = entry.get("spill_bytes") or 0
        if spill:
            out["legal"] = False
            out["reasons"].append(f"{arch} spills {spill} bytes")
    if geo["tg_bytes"] > TG_LIMIT_BYTES:
        out["legal"] = False
        out["reasons"].append(
            f"threadgroup memory {geo['tg_bytes']} > {TG_LIMIT_BYTES}")
    if geo["family"] == "rows" or (
            geo["family"] == "tgshare" and not geo["produce"]):
        return out
    key = ("fused" if geo["family"] == "fused"
           else f"tgshare:{geo['vt']}")
    priced = model["families"][key]["predicted_pooled_ms"]
    if geo["rows_per_tg"] < 8:
        priced += PAIRING_FLOOR_POOLED_MS
        out["reasons"].append(
            "tile is narrower than the shipped 8 rows, so the E216 "
            f"coalescing loss of {PAIRING_FLOOR_POOLED_MS} applies")
    out["predicted_pooled_ms"] = round(priced, 3)
    out["clears_stop_rule"] = priced >= STOP_POOLED_MS
    out["clears_promotion"] = priced >= PROMOTION_POOLED_MS
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ARTIFACTS / "e222-geometry.json"))
    args = parser.parse_args()

    live = e220_census.live_text("qwen35E120QMVHeader")
    geos = geometries()
    model = unit_cost_model()

    result = {
        "experiment": "e222-stage0-geometry",
        "harness": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "source": "Qwen35.swift qwen35E120QMVHeader, read live",
        "arches": list(ARCHES),
        "constants": {
            "ceiling_pooled_ms": CEILING_POOLED_MS,
            "ceiling_pooled_ci95u_ms": CEILING_POOLED_CI95U_MS,
            "pairing_floor_pooled_ms": PAIRING_FLOOR_POOLED_MS,
            "raw_byte_staging_floor_pooled_ms": RAW_BYTE_STAGING_FLOOR_POOLED_MS,
            "promotion_pooled_ms": PROMOTION_POOLED_MS,
            "stop_pooled_ms": STOP_POOLED_MS,
            "threadgroup_limit_bytes": TG_LIMIT_BYTES,
            "g2_splits": {str(k): list(v) for k, v in G2_SPLITS.items()},
        },
        "unit_cost_model": model,
        "geometries": {},
    }

    for geo in geos:
        record = compile_one(geo, live)
        result["geometries"][geo["name"]] = {
            "geometry": geo,
            "compile": record,
            "verdict": verdict(geo, record, model),
        }
        entry = record.get("g17s") or {}
        print(f"{geo['name']:<34} builds={record.get('builds')} "
              f"g17s_regs={entry.get('registers')} "
              f"spill={entry.get('spill_bytes')} "
              f"tg_bytes={geo['tg_bytes']}")

    result["survivors"] = sorted(
        (name for name, entry in result["geometries"].items()
         if entry["verdict"]["legal"]
         and entry["verdict"].get("clears_stop_rule")),
        key=lambda name:
        -result["geometries"][name]["verdict"]["predicted_pooled_ms"])
    result["survivors_ranked_only"] = sorted(
        name for name, entry in result["geometries"].items()
        if entry["verdict"].get("clears_stop_rule")
        and not (entry["compile"].get("g17s") or {}).get("spill_bytes")
        and entry["compile"].get("builds"))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
