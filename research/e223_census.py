#!/usr/bin/env python3
"""E223 stage 0: attribute the per-column inner-loop cost of the scored QMV.

E221 separated two ways of buying the same activation byte: buying it with
extra simdgroups costs 0.05-0.30 us/MB, buying it by widening `NA` costs
0.5-2.3 us/MB (FINDING 576). The excess is per-column instruction work, not
byte volume. This census asks WHICH instruction family carries the `NA` slope,
before any GPU second is spent.

Three witnesses over the same live `qwen35E120QMVHeader` text:

  air        `xcrun metal -O2 -S -emit-llvm`, then a real loop-nest walk. Every
             basic block is priced by the product of its enclosing loop trip
             counts, so the census reports OPS PER K-BLOCK PER LANE per family,
             not static instruction counts. Architecture independent, so it is
             a proxy for the AGX stream, never a measurement of it.
  agx        `xcrun metal-tt` for `applegpu_g16s` (local) and `applegpu_g17s`
             (ranked), through `agx_crossarch`: registers, spill bytes and real
             machine-code size. RULE 407: g17s decides legality, g16s decides
             local-timing fidelity.
  bytes/insn AGX machine code is variable length, so `text_bytes` alone cannot
             be read as an instruction count. Synthetic probes of known op
             count calibrate bytes per instruction on each architecture, which
             turns the `d(text_bytes)/d(NA)` slope into a per-column
             instruction estimate.

The families are then reconciled against E221's measured `NA` slope per cell
(`research/e221-artifacts/e221-rows-report.json`), so the report states which
family can explain the measured slope and what an access-path change can
address at most.

Zero GPU seconds. Nothing here is timed, scored or shipped.

  python3 research/e223_census.py
  python3 research/e223_census.py --skip-agx      # AIR witness only
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agx_crossarch  # noqa: E402
import e223_variants  # noqa: E402

REPO = HERE.parent
QWEN35 = REPO / "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
ARTIFACTS = HERE / "e223-artifacts"
E221_REPORT = HERE / "e221-artifacts/e221-rows-report.json"

ARCHES = ("applegpu_g16s", "applegpu_g17s")
NAS = tuple(range(2, 10))

# The three cells E221 timed, as `(k, n, invocations per round)`. `mlp.down` is
# reported per cell and never pooled (FINDING 564 census weighting).
CELLS = {
    "mlp.gate_up": (5120, 34816, 64, True),
    "mlp.down": (17408, 5120, 64, False),
    "gdn.in_proj": (5120, 16480, 48, True),
}

PREAMBLE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat bfloat16_t;
"""

ENTRY = """
kernel void {name}(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device {xtype}* x [[buffer(3)]],
    const device float* xsums [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]],
    constant int& qmv_k [[buffer(6)]],
    constant int& qmv_n [[buffer(7)]],
    constant int& qmv_stride [[buffer(8)]],
    uint3 qmv_tid [[threadgroup_position_in_grid]],
    uint qmv_lid [[thread_index_in_simdgroup]],
    uint qmv_sgid [[simdgroup_index_in_threadgroup]])
{{
    const int qmv_out_row = int(qmv_tid.y) * 8 + int(qmv_sgid) * 4;
    qwen_e120_qmv_wide<{na}, {table}>(
        w, scales, biases, x, xsums, y,
        qmv_k, qmv_n, qmv_stride,
        int(qmv_tid.x) * {na}, qmv_out_row, qmv_lid);
}}
"""

# One synthetic probe per op count, for the bytes-per-instruction calibration.
# The loop is rolled and the chain is dependent, so the backend emits the fma
# sequence once and cannot vectorize or delete it.
CALIBRATION = """
kernel void calib_{n}(
    device float* o [[buffer(0)]],
    const device float* a [[buffer(1)]],
    constant int& reps [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{{
    float acc = a[i];
    const float c = a[i + 1];
    for (int t = 0; t < reps; t++) {{
{chain}
    }}
    o[i] = acc;
}}
"""

LABEL = re.compile(r"^(\d+|[\w.\-]+):(?:\s+;\s*preds\s*=\s*(.*))?$")
DEF = re.compile(r"^\s*(?:(%[\w.\-]+)\s*=\s*)?([a-z][\w.]*)\b(.*)$")
NIBBLE_CONVERT = "air.convert.f.f32.s.i32"


def live_header() -> str:
    text = QWEN35.read_text()
    match = re.search(
        r'let qwen35E120QMVHeader = """\n(.*?)\n    """\n', text, re.S)
    if match is None:
        raise SystemExit("e223: could not read qwen35E120QMVHeader")
    return match.group(1)


def emit_air(source: str, workdir: pathlib.Path) -> str:
    src = workdir / "census.metal"
    src.write_text(source)
    out = workdir / "census.ll"
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O2",
         "-fno-fast-math", "-S", "-emit-llvm", str(src), "-o", str(out)],
        check=True, capture_output=True, text=True)
    return out.read_text()


def functions(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        define = re.match(r"^define\s+[^@]*@([\w.$]+)\(", line)
        if define:
            name = define.group(1)
            out[name] = []
            continue
        if line == "}":
            name = None
            continue
        if name is not None and line.strip():
            out[name].append(line)
    return out


class Block:
    def __init__(self, label: str, index: int) -> None:
        self.label = label
        self.index = index
        self.lines: list[str] = []
        self.succs: list[str] = []


def blocks_of(body: list[str]) -> dict[str, Block]:
    """Basic blocks with successors, in source order."""
    out: dict[str, Block] = {}
    current = Block("entry", 0)
    out["entry"] = current
    for line in body:
        stripped = line.strip()
        label = LABEL.match(stripped)
        if label:
            current = Block(label.group(1), len(out))
            out[current.label] = current
            continue
        current.lines.append(line)
        if stripped.startswith("br ") or stripped.startswith("switch "):
            current.succs = re.findall(r"label %([\w.\-]+)", stripped)
    return out


def dominators(blocks: dict[str, Block]) -> dict[str, set[str]]:
    """Iterative dominator sets over the CFG, entry first."""
    labels = list(blocks)
    preds: dict[str, list[str]] = {label: [] for label in labels}
    for block in blocks.values():
        for succ in block.succs:
            preds[succ].append(block.label)
    dom = {label: set(labels) for label in labels}
    dom[labels[0]] = {labels[0]}
    changed = True
    while changed:
        changed = False
        for label in labels[1:]:
            incoming = [dom[p] for p in preds[label]]
            new = set.intersection(*incoming) | {label} if incoming else {label}
            if new != dom[label]:
                dom[label] = new
                changed = True
    return dom


def loop_nest(blocks: dict[str, Block]) -> list[dict]:
    """Natural loops, found from real back edges rather than block order.

    LLVM places some exit blocks before their loop, so "branches to an earlier
    block" is not a back edge. An edge `latch -> header` is a back edge only
    when the header dominates the latch; the natural loop is then the header
    plus every block that reaches the latch without passing through the header.
    """
    dom = dominators(blocks)
    succs = {label: block.succs for label, block in blocks.items()}
    found = []
    for label, block in blocks.items():
        for succ in block.succs:
            if succ not in dom[label]:
                continue
            header, latch = succ, label
            body = {header}
            stack = [latch] if latch != header else []
            seen = {latch} if latch != header else set()
            while stack:
                node = stack.pop()
                body.add(node)
                for pred, targets in succs.items():
                    if node in targets and pred != header and pred not in seen:
                        seen.add(pred)
                        stack.append(pred)
            found.append({
                "header": header,
                "latch": latch,
                "trip": trip_count(blocks[latch], blocks[header]),
                "body": sorted(body, key=lambda x: blocks[x].index),
            })
    found.sort(key=lambda loop: (-len(loop["body"]), loop["header"]))
    return found


def trip_count(latch: Block, header: Block) -> int | str:
    """Trip count of the loop this latch closes.

    A counted loop closes on `icmp eq i32 %i, N`; the `k` loop closes on a
    comparison against a runtime value and is reported symbolically. A two-trip
    loop carries no compare at all: the frontend replaces the counter test with
    an `i1` phi that is `true` on entry and `false` on the back edge, so that
    shape is read as trip 2.
    """
    for line in reversed(latch.lines):
        text = line.split(", !")[0].strip()
        match = re.search(r"icmp\s+\w+\s+i32\s+[^,]+,\s+(-?\d+)\s*$", text)
        if match:
            return int(match.group(1))
        if re.search(r"icmp\s+\w+\s+i32\s+", text):
            return "runtime"
    for block in (latch, header):
        for line in block.lines:
            if re.search(r"=\s*phi i1 \[ (true|false),", line):
                return 2
    return "runtime"


def executions(blocks: dict[str, Block], loops: list[dict]) -> dict[str, dict]:
    """Executions of each block per dispatch and per k-block.

    The `k` loop is the one loop with a runtime trip count; everything inside
    it is priced per k-block, everything outside it per dispatch.
    """
    kloop = [loop for loop in loops if loop["trip"] == "runtime"]
    if len(kloop) != 1:
        raise SystemExit(f"e223: expected one runtime loop, found {len(kloop)}")
    outer = set(kloop[0]["body"])
    out: dict[str, dict] = {}
    for block in blocks.values():
        inside = block.label in outer
        factor = 1
        for loop in loops:
            if loop["trip"] == "runtime":
                continue
            if block.label in loop["body"]:
                factor *= loop["trip"]
        out[block.label] = {
            "in_k_loop": inside,
            "per_k_block": factor if inside else 0,
            "per_dispatch_outside_k": 0 if inside else factor,
        }
    return out


FAMILY_RULES = (
    # (family, predicate on the normalized instruction text)
    ("device_load", lambda t: t.startswith("load") and "addrspace(1)" in t),
    ("device_store", lambda t: t.startswith("store") and "addrspace(1)" in t),
    ("private_load", lambda t: t.startswith("load") and "addrspace(" not in t),
    ("private_store", lambda t: t.startswith("store") and "addrspace(" not in t),
    ("address_arith", lambda t: t.startswith("getelementptr")
     or t.startswith("bitcast") or t.startswith("ptrtoint")),
    ("int_arith", lambda t: t.split()[0] in {
        "add", "sub", "mul", "shl", "lshr", "ashr", "and", "or", "xor",
        "sdiv", "udiv", "srem", "urem", "zext", "sext", "trunc", "icmp"}),
    ("fma", lambda t: "llvm.fmuladd" in t or "llvm.fma." in t
     or "air.fma" in t),
    ("fp_arith", lambda t: t.split()[0] in {
        "fadd", "fsub", "fmul", "fdiv", "fneg"}),
    ("convert_i2f", lambda t: NIBBLE_CONVERT in t or t.startswith("sitofp")
     or t.startswith("uitofp")),
    ("convert_bf16_f32", lambda t: t.startswith("fpext")),
    ("convert_f32_bf16", lambda t: t.startswith("fptrunc")),
    ("vector_shuffle", lambda t: t.split()[0] in {
        "insertelement", "extractelement", "shufflevector"}),
    ("simd_reduce", lambda t: "air.simd_sum" in t),
    ("control", lambda t: t.split()[0] in {"br", "phi", "switch", "ret"}),
)

IGNORED = ("llvm.lifetime", "alloca", "call void @llvm.assume")


# Families whose AIR instruction is elementwise over an `<N x T>` vector. AGX
# lanes are scalar, so one such instruction costs N machine operations; every
# other family (a memory access, one element move, a branch) costs one.
ELEMENTWISE = frozenset({
    "fma", "fp_arith", "convert_i2f", "convert_bf16_f32", "convert_f32_bf16"})
VECTOR_TYPE = re.compile(r"<(\d+) x [\w.]+>")


def classify(line: str) -> tuple[str, int] | None:
    """`(family, lane operations)` for one AIR instruction."""
    text = re.sub(r"%[\w.\-]+", "%", line.split(", !")[0]).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or any(token in text for token in IGNORED):
        return None
    body = text.split("=", 1)[1].strip() if text.startswith("%") else text
    family = "other"
    for candidate, rule in FAMILY_RULES:
        try:
            if rule(body):
                family = candidate
                break
        except IndexError:
            continue
    lanes = 1
    if family in ELEMENTWISE:
        width = VECTOR_TYPE.search(body)
        if width:
            lanes = int(width.group(1))
    return family, lanes


ROLE_RULES = (
    ("activation_gather",
     lambda lines: any("load <4 x bfloat>" in line for line in lines)),
    ("dequant_fma",
     lambda lines: any(NIBBLE_CONVERT in line for line in lines)),
    ("weight_load",
     lambda lines: any(re.search(r"load i16, i16 addrspace\(1\)", line)
                       for line in lines)),
    ("scale_bias_load",
     lambda lines: any("load bfloat, bfloat addrspace(1)" in line
                       for line in lines)),
    ("xsums_load",
     lambda lines: any("load float, float addrspace(1)" in line
                       for line in lines)),
    ("epilogue_reduce",
     lambda lines: any("air.simd_sum" in line for line in lines)),
    ("output_store",
     lambda lines: any("store bfloat" in line and "addrspace(1)" in line
                       for line in lines)),
)


def role_of(block: Block) -> str:
    for role, rule in ROLE_RULES:
        if rule(block.lines):
            return role
    return "other"


def census_one(body: list[str]) -> dict:
    blocks = blocks_of(body)
    loops = loop_nest(blocks)
    counts = executions(blocks, loops)
    per_kblock: collections.Counter = collections.Counter()
    lanes_per_kblock: collections.Counter = collections.Counter()
    per_dispatch: collections.Counter = collections.Counter()
    lanes_per_dispatch: collections.Counter = collections.Counter()
    per_role: dict[str, collections.Counter] = {}
    detail = {}
    for block in blocks.values():
        family_counts: collections.Counter = collections.Counter()
        lane_counts: collections.Counter = collections.Counter()
        for line in block.lines:
            found = classify(line)
            if found:
                family, lanes = found
                family_counts[family] += 1
                lane_counts[family] += lanes
        role = role_of(block)
        weight = counts[block.label]
        bucket = per_role.setdefault(role, collections.Counter())
        for family, n in family_counts.items():
            if weight["per_k_block"]:
                per_kblock[family] += n * weight["per_k_block"]
                lanes_per_kblock[family] += (
                    lane_counts[family] * weight["per_k_block"])
                bucket[family] += lane_counts[family] * weight["per_k_block"]
            else:
                per_dispatch[family] += n * weight["per_dispatch_outside_k"]
                lanes_per_dispatch[family] += (
                    lane_counts[family] * weight["per_dispatch_outside_k"])
        detail[block.label] = {
            "role": role,
            "executions_per_k_block": weight["per_k_block"],
            "executions_per_dispatch_outside_k":
                weight["per_dispatch_outside_k"],
            "instructions": dict(family_counts),
            "lane_ops": dict(lane_counts),
        }
    return {
        "loops": loops,
        "blocks": detail,
        "instructions_per_k_block": dict(per_kblock),
        "ops_per_k_block": dict(lanes_per_kblock),
        "ops_per_dispatch_outside_k_loop": dict(lanes_per_dispatch),
        "instructions_per_dispatch_outside_k_loop": dict(per_dispatch),
        "ops_per_k_block_by_role": {
            role: dict(counter) for role, counter in per_role.items()
            if counter},
        "ops_per_k_block_total": sum(lanes_per_kblock.values()),
        "instructions_per_k_block_total": sum(per_kblock.values()),
    }


def linear_fit(points: dict[int, float]) -> dict | None:
    if len(points) < 3:
        return None
    xs = sorted(points)
    n = len(xs)
    mx = sum(xs) / n
    my = sum(points[x] for x in xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (points[x] - my) for x in xs)
    slope = sxy / sxx
    intercept = my - slope * mx
    residual = sum((points[x] - (intercept + slope * x)) ** 2 for x in xs)
    total = sum((points[x] - my) ** 2 for x in xs)
    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(1.0 - residual / total, 5) if total else 1.0,
        "na_domain": [xs[0], xs[-1]],
    }


def calibration_source(counts: tuple[int, ...]) -> str:
    parts = []
    for n in counts:
        chain = "\n".join(
            f"        acc = fma(acc, c, {float(j) + 1.0}f);" for j in range(n))
        parts.append(CALIBRATION.format(n=n, chain=chain))
    return PREAMBLE + "".join(parts)


def calibrate(workdir: pathlib.Path, counts=(16, 32, 64, 128)) -> dict:
    """Bytes of AGX machine code per emitted instruction, per architecture."""
    source = calibration_source(counts)
    lib = agx_crossarch.build_metallib(source, workdir / "calib")
    out: dict[str, dict] = {}
    for arch in ARCHES:
        found = agx_crossarch.translate(lib, arch, workdir / "calib")
        sizes = {n: found[f"calib_{n}"]["text_bytes"] for n in counts}
        fit = linear_fit({n: float(v) for n, v in sizes.items()})
        out[arch.replace("applegpu_", "")] = {
            "text_bytes_by_fma_count": sizes,
            "bytes_per_instruction": fit["slope"] if fit else None,
            "fit": fit,
        }
    return out


def measured_na_slope() -> dict:
    """E221's measured `d(ms/round)/d(NA)` at the shipped geometry."""
    if not E221_REPORT.exists():
        return {}
    report = json.loads(E221_REPORT.read_text())
    out = {}
    for key, row in report.get("na_slope_per_geometry", {}).items():
        cell, groups, thermal = key.split("/")
        out[key] = {
            "cell": cell,
            "groups": groups,
            "thermal": thermal,
            "invocations_per_round": row["invocations_per_round"],
            "rows4_ms_per_round_per_na": row["rows4_ms_per_round_per_na"],
        }
    return out


def lane_k_blocks(k: int, n: int, rows: int = 4, lanes: int = 32) -> int:
    """Lane-k-block executions of the inner loop for one dispatch."""
    return (n // rows) * lanes * (k // 512)


def regime_fits(sizes: dict[int, float], bytes_per_insn: float | None) -> dict:
    """Code-size fits over the two unroll regimes the backend actually uses.

    Between NA = 5 and NA = 6 the AGX backend changes strategy: machine-code
    size drops and spilling starts, so one fit over NA 2..9 describes neither
    regime. `instructions_per_na` divides the slope by the calibrated bytes per
    instruction, which turns code size into a per-column instruction estimate
    for the fully unrolled regime.
    """
    out = {}
    for name, domain in (("na2_5", (2, 3, 4, 5)), ("na6_9", (6, 7, 8, 9))):
        fit = linear_fit({na: sizes[na] for na in domain})
        if fit and bytes_per_insn:
            fit["instructions_per_na"] = round(fit["slope"] / bytes_per_insn, 1)
        out[name] = fit
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-agx", action="store_true")
    parser.add_argument("--variant", default="base",
                        choices=e223_variants.NAMES)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    out_path = args.out or ARTIFACTS / f"e223-census-{args.variant}.json"

    header = e223_variants.apply(live_header(), args.variant)
    # `xtf32` and `xf32ship` drop the in-kernel chunk-sum branch, so only the
    # table path is meaningful for them.
    tables = ((True,) if args.variant in ("xtf32", "xf32ship")
              else (True, False))
    arms = [(na, table) for na in NAS for table in tables]
    source = PREAMBLE + header + "".join(
        ENTRY.format(name=f"qmv_na{na}_{'tbl' if table else 'plain'}", na=na,
                     table="true" if table else "false",
                     xtype=e223_variants.activation_type(args.variant))
        for na, table in arms)

    result = {
        "experiment": "e223-percolumn-inner-loop",
        "stage": "0-attribution-census",
        "harness": "local",
        "measurement": "static_compile",
        "gpu_seconds": 0,
        "official_or_ranked_score": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "source_symbol": "qwen35E120QMVHeader",
        "variant": args.variant,
        "source_sha8": hashlib.sha256(header.encode()).hexdigest()[:8],
        "host_chip": "Apple M4 Pro",
        "local_arch": agx_crossarch.LOCAL_ARCH,
        "ranked_arch": agx_crossarch.RANKED_ARCH,
        "arms": [{"na": na, "use_table": table} for na, table in arms],
        "air": {},
        "agx": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        air = emit_air(source, workdir)
        bodies = functions(air)
        for na, table in arms:
            symbol = (f"_Z18qwen_e120_qmv_wideILi{na}ELb"
                      f"{1 if table else 0}E")
            match = [name for name in bodies if name.startswith(symbol)]
            if len(match) != 1:
                raise SystemExit(
                    f"e223: {len(match)} AIR bodies for NA={na} table={table}")
            key = f"na{na}_{'tbl' if table else 'plain'}"
            result["air"][key] = census_one(bodies[match[0]])
            result["air"][key]["na"] = na
            result["air"][key]["use_table"] = table

        if not args.skip_agx:
            lib = agx_crossarch.build_metallib(source, workdir / "agx")
            for arch in ARCHES:
                found = agx_crossarch.translate(lib, arch, workdir / "agx")
                short = arch.replace("applegpu_", "")
                result["agx"][short] = {
                    f"na{na}_{'tbl' if table else 'plain'}": {
                        "registers": found[
                            f"qmv_na{na}_{'tbl' if table else 'plain'}"
                        ]["registers"],
                        "spill_bytes": found[
                            f"qmv_na{na}_{'tbl' if table else 'plain'}"
                        ]["spill_bytes"],
                        "text_bytes": found[
                            f"qmv_na{na}_{'tbl' if table else 'plain'}"
                        ]["text_bytes"],
                        "text_sha8": found[
                            f"qmv_na{na}_{'tbl' if table else 'plain'}"
                        ]["text_sha8"],
                    }
                    for na, table in arms
                }
            result["calibration"] = calibrate(workdir)

    # Family slopes over NA, per path.
    result["fits"] = {}
    for path in ["tbl" if table else "plain" for table in tables]:
        families = set()
        for na in NAS:
            families |= set(result["air"][f"na{na}_{path}"]["ops_per_k_block"])
        for family in sorted(families):
            points = {
                na: float(result["air"][f"na{na}_{path}"]["ops_per_k_block"]
                          .get(family, 0))
                for na in NAS}
            fit = linear_fit(points)
            if fit:
                result["fits"][f"air/{path}/ops_per_k_block/{family}"] = fit
        totals = {na: float(result["air"][f"na{na}_{path}"]
                            ["ops_per_k_block_total"]) for na in NAS}
        result["fits"][f"air/{path}/ops_per_k_block/TOTAL"] = linear_fit(totals)
        for arch in result["agx"]:
            sizes = {na: float(result["agx"][arch][f"na{na}_{path}"]
                               ["text_bytes"]) for na in NAS}
            result["fits"][f"agx/{arch}/{path}/text_bytes"] = linear_fit(sizes)
            bytes_per_insn = (result.get("calibration", {}).get(arch, {})
                              .get("bytes_per_instruction"))
            result["fits"][f"agx/{arch}/{path}/text_bytes_by_regime"] = (
                regime_fits(sizes, bytes_per_insn))
            regs = {na: float(result["agx"][arch][f"na{na}_{path}"]
                              ["registers"]) for na in NAS}
            result["fits"][f"agx/{arch}/{path}/registers"] = linear_fit(regs)

    result["measured_na_slope"] = measured_na_slope()
    result["lane_k_blocks_per_dispatch"] = {
        cell: lane_k_blocks(k, n) for cell, (k, n, _, _) in CELLS.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"e223: wrote {out_path}  (variant {args.variant})")
    for path in ["tbl" if table else "plain" for table in tables]:
        fit = result["fits"][f"air/{path}/ops_per_k_block/TOTAL"]
        print(f"  air {path}: lane ops/k-block = {fit['intercept']:.1f}"
              f" + {fit['slope']:.1f}*NA (R2 {fit['r_squared']})")
        for arch in result.get("agx", {}):
            regime = result["fits"][f"agx/{arch}/{path}/text_bytes_by_regime"]
            regs = result["fits"][f"agx/{arch}/{path}/registers"]
            unrolled = regime["na2_5"]
            print(f"    {arch}: NA2-5 text_bytes slope {unrolled['slope']:.0f}"
                  f" B/NA = {unrolled.get('instructions_per_na')} insn/NA"
                  f" (R2 {unrolled['r_squared']});"
                  f" registers {regs['intercept']:.0f}"
                  f"+{regs['slope']:.2f}*NA;"
                  f" spills NA>=6: "
                  + ",".join(
                      str(result["agx"][arch][f"na{na}_{path}"]["spill_bytes"])
                      for na in (6, 7, 8, 9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
