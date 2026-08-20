#!/usr/bin/env python3
"""E78 rung 0: emit the four arm tables, census their registers, pre-register.

No GPU work. Everything here is a compile, a translation to the two Apple GPU
generations, or arithmetic over measurements the campaign already owns. The
predictions this file emits are committed before the first timed E78 leg.

  python3 research/e78_rung0.py --out research/e78-artifacts/rung0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch  # noqa: E402
import e78_arms  # noqa: E402
from jit_string_compile import PREAMBLE_BODY, PREAMBLES, preamble, template_def  # noqa: E402

ARCHS = ("applegpu_g17s", "applegpu_g16s")
# The stop-rule ceilings the assignment names, measured on the base table.
REGISTER_CEILING = {"applegpu_g17s": 111, "applegpu_g16s": 96}

# The dispatch entry the scored worker JIT-compiles for the affine-4 group-64
# path. `batched=false` is the decode shape; the batched twin is built too
# because the same switch lives in it.
COMBINED_CELLS = (
    "affine_qmv_fast<bfloat16_t, 64, 4, false>",
    "affine_qmv_fast<bfloat16_t, 64, 4, true>",
)

CELL_ENTRY = """
[[kernel]] void e78_cell_m{m}_ipg{ipg}(
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
  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, {m}, {ipg}, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid);
}}
"""

# Widths the wide branch dispatches through the multi-row helper, and the IPG
# each table gives them. M=2 uses the pair kernel and is untouched.
SHIP_TABLE = {3: 3, 4: 4, 5: 5, 6: 6, 7: 4, 8: 4, 9: 5}
CROWN_TABLE = {3: 3, 4: 4, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3}
MOVED_WIDTHS = (5, 6, 9)

# --- the scored affine-4 group-64 families -----------------------------------
#
# n, k and the per-verify call count are E33 section 8.2, quoted through
# research/xgroup_census.py. `head.compact_draft_vocab` is the 2-bit coarse
# readout at M=1 and never reaches this switch at M >= 3, so it is listed but
# carries no width-M call.
FAMILIES = [
    # name, n (out_vec_size), k (in_vec_size), calls per verify round
    ("mlp.gate_up_fused", 34816, 5120, 64),
    ("mlp.down", 5120, 17408, 64),
    ("linear_attn.in_proj_fused", 16480, 5120, 48),
    ("linear_attn.out_proj", 5120, 6144, 48),
    ("full_attn.qkv_proj_fused", 14336, 5120, 16),
    ("full_attn.o_proj", 5120, 6144, 16),
    ("head.lm_head", 248320, 5120, 1),
]

CORES = 20  # this host, read from the device below and asserted against

# E74's fitted in-situ occupancy law, senpai/campaign-ledger.md and
# research/e74-results.md rung 2.
E74_KNEE_TGS = 1558.0
E74_A = 0.2132

# E33's measured one-group / two-group cost ratio at M=6, ledger item 130.
# A ratio above 1 means one group (our table) is SLOWER than two (the crown's).
E33_RATIO_AT_M6 = {
    "head.lm_head": 0.9830,
    "head.compact_draft_vocab": 0.9868,
    "mlp.gate_up_fused": 0.9941,
    "linear_attn.in_proj_fused": 0.9947,
    "full_attn.qkv_proj_fused": 1.0148,
    "full_attn.o_proj": 1.0414,
    "linear_attn.out_proj": 1.0492,
    "mlp.down": 1.0592,
}

# E74's measured in-situ family tax at M=6, in ms per decode block, for the
# five families the pinning census can reach. research/e74-results.md rung 1.
E74_TAX_MS_AT_M6 = {
    "head.lm_head": 2.208,
    "mlp.gate_up_fused": 21.281,
    "mlp.down": 16.896,
    "linear_attn.out_proj": 3.168,   # census name gdn_out_proj
    "full_attn.o_proj": 1.194,       # census name fa_o_proj
}


def gpu_core_count() -> tuple[int, str]:
    out = subprocess.run(["ioreg", "-l"], capture_output=True, text=True,
                         errors="replace").stdout
    import re
    hits = re.findall(r'"gpu-core-count"\s*=\s*(\d+)', out)
    if hits:
        return int(hits[0]), 'ioreg -l "gpu-core-count"'
    raise SystemExit("e78_rung0: could not read the GPU core count from the device")


def working_tgs(m: int, ipg: int, n: int) -> int:
    """ceil(M/IPG) x-groups survive the early return; each covers ceil(n/8) rows."""
    return math.ceil(m / ipg) * math.ceil(n / 8)


def arm_ipg(arm: str, m: int, n: int) -> int:
    spec = e78_arms.ARMS[arm]
    if spec["cutoff"] is None:
        table = CROWN_TABLE if spec["cells"][5] == 3 else SHIP_TABLE
        return table[m]
    if m not in MOVED_WIDTHS:
        return SHIP_TABLE[m]
    return spec["cells"][m] if n >= spec["cutoff"] else spec["narrow_cells"][m]


def reachable_cells(arm: str) -> list[tuple[int, int]]:
    cells = {(m, SHIP_TABLE[m]) for m in (3, 4, 7, 8)}
    for m in MOVED_WIDTHS:
        for _, n, _, _ in FAMILIES:
            cells.add((m, arm_ipg(arm, m, n)))
    return sorted(cells)


def arm_source(arm: str) -> str:
    """The runtime-effective JIT string for this arm, byte-for-byte."""
    parts = []
    for stem in PREAMBLES:
        if stem == "quantized":
            text = e78_arms.apply_arm(e78_arms.TWIN, arm)
            match = PREAMBLE_BODY.search(text)
            if not match:
                raise SystemExit("e78_rung0: patched twin has no preamble body")
            parts.append(match.group(1) + "\n")
        else:
            parts.append(preamble(stem, None))
    parts += [template_def(cell) for cell in COMBINED_CELLS]
    parts += [CELL_ENTRY.format(m=m, ipg=ipg) for m, ipg in reachable_cells(arm)]
    return "".join(parts)


def census(arm: str, workdir: pathlib.Path) -> dict:
    source = arm_source(arm)
    lib = agx_crossarch.build_metallib(source, workdir)
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "cells": ["m%d_ipg%d" % c for c in reachable_cells(arm)],
        "by_arch": {arch: agx_crossarch.translate(lib, arch, workdir)
                    for arch in ARCHS},
    }


def family_table(cores: int) -> list[dict]:
    rows = []
    for name, n, k, calls in FAMILIES:
        for m in MOVED_WIDTHS:
            ship, crown = SHIP_TABLE[m], CROWN_TABLE[m]
            w_ship = working_tgs(m, ship, n)
            w_crown = working_tgs(m, crown, n)

            def deficit(w: int) -> float:
                return max(0.0, math.log(E74_KNEE_TGS) - math.log(w))

            rows.append({
                "family": name, "n": n, "k": k, "calls_per_round": calls,
                "M": m,
                "ipg_ship": ship, "ipg_crown": crown,
                "x_groups_ship": math.ceil(m / ship),
                "x_groups_crown": math.ceil(m / crown),
                "working_tgs_ship": w_ship,
                "working_tgs_crown": w_crown,
                "tgs_per_core_ship": w_ship / cores,
                "tgs_per_core_crown": w_crown / cores,
                "k_blocks": math.ceil(k / 64),
                "gb_per_call": (n * k / 2 + n * (k / 64) * 4) / 1e9,
                # E74's law, grid term only. Below 1 means the crown's extra
                # x-group buys more occupancy than it costs in weight passes.
                "e74_grid_factor_crown_over_ship":
                    math.exp(E74_A * (deficit(w_crown) - deficit(w_ship))),
                "in_c_hybrid24928": arm_ipg("c_hybrid24928", m, n),
                "in_d_hybrid8192": arm_ipg("d_hybrid8192", m, n),
            })
    return rows


def leg_prediction() -> dict:
    """Pre-registered per-arm effect at M=6, from E33 ratios x E74 family times.

    E33 measured the ratio at M=6 only. Applying it at M=5 assumes the same
    one-group-against-two contrast holds one width down, which is the same
    1 -> 2 x-group step. M=9 is a 2 -> 3 x-group step and is NOT covered by
    E33, so it is reported separately and never folded into the headline.
    """
    def arm_delta(cutoff: float) -> dict:
        total, terms = 0.0, {}
        for name, tax in E74_TAX_MS_AT_M6.items():
            n = {f[0]: f[1] for f in FAMILIES}[name]
            if n >= cutoff:
                terms[name] = 0.0
                continue
            # crown time = our time / (one-group / two-group ratio)
            delta = tax * (1.0 / E33_RATIO_AT_M6[name] - 1.0)
            terms[name] = delta
            total += delta
        return {"delta_ms": total, "by_family": terms}

    block_ms_e74 = 122.111   # E74 in-situ block at M=6, seed 768
    round_ms_ranked = 64.6   # advisor's in-situ M=6 round used for the assignment
    out = {}
    for arm, cutoff in (("b_crown", math.inf), ("c_hybrid24928", 24928.0),
                        ("d_hybrid8192", 8192.0)):
        pred = arm_delta(cutoff)
        pred["fraction_of_e74_block"] = pred["delta_ms"] / block_ms_e74
        pred["fraction_of_advisor_round"] = pred["delta_ms"] / round_ms_ranked
        out[arm] = pred
    return {
        "basis": "E33 ledger-130 ratios at M=6 x E74 in-situ family tax at M=6",
        "caveat": "five interceptable families only; qkv_proj (14336) and "
                  "in_proj (16480) have an E33 ratio but no E74 in-situ tax, "
                  "so the C-minus-D contrast is not predicted here",
        "arms": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("research/e78-artifacts/rung0.json"))
    ap.add_argument("--skip-census", action="store_true")
    args = ap.parse_args()

    cores, cores_from = gpu_core_count()
    payload = {
        "base_sha": e78_arms.BASE_SHA,
        "crown_sha": e78_arms.CROWN_SHA,
        "harness": "local",
        "gpu_cores": cores,
        "gpu_cores_source": cores_from,
        "ship_table": SHIP_TABLE,
        "crown_table": CROWN_TABLE,
        "arms": {name: {"doc": spec["doc"], "cutoff": spec["cutoff"],
                        "cells": spec["cells"],
                        "narrow_cells": spec.get("narrow_cells"),
                        "reachable_cells": ["m%d_ipg%d" % c
                                            for c in reachable_cells(name)]}
                 for name, spec in sorted(e78_arms.ARMS.items())},
        "families": family_table(cores),
        "prediction": leg_prediction(),
        "register_ceiling": REGISTER_CEILING,
    }

    if not args.skip_census:
        censuses = {}
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("a_ship", "b_crown", "b_crown_exact",
                        "c_hybrid24928", "d_hybrid8192"):
                workdir = pathlib.Path(tmp) / arm
                censuses[arm] = census(arm, workdir)
        payload["register_census"] = censuses
        payload["stop_rule"] = stop_rule(censuses)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload.get("stop_rule", {}), indent=2, sort_keys=True))
    print("e78_rung0: wrote %s" % args.out)
    return 0


def stop_rule(censuses: dict) -> dict:
    """The rung-0 gate: cells must be untouched, and no new register maximum."""
    cell_drift, ceiling_breach, combined = [], [], {}
    reference: dict[tuple[str, str], dict] = {}
    for arm, data in censuses.items():
        for arch, kernels in data["by_arch"].items():
            for name, rec in kernels.items():
                if name.startswith("e78_cell_"):
                    key = (arch, name)
                    if key in reference:
                        prev = reference[key]
                        if (prev["registers"], prev["spill_bytes"],
                                prev["text_sha8"]) != (
                                rec["registers"], rec["spill_bytes"],
                                rec["text_sha8"]):
                            cell_drift.append(
                                {"arch": arch, "cell": name, "arm": arm,
                                 "reference": prev, "observed": rec})
                    else:
                        reference[key] = rec
                else:
                    combined.setdefault(arch, {})[arm + "/" + name] = rec
                    if rec["registers"] > REGISTER_CEILING[arch]:
                        ceiling_breach.append(
                            {"arch": arch, "arm": arm, "kernel": name,
                             "registers": rec["registers"],
                             "ceiling": REGISTER_CEILING[arch]})
    return {
        "cell_drift": cell_drift,
        "cell_drift_count": len(cell_drift),
        "ceiling_breach": ceiling_breach,
        "ceiling_breach_count": len(ceiling_breach),
        "combined_kernels": combined,
        "passed": not cell_drift and not ceiling_breach,
    }


if __name__ == "__main__":
    raise SystemExit(main())
