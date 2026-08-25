#!/usr/bin/env python3
"""E222 desk price: what fused register value sharing can be worth, per arm.

Zero GPU seconds. Everything here is arithmetic over three measured inputs:

  1. The E219 pass-anatomy coefficients (`research/e219-artifacts/pass-anatomy.json`,
     W&B run `42kjlmwy`) give the per-dispatch cost `a` and the per-streamed-MiB
     cost `b`. Fusing two staged dispatches into one deletes exactly one
     dispatch and one copy of the weight + scale/bias stream, so `a + b * stream`
     is the fusion saving.
  2. The E221 issued-byte census (`research/e221-artifacts/e221-rows-report.json`,
     W&B run `rmgosylr`) gives the MEASURED price of the geometry route -- what a
     megabyte costs when it is bought by re-reading the activation slab from a
     second simdgroup, rather than by widening NA. Edward measured 0.031-0.302
     us per issued MiB on that route against 0.52-2.33 us per issued MiB on the
     width route: a factor of 3.6 to 40. This supersedes the FINDING 557/576
     byte-linear conversion for pricing anything that moves `rows_per_simd`.
  3. The FINDING 564 pooled width census weights each width by the rounds that
     actually route there.

The arithmetic below is deliberately explicit rather than clever, because the
whole point is that the previous desk price for `rows = 2` was wrong by the
ratio between those two byte rates.

Byte model, validated to the byte against E221's own census (see
`_selftest_against_e221`): a pass at `(k, n, NA, rows)` issues

    weight       = n * k / 2
    scale_bias   = n * (k / 64) * 4
    activation   = (n / rows) * NA * k * 2
    xsums        = (n / rows) * (k / 512) * 32 * NA * 4

`weight` and `scale_bias` are per DISPATCH and do not depend on `rows`;
`activation` and `xsums` are per SIMDGROUP and scale as `1 / rows`. That single
asymmetry is the whole experiment: fusing the two staged token groups deletes
one copy of the first pair, and any `rows` reduction needed to fit the fused
accumulators in registers pays for it out of the second pair.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MIB = 1024.0 * 1024.0

# --- measured inputs -------------------------------------------------------

E219_ARTIFACT = HERE / "e219-artifacts" / "pass-anatomy.json"
E221_ARTIFACT = "research/e221-artifacts/e221-rows-report.json"
E221_REF = "origin/senpai/qwen38-mtp-r1"

# `research/e222-artifacts/e222-geometry.json`, Stage 0. Peak registers and
# spill bytes for the fused body on the local (g16s) and ranked (g17s) AGX
# backends. FINDING 577: the budgets differ by generation, 96 against 126.
G17S_BUDGET = 126
G16S_BUDGET = 96

# FINDING 564 pooled width census: rounds observed at each MTP width over five
# prompts. `m` below 6 never reaches a G >= 2 staged plan at these cells.
WIDTH_ROUNDS = {2: 3, 3: 61, 4: 128, 5: 166, 6: 5, 7: 7, 8: 10, 9: 257}
TOTAL_ROUNDS = sum(WIDTH_ROUNDS.values())

# The three cells E219 scored, which are the three the fitted coefficients and
# the 11.614 / 2.926 ceilings are derived from.
CELLS = [
    {"name": "mlp.gate_up", "k": 5120, "n": 34816, "invocations": 64},
    {"name": "mlp.down", "k": 17408, "n": 5120, "invocations": 64},
    {"name": "gdn.in_proj", "k": 5120, "n": 16480, "invocations": 48},
]

# `Qwen35QMVKernelVariant.pairs`, Vendor/mlx-swift-lm/.../Qwen35.swift:1591.
STAGED_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 5}
SINGLE_IPG = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 3}

WIDTHS = [6, 7, 8, 9]


def shipped_variant(cell_name: str, m: int) -> str:
    """`qwen35QMVVariant`, Qwen35.swift:1649. `singlePass` only at `m == 6`,
    and never for `mlp.down`."""
    return "singlePass" if (m == 6 and cell_name != "mlp.down") else "staged"


def group_widths(m: int, ipg: int) -> list[int]:
    """The NA of each dispatched group under a width plan."""
    out = []
    first = 0
    while first < m:
        out.append(min(ipg, m - first))
        first += ipg
    return out


# --- byte model ------------------------------------------------------------


def pass_bytes(k: int, n: int, na: int, rows: int) -> dict[str, float]:
    simdgroups = n / rows
    return {
        "weight": n * (k / 2),
        "scale_bias": n * (k / 64) * 4,
        "activation": simdgroups * na * k * 2,
        "xsums": simdgroups * (k / 512) * 32 * na * 4,
    }


def config_bytes(k: int, n: int, nas: list[int], rows: int) -> dict[str, float]:
    total = {"weight": 0.0, "scale_bias": 0.0, "activation": 0.0, "xsums": 0.0}
    for na in nas:
        for key, value in pass_bytes(k, n, na, rows).items():
            total[key] += value
    total["stream"] = total["weight"] + total["scale_bias"]
    total["slab"] = total["activation"] + total["xsums"]
    total["dispatches"] = float(len(nas))
    return total


def _selftest_against_e221(e221: dict) -> list[dict]:
    """E221 published `delta_issued_mb` for `rows` 4 -> 2 per cell and NA. This
    model must reproduce those numbers, or its `rows = 2` price means nothing."""
    checks = []
    for key, entry in e221["issued_byte_conversion"].items():
        cell_name = key.split("/")[0]
        cell = next(c for c in CELLS if c["name"] == cell_name)
        groups = int(key.split("/")[1][1:])
        for row in entry["geometry_route_rows4_to_rows2"]:
            na = row["na"]
            nas = [na] * groups
            four = config_bytes(cell["k"], cell["n"], nas, 4)
            two = config_bytes(cell["k"], cell["n"], nas, 2)
            mine = (two["slab"] - four["slab"]) / MIB
            theirs = row["delta_issued_mb"]
            checks.append(
                {
                    "key": key,
                    "na": na,
                    "model_delta_mib": mine,
                    "e221_delta_mib": theirs,
                    "rel_error": abs(mine - theirs) / theirs,
                }
            )
    return checks


# --- pricing ---------------------------------------------------------------


def geometry_rate_us_per_mib(e221: dict) -> dict[str, dict[str, float]]:
    """Measured cost of a megabyte bought on the GEOMETRY route, per cell.

    Reported as the min/median/max over E221's `(groups, thermal, NA)` cells for
    that cell, because the spread is the honest uncertainty in this price and
    the `NA = 5` points are the ones nearest this experiment's `NA = 9`."""
    per_cell: dict[str, list[float]] = {}
    per_cell_na5: dict[str, list[float]] = {}
    for key, entry in e221["issued_byte_conversion"].items():
        cell_name = key.split("/")[0]
        for row in entry["geometry_route_rows4_to_rows2"]:
            per_cell.setdefault(cell_name, []).append(row["us_per_issued_mb"])
            if row["na"] == 5:
                per_cell_na5.setdefault(cell_name, []).append(
                    row["us_per_issued_mb"])
    out = {}
    for cell_name, values in per_cell.items():
        near = sorted(per_cell_na5.get(cell_name, values))
        values = sorted(values)
        out[cell_name] = {
            "low": values[0],
            "high": values[-1],
            "near_na5_low": near[0],
            "near_na5_high": near[-1],
            "n_points": len(values),
        }
    return out


def price_arm(cell: dict, m: int, rows: int, coeff: dict, rate: dict) -> dict:
    """Per-invocation and per-round price of running `m` as ONE fused dispatch
    at `rows_per_simd = rows`, against that cell's own shipped plan at `m`."""
    k, n = cell["k"], cell["n"]
    variant = shipped_variant(cell["name"], m)
    ipg = (STAGED_IPG if variant == "staged" else SINGLE_IPG)[m]
    base_nas = group_widths(m, ipg)
    base = config_bytes(k, n, base_nas, 4)
    fused = config_bytes(k, n, [m], rows)

    dispatches_saved = base["dispatches"] - fused["dispatches"]
    stream_saved_mib = (base["stream"] - fused["stream"]) / MIB
    slab_added_mib = (fused["slab"] - base["slab"]) / MIB

    a_us = dispatches_saved * coeff["a_dispatch_us"]
    b_us = stream_saved_mib * coeff["b_stream_us_per_mib"]
    penalty_low = slab_added_mib * rate["near_na5_low"]
    penalty_high = slab_added_mib * rate["near_na5_high"]

    inv = cell["invocations"]
    return {
        "cell": cell["name"],
        "m": m,
        "rows": rows,
        "shipped_variant": variant,
        "shipped_groups": base_nas,
        "dispatches_saved": dispatches_saved,
        "stream_saved_mib": stream_saved_mib,
        "slab_added_mib": slab_added_mib,
        "a_term_us_per_invocation": a_us,
        "b_term_us_per_invocation": b_us,
        "slab_penalty_us_per_invocation": [penalty_low, penalty_high],
        "net_us_per_invocation": [a_us + b_us - penalty_high,
                                  a_us + b_us - penalty_low],
        "net_ms_per_round_at_this_width": [
            (a_us + b_us - penalty_high) * inv / 1000.0,
            (a_us + b_us - penalty_low) * inv / 1000.0,
        ],
    }


# --- register legality -----------------------------------------------------

# Stage 0 measured peak registers and spill bytes for the fused body. Key is
# `(m, rows)`; value is `(g16s_regs, g16s_spill, g17s_regs, g17s_spill)`.
#
# Read from `research/e222-artifacts/e222-geometry.json` at the `both` schedule
# (lazy packed load plus late scale/bias read), which Stage 0 measured as
# strictly dominant: at every `(m, rows)` it uses fewer registers AND fewer AIR
# instructions than the eager schedule, at identical results. Both timed fused
# arms therefore use it, so these are the numbers that decide legality.
FUSED_REGISTERS = {
    (6, 2): (68, 0, 73, 0),
    (7, 2): (73, 0, 78, 0),
    (8, 2): (79, 0, 84, 0),
    (9, 2): (88, 0, 96, 0),
    (6, 4): (95, 0, 100, 0),
    (7, 4): (96, 48, 111, 0),
    (8, 4): (96, 80, 121, 0),
    (9, 4): (96, 128, 126, 32),
}


def legality(m: int, rows: int) -> dict:
    g16r, g16s, g17r, g17s = FUSED_REGISTERS[(m, rows)]
    return {
        "g16s_registers": g16r,
        "g16s_spill_bytes": g16s,
        "g16s_clean": g16s == 0,
        "g17s_registers": g17r,
        "g17s_spill_bytes": g17s,
        "g17s_clean": g17s == 0,
        "ranked_clean": g17s == 0,
        "local_measurement_transferable": g16s == 0 and g17s == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="research/e222-artifacts/e222-desk.json")
    args = parser.parse_args()

    e219 = json.loads(E219_ARTIFACT.read_text())
    coeff_raw = e219["composition_fit"]["coefficients"]
    coeff = {
        "a_dispatch_us": coeff_raw["a_dispatch_us"]["value"],
        "a_dispatch_us_ci95": coeff_raw["a_dispatch_us"]["ci95"],
        "b_stream_us_per_mib": coeff_raw["b_stream_us_per_mib"]["value"],
        "b_stream_us_per_mib_ci95": coeff_raw["b_stream_us_per_mib"]["ci95"],
        "e_threadgroup_us_per_1k": coeff_raw["e_threadgroup_us_per_1k"]["value"],
        "e_threadgroup_resolved_away_from_zero": coeff_raw[
            "e_threadgroup_us_per_1k"]["resolved_away_from_zero"],
    }

    e221 = json.loads(
        subprocess.run(
            ["git", "show", f"{E221_REF}:{E221_ARTIFACT}"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout)

    selftest = _selftest_against_e221(e221)
    worst = max(c["rel_error"] for c in selftest)

    rates = geometry_rate_us_per_mib(e221)

    arms = []
    for rows in (4, 2):
        for m in WIDTHS:
            per_cell = [
                price_arm(cell, m, rows, coeff, rates[cell["name"]])
                for cell in CELLS
            ]
            lo = sum(c["net_ms_per_round_at_this_width"][0] for c in per_cell)
            hi = sum(c["net_ms_per_round_at_this_width"][1] for c in per_cell)
            share = WIDTH_ROUNDS[m] / TOTAL_ROUNDS
            arms.append(
                {
                    "arm": f"fused_r{rows}",
                    "m": m,
                    "rounds_at_this_width": WIDTH_ROUNDS[m],
                    "round_share": share,
                    "legality": legality(m, rows),
                    "per_cell": per_cell,
                    "ms_per_round_at_this_width": [lo, hi],
                    "pooled_ms_per_round": [lo * share, hi * share],
                }
            )

    # Pool each arm over the widths it can legally serve on the RANKED device.
    pooled = {}
    for rows in (4, 2):
        rows_arms = [a for a in arms if a["arm"] == f"fused_r{rows}"]
        for label, keep in (
            ("ranked_clean_widths_only",
             lambda a: a["legality"]["ranked_clean"]),
            ("all_g2_widths", lambda a: True),
        ):
            chosen = [a for a in rows_arms if keep(a)]
            pooled[f"fused_r{rows}/{label}"] = {
                "widths": [a["m"] for a in chosen],
                "round_mass": sum(a["rounds_at_this_width"] for a in chosen),
                "pooled_ms_per_round": [
                    sum(a["pooled_ms_per_round"][0] for a in chosen),
                    sum(a["pooled_ms_per_round"][1] for a in chosen),
                ],
            }

    # Nothing forces one geometry at every width. `rows` is already a per-width
    # decision in the shipped plan, so the deliverable is the per-width argmax
    # over the ranked-legal arms: keep the whole fusion saving at `rows = 4`
    # where the registers allow it, and buy `m = 9` with `rows = 2`.
    best = []
    for m in WIDTHS:
        legal = [
            a for a in arms
            if a["m"] == m and a["legality"]["ranked_clean"]
        ]
        if not legal:
            continue
        pick = max(legal, key=lambda a: a["pooled_ms_per_round"][0])
        best.append(pick)
    pooled["per_width_argmax_over_ranked_legal_arms"] = {
        "plan": {a["m"]: a["arm"] for a in best},
        "widths": [a["m"] for a in best],
        "round_mass": sum(a["rounds_at_this_width"] for a in best),
        "pooled_ms_per_round": [
            sum(a["pooled_ms_per_round"][0] for a in best),
            sum(a["pooled_ms_per_round"][1] for a in best),
        ],
    }

    report = {
        "harness": "desk-arithmetic",
        "gpu_seconds": 0,
        "official_or_ranked_score": None,
        "whole_leg_or_ranked_number": None,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "inputs": {
            "e219_coefficients": coeff,
            "e221_geometry_route_us_per_mib": rates,
            "e221_width_route_note":
                "E221 measured 0.52-2.33 us per issued MiB on the WIDTH route "
                "against 0.031-0.302 on the GEOMETRY route. The FINDING "
                "557/576 byte-linear dose is a width-route rate and overprices "
                "a rows_per_simd change by 3.6x to 40x.",
            "width_census_rounds": WIDTH_ROUNDS,
            "total_rounds": TOTAL_ROUNDS,
        },
        "byte_model_selftest": {
            "description":
                "this model's rows 4 -> 2 slab delta against E221's published "
                "delta_issued_mb, per (cell, groups, thermal, NA)",
            "points": len(selftest),
            "worst_relative_error": worst,
            "passes": worst < 0.01,
            "detail": selftest,
        },
        "arms": arms,
        "pooled": pooled,
        "accounting_note":
            "The saving is decomposed into the a term (QMV dispatch fusion, "
            "parked at 2.926 pooled in the E219 shortlist) and the b term "
            "(dequantisation sharing across token groups, 11.614 pooled). "
            "Report them separately so the parked 2.926 is not counted twice.",
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"byte-model selftest: {len(selftest)} points, worst relative "
          f"error {worst:.5f} -> {'PASS' if worst < 0.01 else 'FAIL'}")
    print()
    print("per-width net, ms/round (low, high over the measured byte-rate "
          "spread); * = spills on the ranked g17s device")
    for rows in (4, 2):
        print(f"  fused rows={rows}")
        for a in (x for x in arms if x["arm"] == f"fused_r{rows}"):
            lo, hi = a["ms_per_round_at_this_width"]
            plo, phi = a["pooled_ms_per_round"]
            mark = " " if a["legality"]["ranked_clean"] else "*"
            print(f"    m={a['m']}{mark} regs g16s/g17s "
                  f"{a['legality']['g16s_registers']}+"
                  f"{a['legality']['g16s_spill_bytes']}/"
                  f"{a['legality']['g17s_registers']}+"
                  f"{a['legality']['g17s_spill_bytes']:<3} "
                  f"at-width [{lo:8.3f}, {hi:8.3f}]  "
                  f"pooled [{plo:7.3f}, {phi:7.3f}]")
    print()
    print("pooled over the widths each arm can legally serve")
    for key in sorted(pooled):
        p = pooled[key]
        lo, hi = p["pooled_ms_per_round"]
        print(f"  {key:44s} widths {str(p['widths']):16s} "
              f"mass {p['round_mass']:3d}/{TOTAL_ROUNDS}  "
              f"[{lo:7.3f}, {hi:7.3f}] ms/round")
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
