#!/usr/bin/env python3
"""E221 step 1: price rows_per_simd 4 -> 2 against the relief ceiling.

Reads the `rows` phase of an E219 session and answers one question: does
halving outputs per simdgroup relieve the NA slope (the register-pressure
premise) or cost more than it saves (the activation-re-read premise,
FINDING 576)?

Three arms per cell/NA/G/thermal cell:
  rows4       shipped header, the reconciliation anchor
  rows4param  parameterized header at the shipped geometry, must be inert
  rows2param  the treatment

Per-round cost uses each cell's own invocations_per_round, so a cell result is
never pooled with a different invocation count by accident. `mlp.down` is
reported per cell and never folded into a pooled figure.

NOT gate-qualified. harness=local-microbench.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e219_analyze import chain_slopes  # noqa: E402

# E221 assignment constants.
RELIEF_CEILING_MS = 23.066
STOP_RULE_MS = 0.3
NOISE_FLOOR_MS = 0.24
# FINDING 564 pooled per-round invocation census, for the pooled figure only.
POOLED_CELLS = ("mlp.gate_up", "mlp.down", "gdn.in_proj")


def key(fields: dict) -> tuple:
    return (fields["cell"], fields["na"], fields["groups"], fields["cold"])


def overlaps(a: list[float], b: list[float]) -> bool:
    """Do two bootstrap CI95 intervals overlap?"""
    return a[0] <= b[1] and b[0] <= a[1]


def issued_bytes(k: int, n: int, na: int, rows: int) -> int:
    """Issued device read bytes for one dispatch.

    From research/e221-artifacts/qmv-geometry-byte-census.md: one lane reads
    `12 * rows + 36 * na` bytes per k-block, and covering `n` rows takes
    `n / rows` simdgroups of 32 lanes.
    """
    return (n // rows) * (k // 512) * 32 * (12 * rows + 36 * na)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_json", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    blob = json.loads(args.rows_json.read_text())
    if blob.get("phase") != "rows":
        raise SystemExit("not an E221 rows phase report: %s" % args.rows_json)
    units = chain_slopes(blob)

    # group by (cell, na, groups, cold) -> geometry -> unit
    grid: dict[tuple, dict[str, dict]] = {}
    for unit in units.values():
        f = unit["fields"]
        grid.setdefault(key(f), {})[f["geometry"]] = unit

    comparisons = []
    for cell_key in sorted(grid):
        arms = grid[cell_key]
        if "rows4" not in arms:
            continue
        cell, na, groups, cold = cell_key
        inv = arms["rows4"]["fields"]["invocations_per_round"]
        base = arms["rows4"]
        row = {
            "cell": cell, "na": na, "groups": groups, "cold": cold,
            "invocations_per_round": inv,
            "rows4_us_per_invocation": base["slope_us"],
            "rows4_ci95_us": base["slope_ci95"],
            "rows4_ms_per_round": base["slope_us"] * inv / 1000.0,
        }
        for arm in ("rows4param", "rows2param"):
            if arm not in arms:
                continue
            got = arms[arm]
            d_us = got["slope_us"] - base["slope_us"]
            row[arm] = {
                "us_per_invocation": got["slope_us"],
                "ci95_us": got["slope_ci95"],
                "ci95_half_width_ms_per_round":
                    0.5 * (got["slope_ci95"][1] - got["slope_ci95"][0])
                    * inv / 1000.0,
                "ms_per_round": got["slope_us"] * inv / 1000.0,
                # Positive delta_ms_per_round = the arm is SLOWER.
                "delta_ms_per_round": d_us * inv / 1000.0,
                "ratio": (got["slope_us"] / base["slope_us"]
                          if base["slope_us"] else float("nan")),
                "bytes_per_row_per_kblock":
                    got["fields"]["bytes_per_row_per_kblock"],
                # Relief is the NEGATIVE of the delta: a faster arm relieves.
                "relief_ms_per_round": -d_us * inv / 1000.0,
                "ci95_overlaps_rows4": overlaps(
                    got["slope_ci95"], base["slope_ci95"]),
            }
        # Same-header contrast. rows2param and rows4param share one header
        # text, so this isolates the GEOMETRY from the substitution and is the
        # contrast the hypothesis is actually about.
        if "rows4param" in arms and "rows2param" in arms:
            ref, got = arms["rows4param"], arms["rows2param"]
            d_us = got["slope_us"] - ref["slope_us"]
            row["rows2_vs_rows4param"] = {
                "delta_ms_per_round": d_us * inv / 1000.0,
                "relief_ms_per_round": -d_us * inv / 1000.0,
                "ratio": (got["slope_us"] / ref["slope_us"]
                          if ref["slope_us"] else float("nan")),
                "ci95_overlaps": overlaps(
                    got["slope_ci95"], ref["slope_ci95"]),
            }
        comparisons.append(row)

    # Pooled figure over the FINDING 564 census, cold arms only.
    pooled = {}
    for na in sorted({c["na"] for c in comparisons}):
        for groups in sorted({c["groups"] for c in comparisons}):
            for cold in (True, False):
                sel = [c for c in comparisons
                       if c["na"] == na and c["groups"] == groups
                       and c["cold"] == cold and c["cell"] in POOLED_CELLS]
                if len(sel) != len(POOLED_CELLS):
                    continue
                entry = {"cells": len(sel),
                         "rows4_ms_per_round":
                             sum(c["rows4_ms_per_round"] for c in sel)}
                for arm in ("rows4param", "rows2param"):
                    if not all(arm in c for c in sel):
                        continue
                    entry[arm] = {
                        "ms_per_round": sum(c[arm]["ms_per_round"]
                                            for c in sel),
                        "delta_ms_per_round": sum(c[arm]["delta_ms_per_round"]
                                                  for c in sel),
                        "relief_ms_per_round": sum(c[arm]["relief_ms_per_round"]
                                                   for c in sel),
                    }
                pooled["na%d/g%d/%s" % (na, groups, "cold" if cold else "hot")] \
                    = entry

    # NA slope per geometry, from the NA=4 -> NA=5 pair at fixed G.
    slopes = {}
    for cell in sorted({c["cell"] for c in comparisons}):
        for groups in sorted({c["groups"] for c in comparisons}):
            for cold in (True, False):
                at = {c["na"]: c for c in comparisons
                      if c["cell"] == cell and c["groups"] == groups
                      and c["cold"] == cold}
                if 4 not in at or 5 not in at:
                    continue
                tag = "%s/g%d/%s" % (cell, groups, "cold" if cold else "hot")
                inv = at[4]["invocations_per_round"]
                entry = {
                    "rows4_ms_per_round_per_na":
                        at[5]["rows4_ms_per_round"] - at[4]["rows4_ms_per_round"],
                    "invocations_per_round": inv,
                }
                for arm in ("rows4param", "rows2param"):
                    if arm in at[4] and arm in at[5]:
                        entry["%s_ms_per_round_per_na" % arm] = (
                            at[5][arm]["ms_per_round"]
                            - at[4][arm]["ms_per_round"])
                slopes[tag] = entry

    verdict = {}
    for tag, entry in pooled.items():
        if "rows2param" not in entry:
            continue
        relief = entry["rows2param"]["relief_ms_per_round"]
        verdict[tag] = {
            "relief_ms_per_round": relief,
            "relief_ceiling_ms_per_round": RELIEF_CEILING_MS,
            "stop_rule_ms_per_round": STOP_RULE_MS,
            "noise_floor_ms_per_round": NOISE_FLOOR_MS,
            "meets_stop_rule": relief >= STOP_RULE_MS,
            "outside_noise_floor": abs(relief) > NOISE_FLOOR_MS,
        }

    # Inertness is judged by CI95 OVERLAP, not against the pooled +/-0.24
    # ms/round noise floor. That floor describes the pooled three-cell figure;
    # a single-cell bootstrap half-width here is already 0.1 to 0.5 ms/round,
    # so testing one cell against the pooled floor would report the
    # instrument's own resolution as a treatment effect.
    inertness = {}
    for row in comparisons:
        if "rows4param" not in row:
            continue
        tag = "%s/na%d/g%d/%s" % (row["cell"], row["na"], row["groups"],
                                  "cold" if row["cold"] else "hot")
        arm = row["rows4param"]
        inertness[tag] = {
            "delta_ms_per_round": arm["delta_ms_per_round"],
            "ratio": arm["ratio"],
            "ci95_half_width_ms_per_round":
                arm["ci95_half_width_ms_per_round"],
            "ci95_overlaps_rows4": arm["ci95_overlaps_rows4"],
        }

    # How much time does one extra ISSUED megabyte actually cost, depending on
    # HOW it is added? Two routes add issued bytes to the same kernel:
    #   geometry: rows 4 -> 2 doubles the activation and chunk-sum re-read;
    #   width:    NA 4 -> 5 adds one more activation column per lane.
    # If the NA slope were an activation-byte VOLUME effect, both routes would
    # convert at the same rate. Comparing the two rates is therefore the
    # separating measurement for the reframed E221 question.
    conversion = {}
    for cell_name in sorted({c["cell"] for c in comparisons}):
        sample = next(c for c in comparisons if c["cell"] == cell_name)
        k = next(u["fields"]["k"] for u in units.values()
                 if u["fields"]["cell"] == cell_name)
        n = next(u["fields"]["n"] for u in units.values()
                 if u["fields"]["cell"] == cell_name)
        inv = sample["invocations_per_round"]
        for groups in sorted({c["groups"] for c in comparisons}):
            for cold in (True, False):
                at = {c["na"]: c for c in comparisons
                      if c["cell"] == cell_name and c["groups"] == groups
                      and c["cold"] == cold}
                tag = "%s/g%d/%s" % (cell_name, groups,
                                     "cold" if cold else "hot")
                entry: dict = {"k": k, "n": n, "invocations_per_round": inv}
                geo = []
                for na, row in sorted(at.items()):
                    if "rows2_vs_rows4param" not in row:
                        continue
                    d_mb = (issued_bytes(k, n, na, 2)
                            - issued_bytes(k, n, na, 4)) * groups / 1048576.0
                    d_us = (row["rows2_vs_rows4param"]["delta_ms_per_round"]
                            * 1000.0 / inv)
                    geo.append({
                        "na": na, "delta_issued_mb": d_mb,
                        "delta_us_per_invocation": d_us,
                        "us_per_issued_mb": d_us / d_mb if d_mb else None,
                    })
                if geo:
                    entry["geometry_route_rows4_to_rows2"] = geo
                if 4 in at and 5 in at:
                    d_mb = (issued_bytes(k, n, 5, 4)
                            - issued_bytes(k, n, 4, 4)) * groups / 1048576.0
                    d_us = (at[5]["rows4_us_per_invocation"]
                            - at[4]["rows4_us_per_invocation"])
                    entry["width_route_na4_to_na5_at_rows4"] = {
                        "delta_issued_mb": d_mb,
                        "delta_us_per_invocation": d_us,
                        "us_per_issued_mb": d_us / d_mb if d_mb else None,
                    }
                if "geometry_route_rows4_to_rows2" in entry \
                        and "width_route_na4_to_na5_at_rows4" in entry:
                    w = entry["width_route_na4_to_na5_at_rows4"][
                        "us_per_issued_mb"]
                    g5 = next((e["us_per_issued_mb"] for e in geo
                               if e["na"] == 5), None)
                    if w and g5:
                        entry["width_over_geometry_rate_ratio"] = w / g5
                conversion[tag] = entry

    report = {
        "probe": "e221-rows-per-simd",
        "harness": "local-microbench",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "whole_leg_or_ranked_number": False,
        "source_report": str(args.rows_json),
        "shipped_rows_per_simd": blob["shipped_rows_per_simd"],
        "write_census": blob.get("e221_write_census"),
        "register_witness": blob.get("e221_register_witness"),
        "gpu_temperature_c": blob.get("gpu_temperature_c"),
        "blocks": blob.get("blocks"),
        "chains": blob.get("chains"),
        "comparisons": comparisons,
        "pooled": pooled,
        "na_slope_per_geometry": slopes,
        "parameterized_header_inertness": inertness,
        "issued_byte_conversion": conversion,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("wrote %s\n" % args.out)

    print("per-cell, cold, G=1  (positive delta = rows2 is SLOWER)")
    print("%-13s %3s %6s %10s %10s %8s %10s"
          % ("cell", "NA", "inv", "rows4 ms", "rows2 ms", "ratio", "delta ms"))
    for c in comparisons:
        if not c["cold"] or c["groups"] != 1 or "rows2param" not in c:
            continue
        r2 = c["rows2param"]
        print("%-13s %3d %6d %10.3f %10.3f %8.3f %+10.3f"
              % (c["cell"], c["na"], c["invocations_per_round"],
                 c["rows4_ms_per_round"], r2["ms_per_round"], r2["ratio"],
                 r2["delta_ms_per_round"]))

    n_overlap = sum(1 for e in inertness.values()
                    if e["ci95_overlaps_rows4"])
    print("\nparameterized-header inertness at rows=4: CI95 overlaps rows4 in "
          "%d of %d cells" % (n_overlap, len(inertness)))
    worst = sorted(inertness.items(),
                   key=lambda kv: -abs(kv[1]["delta_ms_per_round"]))
    for tag, e in worst[:6]:
        print("  %-34s %+8.3f ms/round  ratio %.4f  CI95 half-width %.3f  %s"
              % (tag, e["delta_ms_per_round"], e["ratio"],
                 e["ci95_half_width_ms_per_round"],
                 "overlaps" if e["ci95_overlaps_rows4"] else "DISJOINT"))

    same = [c for c in comparisons if "rows2_vs_rows4param" in c]
    n_slower = sum(1 for c in same
                   if c["rows2_vs_rows4param"]["delta_ms_per_round"] > 0)
    n_disjoint = sum(1 for c in same
                     if not c["rows2_vs_rows4param"]["ci95_overlaps"])
    print("\nsame-header contrast rows2param vs rows4param: rows2 is slower in "
          "%d of %d cells, CI95 disjoint in %d"
          % (n_slower, len(same), n_disjoint))
    for c in same:
        e = c["rows2_vs_rows4param"]
        print("  %-13s na%d g%d %-5s %+8.3f ms/round  ratio %.4f  %s"
              % (c["cell"], c["na"], c["groups"],
                 "cold" if c["cold"] else "hot", e["delta_ms_per_round"],
                 e["ratio"],
                 "overlaps" if e["ci95_overlaps"] else "disjoint"))

    print("\nNA slope per geometry (ms/round per unit NA, cold, G=1)")
    for tag, e in sorted(slopes.items()):
        if "/g1/cold" not in tag:
            continue
        print("  %-26s rows4 %+8.3f   rows2 %+8.3f"
              % (tag, e["rows4_ms_per_round_per_na"],
                 e.get("rows2param_ms_per_round_per_na", float("nan"))))

    print("\nissued-byte conversion rate, cold, by route (us per issued MB)")
    print("  %-26s %12s %12s %8s"
          % ("cell/G", "geometry", "width NA4->5", "ratio"))
    for tag, e in sorted(conversion.items()):
        if not tag.endswith("/cold"):
            continue
        g5 = next((x["us_per_issued_mb"]
                   for x in e.get("geometry_route_rows4_to_rows2", [])
                   if x["na"] == 5), None)
        w = e.get("width_route_na4_to_na5_at_rows4", {}).get(
            "us_per_issued_mb")
        print("  %-26s %12s %12s %8s"
              % (tag,
                 "%.4f" % g5 if g5 else "-",
                 "%.4f" % w if w else "-",
                 "%.1fx" % e["width_over_geometry_rate_ratio"]
                 if "width_over_geometry_rate_ratio" in e else "-"))

    print("\npooled over %s (FINDING 564 census)" % ", ".join(POOLED_CELLS))
    for tag, e in sorted(verdict.items()):
        print("  %-16s relief %+9.3f ms/round   ceiling %.3f   "
              "stop rule %s"
              % (tag, e["relief_ms_per_round"], RELIEF_CEILING_MS,
                 "MET" if e["meets_stop_rule"] else "NOT MET -> STOP"))


if __name__ == "__main__":
    main()
