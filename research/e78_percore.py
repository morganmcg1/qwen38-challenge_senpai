#!/usr/bin/env python3
"""Grade the grid-starvation model against the rung 2a cells.

The advisor's model says one variable decides whether splitting a QMV row
block into more weight-stream groups helps: the threadgroups-per-core count of
the *fewer group* dispatch. Below a knee the grid starves the machine, so
doubling the grid repays the doubled weight traffic. Above the knee it does
not.

This script grades that model, replicates the E33 sign flip as a positive
control, and reports where the boundary sits with an interval.
"""

import json
import math
from pathlib import Path

ART = Path(__file__).resolve().parent / "e78-artifacts"
CELLS = ART / "rung2a-cells.json"

CORES_LOCAL = 20
CORES_RANKED = 40

# Ranked share of verify rounds by row width, from the campaign width census.
RANKED_WIDTH_SHARE = {4: 14.2, 5: 24.1, 6: 33.4, 7: 12.2, 8: 7.35, 9: 5.75}

# E33's two probe shapes. The advisor reports the g16s flip between them.
E33_PROBES = {"full_attn.qkv_proj_fused": 14336, "linear_attn.in_proj_fused_qkvzba": 16480}


def load_cells():
    doc = json.loads(CELLS.read_text())
    by_key = {}
    for c in doc["cells"]:
        by_key[(c["shape"], c["m"], c["ipg"])] = c
    return doc, by_key


def contested(by_key):
    """Every (shape, M) where the two measured group counts differ."""
    pairs = {}
    for (shape, m, ipg), c in by_key.items():
        pairs.setdefault((shape, m), []).append(c)
    out = []
    for (shape, m), cs in sorted(pairs.items()):
        if len(cs) != 2:
            continue
        lo, hi = sorted(cs, key=lambda c: c["groups"])
        if lo["groups"] == hi["groups"]:
            continue
        lo_ms = lo["ms_per_dispatch"]["mean"]
        hi_ms = hi["ms_per_dispatch"]["mean"]
        out.append(
            {
                "shape": shape,
                "m": m,
                "k": lo["k"],
                "n": lo["n"],
                "few_groups": lo["groups"],
                "few_ipg": lo["ipg"],
                "few_ms": lo_ms,
                "more_groups": hi["groups"],
                "more_ipg": hi["ipg"],
                "more_ms": hi_ms,
                # Grid of the fewer-group dispatch: the starvation candidate.
                "few_tgs": lo["working_threadgroups"],
                "few_tgs_per_core_local": lo["working_threadgroups"] / CORES_LOCAL,
                "few_tgs_per_core_ranked": lo["working_threadgroups"] / CORES_RANKED,
                "split_delta_ms": hi_ms - lo_ms,
                "split_pct": 100.0 * (hi_ms - lo_ms) / lo_ms,
                "split_helps": hi_ms < lo_ms,
                "rel_sem_pct": max(
                    lo["ms_per_dispatch"]["rel_sem_pct"],
                    hi["ms_per_dispatch"]["rel_sem_pct"],
                ),
            }
        )
    return out


def grade_boundary(rows):
    """A single threshold on fewer-group TGs/core can separate the signs only
    if every 'split helps' row sits strictly below every 'split hurts' row."""
    helps = [r for r in rows if r["split_helps"]]
    hurts = [r for r in rows if not r["split_helps"]]
    if not helps:
        return {
            "separable": None,
            "reason": "no cell in the measured set is faster with more groups",
            "helps_n": 0,
            "hurts_n": len(hurts),
            "hurts_min_tgs_per_core_local": min(
                r["few_tgs_per_core_local"] for r in hurts
            ),
        }
    hi_help = max(r["few_tgs_per_core_local"] for r in helps)
    lo_hurt = min(r["few_tgs_per_core_local"] for r in hurts) if hurts else math.inf
    separable = hi_help < lo_hurt
    # Cells that break the threshold: same-or-lower grid but the opposite sign.
    violations = []
    for h in helps:
        for x in hurts:
            if x["few_tgs_per_core_local"] <= h["few_tgs_per_core_local"]:
                violations.append(
                    {
                        "helps": f"{h['shape']} M={h['m']} k={h['k']}",
                        "hurts": f"{x['shape']} M={x['m']} k={x['k']}",
                        "tgs_per_core_local_helps": h["few_tgs_per_core_local"],
                        "tgs_per_core_local_hurts": x["few_tgs_per_core_local"],
                    }
                )
    return {
        "separable": separable,
        "helps_n": len(helps),
        "hurts_n": len(hurts),
        "highest_helping_tgs_per_core_local": hi_help,
        "lowest_hurting_tgs_per_core_local": lo_hurt,
        "boundary_interval_local": [hi_help, lo_hurt] if separable else None,
        "violations": violations,
    }


def e33_control(rows):
    """Positive control: does the E33 sign flip still sit between n=14336 and
    n=16480 at M=6 on this host?"""
    probes = []
    for r in rows:
        if r["shape"] in E33_PROBES and r["m"] == 6:
            probes.append(r)
    probes.sort(key=lambda r: r["n"])
    verdict = "not replicated"
    if len(probes) == 2:
        a, b = probes
        if a["split_helps"] and not b["split_helps"]:
            verdict = "replicated: the flip sits between the two probes"
        elif a["split_helps"] == b["split_helps"]:
            side = "split helps" if a["split_helps"] else "split hurts"
            verdict = (
                f"not replicated: both probes are on the '{side}' side, "
                "so the flip is not between them on this host"
            )
        else:
            verdict = "inverted: the larger grid helps and the smaller hurts"
    return {"probes": probes, "verdict": verdict}


def scored_shape_sides(rows, boundary_local):
    """Which scored shapes sit each side of the boundary at 20 and 40 cores."""
    seen = {}
    for r in rows:
        seen.setdefault((r["shape"], r["n"]), r["few_tgs"])
    out = []
    for (shape, n), tgs in sorted(seen.items(), key=lambda kv: kv[0][1]):
        out.append(
            {
                "shape": shape,
                "n": n,
                "one_group_tgs": n // 8 if n % 8 == 0 else n // 8 + 1,
                "tgs_per_core_20": (n // 8) / CORES_LOCAL,
                "tgs_per_core_40": (n // 8) / CORES_RANKED,
                "boundary_local": boundary_local,
            }
        )
    return out


def main():
    doc, by_key = load_cells()
    rows = contested(by_key)
    boundary = grade_boundary(rows)
    control = e33_control(rows)

    report = {
        "experiment": "e78",
        "analysis": "per-core grid-starvation grading",
        "harness": "local",
        "official_or_ranked_score": False,
        "gate_qualified_for_timing": doc["gate_qualified_for_timing"],
        "cool_gate_passed_real_gate": doc["cool_gate_passed_real_gate"],
        "device": doc["device"],
        "gpu_cores_local": doc["gpu_cores_local"],
        "gpu_cores_ranked": doc["gpu_cores_ranked"],
        "head_sha": doc["head_sha"],
        "source_sha256": doc["source_sha256"],
        "contested_cells": rows,
        "boundary": boundary,
        "e33_positive_control": control,
        "scored_shape_sides": scored_shape_sides(rows, boundary.get("boundary_interval_local")),
    }
    out = ART / "rung2a-percore.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"# E78 per-core grading  (harness=local, {doc['device']}, "
          f"{doc['gpu_cores_local']} cores; ranked host {doc['gpu_cores_ranked']} cores)\n")
    print("## Contested cells: fewer groups vs more groups\n")
    print("| shape | k | n | M | few grp (IPG) | more grp (IPG) | few ms | more ms | "
          "split % | TG/core@20 | TG/core@40 | rel-SEM % | verdict |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda r: (r["few_tgs_per_core_local"], r["m"])):
        print(
            f"| {r['shape']} | {r['k']} | {r['n']} | {r['m']} | "
            f"{r['few_groups']} ({r['few_ipg']}) | {r['more_groups']} ({r['more_ipg']}) | "
            f"{r['few_ms']:.5f} | {r['more_ms']:.5f} | {r['split_pct']:+.2f} | "
            f"{r['few_tgs_per_core_local']:.1f} | {r['few_tgs_per_core_ranked']:.1f} | "
            f"{r['rel_sem_pct']:.3f} | "
            f"{'split HELPS' if r['split_helps'] else 'split hurts'} |"
        )
    print("\n## Boundary grading\n")
    print(json.dumps(boundary, indent=2))
    print("\n## E33 positive control at M=6\n")
    print(control["verdict"])
    for p in control["probes"]:
        print(
            f"  {p['shape']:<34} n={p['n']:<6} 1grp(IPG{p['few_ipg']})={p['few_ms']:.5f} ms  "
            f"2grp(IPG{p['more_ipg']})={p['more_ms']:.5f} ms  {p['split_pct']:+.2f}%  "
            f"TG/core@20={p['few_tgs_per_core_local']:.1f} TG/core@40={p['few_tgs_per_core_ranked']:.1f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
