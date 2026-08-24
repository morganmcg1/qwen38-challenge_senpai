#!/usr/bin/env python3
"""E173: assemble the inventory of the M-independent per-round fixed cost `F`.

Inputs (all optional; a missing input is reported as a gap, never imputed):
  research/e173-artifacts/admission.json  host admission-path census
  research/e173-artifacts/gpu.json        non-weight-pass GPU line items
  research/e173-artifacts/tablepays.json  tablePays natural experiment

Every row carries its own method and side tag. `harness=local` throughout: the
target is F_local = 11.792 ms/round (FINDING 404, M4 Pro class), NOT the ranked
F = 6.899 ms. No ranked transfer is claimed here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

ART = pathlib.Path(__file__).resolve().parent / "e173-artifacts"

# FINDING 404, harness=local, M4 Pro class host (g16s). The fixed term of the
# local round law. This is the budget the inventory has to explain.
F_LOCAL_MS = 11.792

# FINDING 363 / 399 / 405, harness=local, prior campaign measurements on this
# host class. Carried as prior-ledger rows so the inventory is comparable with
# earlier work instead of silently re-deriving it.
PRIOR_HOST_ROWS = [
    {
        "name": "protocol_gap",
        "mechanism": "worker protocol round-trip floor between parent and worker",
        "ms_per_round": 0.3566,
        "method": "prior-ledger (FINDING 399 floor)",
        "side": "host",
    },
    {
        "name": "commit",
        "mechanism": "commit of the pending primary token at round entry",
        "ms_per_round": 0.4331,
        "method": "prior-ledger (FINDING 405)",
        "side": "host",
    },
    {
        "name": "post_eval_host_tail",
        "mechanism": "host work after the verify eval returns (row ledger, "
        "acceptance decision, response encode)",
        "ms_per_round": 0.655,
        "method": "prior-ledger (FINDING 363, 635-675 us range midpoint)",
        "side": "host",
    },
    {
        "name": "head_chain_fixed",
        "mechanism": "proposal-head chain per-round fixed component",
        "ms_per_round": 1.2288,
        "method": "prior-ledger (FINDING 363)",
        "side": "mixed",
    },
]


def load(name: str) -> dict[str, Any] | None:
    path = ART / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares intercept and slope. Intercept is the M-independent part."""
    n = len(xs)
    if n < 2:
        return ys[0], 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def admission_rows(payload: dict[str, Any]) -> tuple[list[dict], dict]:
    """Price the admission path per drafting round.

    Every one of the 257 routed cells enters `routable` once per weight pass.
    At m >= 4 the call is accepted; at m = 1 (and at prefill) the same 257
    entries are refused at `widths.contains(m)`. Both are host-only costs that
    sit inside `F`.
    """
    rows: list[dict] = []
    by_m: dict[int, list[dict]] = {}
    for r in payload["routable"]:
        by_m.setdefault(r["m"], []).append(r)

    detail: dict[str, Any] = {"cells_per_round": 0, "per_m_us": {}}
    for m, entries in sorted(by_m.items()):
        total_ns = sum(e["host_ns_per_call"] * e["calls_per_round"] for e in entries)
        cells = sum(e["calls_per_round"] for e in entries)
        accepted = sum(e["calls_per_round"] for e in entries if e["accepted"])
        detail["cells_per_round"] = cells
        detail["per_m_us"][m] = {
            "total_us": total_ns / 1e3,
            "cells": cells,
            "accepted_cells": accepted,
            "refused_cells": cells - accepted,
            "us_per_cell": total_ns / 1e3 / cells,
        }

    drafting = detail["per_m_us"].get(4) or detail["per_m_us"][max(detail["per_m_us"])]
    rows.append(
        {
            "name": "admission.routable",
            "mechanism": f"Qwen35CustomQMV.routable over {detail['cells_per_round']} "
            "routed cells per weight pass (m=4: all accepted, 0 refusals)",
            "ms_per_round": drafting["total_us"] / 1e3,
            "method": "measured-host (E173 admission census, no eval, no device buffer)",
            "side": "host",
        }
    )

    sidecar = payload.get("xsums_sidecar") or []
    if sidecar:
        # `take` scans all 8 slots with weak loads under an NSLock on every
        # call. Miss is the worst case and is what a cold round pays.
        worst = max(s["host_ns_per_call"] for s in sidecar)
        rows.append(
            {
                "name": "admission.xsums_sidecar_take",
                "mechanism": "Qwen35XSumsSidecar.take: NSLock plus 8 weak-slot "
                f"scan, once per routed cell ({detail['cells_per_round']} cells)",
                "ms_per_round": worst * detail["cells_per_round"] / 1e6,
                "method": "measured-host (E173 admission census, worst of hit/miss)",
                "side": "host",
            }
        )

    pair = payload.get("counter_increment_pair")
    if pair:
        rows.append(
            {
                "name": "admission.counter_increments",
                "mechanism": "qwen35XSumsStandaloneFills / SidecarHits unsynchronised "
                "counter pair, incremented per routed cell at m >= 4",
                "ms_per_round": pair["host_ns_per_call"]
                * detail["cells_per_round"]
                / 1e6,
                "method": "measured-host (E173 admission census)",
                "side": "host",
            }
        )

    build = payload.get("routed_entry_point_graph_build") or []
    if build:
        by_bm: dict[int, float] = {}
        for b in build:
            by_bm.setdefault(b["m"], 0.0)
            by_bm[b["m"]] += b["host_ns_per_call"] * b["calls_per_round"]
        drafting_m = 4 if 4 in by_bm else max(by_bm)
        rows.append(
            {
                "name": "admission.routed_graph_build",
                "mechanism": "qwen35RoutedQuantizedMM host graph construction for "
                f"all {detail['cells_per_round']} cells (lazy, no eval)",
                "ms_per_round": by_bm[drafting_m] / 1e6,
                "method": "measured-host (E173 admission census)",
                "side": "host",
            }
        )
    return rows, detail


def gpu_rows(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Split each GPU line item into an M-independent intercept and an M slope.

    Only the intercept belongs in `F`. The slope is the `pass*groups(M)` and
    `h*d` part of the round law and is reported separately so the two are never
    conflated.
    """
    groups: dict[str, list[dict]] = {}
    for item in payload["items"]:
        groups.setdefault(item["name"], []).append(item)

    rows: list[dict] = []
    slopes: list[dict] = []
    for name, items in groups.items():
        items.sort(key=lambda i: i["m"])
        xs = [float(i["m"]) for i in items]
        ys = [float(i["ms_per_round"]) for i in items]
        intercept, slope = linear_fit(xs, ys)
        family = items[0]["family"]
        calls = items[0]["calls_per_round"]
        rows.append(
            {
                "name": name,
                "mechanism": f"{items[0]['note']} ({calls} calls/round, "
                f"family {family})",
                "ms_per_round": max(0.0, intercept),
                "method": f"measured-gpu (E173 GPU line items, intercept of a "
                f"linear fit over m in {[int(x) for x in xs]})",
                "side": "gpu",
            }
        )
        slopes.append(
            {
                "name": name,
                "ms_per_round_per_m": slope,
                "measured_ms_per_round": {int(i["m"]): i["ms_per_round"] for i in items},
            }
        )
    return rows, slopes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "f-inventory.json"))
    args = ap.parse_args()

    admission = load("admission.json")
    gpu = load("gpu.json")
    tablepays = load("tablepays.json")

    rows: list[dict] = []
    gaps: list[str] = []
    detail: dict[str, Any] = {}
    slopes: list[dict] = []

    if admission:
        a_rows, detail = admission_rows(admission)
        rows.extend(a_rows)
    else:
        gaps.append("admission.json missing: admission path unpriced")

    if gpu:
        g_rows, slopes = gpu_rows(gpu)
        rows.extend(g_rows)
    else:
        gaps.append("gpu.json missing: GPU line items unpriced")

    rows.extend(PRIOR_HOST_ROWS)

    explained = sum(r["ms_per_round"] for r in rows)
    host = sum(r["ms_per_round"] for r in rows if r["side"] == "host")
    gpu_side = sum(r["ms_per_round"] for r in rows if r["side"] == "gpu")
    mixed = sum(r["ms_per_round"] for r in rows if r["side"] == "mixed")
    residual = F_LOCAL_MS - explained

    over_bar = sorted(
        (r for r in rows if r["ms_per_round"] >= 0.5),
        key=lambda r: -r["ms_per_round"],
    )

    verdict = (
        "instrument-not-seeing-F"
        if residual > 1.5
        else ("diffuse" if not over_bar else "named-line-items-found")
    )

    report = {
        "experiment": "e173-decompose-the-per-round-fixed-cost",
        "harness": "local",
        "target_f_ms": F_LOCAL_MS,
        "target_source": "FINDING 404 fixed term, M4 Pro class (g16s)",
        "ranked_f_ms_not_claimed": 6.899,
        "rows": sorted(rows, key=lambda r: -r["ms_per_round"]),
        "explained_ms": explained,
        "host_ms": host,
        "gpu_ms": gpu_side,
        "mixed_ms": mixed,
        "residual_ms": residual,
        "items_at_or_above_0p5ms": [r["name"] for r in over_bar],
        "verdict": verdict,
        "admission_detail": detail,
        "m_slopes_excluded_from_F": slopes,
        "tablepays": (
            {
                "boundary_curvature_ms": tablepays.get("boundary_contrast"),
                "controlled_m3_to_m4_increment": tablepays.get("restricted_m3_m4"),
                "uncontrolled_increments_ms": tablepays.get(
                    "uncontrolled_increments_ms"
                ),
            }
            if tablepays
            else None
        ),
        "gaps": gaps,
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"F_local target      {F_LOCAL_MS:8.3f} ms/round  (harness=local)")
    print(f"explained           {explained:8.3f} ms  "
          f"(host {host:.3f} / gpu {gpu_side:.3f} / mixed {mixed:.3f})")
    print(f"residual            {residual:8.3f} ms")
    print(f"verdict             {verdict}")
    print()
    print(f"{'ms/round':>9}  {'side':<6} {'name':<34} method")
    for r in report["rows"]:
        print(f"{r['ms_per_round']:9.3f}  {r['side']:<6} {r['name']:<34} {r['method']}")
    for g in gaps:
        print(f"GAP: {g}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
