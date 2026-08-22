#!/usr/bin/env python3
"""E137 F2 step 1: Route B coverage per realised verify width, from one leg.

THE QUESTION. F2 section 5: at M >= 6, do some scored linear cells decline
`routable` (`Qwen35.swift:2109-2135`) and fall into the generic gate, where
`quantized.h:1938-1948` gives `case 6: <T, 6, 3, true>`, ipg 3, two weight
passes? If they do, that is both the M=5 to M=6 step and the FINDING 156
shortfall in one fact.

THE INSTRUMENT. `Qwen35CustomQMV.notePipeline` (`Qwen35.swift:1998-2013`, read
this session) increments `pipelineWidths[width]` once for every dispatch Route
B CLAIMS at that width, and never for a cell that fell back. A target verify
forward at width M issues exactly 257 linear cells:

    gdn.in_proj  48   gdn.out_proj 48   fa.qkv 16   fa.o_proj 16
    mlp.gate_up  64   mlp.down     64   lm_head 1                 = 257

THE TEST. The same leg's trace gives `R[M]`, the number of scored verify
forwards at width M. Route B's own count minus `257 * R[M]` must leave only
warm-up forwards and proposal-head forwards, both small and both positive. A
width whose residual is negative, or far below its neighbours', is a width at
which the scored path left Route B.

The residual is NOT zero by construction: `warmAllDepthShapes`
(`Qwen36MTPBlockSession.swift:416-441`) runs one throwaway target forward at
every width 1 through maxDepth+1 plus one extra at width 3, and the proposal
head runs its own routed cells over committed history rows whose count also
lands in 3...9. Both only ADD. The comparison across widths is therefore the
test, not the absolute value.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

FIELD = re.compile(r"(\w+)=(-?\d+)")
ROOT = pathlib.Path(__file__).resolve().parent.parent

LINEAR_CELLS_PER_FORWARD = 257
SHAPES = {
    "linear_attn.in_proj_fused_qkvzba": 48,
    "linear_attn.out_proj": 48,
    "full_attn.qkv_proj_fused": 16,
    "full_attn.o_proj": 16,
    "mlp.gate_up_fused": 64,
    "mlp.down": 64,
    "head.lm_head": 1,
}


def read_widths(trace: pathlib.Path) -> tuple[Counter, int]:
    """Realised verify width per scored round, `M = d + 1`."""
    widths: Counter = Counter()
    tokens = 0
    for line in open(trace, errors="replace"):
        if not line.startswith("mtp-trace: round="):
            continue
        fields = {k: int(v) for k, v in FIELD.findall(line)}
        widths[fields["d"] + 1] += 1
        tokens += fields["acc"] + 1
    return widths, tokens


def read_meta(path: pathlib.Path) -> dict:
    meta = {}
    if not path.exists():
        return meta
    for line in open(path, errors="replace"):
        if "=" in line:
            key, value = line.rstrip("\n").split("=", 1)
            meta[key] = value
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="e137pipe512")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    leg = ROOT / "research" / "out" / args.tag
    pipeline_path = ROOT / "research" / "out" / f"{args.tag}-pipelines.json"
    if not pipeline_path.exists():
        raise SystemExit(f"no pipeline log at {pipeline_path}")
    pipelines = json.loads(pipeline_path.read_text())
    widths, tokens = read_widths(leg / "trace.txt")
    meta = read_meta(leg / "meta.txt")

    by_width = {int(k): v for k, v in pipelines["by_width"].items()}
    first_index = {
        int(k): v for k, v in pipelines.get("first_index_by_width", {}).items()
    }

    rows = []
    for m in sorted(set(by_width) | set(widths)):
        claimed = by_width.get(m, 0)
        rounds = widths.get(m, 0)
        scored = LINEAR_CELLS_PER_FORWARD * rounds
        residual = claimed - scored
        rows.append({
            "m": m,
            "route_b_dispatches": claimed,
            "scored_rounds_at_this_width": rounds,
            "scored_linear_cells": scored,
            "residual_over_scored": residual,
            "implied_forwards": round(claimed / LINEAR_CELLS_PER_FORWARD, 3),
            "first_dispatch_ordinal": first_index.get(m),
            "route_b_covered_every_scored_cell": residual >= 0,
        })

    covered = all(row["route_b_covered_every_scored_cell"] for row in rows)
    # A decline that took a whole shape out at one width would show as a
    # residual hundreds below its neighbours. Report the spread so a reader can
    # see that no width is an outlier rather than only that none is negative.
    residuals = {row["m"]: row["residual_over_scored"] for row in rows}
    at6 = residuals.get(6)
    neighbours = [residuals[m] for m in (5, 7) if m in residuals]

    result = {
        "experiment": "e137-f2-step1-route-b-coverage",
        "tag": args.tag,
        "question":
            "does every scored linear cell reach Route B at every realised "
            "verify width, or do some decline `routable` at M >= 6?",
        "not_a_timing_leg":
            "MLX_E120_QMV_PIPELINE_LOG adds a host dictionary update to every "
            "routed dispatch; this leg's seconds per token are not comparable "
            "with any timed arm.",
        "linear_cells_per_forward": LINEAR_CELLS_PER_FORWARD,
        "shape_cell_counts": SHAPES,
        "route": {
            "arm": pipelines.get("arm"),
            "entry": pipelines.get("entry"),
            "table": pipelines.get("table"),
            "grid": pipelines.get("grid"),
            "plan": pipelines.get("plan"),
            "default_route": pipelines.get("default_route"),
            "qmv_specializations": pipelines.get("qmv_specializations"),
            "total_dispatches": pipelines.get("dispatches"),
            "by_key": pipelines.get("by_key"),
        },
        "leg": {
            "base_sha": meta.get("base_sha"),
            "dirty": meta.get("dirty"),
            "tokens": meta.get("tokens"),
            "local_mode": meta.get("local_mode"),
            "chip": meta.get("chip"),
            "head_dir": meta.get("head_dir"),
            "worker_sha256": meta.get("worker_sha256"),
            "cool_gate": meta.get("cool_gate"),
            "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate"),
            "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
            "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
            "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
            "scored_rounds": sum(widths.values()),
            "scored_tokens": tokens,
            "width_histogram": dict(sorted(widths.items())),
        },
        "per_width": rows,
        "verdict": {
            "route_b_covered_every_scored_cell_at_every_width": covered,
            "width_6_residual": at6,
            "width_5_and_7_residuals": neighbours,
            "hypothesis_scored_cells_decline_routable_at_m6":
                "alive" if (at6 is not None and neighbours
                            and at6 < min(neighbours) - LINEAR_CELLS_PER_FORWARD)
                else "dead",
        },
    }

    out = pathlib.Path(args.out) if args.out else (
        ROOT / "research" / "e137-artifacts" / "f2-step1-route-b-coverage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))

    print(f"leg {args.tag}  base {result['leg']['base_sha']}  "
          f"{result['leg']['scored_rounds']} rounds  "
          f"{result['leg']['scored_tokens']} tokens")
    print(f"route {pipelines.get('default_route')}  "
          f"{pipelines.get('qmv_specializations')} specializations  "
          f"{pipelines.get('dispatches')} routed dispatches")
    print()
    print("  M   routeB   rounds   scored   residual   forwards   first")
    for row in rows:
        print(f"{row['m']:>3} {row['route_b_dispatches']:>8} "
              f"{row['scored_rounds_at_this_width']:>8} "
              f"{row['scored_linear_cells']:>8} "
              f"{row['residual_over_scored']:>10} "
              f"{row['implied_forwards']:>10} "
              f"{str(row['first_dispatch_ordinal']):>7}")
    print()
    print("every scored cell reached Route B at every width: "
          f"{covered}")
    print("hypothesis 'scored cells decline routable at M>=6': "
          f"{result['verdict']['hypothesis_scored_cells_decline_routable_at_m6']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
