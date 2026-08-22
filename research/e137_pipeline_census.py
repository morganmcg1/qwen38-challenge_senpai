#!/usr/bin/env python3
"""E137 F2 step 1: what `MLX_E120_QMV_PIPELINE_LOG` can and cannot show.

THE QUESTION F2 ASKED. At M >= 6, do some scored linear cells decline
`routable` (`Qwen35.swift:2109-2135`) and fall into the generic gate, where
`quantized.h:1938-1948` gives `case 6: <T, 6, 3, true>`, ipg 3, two weight
passes? F2 set a stop condition: post immediately if width 6 shows fewer than
257 dispatches.

WHY THAT STOP CONDITION CANNOT BE READ OFF THIS FILE. `notePipeline`
(`Qwen35.swift:1999-2013`, read this session) does increment
`pipelineWidths[width]` on every routed dispatch, so the in-memory counter is
a true per-dispatch count. The FILE is not. `flushPipelineLog` runs only when
`isNew` is set, that is only when a key or a width is seen for the FIRST time,
plus once from the `atexit` handler registered at `Qwen35.swift:1977`.
`QwenRuntimeWorker.swift:1905` sends the worker `terminate()` (SIGTERM) and
`:1911` escalates to SIGKILL, and neither runs `atexit`. The last flush is
therefore the first dispatch of the last new width, which `warmAllDepthShapes`
(`Qwen36MTPBlockSession.swift:416-441`) reaches at the end of warm-up, before
any scored token. Every scored dispatch increments a counter that is never
written to disk.

WHAT THE FILE DOES PROVE. It is exactly the warm-up gate its own doc comment
claims to be. `warmAllDepthShapes` runs one throwaway target forward at each
width in ascending order, so `first_index_by_width` must be an arithmetic
progression whose step is the routed cell count of one forward. A pipeline
first compiled inside the timed window would break that progression by orders
of magnitude. This script tests the progression and reports the snapshot point
instead of pretending to a scored-window count.

A target verify forward at width M issues exactly 257 linear cells:

    gdn.in_proj  48   gdn.out_proj 48   fa.qkv 16   fa.o_proj 16
    mlp.gate_up  64   mlp.down     64   lm_head 1                 = 257
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

    # `pipelineDispatches` only counts width-carrying dispatches, so the
    # snapshot ordinal is the sum over widths and excludes the `xsums_v1` fill.
    snapshot_ordinal = sum(by_width.values())
    routed_widths = sorted(first_index)
    ordinals = [first_index[m] for m in routed_widths]
    gaps = [b - a for a, b in zip(ordinals, ordinals[1:])]
    progression_ok = bool(gaps) and all(
        g == LINEAR_CELLS_PER_FORWARD for g in gaps)
    # Warm-up alone accounts for one full forward at every routed width below
    # the last, plus the single dispatch that triggered the final flush.
    warmup_only = LINEAR_CELLS_PER_FORWARD * (len(routed_widths) - 1) + 1
    snapshot_is_warmup = snapshot_ordinal == warmup_only

    rows = []
    for m in sorted(set(by_width) | set(widths)):
        claimed = by_width.get(m, 0)
        rounds = widths.get(m, 0)
        rows.append({
            "m": m,
            "dispatches_at_snapshot": claimed,
            "equals_one_forward": claimed == LINEAR_CELLS_PER_FORWARD,
            "scored_rounds_at_this_width": rounds,
            "scored_linear_cells_unobservable":
                LINEAR_CELLS_PER_FORWARD * rounds,
            "first_dispatch_ordinal": first_index.get(m),
            "warmed_before_any_scored_token":
                first_index.get(m) is not None
                and first_index[m] < snapshot_ordinal,
        })

    result = {
        "experiment": "e137-f2-step1-route-b-coverage",
        "tag": args.tag,
        "question":
            "does every scored linear cell reach Route B at every realised "
            "verify width, or do some decline `routable` at M >= 6?",
        "answer_status": "unobservable_with_this_instrument",
        "instrument_limit":
            "flushPipelineLog runs only on a first-seen key or width plus an "
            "atexit handler; QwenRuntimeWorker.swift:1905 SIGTERMs the worker "
            "and :1911 SIGKILLs it, so atexit never runs and the on-disk file "
            "is frozen at the first dispatch of the last new width.",
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
            "total_key_increments": pipelines.get("dispatches"),
            "by_key": pipelines.get("by_key"),
            "first_index_by_key": pipelines.get("first_index_by_key"),
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
        "warmup_gate": {
            "snapshot_ordinal": snapshot_ordinal,
            "warmup_only_prediction": warmup_only,
            "snapshot_is_end_of_warmup": snapshot_is_warmup,
            "routed_widths": routed_widths,
            "first_dispatch_ordinals": ordinals,
            "ordinal_gaps": gaps,
            "gaps_equal_one_forward": progression_ok,
            "every_routed_width_warmed_before_scored_window": all(
                row["warmed_before_any_scored_token"] for row in rows
                if row["first_dispatch_ordinal"] is not None),
            "no_pipeline_first_compiled_in_timed_window":
                progression_ok and snapshot_is_warmup,
        },
        "verdict": {
            "f2_stop_condition": "width 6 shows fewer than 257 dispatches",
            "width_6_dispatches_at_snapshot": by_width.get(6),
            "f2_stop_condition_fired":
                by_width.get(6, 0) < LINEAR_CELLS_PER_FORWARD,
            "width_6_identical_to_every_other_fully_warmed_width": len({
                v for m, v in by_width.items() if m in routed_widths[:-1]
            }) == 1,
            "hypothesis_scored_cells_decline_routable_at_m6":
                "not_testable_with_this_instrument",
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
          f"{pipelines.get('qmv_specializations')} specializations")
    print()
    print("  M   dispatch@snapshot  =1fwd   rounds   first   warmed")
    for row in rows:
        print(f"{row['m']:>3} {row['dispatches_at_snapshot']:>18} "
              f"{str(row['equals_one_forward']):>6} "
              f"{row['scored_rounds_at_this_width']:>8} "
              f"{str(row['first_dispatch_ordinal']):>7} "
              f"{str(row['warmed_before_any_scored_token']):>7}")
    print()
    print(f"snapshot ordinal {snapshot_ordinal}, warm-up-only prediction "
          f"{warmup_only}, match {snapshot_is_warmup}")
    print(f"first-dispatch gaps {gaps}, all equal one forward {progression_ok}")
    print("F2 stop condition (width 6 < 257) fired: "
          f"{result['verdict']['f2_stop_condition_fired']}")
    print("scored-window routing: not testable with this instrument")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
