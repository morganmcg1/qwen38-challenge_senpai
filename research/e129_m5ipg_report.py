#!/usr/bin/env python3
"""E129 -- summarise one width-table QMV probe.

    usage: research/e129_m5ipg_report.py OUT_DIR [--histogram 4:16,5:20,...]

OUT_DIR is a `research/e120_probe.sh` output directory holding `m5ipg.json`
and `m5ipg-exact.json`.

The timing arms are `a_compare` (the candidate width table) and `b_shipped`.
Each block times both arms forward and then in reverse, so the per-block arm
mean is the palindromic mean and monotone thermal drift cancels to first
order. `harness=local`, ungated: `cool_gate_passed_real_gate=false`.

`compare_gain` is the fraction of shipped time the candidate table saves,
`(b - a) / b`, positive when the candidate is faster. Cells whose pass count
is unchanged execute byte-identical code in both pipelines, so they are
controls that isolate the shared entry point's occupancy channel; cells whose
pass count changes are treated.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

DEFAULT_HISTOGRAM = {4: 16, 5: 20, 6: 20, 7: 12, 8: 240}

# Wide affine-4/group-64 matvecs per decode round, by call site. 64 layers:
# 48 gated-DeltaNet and 16 full-attention, plus one lm_head.
CALLS_PER_ROUND = {
    "mlp.gate_up": 64,
    "mlp.down": 64,
    "gdn.in_proj": 48,
    "gdn.out_proj": 48,
    "fa.qkv": 16,
    "fa.o_proj": 16,
    "lm_head": 1,
}

# `gdn.out_proj` and `fa.o_proj` are the same (6144, 5120) cell. Measuring one
# of them is enough; the names are kept apart only so the report names the call
# site.
SHAPE_ALIAS = {"gdn.out_proj": "fa.o_proj"}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def arm_mean(arm: dict) -> float:
    return 0.5 * (arm["forward_us"] + arm["reverse_us"])


def summarise_timing(payload: dict) -> dict:
    cells: dict[tuple[str, int], list[dict]] = {}
    for record in payload["cells"]:
        cells.setdefault((record["shape"], record["width"]), []).append(record)

    rows = []
    for (shape, width), blocks in sorted(cells.items()):
        fractions = []
        saved = []
        a_us = []
        b_us = []
        for block in blocks:
            arms = {arm["arm"]: arm_mean(arm) for arm in block["arms"]}
            a_name = next(name for name in arms if name.startswith("a_"))
            b_name = next(name for name in arms if name.startswith("b_"))
            a_us.append(arms[a_name])
            b_us.append(arms[b_name])
            saved.append(arms[a_name] - arms[b_name])
            fractions.append((arms[a_name] - arms[b_name]) / arms[a_name])
        temps = [b["gpu_temp_entry_c"] for b in blocks if "gpu_temp_entry_c" in b]
        gains = [-f / (1 - f) for f in fractions]
        head = blocks[0]
        rows.append(
            {
                "shape": shape,
                "width": width,
                "blocks": len(blocks),
                "replicates": head["replicates"],
                "passes_shipped": head.get("passes_shipped"),
                "passes_compare": head.get("passes_compare"),
                "treated": head.get("passes_shipped") != head.get("passes_compare"),
                "a_us_median": statistics.median(a_us),
                "b_us_median": statistics.median(b_us),
                "saved_us_median": statistics.median(saved),
                "saved_fraction_median": statistics.median(fractions),
                "compare_gain_median": statistics.median(gains),
                "compare_gain_min": min(gains),
                "compare_gain_max": max(gains),
                "compare_sign_agree": sum(1 for g in gains if g > 0),
                "gpu_temp_entry_min_c": min(temps) if temps else None,
                "gpu_temp_entry_max_c": max(temps) if temps else None,
            }
        )
    return {"cells": rows}


def control_summary(rows: list[dict]) -> dict:
    """Occupancy-only channel: the cells whose pass count did not change."""
    gains = [row["compare_gain_median"] for row in rows if not row["treated"]]
    if len(gains) < 2:
        return {"cells": len(gains)}
    mean = statistics.fmean(gains)
    sd = statistics.stdev(gains)
    return {
        "cells": len(gains),
        "mean_gain": mean,
        "sd": sd,
        "sem": sd / len(gains) ** 0.5,
        "positive_cells": sum(1 for g in gains if g > 0),
    }


def round_model(rows: list[dict], histogram: dict[int, int]) -> dict:
    """Weight the per-cell saving by the routed rounds of one decode leg."""
    by_key = {(row["shape"], row["width"]): row for row in rows}
    total_rounds = sum(histogram.values())
    base_us = 0.0
    saved_us = 0.0
    missing = []
    for width, rounds in sorted(histogram.items()):
        for shape, calls in CALLS_PER_ROUND.items():
            row = by_key.get((shape, width))
            if row is None and shape in SHAPE_ALIAS:
                row = by_key.get((SHAPE_ALIAS[shape], width))
            if row is None:
                missing.append(f"{shape}@M={width}")
                continue
            base_us += rounds * calls * row["b_us_median"]
            saved_us -= rounds * calls * row["saved_us_median"]
    return {
        "histogram": {str(k): v for k, v in sorted(histogram.items())},
        "routed_rounds": total_rounds,
        "wide_qmv_us_shipped": base_us,
        "wide_qmv_us_saved_by_compare": saved_us,
        "wide_qmv_gain_fraction": saved_us / base_us if base_us else None,
        "us_saved_per_routed_round": saved_us / total_rounds if total_rounds else None,
        "missing_cells": missing,
    }


def summarise_exactness(payload: dict) -> dict:
    records = payload["records"]
    return {
        "cells": len(records),
        "elements": sum(r["elements"] for r in records),
        "differing_elements": sum(r["differing_elements"] for r in records),
        "shipped_vs_mlx_differing": sum(r["shipped_vs_mlx"] for r in records),
        "compare_vs_mlx_differing": sum(r["compare_vs_mlx"] for r in records),
        "max_abs_diff": max(r["max_abs_diff"] for r in records),
        "bit_exact_cells": sum(1 for r in records if r["bit_exact"]),
        "x_hit_min": min(r["x_hit"] for r in records),
        "controls_can_fail": all(r["positive_control_can_fail"] for r in records),
        "widths": sorted({r["width"] for r in records}),
        "shapes": sorted({r["shape"] for r in records}),
    }


def parse_histogram(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for token in text.split(","):
        width, _, rounds = token.partition(":")
        out[int(width)] = int(rounds)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    histogram = DEFAULT_HISTOGRAM
    if "--histogram" in sys.argv:
        histogram = parse_histogram(sys.argv[sys.argv.index("--histogram") + 1])

    report: dict = {"out_dir": str(out_dir), "harness": "local"}

    exact_path = out_dir / "m5ipg-exact.json"
    if exact_path.exists():
        report["exactness"] = summarise_exactness(load(exact_path))

    timing_path = out_dir / "m5ipg.json"
    if timing_path.exists():
        payload = load(timing_path)
        timing = summarise_timing(payload)
        report["cases_shipped"] = payload["cases_shipped"]
        report["cases_compare"] = payload["cases_compare"]
        report["passes_shipped"] = payload["passes_shipped"]
        report["passes_compare"] = payload["passes_compare"]
        report["cool_gate_passed_real_gate"] = payload["cool_gate_passed_real_gate"]
        report["gate_qualified_for_timing"] = payload["gate_qualified_for_timing"]
        report["timing"] = timing
        report["occupancy_control"] = control_summary(timing["cells"])
        report["round_model"] = round_model(timing["cells"], histogram)

    out_path = out_dir / "m5ipg-report.json"
    with out_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if "exactness" in report:
        exact = report["exactness"]
        print(
            f"exactness: {exact['cells']} cells, {exact['elements']} elements, "
            f"{exact['differing_elements']} differing, "
            f"max_abs_diff={exact['max_abs_diff']}, "
            f"controls_can_fail={exact['controls_can_fail']}, "
            f"x_hit_min={exact['x_hit_min']}"
        )
    if "timing" in report:
        print(f"\nshipped {report['cases_shipped']}")
        print(f"compare {report['cases_compare']}   positive gain = compare is faster")
        print(f"{'shape':<14}{'M':>3}{'pass':>6}{'shipped_us':>12}{'compare_us':>12}"
              f"{'gain_%':>9}{'sign':>7}")
        for row in report["timing"]["cells"]:
            passes = f"{row['passes_shipped']}>{row['passes_compare']}"
            print(
                f"{row['shape']:<14}{row['width']:>3}{passes:>6}"
                f"{row['b_us_median']:>12.3f}{row['a_us_median']:>12.3f}"
                f"{100 * row['compare_gain_median']:>9.3f}"
                f"{row['compare_sign_agree']:>4}/{row['blocks']}"
            )
        control = report["occupancy_control"]
        if "mean_gain" in control:
            print(
                f"\noccupancy-only control ({control['cells']} unchanged-pass cells): "
                f"mean gain {100 * control['mean_gain']:+.4f} %, "
                f"sd {100 * control['sd']:.4f}, sem {100 * control['sem']:.4f}, "
                f"{control['positive_cells']}/{control['cells']} positive"
            )
        model = report["round_model"]
        print(
            f"\nrouted-round model over {model['routed_rounds']} rounds: "
            f"wide QMV {model['wide_qmv_us_shipped']:.0f} us shipped, compare saves "
            f"{model['wide_qmv_us_saved_by_compare']:.0f} us "
            f"({100 * model['wide_qmv_gain_fraction']:+.3f} %), "
            f"{model['us_saved_per_routed_round']:+.1f} us/round"
        )
        if model["missing_cells"]:
            print(f"missing cells: {', '.join(model['missing_cells'])}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
