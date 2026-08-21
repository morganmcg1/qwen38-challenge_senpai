#!/usr/bin/env python3
"""E105 rung 1: split each dispatch family's cost into launch / memory / residual.

Two measurement FRAMES are in play and they must never be mixed silently.

Frame A -- census leg (`MLX_E58_BUFFER_LIMIT_OPS=0`, `MLX_E58_BUFFER_LIMIT_MB=1`).
    Every dispatch is committed on its own command buffer, so
    `GPUEndTime - GPUStartTime` is that dispatch's ISOLATED GPU interval,
    including the buffer's own start/end ramp. Forced serialisation inflates
    both the per-family cost AND the GPU-busy round by roughly the same ramp,
    so Frame A is valid for SHARES and for STRUCTURE (dispatch counts, grid,
    threadgroups, bytes) and invalid as an absolute free-running time budget.

Frame B -- dose leg (ordinary `--local-iterate`, no census instrument).
    F is the marginal wall-clock cost of one extra dispatch and the round comes
    from the same legs' seconds per token. The promotion arithmetic
    `N * F / R` lives here.

This tool reports the Frame A decomposition. It takes the empty-dispatch floor
from the cheapest census family rather than from Frame B, because importing a
Frame B constant into a Frame A subtraction is exactly the error the frames
exist to prevent.

    usage: research/e105_rung1_decompose.py CENSUS_REPORT_JSON [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# bf16 multiply-add throughput is ~4 orders of magnitude above what these
# kernels ask for, so the arithmetic term is reported as a bound, not a fit.
ARITHMETIC_BOUND_US = 0.2


def decompose(report: dict) -> dict:
    families = report["families"]
    if not families:
        raise SystemExit("census report has no families")

    # The empty-dispatch floor. The cheapest family is the closest thing to a
    # null kernel that the live path actually contains: its roofline share is
    # the smallest and its threadgroup count is the smallest, so whatever it
    # costs above its own bytes is dispatch machinery, not work.
    floor_family = min(families, key=lambda f: f["us_per_dispatch"])
    floor_us = floor_family["us_per_dispatch"]

    rows = []
    for fam in families:
        total = fam["us_per_dispatch"]
        memory = fam["roofline_us_per_dispatch"]
        # Only the part of the floor that is not itself memory traffic counts
        # as launch, or the floor family's own bytes get charged twice.
        launch = floor_us - floor_family["roofline_us_per_dispatch"]
        residual = total - launch - memory
        n = fam["dispatches_per_round"]
        rows.append(
            {
                "family": fam["family"],
                "label": fam["label"],
                "dispatches_per_round": n,
                "threadgroup_count": fam["threadgroup_count"],
                "waves_over_cores": fam["waves_over_cores"],
                "us_per_dispatch": total,
                "launch_us": launch,
                "memory_us": memory,
                "arithmetic_bound_us": ARITHMETIC_BOUND_US,
                "residual_us": residual,
                "residual_share_of_dispatch": 100.0 * residual / total,
                "launch_us_per_round": launch * n,
                "memory_us_per_round": memory * n,
                "residual_us_per_round": residual * n,
            }
        )

    # Anchors are (name, rounds observed, us per round). The value is already
    # divided by the round count.
    want = f"w{report['width']}"
    round_us = next(
        (val for name, _, val in report.get("width_anchors", []) if name == want),
        None,
    )
    if round_us is None:
        raise SystemExit(f"no width anchor named {want}")

    totals = {
        "launch_us_per_round": sum(r["launch_us_per_round"] for r in rows),
        "memory_us_per_round": sum(r["memory_us_per_round"] for r in rows),
        "residual_us_per_round": sum(r["residual_us_per_round"] for r in rows),
    }
    totals["total_us_per_round"] = sum(totals.values())

    return {
        "frame": "A (census, forced one dispatch per command buffer)",
        "tag": report["tag"],
        "width": report["width"],
        "frame_a_round_us": round_us,
        "floor_family": floor_family["family"],
        "floor_us_per_dispatch": floor_us,
        "empty_dispatch_launch_us": floor_us - floor_family["roofline_us_per_dispatch"],
        "rows": rows,
        "totals": totals,
        "pools_pct_of_frame_a_round": {
            k.replace("_us_per_round", ""): 100.0 * v / round_us
            for k, v in totals.items()
        },
    }


def render(out: dict) -> str:
    lines = [
        f"E105 rung 1 decomposition -- frame {out['frame']}",
        f"  tag {out['tag']}  width {out['width']}  round {out['frame_a_round_us']:,.1f} us",
        f"  empty-dispatch launch floor {out['empty_dispatch_launch_us']:.2f} us"
        f" (from {out['floor_family']})",
        "",
        f"{'family':<16} {'n':>4} {'tg':>5} {'wav':>6} {'total':>7} {'launch':>7}"
        f" {'mem':>6} {'resid':>7} {'resid%':>7} {'resid/rnd':>10}",
    ]
    for r in out["rows"]:
        lines.append(
            f"{r['family']:<16} {r['dispatches_per_round']:>4.0f}"
            f" {r['threadgroup_count']:>5d} {r['waves_over_cores']:>6.2f}"
            f" {r['us_per_dispatch']:>7.2f} {r['launch_us']:>7.2f}"
            f" {r['memory_us']:>6.2f} {r['residual_us']:>7.2f}"
            f" {r['residual_share_of_dispatch']:>6.1f}% {r['residual_us_per_round']:>10.1f}"
        )
    t = out["totals"]
    p = out["pools_pct_of_frame_a_round"]
    lines += [
        "",
        f"  launch   pool {t['launch_us_per_round']:>8.1f} us/round  {p['launch']:.3f}% of round",
        f"  memory   pool {t['memory_us_per_round']:>8.1f} us/round  {p['memory']:.3f}% of round",
        f"  residual pool {t['residual_us_per_round']:>8.1f} us/round  {p['residual']:.3f}% of round",
        f"  addressable   {t['total_us_per_round']:>8.1f} us/round  {p['total']:.3f}% of round",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out = decompose(json.loads(args.report.read_text()))
    print(render(out))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
