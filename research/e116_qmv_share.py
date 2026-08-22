#!/usr/bin/env python3
"""E116 rung 4: the MEASURED wide-QMV share of a decode round.

    usage: research/e116_qmv_share.py TAG [--json OUT] [--min-width 2]

WHAT THIS REPLACES. The campaign has been pricing quantized matrix-vector work
with a standing 0.7786 share. That number reconstructs as 80,113 / 102,864 us
(`senpai/campaign-ledger.md:33544-33547`), a single forced-width census frame,
and E111 reports 0.877 for what reads as the same quantity
(`senpai/campaign-ledger.md:34035-34042`). Two published values that differ by
13 % cannot both be the share, and neither was measured at the widths the
shipped schedule actually chooses. This reducer measures it at the realised
widths of one leg and reports the round-count-weighted result.

FRAMES, NAMED (campaign rule 34).

  `mtp_round`     decode rounds of verify width >= `--min-width`, which is the
                  candidate MTP leg. This is the frame an MTP-side arm has to
                  be priced in.
  `serial_round`  decode rounds of verify width 1, the local wrapper's serial
                  leg. Reported separately and never mixed in.
  `all_decode`    both, round-count weighted. Reported for completeness.

Prefill rounds (`w0`) are excluded from every frame: they are one-time seed
work, not a decode round.

WIDE VERSUS NARROW. A `affine_qmv_fast` dispatch carries its grid in the
census key. The first grid dimension is the row count, so `grid=8x4352x1` is a
width-8 verify and `grid=1x4352x1` is a single row. Wide and narrow QMV are
counted separately because they are different dispatch shapes with different
arithmetic intensity, and an arm that changes one does not automatically
change the other.

THIS LEG MUST BE ISOLATED. `exclusive_kernels` is only a per-kernel exclusive
GPU time when one command buffer holds one dispatch, so the leg must have run
with `MLX_E58_BUFFER_LIMIT_OPS=0`. A leg without it populates `by_width_phase`
only and this reducer refuses to guess.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

GRID = re.compile(r"grid=(\d+)x")


def load(tag: str) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    path = pathlib.Path("research/out") / tag / "census.jsonl"
    if not path.exists():
        sys.exit(f"e116_qmv_share: no census at {path}")
    rounds: collections.Counter[str] = collections.Counter()
    kernels: dict[str, dict[str, float]] = {}
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "gputime":
            continue
        for width in {k.split("|", 1)[0] for k in rec.get("by_width_phase", {})}:
            rounds[width] += rec.get("rounds", 1)
        for key, value in rec.get("exclusive_kernels", {}).items():
            entry = kernels.setdefault(key, {"buffers": 0, "gpu_ns": 0.0})
            entry["buffers"] += value["buffers"]
            entry["gpu_ns"] += value["gpu_ns"]
    if not kernels:
        sys.exit(f"e116_qmv_share: {tag} has no exclusive_kernels; it was not "
                 "run with MLX_E58_BUFFER_LIMIT_OPS=0 and cannot give a "
                 "per-kernel share")
    return dict(rounds), kernels


def classify(kernel: str) -> str:
    """`w8|target_verify|affine_qmv_fast_... grid=8x4352x1 tg=...`."""
    _, _, rest = kernel.partition("|")
    phase, _, name = rest.partition("|")
    if "e116_dose" == phase:
        return "dose"
    if "qmv" not in name:
        return "other"
    match = GRID.search(name)
    rows = int(match.group(1)) if match else 1
    if "gather_qmv" in name:
        return "gather_qmv"
    return "wide_qmv" if rows > 1 else "narrow_qmv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--min-width", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()

    rounds, kernels = load(args.tag)
    per_width: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for kernel, value in kernels.items():
        width_key = kernel.split("|", 1)[0]
        width = int(width_key[1:])
        per_width[width][classify(kernel)] += value["gpu_ns"] / 1e3

    frames = {
        "mtp_round": [w for w in per_width if w >= args.min_width],
        "serial_round": [w for w in per_width if w == 1],
        "all_decode": [w for w in per_width if w >= 1],
    }
    classes = ("wide_qmv", "narrow_qmv", "gather_qmv", "other", "dose")

    out: dict[str, object] = {
        "harness": "local",
        "experiment":
            "e116-measured-transfer-from-kernel-percent-to-leg-seconds",
        "rung": 4,
        "tag": args.tag,
        "isolated_command_buffers": True,
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "rounds_by_width": {k: v for k, v in sorted(rounds.items())},
        "min_mtp_width": args.min_width,
        "per_width_us_per_round": {},
        "frames": {},
    }

    for width in sorted(per_width):
        n = max(rounds.get(f"w{width}", 0), 1)
        out["per_width_us_per_round"][f"w{width}"] = {
            "rounds": rounds.get(f"w{width}", 0),
            **{c: per_width[width][c] / n for c in classes},
            "total": sum(per_width[width][c] for c in classes) / n,
        }

    for frame, widths in frames.items():
        totals = {c: 0.0 for c in classes}
        n_rounds = 0
        for width in widths:
            totals_width = per_width[width]
            for c in classes:
                totals[c] += totals_width[c]
            n_rounds += rounds.get(f"w{width}", 0)
        # The dose is injected work, never part of the model's own round.
        model_total = sum(totals[c] for c in classes if c != "dose")
        out["frames"][frame] = {
            "widths": sorted(widths),
            "rounds": n_rounds,
            "round_busy_us_per_round": model_total / max(n_rounds, 1),
            "wide_qmv_us_per_round": totals["wide_qmv"] / max(n_rounds, 1),
            "narrow_qmv_us_per_round": totals["narrow_qmv"] / max(n_rounds, 1),
            "gather_qmv_us_per_round": totals["gather_qmv"] / max(n_rounds, 1),
            "dose_us_per_round": totals["dose"] / max(n_rounds, 1),
            "wide_qmv_share": totals["wide_qmv"] / model_total
            if model_total else float("nan"),
            "all_qmv_share": (totals["wide_qmv"] + totals["narrow_qmv"]
                              + totals["gather_qmv"]) / model_total
            if model_total else float("nan"),
        }

    print(f"E116 rung 4 -- measured QMV share of the round   harness=local")
    print(f"  leg {args.tag}, isolated command buffers, timing_valid=false")
    print(f"  rounds by width {out['rounds_by_width']}")
    print()
    print(f"{'width':>6} {'rounds':>7} {'wide qmv':>11} {'narrow qmv':>11}"
          f" {'gather':>9} {'other':>11} {'round busy':>12} {'wide share':>11}")
    for width in sorted(per_width):
        row = out["per_width_us_per_round"][f"w{width}"]
        busy = row["total"] - row["dose"]
        print(f"{width:>6} {row['rounds']:>7} {row['wide_qmv']:>11,.1f}"
              f" {row['narrow_qmv']:>11,.1f} {row['gather_qmv']:>9,.1f}"
              f" {row['other']:>11,.1f} {busy:>12,.1f}"
              f" {row['wide_qmv'] / busy if busy else float('nan'):>11.4f}")
    print()
    for frame, value in out["frames"].items():
        print(f"  {frame:<13} widths {value['widths']}"
              f" rounds {value['rounds']:>4}"
              f"  round busy {value['round_busy_us_per_round']:>10,.1f} us"
              f"  wide-QMV {value['wide_qmv_us_per_round']:>10,.1f} us"
              f"  share {value['wide_qmv_share']:.4f}"
              f"  all-QMV share {value['all_qmv_share']:.4f}")

    if args.json:
        path = pathlib.Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
