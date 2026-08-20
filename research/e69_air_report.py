#!/usr/bin/env python3
"""Print the E69 rung-0 AIR census as the tables the PR comment quotes.

  python3 research/e69_air_report.py [--in research/e69-artifacts/rung0-air.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib

ARMS = ["plain", "wvec", "xvec", "wxvec", "tgx", "rows8", "rows8wxvec"]


def table(cells: dict, title: str, pick) -> None:
    print(f"\n{title}")
    print("| NA | " + " | ".join(ARMS) + " |")
    print("|" + "---|" * (len(ARMS) + 1))
    for na in sorted(cells, key=int):
        row = [pick(cells[na][arm]) for arm in ARMS]
        print(f"| {na} | " + " | ".join(row) + " |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", type=pathlib.Path,
                    default=pathlib.Path("research/e69-artifacts/rung0-air.json"))
    args = ap.parse_args()
    data = json.loads(args.path.read_text())
    cells = data["cells"]

    print(f"head={data['head']} dirty_paths={data['dirty_paths']}")
    print(f"flags={data['flags']}")
    print(f"probe={data['probe']}")
    total = 0
    failed = []
    for na, per_na in data["checks"].items():
        for name, passed in per_na.items():
            total += 1
            if passed is not True:
                failed.append(f"na={na} {name}")
    print(f"checks total={total} failed={len(failed)} {failed}")

    table(cells, "peak_live_cfg_loop (registers live across the k-loop)",
          lambda c: str(c["peak_live_cfg_loop"]))
    table(cells, "device loads per output row per lane per k-block",
          lambda c: f"{c['per_lane_per_k_block']['device_loads_per_output_row']:.2f}")
    table(cells, "device x loads per lane per k-block",
          lambda c: str(c["per_lane_per_k_block"]["device_x_loads"]))
    table(cells, "weight loads per lane per k-block",
          lambda c: str(c["per_lane_per_k_block"]["weight_loads"]))
    table(cells, "scale/bias loads per lane per k-block",
          lambda c: str(c["per_lane_per_k_block"]["scale_bias_loads"]))
    table(cells, "staged (threadgroup) x loads per lane per k-block",
          lambda c: str(c["per_lane_per_k_block"]["staged_x_loads"]))
    table(cells, "loop fadd per output row",
          lambda c: f"{c['loop_fp_per_output_row']['fadd']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
