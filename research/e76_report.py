#!/usr/bin/env python3
"""E76: join the register census, the device parity run and the timed session.

Three independent facts decide each arm and they arrive in three different
files, so this joins them into the one table the assignment asks for:

  registers   research/e76-artifacts/rung1.json, per architecture.
  bit-identity  research/e76-artifacts/parity-na*-b*.json, summed over all seven
              scored shapes, against `plain` in position 0.
  cost        research/e76-artifacts/timed-na*.json, median GPU seconds per
              dispatch per shape, when a timed session has run.

`max_total_threads_per_threadgroup` comes from the parity log rather than the
JSON. It is the only occupancy-adjacent number the Metal API exposes here: the
runtime derives it from the pipeline's register usage, so a step in it is a real
allocation granule on THIS host.

  python3 research/e76_report.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e76_wide_gen import ARMS  # noqa: E402
from e76_rung1_census import TARGET_G17S_REGISTERS  # noqa: E402
from agx_crossarch import LOCAL_ARCH, RANKED_ARCH  # noqa: E402

ARTIFACTS = pathlib.Path("research/e76-artifacts")
ARM_LINE = re.compile(
    r"ARM na=(\d+) arm=(\S+) max_total_threads_per_threadgroup=(\d+) "
    r"static_threadgroup_memory_bytes=(\d+) thread_execution_width=(\d+)")

# Dispatches of each shape in one target-verify round, from the fourth field of
# research/qmv_cost_curve.py SCORED_SHAPES. `full_attn.o_proj` is the same
# (6144, 5120) shape as `linear_attn.out_proj`, so its 16 dispatches are added
# to that row. `head.compact_draft_vocab` is the 2-bit draft readout and takes
# no affine-4 dispatch, so a per-round price must not charge it.
DISPATCHES_PER_ROUND = {
    "linear_attn.in_proj_fused_qkvzba": 48,
    "linear_attn.out_proj": 48 + 16,
    "full_attn.qkv_proj_fused": 16,
    "mlp.gate_up_fused": 64,
    "mlp.down": 64,
    "head.lm_head": 1,
    "head.compact_draft_vocab": 0,
}


def parity() -> dict[tuple[int, str], dict]:
    found: dict[tuple[int, str], dict] = {}
    for path in sorted(ARTIFACTS.glob("parity-na*.json")):
        report = json.loads(path.read_text())
        na = report["na"]
        for shape in report["shapes"]:
            for arm, differing in shape["parity_differing_vs_plain"].items():
                entry = found.setdefault(
                    (na, arm), {"differing": 0, "elements": 0, "shapes": []})
                entry["differing"] += differing
                entry["elements"] += na * shape["n"]
                entry["shapes"].append(shape["shape"])
    return found


def threadgroup_limits(log: pathlib.Path) -> dict[tuple[int, str], int]:
    if not log.exists():
        return {}
    found = {}
    for line in log.read_text().splitlines():
        match = ARM_LINE.search(line)
        if match:
            found[(int(match.group(1)), match.group(2))] = int(match.group(3))
    return found


def timings() -> dict[tuple[int, str], dict[str, float]]:
    found: dict[tuple[int, str], dict[str, float]] = {}
    for path in sorted(ARTIFACTS.glob("timed-na*.json")):
        report = json.loads(path.read_text())
        na = report["na"]
        for shape in report["shapes"]:
            per_arm: dict[str, list[float]] = {}
            for leg in shape["legs"]:
                per_arm.setdefault(leg["arm"], []).append(
                    leg["seconds_per_dispatch"])
            for arm, values in per_arm.items():
                found.setdefault((na, arm), {})[shape["shape"]] = \
                    statistics.median(values)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=pathlib.Path,
                        default=ARTIFACTS / "rung1.json")
    parser.add_argument("--log", type=pathlib.Path,
                        default=ARTIFACTS / "parity.log")
    parser.add_argument("--out", type=pathlib.Path,
                        default=ARTIFACTS / "rung1-table.md")
    args = parser.parse_args()

    census = json.loads(args.census.read_text())
    checked = parity()
    limits = threadgroup_limits(args.log)
    timed = timings()
    rows = []

    lines = [
        "| variant | NA | rows_per_simd | g16s regs / spill | "
        "g17s regs / spill | bit-identical? | machine-code digest changed? | "
        "max threads / threadgroup |",
        "|---|---:|---:|---:|---:|:--:|:--:|---:|",
    ]
    for na in census["widths"]:
        shipped = {
            arch: census["census"][arch][f"shipped_na{na}"]["text_sha8"]
            for arch in census["architectures"]}
        for arm, rps, unroll, _ in ARMS:
            name = f"e76_{arm}_na{na}"
            local = census["census"][LOCAL_ARCH][name]
            ranked = census["census"][RANKED_ARCH][name]
            moved = any(census["census"][arch][name]["text_sha8"] != shipped[arch]
                        for arch in census["architectures"])
            check = checked.get((na, arm))
            if arm == "plain":
                identical = "reference"
            elif check is None:
                identical = "not run"
            else:
                identical = "yes" if check["differing"] == 0 else \
                    f"NO ({check['differing']})"
            limit = limits.get((na, arm))
            rows.append({
                "arm": arm, "na": na, "rows_per_simd": rps,
                "row_block_loop_unrolled": unroll,
                "g16s_registers": local["registers"],
                "g16s_spill_bytes": local["spill_bytes"],
                "g17s_registers": ranked["registers"],
                "g17s_spill_bytes": ranked["spill_bytes"],
                "g16s_text_sha8": local["text_sha8"],
                "g17s_text_sha8": ranked["text_sha8"],
                "machine_code_changed": moved,
                "parity_differing_vs_plain": None if check is None
                    else check["differing"],
                "parity_elements": None if check is None else check["elements"],
                "parity_shapes": None if check is None
                    else len(check["shapes"]),
                "max_total_threads_per_threadgroup": limit,
                "reaches_target": (ranked["registers"] <= TARGET_G17S_REGISTERS
                                   and not ranked["spill_bytes"]),
                "median_seconds_per_dispatch": timed.get((na, arm)),
            })
            lines.append(
                f"| `{arm}` | {na} | {rps} | "
                f"{local['registers']} / {local['spill_bytes']} | "
                f"{ranked['registers']} / {ranked['spill_bytes']} | "
                f"{identical} | {'yes' if moved else 'no'} | "
                f"{limit if limit is not None else '-'} |")

    covered = {(na, arm) for (na, arm) in checked}
    body = "\n".join(lines)

    # One target-verify round at this width, so the cost side is priced in the
    # unit the score is made of rather than per shape.
    cost = []
    for na in census["widths"]:
        base = timed.get((na, "plain"))
        if not base:
            continue
        base_round = sum(DISPATCHES_PER_ROUND[shape] * seconds
                         for shape, seconds in base.items())
        for arm, _, _, _ in ARMS:
            per_shape = timed.get((na, arm))
            if not per_shape:
                continue
            per_round = sum(DISPATCHES_PER_ROUND[shape] * seconds
                            for shape, seconds in per_shape.items())
            cost.append({"na": na, "arm": arm,
                         "seconds_per_round": per_round,
                         "delta_vs_plain_pct":
                             100.0 * (per_round / base_round - 1.0)})
    if cost:
        body += ("\n\n| variant | NA | seconds per verify round | "
                 "vs `plain` |\n|---|---:|---:|---:|\n")
        body += "\n".join(
            f"| `{row['arm']}` | {row['na']} | {row['seconds_per_round']:.6f} | "
            f"{row['delta_vs_plain_pct']:+.2f} % |" for row in cost)
    for row in rows:
        for entry in cost:
            if entry["na"] == row["na"] and entry["arm"] == row["arm"]:
                row["seconds_per_round"] = entry["seconds_per_round"]
                row["delta_vs_plain_pct"] = entry["delta_vs_plain_pct"]

    print(body)
    print()
    print(f"parity: {len(covered)} arm-width pairs checked over "
          f"{len(set(s for c in checked.values() for s in c['shapes']))} shapes; "
          f"{sum(1 for c in checked.values() if c['differing'])} pairs differ")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body + "\n")
    (args.out.with_suffix(".json")).write_text(
        json.dumps(rows, indent=2, sort_keys=True))
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
