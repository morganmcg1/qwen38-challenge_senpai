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

# The advisor's occupancy model, carried here unchanged and NOT verified by this
# experiment: `floor(register_file / (32 lanes * 4 bytes * registers))` resident
# simdgroups per core on the ranked architecture. research/e76_occupancy_probe.m
# could not confirm it, because `maxTotalThreadsPerThreadgroup` is 1024 for every
# kernel from 14 to 126 registers on this host. Every number derived from this
# model is labelled as modelled, not measured.
ADVISOR_REGISTER_FILE_BYTES = 208 * 1024
BYTES_PER_REGISTER_PER_SIMDGROUP = 32 * 4
# The crown's ranked table never puts more than three proposals in one group, so
# its scored cells compile at the NA=3 register count.
CROWN_LARGEST_GROUP = 3


def resident_simdgroups(registers: int) -> int:
    return ADVISOR_REGISTER_FILE_BYTES // (
        BYTES_PER_REGISTER_PER_SIMDGROUP * registers)


def block_bytes_per_lane(partition: tuple[int, ...], rows_per_simd: int) -> int:
    """Bytes one lane reads per k-block to cover the whole four-row block.

    Read straight off the shipped body with T = bfloat16. Per call the lane
    reads `4 i-steps * NA m * 4 elements * 2 B` of x, `rows_per_simd * 4 * 2 B`
    of packed weights and `rows_per_simd * 2 * 2 B` of scale and bias. The block
    needs `4 / rows_per_simd` calls, so the row-side terms are invariant at 32
    and 16 bytes per group and only the x term moves with the row block.
    Splitting the proposal width into groups is the mirror image: x is invariant
    and the row side repeats per group. This is the model-free part of the
    recommendation, because it needs no register-file constant.
    """
    calls = 4 // rows_per_simd
    return sum(calls * 32 * na + 32 + 16 for na in partition)


def greedy_partition(width: int, largest: int) -> tuple[int, ...]:
    groups = []
    while width > 0:
        groups.append(min(largest, width))
        width -= groups[-1]
    return tuple(groups)


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

    # Rung 3. Every timed arm is priced against the occupancy it would buy on the
    # ranked architecture, so the recommendation is a division and not a
    # judgement call. `break_even_conversion` is the fraction of the modelled
    # occupancy gain that would have to become throughput for the arm to pay for
    # its own measured cost. Above 100 % the arm cannot pay even if every extra
    # resident simdgroup were free throughput.
    crown = census["census"][RANKED_ARCH].get(
        f"e76_plain_na{CROWN_LARGEST_GROUP}")
    advice = []
    for row in rows:
        if "delta_vs_plain_pct" not in row or not row["reaches_target"]:
            continue
        base = next(r for r in rows
                    if r["na"] == row["na"] and r["arm"] == "plain")
        r0 = resident_simdgroups(base["g17s_registers"])
        r1 = resident_simdgroups(row["g17s_registers"])
        gain = 100.0 * (r1 / r0 - 1.0)
        c = row["delta_vs_plain_pct"]
        need = None if gain <= 0 else 100.0 * c / gain
        one_group = block_bytes_per_lane((row["na"],), 4)
        arm_bytes = block_bytes_per_lane((row["na"],), row["rows_per_simd"])
        saved = base["g17s_registers"] - row["g17s_registers"]
        row.update({
            "modelled_resident_simdgroups": r1,
            "modelled_resident_simdgroups_plain": r0,
            "modelled_occupancy_gain_pct": gain,
            "break_even_conversion_pct": need,
            "pays_for_itself_possible": bool(need is not None and need <= 100.0),
            "block_bytes_per_lane": arm_bytes,
            "extra_block_bytes_per_lane": arm_bytes - one_group,
            "g17s_registers_saved": saved,
            "extra_bytes_per_register_saved":
                None if saved <= 0 else (arm_bytes - one_group) / saved,
        })
        advice.append(row)
    if advice:
        body += (
            "\n\n### Rung 3: does the qualifying arm pay for itself?\n\n"
            f"Modelled columns use the advisor's unverified "
            f"`floor({ADVISOR_REGISTER_FILE_BYTES // 1024} KiB / "
            f"({BYTES_PER_REGISTER_PER_SIMDGROUP} B * regs))` occupancy model. "
            "The cost column is measured on this host behind the real 40 C "
            "gate.\n\n"
            "| variant | NA | g17s regs | modelled resident simdgroups "
            "(`plain` -> arm) | modelled occupancy gain | measured cost per "
            "verify round | conversion needed to break even | can it pay? |\n"
            "|---|---:|---:|:--:|---:|---:|---:|:--:|\n")
        body += "\n".join(
            f"| `{row['arm']}` | {row['na']} | {row['g17s_registers']} | "
            f"{row['modelled_resident_simdgroups_plain']} -> "
            f"{row['modelled_resident_simdgroups']} | "
            f"{row['modelled_occupancy_gain_pct']:+.1f} % | "
            f"{row['delta_vs_plain_pct']:+.2f} % | "
            + (f"{row['break_even_conversion_pct']:.0f} %"
               if row["break_even_conversion_pct"] is not None else "no gain")
            + f" | {'possible' if row['pays_for_itself_possible'] else 'NO'} |"
            for row in advice)
        # The model-free comparison. Both routes buy registers with traffic, so
        # price each one in extra bytes per register saved and the recommendation
        # needs no register-file constant to rank them.
        if crown:
            body += (
                "\n\n### Rung 3, model-free: what each route pays per register\n"
                "\n| M | route | partition | rows_per_simd | g17s regs | "
                "bytes per lane per k-block | extra vs one-group shipped | "
                "extra bytes per register saved |\n"
                "|---:|---|---|---:|---:|---:|---:|---:|\n")
            lines3 = []
            for na in census["widths"]:
                shipped_bytes = block_bytes_per_lane((na,), 4)
                shipped_regs = census["census"][RANKED_ARCH][
                    f"e76_plain_na{na}"]["registers"]
                entries = [("shipped one group", (na,), 4, shipped_regs)]
                part = greedy_partition(na, CROWN_LARGEST_GROUP)
                if len(part) > 1:
                    entries.append(("crown partition", part, 4,
                                    crown["registers"]))
                for row in advice:
                    if row["na"] == na:
                        entries.append((f"one group, `{row['arm']}`", (na,),
                                        row["rows_per_simd"],
                                        row["g17s_registers"]))
                for label, partition, rps, regs in entries:
                    total = block_bytes_per_lane(partition, rps)
                    extra = total - shipped_bytes
                    saved = shipped_regs - regs
                    per = f"{extra / saved:.0f}" if saved > 0 else "-"
                    lines3.append(
                        f"| {na} | {label} | "
                        f"[{','.join(str(g) for g in partition)}] | {rps} | "
                        f"{regs} | {total} | {extra:+d} | {per} |")
            body += "\n".join(lines3)

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
