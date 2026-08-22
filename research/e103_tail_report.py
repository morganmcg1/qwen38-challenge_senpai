#!/usr/bin/env python3
"""E103 rung 0 report: cost, resource and occupancy table for the SDPA tail arms.

Joins three zero-GPU sources:

  * the cross-architecture register census (`research/agx_crossarch.py census`)
    on g16s and g17s, which campaign rule 56 requires for any arm that touches
    the register channel;
  * the tail-region AIR instruction accounting
    (`research/e103_tail_air.py`), because Askeladd's `n_nosums` result says
    total instruction issue is the binding resource here; and
  * the threadgroup allocations declared in the AIR, which set occupancy
    against the 32,768-byte per-core pool that E110 validated against
    `maxThreadgroupMemoryLength`.

  python3 research/e103_tail_report.py --air ARMS.ll --census-nfm FILE \\
      --census-fm FILE --tail-air tail_air.json [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

TG_POOL_BYTES = 32768
THREADS_PER_THREADGROUP = 1024
SIMD_SIZE = 32
# E77: register file per core, in bytes, divided by the 128 bytes a simdgroup
# needs per register. Local is the g16s host, ranked is the M5.
SIMDGROUPS_LOCAL = 384 * 1024 // 128
SIMDGROUPS_RANKED = 496 * 1024 // 128
OCCUPANCY_GAMMA = 0.01346

TG_GLOBAL = re.compile(
    r"^@([\w.$]+)\s*=.*addrspace\(3\) global \[(\d+) x (\w+)\]"
)
WIDTH = {"float": 4, "half": 2, "bfloat": 2, "int": 4, "uint": 4}


def threadgroup_bytes(air: pathlib.Path) -> dict[str, int]:
    """Bytes of threadgroup memory each entry point reaches.

    The AIR names a threadgroup global after the mangled template instance that
    declares it, while the entry point carries the host name from
    `instantiate_kernel`. Two host names can therefore share one allocation, and
    a name match would miss that. Summing the distinct globals each function
    body actually references is exact and needs no demangling.
    """
    text = air.read_text()
    size = {
        m.group(1): int(m.group(2)) * WIDTH[m.group(3)]
        for line in text.splitlines()
        if (m := TG_GLOBAL.match(line))
    }

    total: dict[str, int] = {}
    name = None
    seen: set[str] = set()
    for line in text.splitlines():
        define = re.match(r"^define weak_odr void @([\w.]+)\(", line)
        if define:
            name, seen = define.group(1), set()
            total[name] = 0
        elif line == "}":
            name = None
        elif name is not None:
            for reference in re.findall(r"@([\w.$]+)", line):
                if reference in size and reference not in seen:
                    seen.add(reference)
                    total[name] += size[reference]
    return total


def census(path: pathlib.Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        arch, rest = line.split(" ", 1)
        out[arch] = json.loads(rest)
    return out


def occupancy(registers: int, tg_bytes: int) -> dict:
    simdgroups_per_tg = THREADS_PER_THREADGROUP // SIMD_SIZE
    # A kernel with no threadgroup allocation is never memory bound.
    by_memory = TG_POOL_BYTES // tg_bytes if tg_bytes else 1 << 20
    result = {"threadgroup_bytes": tg_bytes, "threadgroups_by_memory": by_memory}
    for tag, budget in (("local", SIMDGROUPS_LOCAL), ("ranked", SIMDGROUPS_RANKED)):
        by_register = budget // registers
        groups = min(by_register // simdgroups_per_tg, by_memory)
        resident = groups * simdgroups_per_tg
        result[f"{tag}_simdgroups_by_register"] = by_register
        result[f"{tag}_threadgroups"] = groups
        result[f"{tag}_resident_simdgroups"] = resident
        result[f"{tag}_binding"] = (
            "memory"
            if by_memory <= by_register // simdgroups_per_tg
            else "register"
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--air", type=pathlib.Path, required=True)
    ap.add_argument("--census-nfm", type=pathlib.Path, required=True)
    ap.add_argument("--census-fm", type=pathlib.Path)
    ap.add_argument("--tail-air", type=pathlib.Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    tg = threadgroup_bytes(args.air)
    nfm = census(args.census_nfm)
    fm = census(args.census_fm) if args.census_fm else None
    tail = json.loads(args.tail_air.read_text())

    rows = {}
    for arm in args.arms:
        bytes_used = tg[arm]
        g16 = nfm["applegpu_g16s"][arm]
        g17 = nfm["applegpu_g17s"][arm]
        row = {
            "g16s": g16,
            "g17s": g17,
            "tail_air": tail[arm]["dynamic_tail_only"],
            "occupancy_g17s": occupancy(g17["registers"], bytes_used),
            "occupancy_g16s": occupancy(g16["registers"], bytes_used),
        }
        if fm:
            row["g16s_fastmath"] = fm["applegpu_g16s"][arm]
            row["g17s_fastmath"] = fm["applegpu_g17s"][arm]
        rows[arm] = row

    base = rows[args.arms[0]]
    print(
        f"{'arm':16s} {'tgB':>6s} {'R16':>4s} {'R17':>4s} {'sp':>4s} "
        f"{'txt16':>6s} {'txt17':>6s} {'instr':>6s} {'barr':>5s} "
        f"{'dTG':>4s} {'occL':>5s} {'occR':>5s} {'bind':>7s}"
    )
    for arm, row in rows.items():
        occ_local = row["occupancy_g16s"]
        occ_ranked = row["occupancy_g17s"]
        print(
            f"{arm:16s} {occ_ranked['threadgroup_bytes']:6d} "
            f"{row['g16s']['registers']:4d} {row['g17s']['registers']:4d} "
            f"{row['g17s']['spill_bytes']:4d} "
            f"{row['g16s']['text_bytes']:6d} {row['g17s']['text_bytes']:6d} "
            f"{row['tail_air']['instructions']:6d} "
            f"{row['tail_air']['barrier']:5d} "
            f"{row['tail_air']['tg_store'] + row['tail_air']['tg_load']:4d} "
            f"{occ_local['local_resident_simdgroups']:5d} "
            f"{occ_ranked['ranked_resident_simdgroups']:5d} "
            f"{occ_ranked['ranked_binding']:>7s}"
        )

    print()
    print("deltas against", args.arms[0])
    for arm, row in list(rows.items())[1:]:
        d_instr = (
            row["tail_air"]["instructions"] - base["tail_air"]["instructions"]
        )
        d_barrier = row["tail_air"]["barrier"] - base["tail_air"]["barrier"]
        ratio = (
            base["occupancy_g17s"]["ranked_resident_simdgroups"]
            / row["occupancy_g17s"]["ranked_resident_simdgroups"]
            if row["occupancy_g17s"]["ranked_resident_simdgroups"]
            else float("inf")
        )
        penalty = ratio**OCCUPANCY_GAMMA - 1.0
        print(
            f"  {arm:16s} d_tail_instr={d_instr:+5d} d_barrier={d_barrier:+3d} "
            f"occupancy_ratio={ratio:.3f} omega_penalty={penalty * 100:+.3f}%"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
