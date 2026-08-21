#!/usr/bin/env python3
"""E104 rung 0.5: register and spill cost of every verify width, both arches.

The advisor's break-even law needs the isolated one-group rate at each verify
width M, and E77's occupancy law says any collapse gain is charged a register
tax first. This census prices that tax before any GPU time is spent.

`research/e100_reg_census.py` censused the shipped dispatcher, which allocates
ONE register count for every width branch, so it cannot separate widths. The
ladder sources emit `e104_iso_na<M>` as its own entry point per width, so the
backend allocates each cell on its own and the count is attributable.

Two arms, both compiled from the runtime-effective JIT string:

  x_onegroup  every M routed to IPG == M, one weight stream.
  y_split     the two-group partition a collapse at that width replaces.

Research only: NA = 6, 7 and 8 exist here as private template instantiations.
Nothing in this script touches the candidate surface.

  python3 research/e104_variant_sources.py --outdir DIR --set ladder
  python3 research/e104_ladder_census.py --dir DIR --out research/out/....json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH,
    RANKED_ARCH,
    build_metallib,
    translate,
)

KERNEL_RE = re.compile(r"e104_iso_na(\d+)$")

# E76 read these g17s counts for the `_wide` BODY at NA = 4, 5 and 6, against a
# hard ceiling of 126. They are printed next to the new numbers so a reader can
# see at once whether this census reproduces the earlier ladder or contradicts
# it; a body census and an entry-point census need not agree.
E76_G17S_BODY = {4: 91, 5: 98, 6: 111}
E76_CEILING = 126


def census(name: str, source: pathlib.Path, workdir: pathlib.Path) -> dict:
    arm_dir = workdir / name
    try:
        lib = build_metallib(source.read_text(), arm_dir)
    except subprocess.CalledProcessError as error:
        return {"arm": name, "compiled": False,
                "error": error.stderr.decode()[-2000:]}
    row: dict = {"arm": name, "compiled": True, "source_bytes": source.stat().st_size}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for kernel, record in translate(lib, arch, arm_dir).items():
            found = KERNEL_RE.search(kernel)
            if found is None:
                continue
            row.setdefault(arch, {})[found.group(1)] = {
                "registers": record.get("registers"),
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "text_sha8": record.get("text_sha8"),
            }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=pathlib.Path,
                        help="directory written by e104_variant_sources --set ladder")
    parser.add_argument("--arms", default="x_onegroup,y_split")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    partitions = json.loads((args.dir / "partitions.json").read_text())
    arms = args.arms.split(",")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for name in arms:
            rows.append(census(name, args.dir / f"arm_{name}.metal", workdir))

    widths = sorted({int(k) for row in rows if row["compiled"]
                     for k in row.get(LOCAL_ARCH, {})})
    for row in rows:
        if not row["compiled"]:
            print(f"{row['arm']}  DID NOT COMPILE")
            print(row["error"])
            continue
        print(f"\n{row['arm']}")
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            cells = []
            for na in widths:
                value = row[arch][str(na)]
                spill = value["spill_bytes"] or 0
                mark = f"/s{spill}" if spill else ""
                part = partitions[row["arm"]][str(na)]["partition"]
                cells.append(f"M{na}[{part}]={value['registers']}{mark}")
            print(f"  {arch:>14}  " + "  ".join(cells))

    have = {row["arm"]: row for row in rows if row["compiled"]}
    # The tax only means anything for the two-arm partition ladder. Other arm
    # sets reuse this census for the per-width counts alone.
    if len(arms) == 2 and len(have) == 2:
        one, two = have[arms[0]], have[arms[1]]
        print("\nregister tax of collapsing to one group (ranked arch "
              f"{RANKED_ARCH})")
        for na in widths:
            a = one[RANKED_ARCH][str(na)]
            b = two[RANKED_ARCH][str(na)]
            same = " same-code" if a["text_sha8"] == b["text_sha8"] else ""
            note = ""
            if na in E76_G17S_BODY:
                note = f"  (E76 body {E76_G17S_BODY[na]}/{E76_CEILING})"
            print(f"  M{na}: one-group {a['registers']:>4}  "
                  f"split {b['registers']:>4}  "
                  f"delta {a['registers'] - b['registers']:+d}"
                  f"{same}{note}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"partitions": partitions, "arms": rows,
             "e76_g17s_body": E76_G17S_BODY, "e76_ceiling": E76_CEILING},
            indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
