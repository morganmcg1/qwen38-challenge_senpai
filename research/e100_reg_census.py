#!/usr/bin/env python3
"""E100: register cost of raising the wide cross-row QMV `NA` cap.

`affine_qmv_fast` is ONE Metal kernel. Every width is a `case` of one
`switch (ntg.x)` inside it, so the backend allocates one register count for the
whole program and every width pays the widest branch. That is what makes the
`NA` cap a shared decision rather than a per-width one, and it is the number
this census reports.

E76 censused the `_wide` BODY on its own and read 83/90/91/98/111 g17s
registers at NA = 2..6. A body census cannot answer this question, because the
scored object is the dispatcher, not the body.

Three arms, all compiled from the runtime-effective JIT string that
`get_quantized_kernel` builds (four `mlx-generated/*.cpp` preambles concatenated
with no include path):

  base      the recorded experiment base, cap NA <= 4.
  cap5      the working tree: M = 5 -> [5] and M = 9 -> [5, 4].
  cap8      research-only text patch on top of the working tree that also
            collapses M = 6 -> [6], M = 7 -> [7] and M = 8 -> [8]. It prices
            the follow-up without shipping it, and it is never built into a
            candidate here.

  python3 research/e100_reg_census.py --base <sha> --out research/out/e100-reg.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from jit_string_compile import assemble  # noqa: E402
from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH,
    RANKED_ARCH,
    build_metallib,
    translate,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The scored dispatcher cells. `affine_qmv_fast<T, group_size, bits, batched>`
# is the entry point the worker launches for every 4-bit g64 QMV in the target.
CELLS = (
    "affine_qmv_fast<bfloat16_t, 64, 4, false>",
    "affine_qmv_fast<bfloat16_t, 64, 4, true>",
)

CAP8_PATCH = (
    ("NA >= 2 && NA <= 5", "NA >= 2 && NA <= 8"),
    ("wide multi-row QMV supports NA in [2, 5]",
     "wide multi-row QMV supports NA in [2, 8]"),
    ("qmv_fast_crossrow_affine4_g64_m<T, 6, 3, true>",
     "qmv_fast_crossrow_affine4_g64_m<T, 6, 6, true>"),
    ("qmv_fast_crossrow_affine4_g64_m<T, 7, 4, true>",
     "qmv_fast_crossrow_affine4_g64_m<T, 7, 7, true>"),
    ("qmv_fast_crossrow_affine4_g64_m<T, 8, 4, true>",
     "qmv_fast_crossrow_affine4_g64_m<T, 8, 8, true>"),
)


def patch(source: str, rules: tuple[tuple[str, str], ...]) -> str:
    for old, new in rules:
        if old not in source:
            raise SystemExit(f"census patch target missing: {old!r}")
        source = source.replace(old, new)
    return source


def partitions(source: str) -> dict[str, str]:
    """Read back the dispatch table the arm actually compiles."""
    import re
    found = {}
    for m, ipg in re.findall(
        r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>", source
    ):
        m_i, ipg_i = int(m), int(ipg)
        groups, left = [], m_i
        while left > 0:
            take = min(ipg_i, left)
            groups.append(take)
            left -= take
        found[m] = "+".join(str(g) for g in groups)
    for m in re.findall(r"qmv_fast_crossrow_affine4_g64<T, (\d+)>", source):
        found.setdefault(m, m)
    return dict(sorted(found.items(), key=lambda kv: int(kv[0])))


def census(name: str, source: str, workdir: pathlib.Path) -> dict:
    arm_dir = workdir / name
    try:
        lib = build_metallib(source, arm_dir)
    except subprocess.CalledProcessError as error:
        return {"arm": name, "compiled": False,
                "error": error.stderr.decode()[-2000:]}
    row = {"arm": name, "compiled": True, "partitions": partitions(source),
           "source_bytes": len(source)}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for kernel, record in translate(lib, arch, arm_dir).items():
            row.setdefault(arch, {})[kernel] = {
                "registers": record.get("registers"),
                "spill_bytes": record.get("spill_bytes", 0),
            }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="git sha of the experiment base arm")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    tree = assemble(CELLS, None)
    arms = {
        "base": assemble(CELLS, args.base),
        "cap5": tree,
        "cap8": patch(tree, CAP8_PATCH),
    }

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for name, source in arms.items():
            rows.append(census(name, source, workdir))

    for row in rows:
        if not row["compiled"]:
            print(f"{row['arm']:>5}  DID NOT COMPILE")
            print(row["error"])
            continue
        table = " ".join(f"M{m}=[{p}]" for m, p in row["partitions"].items())
        print(f"{row['arm']:>5}  {table}")
        for arch in (LOCAL_ARCH, RANKED_ARCH):
            for kernel, value in row[arch].items():
                spill = value["spill_bytes"] or 0
                mark = f" spill={spill}B" if spill else ""
                print(f"        {arch:>14} {kernel} "
                      f"registers={value['registers']}{mark}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"base_sha": args.base, "arms": rows}, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
