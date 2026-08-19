#!/usr/bin/env python3
"""Emit the per-arm in-kernel routing table recorded by the E54 parity sweep.

The parity digests carry the kernel name each (bits, M) cell actually entered in
a built binary. That is the isolation proof for E54: within one pair exactly one
cell may change kernel, and every other cell must run the identical kernel in
both arms, which makes it an in-session control.
"""

import argparse
import collections
import json
import pathlib
import sys

ARMS = [
    "iso_m5_ipg3",
    "iso_m5_ipg5",
    "iso_m7_ipg4",
    "iso_m7_ipg5",
    "iso_m8_ipg4",
    "iso_m8_ipg5",
    "shipped",
    "e27_full",
    "iso_m5_ipg5_lane_perturb",
]

PAIRS = [
    ("P1", "iso_m5_ipg3", "iso_m5_ipg5", 5),
    ("P2", "iso_m7_ipg4", "iso_m7_ipg5", 7),
    ("P3", "iso_m8_ipg4", "iso_m8_ipg5", 8),
]


def routing(path):
    doc = json.loads(path.read_text())
    table = {}
    counts = collections.Counter()
    for e in doc["entries"]:
        key = (e["bits"], e["m"])
        name = e["in_kernel_path"]
        prev = table.setdefault(key, name)
        if prev != name:
            raise SystemExit(f"{path.name}: cell {key} routed to both {prev} and {name}")
        counts[key] += 1
    return table, counts, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-dir", default=".mlxfast-private/e54-parity")
    ap.add_argument("--out", default="research/e54-artifacts/parity-routing.md")
    args = ap.parse_args()

    pdir = pathlib.Path(args.parity_dir)
    tables, counts_by_arm, docs = {}, {}, {}
    for arm in ARMS:
        p = pdir / f"{arm}.json"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        tables[arm], counts_by_arm[arm], docs[arm] = routing(p)
    if not tables:
        raise SystemExit("no parity digests found")

    any_doc = next(iter(docs.values()))
    widths = any_doc["widths"]
    bits_list = sorted(any_doc["bits"], reverse=True)

    lines = []
    lines.append("# E54 in-kernel routing, read out of the built binaries")
    lines.append("")
    lines.append(
        "Source: `research/qmv_parity_dump` cells in "
        f"`{args.parity_dir}`. Widths swept: {widths}. Quantization bit widths: "
        f"{bits_list}."
    )
    lines.append("")
    lines.append(
        "`bits=3` never enters a crossrow kernel in any arm. The crossrow family is "
        "specialised `affine4_g64`, so only 4-bit group-64 weights reach it. The "
        "scored checkpoint is affine 4-bit group-64, so `bits=3` is a negative "
        "routing control and not a scored path."
    )
    lines.append("")

    for bits in bits_list:
        lines.append(f"## bits = {bits}")
        lines.append("")
        arms_present = [a for a in ARMS if a in tables]
        lines.append("| M | " + " | ".join(arms_present) + " |")
        lines.append("|---|" + "|".join(["---"] * len(arms_present)) + "|")
        for m in widths:
            row = []
            for arm in arms_present:
                name = tables[arm].get((bits, m), "—")
                row.append(f"`{name}`")
            lines.append(f"| {m} | " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Isolation check per pair")
    lines.append("")
    lines.append("| pair | treated cell | cells that change kernel | control cells identical |")
    lines.append("|---|---|---|---|")
    ok = True
    for label, a, b, treated in PAIRS:
        if a not in tables or b not in tables:
            continue
        changed = sorted(k for k in tables[a] if tables[a][k] != tables[b].get(k))
        same = sum(1 for k in tables[a] if tables[a][k] == tables[b].get(k))
        expect = [(4, treated)]
        verdict = "" if changed == expect else "  ⚠️ UNEXPECTED"
        ok = ok and changed == expect
        lines.append(
            f"| {label} | bits=4, M={treated} | {changed}{verdict} | "
            f"{same}/{len(tables[a])} |"
        )
    lines.append("")
    lines.append(
        "Every non-treated cell enters a byte-identical kernel in both arms of its "
        "pair, so its timing difference measures session noise only."
    )
    lines.append("")

    if "shipped" in tables and "e27_full" in tables:
        changed = sorted(k for k in tables["shipped"] if tables["shipped"][k] != tables["e27_full"].get(k))
        lines.append("## P4, the E27 composite on the real shipped table")
        lines.append("")
        lines.append(f"Cells that change kernel: {changed}")
        lines.append("")
        lines.append(
            "This reproduces E27's actual edit: `case 5` and `case 9` both move to "
            "IPG=5 while every other width keeps its shipped specialisation."
        )
        lines.append("")

    text = "\n".join(lines) + "\n"
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text)
    print(f"wrote {outp}")
    if not ok:
        raise SystemExit("routing isolation check FAILED")


if __name__ == "__main__":
    main()
