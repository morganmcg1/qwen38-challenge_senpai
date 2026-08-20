#!/usr/bin/env python3
"""Summarise a research/run-qmv-parity.sh sweep into one auditable artifact.

    research/e61_parity_report.py [--in DIR] [--out PATH]

The comparator that run-qmv-parity.sh calls always exits 0 and only prints a
verdict, so a green job is not evidence. This recomputes the per-cell digest
comparison from the raw arm JSON and records the two facts that decide whether
the sweep has any power at all:

  covering_cells_by_bits  how many cells reach a crossrow body per bit width.
                          A bit width with zero covering cells cannot observe a
                          change to the crossrow table, however many cells it
                          contributes.
  negative_control        the perturbed arm must diverge. If it does not, the
                          BIT-IDENTICAL verdicts carry no information.
"""

from __future__ import annotations

import argparse
import json
import pathlib

REF = "ref"
PERTURB = "t6_lane_perturb"


def digests(d: pathlib.Path, name: str) -> dict[tuple, str]:
    payload = json.loads((d / f"{name}.json").read_text())
    return {(e["bits"], e["shape"], e["m"]): e["digest"] for e in payload["entries"]}


def twins(d: pathlib.Path, name: str) -> dict[str, str]:
    text = (d / f"{name}.twins.txt").read_text().strip().splitlines()
    return {line.split()[1]: line.split()[0] for line in text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=".mlxfast-private/qmv-parity")
    ap.add_argument("--out", default="research/e61-artifacts/e61-qmv-parity.json")
    ap.add_argument("--base-sha", default="d2139c924c7a7d98ca6026eea63867c2776abbca")
    ap.add_argument("--head-sha", default="")
    args = ap.parse_args()

    d = pathlib.Path(args.indir)
    ref_payload = json.loads((d / f"{REF}.json").read_text())
    ref = digests(d, REF)
    names = sorted(p.stem for p in d.glob("*.json"))

    arms: dict[str, dict] = {}
    for name in names:
        other = digests(d, name)
        differing = [k for k in ref if ref[k] != other[k]]
        loci = sorted({(k[0], k[2]) for k in differing})
        arms[name] = {
            "twin_sha256": twins(d, name),
            "cells_differing_vs_ref": len(differing),
            "differing_bits_m": [list(x) for x in loci],
            "shapes_differing": sorted({k[1] for k in differing}),
            "verdict": "BIT-IDENTICAL" if not differing else "DIVERGES",
        }

    covering = ref_payload["covering_cells_by_bits"]
    distinct_twins = len({tuple(sorted(a["twin_sha256"].items())) for a in arms.values()})
    perturb = arms.get(PERTURB, {})
    payload = {
        "experiment": "e61",
        "control": "qmv-bitwise-parity-digest",
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "host": "Apple M4 Pro 48GiB",
        "cells_total_per_arm": len(ref),
        "cells_by_bits": ref_payload["cells_by_bits"],
        "covering_cells_by_bits": covering,
        "crossrow_guard": (
            "!batched && group_size==64 && bits==4 && out_vec_size>=1024; the "
            "multi-row ntg.x switch is nested inside out_vec_size>=4096"
        ),
        "bits3_has_no_power": covering.get("3", 0) == 0,
        "bits3_note": (
            "every bits=3 cell routes qmv_fast_impl, so the 96 bits=3 cells are "
            "a leakage control only and have NO power over the crossrow change"
        ),
        "arms_have_distinct_twins": distinct_twins == len(arms),
        "negative_control": {
            "arm": PERTURB,
            "fired": perturb.get("verdict") == "DIVERGES",
            "cells_differing": perturb.get("cells_differing_vs_ref"),
            "loci": perturb.get("differing_bits_m"),
            "proves": [
                "the digest observes kernel output bits in the covering region",
                "the t6 <T,6,6> route at M=6 is live on every scored shape",
                "sensitivity is localised; no spurious divergence elsewhere",
            ],
            "does_not_prove": (
                "power at any other M: the perturbation is gated on NA==6, so "
                "M=6 is the only locus it can reach"
            ),
        },
        "limitations": [
            "synthetic activations and synthetic quantized weights, one call per cell",
            "kernel-level bitwise check only; it is NOT the end-to-end 512-token "
            "exactness check, which is rung 2 PATH C",
        ],
        "arms": arms,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for name in names:
        a = arms[name]
        print(f"{name:20s} {a['verdict']:14s} differing={a['cells_differing_vs_ref']:3d} "
              f"{a['differing_bits_m']}")
    print(f"covering_cells_by_bits={covering}  bits3_has_no_power="
          f"{payload['bits3_has_no_power']}  distinct_twins={payload['arms_have_distinct_twins']}")
    print(f"negative control fired: {payload['negative_control']['fired']}")
    print(f"wrote {out}")
    return 0 if payload["negative_control"]["fired"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
