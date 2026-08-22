#!/usr/bin/env python3
"""Run the E131 rung 3 acceptance suite for `senpai/entry-point-cliff-census.sh`.

    usage: python3 research/e131_rung3_receipt.py --outdir research/e131-artifacts

The suite runs four censuses and composes one receipt:

  coverage      HEAD vs the working tree, to list every gated entry point.
  e121_fails    da025231 -> 5d97175c must exit 1; E121 raised the wide-QMV
                entry point from 101 to 102 registers on applegpu_g17s and
                lost one derived resident simdgroup.
  e126_passes   5d97175c -> 04171655 must exit 0; the revert gives the
                simdgroup back.
  revert_clean  da025231 -> 04171655 must exit 0 with no register delta at
                all, which proves the revert restored the pre-E121 text.

Compile-only. No GPU, no model, no timing. Every simdgroup number here is
`derived` under Rule 89.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

GATE = "senpai/entry-point-cliff-census.sh"

PROOFS = (
    ("e121_fails", "da025231", "5d97175c", 1,
     "the E121 chunk-sum transplant loses a derived g17s simdgroup"),
    ("e126_passes", "5d97175c", "04171655", 0,
     "the E126 revert gives the derived g17s simdgroup back"),
    ("revert_clean", "da025231", "04171655", 0,
     "the E126 revert restores the pre-E121 register counts exactly"),
)


def run_gate(base: str, candidate: str | None, out: pathlib.Path) -> tuple[int, dict]:
    argv = [GATE, "--base", base, "--json", str(out)]
    if candidate is not None:
        argv += ["--candidate", candidate]
    proc = subprocess.run(argv, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode == 2:
        raise SystemExit("gate error: %s" % proc.stderr)
    return proc.returncode, json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="research/e131-artifacts")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    _, coverage = run_gate("HEAD", None, outdir / "rung3-coverage.json")

    proofs = []
    all_pass = True
    for name, base, candidate, want, claim in PROOFS:
        code, payload = run_gate(base, candidate, outdir / ("rung3-proof-%s.json" % name))
        ok = code == want
        all_pass &= ok
        deltas = {
            cell["kernel"]: cell["applegpu_g17s"]["simdgroups_delta_derived"]
            for cell in payload["cells"]
            if cell["applegpu_g17s"]["simdgroups_delta_derived"]
        }
        proofs.append({
            "proof": name,
            "claim": claim,
            "base_ref": base,
            "base_sha": payload["base_sha"],
            "candidate_ref": candidate,
            "candidate_sha": payload["candidate_sha"],
            "expected_exit_code": want,
            "observed_exit_code": code,
            "verdict": payload["verdict"],
            "g17s_simdgroup_deltas_derived": deltas,
            "runtime_seconds": payload["runtime_seconds"],
            "passed": ok,
        })

    slowest = max([coverage["runtime_seconds"]] + [p["runtime_seconds"] for p in proofs])
    receipt = {
        "experiment": "E131",
        "rung": 3,
        "tool": GATE,
        "schema_version": 1,
        "gpu_used": False,
        "model_loaded": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "harness": "compile-only census (xcrun metal-tt), not ranked and not local timing",
        "occupancy_label": "derived",
        "occupancy_rule": coverage["occupancy_rule"],
        "toolchain": coverage["toolchain"],
        "base_sha": coverage["base_sha"],
        "simdgroup_budget": coverage["simdgroup_budget"],
        "cells": {
            cell["kernel"]: {
                arch: {
                    "registers": cell[arch]["candidate_registers"],
                    "spill_bytes": cell[arch]["candidate_spill_bytes"],
                    "text_bytes": cell[arch]["candidate_text_bytes"],
                    "text_sha8": "",
                    "simdgroups": cell[arch]["candidate_simdgroups_derived"],
                }
                for arch in ("applegpu_g16s", "applegpu_g17s")
            }
            for cell in coverage["cells"]
        },
        "gate": {
            "entry_points_covered": len(coverage["cells"]),
            "jit_twin_cells": sum(1 for c in coverage["cells"] if c["source_form"] == "jit_twin"),
            "route_b_cells": sum(1 for c in coverage["cells"] if c["source_form"] == "swift_metal_kernel"),
            "coverage_verdict": coverage["verdict"],
            "proofs_passed": sum(1 for p in proofs if p["passed"]),
            "proofs_total": len(proofs),
            "all_proofs_passed": all_pass,
            "slowest_single_run_seconds": slowest,
            "suite_seconds": round(time.time() - started, 2),
            "budget_seconds": 30,
            "within_budget": slowest < 30,
        },
        "proofs": proofs,
        "coverage_cells": coverage["cells"],
        "warnings": coverage["warnings"],
    }
    path = outdir / "rung3-gate.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("wrote %s   proofs %d/%d   slowest %.2f s"
          % (path, receipt["gate"]["proofs_passed"], len(proofs), slowest))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
