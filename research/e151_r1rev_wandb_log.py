#!/usr/bin/env python3
"""E151 revision r2: log the section 4 artifacts and the single gate pass.

One run, written twice. The first call registers the section 4 histogram and
the gating decision BEFORE any gate re-runs, which is what the r2 request
requires. The second call resumes the same run and adds the gate results.

HARNESS LABELLING.
  harness=offline  static source analysis. The section 4 histogram, the
                   dispatch model and the self-contained re-application patch.
                   Zero GPU seconds, no timing.
  harness=local    anything measured on this M4 Pro. `is_nax_available()` is
                   false here, so the retiled NAX kernel NEVER EXECUTES on this
                   host. The local gate chain proves only that the non-NAX path
                   is unbroken and that the candidate is submission ready. It
                   is not, and cannot be, an effect estimate for the arm.
  harness=ranked   published receipt values only. Not re-derived here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import wandb

PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
RUN_ID = "e151r1rev2"

BASE_SHA = "14247cce11216639a04ecfc2798cf0798091ff92"
GROWTH_BASE_SHA = "770a3ff2f8fbd1bb75d15e3c37ae3c5b076ebbcf"
GATE_PROVED_SURFACE = {
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized_nax.h":
        "50cf7876",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp":
        "e7c55209",
}

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def digest8(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:8]


def flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}_{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj)
    else:
        out[prefix] = obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", help="gate-result JSON to append")
    args = ap.parse_args()

    hist = json.loads((ROOT / "research/e151-r1-m-histogram.json").read_text())

    observed = {rel: digest8(rel) for rel in GATE_PROVED_SURFACE}
    byte_identical = observed == GATE_PROVED_SURFACE

    summary = {
        "e151_harness_note": "offline + local only; no ranked measurement here",
        "e151_base_sha": BASE_SHA,
        "e151_head_sha": sh("git", "rev-parse", "HEAD"),
        "e151_r2_parked_sha": "a6fccd2b",
        "e151_r1_submitted_surface_byte_identical_to_gate_proved_tree":
            int(byte_identical),
    }
    for rel, dig in observed.items():
        summary[f"e151_surface_digest_{pathlib.Path(rel).name}"] = dig

    flatten("e151_r1_m_histogram", hist["e151_r1_scored_m_histogram"], summary)
    summary.update(hist["verdict"])
    flatten("e151_r1_gating_decision", hist["registered_gating_decision"],
            summary)

    if args.gates:
        gates = json.loads(pathlib.Path(args.gates).read_text())
        flatten("", gates, summary)

    run = wandb.init(
        entity=ENTITY, project=PROJECT, id=RUN_ID, resume="allow",
        name="e151-r1-revision-r2",
        job_type="offline-analysis" if not args.gates else "gate-chain",
        tags=["e151", "r1", "revision-r2", "harness=offline",
              "harness=local", "prefill", "nax-retile"],
        config={
            "experiment": "E151",
            "revision": "r2",
            "base_sha": BASE_SHA,
            "growth_base_sha": GROWTH_BASE_SHA,
            "host": "aws-mac g16s (is_nax_available=false)",
            "arm": "128x32 NAX seed-prefill retile, ON",
        },
    )
    run.summary.update(summary)
    print(f"run={run.id} url={run.url}")
    print(f"byte_identical={byte_identical} observed={observed}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
