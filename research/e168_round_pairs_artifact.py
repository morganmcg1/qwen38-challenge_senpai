#!/usr/bin/env python3
"""Publish the E168 per-round cost pairs to W&B so E173 can decompose `F`.

    usage: research/e168_round_pairs_artifact.py [--root research/out/e168]

Alphonse owns E173 and needs the per-round `(block_request_seconds,
effective_draft_length)` pairs that produced the E168 round-cost table. He
cannot fetch them from this branch, so they go to the campaign project as an
artifact.

The artifact carries the raw `04-mtp-timed.json` reports and their `meta.txt`
provenance, plus one tidy `round_pairs.csv` so a reader does not have to
re-derive the pairing. Every caveat that limits what these numbers can support
is written into the artifact description and metadata, not left to the reader.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import tempfile

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
GROUP = "e168-margin-clamp-calibration"

# `activeInputGroups` at Qwen35.swift:1709-1728, as a weight-pass count.
GROUPS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}

DESCRIPTION = """\
Per-round cost pairs from the E168 traced census, for the E173 decomposition
of the fixed round term `F`.

CONTENTS
  round_pairs.csv   one row per decode round, every arm and prompt
  raw/<arm>/<prompt>/04-mtp-timed.json   the trusted parent's own report
  raw/<arm>/<prompt>/meta.txt            provenance and gate flags for that leg

round_pairs.csv columns
  arm                 adapt | p2 | p7   (p<N> pins the proposed draft count to N
                      through MLX_E159_FIXED_DRAFT_DEPTH; adapt is the shipped
                      adaptive schedule)
  prompt              prompt id; `.r2`/`.r3` suffixes are repeats of one prompt
  round_index         position of the round in the decode sequence, 0-based
  block_seconds       block_request_seconds for that round
  effective_draft_len drafts actually proposed in that round
  m                   1 + effective_draft_len, the verified width
  weight_passes       activeInputGroups(m): 1 for m <= 5, 2 for m in 6..8, 3 at 9

CAVEATS. All three limit what these numbers can support.

1. NOT GATE-QUALIFIED TIMING. Every leg ran with MLXFAST_LOCAL_COOL_GATE=0 and
   with the per-round phase trace ON. The leg metadata says so verbatim:
   cool_gate_passed_real_gate=false, gate_qualified_for_timing=false,
   timing_claims_permitted=false, trace_perturbs_timing=true. No value here is
   a score, and none may be compared with a gated historical run.

2. THE TRACE ADDS A PER-ROUND TAX. FINDING 413 prices the research
   instrumentation at about 592 microseconds per drafting round, and finds it
   depth-independent. Because it is a fixed per-round addend, it inflates every
   round roughly equally and therefore biases per-token cost against SHALLOW
   rounds, which amortise it over fewer tokens.

3. M BINS ARE NOT RANDOMLY ASSIGNED. This is the important one. Deep rounds
   occur where the text is predictable, so `m` is confounded with text
   predictability and with acceptance. Do not read a difference between `m`
   bins as a cost of `m`. `round_index` is included precisely so text position
   can be controlled for, and the accept fields in the raw reports allow
   conditioning on acceptance.

HOST. g16s, not the ranked M5. FINDING 410 records that the within-group
pass-efficiency curve is much steeper on this host than the ranked receipts
permit, so the SHAPE transfers and the STEEPNESS does not.
"""


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research/out/e168")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    reports = sorted(root.glob("*/*/reports/04-mtp-timed.json"))
    if not reports:
        raise SystemExit(f"{root}: no 04-mtp-timed.json reports")

    run = wandb.init(
        entity=ENTITY, project=PROJECT, group=GROUP,
        name="e168-round-cost-pairs", job_type="artifact",
        config={
            "experiment": "e168-round-cost-pairs",
            "harness": "local",
            "host": "g16s",
            "leg_kind": "traced-census",
            "cool_gate_passed_real_gate": False,
            "gate_qualified_for_timing": False,
            "timing_claims_permitted": False,
            "trace_perturbs_timing": True,
            "official_score": False,
            "for_experiment": "e173-fixed-term-decomposition",
            "candidate_head": git("rev-parse", "HEAD"),
        },
    )

    artifact = wandb.Artifact(
        "e168-round-cost-pairs", type="measurement",
        description=DESCRIPTION,
        metadata={
            "reports": len(reports),
            "gate_qualified_for_timing": False,
            "trace_perturbs_timing": True,
            "m_bins_randomly_assigned": False,
            "host": "g16s",
        },
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        csv_path = tmp_path / "round_pairs.csv"
        rows = 0
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "arm", "prompt", "round_index", "block_seconds",
                "effective_draft_len", "m", "weight_passes",
            ])
            for report_path in reports:
                prompt_dir = report_path.parent.parent
                arm = prompt_dir.parent.name
                prompt = prompt_dir.name
                report = json.loads(report_path.read_text())
                seconds = report["block_request_seconds"]
                lengths = report["effective_draft_lengths"]
                if len(seconds) != len(lengths):
                    raise SystemExit(
                        f"{report_path}: {len(seconds)} block times against "
                        f"{len(lengths)} draft lengths; the pairing is unsound"
                    )
                for index, (block, length) in enumerate(zip(seconds, lengths)):
                    m = 1 + length
                    writer.writerow([
                        arm, prompt, index, block, length, m,
                        GROUPS.get(m, 1),
                    ])
                    rows += 1
                artifact.add_file(
                    str(report_path), name=f"raw/{arm}/{prompt}/04-mtp-timed.json"
                )
                meta_path = prompt_dir / "meta.txt"
                if meta_path.exists():
                    artifact.add_file(
                        str(meta_path), name=f"raw/{arm}/{prompt}/meta.txt"
                    )

        artifact.add_file(str(csv_path), name="round_pairs.csv")
        artifact.metadata["rounds"] = rows
        run.log_artifact(artifact)
        run.summary.update({"reports": len(reports), "rounds": rows})
        print(f"{len(reports)} reports, {rows} rounds")
        print(f"logged {run.url}")
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
