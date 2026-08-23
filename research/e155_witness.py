#!/usr/bin/env python3
"""E155 -- two-sided witness that the exact readout is gated off by default.

The claim under test is that the audit instrument does nothing unless
`MLX_E155_RECALL_AUDIT=1` is set. One side alone cannot support it:

  off side  the gate is unset, so the audit file must not appear and the
            decode must be digit-identical to a run of the same build without
            the instrument, and to the pre-instrument E153 legs.
  on  side  the gate is set, so the audit file must appear with one line per
            proposal slot. Without this side the off side is vacuous, because
            a check that can never fire also reports "off".

Both sides run the same worker binary. Only the environment differs, so a
difference between them is caused by the gate and by nothing else.

usage: research/e155_witness.py OUT.json
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OFF = ROOT / ".mlxfast-private/e128/runs-e155-off"
ON = ROOT / ".mlxfast-private/e128/runs-e155-on"
PRIOR = ROOT / ".mlxfast-private/e128/runs-e153r1"
AUDIT = ROOT / ".mlxfast-private/e155"
PROMPTS = ("beagle_a", "essays_montaigne", "benchfixture")


def meta(path: pathlib.Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


def report(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    legs = []
    off_matches_on = True
    off_matches_prior = True
    audit_present_off = False
    audit_present_on = True

    for prompt in PROMPTS:
        off_meta = meta(OFF / prompt / "meta.txt")
        on_meta = meta(ON / prompt / "meta.txt")
        off_report = report(OFF / prompt / "report.json")
        on_report = report(ON / prompt / "report.json")
        audit_path = AUDIT / ("%s.jsonl" % prompt)
        # Slot rows only. The instrument also writes install and per-round
        # diagnostic lines, which carry no measurement.
        audit_lines = (
            sum(1 for line in audit_path.open() if '"slot":' in line)
            if audit_path.exists() else 0
        )

        prior_path = PRIOR / prompt / prompt / "report.json"
        prior_report = report(prior_path) if prior_path.exists() else None

        same_drafts = (
            off_report["effective_draft_lengths"]
            == on_report["effective_draft_lengths"]
        )
        same_tokens = (
            off_report["accepted_draft_total"]
            == on_report["accepted_draft_total"]
            and off_report["round_count"] == on_report["round_count"]
            and off_report["declared_rows_total"]
            == on_report["declared_rows_total"]
        )
        same_prior = (
            prior_report is not None
            and prior_report["effective_draft_lengths"]
            == off_report["effective_draft_lengths"]
        )
        expected_lines = sum(off_report["effective_draft_lengths"])

        off_matches_on = off_matches_on and same_drafts and same_tokens
        if prior_report is not None:
            off_matches_prior = off_matches_prior and same_prior
        audit_present_on = audit_present_on and audit_lines == expected_lines

        legs.append({
            "prompt": prompt,
            "worker_sha256_off": off_meta["worker_sha256"],
            "worker_sha256_on": on_meta["worker_sha256"],
            "same_worker_binary": off_meta["worker_sha256"]
            == on_meta["worker_sha256"],
            "base_sha_off": off_meta["base_sha"],
            "base_sha_on": on_meta["base_sha"],
            "golden_sha256": off_meta["golden_sha256"],
            "rounds_off": off_report["round_count"],
            "rounds_on": on_report["round_count"],
            "accepted_off": off_report["accepted_draft_total"],
            "accepted_on": on_report["accepted_draft_total"],
            "all_tokens_matched_off": off_report["all_tokens_matched"],
            "all_tokens_matched_on": on_report["all_tokens_matched"],
            "residual_divergence_off": off_report["residual_divergence_count"],
            "residual_divergence_on": on_report["residual_divergence_count"],
            "effective_draft_lengths_identical_off_vs_on": same_drafts,
            "round_ledger_identical_off_vs_on": same_tokens,
            "effective_draft_lengths_identical_vs_e153": same_prior,
            "audit_rows_written_on": audit_lines,
            "audit_rows_expected": expected_lines,
            "gpu_temp_entry_c_off": float(off_meta["gpu_temp_entry_c"]),
            "gpu_temp_exit_c_off": float(off_meta["gpu_temp_exit_c"]),
            "gpu_temp_entry_c_on": float(on_meta["gpu_temp_entry_c"]),
            "gpu_temp_exit_c_on": float(on_meta["gpu_temp_exit_c"]),
        })

    # Off side: the audit creates its file at install time, before any round,
    # so file absence is the direct observable. It does not depend on the
    # worker's stderr, which the trusted harness drops.
    #
    # The three audit-off legs ran with neither variable set. The probe leg
    # ran with the output path set and the gate variable unset, which is what
    # names the gate variable as the cause.
    probe = report(ROOT / "research/e155-artifacts/probe.json")
    audit_present_off = probe["audit_file_created"]

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()

    payload = {
        "harness": "local",
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "timing_valid": False,
        "base_sha": head,
        "host": os.uname().nodename,
        "e155_exact_path_is_gated_off_by_default": bool(
            off_matches_on and audit_present_on and not audit_present_off
        ),
        "witness_off_side_no_audit_output": not audit_present_off,
        "witness_off_side_probe": probe,
        "witness_off_side_decode_identical_to_audit_on": off_matches_on,
        "witness_off_side_decode_identical_to_pre_instrument_e153":
            off_matches_prior,
        "witness_on_side_audit_output_complete": audit_present_on,
        "legs": legs,
        # Nothing was deleted from the submitted surface by this experiment,
        # and the instrument is reverted before the PR head, so the submitted
        # byte count returns to its base value.
        "e155_growth_reclaimed_bytes": 0,
    }
    with open(sys.argv[1], "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(
        {k: v for k, v in payload.items() if k != "legs"}, indent=2))
    return 0 if payload["e155_exact_path_is_gated_off_by_default"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
