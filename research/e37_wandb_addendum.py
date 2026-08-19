#!/usr/bin/env python3
"""Attach the E37 r2 addendum evidence to the EXISTING census run.

The addendum adds no measurement -- it re-derives the startup memory regime and
the shipped-surface baseline from source -- so it belongs on the run that
already carries the census rather than on a new run id that would fork the
record. This resumes h977ws5a in place.

  python3 research/e37_wandb_addendum.py [--run-id h977ws5a]

Refuses to run unless both addendum gates pass, so the record cannot claim a
verification that did not happen.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import wandb

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "qwen38-mlx-challenge-senpai"
ENTITY = "wandb-applied-ai-team"
REPORT = "research/results/qwen38-r1-e37-draft-width-census-beagle-medicine.md"
REGIME_JSON = "research/results/e37/r2-alloc-regime.json"
SURFACE_JSON = "research/results/e37/r2-shipped-surface.json"
REGIME_TXT = "research/results/e37/r2-alloc-regime.txt"
SURFACE_TXT = "research/results/e37/r2-shipped-surface.txt"


def rerun_gates() -> None:
    """Never log a stale verdict: regenerate both artifacts first."""
    for script, out in (("research/e37_alloc_regime.py", REGIME_TXT),
                        ("research/e37_shipped_surface.py", SURFACE_TXT)):
        proc = subprocess.run([sys.executable, script], cwd=REPO,
                              capture_output=True, text=True)
        (REPO / out).write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit("%s exited %d -- refusing to log a failing gate"
                             % (script, proc.returncode))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="h977ws5a")
    args = ap.parse_args()

    rerun_gates()
    regime = json.loads((REPO / REGIME_JSON).read_text())
    surface = json.loads((REPO / SURFACE_JSON).read_text())
    if not (regime["gate_pass"] and surface["gate_pass"]):
        raise SystemExit("a gate reported FAIL; not logging")

    run = wandb.init(entity=ENTITY, project=PROJECT, id=args.run_id,
                     resume="must")

    checks = wandb.Table(columns=["id", "check", "pass", "detail", "evidence"])
    for c in regime["checks"]:
        checks.add_data(c["id"], c["what"], c["pass"], c["detail"], c["evidence"])

    controls = wandb.Table(columns=["mutation", "target_check", "file",
                                    "killed_the_check", "note"])
    for c in regime["negative_controls"]:
        controls.add_data(c["id"], c["target"], c["file"], c["ok"], c["why"])

    table = wandb.Table(columns=["host_gib", "profile", "mlx_max_mb_per_buffer",
                                 "mlx_max_ops_per_buffer", "memory_cache_limit",
                                 "postwarm_clear_cache", "wired_residency"])
    for r in regime["regime"]:
        table.add_data(r["gib"], "low" if r["low_profile"] else "full(unapplied)",
                       r["mlx_max_mb_per_buffer"], r["mlx_max_ops_per_buffer"],
                       r["memory_cache_limit"], r["postwarm_clear_cache"],
                       r["wired_residency"])

    run.log({"addendum_alloc_checks": checks,
             "addendum_alloc_negative_controls": controls,
             "addendum_alloc_regime": table})

    local = next(r for r in regime["regime"] if r["gib"] == 48)
    ranked = next(r for r in regime["regime"] if r["gib"] == 128)
    run.summary.update({
        "addendum/alloc_checks_passed": sum(1 for c in regime["checks"] if c["pass"]),
        "addendum/alloc_checks_total": len(regime["checks"]),
        "addendum/alloc_controls_killed": sum(1 for c in regime["negative_controls"]
                                              if c["ok"]),
        "addendum/alloc_controls_total": len(regime["negative_controls"]),
        "addendum/alloc_gate_pass": regime["gate_pass"],
        # The one row of the advisor's table that inverts on this path.
        "addendum/local_postwarm_clear_cache": local["postwarm_clear_cache"],
        "addendum/ranked_postwarm_clear_cache": ranked["postwarm_clear_cache"],
        "addendum/local_wired_residency": local["wired_residency"],
        "addendum/ranked_wired_residency": ranked["wired_residency"],
        "addendum/clear_flag_inert_on_mtp_path": True,
        "addendum/full_profile_applied_on_any_host": False,
        "addendum/warm_negative_is_regime_invariant": True,
        "addendum/warm_negative_holds_a_fortiori_on_ranked": True,
        "addendum/counts_affected_by_memory_regime": False,
        "surface/campaign_shipped_files": surface["campaign_shipped_surface"]["n_files"],
        "surface/campaign_shipped_insertions":
            surface["campaign_shipped_surface"]["insertions"],
        "surface/campaign_shipped_deletions":
            surface["campaign_shipped_surface"]["deletions"],
        "surface/baseline_commit": surface["baseline"][:8],
        "surface/branch_shipped_files": surface["branch_on_shipped_surface"]["n_files"],
        "surface/branch_non_research_files": len(surface["branch_non_research_files"]),
        "surface/scope_pass": surface["scope_pass"],
        "surface/negative_control_fires": surface["negative_control_fires"],
    })

    art = wandb.Artifact("e37-addendum", type="verification")
    for rel in (REGIME_JSON, SURFACE_JSON, REGIME_TXT, SURFACE_TXT,
                "research/e37_alloc_regime.py", "research/e37_shipped_surface.py",
                REPORT):
        art.add_file(str(REPO / rel), name=rel)
    run.log_artifact(art)
    print("updated %s" % run.url)
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
