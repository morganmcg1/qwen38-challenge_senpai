#!/usr/bin/env python3
"""Turn the E59 geometry probes into one pass/fail artifact.

Driven by ``research/e59_geometry_proof.sh``; see that script for what each
probe launches and why. This module only decides verdicts.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# name -> (nonzero_exit, low_memory_notice, precondition_message)
PROFILE_EXPECTATIONS = {
    "bogus": (True, 0, 1),
    "low": (True, 1, 0),
    "full": (True, 0, 0),
    "auto": (True, 1, 0),
}

# The dose response has to clear ordinary leg-to-leg noise by a wide margin to
# mean anything. Same-arm same-session spreads run well under 1 %; a command
# buffer committed every 8 operations instead of every 50 should cost far more.
DOSE_MIN_EFFECT_PCT = 3.0


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def load_dose(tag: str) -> dict[str, object]:
    root = pathlib.Path(
        os.environ.get("E59_E2E_ROOT", REPO / ".mlxfast-private" / "e59-e2e")
    )
    run = root / "runs" / tag
    meta = read_meta(run / "meta.txt")
    score = json.loads((run / "score.json").read_text())
    metrics = score["metrics"]
    return {
        "tag": tag,
        "requested_ops_per_buffer": int(meta["mlx_max_ops_per_buffer"]),
        "startup_memory_profile": meta["startup_memory_profile"],
        "wired_residency_active": meta.get("wired_residency_active"),
        "commit": meta.get("commit"),
        "decode_tokens": metrics["decode_tokens"],
        "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
        "serial_seconds_per_token": metrics["serial_seconds_per_token"],
        "all_tokens_matched": metrics["all_tokens_matched"],
        "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
        "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    work = pathlib.Path(os.environ["E59_GEOM_WORK"])
    dose_ran = os.environ.get("E59_GEOM_DOSE_RAN") == "1"

    profiles = []
    all_passed = True
    for line in (work / "profile-probes.tsv").read_text().splitlines():
        name, profile, rc, notice, precond = line.split("\t")
        want_nonzero, want_notice, want_precond = PROFILE_EXPECTATIONS[name]
        passed = (
            (int(rc) != 0) == want_nonzero
            and int(notice) == want_notice
            and int(precond) == want_precond
        )
        all_passed &= passed
        profiles.append(
            {
                "probe": name,
                "requested_profile": profile,
                "exit_code": int(rc),
                "low_memory_notices": int(notice),
                "precondition_messages": int(precond),
                "expected": {
                    "nonzero_exit": want_nonzero,
                    "low_memory_notices": want_notice,
                    "precondition_messages": want_precond,
                },
                "passed": passed,
            }
        )

    dose: dict[str, object] = {"ran": dose_ran}
    if dose_ran:
        slow = load_dose("e59-geom-ops8")
        fast = load_dose("e59-geom-ops50")
        effect = (
            (slow["mtp_seconds_per_token"] - fast["mtp_seconds_per_token"])
            / fast["mtp_seconds_per_token"]
            * 100.0
        )
        # ops=8 ran first and therefore colder, so thermal drift pushes this
        # effect DOWN. A positive effect that clears the floor is conservative.
        passed = effect >= DOSE_MIN_EFFECT_PCT
        all_passed &= passed
        dose.update(
            {
                "legs": [slow, fast],
                "ops8_slower_than_ops50_pct": effect,
                "min_effect_pct": DOSE_MIN_EFFECT_PCT,
                "order": "ops8 first (colder), ops50 second (warmer)",
                "passed": passed,
            }
        )
    else:
        all_passed = False
        dose["passed"] = False
        dose["note"] = "dose response skipped; geometry claim is unproven"

    artifact = {
        "experiment": "E59",
        "stage": "geometry-proof",
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "physical_memory_gib": int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        >> 30,
        "replaces": "worker_low_memory_notices grep (withdrawn: unfalsifiable "
        "on a timed leg, mtp-timed swallows worker stderr)",
        "profile_probes": profiles,
        "dose_response": dose,
        "all_passed": all_passed,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")

    for entry in profiles:
        print(
            f"profile {entry['probe']:6s} rc={entry['exit_code']:4d} "
            f"notice={entry['low_memory_notices']} "
            f"precond={entry['precondition_messages']} "
            f"{'PASS' if entry['passed'] else 'FAIL'}"
        )
    if dose_ran:
        print(
            f"dose    ops8 vs ops50 = {dose['ops8_slower_than_ops50_pct']:+.3f} % "
            f"(floor {DOSE_MIN_EFFECT_PCT} %) "
            f"{'PASS' if dose['passed'] else 'FAIL'}"
        )
    print(f"all_passed={all_passed} -> {out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
