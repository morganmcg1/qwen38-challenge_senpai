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

# name -> (exact_exit_code_or_None, low_memory_notice, precondition_message)
#
# The release build strips `preconditionFailure` text, so a rejected profile
# name proves the parser is live only through its trap exit code 133 (128 + 5,
# SIGTRAP). Every other probe reaches the weight load and fails there, so those
# rows only require a nonzero exit.
PROFILE_EXPECTATIONS = {
    "bogus": (133, 0, 0),
    "low": (None, 1, 0),
    "full": (None, 0, 0),
    "auto": (None, 1, 0),
}

# The dose response has to clear ordinary leg-to-leg noise by a wide margin to
# mean anything. Same-arm same-session spreads run well under 1 %; a command
# buffer committed every 8 operations instead of every 50 should cost far more.
DOSE_MIN_EFFECT_PCT = 3.0

# label -> (tight-cap tag, loose-cap tag). The moderate pair asks whether the
# shipped 50-op cap is anywhere near binding. The extreme pair asks the much
# weaker question of whether the variable reaches MLX at all: at a cap of 1 the
# encoder commits roughly every second operation.
DOSE_PAIRS = [
    ("moderate ops=8 vs ops=50", "e59-geom-ops8", "e59-geom-ops50"),
    ("extreme ops=1 vs ops=50", "e59-geom-ops1", "e59-geom-ops50x"),
]


def read_meta(path: pathlib.Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()
    return meta


def load_dose(tag: str) -> dict[str, object] | None:
    root = pathlib.Path(
        os.environ.get("E59_E2E_ROOT", REPO / ".mlxfast-private" / "e59-e2e")
    )
    run = root / "runs" / tag
    if not (run / "score.json").exists():
        return None
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
        want_rc, want_notice, want_precond = PROFILE_EXPECTATIONS[name]
        rc_ok = int(rc) == want_rc if want_rc is not None else int(rc) != 0
        passed = rc_ok and int(notice) == want_notice and int(precond) == want_precond
        all_passed &= passed
        profiles.append(
            {
                "probe": name,
                "requested_profile": profile,
                "exit_code": int(rc),
                "low_memory_notices": int(notice),
                "precondition_messages": int(precond),
                "expected": {
                    "exit_code": want_rc if want_rc is not None else "nonzero",
                    "low_memory_notices": want_notice,
                    "precondition_messages": want_precond,
                },
                "passed": passed,
            }
        )

    dose: dict[str, object] = {"ran": dose_ran, "pairs": []}
    if dose_ran:
        for label, slow_tag, fast_tag in DOSE_PAIRS:
            slow = load_dose(slow_tag)
            fast = load_dose(fast_tag)
            if slow is None or fast is None:
                continue
            effect = (
                (slow["mtp_seconds_per_token"] - fast["mtp_seconds_per_token"])
                / fast["mtp_seconds_per_token"]
                * 100.0
            )
            # The tighter cap always runs first and therefore colder, so
            # thermal drift pushes this effect DOWN. A positive effect that
            # clears the floor is conservative.
            dose["pairs"].append(
                {
                    "label": label,
                    "legs": [slow, fast],
                    "tight_slower_than_loose_pct": effect,
                    "min_effect_pct": DOSE_MIN_EFFECT_PCT,
                    "order": "tight cap first (colder), loose cap second (warmer)",
                    "passed": effect >= DOSE_MIN_EFFECT_PCT,
                }
            )
        # One responding dose proves the export reaches MLX. A null at the
        # moderate cap alone cannot separate "export ignored" from "commits are
        # free at the roofline"; the extreme cap can.
        dose["passed"] = any(pair["passed"] for pair in dose["pairs"])
        dose["note"] = (
            "any responding pair proves MLX_MAX_OPS_PER_BUFFER reaches the "
            "worker; an all-null result leaves the leg geometry unproven"
        )
        all_passed &= bool(dose["passed"])
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
    for pair in dose["pairs"]:
        print(
            f"dose    {pair['label']:26s} = "
            f"{pair['tight_slower_than_loose_pct']:+.3f} % "
            f"(floor {DOSE_MIN_EFFECT_PCT} %) "
            f"{'PASS' if pair['passed'] else 'FAIL'}"
        )
    print(f"all_passed={all_passed} -> {out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
