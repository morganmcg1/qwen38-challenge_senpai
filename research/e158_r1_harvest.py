#!/usr/bin/env python3
"""E158 R1.C -- harvest the precision-island ABBA legs into one artifact.

    usage: python3 research/e158_r1_harvest.py [--out PATH]

Reads whichever of the two sessions have run:

  perprompt  .mlxfast-private/e128/runs-e158r1-<id>-<slot>-<arm>/<id>/
             `mtp-timed` legs. Candidate MTP leg only, so
             `parent_measured_seconds_per_token` is the leg RULE 177 prices.
             Ungated, ABBA-counterbalanced, entry and exit temperature kept.

  gated      research/out/e158r1g-<slot>-<arm>/
             `--local-iterate` legs behind the real 40 C gate, on the one public
             fixture. Carries the serial control leg as well, so the local
             serial-to-MTP ratio is available. That ratio is admissible for this
             change only because the causal path is confined to the candidate
             MTP leg: the serial control leg never drafts.

ABBA estimator. The slot order is all, none, none, all. Slots 1 and 4 average to
position 2.5 and slots 2 and 3 average to position 2.5, so a linear drift in
session position cancels exactly in mean(none) - mean(all).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = ["all", "none", "none", "all"]
PROMPTS = ["beagle_a", "essays_montaigne", "benchfixture"]

# Measured median weights supplied by the advisor on 2026-08-23: the share of
# ranked prompts for which each local prompt sets the published median.
MEDIAN_WEIGHTS = {"beagle_a": 0.4741, "essays_montaigne": 0.5259}

# Absolute-mtp noise floor for this host, in percent (advisor, 2026-08-23).
NOISE_FLOOR_PCT = 0.039


def read_meta(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def read_json(path: pathlib.Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def witness(directory: pathlib.Path) -> str | None:
    path = directory / "witness.txt"
    if not path.is_file():
        return None
    lines = sorted(
        {
            line.strip()
            for line in path.read_text().splitlines()
            if line.startswith("qwen-mtp-island-arm: ")
        }
    )
    if len(lines) != 1:
        return None
    return lines[0]


def perprompt_legs() -> list[dict]:
    legs = []
    for prompt in PROMPTS:
        for slot, arm in enumerate(ARMS, start=1):
            run = (
                ROOT
                / ".mlxfast-private/e128"
                / f"runs-e158r1-{prompt}-{slot}-{arm}"
                / prompt
            )
            report = read_json(run / "report.json")
            meta = read_meta(run / "meta.txt")
            if report is None:
                continue
            legs.append(
                {
                    "session": "perprompt",
                    "prompt": prompt,
                    "slot": slot,
                    "arm": arm,
                    "witness": witness(
                        ROOT
                        / ".mlxfast-private/e158r1"
                        / f"perprompt-{prompt}-{slot}-{arm}"
                    ),
                    "mtp_seconds_per_token": report[
                        "parent_measured_seconds_per_token"
                    ],
                    "decode_seconds": report["decode_seconds"],
                    "decode_token_count": report["decode_token_count"],
                    "round_count": report["round_count"],
                    "non_drafting_round_count": report.get(
                        "non_drafting_round_count"
                    ),
                    "effective_mean_draft_len": report[
                        "effective_mean_draft_len"
                    ],
                    "accepted_draft_rate": report["accepted_draft_rate"],
                    "accepted_draft_total": report["accepted_draft_total"],
                    "rejected_draft_total": report["rejected_draft_total"],
                    "all_tokens_matched": report["all_tokens_matched"],
                    "residual_divergence_count": report[
                        "residual_divergence_count"
                    ],
                    "p50_block_request_seconds": report[
                        "p50_block_request_seconds"
                    ],
                    "head_provenance_sha256": (report.get("head_provenance") or {}).get(
                        "sha256"
                    ),
                    "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
                    "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
                    "timing_valid": meta.get("timing_valid"),
                    "cool_gate_passed_real_gate": meta.get(
                        "cool_gate_passed_real_gate"
                    ),
                    "gate_qualified_for_timing": meta.get(
                        "gate_qualified_for_timing"
                    ),
                    "official_or_ranked_score": meta.get(
                        "official_or_ranked_score"
                    ),
                    "base_sha": meta.get("base_sha"),
                    "worker_sha256": meta.get("worker_sha256"),
                    "cli_sha256": meta.get("cli_sha256"),
                    "golden_sha256": meta.get("golden_sha256"),
                    "prompt_sha256": meta.get("prompt_sha256"),
                    "host": meta.get("host"),
                    "chip": meta.get("chip"),
                }
            )
    return legs


def gated_legs() -> list[dict]:
    legs = []
    for slot, arm in enumerate(ARMS, start=1):
        out = ROOT / "research/out" / f"e158r1g-{slot}-{arm}"
        score = read_json(out / "score.json")
        meta = read_meta(out / "meta.txt")
        if score is None:
            continue
        metrics = score["metrics"]
        legs.append(
            {
                "session": "gated",
                "prompt": "benchfixture",
                "slot": slot,
                "arm": arm,
                "witness": witness(
                    ROOT / ".mlxfast-private/e158r1" / f"gated-{slot}-{arm}"
                ),
                "mtp_seconds_per_token": metrics["mtp_seconds_per_token"],
                "serial_seconds_per_token": metrics["serial_seconds_per_token"],
                "mtp_decode_speedup": metrics["mtp_decode_speedup"],
                "decode_token_count": metrics["decode_tokens"],
                "effective_mean_draft_len": metrics["effective_mean_draft_len"],
                "accepted_draft_rate": metrics["accepted_draft_rate"],
                "all_tokens_matched": metrics["all_tokens_matched"],
                "residual_divergence_count": metrics["residual_divergence_count"],
                "head_provenance_sha256": metrics["head_provenance_sha256"],
                "gpu_temp_entry_c": meta.get("gpu_temp_entry_c"),
                "gpu_temp_exit_c": meta.get("gpu_temp_exit_c"),
                "cool_gate_passed_real_gate": meta.get(
                    "cool_gate_passed_real_gate"
                ),
                "gate_qualified_for_timing": meta.get("gate_qualified_for_timing"),
                "official_or_ranked_score": meta.get("official_or_ranked_score"),
                "island_arm_requested": meta.get("island_arm_requested"),
                "base_sha": meta.get("base_sha"),
                "worker_sha256": meta.get("worker_sha256"),
                "cli_sha256": meta.get("cli_sha256"),
                "host": meta.get("host"),
                "chip": meta.get("chip"),
                "started": meta.get("started"),
                "finished": meta.get("finished"),
            }
        )
    return legs


def abba(legs: list[dict], field: str) -> dict | None:
    by_arm: dict[str, list[float]] = {"all": [], "none": []}
    for leg in legs:
        value = leg.get(field)
        if value is None:
            return None
        by_arm[leg["arm"]].append(float(value))
    if len(by_arm["all"]) != 2 or len(by_arm["none"]) != 2:
        return None
    mean_all = statistics.fmean(by_arm["all"])
    mean_none = statistics.fmean(by_arm["none"])
    delta = mean_none - mean_all
    return {
        "all": by_arm["all"],
        "none": by_arm["none"],
        "mean_all": mean_all,
        "mean_none": mean_none,
        "delta_none_minus_all": delta,
        "delta_pct_of_all": 100.0 * delta / mean_all if mean_all else None,
        "within_arm_spread_all": abs(by_arm["all"][0] - by_arm["all"][1]),
        "within_arm_spread_none": abs(by_arm["none"][0] - by_arm["none"][1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="research/e158-artifacts/r1-abba.json"
    )
    args = parser.parse_args()

    legs = perprompt_legs() + gated_legs()
    if not legs:
        print("e158_r1_harvest: no legs found", file=sys.stderr)
        return 1

    summary: dict = {
        "experiment": "e158-r1-precision-island-price",
        "harness": "local",
        "noise_floor_pct_absolute_mtp": NOISE_FLOOR_PCT,
        "median_weights": MEDIAN_WEIGHTS,
        "abba_order": ARMS,
        "legs": legs,
        "per_prompt": {},
        "gated": {},
    }

    for prompt in PROMPTS:
        group = [
            leg
            for leg in legs
            if leg["session"] == "perprompt" and leg["prompt"] == prompt
        ]
        if len(group) != 4:
            continue
        summary["per_prompt"][prompt] = {
            "mtp_seconds_per_token": abba(group, "mtp_seconds_per_token"),
            "effective_mean_draft_len": abba(group, "effective_mean_draft_len"),
            "accepted_draft_rate": abba(group, "accepted_draft_rate"),
            "round_count": abba(group, "round_count"),
            "rejected_draft_total": abba(group, "rejected_draft_total"),
            "all_tokens_matched": all(leg["all_tokens_matched"] for leg in group),
            "witnesses_ok": all(
                leg["witness"] is not None
                and leg["witness"].startswith(
                    f"qwen-mtp-island-arm: {leg['arm']} "
                )
                for leg in group
            ),
        }

    weighted = 0.0
    complete = True
    for prompt, weight in MEDIAN_WEIGHTS.items():
        cell = summary["per_prompt"].get(prompt, {}).get("mtp_seconds_per_token")
        if not cell:
            complete = False
            continue
        weighted += weight * cell["delta_pct_of_all"]
    if complete:
        gain = -weighted / 100.0
        summary["median_weighted_candidate_leg"] = {
            "delta_pct_of_all": weighted,
            "gain_fraction_g": gain,
            "published_pct_rule176": 100.0 * gain / (1.0 - gain)
            if gain < 1.0
            else None,
            "beats_noise_floor": abs(weighted) > NOISE_FLOOR_PCT,
        }

    gated = [leg for leg in legs if leg["session"] == "gated"]
    if len(gated) == 4:
        summary["gated"] = {
            "mtp_seconds_per_token": abba(gated, "mtp_seconds_per_token"),
            "serial_seconds_per_token": abba(gated, "serial_seconds_per_token"),
            "mtp_decode_speedup": abba(gated, "mtp_decode_speedup"),
            "effective_mean_draft_len": abba(gated, "effective_mean_draft_len"),
            "accepted_draft_rate": abba(gated, "accepted_draft_rate"),
            "all_tokens_matched": all(leg["all_tokens_matched"] for leg in gated),
            "gate_qualified_for_timing": all(
                leg["gate_qualified_for_timing"] == "true" for leg in gated
            ),
            "witnesses_ok": all(
                leg["witness"] is not None
                and leg["witness"].startswith(
                    f"qwen-mtp-island-arm: {leg['arm']} "
                )
                for leg in gated
            ),
        }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            key: summary[key]
            for key in ("per_prompt", "gated", "median_weighted_candidate_leg")
            if key in summary
        },
        indent=2,
        sort_keys=True,
    ))
    print(f"e158_r1_harvest: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
