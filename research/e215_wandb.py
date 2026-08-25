#!/usr/bin/env python3
"""Publish the E215 schedule-surface evidence to W&B.

    python3 research/e215_wandb.py [--dry-run]

Three runs in group `qwen38-r1-e215-schedule-evidence`:

  exactness   Q1: fixed-window exactness through and past the stop token on
              five fixtures and both schedule arms.
  census      Q2: the ship-vs-stepq schedule census per prompt.
  pin         Q3: the Swift pin test and the suite floor check.

RULE 79. Every leg behind these runs is ungated, traced and unsandboxed, so no
second measured in them is a price. No timing field is published here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e215-schedule-evidence"
ARTIFACTS = pathlib.Path("research/e215-artifacts")
ASSIGNMENT_BASE = "32a53a58fc7bb33f996299b63c49882db673c8cc"


def head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def load(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def identity(extra: dict) -> dict:
    return {
        "experiment": "e215-schedule-evidence",
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "commitSha": head_sha(),
        "shippedArm": "stepq",
        "controlArm": "ship",
        "armDiff": "one line: depthPriceArm .stepq -> .ship",
        "scoredSurfaceDiff": "none; research/ and Tests/ only",
        "harness": "local",
        "officialOrRankedScore": False,
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "timingClaimsPermitted": False,
        "secondsPerTokenIsAPrice": False,
        **extra,
    }


def exactness_summary(exactness: dict) -> dict:
    out = {
        "allLegsMatched": exactness["all_legs_matched"],
        "legCount": len(exactness["legs"]),
    }
    for leg in exactness["legs"]:
        key = "%s/%s" % (leg["prompt"], leg["arm"])
        out["%s/allTokensMatched" % key] = leg["all_tokens_matched"]
        out["%s/residualDivergenceCount" % key] = leg["residual_divergence_count"]
        out["%s/parityAllOk" % key] = leg["parity_all_ok"]
        out["%s/rowLedgerClosed" % key] = leg["row_ledger_closed"]
        out["%s/rowLedgerParts" % key] = leg["row_ledger_parts"]
        out["%s/declaredRowsTotal" % key] = leg["declared_rows_total"]
        out["%s/targetCacheOffsetFinal" % key] = leg["target_cache_offset_final"]
        out["%s/firstStopTokenIndex" % key] = leg["first_stop_token_index"]
        out["%s/firstStopTokenId" % key] = leg["first_stop_token_id"]
        out["%s/stopTokenCountInWindow" % key] = leg["stop_token_count_in_window"]
        out["%s/tokensAfterFirstStopToken" % key] = leg[
            "tokens_after_first_stop_token"
        ]
        out["%s/printedPriceArm" % key] = leg["printed_price_arm"]
        out["%s/workerSha256" % key] = leg["worker_sha256"]
        out["%s/gpuTempEntryC" % key] = leg["gpu_temp_entry_c"]
        out["%s/gpuTempExitC" % key] = leg["gpu_temp_exit_c"]
    return out


def census_summary(census: dict, widths: dict) -> dict:
    out = {
        "directionHoldsOnEveryPrompt": census["direction_holds_on_every_prompt"],
        "promptsWithDirection": census["prompts_with_direction"],
        "promptCount": census["prompt_count"],
    }
    for prompt, entry in census["prompts"].items():
        for arm in ("stepq", "ship"):
            out["%s/%s/rounds" % (prompt, arm)] = entry[arm]["rounds"]
            out["%s/%s/edl" % (prompt, arm)] = entry[arm]["effective_mean_draft_len"]
            out["%s/%s/acceptedRate" % (prompt, arm)] = entry[arm][
                "accepted_draft_rate"
            ]
            out["%s/%s/declaredRows" % (prompt, arm)] = entry[arm][
                "declared_rows_total"
            ]
        for key, value in entry["delta"].items():
            out["%s/delta/%s" % (prompt, key)] = value
    if widths:
        for prompt, entry in widths["prompts"].items():
            for width, count in entry["rounds_by_served_width"].items():
                out["widthCensus/%s/w%s" % (prompt, width)] = count
        for width, count in widths["pooled_all_prompts"][
            "rounds_by_served_width"
        ].items():
            out["widthCensus/pooled/w%s" % width] = count
    return out


def pin_summary(pin: dict) -> dict:
    return {
        "pinTestsPassed": pin["pin_tests_passed"],
        "pinTestCount": pin["pin_test_count"],
        "suiteIssuesBase": pin["suite_issues_base"],
        "suiteIssuesBranch": pin["suite_issues_branch"],
        "suiteFailingNameDiffEmpty": pin["failing_name_diff_empty"],
        "suiteFailingNamesAdded": ",".join(pin["failing_names_added"]) or "none",
        "suiteFailingNamesRemoved": ",".join(pin["failing_names_removed"]) or "none",
        "pinnedArm": pin["pinned_arm"],
        "pinnedThresholdsHex": ",".join(pin["pinned_thresholds_hex"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exactness = load("exactness.json")
    census = load("census.json")
    widths = load("width-census-stepq.json")
    pin = load("pin-test.json")

    specs = {}
    if exactness:
        specs["exactness"] = (
            {
                "job_type": "exactness",
                "name": "e215-post-eos-exactness",
                "config": identity(
                    {
                        "deliverable": "fixed-window exactness through and past "
                        "the stop token, both arms, five fixtures",
                        "gpuUsed": True,
                        "tokens": 512,
                        "localMode": "--local-iterate",
                    }
                ),
            },
            exactness_summary(exactness),
        )
    if census:
        specs["census"] = (
            {
                "job_type": "census",
                "name": "e215-multi-prompt-census",
                "config": identity(
                    {
                        "deliverable": "ship vs stepq schedule census on five "
                        "prompts, plus the composed-base served-width census",
                        "gpuUsed": True,
                        "tokens": 512,
                        "localMode": "--local-iterate",
                    }
                ),
            },
            census_summary(census, widths),
        )
    if pin:
        specs["pin"] = (
            {
                "job_type": "pin",
                "name": "e215-table-pin-test",
                "config": identity(
                    {
                        "deliverable": "Swift pin test for the shipped arm and "
                        "threshold table, against the suite floor",
                        "gpuUsed": False,
                    }
                ),
            },
            pin_summary(pin),
        )

    if not specs:
        raise SystemExit("no E215 artifacts to publish yet")

    if args.dry_run:
        print(
            json.dumps(
                {k: {"config": s["config"], "summary": v} for k, (s, v) in specs.items()},
                indent=1,
                sort_keys=True,
                default=float,
            )
        )
        return

    import wandb

    path = ARTIFACTS / "wandb.json"
    published = json.loads(path.read_text()) if path.exists() else {}
    existing = {key: value["run"] for key, value in published.items()}
    for key, (spec, summary) in specs.items():
        extra = (
            {"id": existing[key], "resume": "allow"} if key in existing else {}
        )
        run = wandb.init(
            project=PROJECT.split("/")[-1],
            entity=PROJECT.split("/")[0],
            group=GROUP,
            **spec,
            **extra,
        )
        run.summary.update(summary)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(published, indent=1) + "\n")
    print(json.dumps(published, indent=1))


if __name__ == "__main__":
    main()
