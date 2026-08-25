#!/usr/bin/env python3
"""Publish the E209 stage-0 evidence to W&B.

    python3 research/e209_wandb.py [--dry-run]

Two runs in group `qwen38-r1-e209-variance-tax-controller`:

  desk    the pre-registered controller replay on the E168 `p7` corpus.
          Every number is harness=local.
  ranked  the cross-check on the FINDING 520 instrument. Every number is
          harness=ranked.

The two runs are kept apart on purpose. The desk run answers the assigned
question and the ranked run prices the same family on the published median;
mixing them would let a local cancellation term leak into ranked pricing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e209-variance-tax-controller"
ARTIFACTS = pathlib.Path("research/e209-artifacts")
ASSIGNMENT_BASE = "22ca1b48c455b83b489afcd59badec2cab84ec5d"


def head_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def identity(harness):
    """RULE 386 identity tuple for a desk replay.

    A desk replay has no built worker, no host thermal state and no timed leg,
    so the fields that describe a measurement are recorded as `desk-replay`
    rather than left blank or borrowed from an unrelated run.
    """
    return {
        "harness": harness,
        "experiment": "e209-variance-tax-controller",
        "stage": "0",
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "commitSha": head_sha(),
        "corpus": "e168 p7 pinned depth 7, 9 prompts, 512 decode tokens",
        "corpusProvenance": "8 prose prompts self-authored under research/; "
                            "benchfixture is the organizer public longcopy "
                            "gate prompt",
        "costTable": "research/out/e168/round_cost.json (E168 tree, DATED)",
        "cap": 7,
        "capNote": "cap 8 not evaluated; width 9 has no measured cost cell",
        "port": "e207_desk depth_walk / update_ema; positive control 1955 "
                "rounds, 0 mismatches",
        "worker": "desk-replay",
        "host": "desk-replay",
        "coolGatePassedRealGate": False,
        "gateQualifiedForTiming": False,
        "timingClaimsPermitted": False,
        "officialOrRankedScore": False,
    }


def desk_summary():
    payload = json.loads((ARTIFACTS / "stage0-controllers.json").read_text())
    summary = payload["summary"]
    best = summary["best_controller"]
    row = summary["controllers"][best]
    out = {
        "harness": "local",
        "bestController": best,
        "bestProseMedianPctFaster": summary["best_prose_median_pct"],
        "bestProseMinPctFaster": summary["best_prose_min_pct"],
        "bestBenchfixturePctFaster": row["benchfixture_pct"],
        "bestProseMedianMsPerRound": row["prose_median_ms_per_round"],
        "bestSwitchesPerLeg": row["switches_per_leg"],
        "shippedSwitchesPerLeg": row["shipped_switches_per_leg"],
        "fractionOfCeilingCaptured": row["fraction_of_ceiling"],
        "ceilingProseMedianPctFaster": summary["ceiling_prose_median_pct"],
        "varianceOnlyProseMedianPctFaster":
            summary["variance_only_prose_median_pct"],
        "lopoMedianPctFaster": summary["lopo_median_pct"],
        "lopoMinPctFaster": summary["lopo_min_pct"],
        "benchfixtureSignAgreesWithProse":
            summary["benchfixture_sign_agrees_with_prose"],
        "signSurvivesLinearCostTable":
            summary["sign_survives_linear_cost_table"],
        "linearCostTableProseMedianPctFaster":
            summary["cost_table_sensitivity"]["linear"]["prose_median_pct"],
        "constantDepth3ProseMedianPctFaster":
            summary["constant_depth_null"]["fixed-d3"]["prose_median_pct"],
        "constantDepth3BenchfixturePctFaster":
            summary["constant_depth_null"]["fixed-d3"]["benchfixture_pct"],
        "constantDepth2ProseMedianPctFaster":
            summary["constant_depth_null"]["fixed-d2"]["prose_median_pct"],
        "constantDepth2BenchfixturePctFaster":
            summary["constant_depth_null"]["fixed-d2"]["benchfixture_pct"],
        "draws": summary["draws"],
        "verdict": summary["verdict"],
    }
    for name, controller in summary["controllers"].items():
        out["controller/%s/proseMedianPctFaster" % name] = \
            controller["prose_median_pct"]
        out["controller/%s/proseMinPctFaster" % name] = \
            controller["prose_min_pct"]
        out["controller/%s/benchfixturePctFaster" % name] = \
            controller["benchfixture_pct"]
        out["controller/%s/switchesPerLeg" % name] = \
            controller["switches_per_leg"]
    for prompt, value in row["per_prompt_pct"].items():
        out["bestPerPrompt/%s/pctFaster" % prompt] = value
    for label, scenario in summary["cost_table_sensitivity"].items():
        out["costSensitivity/%s/proseMedianPctFaster" % label] = \
            scenario["prose_median_pct"]
    return out


def ranked_summary():
    payload = json.loads((ARTIFACTS / "ranked-crosscheck.json").read_text())
    out = {
        "harness": "ranked",
        "receiptA": payload["receipt_A"],
        "crown": payload["crown"],
        "shippedCap7": payload["shipped_cap7"],
        "shippedCap8": payload["shipped_cap8"],
        "oraclePerPromptFixed": payload["oracle_per_prompt_fixed"],
        "oracleVsShippedCap8Pct": payload["oracle_vs_shipped_cap8_pct"],
        "familyCeilingIsNegative": payload["oracle_vs_shipped_cap8_pct"] < 0.0,
        "medianSettingPrompts": ",".join(payload["median_setting_prompts"]),
    }
    for depth, value in payload["const_published_median"].items():
        out["constDepth/%s/publishedMedian" % depth] = value
        out["constDepth/%s/pctVsShippedCap8" % depth] = \
            100.0 * (value - payload["shipped_cap8"]) / payload["shipped_cap8"]
    for name, depth in payload["per_prompt_best_depth"].items():
        out["rankedPrompt/%s/bestFixedDepth" % name] = depth
        out["rankedPrompt/%s/shippedCap8Raw" % name] = \
            payload["per_prompt_shipped_cap8_raw"][name]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = {
        "desk": ({"job_type": "desk", "name": "e209-stage0-controllers",
                  "config": {**identity("local"),
                             "deliverable": "pre-registered online "
                                            "depth-controller family",
                             "preregistration": "research/e209_controllers.py "
                                                "committed before the replay"}},
                 desk_summary()),
        "ranked": ({"job_type": "desk", "name": "e209-ranked-crosscheck",
                    "config": {**identity("ranked"),
                               "instrument": "FINDING 520 survival-pinned "
                                             "latent-q, imported unmodified "
                                             "from e201/e203",
                               "deliverable": "published-median ceiling for "
                                              "the depth-policy family"}},
                   ranked_summary()),
    }

    if args.dry_run:
        print(json.dumps({key: {"config": spec["config"], "summary": summary}
                          for key, (spec, summary) in specs.items()},
                         indent=1, sort_keys=True))
        return

    import wandb

    published = {}
    for key, (spec, summary) in specs.items():
        run = wandb.init(project=PROJECT.split("/")[-1],
                         entity=PROJECT.split("/")[0], group=GROUP, **spec)
        run.summary.update(summary)
        published[key] = {"run": run.id, "url": run.url}
        run.finish()
    print(json.dumps(published, indent=1))


if __name__ == "__main__":
    main()
