#!/usr/bin/env python3
"""Publish the E207 stage-0 evidence to W&B.

    python3 research/e207_wandb.py --screen e207-exact-1-frozen [--dry-run]

Two runs in group `qwen38-r1-e207-ema-subtraction`:

  screen   the arm-ON exactness leg. harness=local. Carries the RULE 386
           identity tuple, the RULE 391(b) witness census, the exact-token
           result and the RULE 179 closure of the leg.
  desk     the stage-0 desk. Local open-loop replay numbers are labelled
           harness=local and the published-median projection harness=ranked;
           every number keeps its own label.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e203_traces as T  # noqa: E402

PROJECT = "wandb-applied-ai-team/qwen38-mlx-challenge-senpai"
GROUP = "qwen38-r1-e207-ema-subtraction"
ARTIFACTS = pathlib.Path("research/e207-artifacts")
ASSIGNMENT_BASE = "9058dbd0316baf6181b345d4cea41e527b4cc6cb"
RECEIPT_A = 3.70784519415395
SEED_RE = re.compile(r"^mtp-trace: begin .*\bwall_us=(\d+)")


def read_meta(tag):
    meta = {}
    for line in (pathlib.Path("research/out") / tag / "meta.txt").read_text(
            errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def witness_census(tag):
    """Arm witness evidence: the census of `e207=` and `ema0_post=` values."""
    path = pathlib.Path("research/out") / tag / "trace.txt"
    arms = collections.Counter()
    ema0 = collections.Counter()
    ema_vectors = set()
    for line in path.read_text(errors="replace").splitlines():
        found = re.search(r"e207=(\w+) ema0_post=(\S+)", line)
        if found:
            arms[found.group(1)] += 1
            ema0[found.group(2)] += 1
        vector = re.search(r" ema=([0-9.,]+)", line)
        if vector:
            ema_vectors.add(vector.group(1))
    return {"armWitnessCounts": dict(arms),
            "ema0PostCounts": dict(ema0),
            "distinctPreRoundEMAVectors": len(ema_vectors),
            "preRoundEMAVectors": sorted(ema_vectors)[:4]}


def screen_summary(tag):
    score = json.loads(
        (pathlib.Path("research/out") / tag / "score.json").read_text())
    metrics = score["metrics"]
    rounds = T.read_leg(pathlib.Path("research/out") / tag)
    mtp = [r for r in rounds if r["depth"] > 0]
    seed_ms = 0.0
    for line in (pathlib.Path("research/out") / tag
                 / "trace.txt").read_text(errors="replace").splitlines():
        found = SEED_RE.match(line)
        if found:
            seed_ms = int(found.group(1)) / 1000.0
            break
    decode_ms = sum(r["round_us"] for r in mtp) / 1000.0
    tokens = len(mtp) + sum(r["accepted"] for r in mtp)
    edl = statistics.fmean([r["accepted"] for r in mtp]) if mtp else 0.0
    modelled = (seed_ms + decode_ms) / 1000.0 / tokens
    return {
        "harness": "local",
        "allTokensMatched": metrics["all_tokens_matched"],
        "residualDivergenceCount": metrics["residual_divergence_count"],
        "publicDriftTripwirePassed": metrics["public_drift_tripwire_passed"],
        "decodeTokens": metrics["decode_tokens"],
        "mtpSecondsPerToken": metrics["mtp_seconds_per_token"],
        "serialSecondsPerToken": metrics["serial_seconds_per_token"],
        "mtpDecodeSpeedup": metrics["mtp_decode_speedup"],
        "effectiveMeanDraftLen": metrics["effective_mean_draft_len"],
        "acceptedDraftRate": metrics["accepted_draft_rate"],
        "headProvenanceSha256": metrics["head_provenance_sha256"],
        # RULE 179 closure of this leg: the emitted tokens must equal
        # rounds * (1 + edl), and the wall time must equal the seed prologue
        # plus the summed round costs.
        "rule179TokensFromRounds": len(mtp) * (1.0 + edl),
        "rule179TokensEmitted": tokens,
        "rule179SeedMs": seed_ms,
        "rule179DecodeMs": decode_ms,
        "rule179ModelledSecondsPerToken": modelled,
        "rule179ResidualPct": 100.0 * (
            modelled / metrics["mtp_seconds_per_token"] - 1.0),
        "mtpRounds": len(mtp),
        "roundMsMedian": statistics.median(
            [r["round_us"] / 1000.0 for r in mtp]) if mtp else 0.0,
        "depthHistogram": {str(d): n for d, n in
                           collections.Counter(r["depth"] for r in mtp).items()},
        **witness_census(tag),
    }


def identity(tag, meta):
    return {
        "harness": "local",
        "experiment": "e207-ema-subtraction",
        "stage": "0",
        "leg": tag,
        "arm": meta.get("e207_arm", ""),
        "armEnv": meta.get("e207_arm_env", ""),
        "instrument": meta.get("e207_instrument", ""),
        "baseSha": meta.get("base_sha", ""),
        "assignmentBaseSha": ASSIGNMENT_BASE,
        "workerSha256": meta.get("worker_sha256", ""),
        "postRunWorkerSha256": meta.get("post_run_worker_sha256", ""),
        "cliSha256": meta.get("cli_sha256", ""),
        "host": meta.get("host", ""),
        "chip": meta.get("chip", ""),
        "memoryBytes": meta.get("memory_bytes", ""),
        "tokens": meta.get("tokens", ""),
        "localMode": meta.get("local_mode", ""),
        "headDir": meta.get("head_dir", ""),
        "sandbox": meta.get("sandbox", ""),
        "metallibSourceFingerprint": meta.get(
            "metallib_source_fingerprint", ""),
        "dirtyCandidatePaths": meta.get("dirty_candidate_paths", ""),
        "coolGate": meta.get("cool_gate", ""),
        "coolGatePassedRealGate": meta.get("cool_gate_passed_real_gate", ""),
        "gateQualifiedForTiming": meta.get("gate_qualified_for_timing", ""),
        "officialOrRankedScore": meta.get("official_or_ranked_score", ""),
        "gpuTempEntryC": meta.get("gpu_temp_entry_c", ""),
        "gpuTempExitC": meta.get("gpu_temp_exit_c", ""),
        "trace": meta.get("trace", ""),
        "traceRounds": meta.get("trace_rounds", ""),
    }


def desk_payload():
    return json.loads((ARTIFACTS / "stage0-desk.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="e207-exact-1-frozen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    meta = read_meta(args.screen)
    screen = screen_summary(args.screen)
    desk = desk_payload()

    local = desk["harness_local"]["prompts"]
    ranked = desk["harness_ranked"]
    prereg = desk["prereg_local_stage1"]
    desk_summary = {
        "harness": "mixed-labelled",
        "walkPortRoundsChecked": desk["walk_control"]["rounds_checked"],
        "walkPortMismatches": desk["walk_control"]["mismatches"],
        "frozenDepthCap7": desk["frozen_depth_closed_form"]["7"],
        "frozenDepthCap8": desk["frozen_depth_closed_form"]["8"],
        "local_predictedStage1DeltaPct":
            prereg["predicted_stage1_delta_pct"],
        "local_benchfixtureLiveMsPerToken":
            prereg["benchfixture_live_leg_ms_per_token"],
        "local_benchfixtureFrozenMsPerToken":
            prereg["benchfixture_frozen_leg_ms_per_token"],
        "local_wideCellTieScale": prereg["wide_cell_tie_scale"],
        "local_signRobust": prereg["sign_robust_to_wide_cell_staleness"],
        "ranked_receiptA": ranked["receipt_A"],
        "ranked_shipCap7": ranked["ship_cap7"],
        "ranked_shipCap8": ranked["ship_cap8"],
        "ranked_frozenConstDepthMedian": ranked["const_table"]["4"],
        "ranked_frozenMarginMixMedian": ranked["frozen_margin_mix"],
        "ranked_bracketPctVsA": ranked["bracket_pct_vs_A"],
        "ranked_bracketPctVsCap8": ranked["bracket_pct_vs_cap8"],
        "ranked_predictedSign": ranked["predicted_sign"],
    }
    for prompt, result in local.items():
        for arm in ("live", "frozen", "live_nomargin", "frozen_nomargin"):
            desk_summary["local_%s_%s_msPerToken" % (prompt, arm)] = \
                result[arm]["ms_per_token"]
        desk_summary["local_%s_liveMeanDepth" % prompt] = \
            result["live"]["mean_depth"]
        desk_summary["local_%s_frozenMeanDepth" % prompt] = \
            result["frozen"]["mean_depth"]

    if args.dry_run:
        print(json.dumps({"screen": screen, "desk": desk_summary}, indent=1))
        return

    import wandb

    run = wandb.init(project=PROJECT.split("/")[-1],
                     entity=PROJECT.split("/")[0], group=GROUP,
                     job_type="exactness-screen",
                     name="e207-screen-frozen-64",
                     config={**identity(args.screen, meta),
                             "commitSha": head})
    run.summary.update(screen)
    screen_url, screen_id = run.url, run.id
    run.finish()

    run = wandb.init(project=PROJECT.split("/")[-1],
                     entity=PROJECT.split("/")[0], group=GROUP,
                     job_type="desk", name="e207-stage0-desk",
                     config={"experiment": "e207-ema-subtraction",
                             "stage": "0",
                             "commitSha": head,
                             "assignmentBaseSha": ASSIGNMENT_BASE,
                             "corpus": "e168 p7 pinned-depth-7, 9 prompts",
                             "rankedInstrument": "FINDING 520 survival-pinned "
                                                 "latent-q via e201/e203",
                             "localCostTable": "research/out/e168/"
                                               "round_cost.json (DATED)",
                             "receiptA": RECEIPT_A})
    run.summary.update(desk_summary)
    print(json.dumps({"screen_run": screen_id, "screen_url": screen_url,
                      "desk_run": run.id, "desk_url": run.url}, indent=1))
    run.finish()


if __name__ == "__main__":
    main()
