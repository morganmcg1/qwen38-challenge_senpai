#!/usr/bin/env python3
"""Publish the E101 fused row top-32 selection ABBA legs to W&B.

    usage: research/e101_wandb_log.py [r5|r5s|all]

One run per timed leg plus one analysis run per block. Each leg run carries its
full experiment identity tuple, the path witness that proves which arm actually
executed, every fidelity field, the per-round counter series, and the
contamination bookkeeping. Each analysis run carries the drift-corrected score
contrast and the per-counter table at the tier the block supports.

MEASUREMENT MODE. Every leg ran with the local cool gate disabled under the
standing permitted mode in `senpai/program.md`, so `cool_gate_passed_real_gate`
and `gate_qualified_for_timing` are published false verbatim on every run and
`official_or_ranked_score` is false. Entry and exit GPU temperature are logged
for each leg so the counterbalancing can be audited. These runs are directional
causal evidence inside one counterbalanced session, not ranked scores.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import sys

import wandb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import e101_abba_analyse as an  # noqa: E402

ENTITY = "wandb-applied-ai-team"
PROJECT = "qwen38-mlx-challenge-senpai"
HOST = "apple-m4-pro-mac16-11-20core-48gib"
WITNESS = re.compile(r"sel_env=(\S+) sel_fused=(\d+) sel_argpart=(\d+)")

BLOCKS = {
    "r5": ("e101-row-top32-abba-production",
           "Production ABBA. Head-chain GPU time sits inside verify_build_us."),
    "r5s": ("e101-row-top32-abba-sync-head",
            "MLX_QWEN_MTP_TRACE_SYNC_HEAD=1 drains the head chain before the "
            "verify window, moving its GPU time into d_submit2_us."),
}


def witness(tag):
    """Last path witness in the leg trace, proving which arm really ran."""
    path = f"research/out/{tag}/trace.txt"
    found = None
    if os.path.exists(path):
        for line in open(path):
            m = WITNESS.search(line)
            if m:
                found = m
    if not found:
        return {}
    return {"sel_env": found.group(1),
            "sel_fused": int(found.group(2)),
            "sel_argpart": int(found.group(3))}


def ledger(tag):
    """Row ledger recovered from the trace, independent of score.json."""
    rows = an.read_rounds_unfiltered(tag)
    if not rows:
        return {}
    n = len(rows)
    drafts = sum(r["d"] for r in rows)
    acc = sum(r["acc"] for r in rows)
    return {"ledger_rounds": n, "ledger_target_rows": drafts + n,
            "ledger_emitted": acc + n, "ledger_drafts": drafts,
            "ledger_accepted": acc, "ledger_rejected": drafts - acc}


def log_leg(leg, block, group, notes):
    tag = leg["tag"]
    config = {k: leg.get(k) for k in an.META}
    config.update({
        "experiment": "e101-selection-chain-custom-topk",
        "block": block, "arm": leg["e101_arm"], "position": leg["position"],
        "harness": "local", "host_profile": HOST,
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "sync_head": block == "r5s",
    })
    config.update(witness(tag))
    run = wandb.init(entity=ENTITY, project=PROJECT, group=group, name=tag,
                     job_type="abba-leg", config=config, notes=notes,
                     reinit=True)

    for step, row in enumerate(an.read_rounds_unfiltered(tag)):
        run.log({f"round/{k}": v for k, v in row.items()}, step=step)

    summary = {f"score/{k}": leg.get(k) for k in an.FIELDS}
    summary.update({f"counter/{k}": v for k, v in leg["trace"].items()})
    summary.update({f"counter_all/{k}": v for k, v in leg["trace_all"].items()})
    summary.update(ledger(tag))
    summary.update({
        "rounds_kept": leg["kept"], "rounds_total": leg["total"],
        "kept_fraction": leg["kept_fraction"],
        "clean_tier": leg["clean"], "strict_tier": leg["strict"],
    })
    run.summary.update(summary)
    run.finish()
    return run.url, run.id


def log_analysis(legs, block, group, notes, reference, threshold):
    tier = [leg for leg in legs if leg["strict"]] or legs
    drafts = statistics.mean([leg["trace"]["d"] for leg in tier
                              if "d" in leg["trace"]] or [float("nan")])
    off = [leg["mtp_seconds_per_token"] for leg in tier
           if leg["e101_arm"] == "off"]
    on = [leg["mtp_seconds_per_token"] for leg in tier
          if leg["e101_arm"] == "on"]
    run = wandb.init(entity=ENTITY, project=PROJECT, group=group,
                     name=f"e101{block}-analysis", job_type="analysis",
                     config={"experiment": "e101-selection-chain-custom-topk",
                             "block": block, "harness": "local",
                             "host_profile": HOST,
                             "round_cpu_ratio": an.ROUND_CPU_RATIO,
                             "min_clean_fraction": an.MIN_CLEAN_FRACTION,
                             "min_counter_fraction": an.MIN_COUNTER_FRACTION,
                             "pooled_cpu_median_ns": reference,
                             "stall_threshold_ns": threshold,
                             "tier_legs": [leg["tag"] for leg in tier],
                             "gate_qualified_for_timing": False,
                             "official_or_ranked_score": False},
                     notes=notes, reinit=True)
    summary = {"drafts_per_round": drafts, "n_off": len(off), "n_on": len(on)}
    if off and on:
        pct = (statistics.mean(on) - statistics.mean(off)) \
            / statistics.mean(off) * 100
        summary.update({"score/off_mtp_s_per_tok": statistics.mean(off),
                        "score/on_mtp_s_per_tok": statistics.mean(on),
                        "score/contrast_pct": pct})
    for key in an.COUNTERS:
        c = an.counter_contrast(tier, key, drafts)
        if c:
            summary[f"contrast/{key}/off"] = c["off"]
            summary[f"contrast/{key}/on"] = c["on"]
            summary[f"contrast/{key}/delta_per_round"] = c["delta"]
            summary[f"contrast/{key}/delta_per_draft"] = c["per_draft"]
            summary[f"contrast/{key}/ranked_pct_median_pair"] = \
                an.price_us_per_draft(-c["per_draft"])
    run.summary.update(summary)
    run.finish()
    return run.url, run.id


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    blocks = list(BLOCKS) if which == "all" else [which]
    index = {}
    for block in blocks:
        group, notes = BLOCKS[block]
        legs = an.load(block)
        if not legs:
            print(f"{block}: no legs found")
            continue
        reference, threshold = an.classify(legs)
        index[block] = {"legs": {}, "analysis": None}
        for leg in legs:
            url, rid = log_leg(leg, block, group, notes)
            index[block]["legs"][leg["tag"]] = {"url": url, "id": rid}
            print(f"{leg['tag']:14s} {leg['e101_arm']:3s} pos {leg['position']} "
                  f"kept {leg['kept']}/{leg['total']}  {url}")
        url, rid = log_analysis(legs, block, group, notes, reference, threshold)
        index[block]["analysis"] = {"url": url, "id": rid}
        print(f"{block} analysis  {url}")
    out = pathlib.Path("research/out/e101-wandb-runs.json")
    out.write_text(json.dumps(index, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
